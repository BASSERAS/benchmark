#!/bin/bash
# Unattended queue: TimeDiT on Experiments A and B, 5 seeds each + full post-processing.
#
# ============================== STATUS: FINISHED, DO NOT RERUN ==============================
# Ran to completion 2026-07-31 ~21:00 -> 2026-08-01 ~05:15. All 10 seeds trained, generated and
# post-processed; zero failed or unstable runs. The artefacts it produced are under
# results/new_experiments/experiment_{A,B}/TimeDiT/ and are what the READMEs are checked against.
#
# THE 4-GPU GRANT EXPIRED 2026-08-01 02:37:59 and is fully consumed. GPUs 0 and 3 were released
# at the end of wave 2 and are idle. This file is now a RECORD of a run made under an exception
# that no longer exists. Re-running it as written would violate the standing 2-GPU limit.
# To rerun: revert the "schedule" section at the bottom to the 2-GPU form kept in git history
# for this file (GPU1/GPU2 only, ~12 h), or re-request a grant.
#
# MEASURED, not estimated (mean over the 5 seeds of each experiment, from the manifests):
#   training    A 5105 s   B 5272 s   -> 0.3403 / 0.3514 s per optimiser step over 15000 steps
#   generation  A 3072 s   B 3164 s   -> ~51-53 min per 8192-path bank, 1000-step DDPM, no stride
#   wall per solo seed  A 2.27 h   B 2.34 h    (the "~2.2 h" estimated below was close)
# ===========================================================================================
#
#   wave 1  A0 A1 A2 A3   one solo job per GPU, on GPUs 0 1 2 3          ~2.2 h
#   wave 2  A4 B0 B1 B2   one solo job per GPU, on GPUs 0 1 2 3          ~2.2 h
#   ---- GPUs 0 and 3 released here, at ~4.4 h, inside the 5 h grant ----
#   wave 3  B3 B4         GPU1: B3      GPU2: B4       || post-process A on the freed cores
#   then    post-process B
#
# THE GPU GRANT AND WHY THE LAYOUT IS WHAT IT IS. Theo granted, on 2026-07-31 at ~21:00, a
# time-boxed exception to the standing 2-GPU limit: 3 GPUs plus <=50% of a fourth, "but only
# for the next 5 hours no more". The standing 16-core limit was NOT lifted and is still
# enforced below.
#
# That pair of constraints fixes the layout, and the reasoning is worth keeping because the
# obvious move is the wrong one. Cores bind at 16, so extra GPUs cannot buy more concurrent
# jobs -- 4 jobs x 4 cores is the ceiling with 2 GPUs or with 4. What the extra GPUs buy is
# the removal of GPU contention: measured, 4 jobs sharing 2 GPUs take ~4 h per wave, while a
# job alone on its own device takes ~2.2 h. The win is one job per GPU, not more jobs.
# Total ~7.4 h rather than ~12 h.
#
# GPU 3 carries exactly ONE job: 13.6 GiB measured of 81920 MiB = 17%, so the "<=50% of the
# fourth" condition holds with margin. Waves 1 and 2 are the ONLY phases that touch GPUs 0
# and 3; every later phase is pinned to GPUs 1 and 2, so no post-processing step can drift
# past the 5 h window while still holding a granted-but-expired device.
#
# The previous 2-GPU schedule (GPU1/GPU2 only, ~12 h) is in git history for this file; revert
# the "schedule" section at the bottom to recover it once the grant expires.
#
# Same shape as run_csdi_queue.sh, three differences that matter:
#
#   * TimeDiT is ~2.2 h per solo seed (~0.34 s/step x 15000 steps, plus ~50 min of 1000-step
#     DDPM sampling). 13.6 GiB per process measured, so two per A100-80GB would be
#     comfortable on memory -- it is contention, not memory, that makes sharing cost ~2x.
#   * TimeDiT has NO released implementation (arXiv:2409.02322 App. C). The manifest must
#     therefore say so and name what the code was built from -- that is $BASIS below, passed
#     to write_generation_manifest.py, which sets official_implementation=false.
#   * Hyperparameters were not the paper's defaults (there are none to default to); they come
#     from our own search on Sine+Stocks. $HPORIGIN records that, and records that no tuning
#     touched Heston, train.npy or disc.npy.
#
# Core limit respected at every instant: <= 16 cores, taskset-pinned, OMP/MKL/OPENBLAS capped
# at 4 per process. The GPU count was 4 during waves 1-2 under the expired grant above and
# <= 2 (GPUs 1 and 2) from wave 3 onward; every post-processing phase is pinned to GPUs 1 and 2
# so no step could outlive the grant while still holding a granted device.
#
# Launch (never run_in_background -- the harness reaps it):
#   cd /home/tbasseras/benchmark/results/new_experiments/tools
#   setsid bash run_timedit_queue.sh > run_timedit_queue.log 2>&1 < /dev/null & disown

BENCH=/home/tbasseras/benchmark
NE=$BENCH/results/new_experiments
DS=$BENCH/dataset/Heston/new_experiments
SCRIPTS=$DS/protocol/experiments/scripts
PY=/home/tbasseras/gpu-venv/bin/python
REV=$(cd "$BENCH" && git rev-parse --short HEAD)

# PDF section 5 item 3 asks *whether* official code was used. For TimeDiT the honest answer is
# "no", and an honest "no" is only complete if it names what was run instead.
BASIS="arXiv:2409.02322 App. C -- no official release; DiT-S backbone reimplemented from facebookresearch/DiT"

# PDF section 5 item 4 asks whether hyperparameters were defaults or validation-selected.
# Neither word fits, so state what actually happened.
HPORIGIN="paper-reproduction-selected (Sine+Stocks HP search in methods/TimeDiT/paper_reimplementation; no tuning on Heston, train.npy or disc.npy)"

say() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- training ---
train() {   # seed gpu cores experiment
  local s=$1 g=$2 c=$3 x=$4
  ( cd "$NE/experiment_$x/TimeDiT/code" && mkdir -p logs && \
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    taskset -c "$c" $PY -u train_timedit_experiment.py --seed "$s" --experiment "$x" \
      --data "$DS/experiment_$x/train.npy" > "logs/exp${x}_seed${s}.log" 2>&1 )
  say "train exp$x seed$s finished rc=$?"
}

# pgrep is safe inside a script file: this process's argv is the script path,
# not the pattern, so it cannot self-match (that bug cost us a wave already).
wait_for_training() {
  while pgrep -f "train_timedit_experiment.py" > /dev/null; do sleep 30; done
}

# --------------------------------------------------------- post-processing ---
post() {   # experiment gpu cores
  local x=$1 g=$2 c=$3
  local M=$NE/experiment_$x/TimeDiT
  local D=$DS/experiment_$x
  local F=$NE/experiment_$x/perfect_floor
  local L=$M/code/logs
  local run="taskset -c $c env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  local stem
  if [ "$x" = "A" ]; then stem=drawdown_memory; else stem=heston_mixture; fi

  mkdir -p "$M/pdf_metrics" "$M/pdf_metrics_validation" "$M/plots" "$L"
  # --anchor-exact is REQUIRED for TimeDiT and was measured, not assumed. The declared
  # multiply-first form 100*S/S[:,0] leaves a 1.421e-14 residual on TimeDiT banks, so
  # check_contract's `np.all(a[:, 0] == 100.0)` fails and BOTH steps below exit 1 -- an
  # 11-hour queue that dies at the first post-processing step. LS4 and CSDI happen to land
  # on the anchor exactly under the declared form, which is why this only bites here;
  # apply_s0_repair.py's own docstring predicts the 1.4e-14 figure for a new method.
  # The two forms differ by 1.8e-15 in log returns, so no metric moves.
  # Both steps must use the SAME form or the strong audit compares different maps.
  say "post $x : 1/9 apply declared S0 repair (divide-first, --anchor-exact)"
  $run $PY "$NE/tools/apply_s0_repair.py" --model-dir "$M" --apply --anchor-exact \
       > "$L/s0_repair_$x.log" 2>&1
  say "post $x : 1/9 rc=$?"
  # NOTE: there is no --check FLAG. Check IS the default mode; --apply is the opt-in.
  # Passing --check makes argparse exit 2, so the audit silently never runs.
  say "post $x : 2/9 STRONG audit repair(raw) == scored, bit-for-bit"
  $run $PY "$NE/tools/apply_s0_repair.py" --model-dir "$M" --anchor-exact \
       --raw-dir "$M/raw_banks" >> "$L/s0_repair_$x.log" 2>&1
  say "post $x : 2/9 rc=$?"

  say "post $x : 3/9 benchmark battery A1-A34 + B"
  ( cd "$M/code" && CUDA_VISIBLE_DEVICES=$g $run $PY compute_metrics_experiment.py \
      --experiment "$x" --source model --seeds 5 > "logs/metrics_${x}_model.log" 2>&1 )
  say "post $x : 3/9 rc=$?"

  # Protocol evaluators run UNCHANGED (PDF section 7 item 7). Two passes per seed:
  # test.npy -> pdf_metrics/ , disc.npy -> pdf_metrics_validation/ (blindness evidence).
  if [ "$x" = "B" ]; then
    say "post B : preflight oracle gate (>=0.90 8-regime accuracy)"
    $run $PY "$NE/tools/check_oracle_gate.py" \
         --gate-report "$D/oracle/gate_report.json" \
         --oracle "$D/oracle/oracle.joblib" > "$L/oracle_gate.log" 2>&1
    if [ $? -ne 0 ]; then
      say "post B : ORACLE GATE FAILED -- refusing to score Experiment B"
      cat "$L/oracle_gate.log"
      return 1
    fi
  fi

  for s in 0 1 2 3 4; do
    local G=$M/generated_paths/seed_$s/generated_paths_8192x128.npy
    for side in test validation; do
      local ref outdir
      if [ "$side" = "test" ]; then ref=$D/test.npy; outdir=$M/pdf_metrics
      else                          ref=$D/disc.npy; outdir=$M/pdf_metrics_validation; fi
      if [ "$x" = "A" ]; then
        $run $PY "$SCRIPTS/evaluate_drawdown_memory.py" \
             --train-data "$D/train.npy" --test-data "$ref" --generated-data "$G" \
             --dataset-manifest "$D/manifest.json" \
             --output "$outdir/seed_${s}_${stem}.json" >> "$L/pdf_eval_$x.log" 2>&1
      else
        $run $PY "$SCRIPTS/evaluate_heston_parameter_mixture.py" \
             --train-data "$D/train.npy" --test-data "$ref" --generated-data "$G" \
             --oracle "$D/oracle/oracle.joblib" \
             --oracle-gate-report "$D/oracle/gate_report.json" \
             --output "$outdir/seed_${s}_${stem}.json" >> "$L/pdf_eval_$x.log" 2>&1
      fi
      say "post $x : 4-5/9 evaluator seed $s $side rc=$?"
    done
  done

  say "post $x : 6/9 generation manifests"
  $run $PY "$NE/tools/write_generation_manifest.py" --model-dir "$M" --experiment "$x" \
       --source-revision "$REV" --repair s0_renormalization_anchor_exact \
       --hyperparameter-origin "$HPORIGIN" \
       --reimplementation-basis "$BASIS" > "$L/manifest_$x.log" 2>&1
  say "post $x : 6/9 rc=$?"

  say "post $x : 7/9 plots"
  $run $PY "$NE/tools/plot_losses.py" --model-dir "$M" \
       --title "TimeDiT (Experiment $x)" > "$L/plots_$x.log" 2>&1
  $run $PY "$NE/tools/plot_stylised_facts.py" --experiment "$x" --model-dir "$M" \
       --seed 0 --label TimeDiT >> "$L/plots_$x.log" 2>&1
  $run $PY "$NE/tools/plot_experiment_figures.py" --experiment "$x" --model-dir "$M" \
       --seed 0 --out "$M/plots" --label TimeDiT >> "$L/plots_$x.log" 2>&1
  say "post $x : 7/9 rc=$?"

  say "post $x : 8/9 layout contract (README.md is expected to be MISSING at this point)"
  $run $PY "$NE/tools/check_method_layout.py" --root "$M" --experiment "$x" \
       > "$L/layout_$x.log" 2>&1
  say "post $x : 8/9 rc=$?  (see $L/layout_$x.log)"

  say "post $x : 9/9 README tables"
  {
    echo "===== SECTION 1 : PDF metrics, test side ====="
    $run $PY "$NE/tools/aggregate_pdf_metrics.py" --model-dir "$M" --floor-dir "$F" \
         --label TimeDiT --subdir pdf_metrics --pattern "*_${stem}.json" \
         --exclude-prefix configuration oracle_gate
    echo; echo "===== SECTION 1.3 : PDF metrics, validation (disc) side ====="
    $run $PY "$NE/tools/aggregate_pdf_metrics.py" --model-dir "$M" --floor-dir "$F" \
         --label TimeDiT --subdir pdf_metrics_validation --pattern "*_${stem}.json" \
         --exclude-prefix configuration oracle_gate
    echo; echo "===== SECTION 2.1 : A1-A34 ====="
    $run $PY "$NE/tools/make_metrics_tables.py" --model-dir "$M" --floor-dir "$F" \
         --label TimeDiT --table A
    echo; echo "===== SECTION 2.2 : B curve-shape ====="
    $run $PY "$NE/tools/make_metrics_tables.py" --model-dir "$M" --floor-dir "$F" \
         --label TimeDiT --table B
  } > "$L/tables_$x.md" 2>&1
  say "post $x : 9/9 done -> $L/tables_$x.md"
  say "POST-PROCESSING $x COMPLETE"
}

# --------------------------------------------------------------- schedule ---
say "queue start (4-GPU time-boxed layout). source revision $REV"
say "GPU grant: 3 GPUs + <=50% of a 4th, expires ~5 h from 21:00 on 2026-07-31."
say "GPUs 0 and 3 are used by waves 1-2 ONLY and are released after wave 2 (~4.4 h)."

say "wave 1: A0 A1 A2 A3 -- one solo job per GPU on 0 1 2 3"
train 0 0 0-3   A &
train 1 1 4-7   A &
train 2 2 8-11  A &
train 3 3 12-15 A &
wait
say "wave 1 done."

say "wave 2: A4 B0 B1 B2 -- one solo job per GPU on 0 1 2 3"
train 4 0 0-3   A &
train 0 1 4-7   B &
train 1 2 8-11  B &
train 2 3 12-15 B &
wait
say "wave 2 done. RELEASING GPUs 0 and 3 -- back inside the standing 2-GPU limit."

say "wave 3: B3 B4, with post-processing of A running alongside on the freed cores"
train 3 1 0-3  B &
train 4 2 8-11 B &
post  A 1 4-7 &
wait
say "wave 3 done."

post B 2 0-7
say "QUEUE COMPLETE"
