#!/bin/bash
# Unattended queue: CSDI on Experiments A and B, 5 seeds each + full post-processing.
#
#   wave 1  A0 A1 A2 A3   GPU1: A0 A1   GPU2: A2 A3    (launched by hand, already running)
#   wave 2  A4 B0 B1 B2   GPU1: A4 B0   GPU2: B1 B2
#   wave 3  B3 B4         GPU1: B3      GPU2: B4       || post-process A on the freed cores
#   then    post-process B
#
# Hard limits respected at every instant: <= 2 GPUs (1 and 2), <= 16 cores, taskset-pinned,
# OMP/MKL/OPENBLAS capped at 4 per process.
#
# Launch (never run_in_background -- the harness reaps it):
#   cd /home/tbasseras/benchmark/results/new_experiments/tools
#   setsid bash run_csdi_queue.sh > run_csdi_queue.log 2>&1 < /dev/null & disown

BENCH=/home/tbasseras/benchmark
NE=$BENCH/results/new_experiments
DS=$BENCH/dataset/Heston/new_experiments
SCRIPTS=$DS/protocol/experiments/scripts
PY=/home/tbasseras/gpu-venv/bin/python
REV=4189d37

say() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- training ---
train() {   # seed gpu cores experiment
  local s=$1 g=$2 c=$3 x=$4
  ( cd "$NE/experiment_$x/CSDI/code" && \
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    taskset -c "$c" $PY train_csdi_experiment.py --seed "$s" --experiment "$x" \
      --data "$DS/experiment_$x/train.npy" > "logs/exp${x}_seed${s}.log" 2>&1 )
  say "train exp$x seed$s finished rc=$?"
}

# pgrep is safe inside a script file: this process's argv is the script path,
# not the pattern, so it cannot self-match (that bug cost us a wave already).
wait_for_training() {
  while pgrep -f "train_csdi_experiment.py" > /dev/null; do sleep 30; done
}

# --------------------------------------------------------- post-processing ---
post() {   # experiment gpu cores
  local x=$1 g=$2 c=$3
  local M=$NE/experiment_$x/CSDI
  local D=$DS/experiment_$x
  local F=$NE/experiment_$x/perfect_floor
  local L=$M/code/logs
  local run="taskset -c $c env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  local stem
  if [ "$x" = "A" ]; then stem=drawdown_memory; else stem=heston_mixture; fi

  mkdir -p "$M/pdf_metrics" "$M/pdf_metrics_validation" "$M/plots" "$L"
  say "post $x : 1/9 apply declared S0 repair"
  $run $PY "$NE/tools/apply_s0_repair.py" --model-dir "$M" --apply \
       > "$L/s0_repair_$x.log" 2>&1
  # NOTE: there is no --check FLAG. Check IS the default mode; --apply is the opt-in.
  # Passing --check makes argparse exit 2, so the audit silently never runs.
  say "post $x : 2/9 STRONG audit repair(raw) == scored, bit-for-bit"
  $run $PY "$NE/tools/apply_s0_repair.py" --model-dir "$M" \
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
       --source-revision "$REV" --repair s0_renormalization \
       --hyperparameter-origin official-default > "$L/manifest_$x.log" 2>&1
  say "post $x : 6/9 rc=$?"

  say "post $x : 7/9 plots"
  $run $PY "$NE/tools/plot_losses.py" --model-dir "$M" \
       --title "CSDI (Experiment $x)" > "$L/plots_$x.log" 2>&1
  $run $PY "$NE/tools/plot_stylised_facts.py" --experiment "$x" --model-dir "$M" \
       --seed 0 --label CSDI >> "$L/plots_$x.log" 2>&1
  $run $PY "$NE/tools/plot_experiment_figures.py" --experiment "$x" --model-dir "$M" \
       --seed 0 --out "$M/plots" --label CSDI >> "$L/plots_$x.log" 2>&1
  say "post $x : 7/9 rc=$?"

  say "post $x : 8/9 layout contract (README.md is expected to be MISSING at this point)"
  $run $PY "$NE/tools/check_method_layout.py" --root "$M" --experiment "$x" \
       > "$L/layout_$x.log" 2>&1
  say "post $x : 8/9 rc=$?  (see $L/layout_$x.log)"

  say "post $x : 9/9 README tables"
  {
    echo "===== SECTION 1 : PDF metrics, test side ====="
    $run $PY "$NE/tools/aggregate_pdf_metrics.py" --model-dir "$M" --floor-dir "$F" \
         --label CSDI --subdir pdf_metrics --pattern "*_${stem}.json" \
         --exclude-prefix configuration oracle_gate
    echo; echo "===== SECTION 1.3 : PDF metrics, validation (disc) side ====="
    $run $PY "$NE/tools/aggregate_pdf_metrics.py" --model-dir "$M" --floor-dir "$F" \
         --label CSDI --subdir pdf_metrics_validation --pattern "*_${stem}.json" \
         --exclude-prefix configuration oracle_gate
    echo; echo "===== SECTION 2.1 : A1-A34 ====="
    $run $PY "$NE/tools/make_metrics_tables.py" --model-dir "$M" --floor-dir "$F" \
         --label CSDI --table A
    echo; echo "===== SECTION 2.2 : B curve-shape ====="
    $run $PY "$NE/tools/make_metrics_tables.py" --model-dir "$M" --floor-dir "$F" \
         --label CSDI --table B
  } > "$L/tables_$x.md" 2>&1
  say "post $x : 9/9 done -> $L/tables_$x.md"
  say "POST-PROCESSING $x COMPLETE"
}

# --------------------------------------------------------------- schedule ---
say "queue start. waiting for wave 1 (experiment A seeds 0-3) to finish"
wait_for_training
say "wave 1 done."

say "wave 2: A4 B0 B1 B2"
train 4 1 0-3   A &
train 0 1 4-7   B &
train 1 2 8-11  B &
train 2 2 12-15 B &
wait
say "wave 2 done."

say "wave 3: B3 B4, with post-processing of A running alongside on the freed cores"
train 3 1 0-3  B &
train 4 2 8-11 B &
post  A 1 4-7 &
wait
say "wave 3 done."

post B 2 0-7
say "QUEUE COMPLETE"
