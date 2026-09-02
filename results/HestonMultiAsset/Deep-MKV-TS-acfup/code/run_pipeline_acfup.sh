#!/usr/bin/env bash
# Unattended end-to-end campaign for the ACF-UP variant (config A) at d = 8.
#
# WHAT IS DIFFERENT FROM THE PUBLISHED RUN
#   published tree: LAMBDA_SCALE=50  KAPPA_SCALE=100  a=0.25  s=0.125  steps=3000
#   this tree:      LAMBDA_SCALE=25  KAPPA_SCALE=50   a=0.5   s=0.25   steps=2500
#   vol_scale = KAPPA_SCALE / LAMBDA_SCALE = 2 in BOTH, so the discrepancy
#   function's internal vol rescaling is unchanged; what moves is the ACF weight
#   (x2) and the objective magnitude (/2, which is what keeps a=0.5 from
#   diverging -- a=0.5 at LAMBDA_SCALE=50 blew up in the 250-step sweep).
#
# THIS SCRIPT NEVER TOUCHES results/HestonMultiAsset/Deep-MKV-TS/. Every path
# below is derived from METHOD_DIR = the parent of this file's directory, which
# is .../Deep-MKV-TS-acfup. compute_all_multiasset.py is given an explicit
# --results-dir for the same reason. No git, no commit, no push: the instruction
# was queue-and-report, not publish.
#
# GPU 0 is forbidden (standing instruction). GPU 1 belongs to another user
# (jyoussef) as of launch. GPU 2 and GPU 3 are used -- two GPUs, the hard limit.
# 5 concurrent trainers, 3 cores each = 15 cores, under the 16-core limit.
#
# DIVERGENCE IS EXPECTED AND HANDLED. Raising the ACF weight is exactly the
# knob that made the d=8 control explode before. A seed that fails to produce
# step_2500.pt is DROPPED from the reported set and named in the log; the chain
# continues on the survivors instead of aborting after 7 hours of GPU time.
#
#   setsid bash run_pipeline_acfup.sh > logs/pipeline_acfup.log 2>&1 & disown

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python
LOGS="$METHOD_DIR/logs"
STEPS=2500

mkdir -p "$LOGS" "$METHOD_DIR/weights" "$METHOD_DIR/losses" "$HERE/runs"

export PYTHONPATH="$REF/src:$REF/experiments"

echo "=== ACFUP PIPELINE START $(date -Is)"
echo "=== method dir: $METHOD_DIR"
grep -nE '^(LAMBDA_SCALE|KAPPA_SCALE|ABS_RETURN_ACF_WEIGHT|SQUARED_RETURN_ACF_WEIGHT|RIDGE_LAMBDA) =' \
    "$HERE/train_multiasset.py" | sed 's/^/[config] /'

# ------------------------------------------------------------ 1. train -------
# seed -> (gpu, core range).  3 on GPU 2, 2 on GPU 3.
declare -A GPU_OF=( [0]=2 [2]=2 [4]=2 [5]=3 [6]=3 )
declare -A CORES_OF=( [0]=0-2 [2]=3-5 [4]=6-8 [5]=9-11 [6]=12-14 )
ALL_SEEDS="0 2 4 5 6"

for s in $ALL_SEEDS; do
  g="${GPU_OF[$s]}"; c="${CORES_OF[$s]}"
  echo "=== [train seed $s] launching on GPU $g cores $c $(date -Is)"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  OPENBLAS_NUM_THREADS=3 \
  taskset -c "$c" "$PY" "$HERE/train_multiasset.py" \
      --seed "$s" --steps "$STEPS" --device cuda:0 \
      --run-root "$HERE/runs" \
      > "$LOGS/train_seed_${s}.log" 2>&1 &
  sleep 20   # stagger: five simultaneous CUDA context creations on 2 cards
done

echo "=== [train] all 5 launched, waiting $(date -Is)"
wait
echo "=== [train] all trainers exited $(date -Is)"

# --------------------------------------------------- 2. survivor triage ------
# A seed counts only if it reached the final checkpoint. Reporting a seed that
# died at step 1800 using its step_1500 payload would be silently changing the
# protocol between the published run and this one.
SURVIVORS=""
DIVERGED=""
for s in $ALL_SEEDS; do
  if compgen -G "$HERE/runs/seed_$s/training_checkpoints/step_*${STEPS}.pt" >/dev/null; then
    SURVIVORS="$SURVIVORS $s"
  else
    DIVERGED="$DIVERGED $s"
  fi
done
SURVIVORS="$(echo $SURVIVORS)"
DIVERGED="$(echo $DIVERGED)"
echo "=== [triage] survivors: [$SURVIVORS]   diverged/incomplete: [$DIVERGED]"
if [ -z "$SURVIVORS" ]; then
  echo "=== ABORT: every seed failed to reach step $STEPS $(date -Is)" >&2
  exit 1
fi
SEEDS_COMMA="$(echo "$SURVIVORS" | tr ' ' ',')"

run() {
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 taskset -c 0-7 "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc $(date -Is) -- CHAIN STOPPED" >&2
    exit "$rc"
  fi
  echo "=== [$label] OK $(date -Is)"
}

# ------------------------------------------------------------ 3. select ------
run "select checkpoints" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0 --run-root "$HERE/runs"

# ---------------------------------------------------------- 4. sentinel ------
printf '%s seeds=%s (acfup config A; diverged/incomplete:%s)\n' \
    "$(date -Is)" "$SEEDS_COMMA" "${DIVERGED:- none}" \
    > "$METHOD_DIR/weights/.campaign_complete"
echo "=== [sentinel] wrote weights/.campaign_complete"

# -------------------------------------------------------- 5. generation ------
run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0

# ------------------------------------------------------------ 6. gate --------
# Hard gate on purpose: a broken .npy yields normal-looking numbers 70 minutes
# later, so the chain must stop before the metrics, not after them.
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"

# --------------------------------------------------------- 7. metrics --------
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method Deep-MKV-TS-acfup \
    --results-dir "$METHOD_DIR" \
    --seed-list "$SEEDS_COMMA"

# --------------------------------------------------- 8. memorisation ---------
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"

# ----------------------------------------------------------- 9. plots --------
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"

# -------------------------------------------------------- 10. README ---------
# Method page only. tools/render_comparison.py is deliberately NOT run: it would
# rewrite the published comparison table in results/HestonMultiAsset/README.md.
run "render method README" "$PY" "$HERE/render_readme.py"

echo "=== ACFUP PIPELINE COMPLETE $(date -Is)"
echo "=== reported seeds: $SEEDS_COMMA"
echo "=== dropped seeds:  ${DIVERGED:- none}"
