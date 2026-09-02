#!/usr/bin/env bash
# Parameterised d = 8 campaign runner. One script, three variants.
#
# Everything is derived from METHOD_DIR = the parent of this file's directory,
# so dropping a copy into results/HestonMultiAsset/<variant>/code/ is all it
# takes to give that variant its own isolated campaign. The published tree
# results/HestonMultiAsset/Deep-MKV-TS/ is never written to.
#
# ENV KNOBS (all optional, defaults shown)
#   STEPS=2500        training length
#   TRAIN=1           0 = skip training entirely and use whatever is already in
#                     RUN_ROOT. Used by the base2500 control, whose RUN_ROOT is
#                     symlinks to the PUBLISHED checkpoints truncated at 2500 --
#                     retraining identical seeds would only add rerun noise on
#                     top of the step-count change we are trying to isolate.
#   RUN_ROOT=$HERE/runs
#   GPUS="1 1 1 1 1"          one entry per seed, same order as ALL_SEEDS
#   CORES="16-17 ..."         one entry per seed
#   THREADS=2                 OMP/MKL/OPENBLAS per trainer
#   POST_GPU=1                GPU for selection/generation/metrics
#   POST_CORES=16-23
#   WAIT_LOG= WAIT_STR=       block until WAIT_STR appears in WAIT_LOG before
#                             starting. Used to chain a variant behind another
#                             campaign so the two never fight for the same card.
#
# DIVERGENCE IS HANDLED: a seed with no final checkpoint is dropped by name and
# the chain continues on the survivors, instead of throwing away hours of GPU
# time because one seed of five exploded.
#
# No git, no commit, no push.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
METHOD="$(basename "$METHOD_DIR")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python
LOGS="$METHOD_DIR/logs"

STEPS="${STEPS:-2500}"
TRAIN="${TRAIN:-1}"
RUN_ROOT="${RUN_ROOT:-$HERE/runs}"
GPUS="${GPUS:-1 1 1 1 1}"
CORES="${CORES:-16-17 18-19 20-21 22-23 24-25}"
THREADS="${THREADS:-2}"
POST_GPU="${POST_GPU:-1}"
POST_CORES="${POST_CORES:-16-23}"
WAIT_LOG="${WAIT_LOG:-}"
WAIT_STR="${WAIT_STR:-}"

ALL_SEEDS=(0 2 4 5 6)
read -r -a GPU_ARR  <<< "$GPUS"
read -r -a CORE_ARR <<< "$CORES"

mkdir -p "$LOGS" "$METHOD_DIR/weights" "$METHOD_DIR/losses" "$RUN_ROOT"
export PYTHONPATH="$REF/src:$REF/experiments"

echo "=== [$METHOD] PIPELINE START $(date -Is)"
echo "=== method dir: $METHOD_DIR"
echo "=== steps=$STEPS train=$TRAIN run_root=$RUN_ROOT"
grep -nE '^(LAMBDA_SCALE|KAPPA_SCALE|ABS_RETURN_ACF_WEIGHT|SQUARED_RETURN_ACF_WEIGHT|RIDGE_LAMBDA) =' \
    "$HERE/train_multiasset.py" | sed 's/^/[config] /'

# ------------------------------------------------------------- 0. wait -------
if [ -n "$WAIT_LOG" ] && [ -n "$WAIT_STR" ]; then
  echo "=== [wait] blocking until '$WAIT_STR' appears in $WAIT_LOG $(date -Is)"
  while ! grep -q "$WAIT_STR" "$WAIT_LOG" 2>/dev/null; do
    if [ -f "$WAIT_LOG" ] && grep -q "CHAIN STOPPED\|ABORT:" "$WAIT_LOG" 2>/dev/null; then
      echo "=== [wait] upstream campaign FAILED -- refusing to start $(date -Is)" >&2
      exit 1
    fi
    sleep 120
  done
  echo "=== [wait] upstream complete, proceeding $(date -Is)"
fi

# ------------------------------------------------------------ 1. train -------
if [ "$TRAIN" -eq 1 ]; then
  for i in "${!ALL_SEEDS[@]}"; do
    s="${ALL_SEEDS[$i]}"; g="${GPU_ARR[$i]}"; c="${CORE_ARR[$i]}"
    echo "=== [train seed $s] launching on GPU $g cores $c $(date -Is)"
    CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    taskset -c "$c" "$PY" "$HERE/train_multiasset.py" \
        --seed "$s" --steps "$STEPS" --device cuda:0 --run-root "$RUN_ROOT" \
        > "$LOGS/train_seed_${s}.log" 2>&1 &
    sleep 20   # stagger simultaneous CUDA context creation
  done
  echo "=== [train] all launched, waiting $(date -Is)"
  wait
  echo "=== [train] all trainers exited $(date -Is)"
else
  echo "=== [train] SKIPPED (TRAIN=0) -- using checkpoints already in $RUN_ROOT"
fi

# --------------------------------------------------- 2. survivor triage ------
SURVIVORS=""; DIVERGED=""
for s in "${ALL_SEEDS[@]}"; do
  if compgen -G "$RUN_ROOT/seed_$s/training_checkpoints/step_*${STEPS}.pt" >/dev/null; then
    SURVIVORS="$SURVIVORS $s"
  else
    DIVERGED="$DIVERGED $s"
  fi
done
SURVIVORS="$(echo $SURVIVORS)"; DIVERGED="$(echo $DIVERGED)"
echo "=== [triage] survivors: [$SURVIVORS]   diverged/incomplete: [$DIVERGED]"
[ -z "$SURVIVORS" ] && { echo "=== ABORT: no seed reached step $STEPS $(date -Is)" >&2; exit 1; }
SEEDS_COMMA="$(echo "$SURVIVORS" | tr ' ' ',')"

run() {
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  CUDA_VISIBLE_DEVICES="$POST_GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 taskset -c "$POST_CORES" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc $(date -Is) -- CHAIN STOPPED" >&2
    exit "$rc"
  fi
  echo "=== [$label] OK $(date -Is)"
}

# ------------------------------------------------------------ 3. select ------
run "select checkpoints" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0 --run-root "$RUN_ROOT"

# ---------------------------------------------------------- 4. sentinel ------
printf '%s seeds=%s (%s; diverged/incomplete:%s)\n' \
    "$(date -Is)" "$SEEDS_COMMA" "$METHOD" "${DIVERGED:- none}" \
    > "$METHOD_DIR/weights/.campaign_complete"
echo "=== [sentinel] wrote weights/.campaign_complete"

# -------------------------------------------------------- 5. generation ------
run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0

# ------------------------------------------------------------ 6. gate --------
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"

# --------------------------------------------------------- 7. metrics --------
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method "$METHOD" --results-dir "$METHOD_DIR" --seed-list "$SEEDS_COMMA"

# --------------------------------------------------- 8. memorisation ---------
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"

# ----------------------------------------------------------- 9. plots --------
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"

# -------------------------------------------------------- 10. README ---------
# Method page only. tools/render_comparison.py is NOT run: it rewrites the
# published comparison table in results/HestonMultiAsset/README.md.
run "render method README" "$PY" "$HERE/render_readme.py"

echo "=== [$METHOD] PIPELINE COMPLETE $(date -Is)"
echo "=== reported seeds: $SEEDS_COMMA"
echo "=== dropped seeds:  ${DIVERGED:- none}"
