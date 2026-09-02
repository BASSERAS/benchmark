#!/usr/bin/env bash
# Repair supervisor for the quarter campaign.
#
# WHY THIS FILE EXISTS
# Seeds 4, 5 and 6 died at step 1 with torch.OutOfMemoryError on GPU 1. Two
# independent causes:
#   1. another user put a 25 GiB job on GPU 1 five minutes after the card was
#      checked and found empty;
#   2. the per-trainer memory figure used to plan the layout came from a 3-step
#      smoke test that reported 9.09 GiB peak. Steady state is 18.8 GiB. Five
#      trainers need 94 GiB. They would not have fitted on an EMPTY 80 GiB card.
# Cause 2 is the real one. Cause 1 only decided which seeds died.
#
# The original supervisor (run_pipeline_variant.sh, PID 325986) is still alive
# with seeds 0 and 2 training, and it holds that file open on a live fd, so it
# CANNOT be edited in place -- bash reads scripts incrementally. It will finish,
# triage, and publish a 2-seed quarter campaign. This script then trains the
# three missing seeds and re-runs the whole chain over all five, overwriting
# those 2-seed artefacts with 5-seed ones.
#
# ORDER MATTERS. Training of 4/5/6 starts as soon as config A's trainers exit
# (POST_WAIT is checked later, not now), so the three seeds train in parallel
# with the 2-seed chain instead of after it. The chain then waits for the first
# supervisor to be completely done before it touches weights/.
#
# ENV KNOBS
#   TRAIN_SEEDS="4 5 6"   seeds to train here (the OOM victims)
#   ALL_SEEDS_STR="0 2 4 5 6"   seeds to triage/report over
#   GPUS / CORES          one entry per TRAIN_SEEDS entry
#   WAIT_LOG/WAIT_STR     block before training
#   POST_WAIT_LOG/POST_WAIT_STR  block after training, before the chain
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
RUN_ROOT="${RUN_ROOT:-$HERE/runs}"
TRAIN_SEEDS="${TRAIN_SEEDS:-4 5 6}"
ALL_SEEDS_STR="${ALL_SEEDS_STR:-0 2 4 5 6}"
GPUS="${GPUS:-2 3 3}"
CORES="${CORES:-20-21 22-23 24-25}"
THREADS="${THREADS:-2}"
POST_GPU="${POST_GPU:-3}"
POST_CORES="${POST_CORES:-26-33}"
WAIT_LOG="${WAIT_LOG:-}"
WAIT_STR="${WAIT_STR:-}"
POST_WAIT_LOG="${POST_WAIT_LOG:-}"
POST_WAIT_STR="${POST_WAIT_STR:-}"

read -r -a TSEEDS   <<< "$TRAIN_SEEDS"
read -r -a ALL_SEEDS <<< "$ALL_SEEDS_STR"
read -r -a GPU_ARR  <<< "$GPUS"
read -r -a CORE_ARR <<< "$CORES"

mkdir -p "$LOGS" "$METHOD_DIR/weights" "$METHOD_DIR/losses" "$RUN_ROOT"
export PYTHONPATH="$REF/src:$REF/experiments"

echo "=== [$METHOD REPAIR] START $(date -Is)"
echo "=== training seeds: $TRAIN_SEEDS   reporting over: $ALL_SEEDS_STR"

block_until() {   # $1 log  $2 needle  $3 label
  [ -z "$1" ] && return 0
  echo "=== [$3] blocking until '$2' appears in $1 $(date -Is)"
  while ! grep -q "$2" "$1" 2>/dev/null; do
    if grep -q "CHAIN STOPPED\|ABORT:" "$1" 2>/dev/null; then
      echo "=== [$3] upstream FAILED -- refusing to continue $(date -Is)" >&2
      exit 1
    fi
    sleep 120
  done
  echo "=== [$3] released $(date -Is)"
}

block_until "$WAIT_LOG" "$WAIT_STR" "wait-before-train"

# ------------------------------------------------------------ 1. train -------
for i in "${!TSEEDS[@]}"; do
  s="${TSEEDS[$i]}"; g="${GPU_ARR[$i]}"; c="${CORE_ARR[$i]}"
  echo "=== [train seed $s] launching on GPU $g cores $c $(date -Is)"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$THREADS" \
  MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "$c" "$PY" "$HERE/train_multiasset.py" \
      --seed "$s" --steps "$STEPS" --device cuda:0 --run-root "$RUN_ROOT" \
      > "$LOGS/train_seed_${s}.log" 2>&1 &
  sleep 20
done
echo "=== [train] launched, waiting $(date -Is)"
wait
echo "=== [train] repair trainers exited $(date -Is)"

# The 2-seed chain must be completely finished before this one rewrites
# weights/ and generated_paths/, or the two will interleave inside the same
# directories and produce a mixed-provenance artefact set.
block_until "$POST_WAIT_LOG" "$POST_WAIT_STR" "wait-before-chain"

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

run "select checkpoints" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0 --run-root "$RUN_ROOT"

printf '%s seeds=%s (%s 5-seed repair; diverged/incomplete:%s)\n' \
    "$(date -Is)" "$SEEDS_COMMA" "$METHOD" "${DIVERGED:- none}" \
    > "$METHOD_DIR/weights/.campaign_complete"

run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method "$METHOD" --results-dir "$METHOD_DIR" --seed-list "$SEEDS_COMMA"
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"
run "render method README" "$PY" "$HERE/render_readme.py"

echo "=== [$METHOD REPAIR] PIPELINE COMPLETE $(date -Is)"
echo "=== reported seeds: $SEEDS_COMMA"
echo "=== dropped seeds:  ${DIVERGED:- none}"
