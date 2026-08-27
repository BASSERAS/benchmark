#!/usr/bin/env bash
# Production training for Deep-MKV-TS on TrueDataset: 5 seeds over a shared
# work queue.
#
# Why a queue and not fixed waves
# -------------------------------
# Fixed waves are right only when the GPU count divides the seed count and every
# GPU is ours.  Neither held on this campaign: GPU 3 was taken by another user
# mid-run, so the pool was 2 GPUs and 5 seeds, and waves of 2 would have idled a
# GPU for a full 2-hour slot in the last wave.  A queue keeps every worker busy
# until the work runs out, and -- the reason that mattered here -- the pool can
# be WIDENED WHILE THE RUN IS IN FLIGHT by starting another worker against the
# same queue.  No restart, no lost hours.
#
# run_pipeline_true.sh stage 4 therefore DELEGATES to this script rather than
# fanning out waves of its own; it passes WORKERS/SEEDS/STEPS/RIDGE_LAMBDA.  The
# two are not alternatives and running both would double-train: the queue file
# is the interlock, and a seed with runs/seed_N/COMPLETE.json is skipped.
#
# A failed seed does not stop the others.  At d = 8 the Heston campaign lost two
# seeds to a non-finite control; killing the run on the first casualty would
# have thrown away the survivors' hours too.  Failures are recorded and the
# queue continues.  collect_artifacts.py discovers seeds from disk, so a missing
# seed shows up as a smaller count with its number absent -- visible -- rather
# than as a list quietly renumbered into a clean run.
#
# Idempotent: a seed whose runs/seed_N/COMPLETE.json already exists is skipped,
# so this may be re-run to pick up where a previous invocation stopped.
#
# Usage:
#   ./run_seeds_true.sh                      # default pool, default seeds
#   WORKERS="1:0-7 2:8-15" ./run_seeds_true.sh
#   WORKERS="3:16-23" ./run_seeds_true.sh    # add a worker to a LIVE queue
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
LOSSES="$METHOD_DIR/losses"
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python

export PYTHONPATH="$REF/src:$REF/experiments:$HERE"

# GPU 0 is another user's and is never ours. GPU 3 was free when the campaign
# was planned and is not any more, so the default pool is 1 and 2. Override with
# WORKERS to add capacity once a GPU frees up.
WORKERS="${WORKERS:-1:0-7 2:8-15}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-3000}"
# Selected on the VALIDATION bank by the 5-point screen in sweep/lambda_*.json.
# |log NN| is minimised at 100 and bracketed on both sides (1 -> 0.3447,
# 10 -> 0.7540, 100 -> 0.2450, 1000 -> 2.6858), so it is an interior optimum and
# not the grid boundary that section 7.2 warns about. Heston's winner was 1000;
# it moved one decade because 1/sqrt(dt) here is 64.6x larger, which scales the
# ridge targets with it. At 1000 and 10000 the conditional-expectation R^2 is
# NEGATIVE (-0.047, -0.529): the penalty crushes the Z-proxy below the mean
# predictor, so the adjoint signal never reaches the control.
RIDGE_LAMBDA="${RIDGE_LAMBDA:-100}"

QUEUE="$HERE/runs/.seed_queue"
LOCK="$HERE/runs/.seed_queue.lock"
mkdir -p "$HERE/runs" "$LOSSES"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Build the queue once. A second invocation while a queue exists JOINS it rather
# than resetting it, which is what makes adding a worker mid-flight safe.
if [[ ! -f "$QUEUE" ]]; then
  : > "$QUEUE"
  for s in $SEEDS; do
    if [[ -f "$HERE/runs/seed_$s/COMPLETE.json" ]]; then
      echo "[$(stamp)] seed $s already COMPLETE -- not queued"
    else
      echo "$s" >> "$QUEUE"
    fi
  done
  echo "[$(stamp)] queued seeds: $(tr '\n' ' ' < "$QUEUE")"
else
  echo "[$(stamp)] joining existing queue: $(tr '\n' ' ' < "$QUEUE")"
fi

# Pop one seed under flock. Printing nothing means the queue is empty.
pop_seed() {
  flock "$LOCK" bash -c '
    q="$1"
    [[ -s "$q" ]] || exit 0
    head -n 1 "$q"
    tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
  ' _ "$QUEUE"
}

worker() {
  local gpu="$1" cores="$2"
  while true; do
    local seed
    seed="$(pop_seed)"
    [[ -z "$seed" ]] && break
    local log="$LOSSES/train_seed_${seed}.log"
    echo "[$(stamp)] gpu=$gpu cores=$cores :: seed $seed  (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 taskset -c "$cores" \
      "$PY" train_true.py --seed "$seed" --steps "$STEPS" \
      --ridge-lambda "$RIDGE_LAMBDA" --device cuda:0 > "$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "[$(stamp)] gpu=$gpu :: seed $seed OK"
    else
      # Recorded, not fatal: the other seeds keep their hours.
      echo "[$(stamp)] gpu=$gpu :: seed $seed FAILED rc=$rc -- continuing" >&2
      echo "$seed rc=$rc $(stamp)" >> "$LOSSES/failed_seeds.txt"
    fi
  done
  echo "[$(stamp)] gpu=$gpu :: queue empty, worker exiting"
}

pids=()
for w in $WORKERS; do
  worker "${w%%:*}" "${w##*:}" &
  pids+=($!)
done
echo "[$(stamp)] ${#pids[@]} worker(s) started: ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done

rm -f "$QUEUE" "$QUEUE.tmp" "$LOCK"
echo "[$(stamp)] all workers done"
if [[ -f "$LOSSES/failed_seeds.txt" ]]; then
  echo "[$(stamp)] WARNING: some seeds failed -- see $LOSSES/failed_seeds.txt" >&2
fi
for s in $SEEDS; do
  [[ -f "$HERE/runs/seed_$s/COMPLETE.json" ]] && printf '  seed %s COMPLETE\n' "$s"
done
