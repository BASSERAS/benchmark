#!/usr/bin/env bash
# Train all 5 TimeDiT Heston seeds in 2-GPU pairs (GPU 0 + GPU 3), per GUIDELINE §4.2.
# Batch 1: seeds 0+1 ; Batch 2: seeds 2+3 ; Batch 3: seed 4 alone.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/tbasseras/gpu-venv/bin/python
LOG=logs
mkdir -p "$LOG"

run() {  # run <seed> <gpu> <cores>
  local seed=$1 gpu=$2 cores=$3
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c "$cores" \
    "$PY" train_heston.py --seed "$seed" --gpu 0 > "$LOG/seed_${seed}.log" 2>&1
}

echo "=== Batch 1: seeds 0 (GPU0) + 1 (GPU3) ==="
run 0 0 0-7  &
run 1 3 24-31 &
wait
echo "=== Batch 2: seeds 2 (GPU0) + 3 (GPU3) ==="
run 2 0 0-7  &
run 3 3 24-31 &
wait
echo "=== Batch 3: seed 4 (GPU0) ==="
run 4 0 0-7
echo "=== ALL 5 SEEDS DONE ==="
