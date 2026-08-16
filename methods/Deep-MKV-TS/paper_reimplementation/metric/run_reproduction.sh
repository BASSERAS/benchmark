#!/usr/bin/env bash
# Deep-MKV-TS paper reimplementation: 4-seed Heston reproduction of paper Table 1.
#
# Seeds 0, 1, 3, 4 (the paper's locked seed set). Two seeds at a time on two
# A100s, per GUIDELINE.md 4.1. GPUs 2/3 and cores 0-15 are held by a concurrent
# EB-JEPA grid, so this run takes GPU 0 -> cores 16-23 and GPU 1 -> cores 24-31.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_ROOT="$(dirname "$HERE")"
REF="$PR_ROOT/../code/reference"
PY=/home/tbasseras/gpu-venv/bin/python
RUNS="$PR_ROOT/runs"
STEPS="${STEPS:-3000}"

mkdir -p "$RUNS"

run_seed() {
  local seed=$1 gpu=$2 cores=$3
  local out="$RUNS/seed_${seed}"
  if [[ -f "$out/COMPLETE.json" ]]; then echo "SKIP seed $seed (complete)"; return 0; fi
  mkdir -p "$out"
  echo "LAUNCH seed=$seed gpu=$gpu cores=$cores steps=$STEPS"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONPATH="$REF/src:$REF/experiments:$REF/experiments/scripts" \
  taskset -c "$cores" "$PY" \
    "$REF/experiments/scripts/run_matched_control_synthetic_validation.py" \
    --task heston --run-dir "$out" --device cuda:0 \
    --protocol-manifest "$HERE/PROTOCOL.json" \
    --seed "$seed" --bank-size 8192 --sample-batch-size 2048 \
    --lambda-scale 50 --kappa-scale 100 --eta 1 --lr 0.002 --grad-clip-norm 5 \
    --reference-kind guyon_lekeufack_structural_likelihood \
    --joint-volatility-weight 0 --source-joint-volatility-weight 0 \
    --abs-return-acf-weight 0.25 --squared-return-acf-weight 0.125 \
    --source-only --source-steps "$STEPS" --solver online \
    --source-checkpoint-steps 500 1000 1500 2000 2500 3000 \
    --fitted-reference-drift-only \
    --adjoint-weight 0 --adjoint-noise-weight 1 \
    --path-derivative-backend autograd \
    --drift-adjoint-backend analytical_reference \
    > "$out/console.log" 2>&1
  echo "{\"seed\": $seed, \"steps\": $STEPS}" > "$out/COMPLETE.json"
  echo "DONE seed=$seed"
}

# Batch 1: seeds 0 and 1. Batch 2: seeds 3 and 4.
run_seed 0 0 16-23 & P1=$!
run_seed 1 1 24-31 & P2=$!
wait $P1 $P2

run_seed 3 0 16-23 & P3=$!
run_seed 4 1 24-31 & P4=$!
wait $P3 $P4

echo "ALL SEEDS DONE — aggregating"
"$PY" "$HERE/aggregate_paper_table.py"
