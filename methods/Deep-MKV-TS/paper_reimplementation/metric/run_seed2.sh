#!/usr/bin/env bash
# Add seed 2 to the Heston campaign.
#
# The paper reports medians over seeds 0, 1, 3, 4 -- seed 2 is not in the paper's
# set, so run_reproduction.sh never runs it. GUIDELINE.md 4 requires the benchmark
# 5-seed set 0..4, so this script fills the one gap with the *identical*
# configuration, so seed 2 is interchangeable with the other four.
#
# Two invariants this script exists to protect:
#   1. BENCHMARK_HESTON_EVAL is deliberately left UNSET, so seed 2 scores on
#      heston_S_disc_8192x128.npy exactly like seeds 0/1/3/4. Setting it would
#      silently put seed 2 on a different split than the seeds it is pooled with.
#   2. It waits for every other training process to exit before starting. Two
#      trainers writing the same run tree corrupts both, and the GUIDELINE 4.1
#      cap is 2 GPUs -- the reproduction already holds both.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_ROOT="$(dirname "$HERE")"
REF="$PR_ROOT/../code/reference"
PY=/home/tbasseras/gpu-venv/bin/python
RUNS="$PR_ROOT/runs"
OUT="$RUNS/seed_2"
STEPS="${STEPS:-3000}"
SEED=2
GPU="${GPU:-0}"
CORES="${CORES:-16-23}"

if [[ -f "$OUT/COMPLETE.json" ]]; then
  echo "SKIP seed $SEED (already complete)"
  exit 0
fi

# Block until the reproduction's own trainers are gone. `pgrep -f` on the script
# name matches only the trainer processes, not this wrapper.
echo "waiting for a free GPU slot (other trainers must exit first)..."
while pgrep -f run_matched_control_synthetic_validation.py > /dev/null; do
  sleep 60
done
echo "slot free at $(date -Is), launching seed $SEED"

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PYTHONPATH="$REF/src:$REF/experiments:$REF/experiments/scripts" \
taskset -c "$CORES" "$PY" \
  "$REF/experiments/scripts/run_matched_control_synthetic_validation.py" \
  --task heston --run-dir "$OUT" --device cuda:0 \
  --protocol-manifest "$HERE/PROTOCOL.json" \
  --seed "$SEED" --bank-size 8192 --sample-batch-size 2048 \
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
  > "$OUT/console.log" 2>&1

echo "{\"seed\": $SEED, \"steps\": $STEPS}" > "$OUT/COMPLETE.json"
echo "DONE seed=$SEED at $(date -Is)"
