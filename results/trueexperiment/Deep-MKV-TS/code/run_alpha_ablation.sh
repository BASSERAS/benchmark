#!/usr/bin/env bash
# alpha-ablation: Theta(alpha) = eta*sigma_ref^-1 + alpha*Z/sqrt(dt).
# alpha=0 is the untrained reference, alpha=1 is the trained model.
#
# GPU 3 ONLY.  GPUs 0, 1 and 2 are running another user's (jyoussef) jobs at the
# time of writing, and this work does not need them: generation is ~1.2 s per
# bank, so all 45 banks are about a minute of GPU. Cores 16-63; cores 0-15
# belong to the concurrent A-table metrics job.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
D=$B/results/trueexperiment/Deep-MKV-TS/diagnostic_bestfit

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$D/logs"
cd "$B"

echo "== alpha sweep: seeds 0-2 and seeds 3-4, both on GPU 3 =="
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMBA_NUM_THREADS=24 \
taskset -c 16-39 "$PY" "$C/alpha_ablation.py" --seeds 0 1 2 --device cuda:0 \
    > "$D/logs/alpha_a.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMBA_NUM_THREADS=24 \
taskset -c 40-63 "$PY" "$C/alpha_ablation.py" --seeds 3 4 --device cuda:0 \
    > "$D/logs/alpha_b.log" 2>&1 &
PID_B=$!

FAIL=0
wait $PID_A || { tail -20 "$D/logs/alpha_a.log"; echo "alpha seeds 0-2 FAILED"; FAIL=1; }
wait $PID_B || { tail -20 "$D/logs/alpha_b.log"; echo "alpha seeds 3-4 FAILED"; FAIL=1; }
[ "$FAIL" -eq 0 ] || exit 1
echo "   alpha sweep done"

# The Theta-term probe is a separate, single-seed run: it adds an eigvalsh per
# control call, so folding it into the sweep would slow every row to measure a
# quantity that does not depend on the seed. 512 paths is plenty for a norm.
echo "== Theta term-norm probe, seed 0, GPU 3 =="
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMBA_NUM_THREADS=24 \
taskset -c 16-39 "$PY" "$C/alpha_ablation.py" --seeds 0 --device cuda:0 \
    --probe --num-paths 512 \
    > "$D/logs/alpha_probe.log" 2>&1 \
    || { tail -20 "$D/logs/alpha_probe.log"; echo "theta probe FAILED"; exit 1; }

echo "ALPHA ABLATION DONE"
