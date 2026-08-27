#!/usr/bin/env bash
# Fine alpha sweep over the window where vol_ratio crosses 1.0.
#
# The coarse sweep (run_alpha_ablation.sh) showed vol_ratio jumping 1.33 -> 0.52
# between alpha=0 and alpha=0.02 on every seed: the learned control pushes
# volatility in the RIGHT direction (the reference over-vols by 33%) but
# overshoots the target by a factor of ~2.  Seed 3 happened to land near the
# crossing at alpha=0.02 and is the only ADMISSIBLE point in the whole study.
#
# This resolves that window on all five seeds, to answer: does a correctly
# scaled control clear the real-vs-real envelope on TrueDataset, or was seed 3
# a fluke?
#
# GPU 3 only (jyoussef owns 0,1,2 and also has a job on 3; 53 GB free there and
# this needs ~2 GB).  Cores 16-63; 0-15 belong to the concurrent metrics job.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
D=$B/results/trueexperiment/Deep-MKV-TS/diagnostic_bestfit

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$D/logs"
cd "$B"

ALPHAS="0.001 0.002 0.004 0.007 0.01 0.015 0.02 0.03"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
taskset -c 16-39 "$PY" "$C/alpha_ablation.py" --seeds 0 1 2 --device cuda:0 \
    --alphas $ALPHAS --tag fine > "$D/logs/alpha_fine_a.log" 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
taskset -c 40-63 "$PY" "$C/alpha_ablation.py" --seeds 3 4 --device cuda:0 \
    --alphas $ALPHAS --tag fine > "$D/logs/alpha_fine_b.log" 2>&1 &
PID_B=$!

FAIL=0
wait $PID_A || { tail -25 "$D/logs/alpha_fine_a.log"; echo "fine seeds 0-2 FAILED"; FAIL=1; }
wait $PID_B || { tail -25 "$D/logs/alpha_fine_b.log"; echo "fine seeds 3-4 FAILED"; FAIL=1; }
[ "$FAIL" -eq 0 ] || exit 1
echo "ALPHA FINE SWEEP DONE"
