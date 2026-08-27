#!/usr/bin/env bash
# Head-to-head: untrained reference SDE vs trained Deep-MKV-TS, both scored on
# the SAME real-vs-real envelope.  Pure numpy, no GPU.
#
# Cores 16-31: cores 0-15 and GPU 3 belong to the concurrent
# run_diagnostic_bestfit.sh metrics stage while it is alive.
set -euo pipefail

B=/home/tbasseras/benchmark
# gpu-venv, NOT .cc-venv: the criterion module imports SBTS's numba kernels, and
# gpu-venv is the interpreter select_checkpoint_true.py itself ran on, so these
# numbers land on exactly the same axis as logs/select_seed_*.log.
# CUDA_VISIBLE_DEVICES is emptied because none of this touches a GPU.
CC=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
D=$B/results/trueexperiment/Deep-MKV-TS/diagnostic_bestfit

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$D/logs"
cd "$B"

CUDA_VISIBLE_DEVICES="" NUMBA_NUM_THREADS=16 \
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
taskset -c 16-31 "$CC" "$C/score_ref_vs_mkv.py" 2>&1 | tee "$D/logs/ref_vs_mkv.log"
