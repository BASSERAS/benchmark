#!/bin/bash
# Stage 2 of the hill-climb: ridge_lambda, GPU 0 half of a 6-point log grid.
#
# lr is deliberately NOT passed on the command line: every arm reads the
# promoted incumbent from sweep/incumbent.json, so this driver must only be
# launched AFTER the lr stage has been reported with --promote.  Launching it
# early would silently screen ridge_lambda at the stale default lr.
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
for rl in 1 100 10000; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage ridge_lambda --ridge-lambda "$rl" --steps 100 --device cuda:0 \
      > "$L/rl_${rl}.log" 2>&1
done
