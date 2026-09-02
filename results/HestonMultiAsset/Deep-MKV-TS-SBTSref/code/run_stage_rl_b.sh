#!/bin/bash
# Stage 2 of the hill-climb: ridge_lambda, GPU 1 half of a 6-point log grid.
#
# Disjoint from run_stage_rl_a.sh (GPU 0, rl in {1, 100, 10000}) so the two
# chains can never write the same arm file.  Same precondition: the lr winner
# must already be promoted into sweep/incumbent.json, because lr is read from
# there rather than passed on the command line.
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
for rl in 10 1000 100000; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage ridge_lambda --ridge-lambda "$rl" --steps 100 --device cuda:0 \
      > "$L/rl_${rl}.log" 2>&1
done
