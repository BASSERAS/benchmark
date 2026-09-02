#!/bin/bash
# Longer-horizon (750-step) confirmation of the lr screen, GPU 1 half.
#
# Disjoint from run_stage_lrlong_a.sh (GPU 0, lr in {1e-5, 2.5e-5}) so the two
# chains can never write the same arm file.  This half asks the other half of
# the question: does the flat high side (5e-5, 1e-4) hold its ranking once the
# small-lr arms have had time to converge?
#
# Waits for chain H's last 250-step arm (lr = 7e-5) before touching GPU 1, so
# the confirmation never contends with the screen still in flight.
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
until grep -q '^\[out\]' "$L/lr_7e-5.log" 2>/dev/null; do sleep 30; done
for lr in 5e-5 1e-4; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lrlong --lr "$lr" --steps 750 --device cuda:0 \
      > "$L/lrlong_${lr}.log" 2>&1
done
