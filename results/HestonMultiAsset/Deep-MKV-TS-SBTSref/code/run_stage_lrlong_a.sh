#!/bin/bash
# Longer-horizon (750-step) confirmation of the lr screen, GPU 0 half.
#
# WHY: the 250-step screen shows a "cliff" below lr = 2.5e-5 (1e-5 -> 0.1379,
# 2.5e-6 -> 1.0171, i.e. untrained).  That is almost certainly a screen-length
# artefact -- not enough steps for a small lr to move -- rather than a genuine
# stability wall.  If it is an artefact, lr = 1e-5 catches up at 750 steps and
# the neighbourhood penalty the "smoothed" rule charges to 2.5e-5 evaporates.
#
# The stage label is "lrlong", NOT "lr": _tag() omits the step count, so
# reusing --stage lr would silently overwrite the 250-step arm of the same lr
# and corrupt the screen table.  report() globs "lr__*.json", which does not
# match "lrlong__*.json".
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
for lr in 1e-5 2.5e-5; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lrlong --lr "$lr" --steps 750 --device cuda:0 \
      > "$L/lrlong_${lr}.log" 2>&1
done
