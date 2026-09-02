#!/bin/bash
# Stage 3 of the hill-climb: h (SBTS kernel bandwidth), GPU 0 half.
#
# h = 0.31 is the value fitted by the SBTS reference generator, so the grid
# brackets it on both sides rather than searching outward from an endpoint.
# lr and ridge_lambda are NOT passed: every arm reads sweep/incumbent.json
# (lr = 2.5e-05, ridge_lambda = 10), so this must run AFTER both earlier
# stages were promoted.
#
# Caution when reading the results: the kernel has compact support
# ||u|| < h, so a small h can zero out every bank path and collapse the drift
# to 0.  Such an arm is degenerate, not merely bad -- check ce_r2 and the
# discrepancy level before treating a small-h score as a real measurement.
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
for h in 0.20 0.31 0.46; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage h --h "$h" --steps 100 --device cuda:0 \
      > "$L/h_${h}.log" 2>&1
done
