#!/bin/bash
# Stage 3 of the hill-climb: h (SBTS kernel bandwidth), GPU 1 half.
#
# Disjoint from run_stage_h_a.sh (GPU 0, h in {0.20, 0.31, 0.46}) so the two
# chains can never write the same arm file.  Together they give a 6-point
# grid straddling the SBTS-fitted h = 0.31 with four interior arms, which is
# what the "smoothed" selection rule needs to be meaningful.
#
# Same precondition as the GPU 0 half: lr and ridge_lambda come from
# sweep/incumbent.json, not the command line.
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
mkdir -p "$L"
for h in 0.25 0.38 0.55; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage h --h "$h" --steps 100 --device cuda:0 \
      > "$L/h_${h}.log" 2>&1
done
