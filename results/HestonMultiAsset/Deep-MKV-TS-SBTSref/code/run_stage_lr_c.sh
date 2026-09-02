#!/bin/bash
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
for lr in 1e-4 8e-3; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lr --lr $lr --steps 250 --device cuda:0 \
      > $L/lr_${lr}.log 2>&1
done
