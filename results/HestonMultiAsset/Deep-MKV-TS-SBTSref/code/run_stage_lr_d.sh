#!/bin/bash
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
for lr in 2.5e-4; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lr --lr $lr --steps 250 --device cuda:0 \
      > $L/lr_${lr}.log 2>&1
done
