#!/bin/bash
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
until grep -q '^\[out\]' $L/lr_2.5e-5.log 2>/dev/null; do sleep 30; done
for lr in 3.5e-5 7e-5; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lr --lr $lr --steps 250 --device cuda:0 \
      > $L/lr_${lr}.log 2>&1
done
