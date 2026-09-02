#!/bin/bash
set -u
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=../losses/sweeplogs
# wait for the 8e-3 arm to release GPU 0
until grep -q '^\[out\]' $L/lr_8e-3.log 2>/dev/null; do sleep 30; done
for lr in 1e-5 2.5e-6 5e-7; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
      --stage lr --lr $lr --steps 250 --device cuda:0 \
      > $L/lr_${lr}.log 2>&1
done
