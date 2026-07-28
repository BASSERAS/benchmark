#!/usr/bin/env bash
# Full-length GATE run for the stage-2 winner recipe, chained Sine -> Stock on
# GPU2.  Winner (from hpo_stage2_shard0.jsonl):
#   znorm . linear . learn_sigma=False . ddpm_fixed . lr=4e-4 . ema=0.999(+warmup)
# 15000 steps x 3 train-seeds x 5 disc-seeds on FULL data (no subset).
set -u
cd /home/tbasseras/benchmark/methods/TimeDiT/paper_reimplementation

echo "[gate-chain] SINE start $(date -Is)"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 16-23 \
  /home/tbasseras/gpu-venv/bin/python reproduce_gate.py \
    --dataset sine --lr 4e-4 --ema 0.999 > log_gate_sine.txt 2>&1
echo "[gate-chain] SINE done $(date -Is)"

echo "[gate-chain] STOCK start $(date -Is)"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 16-23 \
  /home/tbasseras/gpu-venv/bin/python reproduce_gate.py \
    --dataset stock --lr 4e-4 --ema 0.999 > log_gate_stock.txt 2>&1
echo "[gate-chain] STOCK done $(date -Is)"

echo "GATE_CHAIN_DONE" >> hpo_status.txt
echo "[gate-chain] ALL DONE $(date -Is)"
