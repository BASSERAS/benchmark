#!/usr/bin/env bash
# _recompute_group.sh GPU CORES METHOD [METHOD ...]
# Sequentially runs, per method: compute_all (A1-A34+B, GPU for A18/A19),
# recompute_curve_b (B aggregate mse/pct/nrmse), plot_diagnostics (8-panel + theory).
set -euo pipefail
GPU="$1"; CORES="$2"; shift 2
cd "$(dirname "$0")"
PY=/home/tbasseras/gpu-venv/bin/python
for M in "$@"; do
  echo "############## [$GPU] $M — compute_all ##############"
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    taskset -c "$CORES" "$PY" compute_all.py --method "$M" --dataset Heston --seeds 5
  echo "############## [$GPU] $M — recompute_curve_b ##############"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    taskset -c "$CORES" "$PY" recompute_curve_b.py --method "$M" --seeds 5
  echo "############## [$GPU] $M — plot_diagnostics ##############"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    taskset -c "$CORES" "$PY" plot_diagnostics.py --method "$M" --dataset Heston --seed 0
done
echo "############## [$GPU] GROUP DONE ##############"
