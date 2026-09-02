#!/usr/bin/env bash
# $1 = gpu index, $2 = seed, $3 = first core of this GPU's block
set -u
g="$1"; s="$2"; base="$3"
off=0
for h in 0.28 0.31 0.33; do
  c="$((base + off * 7))-$((base + off * 7 + 6))"
  CUDA_VISIBLE_DEVICES="$g" \
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 \
  taskset -c "$c" "$PY" "$R/sweep_hyperparams.py" \
      --stage hfix \
      --h "$h" \
      --jacobian-lags -1 \
      --weight-grad-mode analytic \
      --seed "$s" \
      --steps "$STEPS" \
      --device cuda:0 \
      > "$L/h${h}_s${s}.log" 2>&1 &
  off=$((off + 1))
done
wait
echo "gpu $g seed $s finished LOW round [0.28 0.31 0.33]" >> "$L/_progress_low.txt"
