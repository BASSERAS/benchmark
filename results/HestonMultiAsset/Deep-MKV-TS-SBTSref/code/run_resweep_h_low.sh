#!/usr/bin/env bash
# Extend the corrected-adjoint bandwidth re-sweep BELOW h = 0.36.
#
# WHY.  The first re-sweep grid was {0.36, 0.50, 0.70, 1.00, 1.50, 2.00} and the
# winner came out at 0.36 -- the lowest point tested -- by an 87% margin, with
# everything at h >= 0.50 flat to within 0.6%:
#
#     h      mean val_discrepancy
#     0.36   0.070327     <- grid boundary, winner
#     0.50   0.131546
#     0.70   0.131835
#     1.00   0.132084
#     1.50   0.132296
#     2.00   0.132387
#
# That plateau is not noise, it is saturation: as h grows the kernel weights go
# uniform, b^ref collapses to the unconditional bank mean, and the reference
# stops depending on the path at all.  So the informative region is entirely to
# the LEFT of the grid, and a minimum sitting on the boundary is not an
# optimum -- reporting it as one is exactly the failure MULTIASSET_GUIDELINE
# 12.3 describes.
#
# WHY THESE THREE VALUES.  From the measured small-h endpoint at K = 20, step
# 120, 256 disjoint held-out queries:
#
#     h      alive     median n_eff   verdict
#     0.20    16/256    1.00          1-NN, 240 rows have b^ref = 0
#     0.31   238/256    3.93          near-degenerate
#     0.36   253/256   16.70          averaging
#
# Below roughly 0.25 most rows have every weight underflow and train against a
# reference drift of exactly zero, which is not a bad bandwidth but a
# meaningless run.  {0.28, 0.31, 0.33} covers the gap between that floor and
# the previous grid edge.  0.31 is also the bandwidth SBTS itself selected at
# d = 8, so it doubles as a cross-check.
#
# PROTOCOL IS UNCHANGED from run_resweep_h.sh -- 250 steps, jacobian_lags = -1,
# same validation discrepancy, three seeds -- so these arms and the six existing
# bandwidths sit on one scale and tabulate_hfix.py can rank them together.
#
# Placement (<= 3 GPUs, <= 64 cores): one GPU per SEED, all three bandwidths
# resident at once, a single round.  Same sharing conditions as before.
#     GPU 0 = seed 0   cores  0-20
#     GPU 1 = seed 1   cores 21-41
#     GPU 2 = seed 2   cores 42-62
set -u

R=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/resweeplogs
PY=/home/tbasseras/gpu-venv/bin/python
STEPS=250

mkdir -p "$L"
cd "$R" || exit 1

cat > "$L/_worker_low.sh" <<'WORKER'
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
WORKER
chmod +x "$L/_worker_low.sh"

: > "$L/_progress_low.txt"

export R L PY STEPS
setsid nohup bash -c "
  cd '$R'
  '$L/_worker_low.sh' 0 0 0  &
  '$L/_worker_low.sh' 1 1 21 &
  '$L/_worker_low.sh' 2 2 42 &
  wait
  echo ALL_DONE_LOW >> '$L/_progress_low.txt'
" > "$L/_driver_low.log" 2>&1 < /dev/null &
disown || true

echo "low-h re-sweep launched: 9 arms (h in {0.28,0.31,0.33} x seeds {0,1,2})"
echo "progress: $L/_progress_low.txt"
