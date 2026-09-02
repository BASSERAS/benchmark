#!/usr/bin/env bash
# Re-sweep the SBTS bandwidth h against the CORRECTED (full-K-lag) adjoint.
#
# Why this exists.  The first 72-arm sweep promoted h = 0.36 while the adjoint
# was hard-wired to a single lag -- a gradient wrong by 100% relative error at
# K = 20.  Its choice of h is therefore not evidence about the code that now
# runs.  Worse, probe_jacobian_tail.py measured the occupancy at the promoted
# point and found the kernel weights are ONE-HOT there:
#
#     median effective bank paths, step 120, out of M = 8192
#        K \ h    0.36    0.50    0.70    1.00    1.50
#         10       2.9    1586    5565    7387    8014
#         20       1.0     2.7    1810    6355    7755
#
# so at (h = 0.36, K = 20) the "conditional expectation" b^ref is a nearest
# neighbour lookup of exactly one training path.  The grid below brackets the
# cliff on both sides instead of searching outward from it.
#
# Protocol is deliberately identical to the first sweep -- 250 steps, same
# validation discrepancy, same target -- so the new arms and the old ones sit
# on one scale.  The only changed factor is jacobian_lags = -1, which the arm
# tag records as a `_jl-1` suffix so nothing can overwrite the old arms.
#
# Three seeds per bandwidth, because the measured seed noise floor is 6.91% and
# a single seed cannot resolve an h effect smaller than that.
#
# Placement (<= 3 GPUs, <= 64 cores):
#   one GPU per SEED, three bandwidths resident at a time, two rounds.
#   Every bandwidth is therefore measured under identical sharing conditions,
#   which a round that mixed seeds and bandwidths would not guarantee.
#     GPU 0 = seed 0   cores  0-20
#     GPU 1 = seed 1   cores 21-41
#     GPU 2 = seed 2   cores 42-62
#   3 arms x 19 GiB = 57 GiB of an 80 GiB card.
set -u

R=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
L=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/resweeplogs
PY=/home/tbasseras/gpu-venv/bin/python
STEPS=250

mkdir -p "$L"
cd "$R"

cat > "$L/_worker.sh" <<'WORKER'
#!/usr/bin/env bash
# $1 = gpu index, $2 = seed, $3 = first core of this GPU's block
set -u
g="$1"; s="$2"; base="$3"
for round in "0.36 0.50 0.70" "1.00 1.50 2.00"; do
  off=0
  for h in $round; do
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
  wait        # round barrier: keeps every bandwidth on equal sharing terms
  echo "gpu $g seed $s finished round [$round]" >> "$L/_progress.txt"
done
WORKER
chmod +x "$L/_worker.sh"

: > "$L/_progress.txt"

export R L PY STEPS
setsid nohup bash -c "
  cd '$R'
  '$L/_worker.sh' 0 0 0  &
  '$L/_worker.sh' 1 1 21 &
  '$L/_worker.sh' 2 2 42 &
  wait
  echo ALL_DONE >> '$L/_progress.txt'
" > "$L/_driver.log" 2>&1 < /dev/null &
disown || true

echo "re-sweep launched: 18 arms (h in {0.36,0.50,0.70,1.00,1.50,2.00} x seeds {0,1,2})"
echo "logs:     $L"
echo "progress: $L/_progress.txt"
