#!/bin/bash
# Wave 2: seed noise floor for the paper sweep.
#
# Wave 1 ranks 16 arms on a single seed. That ranking is unfalsifiable until we
# know how much `gap` moves when only the seed changes, so this replicates the
# `base` arm on seeds 1-3. The dataset is loaded from disk (reproduce_heston.py
# line 110), so --seed perturbs training init and generation noise only, never
# the data -- which is exactly the quantity the ranking needs bounded.
#
# It waits rather than launching immediately: wave 1 owns all 16 cores. The
# long-patience tail is 7 trials (t01 t02 t03 t04 t13 t14 t15), so once the live
# count drops to 7 the fast group has given its cores back and we can move in
# without oversubscribing anyone.

set -u

ROOT=/home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation
OUT=$ROOT/results/sweep
PY=/home/tbasseras/gpu-venv/bin/python
cd "$ROOT/metric" || exit 1

echo "[wave2] waiting for wave-1 fast group to finish (live<=7)..."
while [ "$(pgrep -c -f '[r]eproduce_heston.py --seed')" -gt 7 ]; do
    sleep 60
done
echo "[wave2] $(date '+%H:%M:%S') cores free, launching base replicates"

# Cores 5, 8, 6 belong to the fast group (t05, t08, t06) and are idle by now;
# the tail holds 1-4 and 13-15. GPUs alternate to keep both cards loaded.
i=0
for seed in 1 2 3; do
    case $i in
        0) core=5;  gpu=0 ;;
        1) core=8;  gpu=1 ;;
        2) core=6;  gpu=0 ;;
    esac
    i=$((i + 1))
    echo "[wave2] seed $seed -> GPU $gpu core $core"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c $core "$PY" reproduce_heston.py \
        --seed "$seed" --out "$OUT" --tag "t00s$seed" \
        --M-simu 1000 --mle-jobs 1 > "$OUT/t00s$seed.log" 2>&1 &
done

wait
echo "[wave2] $(date '+%H:%M:%S') all base replicates done"
"$PY" "$ROOT/sweep_paper.py" board --seeds 0,1,2,3
