#!/bin/bash
# Wave 3: does the winning pair add, and is the winner real?
#
# safe_t=1e-3 (gap 0.183) and beta=300 (gap 0.187) each closed ~0.31 of the
# base gap of 0.499, and they act on different mechanisms: safe_t sets how close
# to T the Brownian bridge is ever sampled, beta scales the inverse transport
# map. Wave 1 tests neither together, so:
#
#   t16    safe_t=1e-3 + beta=300   -- do the two effects add?
#   t17    safe_t=5e-4              -- does pushing the better knob further help?
#   t09s1  safe_t=1e-3 on seed 1    -- is the leader real, or a lucky draw off
#                                      the ~0.13 run-to-run noise floor?
#
# Runs on cores 0, 9, 12, which trials t00/t09/t12 have already vacated. Cores
# 5, 6 and 8 are deliberately left alone: 5 and 6 still hold K=8/K=12, and 8 is
# reserved by wave2_noise.sh for the base replicates.

set -u

ROOT=/home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation
OUT=$ROOT/results/sweep
PY=/home/tbasseras/gpu-venv/bin/python
cd "$ROOT/metric" || exit 1

launch() {
    tag=$1; core=$2; gpu=$3; seed=$4; shift 4
    echo "[wave3] $tag -> GPU $gpu core $core seed $seed  extra: $*"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c "$core" "$PY" reproduce_heston.py \
        --seed "$seed" --out "$OUT" --tag "$tag" \
        --M-simu 1000 --mle-jobs 1 "$@" > "$OUT/$tag.log" 2>&1 &
}

launch t16   0 0 0 --set safe_t=1e-3 --set beta=300
launch t17   9 1 0 --set safe_t=5e-4
launch t09s1 12 1 1 --set safe_t=1e-3

wait
echo "[wave3] $(date '+%H:%M:%S') done"
"$PY" "$ROOT/sweep_paper.py" board --seeds 0,1,2,3
