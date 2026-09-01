#!/bin/bash
# Wave 5: is beta=300 real, and is it the optimum?
#
# beta=300 leads the board on `corr` (leverage-spread ratio) at 0.905 against a
# base of 0.785. That is a 0.120 margin on a seed noise of 0.084 -- 1.4 sigma,
# on n=1. No further arm can fix that; only replicates can. So:
#
#   t11s1..3     beta=300 on seeds 1-3  -- is the margin reproducible?
#   t24 t25 t26  beta = 150, 200, 500   -- bracket 300. If corr is flat across
#                150-500 then beta is a threshold, not a tuned optimum, and
#                there is nothing left to sweep.
#
# Runs on cores 0, 7, 9, 10, 11, 12, which are free now. Cores 1-4 and 13-15
# still hold the wave-1 tail, and 5, 6, 8 hold the base replicates from wave 2.
# Nothing here waits: the cores are already idle.

set -u

ROOT=/home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation
OUT=$ROOT/results/sweep
PY=/home/tbasseras/gpu-venv/bin/python
cd "$ROOT/metric" || exit 1

launch() {
    tag=$1; core=$2; gpu=$3; seed=$4; shift 4
    echo "[wave5] $tag -> GPU $gpu core $core seed $seed  extra: $*"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c "$core" "$PY" reproduce_heston.py \
        --seed "$seed" --out "$OUT" --tag "$tag" \
        --M-simu 1000 --mle-jobs 1 "$@" > "$OUT/$tag.log" 2>&1 &
}

# -- replicates of the leader: the only thing that can confirm the margin --
launch t11s1  0 0 1 --set beta=300
launch t11s2  7 1 2 --set beta=300
launch t11s3  9 0 3 --set beta=300

# -- bracket the optimum --
launch t24   10 1 0 --set beta=150
launch t25   11 0 0 --set beta=200
launch t26   12 1 0 --set beta=500

wait
echo "[wave5] $(date '+%H:%M:%S') done"
"$PY" "$ROOT/sweep_paper.py" board --seeds 0,1,2,3
