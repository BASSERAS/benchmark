#!/bin/bash
# Wave 4: spend the whole machine on the one lever that moved, and on bounding
# the noise so the ranking is falsifiable.
#
# Why the wave-1 tail was killed: N_pi (60/120/250) and K (5/8/12) all landed
# inside the 0.13 run-to-run noise floor. Generation-time discretization and
# DSBM depth are therefore not what makes xi and rho under-disperse, which also
# removes the reason to keep testing early-stopping and capacity arms -- they
# probe under-training, and under-training is not the mechanism.
#
# What is left standing is safe_t, the only knob with a monotone dose-response
# (1e-2 -> 0.499, 5e-3 -> 0.359, 1e-3 -> 0.183; span 0.32 = 2.5x the floor), and
# beta, which helps but saturates (300 and 1000 differ by less than the floor).
#
# Three jobs, in priority order:
#   1. noise floor    base on seeds 1-3, safe_t=1e-3 on seeds 2-3, beta=300 on
#                     seeds 1-2. Without n>1 on the arms we actually care about,
#                     no gap difference below ~0.13 means anything.
#   2. extend safe_t  1e-4 and 2e-3, to find where the dose-response turns over.
#   3. cross the two  safe_t x beta at four corners, to see whether they add.
#
# Cores 0, 9, 12 belong to the still-running t16/t17/t09s1 and are left alone.
# The guard loop below waits for the kill to land, so this can be launched
# before or after it without oversubscribing.

set -u

ROOT=/home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation
OUT=$ROOT/results/sweep
PY=/home/tbasseras/gpu-venv/bin/python
cd "$ROOT/metric" || exit 1

echo "[wave4] waiting for the wave-1 tail to be killed (live<=3)..."
while [ "$(pgrep -c -f '[r]eproduce_heston.py --seed')" -gt 3 ]; do
    sleep 20
done
echo "[wave4] $(date '+%H:%M:%S') cores free, launching 13 arms"

launch() {
    tag=$1; core=$2; gpu=$3; seed=$4; shift 4
    echo "[wave4] $tag -> GPU $gpu core $core seed $seed  extra: $*"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c "$core" "$PY" reproduce_heston.py \
        --seed "$seed" --out "$OUT" --tag "$tag" \
        --M-simu 1000 --mle-jobs 1 "$@" > "$OUT/$tag.log" 2>&1 &
}

# -- 1. noise floor: replicates of the control and of the two leaders --
launch t00s1  1 0 1
launch t00s2  2 1 2
launch t00s3 14 1 3
launch t09s2  3 0 2 --set safe_t=1e-3
launch t09s3 15 0 3 --set safe_t=1e-3
launch t11s1  4 1 1 --set beta=300
launch t11s2  5 0 2 --set beta=300

# -- 2. extend the safe_t dose-response past the current end points --
launch t18    6 1 0 --set safe_t=1e-4
launch t19    7 0 0 --set safe_t=2e-3

# -- 3. cross safe_t with beta --
launch t20    8 1 0 --set safe_t=5e-4 --set beta=300
launch t21   10 0 0 --set safe_t=1e-3 --set beta=1000
launch t22   11 1 0 --set safe_t=1e-4 --set beta=300
launch t23   13 0 0 --set safe_t=5e-4 --set beta=1000

wait
echo "[wave4] $(date '+%H:%M:%S') done"
"$PY" "$ROOT/sweep_paper.py" board --seeds 0,1,2,3
