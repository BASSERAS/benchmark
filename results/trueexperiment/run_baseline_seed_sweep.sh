#!/usr/bin/env bash
# Table C baselines over 5 resampling seeds instead of 1.
#
# Why: the two bootstrap rows were reported at a single seed (the author's 1234)
# while every method row carries a 5-seed sd, so the table compared a point
# estimate against a distribution. block_bootstrap and session_bootstrap ARE
# stochastic in --seed (conditional_crps_multiasset.py:449 and :464 both build a
# fresh default_rng(seed)), so a spread here is real, not manufactured.
#
# NOT covered here, deliberately: the real_train_bank row. That bank is
# np.load()'d off disk (line 513) and score() contains no RNG at all -- the only
# three RNG sites in the module are block_bootstrap, session_bootstrap and the
# fixed-seed CI in _mean_ci. Running it under five seeds returns one number five
# times. Its honest analogue is a different real split used as the bank
# (val / valdisc / disc), which is a separate question and a semantic change to
# what "the floor" means, so it is not silently bundled in here.
#
# Seed set: the author's 1234 first (so the existing published number is
# reproduced and can be checked against the old artefacts), then 1235..1238.
#
# Run detached:
#   cd /home/tbasseras/benchmark
#   setsid nohup bash results/trueexperiment/run_baseline_seed_sweep.sh \
#       > results/trueexperiment/baselines/sweep.log 2>&1 & disown
set -euo pipefail

REPO=/home/tbasseras/benchmark
cd "$REPO"

PY=/home/tbasseras/sbts-venv/bin/python
V=$REPO/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
OUT=$REPO/results/trueexperiment/baselines/crps_configs
SEEDS=(1234 1235 1236 1237 1238)

# 5 concurrent x 3 threads = 15 cores, inside the 16-core cap on this shared box.
THREADS=3

mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== table C baseline seed sweep: ${SEEDS[*]} ==="

run_cfg() {   # $1 cfgname  $2 weight-mode  $3 standardize
    local cfg=$1 wmode=$2 std=$3
    say "--- config '$cfg' (weight-mode=$wmode standardize=$std) ---"
    local i=0
    for S in "${SEEDS[@]}"; do
        local out=$OUT/${cfg}__bseed_${S}.json
        if [ -f "$out" ]; then
            say "  seed $S exists -- skipping"
            i=$((i + 1))
            continue
        fi
        local lo=$((i * THREADS)) hi=$((i * THREADS + THREADS - 1))
        # --baselines-only skips the method bank entirely (line 510), so --bank is
        # not required and only the two bootstraps are scored.
        CUDA_VISIBLE_DEVICES="" \
        OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
        taskset -c "${lo}-${hi}" \
        "$PY" metrics/conditional_crps_multiasset.py \
            --data-dir "$V" --seq-tag "$TAG" --bank-size 8192 \
            --baselines-only --seed "$S" \
            --weight-mode "$wmode" --standardize "$std" \
            --out "$out" > "$OUT/${cfg}__bseed_${S}.log" 2>&1 &
        say "  seed $S launched on cores ${lo}-${hi} (pid $!)"
        i=$((i + 1))
    done
    wait
    say "  config '$cfg' done"
}

run_cfg paper  paper  bank
run_cfg perdim perdim realtrain

say "SWEEP_COMPLETE $(date -Is)"
