#!/usr/bin/env bash
# Conditional-CRPS (comparison table C) for the untrained reference SDE.
#
# Why
# ---
# The reference was published with tables A and B complete but table C rendering
# as a "-" row, because it had no losses/crps_configs/paper__*.json. Table C is
# the only conditional metric in the comparison: A and B ask "does the marginal
# and the curve shape look right", C asks "given a real history, is the
# generator's continuation right". A baseline that is absent from C cannot tell
# us whether a learned control improves the CONDITIONAL law or only the
# unconditional one -- which is exactly the question the A/B split raised.
#
# The bank-size trap
# ------------------
# SBTS and CSDI do NOT score CRPS on their 6144-path A/B bank. They score on a
# dedicated 8192-path pool under crps_banks/ (CSDI/code/run_pipeline.sh:163-167,
# --bank-size 8192). CRPS here is a k-NN retrieval score with k=256, so a
# smaller pool is a strictly harder retrieval problem. Scoring the reference on
# its 6144-path A/B bank would have handed it a 25% smaller pool and produced a
# row that looks worse for a reason that has nothing to do with the method.
# Hence stage 1: generate the missing 8192-path pool first.
#
# --out-root exists for the same reason: without it the 8192 pool would be
# written into reference/generated_paths/seed_<S>/ and overwrite the published
# A/B bank's metadata.json (bank_role "ab_bank") and generation_time CSVs.
#
# Two configs, exactly as SBTS/CSDI
# ---------------------------------
#   paper  = --weight-mode paper  --standardize bank      (the reported row)
#   perdim = --weight-mode perdim --standardize realtrain  (the README caveat)
# plus __realbank, the real TRAIN split used as its own bank -- the reference
# every row in table C is read against. It is method-independent (verified: the
# SBTS and CSDI copies agree on every score, differing only in the
# block/session bootstrap `_diag/mean_reuse` diagnostic), but it is recomputed
# here rather than copied so this folder is self-contained.
#
# Parallelisation
# ---------------
# conditional_crps_multiasset.py is pure numpy -- no torch, no numba, no CUDA --
# so the axis is cores, not GPUs. Cores 16-55 belong to the concurrent
# compute_all_multiasset.py job and are left alone; this uses 56-91.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
M=$B/results/trueexperiment/reference
V=$B/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
CRPS_TAG=8192x128x8

mkdir -p "$M/logs" "$M/losses/crps_configs"
cd "$B"

# ---------------------------------------------------------------- stage 1 ----
echo "== stage 1: 8192-path conditional-CRPS pool, 5 seeds in parallel =="
NEED=0
for S in 0 1 2 3 4; do
    [ -f "$M/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy" ] || NEED=1
done
if [ "$NEED" -eq 0 ]; then
    echo "   all 5 pools already present -- skipping"
else
    PIDS=(); LOGS=()
    for S in 0 1 2 3 4; do
        LO=$(( 56 + S * 4 )); HI=$(( LO + 3 ))
        LOG="$M/logs/crps_bank_seed_$S.log"
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        CUDA_VISIBLE_DEVICES="" \
        taskset -c $LO-$HI "$PY" "$M/code/generate_reference_true.py" \
            --seeds "$S" --device cpu --num-paths 8192 \
            --out-root "$M/crps_banks" > "$LOG" 2>&1 &
        PIDS+=($!); LOGS+=("$LOG")
    done
    FAIL=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { tail -25 "${LOGS[$i]}"; echo "pool seed $i FAILED"; FAIL=1; }
    done
    [ "$FAIL" -eq 0 ] || exit 1
fi

for S in 0 1 2 3 4; do
    [ -f "$M/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy" ] \
        || { echo "pool seed $S missing"; exit 1; }
done
echo "   all 5 pools present"

# The A/B bank must be untouched: stage 1 writes to a different root, but this
# is the check that says so rather than the comment.
for S in 0 1 2 3 4; do
    grep -q '"bank_role": "ab_bank"' \
        "$M/generated_paths/seed_$S/metadata.json" \
        || { echo "ABORT: A/B bank metadata for seed $S was modified"; exit 1; }
done
echo "   A/B bank metadata intact"

# ---------------------------------------------------------------- stage 2 ----
# 12 jobs: (paper, perdim) x (5 seeds + realbank). Each gets 3 cores on 56-91.
echo "== stage 2: 12 CRPS jobs in parallel =="
PIDS=(); LOGS=(); TAGS=(); W=0
for CFG in "paper paper bank" "perdim perdim realtrain"; do
    set -- $CFG
    NAME=$1; WMODE=$2; STD=$3
    for S in 0 1 2 3 4 realbank; do
        if [ "$S" = realbank ]; then
            OUT="$M/losses/crps_configs/${NAME}__realbank.json"
            BANK="$V/true_S_$TAG.npy"; SIZE=6144; LABEL=real_train_bank
        else
            OUT="$M/losses/crps_configs/${NAME}__seed_$S.json"
            BANK="$M/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy"
            SIZE=8192; LABEL=reference
        fi
        LO=$(( 56 + W * 3 )); HI=$(( LO + 2 ))
        LOG="$M/logs/crps_${NAME}_${S}.log"
        OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
        CUDA_VISIBLE_DEVICES="" \
        taskset -c $LO-$HI "$PY" metrics/conditional_crps_multiasset.py \
            --data-dir "$V" --seq-tag "$TAG" --bank-size "$SIZE" --label "$LABEL" \
            --weight-mode "$WMODE" --standardize "$STD" \
            --bank "$BANK" --out "$OUT" > "$LOG" 2>&1 &
        PIDS+=($!); LOGS+=("$LOG"); TAGS+=("$NAME $S -> cores $LO-$HI")
        W=$(( W + 1 ))
    done
done
printf 'launched %d workers:\n' "${#PIDS[@]}"
printf '   %s\n' "${TAGS[@]}"

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "WORKER FAILED: ${TAGS[$i]}"; tail -25 "${LOGS[$i]}"; FAIL=1; }
done
[ "$FAIL" -eq 0 ] || { echo "CRPS FAILED"; exit 1; }

for NAME in paper perdim; do
    for S in seed_0 seed_1 seed_2 seed_3 seed_4 realbank; do
        F="$M/losses/crps_configs/${NAME}__${S}.json"
        [ -f "$F" ] || { echo "missing $F"; exit 1; }
    done
done
echo "REFERENCE CRPS DONE"
ls -la "$M/losses/crps_configs/"
