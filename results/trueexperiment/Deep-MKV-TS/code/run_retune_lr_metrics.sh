#!/usr/bin/env bash
# Score the lr=6.4e-06 retune: wait for training, select, generate, measure.
#
# Why the wait is on COMPLETE.json and not on PIDs
# ------------------------------------------------
# The five trainings were launched by two different parents (run_retune_lr.sh
# for seeds 1-4, run_retune_lr_seed0.sh for seed 0 after its OOM), so this
# script is not their parent and cannot `wait` on them. train_true.py writes
# runs/seed_<S>/COMPLETE.json as its LAST action, after the weights and the
# config, so that file appearing is the only signal that means "seed S is
# finished AND its artefacts are on disk".
#
# What the first diagnosis actually is
# ------------------------------------
# select_checkpoint_true.py prints, per seed, the vol error and corr error of
# EVERY checkpoint on the 500..3000 grid before it picks one. That table is the
# first real read on whether lowering lr helped, and it arrives seconds after
# training ends -- no bank generation needed. It is echoed to stdout here and
# kept in $O/selection/seed_<S>_selection.json.
#
# The full suite follows because vol_err cannot see an ACF, and the ACF is where
# the alpha=1 run was destroyed (B_acf_sq_r_der 122x worse than doing no
# learning at all, even after the best post-hoc rescale). A retune is a
# different optimisation trajectory, not a rescale, so that number has to be
# re-measured rather than assumed.
#
# Nothing here touches published artefacts. Weights are READ from $O via
# --weights-root, banks are WRITTEN to $O via --out-root, and the committed
# weights/seed_<S>_model.pt are md5-checked at the end.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
M=$B/results/trueexperiment/Deep-MKV-TS
O=$M/retune_lr
V=$B/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
DT=9.512937595129376e-07

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$O/logs"
cd "$B"

BEFORE=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)

# ---------------------------------------------------------------- stage 0 ----
echo "== stage 0: wait for all 5 trainings =="
for S in 0 1 2 3 4; do
    F="$O/runs/seed_$S/COMPLETE.json"
    while [ ! -f "$F" ]; do
        # A dead worker leaves no COMPLETE.json and no running process; without
        # this the loop would spin forever on a crashed seed.
        if ! pgrep -f "train_true.py --seed $S --steps" > /dev/null 2>&1; then
            sleep 20
            [ -f "$F" ] || { echo "ABORT: seed $S has no process and no COMPLETE.json"; \
                             tail -20 "$O/logs/train_seed_$S.log"; exit 1; }
        fi
        sleep 60
    done
    echo "   seed $S complete at $(date -Is)"
done

# ---------------------------------------------------------------- stage 1 ----
# One process for all 5 seeds: the NNratio denominator must be the same number
# for every seed or the per-seed objectives are measured against different
# rulers (select_checkpoint_true.py:343-346).
echo "== stage 1: checkpoint selection (this IS the first diagnosis) =="
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
taskset -c 16-23 "$PY" "$C/select_checkpoint_true.py" \
    --seeds 0 1 2 3 4 --device cuda:0 \
    --run-root "$O/runs" --out-root "$O" \
    2>&1 | tee "$O/logs/selection.log"

for S in 0 1 2 3 4; do
    [ -f "$O/weights/seed_${S}_model.pt" ] || { echo "no selected weights seed $S"; exit 1; }
done

# ---------------------------------------------------------------- stage 2 ----
echo "== stage 2: 5 A/B banks, one worker per seed =="
PIDS=(); LOGS=(); TAGS=()
for S in 0 1 2 3 4; do
    GPU=$(( S % 4 ))
    LO=$(( 16 + S * 6 )); HI=$(( LO + 5 ))
    LOG="$O/logs/gen_seed_$S.log"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    taskset -c $LO-$HI "$PY" "$C/generate_bank_true.py" \
        --seed "$S" --device cuda:0 --num-paths 6144 \
        --data-dir "$V" --seq-tag "$TAG" \
        --weights-root "$O" --out-root "$O" > "$LOG" 2>&1 &
    PIDS+=($!); LOGS+=("$LOG"); TAGS+=("gen seed $S -> gpu $GPU cores $LO-$HI")
done
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "WORKER FAILED: ${TAGS[$i]}"; tail -25 "${LOGS[$i]}"; FAIL=1; }
done
[ "$FAIL" -eq 0 ] || { echo "GENERATION FAILED"; exit 1; }

for S in 0 1 2 3 4; do
    [ -f "$O/generated_paths/seed_$S/generated_paths_$TAG.npy" ] \
        || { echo "bank seed $S missing"; exit 1; }
done
echo "   all 5 banks present"

# ---------------------------------------------------------------- stage 3 ----
# --gen-root is the directory HOLDING generated_paths/, so it is $O.
echo "== stage 3: full A1-A32 + curve B =="
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 16-55 "$PY" metrics/compute_all_multiasset.py \
    --method "Deep-MKV-TS-lr6.4e-06-diagnostic" --dataset TrueDataset --seeds 5 \
    --gen-root "$O" --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$O" \
    > "$O/logs/metrics.log" 2>&1

[ -f "$O/metrics_summary.csv" ] \
    || { tail -30 "$O/logs/metrics.log"; echo "metrics_summary.csv not produced"; exit 1; }

AFTER=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] \
    || { echo "ABORT: published weights were modified ($BEFORE -> $AFTER)"; exit 1; }
echo "published weights unchanged: $AFTER"

echo "RETUNE METRICS DONE -- $O/metrics_summary.csv"
