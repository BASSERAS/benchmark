#!/usr/bin/env bash
# Score each retune seed THE MOMENT it finishes, not after all five.
#
# Operator instruction, verbatim:
#   "give priority to the 5 seeds iu mean when the first seed land do
#    immediately measures on it and if it improves the reference model clearly
#    do not do the sweep but it does not do the sweep pls and queue it pls"
#
# So the loop is completion-ordered, not seed-ordered: whichever seed writes
# COMPLETE.json first gets selected, generated and scored immediately, and its
# verdict is on disk within ~15 min of that seed ending rather than ~60 min
# later. Seeds 1-3 finish ~30 min before seeds 0 and 4 (which share GPU 0), so
# this buys a real head start, not a cosmetic one.
#
# Why per-seed results go to a separate directory
# -----------------------------------------------
# compute_all_multiasset.py writes metrics_summary.csv into --results-dir with
# ONE column per scored seed. A per-seed run therefore produces a summary with a
# single seed column, and running it five times into the same directory would
# leave only the last seed. Early runs land in early/seed_<S>/; the final
# 5-seed run at the end writes the real aggregate into $O/.
#
# The banks themselves are shared: generated_paths/seed_<S>/ under $O, written
# once, read by both the early single-seed run and the final aggregate. No path
# is generated twice.
#
# Nothing here touches published artefacts. Weights are READ from $O
# (--weights-root), banks WRITTEN to $O (--out-root), and the committed
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
mkdir -p "$O/logs" "$O/early"
cd "$B"

BEFORE=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)

DONE=""
FIRST=1

is_done () { case " $DONE " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------------------
# Completion-ordered loop. train_true.py writes COMPLETE.json as its LAST
# action, after the weights and the config, so that file is the only signal
# meaning "seed S is finished AND its artefacts are readable".
# ---------------------------------------------------------------------------
while [ "$(echo $DONE | wc -w)" -lt 5 ]; do
    PROGRESS=0
    for S in 0 1 2 3 4; do
        is_done "$S" && continue
        [ -f "$O/runs/seed_$S/COMPLETE.json" ] || continue

        echo ""
        echo "======================================================================"
        echo "== seed $S finished at $(date -Is) -- scoring it now =="
        echo "======================================================================"

        # -- select ---------------------------------------------------------
        # Per-seed invocation. The NNratio denominator is loaded from the same
        # dataset files every time, so the ruler is identical across seeds even
        # though the process is not shared.
        GPU=$(( S % 4 ))
        LO=$(( 16 + S * 6 )); HI=$(( LO + 5 ))
        CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
        OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c $LO-$HI "$PY" "$C/select_checkpoint_true.py" \
            --seeds "$S" --device cuda:0 \
            --run-root "$O/runs" --out-root "$O" \
            2>&1 | tee "$O/logs/selection_seed_$S.log"

        [ -f "$O/weights/seed_${S}_model.pt" ] \
            || { echo "ABORT: no selected weights for seed $S"; exit 1; }

        # -- generate -------------------------------------------------------
        CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
        OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c $LO-$HI "$PY" "$C/generate_bank_true.py" \
            --seed "$S" --device cuda:0 --num-paths 6144 \
            --data-dir "$V" --seq-tag "$TAG" \
            --weights-root "$O" --out-root "$O" \
            > "$O/logs/gen_seed_$S.log" 2>&1 \
            || { echo "ABORT: generation failed seed $S"; \
                 tail -25 "$O/logs/gen_seed_$S.log"; exit 1; }

        [ -f "$O/generated_paths/seed_$S/generated_paths_$TAG.npy" ] \
            || { echo "ABORT: bank seed $S missing"; exit 1; }

        # -- score this seed alone ------------------------------------------
        # --seed-list exists precisely so a subset can be scored; the default
        # range(--seeds) would demand all five banks.
        mkdir -p "$O/early/seed_$S"
        CUDA_VISIBLE_DEVICES=$GPU \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        taskset -c 24-55 "$PY" metrics/compute_all_multiasset.py \
            --method "Deep-MKV-TS-lr6.4e-06-seed$S" --dataset TrueDataset \
            --seed-list "$S" --gen-root "$O" --data-dir "$V" \
            --seq-tag "$TAG" --dt "$DT" --results-dir "$O/early/seed_$S" \
            > "$O/logs/metrics_seed_$S.log" 2>&1 \
            || { echo "ABORT: metrics failed seed $S"; \
                 tail -30 "$O/logs/metrics_seed_$S.log"; exit 1; }

        # -- verdict --------------------------------------------------------
        "$PY" "$C/compare_retune_early.py" --seed "$S" \
            --retune-root "$O/early/seed_$S" \
            2>&1 | tee "$O/logs/verdict_seed_$S.log"

        if [ "$FIRST" -eq 1 ]; then
            cp "$O/logs/verdict_seed_$S.log" "$O/FIRST_VERDICT_seed_$S.log"
            echo "*** FIRST VERDICT WRITTEN: $O/FIRST_VERDICT_seed_$S.log ***"
            FIRST=0
        fi

        DONE="$DONE $S"
        PROGRESS=1
    done

    if [ "$(echo $DONE | wc -w)" -lt 5 ] && [ "$PROGRESS" -eq 0 ]; then
        # Nothing new finished. Before sleeping, make sure the seeds we are
        # still waiting on are actually alive -- a crashed worker leaves no
        # COMPLETE.json and would hang this loop forever.
        for S in 0 1 2 3 4; do
            is_done "$S" && continue
            if ! pgrep -f "train_true.py --seed $S --steps" > /dev/null 2>&1; then
                sleep 30
                [ -f "$O/runs/seed_$S/COMPLETE.json" ] || {
                    echo "ABORT: seed $S has no process and no COMPLETE.json"
                    tail -20 "$O/logs/train_seed_$S.log"; exit 1; }
            fi
        done
        sleep 60
    fi
done

# ---------------------------------------------------------------- final ----
echo ""
echo "== all 5 seeds scored -- full 5-seed aggregate =="
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 16-55 "$PY" metrics/compute_all_multiasset.py \
    --method "Deep-MKV-TS-lr6.4e-06-diagnostic" --dataset TrueDataset --seeds 5 \
    --gen-root "$O" --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$O" \
    > "$O/logs/metrics.log" 2>&1

[ -f "$O/metrics_summary.csv" ] \
    || { tail -30 "$O/logs/metrics.log"; echo "metrics_summary.csv not produced"; exit 1; }

for S in 0 1 2 3 4; do
    "$PY" "$C/compare_retune_early.py" --seed "$S" --retune-root "$O" \
        2>&1 | tail -6 | tee -a "$O/logs/verdict_aggregate.log"
done

AFTER=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] \
    || { echo "ABORT: published weights were modified ($BEFORE -> $AFTER)"; exit 1; }
echo "published weights unchanged: $AFTER"

echo "RETUNE PROGRESSIVE DONE -- $O/metrics_summary.csv"
