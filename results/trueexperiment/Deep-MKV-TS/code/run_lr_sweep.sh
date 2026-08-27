#!/usr/bin/env bash
# QUEUED, NOT ARMED. Turn the single lr point into a curve.
#
# Operator instruction, verbatim:
#   "yes queue the lr sweep after this one lands pls but give priority to the 5
#    seeds iu mean when the first seed land do immediately measures on it and if
#    it improves the reference model clearly do not do the sweep but it does not
#    do the sweep pls and queue it pls"
#
# So this file is deliberately inert until someone runs it. It must NOT be
# launched if the lr=6.4e-06 run clearly beats the untrained reference -- in
# that case the point is already good enough and the GPUs are better spent on
# the 5-seed aggregate and the CRPS pools.
#
# What is being swept and why these three points
# ----------------------------------------------
# 6.4e-06 came from a SINGLE derivation chain: the post-hoc alpha grid put the
# admissible control magnitude at ~0.0032 of the shipped one, and the lr was
# scaled by the same factor. That step is a heuristic, not a theorem -- alpha
# rescales a CONVERGED control, lr rescales a STEP SIZE, and the two coincide
# only if the optimiser lands proportionally. So the sweep brackets it by half
# an order of magnitude either way:
#
#   2.0e-06   0.31x the derived point   Theta/step = 0.00205
#   6.4e-06   the derived point         Theta/step = 0.00656   (already run)
#   2.0e-05   3.1x the derived point    Theta/step = 0.02051
#
# 2.0e-05 is close to Heston's own Theta/step of 0.03175, so it doubles as the
# "what if the dimensionless matching is the right idea but the alpha grid
# mis-calibrated it" control.
#
# Two seeds per point, not five
# -----------------------------
# A sweep answers "which lr", not "how stable is this lr". Five seeds per point
# would be 6 trainings x ~2 h on 4 GPUs = 3 waves. Two seeds x 2 new points = 4
# trainings = ONE wave, and seeds 1 and 3 are chosen because they were the two
# clean, non-degenerate seeds under the alpha study (seeds 0 and 2 fail on
# CORRELATION at every alpha tested -- 0.335 and 0.265 against the reference's
# 0.249 and 0.244 -- which is structural and no learning rate will fix it).
#
# Nothing here touches published artefacts: each point gets its own root under
# lr_sweep/<tag>/, and the committed weights are md5-checked at the end.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
M=$B/results/trueexperiment/Deep-MKV-TS
W=$M/lr_sweep
STEPS=3000
SEEDS="1 3"

export PYTHONPATH="$R/src:$R/experiments:$C"
cd "$B"

# Refuse to start while the main run is still using the GPUs. The whole point of
# the operator's ordering is that the 5-seed run has priority.
if pgrep -f "train_true.py --seed" > /dev/null 2>&1; then
    echo "ABORT: the lr=6.4e-06 run is still training. This sweep is queued"
    echo "       BEHIND it -- wait for run_retune_lr_progressive.sh to finish."
    exit 1
fi

BEFORE=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)

# One worker per GPU: 2 points x 2 seeds = 4 trainings, 4 A100s, one wave.
echo "== lr sweep: 2 points x 2 seeds, one wave =="
PIDS=(); LOGS=(); TAGS=(); G=0
for LR in 2.0e-06 2.0e-05; do
    O="$W/lr_$LR"
    mkdir -p "$O/logs" "$O/weights" "$O/losses" "$O/runs"
    for S in $SEEDS; do
        LO=$(( 16 + G * 8 )); HI=$(( LO + 7 ))
        LOG="$O/logs/train_seed_$S.log"
        CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c $LO-$HI "$PY" "$C/train_true.py" \
            --seed "$S" --steps "$STEPS" --device cuda:0 \
            --lr "$LR" --out-root "$O" --run-root "$O/runs" \
            > "$LOG" 2>&1 &
        PIDS+=($!); LOGS+=("$LOG"); TAGS+=("lr=$LR seed $S -> gpu $G cores $LO-$HI")
        G=$(( G + 1 ))
    done
done
printf 'launched %d workers:\n' "${#PIDS[@]}"
printf '   %s\n' "${TAGS[@]}"

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "WORKER FAILED: ${TAGS[$i]}"; tail -30 "${LOGS[$i]}"; FAIL=1; }
done
[ "$FAIL" -eq 0 ] || { echo "LR SWEEP TRAINING FAILED"; exit 1; }

# ------------------------------------------------------------ score ---------
# Same three stages as the main run, but only on the two swept seeds. The
# aggregate columns are meaningless at n=2, so only the per-seed verdicts are
# read -- which is exactly what compare_retune_early.py produces.
for LR in 2.0e-06 2.0e-05; do
    O="$W/lr_$LR"
    for S in $SEEDS; do
        GPU=$(( S % 4 ))
        echo "== lr=$LR seed $S: select, generate, score =="
        CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
        OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c 16-21 "$PY" "$C/select_checkpoint_true.py" \
            --seeds "$S" --device cuda:0 --run-root "$O/runs" --out-root "$O" \
            > "$O/logs/selection_seed_$S.log" 2>&1

        CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
        OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c 16-21 "$PY" "$C/generate_bank_true.py" \
            --seed "$S" --device cuda:0 --num-paths 6144 \
            --data-dir "$B/dataset/TrueDataset/variants/om_2022-07_N6144" \
            --seq-tag 6144x128x8 --weights-root "$O" --out-root "$O" \
            > "$O/logs/gen_seed_$S.log" 2>&1

        mkdir -p "$O/early/seed_$S"
        CUDA_VISIBLE_DEVICES=$GPU \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        taskset -c 24-55 "$PY" metrics/compute_all_multiasset.py \
            --method "Deep-MKV-TS-lr$LR-seed$S" --dataset TrueDataset \
            --seed-list "$S" --gen-root "$O" \
            --data-dir "$B/dataset/TrueDataset/variants/om_2022-07_N6144" \
            --seq-tag 6144x128x8 --dt 9.512937595129376e-07 \
            --results-dir "$O/early/seed_$S" \
            > "$O/logs/metrics_seed_$S.log" 2>&1

        "$PY" "$C/compare_retune_early.py" --seed "$S" \
            --retune-root "$O/early/seed_$S" \
            2>&1 | tee "$O/logs/verdict_seed_$S.log"
    done
done

AFTER=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] \
    || { echo "ABORT: published weights were modified ($BEFORE -> $AFTER)"; exit 1; }

echo ""
echo "== lr curve: temporal wins out of 27, per point =="
"$PY" - <<'PYEOF'
import json, pathlib
W = pathlib.Path("/home/tbasseras/benchmark/results/trueexperiment/Deep-MKV-TS")
points = [("2.0e-06", W / "lr_sweep/lr_2.0e-06"),
          ("6.4e-06", W / "retune_lr"),
          ("2.0e-05", W / "lr_sweep/lr_2.0e-05")]
print(f"{'lr':>10s} {'seed':>5s} {'temporal':>10s} {'marginal':>10s} {'total':>10s}")
for label, root in points:
    for s in (1, 3):
        f = root / "early" / f"seed_{s}" / f"early_verdict_seed_{s}.json"
        if not f.is_file():
            print(f"{label:>10s} {s:5d} {'-':>10s} {'-':>10s} {'-':>10s}")
            continue
        v = json.loads(f.read_text())
        print(f"{label:>10s} {s:5d} "
              f"{v['retune_temporal_wins']:4d}/{v['n_temporal']:<5d} "
              f"{v['retune_marginal_wins']:4d}/{v['n_marginal']:<5d} "
              f"{v['retune_wins']:4d}/{v['n_compared']:<5d}")
PYEOF

echo "LR SWEEP DONE -- roots under $W"
