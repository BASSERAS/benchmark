#!/usr/bin/env bash
# Score a CORRECTLY SCALED control on the full 34-metric suite.
#
# Why
# ---
# The 88-point alpha grid showed that alpha in [0.0029, 0.0044] is admissible on
# seeds 1, 3 and 4 at once, and that at alpha = 0.003233 BOTH aggregate means
# clear the real-vs-real envelope (vol 23.67% <= 24.59%, corr 0.1810 <= 0.1843).
# But `selection_true.score_candidate` only measures vol / corr / kurt / NN.
#
# The A+B table on the shipped alpha=1 banks says the interesting part is
# elsewhere: on marginal statistics the learned control HELPS (A7 terminal MMD
# 7.12x -> 2.20x floor, A8 increment MMD 2.95x -> 0.78x, A18 GRU discriminator
# 4.79x -> 0.45x) while on temporal structure it is destroyed (A21 ACF|r| 5.33x
# -> 10.87x, A31 rolling-vol KS 2.29x -> 6.26x, and B_acf_abs_r_der 7.3x ->
# 3077.9x floor, i.e. 424x worse than doing no learning at all).
#
# That split is exactly what an over-strong per-step control predicts: sigma is
# driven by a network output with no temporal persistence, so the return
# HISTOGRAM can be matched while volatility CLUSTERING is destroyed.
#
# This run answers the question that follows: does turning the control down to
# alpha = 0.003233 restore the temporal structure, or does it merely trade the
# marginal gains back? Only the full suite can say, because vol_err cannot see
# an ACF.
#
# Nothing here touches published artefacts: everything lands under
# diagnostic_bestfit/alpha_fixed/.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
D=$B/results/trueexperiment/Deep-MKV-TS/diagnostic_bestfit
A=$D/alpha_fixed
V=$B/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
DT=9.512937595129376e-07
ALPHA=0.003233

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$A/logs"
cd "$B"

echo "== generate 5 banks at alpha=$ALPHA (one worker per seed, GPUs 0-3) =="
PIDS=(); LOGS=()
for S in 0 1 2 3 4; do
    GPU=$(( S % 4 ))
    LO=$(( 16 + S * 6 )); HI=$(( LO + 5 ))
    LOG="$A/logs/gen_seed_$S.log"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 \
    taskset -c $LO-$HI "$PY" "$C/alpha_ablation.py" --seeds "$S" --device cuda:0 \
        --alphas "$ALPHA" --tag "fixed" --save-bank-root "$A" > "$LOG" 2>&1 &
    PIDS+=($!); LOGS+=("$LOG")
done
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { tail -25 "${LOGS[$i]}"; echo "gen seed $i FAILED"; FAIL=1; }
done
[ "$FAIL" -eq 0 ] || exit 1

for S in 0 1 2 3 4; do
    [ -f "$A/generated_paths/seed_$S/generated_paths_$TAG.npy" ] \
        || { echo "bank seed $S missing"; exit 1; }
done
echo "   all 5 banks present"

# --gen-root is the directory HOLDING generated_paths/, so it is $A.
echo "== full A1-A32 + curve B on the alpha-corrected banks =="
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 16-55 "$PY" metrics/compute_all_multiasset.py \
    --method "Deep-MKV-TS-alpha${ALPHA}-diagnostic" --dataset TrueDataset --seeds 5 \
    --gen-root "$A" --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$A" \
    > "$A/logs/metrics.log" 2>&1

[ -f "$A/metrics_summary.csv" ] \
    || { tail -30 "$A/logs/metrics.log"; echo "metrics_summary.csv not produced"; exit 1; }
echo "ALPHA-FIXED METRICS DONE"
