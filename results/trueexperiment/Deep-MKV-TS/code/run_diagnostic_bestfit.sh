#!/usr/bin/env bash
# DIAGNOSTIC ONLY -- answers "does the learned control beat the untrained
# reference SDE?" after select_checkpoint_true.py aborted on all five seeds.
#
# This is NOT a stage of run_pipeline_post.sh and must never be called from it.
# Every artefact it writes lands under diagnostic_bestfit/, never under the
# published generated_paths/ or metrics_summary.csv, so a later real pipeline
# run neither reads nor skips because of anything here.
#
# GPUs 2 and 3 (GPU 0 is off-limits on this machine, GPU 1 left free), cores
# 0-15 -- the standing 2-GPU / 16-core limit.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
CC=/home/tbasseras/.cc-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
M=$B/results/trueexperiment/Deep-MKV-TS
D=$M/diagnostic_bestfit
V=$B/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
DT=9.512937595129376e-07

export PYTHONPATH="$R/src:$R/experiments:$C"

mkdir -p "$D/logs"
cd "$B"

echo "== stage best-fit checkpoints =="
"$CC" "$C/stage_bestfit_checkpoints.py" 2>&1 | tee "$D/logs/stage.log"

echo "== generate 5 A/B banks (GPUs 2,3) =="
for S in 0 1 2 3 4; do
    OUT=$D/generated_paths/seed_$S/generated_paths_$TAG.npy
    if [ -f "$OUT" ]; then echo "   seed $S bank exists -- skipping"; continue; fi
    GPU=$(( 2 + (S % 2) ))
    LO=$(( (S % 2) * 8 )); HI=$(( LO + 7 ))
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    taskset -c $LO-$HI "$PY" "$C/generate_bank_true.py" --seed "$S" --device cuda:0 \
        --data-dir "$V" --seq-tag "$TAG" --out-root "$D" \
        > "$D/logs/bank_seed_$S.log" 2>&1 &
    if [ $(( S % 2 )) -eq 1 ]; then wait; fi
done
wait

for S in 0 1 2 3 4; do
    [ -f "$D/generated_paths/seed_$S/generated_paths_$TAG.npy" ] \
        || { tail -20 "$D/logs/bank_seed_$S.log"; echo "bank seed $S missing"; exit 1; }
done
echo "   all 5 banks present"

# --gen-root is "directory holding generated_paths/", so it is $D, not
# $D/generated_paths.
echo "== metrics A1-A32 + curve B (GPU 3, cores 0-15) =="
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
taskset -c 0-15 "$PY" metrics/compute_all_multiasset.py \
    --method Deep-MKV-TS-bestfit-diagnostic --dataset TrueDataset --seeds 5 \
    --gen-root "$D" \
    --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$D" \
    > "$D/logs/metrics.log" 2>&1

[ -f "$D/metrics_summary.csv" ] || { tail -30 "$D/logs/metrics.log"; echo "metrics_summary.csv not produced"; exit 1; }
echo "DIAGNOSTIC DONE"
