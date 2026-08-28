#!/usr/bin/env bash
# Score ONE (half, method, seed) triple. Writes only under /tmp/splithalf.
# Args: <half A|B> <method> <seed> <gpu> <cores>
set -u
H=$1 M=$2 S=$3 GPU=$4 CORES=$5

REPO=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
OUT=/tmp/splithalf
TAG=3072x128x8
DT=9.512937595129376e-07

DEST="$OUT/out$H/$M/s$S"
LOG="$OUT/logs/${H}_${M}_s${S}.log"

# Idempotent: a finished seed is never recomputed, so the queue can be re-run.
if [ -f "$DEST/seed_${S}_metrics.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP  $H/$M/seed$S (already done)"
    exit 0
fi

mkdir -p "$DEST"
cd "$REPO" || exit 1
t0=$SECONDS
echo "[$(date +%H:%M:%S)] START $H/$M/seed$S  gpu$GPU cores$CORES"

CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
taskset -c "$CORES" \
"$PY" metrics/compute_all_multiasset.py \
    --method "$M" --dataset TrueDataset --seed-list "$S" \
    --data-dir "$OUT/data$H" --seq-tag "$TAG" --dt "$DT" \
    --gen-root "$OUT/gen/$M" \
    --results-dir "$DEST" \
    > "$LOG" 2>&1
rc=$?

if [ $rc -eq 0 ] && [ -f "$DEST/seed_${S}_metrics.json" ]; then
    echo "[$(date +%H:%M:%S)] OK    $H/$M/seed$S  $((SECONDS - t0))s"
else
    echo "[$(date +%H:%M:%S)] FAILED $H/$M/seed$S rc=$rc  see $LOG"
fi
exit $rc
