#!/usr/bin/env bash
# Reference metrics A1-A32 + curve B on TrueDataset.
#
# Same invocation as CSDI/code/run_pipeline.sh STAGE 5 and
# Deep-MKV-TS/code/run_pipeline_post.sh stage [5]; only --method and
# --results-dir differ.  Kept as a script rather than a shell one-liner so the
# exact flags that produced results/trueexperiment/reference/metrics_summary.csv
# are recorded next to the artefacts they produced.
#
# GPU 3 and cores 96-111: GPUs 1 and 2 plus cores 0-63 belong to the
# Deep-MKV-TS run_pipeline_post.sh job while it is alive, and GPU 0 is
# off-limits on this machine.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
V="$B/dataset/TrueDataset/variants/om_2022-07_N6144"
M="$B/results/trueexperiment/reference"
TAG=6144x128x8
DT=9.512937595129376e-07

mkdir -p "$M/logs"
cd "$B"

CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
taskset -c 96-111 "$PY" metrics/compute_all_multiasset.py \
    --method reference --dataset TrueDataset --seeds 5 \
    --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$M" \
    > "$M/logs/metrics.log" 2>&1

[ -f "$M/metrics_summary.csv" ]    || { echo "metrics_summary.csv not produced"; exit 1; }
[ -f "$M/curve_b_aggregate.json" ] || { echo "curve_b_aggregate.json not produced"; exit 1; }
echo "metrics done"
