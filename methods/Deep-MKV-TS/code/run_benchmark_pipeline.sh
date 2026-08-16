#!/usr/bin/env bash
# Deep-MKV-TS: everything downstream of training, in one command.
#
# Training itself lives in ../paper_reimplementation/metric/{run_reproduction.sh,
# run_seed2.sh}. This script assumes all five seed run trees exist (each with a
# step_2500 checkpoint evaluation) and takes it from there:
#
#   1. export   runs/seed_{0..4}/ -> generated_paths/, weights/, losses/   (GUIDELINE 4.3-4.4)
#   2. metrics  A1-A34 + B curves + grid_tvd                               (GUIDELINE 5)
#   3. plots    8-panel stylised-facts diagnostic                          (GUIDELINE 7.1)
#   4. PS-MC    path-shadowing Monte-Carlo evaluation                      (GUIDELINE 6)
#
# It blocks until every seed has its step-2500 evaluation on disk, so it can be
# launched the moment the last seed starts training.
#
# Usage:  bash code/run_benchmark_pipeline.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(dirname "$HERE")"
BENCH="$(cd "$METHOD_ROOT/../.." && pwd)"
PY="${BENCHMARK_PYTHON:-/home/tbasseras/gpu-venv/bin/python}"
RUNS="$METHOD_ROOT/paper_reimplementation/runs"
SEEDS=(0 1 2 3 4)
CORES="${BENCHMARK_CORES:-16-23}"          # 8 physical cores, GUIDELINE 4.1
GPU="${BENCHMARK_GPU:-0}"                  # single GPU index

log() { echo "[$(date -Is)] $*"; }

# --- 0. wait for every seed to have its reported checkpoint ----------------
for s in "${SEEDS[@]}"; do
  if [[ ! -d "$RUNS/seed_$s" ]]; then
    echo "FATAL: $RUNS/seed_$s does not exist. Train it first:" >&2
    echo "  bash $METHOD_ROOT/paper_reimplementation/metric/run_reproduction.sh" >&2
    echo "  bash $METHOD_ROOT/paper_reimplementation/metric/run_seed2.sh" >&2
    exit 1
  fi
done

waited=0
while :; do
  missing=()
  for s in "${SEEDS[@]}"; do
    [[ -f "$RUNS/seed_$s/volatility_only_online_mp/checkpoint_evaluations/step_2500/metrics.json" ]] \
      || missing+=("$s")
  done
  (( ${#missing[@]} == 0 )) && break
  if (( waited % 600 == 0 )); then
    log "waiting for seeds: ${missing[*]}"
  fi
  # 90 min ceiling: one seed is ~35 min, so anything past this is a hung run,
  # not a slow one, and silently waiting forever would hide that.
  if (( waited > 5400 )); then
    echo "FATAL: still missing seeds ${missing[*]} after 90 min; check console.log" >&2
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
done
log "all 5 seeds have a step_2500 evaluation"

# --- 1. export to the GUIDELINE layout -------------------------------------
log "1/4 export"
"$PY" "$HERE/export_benchmark_artifacts.py" --seeds "${SEEDS[@]}"

# --- 2. A1-A34 + B metrics -------------------------------------------------
# compute_all.py refits the A18 judge and the A19 forecaster per seed, so it
# needs a GPU. One job, 8 cores, per GUIDELINE 4.1.
log "2/4 metrics (A1-A34 + B + grid_tvd)"
cd "$BENCH/metrics"
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
  "$PY" compute_all.py --method Deep-MKV-TS --dataset Heston --seeds 5
# compute_all.py writes the per-seed JSONs and metrics_summary.csv; the two
# aggregate files that render_tables.py reads are produced separately.
OMP_NUM_THREADS=8 taskset -c "$CORES" "$PY" recompute_curve_b.py  --method Deep-MKV-TS --seeds 5
OMP_NUM_THREADS=8 taskset -c "$CORES" "$PY" recompute_grid_tvd.py --method Deep-MKV-TS --seeds 5

# --- 3. stylised-facts diagnostic figure -----------------------------------
log "3/4 diagnostics figure"
OMP_NUM_THREADS=8 taskset -c "$CORES" \
  "$PY" plot_diagnostics.py --method Deep-MKV-TS --dataset Heston --seed 0
# A18 / A19 classifier-training loss figures (GUIDELINE 7.2-7.3). Reads the
# seed_*_{disc,pred}_{gru,mlp}_loss.csv that compute_all.py just wrote.
OMP_NUM_THREADS=8 taskset -c "$CORES" \
  "$PY" plot_score_losses.py --method Deep-MKV-TS --dataset Heston

# --- 4. path-shadowing Monte-Carlo -----------------------------------------
log "4/4 PS-MC"
cd "$METHOD_ROOT/path_shadowing"
OMP_NUM_THREADS=8 taskset -c "$CORES" "$PY" run_eval.py

log "DONE. Outputs:"
log "  $METHOD_ROOT/{generated_paths,weights,losses}/"
log "  $BENCH/results/Heston/Deep-MKV-TS/"
log "  $BENCH/results/Heston/Deep-MKV-TS/path_shadowing/"
