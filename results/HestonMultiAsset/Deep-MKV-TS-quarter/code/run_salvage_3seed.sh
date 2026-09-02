#!/usr/bin/env bash
# Quarter campaign: 3-seed salvage chain.
#
# WHY THIS FILE EXISTS
# Seeds 2 and 5 diverged with a non-finite Theta. collect_artifacts.py enforces
# a 5-seed contract, so both earlier chains (run_pipeline_variant2.sh at
# 03:31:43 and run_final_chain.sh at 04:19:15) died there with rc=1 and never
# reached the metrics stage. Seeds 0, 4 and 6 have complete weights and
# generated paths on disk; this chain scores those three.
#
# WHAT IS DIFFERENT FROM THE OTHER CHAINS
#   1. COLLECT_SEEDS=0,4,6 relaxes the contract gate (env override added to
#      collect_artifacts.py; the default 5-seed list is untouched).
#   2. Training, checkpoint selection and path generation are SKIPPED -- those
#      artefacts already exist. This chain starts at collect.
#   3. The plot/README stages are NON-FATAL. plot_losses.py hardcodes
#      SEEDS = [0, 2, 4, 5, 6] at line 64 and will fail on a 3-seed set; that
#      must not destroy the 71-minute metrics result that precedes it.
#
# HEALTH WARNING carried into every artefact this produces:
# N=3, and the two dropped seeds were dropped because they DIVERGED. A 40%
# divergence rate is itself a result about this config. Row counts from this
# run are NOT comparable to the 5-seed campaigns -- fewer seeds means a larger
# std, which widens the tie band.
#
# No git, no commit, no push.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
METHOD="$(basename "$METHOD_DIR")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python

SEEDS_COMMA="${SEEDS_COMMA:-0,4,6}"
DIVERGED="${DIVERGED:-2,5}"
POST_GPU="${POST_GPU:-1}"
POST_CORES="${POST_CORES:-0-7}"

export PYTHONPATH="$REF/src:$REF/experiments"
export COLLECT_SEEDS="$SEEDS_COMMA"

echo "=== [$METHOD SALVAGE] START $(date -Is)"
echo "=== reporting seeds: $SEEDS_COMMA   (diverged, dropped: $DIVERGED)"
echo "=== gpu: $POST_GPU   cores: $POST_CORES"
echo "=== N=3 -- results NOT row-comparable to the 5-seed campaigns"

# ---------------------------------------------------------- preflight --------
for s in ${SEEDS_COMMA//,/ }; do
  npy="$METHOD_DIR/generated_paths/seed_$s/generated_paths_8192x252x8.npy"
  [ -f "$npy" ] || { echo "=== ABORT: missing $npy" >&2; exit 1; }
done
echo "=== [preflight] all 3 generated-path files present"

run() {                     # fatal: chain stops on failure
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  CUDA_VISIBLE_DEVICES="$POST_GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "$POST_CORES" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc $(date -Is) -- CHAIN STOPPED" >&2
    exit "$rc"
  fi
  echo "=== [$label] OK $(date -Is)"
}

soft() {                    # non-fatal: cosmetic stages may fail on N=3
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  CUDA_VISIBLE_DEVICES="$POST_GPU" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "$POST_CORES" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc -- NON-FATAL, continuing $(date -Is)" >&2
  else
    echo "=== [$label] OK $(date -Is)"
  fi
}

run  "collect artifacts" "$PY" "$HERE/collect_artifacts.py"
run  "compute A1-A34"    "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
     --method "$METHOD" --results-dir "$METHOD_DIR" --seed-list "$SEEDS_COMMA"
soft "memorisation"      "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"
soft "plot diagnostics"  "$PY" "$HERE/plot_diagnostics_multiasset.py"
soft "plot losses"       "$PY" "$HERE/plot_losses.py"
soft "render README"     "$PY" "$HERE/render_readme.py"

echo "=== [$METHOD SALVAGE] PIPELINE COMPLETE $(date -Is)"
echo "=== reported seeds: $SEEDS_COMMA   dropped (diverged): $DIVERGED"
