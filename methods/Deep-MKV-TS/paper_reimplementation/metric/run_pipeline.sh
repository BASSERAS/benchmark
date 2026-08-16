#!/usr/bin/env bash
# Deep-MKV-TS end-to-end driver: reproduction -> paper table -> val/valdisc search -> ranking.
#
# The 4-seed reproduction is usually already in flight when this starts, so
# stage 1 waits for it rather than relaunching (run_reproduction.sh itself skips
# any seed that already has COMPLETE.json, so calling it again is safe either
# way). Stage 3 never touches heston_S_disc: run_hpsearch.sh redirects the
# scored split to heston_S_valdisc via BENCHMARK_HESTON_EVAL. Only the single
# winner is re-run on `disc`, exactly once, per GUIDELINE 11.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_ROOT="$(dirname "$HERE")"
PY=/home/tbasseras/gpu-venv/bin/python
RUNS="$PR_ROOT/runs"
SEEDS=(0 1 3 4)
# Generous ceiling: 4 seeds at roughly 20-25 min each, two at a time.
MAX_WAIT_S="${MAX_WAIT_S:-7200}"

log() { echo "[pipeline $(date -u +%H:%M:%SZ)] $*"; }

all_seeds_done() {
  local s
  for s in "${SEEDS[@]}"; do
    [[ -f "$RUNS/seed_$s/COMPLETE.json" ]] || return 1
  done
  return 0
}

reproduction_alive() {
  pgrep -f "run_matched_control_synthetic_validation.py.*paper_reimplementation/runs/seed_" > /dev/null
}

# ---- Stage 1: reproduction -------------------------------------------------
log "stage 1: 4-seed reproduction (Guyon-Lekeufack reference)"
waited=0
while ! all_seeds_done; do
  if ! reproduction_alive; then
    log "no reproduction process alive; invoking run_reproduction.sh for the remaining seeds"
    bash "$HERE/run_reproduction.sh" >> "$RUNS/reproduction_driver.log" 2>&1 || {
      log "FATAL: run_reproduction.sh failed; see reproduction_driver.log"; exit 1; }
    continue
  fi
  sleep 30
  waited=$((waited + 30))
  if (( waited > MAX_WAIT_S )); then
    log "FATAL: reproduction exceeded ${MAX_WAIT_S}s"; exit 1
  fi
done
log "stage 1 done: all seeds complete"

# ---- Stage 2: paper-vs-ours table -----------------------------------------
log "stage 2: aggregating paper-vs-ours table"
"$PY" "$HERE/aggregate_paper_table.py" > "$RUNS/aggregate.log" 2>&1 \
  || { log "FATAL: aggregation failed; see aggregate.log"; exit 1; }
log "stage 2 done: results/PAPER_VS_OURS.md written"

# ---- Stage 3: hyperparameter search on val/valdisc -------------------------
log "stage 3: hyperparameter search (scored on heston_S_valdisc)"
bash "$HERE/run_hpsearch.sh" > "$RUNS/hpsearch_driver.log" 2>&1 \
  || { log "FATAL: search failed; see hpsearch_driver.log"; exit 1; }
log "stage 3 done: runs/hpsearch/RANKING.md written"

log "PIPELINE COMPLETE"
