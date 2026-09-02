#!/usr/bin/env bash
# Quarter campaign: single 5-seed chain, gate-free.
#
# WHY THIS FILE EXISTS
# The quarter campaign currently has two supervisors layered on top of each
# other for historical reasons:
#
#   325986  run_pipeline_variant.sh   trains seeds 0,2; then runs a 2-seed chain
#   332611  run_pipeline_variant2.sh  waits for that 2-seed chain to say
#                                     "PIPELINE COMPLETE", then runs a 5-seed
#                                     chain that OVERWRITES every artefact the
#                                     2-seed chain just produced
#
# The 2-seed chain is ~1.4 h of GPU time whose entire output is thrown away.
# It exists only because 325986 was already running with a live fd on its own
# script and could not be edited. Removing it saves ~1.4 h of wall clock.
#
# This script replaces BOTH chains with one 5-seed chain that fires the moment
# the last trainer writes its step-2500 checkpoint.
#
# SAFETY INTERLOCK
# Two chains writing weights/ and generated_paths/ concurrently produce a
# mixed-provenance artefact set. So this script REFUSES to run its chain if
# either legacy supervisor is still alive -- in that case it exits quietly and
# lets the old (slower, correct) path proceed untouched. That makes it safe to
# launch right now, before the operator has killed anything.
#
# The trainers themselves are NOT touched. Killing 325986 orphans its two
# python children onto init; they keep training and keep writing checkpoints.
#
# No git, no commit, no push.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
METHOD="$(basename "$METHOD_DIR")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python

STEPS="${STEPS:-2500}"
RUN_ROOT="${RUN_ROOT:-$HERE/runs}"
ALL_SEEDS_STR="${ALL_SEEDS_STR:-0 2 4 5 6}"
POST_GPU="${POST_GPU:-0}"
POST_CORES="${POST_CORES:-46-53}"
LEGACY_PIDS="${LEGACY_PIDS:-325986 332611}"

read -r -a ALL_SEEDS <<< "$ALL_SEEDS_STR"
export PYTHONPATH="$REF/src:$REF/experiments"

echo "=== [$METHOD FINAL] START $(date -Is)"
echo "=== seeds: $ALL_SEEDS_STR   post-gpu: $POST_GPU   cores: $POST_CORES"
echo "=== legacy supervisors watched: $LEGACY_PIDS"

# ------------------------------------------- 1. wait for all 5 trainers ------
# Poll for the final checkpoint of every seed. _save_atomic() in
# train_multiasset.py means a visible step_2500.pt is always complete.
echo "=== [wait] polling for step_${STEPS} checkpoints on all seeds $(date -Is)"
while :; do
  missing=""
  for s in "${ALL_SEEDS[@]}"; do
    compgen -G "$RUN_ROOT/seed_$s/training_checkpoints/step_*${STEPS}.pt" >/dev/null \
      || missing="$missing $s"
  done
  [ -z "$missing" ] && break

  # If no trainer is alive any more, the missing seeds died. Stop waiting and
  # let triage below drop them by name rather than block forever.
  if ! pgrep -f "train_multiasse[t].py .*Deep-MKV-TS-quarter" >/dev/null 2>&1; then
    echo "=== [wait] no quarter trainer alive; missing seeds [$missing ] will be dropped $(date -Is)"
    break
  fi
  sleep 120
done
echo "=== [wait] training phase over $(date -Is)"

# ------------------------------------------------ 2. safety interlock --------
for p in $LEGACY_PIDS; do
  if kill -0 "$p" 2>/dev/null; then
    echo "=== [interlock] legacy supervisor PID $p is STILL ALIVE." >&2
    echo "=== [interlock] it owns the chain. Refusing to run a second one." >&2
    echo "=== [interlock] exiting cleanly; the legacy path will finish the job." >&2
    exit 0
  fi
done
echo "=== [interlock] no legacy supervisor alive -- this chain owns the artefacts"

# --------------------------------------------------- 3. survivor triage ------
SURVIVORS=""; DIVERGED=""
for s in "${ALL_SEEDS[@]}"; do
  if compgen -G "$RUN_ROOT/seed_$s/training_checkpoints/step_*${STEPS}.pt" >/dev/null; then
    SURVIVORS="$SURVIVORS $s"
  else
    DIVERGED="$DIVERGED $s"
  fi
done
SURVIVORS="$(echo $SURVIVORS)"; DIVERGED="$(echo $DIVERGED)"
echo "=== [triage] survivors: [$SURVIVORS]   diverged/incomplete: [$DIVERGED]"
[ -z "$SURVIVORS" ] && { echo "=== ABORT: no seed reached step $STEPS $(date -Is)" >&2; exit 1; }
SEEDS_COMMA="$(echo "$SURVIVORS" | tr ' ' ',')"

run() {
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  CUDA_VISIBLE_DEVICES="$POST_GPU" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  OPENBLAS_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "$POST_CORES" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc $(date -Is) -- CHAIN STOPPED" >&2
    exit "$rc"
  fi
  echo "=== [$label] OK $(date -Is)"
}

run "select checkpoints" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0 --run-root "$RUN_ROOT"

printf '%s seeds=%s (%s final 5-seed chain; diverged/incomplete:%s)\n' \
    "$(date -Is)" "$SEEDS_COMMA" "$METHOD" "${DIVERGED:- none}" \
    > "$METHOD_DIR/weights/.campaign_complete"
echo "=== [sentinel] wrote weights/.campaign_complete"

run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SURVIVORS --device cuda:0
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method "$METHOD" --results-dir "$METHOD_DIR" --seed-list "$SEEDS_COMMA"
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"
run "render method README" "$PY" "$HERE/render_readme.py"

echo "=== [$METHOD FINAL] PIPELINE COMPLETE $(date -Is)"
echo "=== reported seeds: $SEEDS_COMMA"
echo "=== dropped seeds:  ${DIVERGED:- none}"
