#!/usr/bin/env bash
#
# Deep-MKV-TS on TrueDataset: freeze -> ridge -> train -> select, queued.
#
# Why this script exists
# ----------------------
# Every stage below depends on the artefact the previous stage writes, so they
# cannot be launched together; but WITHIN a stage the points are independent and
# are fanned out across GPUs 1, 2 and 3. The script exists so the whole chain
# runs unattended instead of waiting for a human between stages.
#
# GPU 0 is never used: it is another user's. The 3-GPU / 24-core footprint is an
# explicit, time-boxed override of the standing 2-GPU limit, granted for this
# experiment only.
#
# Ordering constraint that forces the shape of this file
# ------------------------------------------------------
# `train_true.build_model` calls `load_kernel`, which reads
# `code/reference/reference_kernel.json`. That file does not exist until stage 1
# writes it. So nothing downstream -- not the ridge sweep, not the timing probe,
# not training -- can start before the reference sweep has chosen its
# hyperparameters. That is a real dependency, not a scheduling preference.
#
# Usage (detached; this runs for hours)
# -------------------------------------
#     cd /home/tbasseras/benchmark/results/trueexperiment/Deep-MKV-TS/code
#     setsid nohup ./run_pipeline_true.sh \
#         > ../losses/pipeline.log 2>&1 < /dev/null & disown
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(dirname "$HERE")"
LOSSES="$METHOD_ROOT/losses"
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python

export PYTHONPATH="$REF/src:$REF/experiments"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

GPUS=(1 2 3)
CORES=("0-7" "8-15" "16-23")

LR=0.01
REF_STEPS=300
TRAIN_STEPS=3000
SEEDS=(0 1 2 3 4)
# Started minimal -- centred on the Heston winner (1000), one decade either side
# -- because the instruction was to detect a large move, not to re-tune from
# scratch. It had to be WIDENED, and the reason is section 7.2 rather than
# impatience: on the 3-point grid (100, 1000, 10000) the winner was 100, the
# LOWER ENDPOINT. A screen that selects at the edge of its own grid has found a
# boundary, not an optimum, and section 7.2 requires the grid be extended until
# the choice is bracketed. Adding 1 and 10 did that. |log NNratio| by lambda:
#     1 -> 0.3447   10 -> 0.7540   100 -> 0.2450   1000 -> 2.6858   1e4 -> 2.5248
# so 100 is now an INTERIOR minimum with both neighbours worse. Do not shrink
# this back to three points.
LAMBDAS=(1 10 100 1000 10000)

cd "$HERE"
mkdir -p "$LOSSES"

banner() { printf '\n%s\n== %s\n%s\n' "$(printf '=%.0s' {1..78})" "$1" "$(printf '=%.0s' {1..78})"; }
stamp()  { date '+%Y-%m-%d %H:%M:%S'; }

# --- run a list of jobs, one per GPU, and fail loudly if any of them fails ----
# `wait $pid` is checked per PID rather than using a bare `wait`, because a bare
# `wait` returns 0 even when a child died, which would let the pipeline march on
# with a missing artefact and only discover it three stages later.
fanout() {
  local -n _cmds=$1
  local -a pids=() tags=()
  local i=0
  for cmd in "${_cmds[@]}"; do
    local g="${GPUS[$((i % ${#GPUS[@]}))]}"
    local c="${CORES[$((i % ${#CORES[@]}))]}"
    echo "[$(stamp)] launch gpu=$g cores=$c :: $cmd"
    CUDA_VISIBLE_DEVICES="$g" taskset -c "$c" bash -c "$cmd" &
    pids+=($!)
    tags+=("gpu$g :: $cmd")
    i=$((i + 1))
  done
  local rc=0 k=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      echo "[$(stamp)] FAILED: ${tags[$k]}" >&2
      rc=1
    fi
    k=$((k + 1))
  done
  return $rc
}

# =============================================================================
banner "stage 0  preconditions"
# =============================================================================
SELECTION="$LOSSES/reference_selection.json"
if [[ ! -f "$SELECTION" ]]; then
  echo "ABORT: $SELECTION missing. The reference sweep has not chosen a kernel yet." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Read the selected point instead of retyping it. A hand-copied hyperparameter
# is how a run ends up reporting a configuration it did not use.
read -r SIGMA_MAX RIDGE_COV RIDGE_DRIFT < <(
  "$PY" - "$SELECTION" <<'PYEOF'
import json, sys
sel = json.loads(open(sys.argv[1]).read())["selected"]
print(float(sel["sigma_max"]), float(sel["ridge_covariance"]), float(sel["ridge_drift"]))
PYEOF
)
echo "[$(stamp)] selected reference kernel: sigma_max=$SIGMA_MAX ridge_cov=$RIDGE_COV ridge_drift=$RIDGE_DRIFT"

# =============================================================================
banner "stage 1  freeze the reference kernel   (~4 min, single GPU)"
# =============================================================================
KERNEL="$HERE/reference/reference_kernel.json"
if [[ -f "$KERNEL" ]]; then
  echo "[$(stamp)] $KERNEL already exists -- not refitting."
else
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" taskset -c "${CORES[0]}" \
    "$PY" fit_reference_true.py \
      --sigma-max "$SIGMA_MAX" \
      --ridge-covariance "$RIDGE_COV" \
      --ridge-drift "$RIDGE_DRIFT" \
      --lr "$LR" --steps "$REF_STEPS" --device cuda:0
  [[ -f "$KERNEL" ]] || { echo "ABORT: fit did not write $KERNEL" >&2; exit 1; }
fi
echo "[$(stamp)] kernel sha256: $(sha256sum "$KERNEL" | cut -c1-16)"

# =============================================================================
banner "stage 2  timing probe   (~3 min, single GPU)"
# =============================================================================
# 20 steps is enough to measure seconds/step once the first-step compile and
# allocator warmup are amortised, and it writes nothing. This replaces the
# projection ("Heston was 7.9 s/step at T=252, halve it for T=127") with a
# measurement on this dataset, this kernel, this box.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" taskset -c "${CORES[0]}" \
  "$PY" train_true.py --seed 0 --time-only 20 --device cuda:0 \
  2>&1 | tee "$LOSSES/timing_probe.log"

# =============================================================================
banner "stage 3  ridge-lambda sweep   (5 points, two waves, ~40 min)"
# =============================================================================
RIDGE_CMDS=()
for lam in "${LAMBDAS[@]}"; do
  RIDGE_CMDS+=("$PY sweep_ridge_lambda_true.py --ridge-lambda $lam --steps 250 --device cuda:0")
done
fanout RIDGE_CMDS
"$PY" sweep_ridge_lambda_true.py --report 2>&1 | tee "$LOSSES/ridge_sweep.log"

# The winner is read back from the artefacts the sweep wrote, for the same
# reason as stage 0: no hand-copied numbers.
#
# THIS STAGE RANKS. IT DOES NOT GATE -- and that is a deliberate reading of
# section 7, not a relaxation of it. Section 7 constrains the checkpoint that is
# REPORTED. What runs here is a 250-step screen: 8 % of the training budget, run
# only to order five values of one hyperparameter against each other. Nothing at
# 250 steps can clear a vol/corr ceiling that was measured real-against-real, so
# applying `sel.select` here does not enforce the criterion -- it aborts the
# pipeline before training and enforces nothing at all.
#
# The ceilings are still enforced, in the only place they are meaningful:
# `select_checkpoint_true.py` at stage 5, on fully-trained checkpoints, on the
# validation split. A lambda that wins here and then produces no admissible
# checkpoint there fails, loudly, at stage 5.
#
# Violations at 250 steps are therefore PRINTED, not acted on. Discarding them
# silently would be the actual dishonesty; a lambda that is already far outside
# the envelope this early is worth seeing even though it is not disqualifying.
BEST_LAMBDA="$(
  "$PY" - "$HERE" <<'PYEOF'
import json, math, sys, pathlib
here = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(here))
import selection_true as sel
from fit_reference_true import DATASET, SEQ_TAG
env = sel.envelope(data_dir=DATASET, seq_tag=SEQ_TAG)
recs = [json.loads(p.read_text()) for p in sorted((here / "sweep").glob("lambda_*.json"))]
recs = [r for r in recs if not r.get("diverged")]
if not recs:
    raise SystemExit("ABORT: every ridge_lambda diverged")

scored = [(sel.objective(r), r) for r in recs]
scored = [(o, r) for o, r in scored if math.isfinite(o)]
if not scored:
    raise SystemExit("ABORT: every surviving ridge_lambda scored non-finite |log NNratio|")
scored.sort(key=lambda t: t[0])

for o, r in scored:
    bad = sel.admissibility(r, env)
    note = "envelope ok at 250 steps" if not bad else "outside envelope: " + "; ".join(bad)
    print(f"  lambda={r['ridge_lambda']:<8g} |logNN|={o:8.4f}   {note}", file=sys.stderr)

best = scored[0][1]
grid = sorted(float(r["ridge_lambda"]) for _, r in scored)
lam = float(best["ridge_lambda"])
# Section 7.2: a winner at either end of the grid is a boundary, not an optimum.
# Say so here rather than letting the README quote it as a tuned value.
if len(grid) > 1 and lam in (grid[0], grid[-1]):
    print(f"  ! WARNING: ridge_lambda={lam:g} is an ENDPOINT of the grid {grid}. "
          f"That is a boundary, not a demonstrated optimum (section 7.2). "
          f"Widen LAMBDAS and re-run before reporting it as selected.",
          file=sys.stderr)
print(f"{lam:g}")
PYEOF
)"
echo "[$(stamp)] selected ridge_lambda = $BEST_LAMBDA"

# =============================================================================
banner "stage 4  production training   (5 seeds over a shared queue, ~2 h/seed)"
# =============================================================================
# A failed seed does not stop the others: at d = 8 the Heston campaign lost two
# seeds to a non-finite control, and killing the pipeline on the first casualty
# would have thrown away the survivors' hours as well.
#
# THIS DELEGATES TO run_seeds_true.sh RATHER THAN FANNING OUT FIXED WAVES, and
# the reason is arithmetic: `fanout` runs one wave per GPU-count, so 5 seeds on
# 3 GPUs is a wave of 3 and then a wave of 2, and the second wave leaves a GPU
# idle for a FULL 2-HOUR SLOT. A shared work queue keeps every worker busy until
# the seeds run out. It is also idempotent -- a seed with runs/seed_N/COMPLETE.json
# is skipped -- so this stage can be re-entered after an interruption without
# retraining what already finished, which fixed waves cannot do.
#
# The queue can additionally be WIDENED IN FLIGHT (`WORKERS="3:16-23"
# ./run_seeds_true.sh` against the live queue) if a GPU frees up mid-run. That
# is what happened on this campaign.
WORKERS=""
for i in "${!GPUS[@]}"; do
  WORKERS+="${GPUS[$i]}:${CORES[$i]} "
done
echo "[$(stamp)] handing ${#SEEDS[@]} seeds to run_seeds_true.sh over workers: $WORKERS"

WORKERS="$WORKERS" SEEDS="${SEEDS[*]}" STEPS="$TRAIN_STEPS" \
  RIDGE_LAMBDA="$BEST_LAMBDA" bash "$HERE/run_seeds_true.sh" \
  || echo "[$(stamp)] WARNING: at least one seed failed; continuing so the survivors finish" >&2

# =============================================================================
banner "stage 5  checkpoint selection   (section-7 rule, ~1 min/seed)"
# =============================================================================
DONE_SEEDS=()
for s in "${SEEDS[@]}"; do
  [[ -f "$HERE/runs/seed_$s/COMPLETE.json" ]] && DONE_SEEDS+=("$s")
done
echo "[$(stamp)] seeds that finished training: ${DONE_SEEDS[*]:-NONE}"
if [[ ${#DONE_SEEDS[@]} -eq 0 ]]; then
  echo "ABORT: no seed completed training." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" taskset -c "${CORES[0]}" \
  "$PY" select_checkpoint_true.py --seeds "${DONE_SEEDS[@]}" --device cuda:0 \
  2>&1 | tee "$LOSSES/checkpoint_selection.log"

banner "pipeline finished at $(stamp)"
echo "Everything downstream of here lives in run_pipeline_post.sh, which is a"
echo "SEPARATE script on purpose: it waits on runs/.seed_queue rather than on this"
echo "process, so it can be launched at any point -- including before training"
echo "finishes -- and will fire the moment the queue drains. Run:"
echo
echo "    setsid nohup bash $HERE/run_pipeline_post.sh \\"
echo "        > $METHOD_ROOT/logs/pipeline_post.log 2>&1 < /dev/null & disown"
echo
echo "It runs: selection -> A/B banks -> CRPS pools -> contract gate -> metrics ->"
echo "memorisation -> 12 conditional-CRPS runs -> figures -> envelope screen -> README."
