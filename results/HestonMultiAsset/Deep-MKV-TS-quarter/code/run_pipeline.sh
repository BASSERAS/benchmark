#!/usr/bin/env bash
# End-to-end d = 8 pipeline, run once seed 6 finishes training.
#
# Chain, in order, each step gated on the previous one succeeding:
#
#   1. wait for the seed 6 trainer to exit
#   2. select_checkpoint_multiasset.py --seeds 6   (validation checkpoint choice)
#   3. weights/.campaign_complete sentinel
#   4. run_all_multiasset.py                       (generate 5 x 8192 test paths)
#   5. collect_artifacts.py                        (GUIDELINE section 4 contract gate)
#   6. compute_all_multiasset.py                   (A1-A34, ~55 min)
#   7. measure_memorisation.py                     (NN-ratio diagnostic)
#   8. plot_diagnostics_multiasset.py, plot_losses.py
#   9. render_readme.py                            (Deep-MKV-TS/README.md)
#  10. tools/render_comparison.py                  (HestonMultiAsset/README.md)
#
# Step 5 is a hard gate on purpose: a broken .npy produces normal-looking numbers
# 55 minutes later, so the chain must stop before step 6, not after it.
#
# GPU 0 is never used -- standing instruction.
#
#   setsid bash run_pipeline.sh > logs/pipeline.log 2>&1 & disown

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
MA_DIR="$(dirname "$METHOD_DIR")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python
GPU="${GPU:-1}"
CORES="${CORES:-8-15}"
SEEDS_SPACE="0 2 4 5 6"
SEEDS_COMMA="0,2,4,5,6"

export PYTHONPATH="$REF/src:$REF/experiments"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"

run() {
  local label="$1"; shift
  echo "=== [$label] START $(date -Is)"
  taskset -c "$CORES" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== [$label] FAILED rc=$rc $(date -Is) -- CHAIN STOPPED" >&2
    exit "$rc"
  fi
  echo "=== [$label] OK $(date -Is)"
}

# -------------------------------------------------------------- 1. wait ------
echo "=== waiting for seed 6 trainer $(date -Is)"
while pgrep -f "train_multiasse[t].py --seed 6" >/dev/null; do sleep 60; done
echo "=== seed 6 trainer exited $(date -Is)"

if [ ! -f "$HERE/runs/seed_6/training_checkpoints/step_3000.pt" ]; then
  echo "ABORT: seed 6 has no step_3000.pt -- it did not finish cleanly." >&2
  exit 1
fi

# ------------------------------------------------------------ 2. select ------
run "select seed 6" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds 6 --device cuda:0

# ---------------------------------------------------------- 3. sentinel ------
# run_campaign.sh writes this only when every launched seed succeeded, which it
# cannot do here: seeds 1 and 3 diverged. The five REPORTED seeds all finished,
# so the sentinel is written by hand and records exactly which seeds it covers.
printf '%s seeds=%s (seeds 1 and 3 diverged, replaced by 5 and 6)\n' \
    "$(date -Is)" "$SEEDS_COMMA" > "$METHOD_DIR/weights/.campaign_complete"
echo "=== [sentinel] wrote weights/.campaign_complete"

# -------------------------------------------------------- 4. generation ------
run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SEEDS_SPACE --device cuda:0

# ------------------------------------------------------------ 5. gate --------
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"

# --------------------------------------------------------- 6. metrics --------
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method Deep-MKV-TS --seed-list "$SEEDS_COMMA"

# --------------------------------------------------- 7. memorisation ---------
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"

# ----------------------------------------------------------- 8. plots --------
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"

# --------------------------------------------------------- 9. README ---------
run "render method README" "$PY" "$HERE/render_readme.py"

# ----------------------------------------------------- 10. comparison --------
run "render comparison" "$PY" "$MA_DIR/tools/render_comparison.py" \
    --methods SBTS,LS4,CSDI,Deep-MKV-TS,reference

# ------------------------------------------------- 11. commit and push -------
# The repo working tree is dirty with a dozen unrelated experiments (DoubleWell,
# OU, CSDI repro, TimeMoDE...).  `git add -A` would sweep all of it into a commit
# that claims to be about the d = 8 Deep-MKV-TS campaign.  Every path below is
# therefore named explicitly, and a guard re-checks the STAGED set afterwards --
# an allowlist that is never verified is just a longer way of trusting yourself.
echo "=== [commit] START $(date -Is)"
cd "$BENCH" || exit 1

git add -- \
  "results/HestonMultiAsset/README.md" \
  "results/HestonMultiAsset/tools/render_comparison.py" \
  "results/HestonMultiAsset/Deep-MKV-TS/README.md" \
  "results/HestonMultiAsset/Deep-MKV-TS/metrics_summary.csv" \
  "results/HestonMultiAsset/Deep-MKV-TS/metrics_per_asset.csv" \
  "results/HestonMultiAsset/Deep-MKV-TS/curve_b_aggregate.json" \
  "results/HestonMultiAsset/Deep-MKV-TS/grid_tvd_aggregate.json" \
  "results/HestonMultiAsset/Deep-MKV-TS/seed_"*"_metrics.json" \
  "results/HestonMultiAsset/Deep-MKV-TS/seed_"*"_loss.csv" \
  "results/HestonMultiAsset/Deep-MKV-TS/losses" \
  "results/HestonMultiAsset/Deep-MKV-TS/plots" \
  "results/HestonMultiAsset/Deep-MKV-TS/weights" \
  "results/HestonMultiAsset/Deep-MKV-TS/generated_paths" \
  "results/HestonMultiAsset/Deep-MKV-TS/code" \
  2>/dev/null

# code/runs/ is 12 MB of intermediate training checkpoints; the SELECTED weights
# are already in weights/.  __pycache__ and .omc are machine state, not results.
git reset -q -- \
  "results/HestonMultiAsset/Deep-MKV-TS/code/runs" \
  "results/HestonMultiAsset/Deep-MKV-TS/code/__pycache__" \
  "results/HestonMultiAsset/Deep-MKV-TS/.omc" 2>/dev/null

# Guard. Refuses on: anything outside the d = 8 Deep-MKV-TS tree (bar the two
# shared README/renderer files), any .npy, or any blob over 50 MB.
"$PY" - <<'PYEOF' || exit 1
import subprocess, sys, os
BENCH = "/home/tbasseras/benchmark"
ALLOW_EXACT = {
    "results/HestonMultiAsset/README.md",
    "results/HestonMultiAsset/tools/render_comparison.py",
}
PREFIX = "results/HestonMultiAsset/Deep-MKV-TS/"
staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        cwd=BENCH, capture_output=True, text=True).stdout.split()
if not staged:
    print("[guard] nothing staged -- nothing to commit"); sys.exit(3)
bad = []
for p in staged:
    if p not in ALLOW_EXACT and not p.startswith(PREFIX):
        bad.append(f"OUT OF SCOPE: {p}")
    if p.endswith(".npy"):
        bad.append(f"PATH BANK: {p}")
    full = os.path.join(BENCH, p)
    if os.path.exists(full) and os.path.getsize(full) > 50 * 1024 * 1024:
        bad.append(f"OVER 50 MB: {p} ({os.path.getsize(full)/1e6:.0f} MB)")
if bad:
    print("[guard] REFUSING TO COMMIT:"); [print("   " + b) for b in bad]; sys.exit(1)
print(f"[guard] ok: {len(staged)} files, all inside the d=8 Deep-MKV-TS tree")
PYEOF
guard_rc=$?
if [ "$guard_rc" -eq 3 ]; then
  echo "=== [commit] nothing to commit -- skipping push"
  echo "=== PIPELINE COMPLETE $(date -Is)"
  exit 0
fi

# Selected steps are READ from the selection records, never typed into the message.
SEL="$("$PY" -c "
import json,glob
D='$HERE/selection'
r=[(json.load(open(p))['seed'],json.load(open(p))['selected_step']) for p in sorted(glob.glob(D+'/seed_*_selection.json'))]
print(', '.join(f'seed {s}->{k}' for s,k in sorted(r)))")"

git commit -F - <<EOF
Deep-MKV-TS d = 8 multi-asset Heston: 5-seed campaign, metrics and READMEs

Full A1-A34 evaluation of Deep-MKV-TS on the d = 8 multi-asset Heston
benchmark, plus the method page and the Deep-MKV-TS column on the
comparison page.

Seeds reported: $SEEDS_COMMA. Seeds 1 and 3 diverged with a non-finite
control and were replaced by 5 and 6; the numbering gaps are kept visible
so a 33% stability failure rate is not laundered into a clean {0..4} run.

Validation-selected checkpoints: $SEL

Hyperparameters: ridge_lambda = 1000, re-selected at d = 8 and bracketed
on both sides (300 worse, 3000 diverges). The four discrepancy weights are
inherited from d = 1 and were verified, not assumed: raising either ACF
weight makes the control explode, and LAMBDA_SCALE / KAPPA_SCALE move the
metric by less than the 46% checkpoint-to-checkpoint noise floor.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
if [ $? -ne 0 ]; then
  echo "=== [commit] FAILED -- not pushing" >&2; exit 1
fi
echo "=== [commit] OK $(date -Is)"

echo "=== [push] START $(date -Is)"
if git push origin master; then
  echo "=== [push] OK $(date -Is)"
else
  echo "=== [push] FAILED -- commit is local only $(date -Is)" >&2
  exit 1
fi

echo "=== PIPELINE COMPLETE $(date -Is)"
