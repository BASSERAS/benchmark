#!/usr/bin/env bash
# End-to-end d = 8 pipeline for Deep-MKV-TS-SBTSref, run once BOTH training
# chains finish.
#
# Chain, in order, each step gated on the previous one succeeding:
#
#   1. wait for every train_multiasset.py process to exit
#   2. preflight: all 5 seeds reached step_3000.pt, all scripts present
#   3. select_checkpoint_multiasset.py            (validation checkpoint choice)
#   4. weights/.campaign_complete sentinel
#   5. run_all_multiasset.py                      (generate 5 x 8192 test paths)
#   6. collect_artifacts.py                       (GUIDELINE section 4 contract gate)
#   7. compute_all_multiasset.py                  (A1-A34, ~55 min)
#   8. measure_memorisation.py                    (NN-ratio diagnostic)
#   9. plot_diagnostics_multiasset.py, plot_losses.py
#  10. render_readme.py                           (this method's README.md)
#  11. tools/render_comparison.py                 (HestonMultiAsset/README.md)
#
# Step 6 is a hard gate on purpose: a broken .npy produces normal-looking numbers
# 55 minutes later, so the chain must stop before step 7, not after it.
#
# Step 2 is the other hard gate, and it is the one the sibling did not have.
# Two of the sibling's seeds died mid-training and the pipeline only found out
# much later. Here the run refuses to start if any of seeds 0-4 is missing its
# step_3000.pt, and it names the missing seeds. It does NOT substitute a spare
# seed and it does NOT renumber: a stability failure is a result to report, not
# a gap to paper over.
#
# There is no reference-fitting step. Unlike the sibling, this method's reference
# is not fitted -- b^ref and sigma^ref are the SBTS kernel evaluated against the
# training bank, rebuilt at load time from the recorded (h, K, npi, grad mode).
#
# Committing and pushing is OPT-IN
# --------------------------------
# The sibling's pipeline committed and pushed at the end. This one does not,
# unless you set PUSH=1. Publishing a result is a decision, and this script runs
# unattended for hours; it should not make that decision on its own. With PUSH=1
# the same explicit allowlist and staged-set guard as the sibling apply.
#
#   setsid bash run_pipeline.sh > logs/pipeline.log 2>&1 & disown
#   PUSH=1 setsid bash run_pipeline.sh > logs/pipeline.log 2>&1 & disown

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(dirname "$HERE")"
MA_DIR="$(dirname "$METHOD_DIR")"
METHOD="$(basename "$METHOD_DIR")"
BENCH=/home/tbasseras/benchmark
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python

# Both training chains occupy GPU 0 (cores 0-7) and GPU 1 (cores 8-15) until they
# exit. By the time this script gets past step 1 they are free, so it takes GPU 0
# and cores 0-7 -- one GPU, eight cores, inside the 2-GPU / 16-core cap.
GPU="${GPU:-0}"
CORES="${CORES:-0-7}"
PUSH="${PUSH:-0}"
SEEDS_SPACE="0 1 2 3 4"
SEEDS_COMMA="0,1,2,3,4"
FINAL_STEP="${FINAL_STEP:-3000}"

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
# The bracket in the pattern keeps pgrep from matching its own command line.
echo "=== waiting for all train_multiasset.py processes $(date -Is)"
while pgrep -f "train_multiasse[t].py" >/dev/null; do sleep 60; done
echo "=== all trainers exited $(date -Is)"

# --------------------------------------------------------- 2. preflight ------
missing_seeds=()
for s in $SEEDS_SPACE; do
  if [ ! -f "$HERE/runs/seed_$s/training_checkpoints/step_$(printf '%04d' "$FINAL_STEP").pt" ]; then
    missing_seeds+=("$s")
  fi
done
if [ "${#missing_seeds[@]}" -ne 0 ]; then
  echo "ABORT: seed(s) ${missing_seeds[*]} never reached step $FINAL_STEP." >&2
  echo "       This is a training failure, not a missing file. Report it." >&2
  echo "       Do NOT substitute another seed and do NOT renumber." >&2
  exit 1
fi
echo "=== [preflight] all 5 seeds reached step $FINAL_STEP"

missing_scripts=()
for f in select_checkpoint_multiasset.py run_all_multiasset.py collect_artifacts.py \
         measure_memorisation.py plot_diagnostics_multiasset.py plot_losses.py \
         render_readme.py; do
  [ -f "$HERE/$f" ] || missing_scripts+=("$f")
done
[ -f "$BENCH/metrics/compute_all_multiasset.py" ] || missing_scripts+=("metrics/compute_all_multiasset.py")
[ -f "$MA_DIR/tools/render_comparison.py" ] || missing_scripts+=("tools/render_comparison.py")
if [ "${#missing_scripts[@]}" -ne 0 ]; then
  echo "ABORT: missing script(s): ${missing_scripts[*]}" >&2
  echo "       Failing now rather than 55 minutes into the metrics stage." >&2
  exit 1
fi
echo "=== [preflight] all pipeline scripts present"

# ------------------------------------------------------------ 3. select ------
run "select checkpoints" "$PY" "$HERE/select_checkpoint_multiasset.py" \
    --seeds $SEEDS_SPACE --device cuda:0

# ---------------------------------------------------------- 4. sentinel ------
printf '%s seeds=%s method=%s\n' \
    "$(date -Is)" "$SEEDS_COMMA" "$METHOD" > "$METHOD_DIR/weights/.campaign_complete"
echo "=== [sentinel] wrote weights/.campaign_complete"

# -------------------------------------------------------- 5. generation ------
run "generate paths" "$PY" "$HERE/run_all_multiasset.py" \
    --seeds $SEEDS_SPACE --device cuda:0

# ------------------------------------------------------------ 6. gate --------
run "collect artifacts" "$PY" "$HERE/collect_artifacts.py"

# --------------------------------------------------------- 7. metrics --------
run "compute A1-A34" "$PY" "$BENCH/metrics/compute_all_multiasset.py" \
    --method "$METHOD" --seed-list "$SEEDS_COMMA"

# --------------------------------------------------- 8. memorisation ---------
run "memorisation" "$PY" "$HERE/measure_memorisation.py" --seeds "$SEEDS_COMMA"

# ----------------------------------------------------------- 9. plots --------
run "plot diagnostics" "$PY" "$HERE/plot_diagnostics_multiasset.py"
run "plot losses" "$PY" "$HERE/plot_losses.py"

# -------------------------------------------------------- 10. README ---------
run "render method README" "$PY" "$HERE/render_readme.py"

# ---------------------------------------------------- 11. comparison ---------
run "render comparison" "$PY" "$MA_DIR/tools/render_comparison.py" \
    --methods "SBTS,LS4,CSDI,Deep-MKV-TS,$METHOD,reference"

if [ "$PUSH" != "1" ]; then
  echo "=== PIPELINE COMPLETE (not committed -- rerun with PUSH=1 to publish) $(date -Is)"
  exit 0
fi

# ------------------------------------------------ 12. commit and push --------
# The repo working tree is dirty with a dozen unrelated experiments. `git add -A`
# would sweep all of it into a commit that claims to be about this campaign.
# Every path below is therefore named explicitly, and a guard re-checks the
# STAGED set afterwards -- an allowlist that is never verified is just a longer
# way of trusting yourself.
echo "=== [commit] START $(date -Is)"
cd "$BENCH" || exit 1

P="results/HestonMultiAsset/$METHOD"
git add -- \
  "results/HestonMultiAsset/README.md" \
  "results/HestonMultiAsset/tools/render_comparison.py" \
  "$P/README.md" \
  "$P/metrics_summary.csv" \
  "$P/metrics_per_asset.csv" \
  "$P/curve_b_aggregate.json" \
  "$P/grid_tvd_aggregate.json" \
  "$P/seed_"*"_metrics.json" \
  "$P/seed_"*"_loss.csv" \
  "$P/losses" \
  "$P/plots" \
  "$P/weights" \
  "$P/generated_paths" \
  "$P/code" \
  2>/dev/null

# code/runs/ is intermediate training checkpoints; the SELECTED weights are
# already in weights/. __pycache__ and .omc are machine state, not results.
git reset -q -- "$P/code/runs" "$P/code/__pycache__" "$P/.omc" 2>/dev/null

# Guard. Refuses on: anything outside this method's tree (bar the two shared
# README/renderer files), any .npy, or any blob over 50 MB.
METHOD="$METHOD" "$PY" - <<'PYEOF' || exit 1
import subprocess, sys, os
BENCH = "/home/tbasseras/benchmark"
METHOD = os.environ["METHOD"]
ALLOW_EXACT = {
    "results/HestonMultiAsset/README.md",
    "results/HestonMultiAsset/tools/render_comparison.py",
}
PREFIX = f"results/HestonMultiAsset/{METHOD}/"
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
print(f"[guard] ok: {len(staged)} files, all inside the {METHOD} tree")
PYEOF
guard_rc=$?
if [ "$guard_rc" -eq 3 ]; then
  echo "=== [commit] nothing to commit -- skipping push"
  echo "=== PIPELINE COMPLETE $(date -Is)"
  exit 0
fi

# Selected steps and hyperparameters are READ from the recorded artefacts, never
# typed into the message by hand.
SEL="$("$PY" -c "
import json,glob
D='$HERE/selection'
r=[(json.load(open(p))['seed'],json.load(open(p))['selected_step']) for p in sorted(glob.glob(D+'/seed_*_selection.json'))]
print(', '.join(f'seed {s}->{k}' for s,k in sorted(r)))")"

CFG="$("$PY" -c "
import json
c=json.load(open('$METHOD_DIR/weights/seed_0_config.json'))
print('lr = {}, ridge_lambda = {}, h = {}, K = {}, npi = {}, weight_grad_mode = {}'.format(
    c['lr'], c['ridge_lambda'], c['reference_h'], c['reference_markov_order'],
    c['reference_npi'], c['reference_weight_grad_mode']))")"

git commit -F - <<EOF
$METHOD d = 8 multi-asset Heston: 5-seed campaign, metrics and READMEs

Full A1-A34 evaluation of Deep-MKV-TS run against an SBTS reference on the
d = 8 multi-asset Heston benchmark, plus the method page and this method's
column on the comparison page.

The reference is not fitted: b^ref is the SBTS Markovian kernel average over
the 8192-path training bank and sigma^ref is the corrected constant diffusion,
both rebuilt at load time from the recorded settings.

Seeds reported: $SEEDS_COMMA (all five launched, all five reported).

Validation-selected checkpoints: $SEL

Hyperparameters: $CFG
h, K and npi are sweep outcomes; weight_grad_mode = analytic is a correctness
setting, not a tuned one -- 'detached' drops d b^ref / d x from the backward
pass entirely and the method is specified to carry it. See code/SWEEP.md.

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
