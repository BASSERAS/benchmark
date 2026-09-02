#!/usr/bin/env bash
# Stage, guard, commit and push the d = 8 Deep-MKV-TS results.
#
# Split out of run_pipeline.sh so it can be invoked by a watcher AFTER the
# pipeline exits, without editing a script bash is currently executing.
#
# IDEMPOTENT: if nothing is staged (already committed) it exits 0 without
# creating an empty commit, so running it twice is harmless.
#
# The repo working tree is dirty with a dozen unrelated experiments (DoubleWell,
# OU, CSDI repro, TimeMoDE...). `git add -A` would sweep all of it into a commit
# that claims to be about the d = 8 campaign. Every path is therefore named
# explicitly, and a guard re-checks the STAGED set afterwards -- an allowlist
# that is never verified is just a longer way of trusting yourself.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
SEEDS_COMMA="0,2,4,5,6"

cd "$BENCH" || exit 1
echo "=== [commit] START $(date -Is)"

git add -- \
  "results/HestonMultiAsset/README.md" \
  "results/HestonMultiAsset/tools/render_comparison.py" \
  "results/HestonMultiAsset/Deep-MKV-TS/README.md" \
  "results/HestonMultiAsset/Deep-MKV-TS/metrics_summary.csv" \
  "results/HestonMultiAsset/Deep-MKV-TS/metrics_per_asset.csv" \
  "results/HestonMultiAsset/Deep-MKV-TS/curve_b_aggregate.json" \
  "results/HestonMultiAsset/Deep-MKV-TS/grid_tvd_aggregate.json" \
  "results/HestonMultiAsset/Deep-MKV-TS/losses" \
  "results/HestonMultiAsset/Deep-MKV-TS/plots" \
  "results/HestonMultiAsset/Deep-MKV-TS/weights" \
  "results/HestonMultiAsset/Deep-MKV-TS/generated_paths" \
  "results/HestonMultiAsset/Deep-MKV-TS/code" \
  2>/dev/null
# Globs are added separately: an unmatched glob makes the whole `git add` fail.
for g in "results/HestonMultiAsset/Deep-MKV-TS/seed_"*"_metrics.json" \
         "results/HestonMultiAsset/Deep-MKV-TS/seed_"*"_loss.csv"; do
  compgen -G "$g" >/dev/null && git add -- $g 2>/dev/null
done

# code/runs/ is 12 MB of intermediate training checkpoints; the SELECTED weights
# are already in weights/. __pycache__ and .omc are machine state, not results.
git reset -q -- \
  "results/HestonMultiAsset/Deep-MKV-TS/code/runs" \
  "results/HestonMultiAsset/Deep-MKV-TS/code/__pycache__" \
  "results/HestonMultiAsset/Deep-MKV-TS/.omc" 2>/dev/null

# Guard. Refuses on: anything outside the d = 8 Deep-MKV-TS tree (bar the two
# shared files), any .npy path bank, or any blob over 50 MB.
"$PY" - <<'PYEOF'
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
    print("[guard] nothing staged -- already committed, nothing to do")
    sys.exit(3)
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
    print("[guard] REFUSING TO COMMIT:")
    for b in bad:
        print("   " + b)
    sys.exit(1)
print(f"[guard] ok: {len(staged)} files, all inside the d=8 Deep-MKV-TS tree")
PYEOF
guard_rc=$?
if [ "$guard_rc" -eq 3 ]; then
  echo "=== [commit] nothing to commit -- done"; exit 0
fi
if [ "$guard_rc" -ne 0 ]; then
  echo "=== [commit] GUARD REFUSED -- not committing" >&2; exit 1
fi

# Selected steps are READ from the selection records, never typed into the message.
SEL="$("$PY" -c "
import json,glob
r=[]
for p in sorted(glob.glob('$HERE/selection/seed_*_selection.json')):
    d=json.load(open(p)); r.append((d['seed'], d['selected_step']))
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
echo "=== [commit+push] COMPLETE $(date -Is)"
