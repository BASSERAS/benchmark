#!/usr/bin/env bash
# Unattended driver: retune -> method root -> full suite -> both READMEs -> push.
#
# Operator instruction, verbatim:
#   "I will come back in 1 hour so pls queue everything st when I come back all
#    the readme and the display is already commited and pushed pls"
# and, on which run represents the method:
#   "retune only and confirm to me pls it has good scores about improving the
#    reference model especially on table B and C"
#
# This is the ONLY script that writes into the Deep-MKV-TS method root. Nothing
# above it does, on purpose: run_retune_lr_progressive.sh md5-checks
# weights/seed_*_model.pt at its very end and aborts if they moved, so the
# promotion cannot begin until that process has exited.
#
# Naming note for the process guards below: this file is `run_publish_retune.sh`
# and the two it waits on are `run_retune_lr_progressive.sh` and
# `run_retune_crps_prestage.sh`. No pgrep pattern used here matches this script's
# own command line, so there is no self-match hazard.
#
# What "promotion" means and why the old weights are kept
# ------------------------------------------------------
# The weights currently at the method root are lr = 0.002, carried unchanged from
# the Heston d = 8 run. On this dataset the control map adds Zhat / sqrt(dt) with
# dt = 9.51e-07, so that rate displaces Theta by 2.05 per step against Heston's
# 0.032 -- a 64.6x overshoot. They were never generated from and never scored.
# They are MOVED to weights_lr2e-3/, never deleted: they are the evidence for the
# claim above.
#
# Why STAGE 5 of the pipeline is allowed to skip
# ----------------------------------------------
# run_retune_lr_progressive.sh already ran compute_all_multiasset.py over all
# five retune banks and wrote metrics_summary.csv, metrics_per_asset.csv,
# curve_b_aggregate.json and grid_tvd_aggregate.json into retune_lr/. Those files
# carry NO method label (verified: `grep -c Deep-MKV-TS metrics_summary.csv` is
# 0; the columns are metric, scope, mean, std, seed_0..seed_4), so copying them
# to the method root is a rename, not a re-attribution. Re-running the stage
# would burn ~40 min producing the same numbers from the same banks.
#
# The CRPS pools and the 12 conditional-CRPS configs come from
# run_retune_crps_prestage.sh, which built them inside retune_lr/ while the
# scorer was still running.
set -u

cd /home/tbasseras/benchmark || exit 1

B=/home/tbasseras/benchmark
R=$B/results/trueexperiment
M=$R/Deep-MKV-TS
C=$M/code
O=$M/retune_lr
TAG=6144x128x8
CC=/home/tbasseras/.cc-venv/bin/python
SEEDS="0 1 2 3 4"

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { say "!! ABORT: $*"; exit 1; }

# ===========================================================================
say "PUBLISH 1  wait for run_retune_lr_progressive.sh"
# ===========================================================================
DEADLINE=$(( $(date +%s) + 5400 ))
while pgrep -f "run_retune_lr_progressive.sh" > /dev/null 2>&1; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && die "progressive scorer still alive after 90 min"
    sleep 30
done
say "     scorer exited"
[ -f "$O/metrics_summary.csv" ] || die "$O/metrics_summary.csv absent -- the 5-seed
     aggregate never ran. Check $O/logs/metrics.log; do NOT publish a partial run."
[ -f "$O/curve_b_aggregate.json" ]  || die "$O/curve_b_aggregate.json absent"
[ -f "$O/grid_tvd_aggregate.json" ] || die "$O/grid_tvd_aggregate.json absent"
# The aggregate must actually contain all five seed columns. A summary with only
# seed_1 would render a 5-seed table from one seed and say nothing about it.
head -1 "$O/metrics_summary.csv" | grep -q "seed_0" || die "no seed_0 column in aggregate"
head -1 "$O/metrics_summary.csv" | grep -q "seed_4" || die "no seed_4 column in aggregate"
say "     5-seed aggregate present and complete"

# ===========================================================================
say "PUBLISH 2  wait for run_retune_crps_prestage.sh"
# ===========================================================================
DEADLINE=$(( $(date +%s) + 5400 ))
while pgrep -f "run_retune_crps_prestage.sh" > /dev/null 2>&1; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && die "CRPS prestage still alive after 90 min"
    sleep 30
done
# Non-fatal by design: if the prestage died, run_pipeline_post.sh's STAGE 3 and
# STAGE 7 rebuild exactly these files. It costs ~25 min, it does not cost
# correctness, and aborting here would throw away everything above.
PRESTAGE_OK=1
for CFG in paper perdim; do
    for S in $SEEDS; do
        [ -f "$O/losses/crps_configs/${CFG}__seed_$S.json" ] || PRESTAGE_OK=0
    done
    [ -f "$O/losses/crps_configs/${CFG}__realbank.json" ] || PRESTAGE_OK=0
done
if [ "$PRESTAGE_OK" -eq 1 ]; then
    say "     all 12 CRPS configs pre-built -- pipeline STAGE 7 will skip"
else
    say "     !! prestage incomplete -- run_pipeline_post.sh will rebuild the"
    say "        missing CRPS jobs itself (~25 min). Continuing."
    tail -n 5 "$O/logs/prestage.log" 2>/dev/null
fi

# ===========================================================================
say "PUBLISH 3  promote retune_lr -> method root"
# ===========================================================================
# Idempotent: if weights_lr2e-3/ already exists this script has run before, and
# moving again would overwrite the preserved lr=0.002 weights with the retune.
if [ -d "$M/weights_lr2e-3" ]; then
    say "     weights_lr2e-3/ exists -- promotion already done, skipping the move"
else
    [ -f "$M/weights/seed_0_model.pt" ] || die "no weights at the method root to preserve"
    mv "$M/weights" "$M/weights_lr2e-3" || die "could not preserve lr=0.002 weights"
    say "     lr=0.002 weights preserved at weights_lr2e-3/"
fi

mkdir -p "$M/weights" "$C/selection" "$M/generated_paths" "$M/crps_banks" \
         "$M/losses/crps_configs" "$M/plots" "$M/logs"

for S in $SEEDS; do
    cp -f "$O/weights/seed_${S}_model.pt"    "$M/weights/"  || die "copy weights seed $S"
    cp -f "$O/weights/seed_${S}_config.json" "$M/weights/"  || die "copy config seed $S"
    cp -f "$O/selection/seed_${S}_selection.json" "$C/selection/" || die "copy selection seed $S"
    cp -f "$O/losses/seed_${S}_losses.csv"   "$M/losses/"   || die "copy losses seed $S"
done

# Byte-identical or the method root is not the retune. md5 rather than size:
# the trainer's own artefact and the selector's payload differ by 1477 bytes,
# which a size check would catch by luck, not by construction.
for S in $SEEDS; do
    a=$(md5sum "$O/weights/seed_${S}_model.pt" | cut -d' ' -f1)
    b=$(md5sum "$M/weights/seed_${S}_model.pt" | cut -d' ' -f1)
    [ "$a" = "$b" ] || die "seed $S weights differ after copy ($a vs $b)"
done
say "     5 selected checkpoints promoted, md5-identical"

cp -rf "$O/generated_paths/." "$M/generated_paths/" || die "copy A/B banks"
cp -rf "$O/crps_banks/."      "$M/crps_banks/"      || die "copy CRPS pools"
if [ "$PRESTAGE_OK" -eq 1 ]; then
    cp -f "$O"/losses/crps_configs/*.json "$M/losses/crps_configs/" || die "copy crps configs"
fi
cp -f "$O/metrics_summary.csv" "$O/curve_b_aggregate.json" \
      "$O/grid_tvd_aggregate.json" "$M/" || die "copy aggregates"
[ -f "$O/metrics_per_asset.csv" ] && cp -f "$O/metrics_per_asset.csv" "$M/"
# plot_score_losses.py (STAGE 8) reads these out of --results-dir. Stage 5 is
# being skipped, so they have to be carried across with the summary they
# describe. Globs may be empty; that is not fatal, STAGE 8 reports it.
cp -f "$O"/seed_*_metrics.json "$M/" 2>/dev/null
cp -f "$O"/seed_*_loss.csv     "$M/" 2>/dev/null

# The envelope screen on disk was computed from the lr=0.002 banks. STAGE 8b
# recomputes it unconditionally, but if 8b were to fail the renderer would print
# the stale number as if it described the retune. Park it rather than trust the
# overwrite.
if [ -f "$M/losses/envelope_screen.json" ] && [ ! -f "$M/losses/envelope_screen_lr2e-3.json" ]; then
    mv "$M/losses/envelope_screen.json" "$M/losses/envelope_screen_lr2e-3.json"
    say "     stale lr=0.002 envelope screen parked as envelope_screen_lr2e-3.json"
fi

for S in $SEEDS; do
    [ -f "$M/generated_paths/seed_$S/generated_paths_$TAG.npy" ] \
        || die "A/B bank seed $S missing after promote"
done
say "     promotion complete"

# ===========================================================================
say "PUBLISH 4  run_pipeline_post.sh (stages 1,2,3,5,7 should all skip)"
# ===========================================================================
# GPUs 0 and 1: 2 and 3 belong to another user and were at ~60% when this was
# queued. Two lanes is enough -- every GPU stage that remains is short.
SEEDS="0 1 2 3 4" GPUS="0 1" LANE_W=20 WAIT_HOURS=1 \
    bash "$C/run_pipeline_post.sh" > "$M/logs/publish_pipeline.log" 2>&1
RC=$?
tail -n 40 "$M/logs/publish_pipeline.log"
[ "$RC" -eq 0 ] || die "run_pipeline_post.sh failed (rc=$RC) -- see $M/logs/publish_pipeline.log"
[ -f "$M/README.md" ] || die "method README not produced"

# ===========================================================================
say "PUBLISH 5  render the cross-method comparison page"
# ===========================================================================
bash "$R/run_render_comparison.sh" > "$M/logs/render_comparison.log" 2>&1 \
    || { tail -n 30 "$M/logs/render_comparison.log"; die "render_comparison"; }

# render_comparison.py's readers return {} for a missing file instead of raising,
# so a method whose artefacts never landed renders as a full column of dashes
# with no error anywhere. This is the only thing standing between that and a
# pushed commit.
$CC - <<'PYEOF' || exit 1
import re, sys, pathlib
p = pathlib.Path("/home/tbasseras/benchmark/results/trueexperiment/README.md")
txt = p.read_text(encoding="utf-8")
if "Deep-MKV-TS" not in txt:
    sys.exit("ABORT: comparison README has no Deep-MKV-TS column")
if "reproduced exactly" in txt:
    sys.exit("ABORT: comparison README contains the banned phrase")
lines = [l for l in txt.splitlines() if l.startswith("|")]
hdr = next((l for l in lines if "Deep-MKV-TS" in l), None)
cols = [c.strip() for c in hdr.strip("|").split("|")]
idx = cols.index(next(c for c in cols if "Deep-MKV-TS" in c))
num = dash = 0
for l in lines:
    cells = [c.strip() for c in l.strip("|").split("|")]
    if len(cells) != len(cols) or cells == cols:
        continue
    v = cells[idx]
    if re.search(r"\d", v):
        num += 1
    elif v in ("-", "--", ""):
        dash += 1
print(f"[gate] Deep-MKV-TS column: {num} numeric rows, {dash} dashes")
if num < 20:
    sys.exit(f"ABORT: only {num} numeric rows in the Deep-MKV-TS column -- "
             "its artefacts did not reach the renderer")
PYEOF
say "     comparison page renders with a populated Deep-MKV-TS column"

# ===========================================================================
say "PUBLISH 6  section-12 registration gates"
# ===========================================================================
# Guideline section 12: verify the ignore rules, never assume them. Every .npy
# that is new in this commit must be matched by an explicit rule, and nothing
# under results/trueexperiment may reach the index as a path bank.
NPY_FAIL=0
while IFS= read -r f; do
    git check-ignore -q "$f" || { say "     NOT IGNORED: $f"; NPY_FAIL=1; }
done < <(find "$M/generated_paths" "$M/crps_banks" "$O" -name '*.npy' 2>/dev/null)
[ "$NPY_FAIL" -eq 0 ] || die "a .npy under the method root is not covered by .gitignore"

LEAK=$(git status --porcelain -uall -- results/trueexperiment | grep '\.npy' || true)
[ -z "$LEAK" ] || { echo "$LEAK"; die "path banks are visible to git under results/trueexperiment"; }
say "     no .npy reachable by git under results/trueexperiment"

# ===========================================================================
say "PUBLISH 7  stage, verify, commit, push"
# ===========================================================================
# Named paths only -- never `git add -A`. `git add <dir>` honours .gitignore, so
# the .npy banks inside these trees stay out; the gate below proves it rather
# than trusting it. diagnostic_bestfit/ is deliberately NOT staged: its banks
# have no metadata.json beside them, so registering that tree would record
# nothing about how those paths were produced.
git add \
    "$M/README.md" \
    "$M/metrics_summary.csv" \
    "$M/curve_b_aggregate.json" \
    "$M/grid_tvd_aggregate.json" \
    "$M/code" \
    "$M/losses" \
    "$M/weights" \
    "$M/weights_lr2e-3" \
    "$M/plots" \
    "$M/generated_paths" \
    "$M/crps_banks" \
    "$O/selection" \
    "$O/early" \
    "$O/losses" \
    "$O/logs" \
    "$R/README.md" \
    "$R/render_comparison.py" \
    "$R/run_render_comparison.sh" \
    2>&1 | tail -n 5
[ -f "$M/metrics_per_asset.csv" ] && git add "$M/metrics_per_asset.csv"

STAGED=$(git diff --cached --name-only)
[ -n "$STAGED" ] || die "nothing staged"

echo "$STAGED" | grep -q '\.npy$' && die "a .npy reached the index"
OUTSIDE=$(echo "$STAGED" | grep -v '^results/trueexperiment/' || true)
[ -z "$OUTSIDE" ] || { echo "$OUTSIDE"; die "staged a path outside results/trueexperiment"; }
say "     $(echo "$STAGED" | wc -l) files staged, all under results/trueexperiment, zero .npy"

git commit -q -F - <<'MSGEOF' || die "commit failed"
trueexperiment: publish Deep-MKV-TS at lr=6.4e-06 and add it to the comparison

The method root previously held lr=0.002 weights carried over unchanged from the
Heston d=8 run, and nothing had ever been generated from or scored against them.
That rate does not transfer: the control map is

    Theta = eta * sigma_ref^{-1} + Zhat / sqrt(dt)

so the per-step displacement of Theta scales as lr / sqrt(dt). Heston uses
dt = 1/252 and TrueDataset dt = 9.512937595129376e-07, a 64.6x difference in
1/sqrt(dt), which put Theta/step at 2.05 here against 0.032 on Heston. The
post-hoc alpha grid independently placed the admissible control magnitude at
0.0032 of the shipped one; scaling the rate by the same factor gives
lr = 6.4e-06 and Theta/step = 0.0066.

Five seeds were retrained at that rate and scored against the untrained
multivariate reference SDE (the same control map with Zhat == 0), using the
floor-relative rule of section 7.1: the real-vs-real floor is a target, not a
minimum, so a row is won by whichever run is closer to it in |log(value/floor)|,
in either direction. A28 is excluded, its target being 1.0 rather than the
floor's own value.

The lr=0.002 weights are preserved under weights_lr2e-3/ rather than deleted --
they are the evidence for the paragraph above.

CRPS is reported under both conventions (paper: weight-mode paper, standardize
bank; perdim: weight-mode perdim, standardize realtrain), per section 14.

Path banks are gitignored and stay so; the metadata.json beside each bank is the
surviving audit record.
MSGEOF

say "     committed: $(git rev-parse --short HEAD)"
git push || die "push failed"
say "=========================================================="
say "PUBLISHED"
say "  method README      $M/README.md"
say "  comparison README  $R/README.md"
say "  commit             $(git rev-parse --short HEAD)"
say "=========================================================="
