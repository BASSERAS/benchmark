#!/usr/bin/env bash
# Re-render results/trueexperiment/README.md, the cross-method comparison page.
#
# Why this wrapper exists: the repo invokes its renderers from shell stages
# (see results/HestonMultiAsset/Deep-MKV-TS/code/run_pipeline.sh:99 for the
# Heston equivalent), and this makes the trueexperiment side match.
#
# What changed and why it is being re-run now: METHODS in render_comparison.py
# gained a third column, `reference` -- the untrained multivariate reference SDE
# (the Deep-MKV-TS control map with Zhat == 0). That column is the baseline every
# learned control starts from, so without it the table cannot answer whether a
# method adds anything over doing no learning at all.
#
# The reference has no losses/crps_configs/paper__*.json, so it renders as a "-"
# row in table C. render_c already handles that case; tables A and B are complete
# for it.
#
# Pure numpy/stdlib, no GPU, no torch. Reads and writes only inside
# results/trueexperiment/. A backup of the previous README is kept in /tmp so the
# diff is inspectable.
set -euo pipefail

T=/home/tbasseras/benchmark/results/trueexperiment
PY=/home/tbasseras/.cc-venv/bin/python

cd "$T"
[ -f README.md ] && cp -f README.md /tmp/README_trueexperiment_before.md

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$PY" "$T/render_comparison.py"

echo "--- previous README saved at /tmp/README_trueexperiment_before.md ---"
