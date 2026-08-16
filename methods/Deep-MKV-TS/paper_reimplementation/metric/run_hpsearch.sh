#!/usr/bin/env bash
# Deep-MKV-TS hyperparameter search, scored on the VALIDATION splits only.
#
# BENCHMARK_HESTON_EVAL redirects the scored split to heston_S_valdisc_8192x128.npy,
# so no trial ever sees heston_S_disc_8192x128.npy (the split the paper table is
# scored on) nor heston_S_test_8192x128.npy. Only the single winner from
# rank_hpsearch.py may afterwards be re-run on `disc`, exactly once.
#
# Two trials at a time on GPUs 0 and 1 (GPUs 2/3 and cores 0-15 are held by a
# concurrent job), 8 cores each per GUIDELINE.md 4.1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_ROOT="$(dirname "$HERE")"
REF="$PR_ROOT/../code/reference"
PY=/home/tbasseras/gpu-venv/bin/python
OUT_ROOT="$PR_ROOT/runs/hpsearch"
SEED="${SEED:-0}"
STEPS="${STEPS:-3000}"

mkdir -p "$OUT_ROOT"

# One line per trial: "<tag>|<extra flags appended to the base command>".
# The base command below carries the settings shared by every trial; each trial
# overrides only what its flag string names, so the tag describes its full delta.
# The reproduction lands close to the paper on SWD and MDD but loses on RV W1,
# |r| ACF and early-future. The grid therefore moves the three weights that
# drive exactly those columns: kappa (volatility MMD), abs-return-ACF, and the
# joint price/volatility term that couples the prefix to the future.
TRIALS=(
  "baseline|"
  "kappa200|--kappa-scale 200"
  "kappa400|--kappa-scale 400"
  "absacf0.50|--abs-return-acf-weight 0.50"
  "absacf1.00|--abs-return-acf-weight 1.00"
  "jointvol0.5|--source-joint-volatility-weight 0.5"
  "jointvol1.0|--source-joint-volatility-weight 1.0"
  "kappa200_absacf0.50|--kappa-scale 200 --abs-return-acf-weight 0.50"
  "kappa200_jointvol0.5|--kappa-scale 200 --source-joint-volatility-weight 0.5"
  "lambda100_kappa200|--lambda-scale 100 --kappa-scale 200"
  "lr0.001|--lr 0.001"
  "kappa200_absacf0.50_jointvol0.5|--kappa-scale 200 --abs-return-acf-weight 0.50 --source-joint-volatility-weight 0.5"
  "kappa400_absacf1.00|--kappa-scale 400 --abs-return-acf-weight 1.00"
  "lambda100_kappa200_absacf0.50|--lambda-scale 100 --kappa-scale 200 --abs-return-acf-weight 0.50"
)

run_trial() {
  local spec=$1 gpu=$2 cores=$3
  local tag="${spec%%|*}" extra="${spec#*|}"
  local out="$OUT_ROOT/$tag"
  if [[ -f "$out/COMPLETE.json" ]]; then echo "SKIP $tag (complete)"; return 0; fi
  mkdir -p "$out"
  echo "LAUNCH $tag gpu=$gpu cores=$cores extra=[$extra]"
  # shellcheck disable=SC2086  # $extra is an intentional flag list
  CUDA_VISIBLE_DEVICES="$gpu" \
  BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONPATH="$REF/src:$REF/experiments:$REF/experiments/scripts" \
  taskset -c "$cores" "$PY" \
    "$REF/experiments/scripts/run_matched_control_synthetic_validation.py" \
    --task heston --run-dir "$out" --device cuda:0 \
    --protocol-manifest "$HERE/PROTOCOL.json" \
    --seed "$SEED" --bank-size 8192 --sample-batch-size 2048 \
    --lambda-scale 50 --kappa-scale 100 --eta 1 --lr 0.002 --grad-clip-norm 5 \
    --reference-kind guyon_lekeufack_structural_likelihood \
    --joint-volatility-weight 0 --source-joint-volatility-weight 0 \
    --abs-return-acf-weight 0.25 --squared-return-acf-weight 0.125 \
    --source-only --source-steps "$STEPS" --solver online \
    --source-checkpoint-steps 500 1000 1500 2000 2500 3000 \
    --fitted-reference-drift-only \
    --adjoint-weight 0 --adjoint-noise-weight 1 \
    --path-derivative-backend autograd \
    --drift-adjoint-backend analytical_reference \
    $extra \
    > "$out/console.log" 2>&1
  echo "{\"tag\": \"$tag\", \"seed\": $SEED, \"steps\": $STEPS, \"extra\": \"$extra\"}" \
    > "$out/COMPLETE.json"
  echo "DONE $tag"
}

if [[ ${#TRIALS[@]} -eq 0 ]]; then
  echo "TRIALS is empty; populate the grid before running." >&2
  exit 1
fi

# Two slots, one per GPU; both must finish before the next pair starts.
i=0
while (( i < ${#TRIALS[@]} )); do
  run_trial "${TRIALS[$i]}" 0 16-23 & A=$!
  if (( i + 1 < ${#TRIALS[@]} )); then
    run_trial "${TRIALS[$((i+1))]}" 1 24-31 & B=$!
    wait $A $B
  else
    wait $A
  fi
  i=$((i + 2))
done

echo "ALL TRIALS DONE, ranking"
cd "$HERE" && "$PY" rank_hpsearch.py
