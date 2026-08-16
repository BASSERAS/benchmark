#!/usr/bin/env bash
# Confirm the top hyperparameter trials on the paper's remaining three seeds.
#
# run_hpsearch.sh scores every trial on a single seed (0). A few percent of
# margin there is inside seed-to-seed noise, so promoting that trial would be
# selection luck, not a result. This re-runs the named tags on seeds 1, 3 and 4
# so the winner can be chosen on the same 4-seed median the paper reports.
#
# Still validation-only: BENCHMARK_HESTON_EVAL keeps every run on
# heston_S_valdisc_8192x128.npy. `disc` is touched exactly once, later, by the
# single winner; `test` is never touched.
#
# Usage: bash run_confirm.sh <tag> [<tag> ...]
# Each tag must already exist under runs/hpsearch/<tag>/COMPLETE.json; its exact
# flag string is read back from there rather than restated, so a confirm run
# cannot silently diverge from the trial it is confirming.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_ROOT="$(dirname "$HERE")"
REF="$PR_ROOT/../code/reference"
PY=/home/tbasseras/gpu-venv/bin/python
SEARCH_ROOT="$PR_ROOT/runs/hpsearch"
OUT_ROOT="$PR_ROOT/runs/hpsearch_confirm"
STEPS="${STEPS:-3000}"
# Seed 0 is already done by the search itself and is reused, not re-run.
CONFIRM_SEEDS=(1 3 4)

if [[ $# -eq 0 ]]; then
  echo "usage: bash run_confirm.sh <tag> [<tag> ...]" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

extra_flags_for() {
  local tag=$1
  local complete="$SEARCH_ROOT/$tag/COMPLETE.json"
  [[ -f "$complete" ]] || { echo "no such trial: $complete" >&2; return 1; }
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["extra"])' "$complete"
}

run_one() {
  local tag=$1 seed=$2 gpu=$3 cores=$4 extra=$5
  local out="$OUT_ROOT/$tag/seed_$seed"
  if [[ -f "$out/COMPLETE.json" ]]; then echo "SKIP $tag seed=$seed (complete)"; return 0; fi
  mkdir -p "$out"
  echo "LAUNCH $tag seed=$seed gpu=$gpu cores=$cores extra=[$extra]"
  # shellcheck disable=SC2086  # $extra is an intentional flag list
  CUDA_VISIBLE_DEVICES="$gpu" \
  BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONPATH="$REF/src:$REF/experiments:$REF/experiments/scripts" \
  taskset -c "$cores" "$PY" \
    "$REF/experiments/scripts/run_matched_control_synthetic_validation.py" \
    --task heston --run-dir "$out" --device cuda:0 \
    --protocol-manifest "$HERE/PROTOCOL.json" \
    --seed "$seed" --bank-size 8192 --sample-batch-size 2048 \
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
  echo "{\"tag\": \"$tag\", \"seed\": $seed, \"steps\": $STEPS, \"extra\": \"$extra\"}" \
    > "$out/COMPLETE.json"
  echo "DONE $tag seed=$seed"
}

# Flatten (tag, seed) into one job list so both GPUs stay busy across tags
# rather than idling at each tag boundary.
JOBS=()
for tag in "$@"; do
  extra="$(extra_flags_for "$tag")"
  for seed in "${CONFIRM_SEEDS[@]}"; do
    JOBS+=("$tag|$seed|$extra")
  done
done

echo "${#JOBS[@]} confirm runs queued across $# tag(s)"

i=0
while (( i < ${#JOBS[@]} )); do
  IFS='|' read -r t s e <<< "${JOBS[$i]}"
  run_one "$t" "$s" 0 16-23 "$e" & A=$!
  if (( i + 1 < ${#JOBS[@]} )); then
    IFS='|' read -r t2 s2 e2 <<< "${JOBS[$((i+1))]}"
    run_one "$t2" "$s2" 1 24-31 "$e2" & B=$!
    wait $A $B
  else
    wait $A
  fi
  i=$((i + 2))
done

echo "ALL CONFIRM RUNS DONE, ranking by 4-seed median"
cd "$HERE" && "$PY" rank_confirm.py "$@"
