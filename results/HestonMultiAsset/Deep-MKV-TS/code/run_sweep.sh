#!/usr/bin/env bash
# Re-select ridge_lambda on the VALIDATION discrepancy bank (d = 8).
#
# Five candidates, seed 0, SHORT runs, scored against
# heston_ma_S_valdisc_8192x252x8.npy.  The test split is never opened.
#
# Everything runs on GPU 1 only: GPU 0 is the user's, GPUs 2 and 3 are the
# parallel session.  Two waves so peak memory stays well under the card.
#
#   wave A  lambda = 1e-3, 1, 10     (1e-3 is the slow one: no Cholesky path)
#   wave B  lambda = 100, 1000
#
#   setsid bash run_sweep.sh > logs/sweep.log 2>&1 & disown

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY=/home/tbasseras/gpu-venv/bin/python
STEPS="${STEPS:-250}"
LOGS="$HERE/logs"
mkdir -p "$LOGS" "$HERE/sweep"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$REF/src:$REF/experiments"
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5

run_wave() {
  local core=0
  local pids=()
  for lam in "$@"; do
    local tag="${lam//./p}"
    echo "[launch] lambda=$lam cores=${core}-$((core + 4))  $(date -Is)"
    taskset -c "${core}-$((core + 4))" "$PY" "$HERE/sweep_ridge_lambda.py" \
      --ridge-lambda "$lam" --steps "$STEPS" --device cuda:0 \
      > "$LOGS/sweep_lambda_${tag}.log" 2>&1 &
    pids+=("$!")
    core=$((core + 5))
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  return "$rc"
}

echo "=== wave A  $(date -Is) ==="
run_wave 1e-3 1 10 || echo "[warn] a candidate in wave A failed; see logs"

echo "=== wave B  $(date -Is) ==="
run_wave 100 1000 || echo "[warn] a candidate in wave B failed; see logs"

echo "=== report  $(date -Is) ==="
taskset -c 0-4 "$PY" "$HERE/sweep_ridge_lambda.py" --report

echo "=== sweep done  $(date -Is) ==="
