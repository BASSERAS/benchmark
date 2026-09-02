#!/usr/bin/env bash
# The 5-seed d = 8 campaign.
#
# Three seeds at a time, one process per GPU, 5 physical cores each, two waves:
#
#   wave 1   seed 0 -> GPU 1     seed 1 -> GPU 2     seed 2 -> GPU 3
#   wave 2   seed 3 -> GPU 1     seed 4 -> GPU 2
#
# GPU 0 IS NEVER USED -- that is an explicit standing instruction, not a default.
# Three processes at 5 cores each is 15 cores, inside the 16-core cap; that is why
# it is 5 and not 8.  `GPUS` is an operating parameter for a shared machine, not a
# degradation path: set GPUS="1" for five sequential waves on one card.  The
# mathematics is identical either way -- only the schedule changes.
#
#   setsid bash run_campaign.sh > logs/campaign.log 2>&1 & disown
#
# Before launching anything this script GATES on the ridge_lambda selection:
# `sweep/winner.json` must exist and its winner must equal the RIDGE_LAMBDA
# compiled into train_multiasset.py.  A campaign started against the placeholder
# 1e-3 would burn ~16 h producing paths from a hyperparameter the sweep rejected,
# and nothing downstream would notice -- the losses would look healthy and every
# metric would be quietly wrong.  See MULTIASSET_GUIDELINE.md section 12.6.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PY="${PY:-/home/tbasseras/gpu-venv/bin/python}"
STEPS="${STEPS:-3000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GPUS="${GPUS:-1 2 3}"
CORES_PER_RUN="${CORES_PER_RUN:-5}"

# Refuse GPU 0 outright rather than trusting the default. A typo in an override
# must not be able to put work on it.
for _g in $GPUS; do
  if [ "$_g" = "0" ]; then
    echo "[gate] REFUSING: GPU 0 is off limits by standing instruction." >&2
    exit 1
  fi
done
LOGS="$HERE/logs"
mkdir -p "$LOGS" "$HERE/weights" "$HERE/losses"

export PYTHONPATH="$REF/src:$REF/experiments"
export OMP_NUM_THREADS="$CORES_PER_RUN"
export MKL_NUM_THREADS="$CORES_PER_RUN"
export OPENBLAS_NUM_THREADS="$CORES_PER_RUN"

# ---------------------------------------------------------------- gate -------
# Compare the selected lambda against the one the trainer will actually use.
# Both numbers are read, never typed: winner.json from the sweep, RIDGE_LAMBDA
# by importing the trainer module itself.
gate() {
  "$PY" - "$HERE" <<'PYEOF'
import json, os, sys
here = sys.argv[1]
winner_path = os.path.join(here, "sweep", "winner.json")
if not os.path.exists(winner_path):
    sys.exit(f"[gate] {winner_path} does not exist -- the ridge_lambda sweep has "
             "not been reported yet.  Run `sweep_ridge_lambda.py --report` first.")
with open(winner_path) as fh:
    winner = json.load(fh)
selected = float(winner["ridge_lambda"])

# A winner.json written mid-sweep is indistinguishable from the final one unless
# its coverage is checked. Require every candidate the sweep was designed around.
EXPECTED = [1e-3, 1.0, 10.0, 100.0, 1000.0]
compared = winner.get("candidates_compared")
if compared is None:
    sys.exit("[gate] winner.json has no `candidates_compared` field -- it was written "
             "by a pre-coverage version of sweep_ridge_lambda.py. Re-run "
             "`sweep_ridge_lambda.py --report` to regenerate it.")
missing = [x for x in EXPECTED if not any(abs(x - c) <= 1e-12 * max(1.0, x) for c in compared)]
if missing:
    sys.exit(f"[gate] REFUSING TO LAUNCH. The sweep compared {compared} but candidates "
             f"{missing} are missing -- this winner came from a PARTIAL sweep. Wait for "
             "run_sweep.sh to finish, then re-run `sweep_ridge_lambda.py --report`.")

sys.path.insert(0, here)
import train_multiasset as tm
compiled = float(tm.RIDGE_LAMBDA)

if abs(selected - compiled) > 1e-12 * max(1.0, abs(selected)):
    sys.exit(f"[gate] REFUSING TO LAUNCH.  The sweep selected ridge_lambda="
             f"{selected!r} (from {winner_path}) but train_multiasset.RIDGE_LAMBDA "
             f"is {compiled!r}.  Set RIDGE_LAMBDA to the selected value and update "
             "the selection paragraph in code/README.md before running the campaign.")

print(f"[gate] ok: ridge_lambda={compiled!r} matches the sweep winner")
print(f"[gate] winner record: val_discrepancy={winner.get('val_discrepancy')!r} "
      f"ce_projection_r2={winner.get('ce_projection_r2')!r}")
PYEOF
}

gate || exit 1

# ---------------------------------------------------------------- waves ------
run_wave() {
  local pids=()
  local labels=()
  local slot=0
  local gpu_list=($GPUS)
  for seed in "$@"; do
    local gpu="${gpu_list[$slot]}"
    local lo=$((slot * CORES_PER_RUN))
    local hi=$((lo + CORES_PER_RUN - 1))
    echo "[launch] seed=$seed gpu=$gpu cores=${lo}-${hi}  $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" taskset -c "${lo}-${hi}" \
      "$PY" "$HERE/train_multiasset.py" \
      --seed "$seed" --steps "$STEPS" --device cuda:0 \
      > "$LOGS/train_seed_${seed}.log" 2>&1 &
    pids+=("$!")
    labels+=("seed=$seed")
    slot=$((slot + 1))
  done
  local rc=0
  local i=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      echo "[done]  ${labels[$i]}  $(date -Is)"
    else
      echo "[FAILED] ${labels[$i]} exited non-zero -- see logs  $(date -Is)"
      rc=1
    fi
    i=$((i + 1))
  done
  return "$rc"
}

# Chop the seed list into waves of size = number of GPUs.
n_gpus=$(wc -w <<<"$GPUS")
wave=()
wave_no=1
overall_rc=0
for seed in $SEEDS; do
  wave+=("$seed")
  if [ "${#wave[@]}" -eq "$n_gpus" ]; then
    echo "=== wave $wave_no: ${wave[*]}  $(date -Is) ==="
    run_wave "${wave[@]}" || overall_rc=1
    wave=()
    wave_no=$((wave_no + 1))
  fi
done
if [ "${#wave[@]}" -gt 0 ]; then
  echo "=== wave $wave_no: ${wave[*]}  $(date -Is) ==="
  run_wave "${wave[@]}" || overall_rc=1
fi

echo "=== campaign finished rc=$overall_rc  $(date -Is) ==="
# Sentinel: downstream steps key off this file, so a partial campaign cannot be
# mistaken for a complete one.
if [ "$overall_rc" -eq 0 ]; then
  date -Is > "$HERE/weights/.campaign_complete"
  echo "[sentinel] wrote weights/.campaign_complete"
else
  echo "[sentinel] NOT written: at least one seed failed"
fi
exit "$overall_rc"
