#!/usr/bin/env bash
# Retrain all 5 seeds at the dt-corrected learning rate.
#
# The hypothesis being tested
# ---------------------------
# The control is  Theta = eta*sigma_ref^-1 + Zhat/sqrt(dt).  `1/sqrt(dt)` is a
# gain on the network output: 15.87 at Heston's dt = 1/252, 1025.28 at this
# panel's dt = 9.51e-07.  train_true.py:121 declares its settings "byte-for-byte
# the committed Heston d = 8 run", and LR = 2e-3 is one of them.  AdamW moves
# each coordinate by ~lr per step regardless of gradient scale, so the Theta
# displacement per optimiser step is lr/sqrt(dt): 0.032 on Heston, 2.05 here.
#
# Measured consequence on the shipped banks: |Zhat/sqrt(dt)| / |eta*sigma_ref^-1|
# = 194, i.e. the reference SDE is erased and Theta is essentially pure
# noise-head output.  The checkpoint grid cannot escape it -- even step 500 is
# 72-477x too strong -- because selection searched checkpoint STEP while the
# broken axis was control MAGNITUDE.
#
# Two independent derivations agree on the correction:
#   (a) post-hoc rescale.  alpha_ablation.py scales Zhat by alpha exactly (the
#       drift head is identically zero and no denormalisation is configured), and
#       an 88-point grid x 5 seeds put the jointly-admissible window at
#       alpha in [0.00291, 0.004434], best joint point 0.003233.
#   (b) dimensionless matching.  Heston's learned term is ~1.7x its reference
#       term; reproducing that ratio here needs |Zhat| ~ 1.7*8/1025 = 0.0133
#       against a measured ~4.0, i.e. a factor ~0.0033.
# Hence lr = 2e-3 * 0.0032 = 6.4e-06, and Theta/step becomes 0.0066.
#
# Why this is not the same experiment as the alpha rescale
# --------------------------------------------------------
# Rescaling multiplies a CONVERGED solution; lowering lr changes the whole
# optimisation trajectory.  It can land somewhere better (the network gets to
# re-fit the residual at the right scale) or worse (3000 steps at 6.4e-06 may
# simply under-train).  The run is therefore checkpointed on the usual grid
# 500..3000 so selection still has a step axis to search, and it is scored on
# the full 34-metric suite rather than on vol/corr alone, because vol_err cannot
# see an ACF and the ACF is where the damage was.
#
# Nothing here touches published artefacts.  --out-root redirects weights/ and
# losses/; --run-root redirects checkpoints.  The published
# weights/seed_<S>_model.pt are asserted unchanged at the end.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
M=$B/results/trueexperiment/Deep-MKV-TS
O=$M/retune_lr
LR=6.4e-06
STEPS=3000

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$O/logs" "$O/weights" "$O/losses" "$O/runs"
cd "$B"

# Fingerprint the published weights BEFORE the run so "untouched" is a measured
# claim afterwards, not an assumption about --out-root.
BEFORE=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
echo "published weights digest before: $BEFORE"

echo "== retrain 5 seeds at lr=$LR (Heston constant 2e-3) =="
PIDS=(); LOGS=(); TAGS=()
for S in 0 1 2 3 4; do
    # Seeds 0-3 take one GPU each; seed 4 doubles up on GPU 0, which is the
    # idlest of the four. All four are shared with another user (~40 GB used of
    # 80 GB each), which is inside the "leave at least 50% free" budget.
    if [ "$S" -eq 4 ]; then GPU=0; else GPU=$S; fi
    LO=$(( 16 + S * 8 )); HI=$(( LO + 7 ))
    LOG="$O/logs/train_seed_$S.log"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
    taskset -c $LO-$HI "$PY" "$C/train_true.py" \
        --seed "$S" --steps "$STEPS" --device cuda:0 \
        --lr "$LR" --out-root "$O" --run-root "$O/runs" \
        > "$LOG" 2>&1 &
    PIDS+=($!); LOGS+=("$LOG"); TAGS+=("seed $S -> gpu $GPU cores $LO-$HI")
done
printf 'launched %d workers:\n' "${#PIDS[@]}"
printf '   %s\n' "${TAGS[@]}"

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "WORKER FAILED: ${TAGS[$i]}"; tail -30 "${LOGS[$i]}"; FAIL=1; }
done
[ "$FAIL" -eq 0 ] || { echo "RETUNE FAILED"; exit 1; }

AFTER=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] \
    || { echo "ABORT: published weights were modified ($BEFORE -> $AFTER)"; exit 1; }
echo "published weights digest after:  $AFTER  (unchanged)"

for S in 0 1 2 3 4; do
    [ -f "$O/weights/seed_${S}_model.pt" ] || { echo "missing weights seed $S"; exit 1; }
done

# The number the whole run is about: |Zhat| at the final layer of the noise
# head. Published run measured 2.31-6.96; the target is ~0.013.
echo "== learned control magnitude =="
"$PY" - <<PYEOF
import torch, json, pathlib
O = pathlib.Path("$O")
HEAD = "expected_adjoint_noise_next_head.2"
dt = 9.512937595129376e-07
print(f"{'seed':>5s} {'|Z|':>10s} {'|Z|/sqrt(dt)':>14s}   (published |Z| was 2.31-6.96)")
out = {}
for s in range(5):
    sd = torch.load(O / f"weights/seed_{s}_model.pt", map_location="cpu",
                    weights_only=True)
    w = sd[f"{HEAD}.weight"].double()
    b = sd[f"{HEAD}.bias"].double()
    n = float((w.pow(2).sum() + b.pow(2).sum()).sqrt())
    out[s] = n
    print(f"{s:5d} {n:10.4f} {n/dt**0.5:14.2f}")
(O / "zhat_norms.json").write_text(json.dumps(out, indent=2) + "\n")
PYEOF

echo "RETUNE DONE -- weights in $O/weights"
