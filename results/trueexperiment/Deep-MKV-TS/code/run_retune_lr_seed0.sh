#!/usr/bin/env bash
# Seed 0 of the lr=6.4e-06 retune, deferred until GPU 0 is free.
#
# Why this file exists
# --------------------
# run_retune_lr.sh put seeds 0 and 4 on GPU 0 on the assumption that two
# trainings fit alongside jyoussef's 38.7 GB. They do not: the eigvalsh in
# specific_entropy_matrix_cost asks for a single 17.41 GiB allocation, and with
# 38.68 (jyoussef) + 18.28 (seed 4) already resident there were only 4.02 GiB
# left. Seed 0 died with CUDA OOM at step 0; seeds 1-4 (one per GPU) are fine.
#
# So the correct capacity is ONE training per GPU while the machine is shared.
# Rather than kill and restart the four healthy runs, this script waits for
# seed 4 to release GPU 0 and then runs seed 0 there with identical arguments.
#
# expandable_segments:True is added because the OOM message named fragmentation
# explicitly (17.43 GiB reserved-but-unallocated against 314 MiB actually in
# use), and seed 0 will be running against whatever jyoussef holds by then.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
M=$B/results/trueexperiment/Deep-MKV-TS
O=$M/retune_lr
LR=6.4e-06
STEPS=3000
# Optional. With a PID, wait for that process to release GPU 0 before starting.
# Without one, start immediately -- which is correct when the other tenant's
# footprint has shrunk enough that seed 0's 36.5 GB fits in what is left.
# Operator instruction, verbatim: "even if jyoussef use all gpu u can complete
# where there is space left".
WAIT_PID=${1:-}

export PYTHONPATH="$R/src:$R/experiments:$C"
cd "$B"

if [ -n "$WAIT_PID" ]; then
    echo "waiting for PID $WAIT_PID (seed 4 on GPU 0) to exit..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "GPU 0 released at $(date -Is)"
else
    echo "starting immediately in the free space on GPU 0 at $(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
fi

# The published weights must be untouched by this run too, not just by the
# parent script whose own check never fired (it exits 1 on the seed-0 failure).
BEFORE=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)

LOG="$O/logs/train_seed_0.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
taskset -c 16-23 "$PY" "$C/train_true.py" \
    --seed 0 --steps "$STEPS" --device cuda:0 \
    --lr "$LR" --out-root "$O" --run-root "$O/runs" \
    > "$LOG" 2>&1 \
    || { echo "SEED 0 FAILED"; tail -30 "$LOG"; exit 1; }

AFTER=$(md5sum "$M"/weights/seed_*_model.pt | md5sum | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] \
    || { echo "ABORT: published weights were modified ($BEFORE -> $AFTER)"; exit 1; }

for S in 0 1 2 3 4; do
    [ -f "$O/weights/seed_${S}_model.pt" ] || { echo "missing weights seed $S"; exit 1; }
done

# Same measurement the parent script never reached: |Zhat| at the noise head.
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

echo "RETUNE DONE (all 5 seeds) -- weights in $O/weights"
