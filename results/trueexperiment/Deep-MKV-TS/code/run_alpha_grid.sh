#!/usr/bin/env bash
# Dense alpha grid: 75 points, all 5 seeds, 10 parallel workers.
#
# Theta(alpha) = eta*sigma_ref^-1 + alpha*Z/sqrt(dt).  The coarse and fine
# sweeps localised the admissible window to alpha in [1e-3, 3e-2] and showed
# vol_ratio crossing its target of 1.0 somewhere inside it.  This resolves the
# whole range at ~22 points per decade so the optimum, its width, and whether
# seeds 0 and 2 have ANY admissible point can all be read off directly instead
# of interpolated.
#
# Parallelisation
# ---------------
# The GPU work (one 6144x128x8 bank, ~1.2 s) is negligible; the cost is
# selection_true.score_candidate, which is numba/CPU.  So the split is by CPU:
# 10 workers x 4 cores on cores 16-55.  Cores 0-15 belong to the concurrent
# compute_all_multiasset.py metrics job and are left alone.
#
# GPUs: workers are spread round-robin over all four cards, sharing with
# jyoussef.  Each worker holds ~2 GB, so 2-3 workers per card sits far inside
# the 53-58 GB free on each, and total occupancy stays well under the 350%
# budget the user set.
#
# The alpha grid is split into two INTERLEAVED chunks (a[0::2], a[1::2]) rather
# than two contiguous halves, so both workers on a seed see the same mix of
# cheap and expensive alphas and finish together.
set -euo pipefail

B=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=$B/methods/Deep-MKV-TS/code/reference
C=$B/results/trueexperiment/Deep-MKV-TS/code
D=$B/results/trueexperiment/Deep-MKV-TS/diagnostic_bestfit

export PYTHONPATH="$R/src:$R/experiments:$C"
mkdir -p "$D/logs"
cd "$B"

# Built by the same expression documented in the header; kept literal so the
# grid that produced the JSONs is recoverable from this file alone.
G=$("$PY" - <<'PYEOF'
import numpy as np
a = np.unique(np.round(np.unique(np.concatenate([
    [0.0],
    np.logspace(np.log10(1e-4), np.log10(0.05), 60),
    np.logspace(np.log10(0.06), np.log10(1.0), 14),
])), 6))
print(' '.join(f'{x:g}' for x in a[0::2]))
print(' '.join(f'{x:g}' for x in a[1::2]))
PYEOF
)
CHUNK1=$(printf '%s\n' "$G" | sed -n 1p)
CHUNK2=$(printf '%s\n' "$G" | sed -n 2p)
echo "chunk1: $(printf '%s ' $CHUNK1 | wc -w) alphas"
echo "chunk2: $(printf '%s ' $CHUNK2 | wc -w) alphas"

PIDS=()
TAGS=()
LOGS=()
W=0
for SEED in 0 1 2 3 4; do
    for CH in 1 2; do
        if [ "$CH" -eq 1 ]; then A="$CHUNK1"; else A="$CHUNK2"; fi
        GPU=$(( W % 4 ))
        LO=$(( 16 + W * 4 )); HI=$(( LO + 3 ))
        LOG="$D/logs/alpha_grid_s${SEED}_g${CH}.log"
        CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
        taskset -c $LO-$HI "$PY" "$C/alpha_ablation.py" --seeds "$SEED" \
            --device cuda:0 --alphas $A --tag "g$CH" > "$LOG" 2>&1 &
        PIDS+=($!)
        TAGS+=("seed $SEED chunk $CH -> gpu $GPU cores $LO-$HI")
        LOGS+=("$LOG")
        W=$(( W + 1 ))
    done
done
printf 'launched %d workers:\n' "${#PIDS[@]}"
printf '   %s\n' "${TAGS[@]}"

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "WORKER FAILED: ${TAGS[$i]}"
        tail -25 "${LOGS[$i]}" || true
        FAIL=1
    fi
done
[ "$FAIL" -eq 0 ] || { echo "ALPHA GRID FAILED"; exit 1; }
echo "ALPHA GRID DONE"
