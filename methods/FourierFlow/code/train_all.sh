#!/usr/bin/env bash
# Fourier Flow — train + generate all 5 Heston seeds in parallel.
#
# Fourier Flow is CPU-bound: the DFT/iDFT in fourier/transforms.py uses
# numpy.fft, and the coupling MLPs are tiny (hidden=200). No GPU is used.
# We therefore parallelise across CPU cores with taskset + OMP_NUM_THREADS,
# honouring the machine's 16-physical-core limit (5 seeds x 3 cores = 15).
#
# Each seed trains one FourierFlow on the 8192x128 Heston price paths, then
# samples 8192 synthetic paths back on the original price scale. Per-seed
# artifacts land in ../weights, ../losses, ../generated_paths (see train_heston.py).
#
# Usage:
#   cd methods/FourierFlow/code/reference     # so SequentialFlows is importable
#   bash ../train_all.sh
#
# Override epochs / hyperparameters by exporting EPOCHS, HIDDEN, NFLOWS, LR
# before calling (all optional; defaults = paper Stocks config).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # methods/FourierFlow/code
REF="$HERE/reference"
LOGD="$HERE/../losses"
PY="${PY:-/home/tbasseras/gpu-venv/bin/python}"
mkdir -p "$LOGD"

EPOCHS="${EPOCHS:-1000}"
HIDDEN="${HIDDEN:-200}"
NFLOWS="${NFLOWS:-3}"
LR="${LR:-1e-3}"
NGEN="${NGEN:-8192}"
# normalize=1 (paper regime). The even seq_len (128) leaves one exactly-zero-variance
# imaginary-DC spectral bin even after the +0 anchor; FourierFlowClamp clamps that
# single bin's std 0->1 so normalize=True is stable (Fix A). See code/README.md.
NORMALIZE="${NORMALIZE:-1}"

echo "[train_all] epochs=$EPOCHS hidden=$HIDDEN n_flows=$NFLOWS lr=$LR normalize=$NORMALIZE"
for s in 0 1 2 3 4; do
  c0=$((s*3)); c1=$((s*3+2))
  taskset -c ${c0}-${c1} env OMP_NUM_THREADS=3 PYTHONPATH="$REF" \
    "$PY" "$HERE/train_heston.py" \
      --seed "$s" --normalize "$NORMALIZE" --epochs "$EPOCHS" \
      --hidden "$HIDDEN" --n-flows "$NFLOWS" --lr "$LR" \
      --n-gen "$NGEN" --display-step 200 \
      > "$LOGD/train_seed_${s}.log" 2>&1 &
  echo "[train_all] launched seed $s on cores ${c0}-${c1} (pid $!)"
done
wait
echo "[train_all] all 5 seeds finished."
