#!/usr/bin/env bash
# Final Deep-MKV-TS-SBTSref campaign: 5 seeds, exact (full-K-lag) adjoint.
#
# One process per seed, all launched at once, each detached with setsid so the
# harness cannot reap them.  There is NO chaining: a chain needs somebody to be
# alive at the handoff, and that is exactly what failed last time.
#
# Placement (<= 3 GPUs, <= 64 cores, per the standing hardware rule):
#   GPU 0 : seeds 0, 1   cores  0-11, 12-23
#   GPU 1 : seeds 2, 3   cores 24-35, 36-47
#   GPU 2 : seed  4      cores 48-59
# Two jobs share a card because one job used 19 GiB of 80 GiB and left the SMs
# latency-bound, not throughput-bound; co-locating keeps every card busy instead
# of idling two of them during a second round.
set -euo pipefail

R=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
LOG=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/campaignlogs_final
PY=/home/tbasseras/gpu-venv/bin/python

mkdir -p "$LOG"
cd "$R"

# seed:gpu:cores
for spec in 0:0:0-11 1:0:12-23 2:1:24-35 3:1:36-47 4:2:48-59; do
  s="${spec%%:*}"; rest="${spec#*:}"; g="${rest%%:*}"; c="${rest#*:}"

  CUDA_VISIBLE_DEVICES="$g" \
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
  setsid nohup taskset -c "$c" "$PY" "$R/train_multiasset.py" \
      --seed "$s" \
      --steps 3000 \
      --lr 2.5e-05 \
      --ridge-lambda 10 \
      --h 0.36 \
      --markov-order 20 \
      --npi 1 \
      --weight-grad-mode analytic \
      --jacobian-lags -1 \
      --device cuda:0 \
      > "$LOG/seed_${s}.log" 2>&1 < /dev/null &
  disown || true
  echo "launched seed=$s gpu=$g cores=$c"
done

echo "all 5 launched; logs in $LOG"
