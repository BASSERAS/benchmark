#!/bin/bash
# SBBTS end-to-end benchmark pipeline: 5 seeds -> metrics -> figures -> PS-MC.
#
# beta = 100: the authors' run_heston.py default, unmodified. Every hyperparameter
# below is theirs; nothing here is tuned.
#
# We did look. The sweep in ../paper_reimplementation/ ran 27 arms, with 4 seed
# replicates on the two arms that looked most promising, and found nothing:
#
#   leverage-spread ratio   beta=100  0.794 +- 0.015 (n=4)
#                           beta=300  0.873 +- 0.038 (n=4)    Welch t = -3.90, p = 0.018
#
# CORRECTED 2026-09-01. The figures above used to read 0.816 +- 0.058 / 0.862 +-
# 0.048, t = 1.22, p = 0.27, and this comment used to conclude "beta is a plateau,
# not a lever". That came from a stale corr_spread_cache.json -- sweep_paper.py
# keys the cache on the trial tag and assumes generated arrays are immutable, but
# a re-run of the same tag overwrites the array. See README "Correction (2026-09-01)"
# and rebuild_corr_cache.py.
#
# There IS a dose response: beta = 150/100/200/300/500/1000 gives 0.767 / 0.785 /
# 0.870 / 0.905 / 0.913 / 0.884, a 0.147 span against a 0.034 seed spread at fixed
# beta = 100.
#
# We ship beta = 100 regardless: it is the authors' run_heston.py default; beta does
# not significantly move xi (p = 0.148) or rho (p = 0.565), which is what the paper's
# claim is about; and with n = 4 and six statistics tested nothing survives Bonferroni.
# Flagged as an open item for the method authors rather than silently re-tuned.
#
# Stage 0 exists because the SBTS bandwidth sweep is still holding cores 0-10; the
# pipeline waits rather than oversubscribing. GUIDELINE 4.1 caps us at 2 GPUs and
# 16 cores and the machine is shared.
#
# Launch detached -- the chain is multi-hour:
#   setsid bash methods/SBBTS/run_benchmark.sh > methods/SBBTS/losses/pipeline.log 2>&1 < /dev/null & disown

set -u

BENCH=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
METHOD=SBBTS
BETA=100
LOGD=$BENCH/methods/$METHOD/losses
mkdir -p "$LOGD"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# -- Stage 0: drain the paper-reimplementation compute first --
# Two things are still live: the SBTS bandwidth sweep (sbts_baseline.py, ~9 CPU
# cores) and the paper sweep arms (reproduce_heston.py, both A100s). Starting on
# top of them would blow the 16-core cap and halve everyone's throughput, so the
# pipeline blocks instead. Kill the redundant wave-4 arms to start sooner.
say "stage 0: waiting for paper-reimplementation jobs to drain"
while pgrep -f "sbts_baseline.py|reproduce_heston.py" > /dev/null 2>&1; do
    sleep 30
done
say "stage 0: cores and GPUs free"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# -- Stages 1-4 live in run_pipeline.sh --
#
# Everything above this line is byte-frozen. An instance of this script (PID
# 3006242) was already parked in the `while pgrep` loop above when the file was
# edited, and bash resumes a script from a recorded byte offset -- so changing
# any earlier byte would have made that process execute garbage on wake. Only
# the tail was rewritten, which is safe because the tail has not been read yet.
#
# Why the split: the gate above waits for EVERY paper-reimplementation process,
# including two long-patience arms (t02 patience=150, t15 patience=100) that had
# already run 4h and hold ~1 core each. run_pipeline.sh replaces that with a
# counting gate that tolerates a small number of stragglers, and takes a lock so
# that whichever instance arrives second exits instead of clobbering stage 1.
exec bash "$BENCH/methods/$METHOD/run_pipeline.sh"
