#!/usr/bin/env bash
# Chain: wait for stage-1 (hpo_sine.py, PID passed as $1) to exit,
# run marginal analysis, then launch the stage-2 refined grid on GPU0.
set -u
cd /home/tbasseras/benchmark/methods/TimeDiT/paper_reimplementation
STAGE1_PID="$1"
echo "[chain] waiting on stage-1 PID $STAGE1_PID $(date -Is)"
while kill -0 "$STAGE1_PID" 2>/dev/null; do sleep 60; done
echo "[chain] stage-1 exited $(date -Is), running analyze_hpo.py"
/home/tbasseras/gpu-venv/bin/python analyze_hpo.py > analyze_stage1.txt 2>&1
echo "STAGE1_ANALYZED" >> hpo_status.txt
echo "[chain] launching stage-2 grid $(date -Is)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  /home/tbasseras/gpu-venv/bin/python hpo_stage2.py --steps 3000 > log_hpo_stage2.txt 2>&1
echo "CHAIN_DONE" >> hpo_status.txt
echo "[chain] done $(date -Is)"
