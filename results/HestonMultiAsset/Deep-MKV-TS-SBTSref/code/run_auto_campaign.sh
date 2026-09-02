#!/usr/bin/env bash
# Unattended driver: wait out the h re-sweep, promote its winner, run the
# 5-seed final campaign.  Ticks every 5 minutes for up to 7 hours.
#
# WHY A SUPERVISOR AND NOT A CHAIN.  The previous attempt chained the campaign
# behind the sweep by having the interactive session watch for a marker and then
# launch.  That needs somebody alive at the handoff, and nobody was: the sweep
# finished, no launch happened, and an hour of three idle A100s was lost.  This
# script owns the handoff itself, is detached with setsid, and holds no
# reference to the session that started it.
#
# WHY IT IS IDEMPOTENT.  Each phase writes a marker into $STATE before doing
# anything irreversible, and refuses to redo a phase whose marker exists.  A
# supervisor that double-launched the campaign would put two processes on the
# same seed, and the second would either fight for the seed lock or silently
# overwrite the first one's checkpoints.
#
# WHY IT POLLS INSTEAD OF USING `wait`.  It is not the parent of the sweep
# processes -- they were detached by a different script -- so `wait` cannot see
# them.  The ALL_DONE marker in _progress.txt is the only reliable completion
# signal, and it is written by the sweep's own round barrier.
set -u

R=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
LOSSES=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses
SWEEPLOG="$LOSSES/resweeplogs"
CAMPLOG="$LOSSES/campaignlogs_auto"
STATE="$LOSSES/_supervisor"
PY=/home/tbasseras/gpu-venv/bin/python

TICK=300          # 5 minutes, as asked
MAX_TICKS=84      # 84 x 5 min = 7 hours, as asked

mkdir -p "$CAMPLOG" "$STATE"
LOG="$STATE/supervisor.log"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "supervisor up: pid=$$ tick=${TICK}s budget=${MAX_TICKS} ticks (7h)"

# --------------------------------------------------------------------------
# phase A -- wait for the 18-arm re-sweep to finish
# --------------------------------------------------------------------------
tick=0
while [ "$tick" -lt "$MAX_TICKS" ]; do
  if [ -f "$STATE/sweep_done" ]; then break; fi
  if grep -q ALL_DONE "$SWEEPLOG/_progress.txt" 2>/dev/null; then
    say "phase A: ALL_DONE seen after $tick ticks"
    touch "$STATE/sweep_done"
    break
  fi
  n_arms=$(ls "$R/sweep"/hfix__*.json 2>/dev/null | wc -l)
  n_live=$(pgrep -c -f 'sweep_hyperparams.py --stage hfix' 2>/dev/null || echo 0)
  say "phase A tick $tick: arms=$n_arms/18 live=$n_live"
  # Every arm has written AND nothing is still running: the sweep is over even
  # if the marker never landed (e.g. the round barrier was killed).  Trusting
  # only the marker would hang here for the full 7 hours.
  if [ "$n_arms" -ge 18 ] && [ "$n_live" -eq 0 ]; then
    say "phase A: 18 arms present and no live sweep process; proceeding"
    touch "$STATE/sweep_done"
    break
  fi
  tick=$((tick + 1))
  sleep "$TICK"
done

if [ ! -f "$STATE/sweep_done" ]; then
  say "ABORT: 7h budget exhausted, sweep never completed. No campaign launched."
  exit 1
fi

# --------------------------------------------------------------------------
# phase B -- tabulate, pick the winner, promote it
# --------------------------------------------------------------------------
if [ -f "$STATE/winner_h" ]; then
  BEST_H=$(cat "$STATE/winner_h")
  say "phase B: winner already chosen, h=$BEST_H"
else
  cd "$R" || exit 1
  "$PY" "$R/tabulate_hfix.py" --promote > "$STATE/table_hfix.txt" 2>&1
  rc=$?
  cat "$STATE/table_hfix.txt" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    say "ABORT: tabulate_hfix.py exited $rc; refusing to launch on an unsound winner."
    exit 1
  fi
  BEST_H=$(grep -oP '^WINNER_H=\K.*' "$STATE/table_hfix.txt" | tail -1)
  if [ -z "$BEST_H" ]; then
    say "ABORT: no WINNER_H line in the table; refusing to guess a bandwidth."
    exit 1
  fi
  echo "$BEST_H" > "$STATE/winner_h"
  say "phase B: promoted h=$BEST_H"
fi

# --------------------------------------------------------------------------
# phase C -- launch the 5-seed campaign at the promoted bandwidth
# --------------------------------------------------------------------------
# Placement: <= 3 GPUs, <= 64 cores.  Two seeds share GPU 0 and GPU 1 because a
# single arm used 19 GiB of an 80 GiB card and left the SMs latency-bound; a
# second round with two idle cards would be strictly slower.
if [ -f "$STATE/campaign_launched" ]; then
  say "phase C: campaign already launched, skipping"
else
  cd "$R" || exit 1
  for spec in 0:0:0-11 1:0:12-23 2:1:24-35 3:1:36-47 4:2:48-59; do
    s="${spec%%:*}"; rest="${spec#*:}"; g="${rest%%:*}"; c="${rest#*:}"
    CUDA_VISIBLE_DEVICES="$g" \
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
    setsid nohup taskset -c "$c" "$PY" "$R/train_multiasset.py" \
        --seed "$s" \
        --steps 3000 \
        --lr 2.5e-05 \
        --ridge-lambda 10 \
        --h "$BEST_H" \
        --markov-order 20 \
        --npi 1 \
        --weight-grad-mode analytic \
        --jacobian-lags -1 \
        --device cuda:0 \
        > "$CAMPLOG/seed_${s}.log" 2>&1 < /dev/null &
    disown || true
    say "phase C: launched seed=$s gpu=$g cores=$c h=$BEST_H"
  done
  touch "$STATE/campaign_launched"
fi

# --------------------------------------------------------------------------
# phase D -- watch the campaign to completion
# --------------------------------------------------------------------------
while [ "$tick" -lt "$MAX_TICKS" ]; do
  n_live=$(pgrep -c -f 'train_multiasset.py --seed' 2>/dev/null || echo 0)
  prog=$(grep -h '^step=' "$CAMPLOG"/seed_*.log 2>/dev/null | tail -1)
  say "phase D tick $tick: live=$n_live  last=[${prog:-none}]"
  if [ "$n_live" -eq 0 ]; then
    # Give the last process a tick to flush its final write before judging.
    sleep 20
    done_n=$(grep -l 'step= 3000' "$CAMPLOG"/seed_*.log 2>/dev/null | wc -l)
    say "phase D: no live trainers; $done_n/5 seeds reached step 3000"
    touch "$STATE/campaign_done"
    break
  fi
  tick=$((tick + 1))
  sleep "$TICK"
done

if [ -f "$STATE/campaign_done" ]; then
  say "ALL PHASES COMPLETE"
  echo ALL_DONE > "$STATE/DONE"
else
  say "7h budget exhausted with the campaign still running; it is NOT killed."
  echo BUDGET_EXHAUSTED > "$STATE/DONE"
fi
