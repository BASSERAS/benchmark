#!/usr/bin/env bash
# Chain the post-training pipeline behind the 5-seed campaign, unattended.
#
# WHY THIS EXISTS AS A SEPARATE SCRIPT.  run_pipeline.sh already does every
# post-training stage (select -> generate -> gate -> metrics -> memorisation ->
# plots -> READMEs), and its step 1 waits for the trainers to exit.  But that
# wait is `while pgrep train_multiasset.py`, which is only a wait if a trainer
# is ALREADY RUNNING.  Launched now -- with the h re-sweep still going and the
# campaign not yet started -- it would find no trainer, fall straight through
# to the preflight, see no step_3000.pt, and abort within a second.  So the
# pipeline cannot simply be launched early; something has to hold it until the
# campaign is actually on the GPUs.  That is this script's whole job.
#
# WHY NOT PUT IT INSIDE run_auto_campaign.sh.  That supervisor is running right
# now (pid 3872760).  bash reads a script lazily, by byte offset, so editing a
# live script corrupts the still-unread remainder.  Appending a phase E would
# mean killing and relaunching the supervisor mid-sweep.  A second detached
# process costs nothing and touches nothing that is already in flight.
#
# WHY IT DOES NOT PUSH.  PUSH stays 0.  run_pipeline.sh's own header says
# publishing is a decision and the script "should not make that decision on its
# own" while running unattended for hours.  The results will be sitting there,
# rendered, for review.
#
# Idempotent: refuses to start a second time once post_launched exists, because
# two concurrent pipelines would race on the same generated_paths/ and metrics
# files and the loser's partial writes would be indistinguishable from real
# output.
set -u

R=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
METHOD_DIR=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref
STATE="$METHOD_DIR/losses/_supervisor"
LOG="$STATE/post_campaign.log"

mkdir -p "$STATE" "$METHOD_DIR/weights"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$STATE/post_launched" ]; then
  say "post-pipeline already launched once; refusing to run a second copy."
  exit 0
fi

say "post-campaign chainer up: pid=$$"

# ---------------------------------------------------------------------------
# wait for the campaign to be ON the GPUs, not merely scheduled
# ---------------------------------------------------------------------------
# Two conditions, both required.  The marker alone is not enough: the supervisor
# touches campaign_launched immediately after firing the five setsid commands,
# and a trainer that dies in its first second (bad flag, OOM) would leave the
# marker set with nothing running -- the pipeline would then abort on preflight
# and the night would be wasted.  Waiting for pgrep to actually see a trainer
# means the handoff happens only once training is genuinely alive.
tick=0
MAX=96          # 96 x 5 min = 8 h, past the supervisor's own 7 h budget
while [ "$tick" -lt "$MAX" ]; do
  if [ -f "$STATE/campaign_launched" ] && pgrep -f "train_multiasse[t].py" >/dev/null; then
    say "campaign is live after $tick ticks; handing over to run_pipeline.sh"
    break
  fi
  if [ -f "$STATE/campaign_launched" ]; then
    say "tick $tick: marker present but no trainer visible yet"
  else
    say "tick $tick: waiting for campaign_launched"
  fi
  tick=$((tick + 1))
  sleep 300
done

if [ ! -f "$STATE/campaign_launched" ]; then
  say "ABORT: 8 h elapsed and the campaign never launched. Nothing run."
  exit 1
fi
if ! pgrep -f "train_multiasse[t].py" >/dev/null; then
  say "ABORT: campaign_launched is set but no trainer is running. Refusing to"
  say "       start the pipeline -- preflight would abort and the real problem"
  say "       is that the trainers died. Read $METHOD_DIR/losses/campaignlogs_auto/."
  exit 1
fi

touch "$STATE/post_launched"

# ---------------------------------------------------------------------------
# hand over
# ---------------------------------------------------------------------------
# GPU 0 / cores 0-7 are run_pipeline.sh's defaults and are free by the time its
# own step-1 wait returns, because that wait ends exactly when the last trainer
# releases them.
say "exec run_pipeline.sh (PUSH=0)"
cd "$R" || exit 1
PUSH=0 GPU=0 CORES=0-7 bash "$R/run_pipeline.sh" >> "$STATE/pipeline.log" 2>&1
rc=$?
say "run_pipeline.sh exited rc=$rc"
if [ "$rc" -eq 0 ]; then
  echo PIPELINE_OK > "$STATE/PIPELINE_DONE"
  say "POST-CAMPAIGN PIPELINE COMPLETE"
else
  echo "PIPELINE_FAILED rc=$rc" > "$STATE/PIPELINE_DONE"
  say "pipeline FAILED; see $STATE/pipeline.log"
fi
