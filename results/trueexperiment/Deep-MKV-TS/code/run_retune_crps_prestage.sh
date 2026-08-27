#!/usr/bin/env bash
# Pre-build the table-C inputs for the lr=6.4e-06 retune WHILE the progressive
# scorer is still running, so the post-pipeline does not have to.
#
# Operator instruction, verbatim:
#   "I will come back in 1 hour so pls queue everything st when I come back all
#    the readme and the display is already commited and pushed pls"
#
# Why this exists rather than just waiting
# ----------------------------------------
# run_pipeline_post.sh's long pole after the metrics are in hand is STAGE 7: the
# 2*(5+1) = 12 conditional-CRPS jobs, ~21 min wall on the SBTS precedent. Those
# jobs need only (a) the 8192-path pools and (b) the real train bank. Both are
# derivable NOW from retune_lr/weights/, which has all five selected checkpoints
# on disk. Running them early moves ~25 min off the critical path.
#
# Every artefact this writes is EXACTLY the file run_pipeline_post.sh's own
# guards test for -- STAGE 3 skips on
#   crps_banks/generated_paths/seed_<S>/generated_paths_8192x128x8.npy
# (run_pipeline_post.sh:261) and STAGE 7 skips on
#   losses/crps_configs/<cfg>__seed_<S>.json
# (run_pipeline_post.sh:355) -- so if any part of this fails, the pipeline simply
# redoes it. There is no state here the pipeline cannot rebuild, and nothing
# already on disk is overwritten.
#
# Why it cannot write into $M yet
# -------------------------------
# run_retune_lr_progressive.sh md5-checks Deep-MKV-TS/weights/seed_*_model.pt at
# its very end and aborts if they moved, and run_pipeline_post.sh's own STAGE 3
# would read those (lr=0.002) weights rather than the retune. So this writes ONLY
# inside retune_lr/ and reads its weights via --weights-root "$O". Promotion into
# the method root happens in the driver, after PID 1304816 exits.
#
# Placement: GPUs 0 and 1 are idle (nvidia-smi at 16:52 -- 19 MiB, 0% each);
# 2 and 3 are another user's at ~60%. Cores 64-99 are used because the
# progressive scorer's metrics stage holds 24-55.
set -u

cd /home/tbasseras/benchmark || exit 1

R=/home/tbasseras/benchmark/results/trueexperiment
M=$R/Deep-MKV-TS
C=$M/code
O=$M/retune_lr
V=/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
CRPS_TAG=8192x128x8
PY=/home/tbasseras/gpu-venv/bin/python
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$REF/src:$REF/experiments:$C"

# Load-bearing: render_readme.py and render_comparison.py key table C on this
# exact string. A mismatch renders table C with baselines only AND NO ERROR.
LABEL=Deep-MKV-TS
SEEDS="0 1 2 3 4"

mkdir -p "$O/logs" "$O/losses/crps_configs" "$O/crps_banks"
say() { echo "[$(date +%H:%M:%S)] $*"; }

# --------------------------------------------------------------- pools ------
# The gate is the SELECTION file, not the weights file.
#
# First attempt at 16:54 died on seed 0 with
#   "ABORT: .../retune_lr/weights/seed_0_model.pt is not a checkpoint_state()
#    payload; select_checkpoint_true.py must run before generation"
# because train_true.py and select_checkpoint_true.py write the SAME path with
# different payloads: the training artefact is 229268 bytes, the selected
# checkpoint 230745. Testing `-f weights/seed_S_model.pt` therefore cannot tell
# a selected seed from an unselected one. selection/seed_<S>_selection.json is
# written only by the selector, so it is the honest signal.
#
# Opportunistic rather than all-or-nothing: run_retune_lr_progressive.sh selects
# seeds one at a time in completion order, so waiting for all five before
# starting any pool would idle both free GPUs for ~20 min. Each seed's pool is
# launched the moment its selection lands.
say "PRESTAGE A  CRPS pools (8192 x 128 x 8) from retune_lr/weights"
POOL_DEADLINE=$(( $(date +%s) + 3600 ))
STARTED=""; PIDS=""; FAIL=0; i=0
started() { case " $STARTED " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

while :; do
    PENDING=""
    for S in $SEEDS; do
        started "$S" && continue
        OUT=$O/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy
        if [ -f "$OUT" ]; then
            say "   seed $S pool exists -- skipping"
            STARTED="$STARTED $S"; continue
        fi
        if [ ! -f "$O/selection/seed_${S}_selection.json" ]; then
            PENDING="$PENDING $S"; continue
        fi
        G=$(( i % 2 ))                       # GPUs 0 and 1 only; 2 and 3 are shared
        LO=$(( 56 + i * 2 )); HI=$(( LO + 1 ))
        say "   seed $S selected -- launching pool on gpu $G cores $LO-$HI"
        CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        taskset -c $LO-$HI "$PY" "$C/generate_bank_true.py" \
            --seed "$S" --m-simu 8192 --device cuda:0 \
            --data-dir "$V" --seq-tag "$TAG" \
            --weights-root "$O" --out-root "$O/crps_banks" \
            > "$O/logs/pool_seed_$S.log" 2>&1 &
        PIDS="$PIDS $!"
        STARTED="$STARTED $S"
        i=$(( i + 1 ))
    done
    [ -z "$PENDING" ] && break
    [ "$(date +%s)" -gt "$POOL_DEADLINE" ] && {
        say "!! seeds [$(echo $PENDING)] never got a selection file within 1 h"; exit 1; }
    sleep 30
done
for p in $PIDS; do wait "$p" || FAIL=1; done
[ "$FAIL" -eq 0 ] || { say "!! pool generation failed"; tail -n 20 "$O"/logs/pool_seed_*.log; exit 1; }
for S in $SEEDS; do
    [ -f "$O/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy" ] \
        || { say "!! pool seed $S missing"; exit 1; }
done
say "     all 5 pools present"

# ---------------------------------------------------------------- crps ------
# Both conventions, always. Guideline section 14: shipping only `paper` is the
# single most common way to make an unfalsifiable claim here, because `perdim`
# is the sharper estimator AND the one that makes the method look worse.
#   paper  = --weight-mode paper  --standardize bank
#   perdim = --weight-mode perdim --standardize realtrain
# 12 jobs on cores 64-99 (36 cores, 3 each). No GPU: conditional_crps_multiasset
# contains zero torch references, and an empty CUDA_VISIBLE_DEVICES makes an
# accidental .cuda() fail loudly instead of stealing a card mid-run.
say "PRESTAGE B  12 conditional-CRPS jobs (no gpu, cores 64-99)"
PIDS=""; FAIL=0; i=0
for CFG in paper perdim; do
    if [ "$CFG" = paper ]; then WMODE=paper; STD=bank
    else                       WMODE=perdim; STD=realtrain; fi
    for T in $SEEDS realbank; do
        if [ "$T" = realbank ]; then
            OUT=$O/losses/crps_configs/${CFG}__realbank.json
            BANK=$V/true_S_$TAG.npy; SIZE=6144; LBL=real_train_bank
        else
            OUT=$O/losses/crps_configs/${CFG}__seed_$T.json
            BANK=$O/crps_banks/generated_paths/seed_$T/generated_paths_$CRPS_TAG.npy
            SIZE=8192; LBL=$LABEL
        fi
        if [ -f "$OUT" ]; then say "   $(basename "$OUT") exists -- skipping"; i=$(( i + 1 )); continue; fi
        LO=$(( 64 + i * 3 )); HI=$(( LO + 2 ))
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
        OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 \
        taskset -c $LO-$HI "$PY" metrics/conditional_crps_multiasset.py \
            --data-dir "$V" --seq-tag "$TAG" --bank-size "$SIZE" --label "$LBL" \
            --weight-mode "$WMODE" --standardize "$STD" --bank "$BANK" --out "$OUT" \
            > "$O/logs/crps_${CFG}__$T.log" 2>&1 &
        PIDS="$PIDS $!"
        i=$(( i + 1 ))
    done
done
for p in $PIDS; do wait "$p" || FAIL=1; done
[ "$FAIL" -eq 0 ] || { say "!! a CRPS job failed"; tail -n 15 "$O"/logs/crps_*.log; exit 1; }

for CFG in paper perdim; do
    for S in $SEEDS; do
        [ -f "$O/losses/crps_configs/${CFG}__seed_$S.json" ] \
            || { say "!! CRPS $CFG seed $S missing"; exit 1; }
    done
    [ -f "$O/losses/crps_configs/${CFG}__realbank.json" ] \
        || { say "!! CRPS $CFG realbank missing"; exit 1; }
done
say "PRESTAGE COMPLETE -- 5 pools + 12 CRPS configs under $O"
