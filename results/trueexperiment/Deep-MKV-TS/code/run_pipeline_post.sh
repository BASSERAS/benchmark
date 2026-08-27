#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deep-MKV-TS / TrueDataset -- unattended POST-TRAINING driver.
#
# Picks up where run_seeds_true.sh leaves off and carries the run through to a
# rendered README, so the whole thing can be launched now and left alone while
# the seeds are still training. Launch detached:
#
#   setsid nohup bash results/trueexperiment/Deep-MKV-TS/code/run_pipeline_post.sh \
#       > results/trueexperiment/Deep-MKV-TS/logs/pipeline_post.log 2>&1 < /dev/null & disown
#
# DESIGN NOTES -- read before editing.
#
# * STAGE 0 WAITS FOR TRAINING. run_seeds_true.sh deletes runs/.seed_queue only
#   after every worker has exited, so the ABSENCE of that file is the definitive
#   "all workers done" signal -- not a process scan, which would race with a
#   worker between two seeds, and not a COMPLETE.json count, which cannot tell
#   "still running" from "died". If the file is already gone this stage returns
#   immediately, so the script is equally correct launched before or after
#   training ends.
#
# * A MISSING SEED STOPS THE PIPELINE. If fewer seeds finished than were
#   launched, this script dies rather than quietly reporting a 4-seed run in the
#   slot a 5-seed run was planned for. That is the one thing the section-10
#   contract is most emphatic about. The operator's two honest options are both
#   available: rerun the seed (run_seeds_true.sh is idempotent and will skip the
#   survivors), or re-launch this script with SEEDS explicitly overridden, which
#   makes the reduced run a recorded decision instead of an accident.
#
# * RESUMABLE. Every stage checks for its own artefact first and skips if
#   present, so a stage that died at hour 2 restarts without redoing the two
#   hours in front of it. The guards are on artefacts, not on a progress file,
#   because the artefact IS the evidence the stage ran.
#
# * THREE LANES. GPU 0 is another user's and is never ours. The pool is 1/2/3
#   with cores pinned 0-7 / 8-15 / 16-23 so the lanes cannot fight over cores.
#   This box is shared -- check `nvidia-smi` and repoint $GPUS before launching
#   rather than trusting whatever it says now.
#
# * real_floor IS NOT RECOMPUTED, and neither is dataset_stats.json. Both were
#   built by the SBTS run from the same splits through the same code. Re-running
#   them would burn ~30 min reproducing bit-identical files, and any difference
#   would mean someone silently rebuilt the dataset.
# ---------------------------------------------------------------------------
set -u

cd /home/tbasseras/benchmark || exit 1

R=/home/tbasseras/benchmark/results/trueexperiment
M=$R/Deep-MKV-TS
C=$M/code
V=/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
CRPS_TAG=8192x128x8
DT=9.512937595129376e-07
PY=/home/tbasseras/gpu-venv/bin/python
CC=/home/tbasseras/.cc-venv/bin/python
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$REF/src:$REF/experiments:$C"

# --label is LOAD-BEARING: render_readme.py keys table C on the method name, and
# a mismatched label renders table C with baselines only AND NO ERROR.
LABEL=Deep-MKV-TS

LAUNCHED_SEEDS="0 1 2 3 4"
SEEDS="${SEEDS:-}"          # empty = discover; set explicitly to force a subset
GPUS="${GPUS:-1 2 3}"
WAIT_HOURS="${WAIT_HOURS:-8}"

STATUS=$M/logs/pipeline_post.status
mkdir -p "$M/logs" "$M/losses/crps_configs" "$M/plots" "$M/weights"
echo "RUNNING $(date -Is)" > "$STATUS"

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() {
    say "!! STAGE FAILED: $*"
    echo "PIPELINE_FAILED $* $(date -Is)" > "$STATUS"
    exit 1
}

# 64 cores are granted for this campaign (a standing override of the usual 16).
# LANE_W is how many of them each GPU lane gets during the SEQUENTIAL stages,
# where the lanes are the only thing running: 3 lanes x 20 = 60 of 64. The old
# value was 8, which left 40 cores idle while stage 2 and 3 did their numpy
# post-processing.
LANE_W="${LANE_W:-20}"

# Run one command pinned to a GPU and its core block. The Nth entry of $GPUS
# gets cores LANE_W*N .. LANE_W*N+LANE_W-1. CUDA_VISIBLE_DEVICES renumbers the
# visible device to 0, which is why every python call below says --device cuda:0
# regardless of which physical card it lands on.
lane() {
    local idx=$1; shift
    local gpu cores lo
    gpu=$(set -- $GPUS; eval echo "\${$((idx + 1))}")
    lo=$((idx * LANE_W))
    cores="$lo-$((lo + LANE_W - 1))"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$LANE_W MKL_NUM_THREADS=$LANE_W \
        OPENBLAS_NUM_THREADS=$LANE_W taskset -c "$cores" "$@"
}

n_lanes() { set -- $GPUS; echo $#; }

# Explicit (gpu, cores, threads) placement.
#
# lane() cannot be used by the STAGE 5-8b concurrent block. lane() derives its
# core range from a lane INDEX, and that block runs four DIFFERENT STAGES at once
# rather than several seeds of one stage -- two of them would map to lane 0 and
# silently fight over the same cores. run_on takes the range explicitly so the
# partition is stated in one place and can be checked by eye.
#
# An empty $1 means "no GPU". CUDA_VISIBLE_DEVICES="" is deliberate rather than
# merely unset: it makes an accidental .cuda() fail loudly instead of quietly
# taking a slice of a card another stage is mid-way through using.
run_on() {   # $1 gpu (""=none)   $2 "lo-hi"   $3 threads   $4.. command
    local gpu=$1 cores=$2 n=$3; shift 3
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" \
        OPENBLAS_NUM_THREADS="$n" NUMEXPR_NUM_THREADS="$n" \
        taskset -c "$cores" "$@"
}

# First two GPUs by name, for the concurrent block's two GPU stages.
GPU_A=$(set -- $GPUS; echo "$1")
GPU_B=$(set -- $GPUS; eval echo "\${2:-\$1}")

# ===========================================================================
say "STAGE 0  wait for run_seeds_true.sh to finish"
# ===========================================================================
QUEUE=$C/runs/.seed_queue
FAILED=$M/losses/failed_seeds.txt
deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))

# Wait on PER-SEED RESOLUTION, not on the queue file.
#
# The queue file is the wrong signal once more than one run_seeds_true.sh
# invocation is feeding the same queue -- and that is exactly what we do to widen
# the worker pool mid-flight. run_seeds_true.sh removes runs/.seed_queue when the
# workers of ITS OWN invocation drain, so the first invocation deletes the file
# while seeds popped by a later invocation are still training. Waiting on the
# file's absence would then release this script early, it would find 3 of 5
# COMPLETE.json, and die on the seed-count contract below -- aborting an
# otherwise healthy unattended chain.
#
# A seed is RESOLVED when it has either runs/seed_N/COMPLETE.json (success) or a
# line in losses/failed_seeds.txt (recorded failure). Every seed reaches one or
# the other, so this terminates on the same events the queue did, but it is
# insensitive to how many invocations are sharing the queue.
seed_resolved() {
    [ -f "$C/runs/seed_$1/COMPLETE.json" ] && return 0
    [ -f "$FAILED" ] && grep -qE "^$1 " "$FAILED" && return 0
    return 1
}

while :; do
    PENDING=""
    for S in $LAUNCHED_SEEDS; do
        seed_resolved "$S" || PENDING="$PENDING $S"
    done
    [ -z "$PENDING" ] && break
    [ "$(date +%s)" -gt "$deadline" ] && \
        die "seeds [$(echo $PENDING)] still unresolved after ${WAIT_HOURS}h -- refusing to wait longer"
    sleep 60
done
say "     all launched seeds resolved (complete or recorded as failed)"
[ -f "$QUEUE" ] && say "     note: $QUEUE still present -- a worker invocation is draining; harmless"

DONE_SEEDS=""
for S in $LAUNCHED_SEEDS; do
    [ -f "$C/runs/seed_$S/COMPLETE.json" ] && DONE_SEEDS="$DONE_SEEDS $S"
done
DONE_SEEDS=$(echo $DONE_SEEDS)
say "     seeds with COMPLETE.json: ${DONE_SEEDS:-none}"

if [ -z "$SEEDS" ]; then
    SEEDS="$DONE_SEEDS"
    if [ "$SEEDS" != "$LAUNCHED_SEEDS" ]; then
        [ -f "$M/losses/failed_seeds.txt" ] && cat "$M/losses/failed_seeds.txt"
        die "launched [$LAUNCHED_SEEDS] but only [$SEEDS] completed. Rerun the
     missing seed with run_seeds_true.sh (idempotent -- it skips the ones that
     already have COMPLETE.json), or relaunch this script with SEEDS='$SEEDS'
     to report the reduced run as a deliberate, recorded decision."
    fi
else
    say "     SEEDS overridden by the operator: [$SEEDS]"
fi
[ -n "$SEEDS" ] || die "no completed seeds"

# Derived, never written by hand. The CSDI 5-seed relaunch tripped the section-4
# contract gate precisely because SEEDS and --expect-seeds were two independent
# copies of the same number and only one got updated.
N_SEEDS=$(set -- $SEEDS; echo $#)
SEEDS_CSV=$(echo $SEEDS | tr ' ' ',')
say "     proceeding with $N_SEEDS seed(s): $SEEDS_CSV"

# Fan a per-seed command across the GPU lanes, waiting between waves.
# $1 is a shell function name taking (lane_index, seed).
fan_seeds() {
    local fn=$1 i=0 pids="" nl rc=0
    nl=$(n_lanes)
    for S in $SEEDS; do
        "$fn" "$((i % nl))" "$S" &
        pids="$pids $!"
        i=$((i + 1))
        if [ $((i % nl)) -eq 0 ]; then
            for p in $pids; do wait "$p" || rc=1; done
            pids=""
        fi
    done
    for p in $pids; do wait "$p" || rc=1; done
    return $rc
}

# ===========================================================================
say "STAGE 1  section-7 checkpoint selection (validation split)"
# ===========================================================================
# One process per seed rather than one process for all five. The envelope and
# the NNratio denominator are deterministic functions of the dataset, so every
# process computes the SAME ruler -- the per-seed objectives stay comparable --
# and a seed with no admissible checkpoint raises SystemExit without taking the
# other seeds' selections down with it.
select_one_seed() {
    local out=$C/selection/seed_$2_selection.json
    if [ -f "$out" ]; then echo "     seed $2 selection exists -- skipping"; return 0; fi
    lane "$1" $PY "$C/select_checkpoint_true.py" --seeds "$2" --device cuda:0 \
        > "$M/logs/select_seed_$2.log" 2>&1
}
fan_seeds select_one_seed || {
    for S in $SEEDS; do tail -5 "$M/logs/select_seed_$S.log" 2>/dev/null; done
    die "select_checkpoint_true"
}
for S in $SEEDS; do
    [ -f "$M/weights/seed_${S}_model.pt" ] || die "weights for seed $S missing"
done
say "     selected steps:"
grep -h SELECTED "$M"/logs/select_seed_*.log 2>/dev/null | sed 's/^/       /'

# ===========================================================================
say "STAGE 2  A/B path banks  (6144 x 128 x 8, one per seed)"
# ===========================================================================
bank_one_seed() {
    local out=$M/generated_paths/seed_$2/generated_paths_$TAG.npy
    if [ -f "$out" ]; then echo "     seed $2 A/B bank exists -- skipping"; return 0; fi
    lane "$1" $PY "$C/generate_bank_true.py" --seed "$2" --device cuda:0 \
        --data-dir "$V" --seq-tag "$TAG" --out-root "$M" \
        > "$M/logs/bank_seed_$2.log" 2>&1
}
fan_seeds bank_one_seed || {
    for S in $SEEDS; do tail -5 "$M/logs/bank_seed_$S.log" 2>/dev/null; done
    die "generate_bank_true (A/B)"
}
say "     all $N_SEEDS A/B banks present"

# ===========================================================================
say "STAGE 3  CRPS pools  (8192 x 128 x 8, SEPARATE draw)"
# ===========================================================================
# A separate draw from the same selected checkpoint, not a resample of the A/B
# bank: table C conditions on a history window and needs more paths per query
# than the A/B bank has, and reusing those paths would correlate the CRPS
# estimate with the very bank the A-table scores.
pool_one_seed() {
    local out=$M/crps_banks/generated_paths/seed_$2/generated_paths_$CRPS_TAG.npy
    if [ -f "$out" ]; then echo "     seed $2 CRPS pool exists -- skipping"; return 0; fi
    lane "$1" $PY "$C/generate_bank_true.py" --seed "$2" --m-simu 8192 --device cuda:0 \
        --data-dir "$V" --seq-tag "$TAG" --out-root "$M/crps_banks" \
        > "$M/logs/pool_seed_$2.log" 2>&1
}
fan_seeds pool_one_seed || {
    for S in $SEEDS; do tail -5 "$M/logs/pool_seed_$S.log" 2>/dev/null; done
    die "generate_bank_true (CRPS pool)"
}
for S in $SEEDS; do
    [ -f "$M/crps_banks/generated_paths/seed_$S/generated_paths_$CRPS_TAG.npy" ] \
        || die "CRPS pool for seed $S missing"
done
say "     all $N_SEEDS CRPS pools present"

# ===========================================================================
say "STAGE 4  section-4 contract gate + provenance copy"
# ===========================================================================
if [ ! -f "$M/losses/dataset_stats.json" ]; then
    cp "$R/SBTS/losses/dataset_stats.json" "$M/losses/dataset_stats.json" \
        || die "could not copy dataset_stats.json"
fi
$CC "$C/collect_artifacts.py" --expect-seeds "$N_SEEDS" \
    > "$M/logs/collect_artifacts.log" 2>&1 \
    || { cat "$M/logs/collect_artifacts.log"; die "collect_artifacts (section-4 contract breach)"; }
say "     contract gate passed"

# ===========================================================================
say "STAGES 5,6,7,8b  metrics / memorisation / CRPS / envelope  (CONCURRENT)"
# ===========================================================================
# These four stages are MUTUALLY INDEPENDENT. Stage 5 writes metrics_summary.csv
# and curve_b_aggregate.json, stage 6 writes losses/memorisation.json, stage 7
# writes losses/crps_configs/*.json, stage 8b writes losses/envelope_screen.json.
# None of them reads another's output. The first thing that needs all four is
# STAGE 9 (render), so that is where the barrier belongs -- not between each pair.
# Run one after another they cost roughly 40 + 10 + 50 + 5 minutes; run together
# they cost the longest one, which is stage 5.
#
# A33/A34 do not exist on this dataset. They are OMITTED from the README, not
# rendered as '-': a dash in a results table reads as "measured, came out
# empty", which is a different and false claim.
#
# CORE BUDGET -- a STATIC, NON-OVERLAPPING partition of the 64 granted cores:
#     stage 5    gpu $GPU_A   cores  0-15   (16)   numpy-heavy, the long pole
#     stage 6    gpu $GPU_B   cores 16-23   ( 8)   torch, short
#     stage 7    no gpu       cores 24-59   (36)   pure numpy, 2*(N_SEEDS+1) jobs
#     stage 8b   no gpu       cores 60-63   ( 4)   pure numpy
# Non-overlapping is the whole point. Twelve CRPS jobs oversubscribing stage 5's
# cores would slow the long pole and win nothing, so a clean partition is worth
# more than the extra cores any single stage could have been given.
#
# Stage 7 gets no GPU because it needs none: conditional_crps_multiasset.py
# contains zero torch and zero cuda references. It was previously fanned over the
# GPU lanes anyway, which serialised 12 CPU jobs into 5 waves on 24 cores.
#
# The THIRD GPU is deliberately unused here. It is shared with another user on
# this machine, and stages 5 and 6 are the only GPU work left in the pipeline.
#
# Each stage keeps its own failure semantics, but the block dies AFTER the
# barrier, not at the first casualty: killing the block the moment stage 7 fails
# would throw away a stage-5 run that was 35 minutes in.
[ -f "$R/real_floor/metrics_summary.csv" ] || die "real_floor reference missing"

stage5_metrics() {
    if [ -f "$M/metrics_summary.csv" ]; then
        say "     [5] metrics_summary.csv exists -- skipping"; return 0
    fi
    say "     [5] metrics A1-A32 + curve B starting  (gpu $GPU_A, cores 0-15)"
    run_on "$GPU_A" 0-15 16 $PY metrics/compute_all_multiasset.py \
        --method "$LABEL" --dataset TrueDataset --seeds "$N_SEEDS" \
        --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$M" \
        > "$M/logs/metrics.log" 2>&1
}

# A real market has no law to re-draw from, so a metric sinking below the floor
# does NOT flag copying here the way it would on Heston. Denominator is val,
# never test.
stage6_memorisation() {
    if [ -f "$M/losses/memorisation.json" ]; then
        say "     [6] memorisation.json exists -- skipping"; return 0
    fi
    say "     [6] memorisation guard starting  (gpu $GPU_B, cores 16-23)"
    run_on "$GPU_B" 16-23 8 $PY "$C/measure_memorisation.py" \
        --data-dir "$V" --seq-tag "$TAG" --seeds "$SEEDS_CSV" \
        > "$M/logs/memorisation.log" 2>&1
}

crps_run() {   # $1 cores  $2 threads  $3 cfg  $4 weight-mode  $5 standardize  $6 seed|realbank
    local out bank size label
    if [ "$6" = realbank ]; then
        out=$M/losses/crps_configs/$3__realbank.json
        bank=$V/true_S_$TAG.npy; size=6144; label=real_train_bank
    else
        out=$M/losses/crps_configs/$3__seed_$6.json
        bank=$M/crps_banks/generated_paths/seed_$6/generated_paths_$CRPS_TAG.npy
        size=8192; label=$LABEL
    fi
    if [ -f "$out" ]; then echo "     $(basename "$out") exists -- skipping"; return 0; fi
    # One log PER JOB, not per convention. Twelve concurrent writers appending to
    # two shared files would interleave mid-line and make a failure unreadable.
    run_on "" "$1" "$2" $PY metrics/conditional_crps_multiasset.py \
        --data-dir "$V" --seq-tag "$TAG" --bank-size "$size" --label "$label" \
        --weight-mode "$4" --standardize "$5" --bank "$bank" --out "$out" \
        > "$M/logs/crps_$3__$6.log" 2>&1
}

# All 2*(N_SEEDS+1) CRPS jobs at once across cores 24-59. Both conventions and
# both realbanks are in the same fan: they are independent processes writing
# distinct files, and the realbank floor is not an input to the seed runs.
stage7_crps() {
    local njobs per i=0 pids="" rc=0 lo hi wmode std
    njobs=$(( 2 * (N_SEEDS + 1) ))
    per=$(( 36 / njobs )); [ "$per" -lt 1 ] && per=1
    say "     [7] $njobs CRPS jobs starting concurrently  (no gpu, cores 24-59, ${per}c each)"
    for CFG in paper perdim; do
        if [ "$CFG" = paper ]; then wmode=paper; std=bank
        else                        wmode=perdim; std=realtrain; fi
        for TARGET in $SEEDS realbank; do
            lo=$(( 24 + i * per )); hi=$(( lo + per - 1 ))
            [ "$hi" -gt 59 ] && hi=59
            crps_run "$lo-$hi" "$per" "$CFG" "$wmode" "$std" "$TARGET" &
            pids="$pids $!"
            i=$(( i + 1 ))
        done
    done
    for p in $pids; do wait "$p" || rc=1; done
    return $rc
}

# NOT part of render_readme.py, which never computes the envelope. Numpy only,
# so it runs on $CC. Non-fatal by design: a FAIL here is a FINDING about the
# method, not a broken pipeline, and must not abort the render that reports it.
stage8b_envelope() {
    say "     [8b] envelope screen starting  (no gpu, cores 60-63)"
    run_on "" 60-63 4 $CC "$C/screen_envelope.py" --data-dir "$V" \
        --seq-tag "$TAG" --seeds "$SEEDS_CSV" \
        > "$M/logs/envelope_screen.log" 2>&1
}

stage5_metrics      & P5=$!
stage6_memorisation & P6=$!
stage7_crps         & P7=$!
stage8b_envelope    & P8B=$!
say "     barrier: 5=$P5  6=$P6  7=$P7  8b=$P8B"

FAILED_STAGES=""
wait "$P5"  || FAILED_STAGES="$FAILED_STAGES 5"
wait "$P6"  || FAILED_STAGES="$FAILED_STAGES 6"
wait "$P7"  || FAILED_STAGES="$FAILED_STAGES 7"
wait "$P8B" || say "     !! envelope screen errored -- see logs/envelope_screen.log"
say "     concurrent block joined"

if [ -n "$FAILED_STAGES" ]; then
    for s in $FAILED_STAGES; do
        case $s in
            5) say "--- stage 5 tail ---"; tail -30 "$M/logs/metrics.log" ;;
            6) say "--- stage 6 tail ---"; tail -20 "$M/logs/memorisation.log" ;;
            7) say "--- stage 7 tails ---"; tail -20 "$M"/logs/crps_*.log ;;
        esac
    done
    die "concurrent block: stage(s)$FAILED_STAGES failed"
fi

[ -f "$M/metrics_summary.csv" ]      || die "metrics_summary.csv not produced"
[ -f "$M/curve_b_aggregate.json" ]   || die "curve_b_aggregate.json not produced"
[ -f "$M/losses/memorisation.json" ] || die "memorisation.json not produced"
for CFG in paper perdim; do
    for S in $SEEDS; do
        [ -f "$M/losses/crps_configs/${CFG}__seed_$S.json" ] || die "CRPS $CFG seed $S missing"
    done
    [ -f "$M/losses/crps_configs/${CFG}__realbank.json" ] || die "CRPS $CFG realbank missing"
done
[ -f "$M/losses/envelope_screen.json" ] || die "envelope_screen.json not produced"
say "     metrics, memorisation, $((2 * (N_SEEDS + 1))) CRPS runs and envelope screen all present"

# ===========================================================================
say "STAGE 8  figures"
# ===========================================================================
# After the barrier, not inside it: plot_score_losses reads --results-dir "$M",
# which stage 5 is still writing while the block runs.
lane 0 $PY "$C/plot_diagnostics_true.py" --data-dir "$V" --seq-tag "$TAG" \
    > "$M/logs/plot_diagnostics.log" 2>&1 || die "plot_diagnostics_true"
$PY "$C/plot_losses.py" --seeds "$SEEDS_CSV" \
    > "$M/logs/plot_losses.log" 2>&1 || die "plot_losses"
lane 0 $PY metrics/plot_score_losses.py --method "$LABEL" --dataset TrueDataset \
    --results-dir "$M" > "$M/logs/plot_scores.log" 2>&1 || die "plot_score_losses"
say "     figures written"

# ===========================================================================
say "STAGE 9  render README"
# ===========================================================================
$PY "$C/render_readme.py" > "$M/logs/render.log" 2>&1 \
    || { cat "$M/logs/render.log"; die "render_readme"; }
[ -f "$M/README.md" ] || die "README.md not produced"

# Section 10.4 structural check: the phrase below is banned in this tree --
# nothing here reproduces a published table exactly, and claiming otherwise is
# the specific failure the guideline was written to prevent.
if [ "$(grep -c 'reproduced exactly' "$M/README.md")" != "0" ]; then
    die "README contains the banned phrase 'reproduced exactly'"
fi

say "=========================================================="
say "PIPELINE COMPLETE"
say "  README     $M/README.md  ($(wc -l < "$M/README.md") lines)"
say "----- convergence + selection verdict (plot_losses) -----"
cat "$M/logs/plot_losses.log"
say "----- memorisation (the guard) -----"
tail -20 "$M/logs/memorisation.log" 2>/dev/null
say "----- section-7 envelope screen (the verdict) -----"
tail -25 "$M/logs/envelope_screen.log" 2>/dev/null
say "=========================================================="
echo "PIPELINE_COMPLETE $(date -Is)" > "$STATUS"
