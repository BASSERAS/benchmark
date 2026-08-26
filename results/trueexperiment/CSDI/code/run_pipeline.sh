#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CSDI / TrueDataset -- unattended end-to-end driver.
#
# Chains everything from "seeds 0 and 1 are still training" through to a
# rendered README, so the whole run can be left alone. Launch detached:
#
#   setsid nohup bash results/trueexperiment/CSDI/code/run_pipeline.sh \
#       > results/trueexperiment/CSDI/logs/pipeline.log 2>&1 < /dev/null & disown
#
# DESIGN NOTES -- read before editing.
#
# * RESUMABLE. Every stage checks for its own output first and skips if present.
#   A stage that died at hour 1 can be restarted without redoing the 40 minutes
#   of training in front of it. The guards are on artefacts, not on a progress
#   file, because the artefact IS the evidence the stage ran.
#
# * FAIL-FAST, NOT FAIL-SILENT. On any stage failure the script writes
#   PIPELINE_FAILED to logs/pipeline.status and exits non-zero. Continuing past
#   a failed stage would produce a README rendered from a partial run, which is
#   worse than no README -- the tables would still render, just wrong.
#
# * TWO LANES, HARD LIMIT. Which physical GPUs the lanes sit on is set by
#   $GPU_A / $GPU_B below and CHANGES BETWEEN RUNS -- this box is shared, so
#   check `nvidia-smi` and repoint those two variables before every launch
#   rather than trusting whatever they say now. Cores stay pinned 0-7 / 8-15 so
#   the two lanes cannot fight. Do not "optimise" this by adding a third lane.
#
# * real_floor IS NOT RECOMPUTED. results/trueexperiment/real_floor/ was built
#   by the SBTS run from the same three held-out real splits through the same
#   code. Re-running it would burn ~30 min reproducing bit-identical files.
# ---------------------------------------------------------------------------
set -u

cd /home/tbasseras/benchmark || exit 1

R=/home/tbasseras/benchmark/results/trueexperiment
M=$R/CSDI
C=$M/code
V=/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
DT=9.512937595129376e-07
PY=/home/tbasseras/gpu-venv/bin/python
CC=/home/tbasseras/.cc-venv/bin/python
SEEDS="0 1 2 3 4"
# Derived, never written out by hand. The 5-seed relaunch tripped the section-4
# contract gate precisely because SEEDS and collect_artifacts' --expect-seeds
# were two independent copies of the same number and only one got updated.
# Anything downstream that needs a COUNT or a COMMA LIST must come from here.
N_SEEDS=$(set -- $SEEDS; echo $#)
SEEDS_CSV=$(echo $SEEDS | tr ' ' ',')
STATUS=$M/logs/pipeline.status

mkdir -p "$M/logs" "$M/losses/crps_configs" "$M/plots"
echo "RUNNING $(date -Is)" > "$STATUS"

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { say "!! STAGE FAILED: $*"; echo "PIPELINE_FAILED $* $(date -Is)" > "$STATUS"; exit 1; }

# Run one command pinned to a GPU lane.
#   lane 0 -> GPU $GPU_A / cores 0-7
#   lane 1 -> GPU $GPU_B / cores 8-15
#
# 2026-08-26: both lanes moved onto GPU 0. The original GPU1+GPU3 split is no
# longer available -- GPU 1 was taken by another user on this box and GPUs 2/3
# are running our own Deep-MKV-TS-acfup seeds for the next ~4 hours. GPU 0 is
# genuinely idle (18 MiB), so both lanes share it.
#
# Sharing one GPU between the two lanes is deliberate rather than serialising to
# a single lane: the remaining work is dominated by stage 7, whose two CRPS
# conventions are independent and each leave the A100 far from saturated. The
# CORE pinning is what actually keeps them apart, and that is unchanged.
GPU_A=0
GPU_B=0
lane() {
    local l=$1; shift
    local gpu cores
    if [ "$l" = 0 ]; then gpu=$GPU_A; cores=0-7; else gpu=$GPU_B; cores=8-15; fi
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        taskset -c $cores "$@"
}

# ===========================================================================
say "STAGE 1  wait for the in-flight seed 0 / seed 1 training runs"
# ===========================================================================
# These were launched by hand before this script existed. Waiting on the
# PROCESS rather than on the artefact avoids a race where the .npy exists but
# np.save has not flushed.
#
# NOT `pgrep -f train_true.py`. The harness `bash -c` wrappers that launched
# these runs are still alive and their OWN command lines contain the string
# "train_true.py --seed 0", so a pgrep guard matches the wrapper forever and
# this stage never exits. Wait on the explicit python PIDs instead, and verify
# via /proc/<pid>/cmdline that the PID has not been recycled onto some
# unrelated process during the wait.
# Emptied 2026-08-26: both runs have since exited and their checkpoints are on
# disk. Leaving stale PIDs here is not merely dead weight -- a recycled PID that
# happened to land on a long-lived process would make this stage block forever.
# The cmdline check below would catch the common case, but empty is exact.
WAIT_PIDS=""
for P in $WAIT_PIDS; do
    if [ -r "/proc/$P/cmdline" ] && \
       tr '\0' ' ' < "/proc/$P/cmdline" | grep -q "python train_true.py"; then
        say "     waiting on pid $P"
        while [ -d "/proc/$P" ]; do sleep 30; done
        say "     pid $P exited"
    else
        say "     pid $P not a live train_true.py -- nothing to wait for"
    fi
done

# Belt and braces: the seed 0/1 artefacts must now exist, or those runs died.
for S in 0 1; do
    [ -f "$M/weights/seed_${S}_model.pt" ] \
        || die "seed $S finished without a checkpoint -- check logs/train_seed$S.log"
done
say "     seeds 0/1 complete"

# ===========================================================================
say "STAGE 2  train seeds 2, 3 and 4 (two lanes, GPU1 + GPU3)"
# ===========================================================================
# Two lanes, three seeds: lane 0 runs 2 then 4, lane 1 runs 3. Written as a
# per-lane QUEUE rather than one seed per background job, because with 3 jobs
# and 2 GPUs a flat `for S in 2 3 4; ... &` would put two trainers on GPU 1 at
# once. The lane is the resource; the loop inside it is the queue.
for QUEUE in "0 2 4" "1 3"; do
    set -- $QUEUE
    L=$1; shift
    (
        for S in "$@"; do
            if [ -f "$M/weights/seed_${S}_model.pt" ] && \
               [ -f "$M/generated_paths/seed_$S/generated_paths_${TAG}.npy" ]; then
                echo "     seed $S already trained -- skipping"
                continue
            fi
            lane $L $PY "$C/train_true.py" --seed "$S" --data-dir "$V" \
                --seq-tag "$TAG" --val-n 256 \
                > "$M/logs/train_seed$S.log" 2>&1 || exit 1
        done
    ) &
done
wait

for S in $SEEDS; do
    [ -f "$M/weights/seed_${S}_model.pt" ] || die "seed $S checkpoint missing"
    [ -f "$M/generated_paths/seed_$S/generated_paths_${TAG}.npy" ] \
        || die "seed $S A/B bank missing"
done
say "     all 5 checkpoints + A/B banks present"

# ===========================================================================
say "STAGE 3  generate the 8192-path conditional-CRPS pools"
# ===========================================================================
# A SEPARATE draw from each checkpoint, not a resample of the A/B bank, so the
# CRPS pool and the A-table bank are two independent samples of one fitted
# model. generate_bank_true.py reads the z-score out of the checkpoint rather
# than refitting it, so the pool cannot silently standardise on another split.
for QUEUE in "0 0 2 4" "1 1 3"; do
    set -- $QUEUE
    L=$1; shift
    (
        for S in "$@"; do
            OUT=$M/crps_banks/generated_paths/seed_$S/generated_paths_8192x128x8.npy
            if [ -f "$OUT" ]; then echo "     crps pool seed $S exists -- skipping"; continue; fi
            lane $L $PY "$C/generate_bank_true.py" --seed "$S" --m-simu 8192 \
                --data-dir "$V" --seq-tag "$TAG" --out-root "$M/crps_banks" \
                >> "$M/logs/crps_banks.log" 2>&1 || exit 1
        done
    ) &
done
wait

for S in $SEEDS; do
    [ -f "$M/crps_banks/generated_paths/seed_$S/generated_paths_8192x128x8.npy" ] \
        || die "CRPS pool for seed $S missing"
done
say "     all 5 CRPS pools present"

# ===========================================================================
say "STAGE 4  section-4 contract gate + provenance copy"
# ===========================================================================
# dataset_stats.json is the locked build's own stats. COPIED, not recomputed:
# the point is that every method in this tree quotes the SAME provenance
# record, so a divergence would mean someone silently rebuilt the dataset.
if [ ! -f "$M/losses/dataset_stats.json" ]; then
    cp "$R/SBTS/losses/dataset_stats.json" "$M/losses/dataset_stats.json" \
        || die "could not copy dataset_stats.json"
fi

$CC "$C/collect_artifacts.py" --expect-seeds "$N_SEEDS" \
    > "$M/logs/collect_artifacts.log" 2>&1 \
    || { cat "$M/logs/collect_artifacts.log"; die "collect_artifacts (section-4 contract breach)"; }
say "     contract gate passed"

# ===========================================================================
say "STAGE 5  metrics A1-A32 + B  (~40 min)"
# ===========================================================================
[ -f "$R/real_floor/metrics_summary.csv" ] || die "real_floor reference missing"

if [ -f "$M/metrics_summary.csv" ]; then
    say "     metrics_summary.csv exists -- skipping"
else
    lane 0 $PY metrics/compute_all_multiasset.py \
        --method CSDI --dataset TrueDataset --seeds "$N_SEEDS" \
        --data-dir "$V" --seq-tag "$TAG" --dt "$DT" --results-dir "$M" \
        > "$M/logs/metrics.log" 2>&1 \
        || { tail -30 "$M/logs/metrics.log"; die "compute_all_multiasset"; }
fi
[ -f "$M/metrics_summary.csv" ]    || die "metrics_summary.csv not produced"
[ -f "$M/curve_b_aggregate.json" ] || die "curve_b_aggregate.json not produced"
say "     metrics done"

# ===========================================================================
say "STAGE 6  memorisation guard"
# ===========================================================================
# The only guard there is on this dataset: a real market has no law to re-draw
# from, so a metric sinking below the floor does NOT flag copying here the way
# it would on Heston. Denominator is val, never test.
if [ -f "$M/losses/memorisation.json" ]; then
    say "     memorisation.json exists -- skipping"
else
    lane 0 $PY "$C/measure_memorisation.py" --data-dir "$V" --seq-tag "$TAG" \
        --seeds "$SEEDS_CSV" > "$M/logs/memorisation.log" 2>&1 \
        || { tail -20 "$M/logs/memorisation.log"; die "measure_memorisation"; }
fi
[ -f "$M/losses/memorisation.json" ] || die "memorisation.json not produced"
say "     memorisation done"

# ===========================================================================
say "STAGE 7  conditional CRPS -- both conventions, 5 seeds + realbank"
# ===========================================================================
# --label CSDI is LOAD-BEARING. render_readme.py keys table C on the method
# name; a mismatched label renders the table with baselines only and no error.
crps_run() {   # $1 lane  $2 cfgname  $3 weight-mode  $4 standardize  $5 seed|realbank
    local out bank size label
    if [ "$5" = realbank ]; then
        out=$M/losses/crps_configs/$2__realbank.json
        bank=$V/true_S_$TAG.npy; size=6144; label=real_train_bank
    else
        out=$M/losses/crps_configs/$2__seed_$5.json
        bank=$M/crps_banks/generated_paths/seed_$5/generated_paths_8192x128x8.npy
        size=8192; label=CSDI
    fi
    if [ -f "$out" ]; then echo "     $(basename "$out") exists -- skipping"; return 0; fi
    lane "$1" $PY metrics/conditional_crps_multiasset.py \
        --data-dir "$V" --seq-tag "$TAG" --bank-size "$size" --label "$label" \
        --weight-mode "$3" --standardize "$4" --bank "$bank" --out "$out" \
        >> "$M/logs/crps_$2.log" 2>&1
}

# Lane 0 takes the paper convention (the one that gets reported), lane 1 the
# perdim alternative that the README's caveat paragraph cites.
(
    for S in $SEEDS realbank; do crps_run 0 paper  paper  bank      "$S" || exit 1; done
) &
(
    for S in $SEEDS realbank; do crps_run 1 perdim perdim realtrain "$S" || exit 1; done
) &
wait

for CFG in paper perdim; do
    for S in $SEEDS; do
        [ -f "$M/losses/crps_configs/${CFG}__seed_$S.json" ] || die "CRPS $CFG seed $S missing"
    done
    [ -f "$M/losses/crps_configs/${CFG}__realbank.json" ] || die "CRPS $CFG realbank missing"
done
say "     all 12 CRPS runs done"

# ===========================================================================
say "STAGE 8  figures"
# ===========================================================================
lane 0 $PY "$C/plot_diagnostics_true.py" --data-dir "$V" --seq-tag "$TAG" \
    > "$M/logs/plot_diagnostics.log" 2>&1 || die "plot_diagnostics_true"
$PY "$C/plot_losses.py" --seeds "$SEEDS_CSV" \
    > "$M/logs/plot_losses.log" 2>&1 || die "plot_losses"
lane 0 $PY metrics/plot_score_losses.py --method CSDI --dataset TrueDataset \
    --results-dir "$M" > "$M/logs/plot_scores.log" 2>&1 || die "plot_score_losses"
say "     figures written"

# ===========================================================================
say "STAGE 8b  section-7 real-vs-real envelope screen"
# ===========================================================================
# NOT part of render_readme.py, which never computes the envelope. Without this
# stage the two HARD CONSTRAINTS of the selection criterion are simply absent
# from the README -- the A-table would show the damage spread across A29/A30/A31
# without ever stating the verdict. Numpy only, so it runs on $CC, not $PY.
# Non-fatal by design: a FAIL here is a FINDING about the method, not a broken
# pipeline, and must not abort the render that reports it.
$CC "$C/screen_envelope.py" --data-dir "$V" --seq-tag "$TAG" --seeds "$SEEDS_CSV" \
    > "$M/logs/envelope_screen.log" 2>&1 \
    || say "     !! envelope screen errored -- see logs/envelope_screen.log"
[ -f "$M/losses/envelope_screen.json" ] || die "envelope_screen.json not produced"
say "     envelope screen written"

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
say "----- convergence verdict (plot_losses) -----"
cat "$M/logs/plot_losses.log"
say "----- memorisation (the guard) -----"
cat "$M/logs/memorisation.log" 2>/dev/null | tail -20
say "----- section-7 envelope screen (the verdict) -----"
cat "$M/logs/envelope_screen.log" 2>/dev/null | tail -25
say "=========================================================="
echo "PIPELINE_COMPLETE $(date -Is)" > "$STATUS"
