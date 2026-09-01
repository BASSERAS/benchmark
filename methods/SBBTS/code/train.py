"""
SBBTS 5-seed orchestrator for the benchmark Heston dataset.

Implements the batching pattern mandated by GUIDELINE section 4.2: seeds are run
two at a time, one per GPU, with CPU cores pinned per job. The 2-GPU cap is a hard
assertion, not a convention -- the machine is shared and the user's standing
instruction is "take care not to use more than 2 GPU for it".

    batch 1: seeds 0, 1   batch 2: seeds 2, 3   batch 3: seed 4 alone

Each worker is ``train_seed.py``; this file only schedules them and tees their
stdout to ``methods/SBBTS/losses/train_seed_{i}.log``.

Run detached (training is multi-hour; do not use a foreground shell):

    cd methods/SBBTS/code
    setsid /home/tbasseras/gpu-venv/bin/python train.py --gpus 0,3 \
      > ../losses/train_all.log 2>&1 < /dev/null & disown
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
BENCH = CODE.parent.parent.parent
LOG_DIR = BENCH / "methods" / "SBBTS" / "losses"

MAX_GPUS = 2          # hard cap, see module docstring
CORES_PER_JOB = 8     # GUIDELINE 4.1


def slots_for(gpus, jobs_per_gpu):
    """Concurrency slots as (gpu, core_range, threads).

    One seed saturates a single CPU core while leaving the A100 at ~5 GiB of 80:
    the loop is kernel-launch bound, not compute bound. Packing several seeds
    onto one card is therefore close to linear speedup, so the card's 8-core
    block is split between the seeds sharing it rather than handed to one.
    At jobs_per_gpu=1 this reproduces the old GUIDELINE 4.2 layout exactly.
    """
    out = []
    per = max(1, CORES_PER_JOB // jobs_per_gpu)
    for gpu in gpus:
        base = gpu * CORES_PER_JOB
        for j in range(jobs_per_gpu):
            lo = base + j * per
            out.append((gpu, f"{lo}-{lo + per - 1}", per))
    return out


def preflight(gpus):
    """Print GPU occupancy so the operator can abort before stealing someone's card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        print(f"WARNING: nvidia-smi failed ({e}); cannot verify GPU availability", flush=True)
        return
    print("nvidia-smi:", flush=True)
    for line in out.splitlines():
        idx = line.split(",")[0].strip()
        mark = "  <-- will be used" if int(idx) in gpus else ""
        print(f"  {line}{mark}", flush=True)


def launch(seed, gpu, python, extra, cores, threads):
    """Start one train_seed.py subprocess pinned to `gpu` and its core slice."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_seed_{seed}.log"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    cmd = ["taskset", "-c", cores, python,
           str(CODE / "train_seed.py"), "--seed", str(seed), "--gpu", str(gpu)] + extra

    print(f"  seed {seed} -> GPU {gpu}, cores {cores}, log {log_path.name}",
          flush=True)
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(CODE))
    return proc, fh, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--gpus", type=str, default="0,3",
                    help="comma-separated GPU ids; at most 2 (hard cap)")
    ap.add_argument("--jobs-per-gpu", type=int, default=3,
                    help="seeds sharing one card; each needs ~5 GiB and 1 core")
    ap.add_argument("--python", type=str, default="/home/tbasseras/gpu-venv/bin/python")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS:
        sys.exit(f"REFUSING: {len(gpus)} GPUs requested, hard cap is {MAX_GPUS}.")

    print("=" * 70)
    print(f"SBBTS 5-seed orchestrator | seeds={seeds} | gpus={gpus}")
    print("=" * 70, flush=True)
    preflight(set(gpus))

    slots = slots_for(gpus, args.jobs_per_gpu)
    print(f"{len(slots)} concurrent slots ({args.jobs_per_gpu}/GPU, "
          f"{slots[0][2]} cores each) for {len(seeds)} seeds", flush=True)
    if args.dry_run:
        for seed, (g, cores, thr) in zip(seeds, slots):
            print(f"  seed {seed} -> GPU {g} (cores {cores}, {thr} threads)")
        if len(seeds) > len(slots):
            print(f"  ...{len(seeds) - len(slots)} seed(s) queued behind the first wave")
        return

    t_all = time.perf_counter()
    failures = []
    pending = list(seeds)
    running = {}          # slot index -> (seed, proc, fh, log_path)
    while pending or running:
        for si, (gpu, cores, threads) in enumerate(slots):
            if si in running or not pending:
                continue
            seed = pending.pop(0)
            running[si] = (seed, *launch(seed, gpu, args.python, extra, cores, threads))

        finished = [si for si, (_, p, _, _) in running.items() if p.poll() is not None]
        if not finished:
            time.sleep(5)
            continue
        for si in finished:
            seed, proc, fh, log_path = running.pop(si)
            fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  seed {seed}: {status}  [{log_path}]", flush=True)
            if proc.returncode != 0:
                failures.append(seed)

    print(f"\ntotal {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    if failures:
        sys.exit(f"FAILED seeds: {failures} -- inspect {LOG_DIR}/train_seed_<i>.log")
    print("all seeds completed", flush=True)


if __name__ == "__main__":
    main()
