"""
SBBTS hyperparameter search on the benchmark Heston dataset.

Design copied from ``methods/TimeMoDE/sweep`` so the two searches are directly
comparable, and the scorer and ranking criterion are *imported* from it rather
than reimplemented:

  * ``sweep_score.score``      -- the 43 headline Heston contests (36 A + 6 B + grid_tvd)
  * ``sweep_run.objective``    -- mean normalised rank vs the 12 published methods
  * ``sweep_run.load_reference`` -- reads methods/TimeMoDE/sweep/refs/*_seed*.json

Selection runs against the **val** draw (``heston_S_val`` + ``heston_S_valdisc``,
seeds 3/4). The canonical test set (seed 1) and its judge (seed 2) stay unseen
until a final confirmation run, per GUIDELINE section 5.0.

Search design: one-axis-at-a-time around the paper baseline. A full grid over 8
axes is not affordable (each trial trains a transformer for tens of minutes), and
OAT is what makes the ``recap.py``-style "which axis actually matters" table
possible.

Screening uses a reduced epoch budget and fewer generated paths; the top
candidates are then re-run at full settings. Screening fidelity must be checked
before trusting the ranking -- see ``board``.

Hardware: 2 GPUs maximum (hard cap, asserted), 8 cores each.

Usage:
    python sweep_run.py trials                       # print the trial queue
    python sweep_run.py run --gpus 0,3               # screen every pending trial
    python sweep_run.py run --gpus 0,3 --full --only 3,7,12   # confirm winners
    python sweep_run.py board                        # leaderboard
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SWEEP = Path(__file__).resolve().parent
METHOD = SWEEP.parent
BENCH = METHOD.parent.parent
CODE = METHOD / "code"
RUNS = SWEEP / "runs"
TRIALS = SWEEP / "trials.jsonl"
TIMODE_SWEEP = BENCH / "methods" / "TimeMoDE" / "sweep"

PY = os.environ.get("SBBTS_PY", "/home/tbasseras/gpu-venv/bin/python")
MAX_GPUS = 2
CORES_PER_JOB = 8

sys.path.insert(0, str(TIMODE_SWEEP))
from sweep_score import CONTEST_KEYS, load_ref, score  # noqa: E402
from sweep_run import load_reference, objective        # noqa: E402


# ── search space ────────────────────────────────────────────────────────────
# Baseline = the authors' run_heston.py, i.e. what train_seed.py defaults to.
BASE = dict(beta=100, K=5, N_pi=60, d_model=128, hidden_dim=64,
            nhead=32, n_layers=2)

# One-axis-at-a-time deviations. beta stays >= 100: below that the reference
# switches to the learned-inverse branch (training_sbbts_inv.py), which is a
# different algorithm, not a hyperparameter setting.
AXES = {
    "beta":       [300, 1000],
    "K":          [3, 8],
    "N_pi":       [30, 120],
    "d_model":    [64, 256],
    "hidden_dim": [32, 128],
    "nhead":      [8, 16],
    "n_layers":   [1, 4],
}

# Screening budget. Full = train_seed.py defaults (n_epochs=1000, M_simu=8192).
SCREEN = dict(n_epochs=120, M_simu=2048)


def valid(cfg):
    """Reject configs the architecture cannot express."""
    if cfg["d_model"] % cfg["nhead"] != 0:
        return f"d_model {cfg['d_model']} not divisible by nhead {cfg['nhead']}"
    if cfg["beta"] < 100:
        return f"beta {cfg['beta']} < 100 selects the learned-inverse branch"
    return None


def gen_trials():
    """Baseline first, then one deviation per axis value."""
    out, seen = [], set()

    def add(cfg, note):
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            return
        why = valid(cfg)
        if why:
            print(f"  skip {note}: {why}", file=sys.stderr)
            return
        seen.add(key)
        out.append(dict(id=len(out), cfg=cfg, note=note))

    add(dict(BASE), "base=paper")
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(BASE)
            cfg[axis] = v
            add(cfg, f"{axis}={v}")
    return out


def read_trials():
    if not TRIALS.exists():
        trials = gen_trials()
        with open(TRIALS, "w") as f:
            for t in trials:
                f.write(json.dumps(t) + "\n")
        return trials
    with open(TRIALS) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── execution ───────────────────────────────────────────────────────────────
def trial_dir(tid, full):
    return RUNS / f"trial_{tid:04d}{'_full' if full else ''}"


def cfg_args(cfg, full):
    a = ["--beta", str(cfg["beta"]), "--K", str(cfg["K"]),
         "--N-pi", str(cfg["N_pi"]), "--d-model", str(cfg["d_model"]),
         "--hidden-dim", str(cfg["hidden_dim"]), "--nhead", str(cfg["nhead"]),
         "--n-layers", str(cfg["n_layers"])]
    if not full:
        a += ["--n-epochs", str(SCREEN["n_epochs"]), "--M-simu", str(SCREEN["M_simu"])]
    return a


def slots_for(gpus, jobs_per_gpu):
    """Concurrency slots as (gpu, core_range, threads).

    One trial saturates a single CPU core while leaving the A100 at ~5 GiB of 80:
    the loop is kernel-launch bound, not compute bound. Packing several trials
    onto one card is therefore close to linear speedup, so the card's 8-core
    block is split between the jobs sharing it rather than handed to one job.
    """
    out = []
    per = max(1, CORES_PER_JOB // jobs_per_gpu)
    for gpu in gpus:
        base = gpu * CORES_PER_JOB
        for j in range(jobs_per_gpu):
            lo = base + j * per
            out.append((gpu, f"{lo}-{lo + per - 1}", per))
    return out


def launch(tid, cfg, full, gpu, seed, cores, threads):
    """Train + generate one trial. train_seed.py writes into <d>/generated_paths/seed_<seed>/."""
    d = trial_dir(tid, full)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(dict(id=tid, cfg=cfg, full=full,
                                                   seed=seed), indent=2))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(threads)

    cmd = (["taskset", "-c", cores, PY, str(CODE / "train_seed.py"),
            "--seed", str(seed), "--gpu", str(gpu), "--out-root", str(d)]
           + cfg_args(cfg, full))
    fh = open(d / "train.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, d


def gen_npy(d, seed):
    hits = sorted((d / "generated_paths" / f"seed_{seed}").glob("generated_paths_*.npy"))
    return hits[0] if hits else None


def score_trial(d, seed, ref_name, metric_seed, score_gpu=0):
    """Score the generated paths and cache metrics.json next to them."""
    import torch
    npy = gen_npy(d, seed)
    if npy is None:
        return None
    fake = np.load(npy)
    S, v, S_disc = load_ref(ref_name)
    # Pin scoring to a card we were actually granted; the parent sees all GPUs.
    device = f"cuda:{score_gpu}" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    r = score(fake, S, v, S_disc, metric_seed, device)
    r["ref"] = ref_name
    r["metric_seed"] = metric_seed
    r["n_paths"] = int(fake.shape[0])
    r["score_sec"] = round(time.perf_counter() - t0, 2)
    (d / "metrics.json").write_text(json.dumps(r, indent=2))
    return r


def cmd_run(a):
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS:
        sys.exit(f"REFUSING: {len(gpus)} GPUs requested, hard cap is {MAX_GPUS}.")

    trials = read_trials()
    if a.only:
        want = {int(x) for x in a.only.split(",")}
        trials = [t for t in trials if t["id"] in want]
    todo = [t for t in trials
            if a.force or not (trial_dir(t["id"], a.full) / "metrics.json").exists()]
    print(f"{len(todo)}/{len(trials)} trials to run "
          f"({'FULL' if a.full else 'screen'}), gpus={gpus}", flush=True)

    ref, methods = load_reference()
    slots = slots_for(gpus, a.jobs_per_gpu)
    print(f"{len(slots)} concurrent slots "
          f"({a.jobs_per_gpu}/GPU, {slots[0][2]} cores each)", flush=True)

    pending = list(todo)
    running = {}          # slot index -> (trial, proc, fh, dir)
    done = 0
    while pending or running:
        for si, slot in enumerate(slots):
            if si in running or not pending:
                continue
            t = pending.pop(0)
            gpu, cores, threads = slot
            print(f"[{t['id']}] {t['note']} -> GPU {gpu} cores {cores}", flush=True)
            running[si] = (t, *launch(t["id"], t["cfg"], a.full, gpu, a.seed, cores, threads))

        # Reap whatever finished; sleep only when every slot is still busy.
        finished = [si for si, (_, p, _, _) in running.items() if p.poll() is not None]
        if not finished:
            time.sleep(5)
            continue
        for si in finished:
            t, proc, fh, d = running.pop(si)
            fh.close()
            done += 1
            if proc.returncode != 0:
                print(f"[{t['id']}] TRAIN FAILED rc={proc.returncode}, see {d}/train.log",
                      flush=True)
                continue
            m = score_trial(d, a.seed, a.ref, a.metric_seed, gpus[0])
            if m is None:
                print(f"[{t['id']}] no generated paths found", flush=True)
                continue
            obj, wins = objective(m, ref)
            print(f"[{t['id']}] {t['note']}  obj={obj:.4f}  "
                  f"wins={wins}/{len(CONTEST_KEYS)}  ({done}/{len(todo)})", flush=True)


def cmd_board(a):
    ref, methods = load_reference()
    trials = {t["id"]: t for t in read_trials()}
    rows = []
    for tid, t in trials.items():
        for full in (False, True):
            p = trial_dir(tid, full) / "metrics.json"
            if not p.exists():
                continue
            obj, wins = objective(json.loads(p.read_text()), ref)
            rows.append(dict(id=tid, note=t["note"], full=full, obj=obj, wins=wins))
    if not rows:
        print("no scored trials yet")
        return
    rows.sort(key=lambda r: r["obj"])
    print(f"Objective = mean normalised rank over {len(CONTEST_KEYS)} contests "
          f"vs {len(methods)} published methods (0 = beats all). Lower better.\n")
    print(f"| # | id | stage  | obj    | wins/{len(CONTEST_KEYS)} | config |")
    print("|---|----|--------|--------|-------|--------|")
    for i, r in enumerate(rows, 1):
        stage = "full" if r["full"] else "screen"
        print(f"| {i} | {r['id']} | {stage} | {r['obj']:.4f} | {r['wins']} | {r['note']} |")


def cmd_trials(a):
    for t in read_trials():
        print(f"{t['id']:>3}  {t['note']:<18}  {t['cfg']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("trials").set_defaults(fn=cmd_trials)

    r = sub.add_parser("run")
    r.add_argument("--gpus", default="0,3")
    r.add_argument("--jobs-per-gpu", type=int, default=4,
                   help="trials sharing one card; each needs ~5 GiB and 1 core")
    r.add_argument("--full", action="store_true", help="full budget instead of screening")
    r.add_argument("--only", default=None, help="comma-separated trial ids")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ref", choices=["val", "test"], default="val")
    r.add_argument("--metric_seed", type=int, default=0)
    r.add_argument("--force", action="store_true", help="re-run already-scored trials")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("board")
    b.set_defaults(fn=cmd_board)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
