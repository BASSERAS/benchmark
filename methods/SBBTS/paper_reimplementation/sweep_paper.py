"""Hyperparameter search on the PAPER experiment, targeting the paper's Figure 2.

Why this exists
---------------
The canonical ``run_heston.py`` settings reproduce the paper's *qualitative* claim
(SBBTS does not collapse xi and rho the way SBTS does) but under-shoot its
*quantitative* one: our std ratios are xi 0.694 / rho 0.768 where the paper's
Figure 2 shows the SBBTS density lying essentially on top of the data density,
i.e. a ratio near 1. This sweep asks which hyperparameter closes that gap.

Objective
---------
The paper publishes no numbers for Figure 2, so the target is the claim itself:
``std_ratio -> 1`` for every parameter, and hardest for the two the paper singles
out. Trials are ranked by

    gap = |log std_ratio(xi)| + |log std_ratio(rho)|

with ``gap_all`` (the same statistic averaged over all five parameters) as the
tie-break. A log ratio is used so 0.5 and 2.0 are penalised equally -- over- and
under-dispersion are both failures to match.

Cost
----
``--M-simu 1000`` instead of 4000: the per-path MLE dominates the run and scales
linearly in path count, while the standard error of a std estimated from ~700
surviving paths is about 2.7% -- far tighter than the 30% effect being measured.
The data-side MLE is disk-cached and shared by every trial.

Run (2 GPUs, 16 cores, one wave):

    cd methods/SBBTS/paper_reimplementation
    setsid /home/tbasseras/gpu-venv/bin/python sweep_paper.py run --gpus 0,1 \
      > results/sweep/sweep.log 2>&1 < /dev/null & disown
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METRIC = ROOT / "metric" / "reproduce_heston.py"
OUT = ROOT / "results" / "sweep"
PY = "/home/tbasseras/gpu-venv/bin/python"

MAX_GPUS = 2          # hard cap; the machine is shared
CORES_PER_GPU = 8     # GUIDELINE 4.1

# One-factor-at-a-time around the run_heston.py baseline. Arms that change model
# capacity or the learning rate also raise `patience`, because with the stock
# patience=15 they would be confounded with "stopped even earlier than base".
#   flags -> passed to reproduce_heston.py verbatim
TRIALS = [
    ("base",            []),
    ("patience=50",     ["--patience", "50"]),
    ("patience=150",    ["--patience", "150"]),
    ("delta=0",         ["--patience", "50", "--delta", "0"]),
    ("batch=64",        ["--patience", "50", "--set", "batch_size=64"]),
    ("K=8",             ["--K", "8"]),
    ("K=12",            ["--K", "12"]),
    ("N_pi=120",        ["--set", "N_pi=120"]),
    ("N_pi=250",        ["--set", "N_pi=250"]),
    ("safe_t=1e-3",     ["--set", "safe_t=1e-3"]),
    ("safe_t=5e-3",     ["--set", "safe_t=5e-3"]),
    ("beta=300",        ["--set", "beta=300"]),
    ("beta=1000",       ["--set", "beta=1000"]),
    ("d_model=256",     ["--patience", "50", "--set", "d_model=256"]),
    ("n_layers=4",      ["--patience", "50", "--set", "n_layers=4"]),
    ("lr=3e-4",         ["--patience", "100", "--set", "lr=3e-4"]),
    # Appended mid-sweep. safe_t=1e-3 and beta=300 each closed ~0.31 of the gap
    # on their own, and they act on different things -- safe_t on how close to T
    # the bridge is ever sampled, beta on the strength of the inverse transport.
    # Nothing yet tests whether the two effects add.
    ("st1e-3+beta300",  ["--set", "safe_t=1e-3", "--set", "beta=300"]),
    ("safe_t=5e-4",     ["--set", "safe_t=5e-4"]),
    # Wave 4. N_pi (60/120/250) and K (5/8/12) all landed inside the 0.13 noise
    # floor, so neither generation-time discretization nor DSBM depth is what
    # makes xi and rho under-disperse. safe_t looked like the only knob with a
    # monotone dose-response (1e-2 -> 5e-3 -> 1e-3 spans 0.32, ~2.5x the floor)
    # and beta looked saturating, so the remaining budget goes to that grid and
    # its crossings rather than to more early-stopping / capacity arms.
    #
    # REFUTED, 2026-09-01. On the corrected corr cache (see rebuild_corr_cache.py)
    # safe_t over 1e-4..1e-2 spans 0.108 with NO ordering -- 0.739, 0.816, 0.832,
    # 0.847, 0.804, 0.785, optimum in the middle, endpoints the two worst -- while
    # beta is the near-monotone one. The wave-4 budget was spent on the wrong knob.
    ("safe_t=1e-4",     ["--set", "safe_t=1e-4"]),
    ("safe_t=2e-3",     ["--set", "safe_t=2e-3"]),
    ("st5e-4+beta300",  ["--set", "safe_t=5e-4", "--set", "beta=300"]),
    ("st1e-3+beta1000", ["--set", "safe_t=1e-3", "--set", "beta=1000"]),
    ("st1e-4+beta300",  ["--set", "safe_t=1e-4", "--set", "beta=300"]),
    ("st5e-4+beta1000", ["--set", "safe_t=5e-4", "--set", "beta=1000"]),
    # Wave 5. Ranking moved to `corr` (leverage-spread ratio, no MLE), which
    # reverses the earlier reading: beta is the lever (100 -> 0.785, 300 ->
    # 0.905, 1000 -> 0.884, span 1.4x the seed noise) and safe_t is not (span
    # 0.047 < the 0.084 seed spread). These bracket beta=300 to test whether it
    # is a real optimum rather than the top of a noisy plateau.
    ("beta=150",        ["--set", "beta=150"]),
    ("beta=200",        ["--set", "beta=200"]),
    ("beta=500",        ["--set", "beta=500"]),
]


DS = ROOT / "dataset" / "X_heston_paper_5000x253x2.npy"
_CORR_CACHE = OUT / "corr_spread_cache.json"


def _corr_spread(levels):
    """Std across paths of corr(price log-return, variance log-return).

    This is the leverage effect read straight off the paths -- the MLE-free
    analogue of rho's dispersion, and one of the authors' own diagnostics
    (``utils/plot_metrics.py:plot_corr_matrix``, which for d=2 has this single
    off-diagonal entry).
    """
    R = np.diff(np.log(levels.astype(np.float64)), axis=1)
    c = np.array([np.corrcoef(p, rowvar=False)[0, 1] for p in R])
    return float(np.nanstd(c))


def _cache():
    if _CORR_CACHE.is_file():
        return json.loads(_CORR_CACHE.read_text())
    return {}


def corr_ratio(tag):
    """Generated leverage spread / data leverage spread. None if no paths on disk.

    1.0 = the generated paths reproduce the real dispersion of the leverage
    effect; below 1.0 = the same under-dispersion the MLE reports for rho.

    Cached by tag. WARNING: this cache is only sound if the array for a tag never
    changes, and that is NOT guaranteed -- ``--redo`` and any re-run reuse the tag
    and overwrite ``X_sbbts_*_<tag>.npy`` while the cache keeps serving the old
    value. This silently corrupted the beta comparison on 2026-09-01. After any
    re-run, run ``rebuild_corr_cache.py`` before reading a board.
    """
    cache = _cache()
    if tag in cache:
        return cache[tag]
    hits = sorted(OUT.glob(f"X_sbbts_*_{tag}.npy"))
    if not hits:
        return None
    gen = np.load(hits[0])
    keep = np.isfinite(gen).all(axis=(1, 2)) & (gen > 0).all(axis=(1, 2))
    if keep.sum() < 50:
        return None
    if "__data__" not in cache:
        cache["__data__"] = _corr_spread(np.load(DS))
    cache[tag] = _corr_spread(gen[keep]) / cache["__data__"]
    _CORR_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[tag]


def tag_of(i, seed=0):
    """Trial tag. Seed 0 keeps the bare ``tNN`` so wave-1 artefacts stay addressable;
    replicates get a suffix so a noise-floor wave cannot overwrite them."""
    return f"t{i:02d}" if seed == 0 else f"t{i:02d}s{seed}"


def slots_for(gpus, jobs_per_gpu):
    """Concurrency slots as (gpu, core_range, threads).

    A trial pins exactly one CPU core and leaves the A100 at ~5 GiB of 80: the
    loop is kernel-launch bound, not compute bound. Packing several trials onto
    one card is therefore close to linear speedup, so each card's 8-core block is
    split between the trials sharing it rather than handed to one.
    """
    out = []
    per = max(1, CORES_PER_GPU // jobs_per_gpu)
    for gpu in gpus:
        base = gpu * CORES_PER_GPU
        for j in range(jobs_per_gpu):
            lo = base + j * per
            out.append((gpu, f"{lo}-{lo + per - 1}", per))
    return out


def launch(i, note, flags, gpu, cores, threads, seed, m_simu):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = tag_of(i, seed)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        env[k] = str(threads)

    cmd = (["taskset", "-c", cores, PY, str(METRIC),
            "--seed", str(seed), "--out", str(OUT), "--tag", tag,
            "--M-simu", str(m_simu), "--mle-jobs", str(threads)] + flags)
    log = OUT / f"{tag}.log"
    fh = open(log, "w")
    fh.write(f"# {note}\n# {' '.join(cmd)}\n")
    fh.flush()
    print(f"[{tag}] {note:<14} -> GPU {gpu} cores {cores}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(METRIC.parent)), fh


def score(i, seed=0):
    """Read one trial's scores JSON; return None if it never got that far."""
    path = OUT / f"sbbts_heston_scores_{tag_of(i, seed)}.json"
    if not path.is_file():
        return None
    s = json.loads(path.read_text())
    by = {r["param"]: r for r in s["summary"]}
    gaps = {p: abs(math.log(by[p]["std_ratio"])) for p in by
            if by[p]["std_ratio"] > 0}
    if not gaps:
        return None
    cr = corr_ratio(tag_of(i, seed))
    return dict(
        tag=tag_of(i, seed),
        idx=i,
        seed=seed,
        ratios={p: by[p]["std_ratio"] for p in by},
        w1={p: by[p]["w1"] for p in by},
        gap=gaps.get("xi", math.nan) + gaps.get("rho", math.nan),
        gap_all=sum(gaps.values()) / len(gaps),
        corr=cr if cr is not None else math.nan,
        corr_gap=abs(math.log(cr)) if cr and cr > 0 else math.nan,
        train_min=s["timing"]["train_min"],
    )


def cmd_run(a):
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS:
        sys.exit(f"REFUSING: {len(gpus)} GPUs requested, hard cap is {MAX_GPUS}.")
    OUT.mkdir(parents=True, exist_ok=True)

    a.seeds = f"0,{a.seed}"      # the closing board must show the wave it just ran
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
    todo = [(i, n, f) for i, (n, f) in enumerate(TRIALS)
            if (only is None or i in only)
            and (a.redo or score(i, a.seed) is None)]
    if not todo:
        print("nothing to run (all selected trials already have scores; "
              "use --redo to force)", flush=True)
        return cmd_board(a)
    slots = slots_for(gpus, a.jobs_per_gpu)
    print(f"{len(todo)}/{len(TRIALS)} trials to run | {len(slots)} slots "
          f"({a.jobs_per_gpu}/GPU, {slots[0][2]} core(s) each) | "
          f"M_simu={a.m_simu}", flush=True)
    if a.dry_run:
        for n_, ((i, note, flags), s) in enumerate(zip(todo, slots * 4)):
            print(f"  {tag_of(i, a.seed)} {note:<14} GPU {s[0]} cores {s[1]}  flags={flags}")
        return

    t0 = time.perf_counter()
    pending = list(todo)
    running = {}
    while pending or running:
        for si, (gpu, cores, threads) in enumerate(slots):
            if si in running or not pending:
                continue
            i, note, flags = pending.pop(0)
            running[si] = (i, note, *launch(i, note, flags, gpu, cores, threads,
                                            a.seed, a.m_simu))
        finished = [si for si, (_, _, p, _) in running.items() if p.poll() is not None]
        if not finished:
            time.sleep(10)
            continue
        for si in finished:
            i, note, proc, fh = running.pop(si)
            fh.close()
            if proc.returncode != 0:
                print(f"[{tag_of(i, a.seed)}] FAILED rc={proc.returncode}, see "
                      f"{OUT / (tag_of(i, a.seed) + '.log')}", flush=True)
                continue
            r = score(i, a.seed)
            if r is None:
                print(f"[{tag_of(i, a.seed)}] finished but wrote no scores JSON", flush=True)
                continue
            print(f"[{tag_of(i, a.seed)}] {note:<14} corr={r['corr']:.3f} "
                  f"xi={r['ratios']['xi']:.3f} "
                  f"rho={r['ratios']['rho']:.3f} gap={r['gap']:.4f}", flush=True)

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    cmd_board(a)


def cmd_board(a):
    """Leaderboard over every trial x seed found on disk, best `corr` first.

    Ranking is on ``corr``, not ``gap``. ``gap`` is built from per-path MLE
    std_ratios and a replicate of one arm moved it 0.183 -> 0.451 on a base of
    0.499, i.e. the seed noise is over half the signal, so it cannot order the
    arms. ``corr`` measures the same underlying quantity -- dispersion of the
    leverage effect -- directly off the paths with no estimator, and on the same
    replicate moved only 0.832 -> 0.748. Both columns are kept because they
    agree on ordering, which is the evidence that either measures anything real.
    """
    seeds = sorted({0, *(int(s) for s in getattr(a, "seeds", "0").split(",") if s.strip())})
    rows = [r for r in (score(i, s) for s in seeds for i in range(len(TRIALS)))
            if r is not None]
    if not rows:
        print("no completed trials yet")
        return
    # NaN corr (paths missing from disk) sorts last rather than poisoning the order.
    rows.sort(key=lambda r: (math.inf if math.isnan(r["corr_gap"]) else r["corr_gap"],
                             r["gap"]))
    names = [n for n, _ in TRIALS]
    print(f"{'tag':>7} {'trial':<14} | {'corr':>6} | {'xi':>6} {'rho':>6} {'kappa':>6} "
          f"{'theta':>6} {'r':>6} | {'gap':>7} {'gap_all':>7} | {'train':>6}")
    print("-" * 100)
    for r in rows:
        q = r["ratios"]
        print(f"{r['tag']:>7} {names[r['idx']]:<14} | {r['corr']:>6.3f} | "
              f"{q['xi']:>6.3f} {q['rho']:>6.3f} "
              f"{q['kappa']:>6.3f} {q['theta']:>6.3f} {q['r']:>6.3f} | "
              f"{r['gap']:>7.4f} {r['gap_all']:>7.4f} | {r['train_min']:>5.1f}m")
    print("\ncorr = std of per-path corr(dlogS, dlogv), generated / data. "
          "1.000 = matched leverage")
    print("       dispersion. This is the ranking key: authors' own diagnostic, no MLE.")
    print("gap  = |log std_ratio(xi)| + |log std_ratio(rho)|, kept as the MLE cross-check.")

    # Replicates of the same arm bound the seed noise, without which no ranking
    # among the arms above is meaningful.
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["idx"], []).append(r)
    reps = {i: rs for i, rs in by_arm.items() if len(rs) > 1}
    if reps:
        print("\nseed replicates (spread here is the noise floor for the ranking):")
        for i, rs in sorted(reps.items()):
            c = [r["corr"] for r in rs]
            g = [r["gap"] for r in rs]
            print(f"  {names[i]:<14} n={len(rs)}  "
                  f"corr {min(c):.3f}..{max(c):.3f} (spread {max(c) - min(c):.3f})  "
                  f"gap {min(g):.3f}..{max(g):.3f} (spread {max(g) - min(g):.3f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpus", type=str, default="0,1")
    r.add_argument("--jobs-per-gpu", type=int, default=8,
                   help="trials sharing one card; each needs ~5 GiB and 1 core")
    r.add_argument("--m-simu", type=int, default=1000,
                   help="generated paths per trial; the MLE cost is linear in this")
    r.add_argument("--seed", type=int, default=0,
                   help="replicates land under tag tNNsSEED, so a noise-floor "
                        "wave never overwrites the seed-0 results")
    r.add_argument("--only", type=str, default="",
                   help="comma-separated trial indices, e.g. --only 0,2 to "
                        "replicate just those arms")
    r.add_argument("--redo", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run, seeds="0")
    b = sub.add_parser("board")
    b.add_argument("--seeds", type=str, default="0",
                   help="comma-separated seeds to include, e.g. --seeds 0,1,2")
    b.set_defaults(fn=cmd_board)
    t = sub.add_parser("trials")
    t.set_defaults(fn=lambda a: [print(f"{tag_of(i)}  {n:<14} {f}")
                                 for i, (n, f) in enumerate(TRIALS)])
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
