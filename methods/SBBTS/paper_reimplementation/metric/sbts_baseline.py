"""The SBTS comparator for Figure 2, on the same Heston dataset as SBBTS.

Why this exists. The paper's Figure 2 has three curves -- Data, SBTS, SBBTS --
and its whole quantitative claim is the *contrast*: SBBTS "captures the full
real range for all parameters" whereas SBTS "failed to reproduce the vol of vol
xi and the correlation rho", which get "projected onto an effective average,
yielding a concentrated distribution around the centre of the parameter range".
Without the SBTS curve we can only say our SBBTS numbers are plausible; with it
we can test the claim, because the claim is about a gap, not a level.

The authors' released SBBTS repo has no SBTS path, so the recipe here is taken
from the SBTS repo's own notebook (``methods/SBTS/code/reference/
experiments_demo.ipynb``, the "Parameters Estimation: Illustration on Heston
Data" section, cells 8-11), which is the same two-channel Heston experiment:

    log_returns = np.zeros((M, N+1, 2))
    log_returns[:,1:] = np.diff(np.log(X_heston), axis=1)
    X_heston = X_heston[:, 1:]
    log_returns_sbts = simulateSB_multi_mark(N, M, d=2, K=1, X=log_returns,
                                             N_pi=100, h=0.05, deltati=1/252,
                                             M_simu=500)
    X_heston_sbts = np.exp(log_returns_sbts.cumsum(axis=1))

Hence the defaults K=1, h=0.05, N_pi=100. Note SBTS is fed *raw* log returns:
unlike SBBTS it has no per-channel rescaling, so none is applied here either.
The preprocessing is otherwise byte-identical to ``reproduce_heston.py``
(``build_training_tensor``), which is what makes the two columns comparable.

The kernel itself is imported unmodified from the SBTS reference; nothing in
this file reimplements the method.

    cd metric && python sbts_baseline.py --M-simu 1000 --workers 3 --mle-jobs 1 \
                     --h 0.05 --tag sbtsh0p05
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # paper_reimplementation/
BENCH = ROOT.parent.parent.parent                    # benchmark/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH / "methods" / "SBTS" / "code" / "reference"))

from heston_mle import get_params_estimation, plot_figure2, summarize  # noqa: E402
from models.sbts_multi_markovian import simulate_kernel_vectorized_mark  # noqa: E402

# Author's notebook settings for the d=2 Heston case (cell 9).
#
# h = 0.05, NOT 0.2. This default was corrected on 2026-09-01 after the SBTS
# author confirmed the Heston notebook uses 0.05 and asked that every further
# SBTS experiment keep h this low. The paper's Appendix Table 4 quotes h = 0.4,
# which over-smooths at this sequence length; the notebook value supersedes it.
# The superseded h = 0.2 run is still on disk under `--tag sbts` and appears as
# one row of the bandwidth sweep in ../README.md §4, but it is not the reported
# SBTS column.
DEFAULTS = dict(K=1, h=0.05, N_pi=100, dt=1 / 252)


def build_log_returns(X_levels):
    """Levels (M, N+1, 2) -> (log returns (M, N+1, 2), levels aligned (M, N, 2)).

    Identical to ``reproduce_heston.build_training_tensor`` minus the rescaling,
    which SBTS does not use. Row 0 of the returns is the zero seed that
    ``simulate_kernel_vectorized_mark`` starts every path from.
    """
    log_returns = np.zeros_like(X_levels, dtype=np.float64)
    log_returns[:, 1:] = np.diff(np.log(X_levels), axis=1)
    return log_returns, X_levels[:, 1:]


def _worker(args):
    """Generate ``n`` paths in log-return space. Returns (n, N, d), initial row dropped."""
    n, X, N, M, d, K, N_pi, h, dt, seed = args
    np.random.seed(seed)
    out = np.zeros((n, N + 1, d))
    for k in range(n):
        out[k] = simulate_kernel_vectorized_mark(N, M, d, K, X, N_pi, h, dt)
    return out[:, 1:]


def generate(log_returns, M_simu, K, h, N_pi, dt, workers, seed):
    """Parallel SBTS generation. Returns (levels (M_simu, N, d), elapsed seconds).

    Workers are forked, so each inherits the compiled Numba kernel and a
    read-only copy-on-write view of the training array; only the per-worker RNG
    seed differs.
    """
    M, T, d = log_returns.shape
    N = T - 1

    base, rem = divmod(M_simu, workers)
    chunks = [c for c in (base + (1 if i < rem else 0) for i in range(workers)) if c > 0]
    args = [(c, log_returns, N, M, d, K, N_pi, h, dt, seed * 10_000 + i)
            for i, c in enumerate(chunks)]
    print(f"      {len(chunks)} workers x ~{chunks[0]} paths", flush=True)

    t0 = time.perf_counter()
    with Pool(len(args)) as pool:
        parts = pool.map(_worker, args)
    elapsed = time.perf_counter() - t0

    gen_returns = np.vstack(parts)                       # (M_simu, N, d)
    return np.exp(gen_returns.cumsum(axis=1)), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5000, help="training paths")
    ap.add_argument("--M-simu", type=int, default=1000)
    ap.add_argument("--K", type=int, default=DEFAULTS["K"])
    ap.add_argument("--h", type=float, default=DEFAULTS["h"])
    ap.add_argument("--N-pi", type=int, default=DEFAULTS["N_pi"])
    ap.add_argument("--workers", type=int, default=6,
                    help="max 16 physical cores on this machine, shared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mle-jobs", type=int, default=1)
    ap.add_argument("--tag", type=str, default="sbts")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "sweep"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_dir = ROOT / "dataset"
    np.random.seed(a.seed)

    # -- 1. Same dataset, same transform as the SBBTS runs --
    x_path = ds_dir / f"X_heston_paper_{a.M}x253x2.npy"
    X_levels = np.load(x_path)
    log_returns, levels_aligned = build_log_returns(X_levels)
    N = log_returns.shape[1] - 1
    print(f"[1/4] {x_path.name} {X_levels.shape} -> returns {log_returns.shape}", flush=True)
    print(f"      K={a.K} h={a.h} N_pi={a.N_pi} dt={DEFAULTS['dt']:.6f} "
          f"(SBTS notebook cell 9; no rescaling)", flush=True)

    # -- 2. Generate --
    print(f"[2/4] generating {a.M_simu} paths", flush=True)
    X_sbts, gen_sec = generate(log_returns, a.M_simu, a.K, a.h, a.N_pi,
                               DEFAULTS["dt"], a.workers, a.seed)
    print(f"      done in {gen_sec / 60:.1f} min  "
          f"price[{X_sbts[:, :, 0].min():.3g}, {X_sbts[:, :, 0].max():.3g}]  "
          f"var[{X_sbts[:, :, 1].min():.3g}, {X_sbts[:, :, 1].max():.3g}]", flush=True)
    gen_name = f"X_sbts_{X_sbts.shape[0]}x{X_sbts.shape[1]}x2_{a.tag}.npy"
    np.save(out / gen_name, X_sbts)

    # The MLE takes log S and divides by v, so it needs both channels positive.
    valid = np.isfinite(X_sbts).all(axis=(1, 2)) & (X_sbts > 0).all(axis=(1, 2))
    X_mle = np.ascontiguousarray(X_sbts[valid].astype(np.float64))
    print(f"      MLE-eligible: {len(X_mle)}/{len(X_sbts)} "
          f"({int((~valid).sum())} dropped)", flush=True)

    # -- 3. The paper's metric, same estimator and same cached data side --
    print(f"[3/4] per-path Heston MLE ({a.mle_jobs} workers)", flush=True)
    t0 = time.perf_counter()
    data_cache = ds_dir / f"params_data_{a.M}x5.npy"
    if data_cache.exists():
        params_data = np.load(data_cache)
        print(f"      data MLE: cache hit {data_cache.name}", flush=True)
    else:
        params_data = get_params_estimation(np.ascontiguousarray(levels_aligned),
                                            dt=DEFAULTS["dt"], n_jobs=a.mle_jobs)
        np.save(data_cache, params_data)
    params_sbts = get_params_estimation(X_mle, dt=DEFAULTS["dt"], n_jobs=a.mle_jobs)
    mle_sec = time.perf_counter() - t0
    np.save(out / f"params_sbts_{a.tag}.npy", params_sbts)

    rows = summarize(params_data, params_sbts)
    fig = plot_figure2(params_data, params_sbts,
                       out / f"figure2_params_kde_{a.tag}.png", gen_label="SBTS")

    print(f"\n{'param':>6} | {'data mean+-std':>22} | {'SBTS mean+-std':>22} | "
          f"{'std ratio':>9} | {'W1':>8}")
    print("-" * 82)
    for r in rows:
        print(f"{r['param']:>6} | {r['data_mean']:>10.4f} +- {r['data_std']:<8.4f} | "
              f"{r['gen_mean']:>10.4f} +- {r['gen_std']:<8.4f} | "
              f"{r['std_ratio']:>9.3f} | {r['w1']:>8.4f}")

    scores = dict(
        method="SBTS", role="baseline comparator for Figure 2",
        config=dict(M=a.M, M_simu=a.M_simu, N=N, d=2, K=a.K, h=a.h,
                    N_pi=a.N_pi, dt=DEFAULTS["dt"], seed=a.seed,
                    source="SBTS reference experiments_demo.ipynb cell 9",
                    rescaling=False),
        timing=dict(gen_min=round(gen_sec / 60, 2), mle_min=round(mle_sec / 60, 2)),
        generated=dict(shape=list(X_sbts.shape), file=gen_name,
                       n_mle_paths=int(len(X_mle)),
                       n_dropped_for_mle=int((~valid).sum())),
        summary=rows,
        paper_reference=(
            "Paper section 5.1: SBTS 'failed to reproduce the vol of vol xi and the "
            "correlation rho', which are 'projected onto an effective average, yielding "
            "a concentrated distribution around the centre of the parameter range', "
            "while kappa, theta, r 'remain identifiable and well recovered'. The "
            "falsifiable statistic is std_ratio << 1 for xi and rho and ~1 for the rest."
        ),
        figure=str(Path(fig).name),
    )
    (out / f"sbts_heston_scores_{a.tag}.json").write_text(json.dumps(scores, indent=2))
    print(f"\n[4/4] wrote {out / f'sbts_heston_scores_{a.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
