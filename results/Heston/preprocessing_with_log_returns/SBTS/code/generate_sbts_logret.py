"""
SBTS generation driver — preprocessing_with_log_returns folder (4096 dataset).

Mirrors the LS4 worked example: produces per-seed generated price paths in the
layout compute_metrics_logret.py expects.

Two modes:
  native  (default): the SBTS volatility-scaled log-return transform IS the
                     preprocessing (paper §6). Delegates to the untouched
                     methods/SBTS/code/sbts_generate.generate_paths.
  --raw   (baseline_no_preproc): the SBTS kernel run directly on
                     globally-standardized RAW prices — no log-return transform.
                     Same hyperparameters (h=0.05, K=20, N_pi=50).

Author hyperparameters (A. Alouadi, 2026-07-27): h=0.05, K=20, N_pi=50, dt=1/250.
Non-parametric kernel — the 5 "seeds" differ ONLY in generation Brownian RNG;
all share the same 4096 train set. CPU/numba only, no GPU, no training loss.

Run (sbts-venv):
  python generate_sbts_logret.py --seed 0 --workers 64
  python generate_sbts_logret.py --seed 0 --raw --workers 64 \
      --out .../SBTS/baseline_no_preproc
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from datetime import datetime, timezone
from multiprocessing import Pool

# ── Path resolution ──────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))            # .../SBTS/code
SBTS_DIR  = os.path.dirname(HERE)                                  # .../SBTS
# SBTS/code → SBTS → plr → Heston → results → benchmark
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

DATA_DIR = os.path.join(
    BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns"
)
DEFAULT_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")

# import untouched SBTS core
SBTS_SRC = os.path.join(BENCH_ROOT, "methods", "SBTS", "code")
sys.path.insert(0, SBTS_SRC)
from sbts_generate import (            # noqa: E402
    generate_paths, warmup_jit, _simulate_one, DT, S0,
)

# ── Author hyperparameters ───────────────────────────────────────────────────
H_BW, K_MARKOV, N_PI = 0.05, 20, 50


# ── Raw-price baseline worker (module scope for multiprocessing pickle) ───────
def _raw_worker(args):
    """Generate chunk_size paths in standardized-price space.

    Unlike the native worker this KEEPS all T columns (no dummy-col drop):
    X_train is standardized prices with a genuine initial column.
    """
    chunk_size, X_train, N, M, K, N_pi, h, deltati, rng_seed = args
    np.random.seed(rng_seed)
    data = np.zeros((chunk_size, X_train.shape[1]), dtype=np.float64)
    for k in range(chunk_size):
        data[k] = _simulate_one(N, M, K, X_train, N_pi, h, deltati)
    return data                       # (chunk_size, T) — keep all columns


def generate_paths_raw(S_train, M_simu, h, K=K_MARKOV, N_pi=N_PI,
                       n_workers=64, seed=0):
    """Baseline: SBTS kernel on globally-standardized RAW prices (no log-ret).

    x_mu, x_sd = global mean/std over all train prices (frozen).
    X_train = (S - x_mu) / x_sd  (M, T) — genuine initial column, no dummy.
    Kernel simulates T-1 intervals starting from X_train[0,0].
    Inverse: S_gen = X_gen * x_sd + x_mu ; anchor S_gen[:,0] = S0.
    """
    M, T = S_train.shape
    N    = T - 1

    x_mu = float(S_train.mean())
    x_sd = float(S_train.std())
    X_train = ((S_train - x_mu) / x_sd).astype(np.float64)
    print(f"  raw-price standardize: mu={x_mu:.4f} sd={x_sd:.4f}", flush=True)

    base, rem = divmod(M_simu, n_workers)
    chunks    = [base + (1 if i < rem else 0) for i in range(n_workers)]
    chunks    = [c for c in chunks if c > 0]
    args_list = [
        (c, X_train, N, M, K, N_pi, h, DT, seed * 10_000 + i)
        for i, c in enumerate(chunks)
    ]
    print(f"  Launching {len(args_list)} workers "
          f"({chunks[0]} paths each)", flush=True)

    t0 = time.perf_counter()
    with Pool(len(args_list)) as pool:
        results = pool.map(_raw_worker, args_list)
    elapsed = time.perf_counter() - t0

    X_gen = np.vstack(results)                     # (M_simu, T) standardized
    S_gen = X_gen * x_sd + x_mu                     # inverse-standardize
    S_gen[:, 0] = S0                                # anchor initial price

    meta = dict(
        seed=int(seed), M_train=int(M), M_simu=int(M_simu), T=int(T),
        h=float(h), K=int(K), N_pi=int(N_pi), dt=float(DT),
        x_mu=x_mu, x_sd=x_sd,
        n_workers=int(len(args_list)), elapsed_sec=round(elapsed, 2),
        shape=[int(M_simu), int(T)],
        min_val=float(S_gen.min()), max_val=float(S_gen.max()),
    )
    print(f"  Done: {elapsed:.1f}s | S_gen min={meta['min_val']:.2f} "
          f"max={meta['max_val']:.2f}", flush=True)
    return S_gen, meta


# ── Save helpers ─────────────────────────────────────────────────────────────
def _save(out_dir, seed, S_gen, meta, raw):
    N = S_gen.shape[0]
    gp_dir = os.path.join(out_dir, "generated_paths", f"seed_{seed}")
    os.makedirs(gp_dir, exist_ok=True)
    np.save(os.path.join(gp_dir, f"generated_paths_{N}x128.npy"), S_gen)

    gen_has_nan = bool(np.isnan(S_gen).any())
    metadata = dict(
        method="SBTS",
        preproc=("none_raw_price" if raw else "sbts_volscaled_logreturn"),
        seed=int(seed),
        n_train=int(meta["M_train"]),
        n_generated=int(N),
        shape=[int(N), 128],
        hyperparams=dict(h=meta["h"], K=meta["K"], N_pi=meta["N_pi"],
                         dt=meta["dt"]),
        gen_has_nan=gen_has_nan,
        gen_min=meta["min_val"], gen_max=meta["max_val"],
        elapsed_sec=meta["elapsed_sec"],
        date=datetime.now(timezone.utc).isoformat(),
    )
    if raw:
        metadata["raw_standardize"] = dict(x_mu=meta["x_mu"], x_sd=meta["x_sd"])
    else:
        metadata["sigma_logret"] = meta["sigma"]
    with open(os.path.join(gp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # bandwidth record (SBTS has no training loss)
    losses_dir = os.path.join(out_dir, "losses")
    os.makedirs(losses_dir, exist_ok=True)
    with open(os.path.join(losses_dir, f"seed_{seed}_bandwidth.json"), "w") as f:
        json.dump(dict(h=meta["h"], K=meta["K"], N_pi=meta["N_pi"],
                       note="SBTS is non-parametric: no training loss"), f,
                  indent=2)

    # generation-time log
    csv_path = os.path.join(losses_dir, "generation_time.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seed", "mode", "n_generated", "n_workers",
                        "elapsed_sec"])
        w.writerow([seed, "raw" if raw else "native", N,
                    meta["n_workers"], meta["elapsed_sec"]])
    print(f"  saved → {gp_dir}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--gen_num", type=int, default=4096)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SBTS_NWORK", "64")))
    ap.add_argument("--raw", action="store_true",
                    help="baseline_no_preproc: kernel on raw prices")
    ap.add_argument("--out", default=SBTS_DIR,
                    help="output root (native→SBTS/, raw→SBTS/baseline_no_preproc)")
    args = ap.parse_args()

    S_train = np.load(args.data).astype(np.float64)
    print(f"[SBTS gen] mode={'raw' if args.raw else 'native'} "
          f"seed={args.seed} S_train={S_train.shape} "
          f"workers={args.workers} out={args.out}", flush=True)

    warmup_jit()

    if args.raw:
        S_gen, meta = generate_paths_raw(
            S_train, args.gen_num, H_BW, K_MARKOV, N_PI,
            n_workers=args.workers, seed=args.seed)
    else:
        S_gen, meta = generate_paths(
            S_train, args.gen_num, H_BW,
            K=K_MARKOV, N_pi=N_PI, n_workers=args.workers, seed=args.seed)

    _save(args.out, args.seed, S_gen, meta, args.raw)


if __name__ == "__main__":
    main()
