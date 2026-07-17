"""
SBTS Univariate Markovian generation — Heston price paths.

Pipeline (SBTS paper §6 Scaling Procedure):
  1. Load price paths S (M, 128)
  2. log-returns R = log(S[:,1:]/S[:,:-1])  shape (M, 127)
  3. global sigma = std(R)  (empirical std over all paths × timesteps)
  4. Scale: R~ = R * sqrt(dt) / sigma        shape (M, 127)
  5. Prepend dummy-0 col → X_train           shape (M, 128)
     (so SBTS generates N=127 intervals → 127 output values)
  6. Generate M_simu paths in parallel workers
  7. Inverse-scale: R_gen = output * sigma / sqrt(dt)   shape (M_simu, 127)
  8. Reconstruct prices: S_gen[:,0]=100, S_gen[:,t+1]=S_gen[:,t]*exp(R_gen[:,t])

Key decisions (SBTS paper Appendix C, Table 4, confirmed 2026-07-17):
  K=1 (Markovian order 1 for Heston)
  h=0.4 (bandwidth, starting point from paper — tuned via CV)
  N_pi=200 (Euler substeps per interval)
  dt=1/250

Parallelisation: Python multiprocessing fork — workers inherit Numba JIT cache.
On this machine: up to 16 physical cores.
"""

import os
import sys
import time
import json
import numpy as np
import numba as nb
from multiprocessing import Pool

# ── Constants ────────────────────────────────────────────────────────────────
DT   = 1.0 / 250.0   # time step
S0   = 100.0          # Heston initial price

# ── Numba-JIT kernel functions (inlined to avoid multiprocessing import-path
# ── issues; functionally identical to reference/models/sbts_uni_markovian.py)

@nb.jit(nopython=True, cache=True)
def _kernel(x, h):
    """Quartic compact-support kernel K_h(x) = (h²-x²)² · 1_{|x|<h}."""
    return np.where(np.abs(x) < h, (h**2 - x**2)**2, 0.0)


@nb.jit(nopython=True, cache=True)
def _simulate_one(N, M, K, X, N_pi, h, deltati):
    """Generate one trajectory in the SCALED space.

    Parameters
    ----------
    N      : int   — number of intervals to simulate (= T-1 = 127)
    M      : int   — number of training paths
    K      : int   — Markovian order (1 for Heston)
    X      : (M, N+1) float64 — training data in scaled space
    N_pi   : int   — Euler substeps per interval
    h      : float — kernel bandwidth
    deltati: float — dt between observations

    Returns
    -------
    (N+1,) array — generated trajectory including initial state
    """
    time_step_Euler = deltati / N_pi
    v_time_step     = np.arange(0.0, deltati + 1e-9, time_step_Euler)
    num_brownian    = N * (len(v_time_step) - 1)
    Brownian        = np.random.normal(0.0, 1.0, num_brownian)

    X_            = X[0, 0]          # start from first training path's initial
    timeSeries    = np.zeros(N + 1)
    timeSeries[0] = X_
    weights       = np.ones(M)
    weights_tilde = np.zeros(M)
    last_K        = np.empty(K, dtype=X.dtype)
    index_queue   = 0
    index_        = 0

    for i in range(N):
        if i > 0:
            if index_queue >= K:
                X_oldest      = last_K[0]
                kernel_oldest = _kernel(X[:, i - K] - X_oldest, h)
                if np.any(kernel_oldest == 0.0):
                    # reset and rebuild from i-K+1
                    weights   = np.ones(M)
                    ind_ref   = i - K
                    for j in range(1, K):
                        weights *= _kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                last_K[:-1] = last_K[1:]
                last_K[-1]  = X_
            else:
                last_K[index_queue] = X_
            index_queue  += 1
            weights[:]   *= _kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1.0 / M    # equal weights at t=0 (dummy column)

        weights_tilde[:] = weights[:] * np.exp(
            (X[:, i + 1] - X_)**2 / 2.0 * deltati
        )

        for k in range(len(v_time_step) - 1):
            timeprev = v_time_step[k]
            timestep = v_time_step[k + 1] - v_time_step[k]
            if k == 0:
                expec_den = np.sum(weights)
                expec_num = np.sum(weights * (X[:, i + 1] - X_))
            else:
                termtoadd  = -(X[:, i + 1] - X_)**2 / (2.0 * (deltati - timeprev))
                termtoadd  = weights_tilde * np.exp(termtoadd)
                expec_den  = np.sum(termtoadd)
                termtoadd *= (X[:, i + 1] - X_)
                expec_num  = np.sum(termtoadd)

            drift = (expec_num / expec_den) / (deltati - timeprev) \
                    if expec_den > 0.0 else 0.0
            X_  += drift * timestep + Brownian[index_] * np.sqrt(timestep)
            index_ += 1

        timeSeries[i + 1] = X_

    return timeSeries


# ── Top-level worker (must be at module scope for multiprocessing pickle) ────

def _worker(args):
    """Generate chunk_size paths; return scaled log-returns (chunk_size, N)."""
    chunk_size, X_train, N, M, K, N_pi, h, deltati, rng_seed = args
    np.random.seed(rng_seed)

    data = np.zeros((chunk_size, X_train.shape[1]), dtype=np.float64)
    for k in range(chunk_size):
        data[k] = _simulate_one(N, M, K, X_train, N_pi, h, deltati)

    return data[:, 1:]   # drop dummy initial column → (chunk_size, N=127)


# ── Public API ───────────────────────────────────────────────────────────────

def scale_log_returns(S):
    """Price paths → scaled log-returns + sigma scalar.

    Parameters
    ----------
    S : (M, T) float64 — price paths

    Returns
    -------
    R_tilde : (M, T-1) float64 — scaled: R~ = R * sqrt(dt) / sigma
    sigma   : float             — empirical std of raw log-returns
    """
    R       = np.log(S[:, 1:] / S[:, :-1])   # (M, T-1)
    sigma   = float(R.std())
    R_tilde = (R * np.sqrt(DT) / sigma).astype(np.float64)
    return R_tilde, sigma


def generate_paths(S_train, M_simu, h,
                   K=1, N_pi=200, n_workers=16, seed=0):
    """Generate M_simu Heston-like price paths using SBTS Markovian.

    Parameters
    ----------
    S_train   : (M, 128) float64 — training price paths (dataset/Heston only)
    M_simu    : int               — paths to generate (8192 for full run)
    h         : float             — kernel bandwidth (paper default: 0.4)
    K         : int               — Markovian order  (paper: 1 for Heston)
    N_pi      : int               — Euler substeps   (paper: 200 for Heston)
    n_workers : int               — CPU workers (hard max 16 on this machine)
    seed      : int               — base random seed

    Returns
    -------
    S_gen : (M_simu, 128) float64 — generated price paths in price space
    meta  : dict                  — generation metadata (saved as metadata.json)
    """
    M, T = S_train.shape
    N    = T - 1          # 127 intervals for T=128

    # 1. Scale training data
    R_tilde, sigma = scale_log_returns(S_train)
    print(f"  sigma(log-returns)={sigma:.6f}  "
          f"sigma(scaled)={float(R_tilde.std()):.6f}", flush=True)

    # 2. Prepend dummy-0 initial column → X_train shape (M, 128)
    X_train = np.hstack([np.zeros((M, 1), dtype=np.float64), R_tilde])

    # 3. Split M_simu across workers
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
        results = pool.map(_worker, args_list)
    elapsed = time.perf_counter() - t0

    # 4. Merge → (M_simu, 127) scaled log-returns
    R_tilde_gen = np.vstack(results)

    # 5. Inverse-scale → raw log-returns
    R_gen = R_tilde_gen * sigma / np.sqrt(DT)   # (M_simu, 127)

    # 6. Reconstruct price paths anchored at S0=100
    S_gen       = np.empty((M_simu, T), dtype=np.float64)
    S_gen[:, 0] = S0
    R_cum       = np.cumsum(R_gen, axis=1)        # cumulative log-returns
    S_gen[:, 1:] = S0 * np.exp(R_cum)

    meta = dict(
        seed=int(seed), M_train=int(M), M_simu=int(M_simu), T=int(T),
        h=float(h), K=int(K), N_pi=int(N_pi), dt=float(DT), sigma=float(sigma),
        n_workers=int(len(args_list)), elapsed_sec=round(elapsed, 2),
        shape=[int(M_simu), int(T)],
        min_val=float(S_gen.min()), max_val=float(S_gen.max()),
    )
    print(f"  Done: {elapsed:.1f}s | "
          f"S_gen min={meta['min_val']:.2f}  max={meta['max_val']:.2f}", flush=True)

    return S_gen, meta


def warmup_jit():
    """Pre-compile Numba JIT with tiny dummy data (avoids cold start in workers)."""
    print("  Warming up Numba JIT...", end=" ", flush=True)
    dummy_X = np.random.randn(5, 4).astype(np.float64)
    _simulate_one(3, 5, 1, dummy_X, 2, 0.4, DT)
    print("done.", flush=True)
