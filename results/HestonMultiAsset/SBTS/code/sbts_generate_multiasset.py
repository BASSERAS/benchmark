"""
SBTS Multivariate Markovian generation — multi-asset Heston price paths (d = 8).

This is the d-dimensional sibling of sbts_generate.py. The univariate file drives
the length-128 single-asset Heston benchmark; this one drives
dataset/HestonMultiAsset (8192 x 252 x 8).

Pipeline (SBTS paper §6 Scaling Procedure, applied per asset)
------------------------------------------------------------
  1. Load price paths S (M, T, d)
  2. log-returns R = log(S[:,1:]/S[:,:-1])           (M, T-1, d)
  3. PER-ASSET sigma_j = std(R[:,:,j])               (d,)
  4. Scale  R~[:,:,j] = R[:,:,j] * sqrt(dt) / sigma_j
  5. Prepend dummy-0 row -> X_train                  (M, T, d)
  6. Generate M_simu paths in parallel workers
  7. Inverse-scale  R_gen[:,:,j] = out[:,:,j] * sigma_j / sqrt(dt)
  8. Reconstruct    S_gen[:,0,j] = 100, S_gen[:,t+1,j] = S_gen[:,t,j]*exp(R_gen)

THREE DELIBERATE DEVIATIONS FROM sbts_generate.py, each forced by d > 1
----------------------------------------------------------------------

(1) PER-ASSET sigma, not one pooled scalar.
    The univariate file uses a single sigma = R.std() over all paths and steps.
    In d = 8 that would leave the eight coordinates on different scales
    (annualised vol ranges 0.155 to 0.261 across our assets), while the
    multivariate kernel is RADIAL -- it measures ||x||_2 and so weights every
    coordinate equally. A pooled sigma would therefore let the high-vol assets
    dominate the conditioning distance and the low-vol assets contribute almost
    nothing. Per-asset scaling is a DIAGONAL linear map, so it rescales the
    marginals without touching the cross-asset correlation structure, and step 7
    inverts it exactly. Cross-asset correlation is what this dataset exists to
    test, so it must survive the transform; it does.

(2) The kernel is radial, K_h(x) = (h^2 - ||x||^2)^2 * 1{||x|| < h}, taken from
    reference/models/sbts_multi_markovian.py. It is NOT a product of d
    univariate kernels.

(3) weights_tilde = weights * exp(+||diff||^2 / (2 dt)).
    The two reference implementations disagree here, upstream, and only one can
    be right:
        sbts_uni_markovian.py  : exp(diff**2 / 2 * deltati)    = diff^2 * dt/2
        sbts_multi_markovian.py: exp(sum(diff**2/(2*deltati))) = diff^2/(2 dt)
    The k = 0 branch of the inner loop uses the raw `weights`, while the k > 0
    branch uses weights_tilde * exp(-||diff||^2 / (2(dt - t_prev))). Continuity
    as t_prev -> 0 requires weights_tilde * exp(-||diff||^2/(2 dt)) == weights,
    i.e. exactly the MULTI form. The uni form looks like an operator-precedence
    slip (`/ 2 * deltati` instead of `/ (2 * deltati)`); at dt = 1/252 it makes
    the correction factor exp(8e-6) ~ 1 instead of exp(0.5) ~ 1.65, so it
    silently disables the bridge correction. We use the multi (correct) form.
    NOTE: methods/SBTS/code/sbts_generate.py copies the uni reference verbatim
    and is left untouched -- the committed 1-D benchmark numbers are not ours to
    silently restate here.

BANDWIDTH: RECALIBRATED FOR d = 8, NOT AUTHOR-SPECIFIED
-------------------------------------------------------
h = 0.05 was confirmed by the SBTS author for the d = 1, T = 128 benchmark.
It CANNOT be carried over. The kernel has compact support, and the median
pairwise distance between scaled training rows grows like sqrt(d):

    d = 1: median ||x - x'|| = 0.037     d = 8: median = 0.264

At h = 0.05 only 0.02% of the 8192 training rows fall inside the support for a
SINGLE lag; the weight is a rolling product over K lags, so the product is zero
almost surely. Zero weights => expec_den = 0 => drift = 0 => the generator
emits pure Brownian motion while reporting success. That is a silent failure,
so h and K are calibrated here by the effective sample size

    ESS = (sum_i w_i)^2 / sum_i w_i^2

measured on the REAL recursion (see ess_diagnostic below), targeting a band that
is neither collapsed (ESS ~ 1 => the "conditional expectation" is one memorised
training path) nor saturated (ESS ~ M => the kernel conditions on nothing and
the drift is the unconditional mean).

Parallelisation: multiprocessing fork; workers inherit the Numba JIT cache.
Hard cap 16 physical cores on this machine.
"""

import time

import numba as nb
import numpy as np
from multiprocessing import Pool

# ── Constants ────────────────────────────────────────────────────────────────
DT = 1.0 / 252.0     # matches dataset/HestonMultiAsset (252 daily steps)
S0 = 100.0           # all assets start here


# ── Numba-JIT kernels (inlined; mirror reference/models/sbts_multi_markovian.py)

@nb.jit(nopython=True, cache=True)
def _kernel(x, h):
    """Radial quartic compact-support kernel.

    K_h(x) = (h^2 - ||x||^2)^2 * 1{||x||_2 < h}

    Parameters
    ----------
    x : (M, d) float64 — difference matrix
    h : float          — bandwidth

    Returns
    -------
    (M,) float64
    """
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    return np.where(x_norm < h, (h ** 2 - x_norm ** 2) ** 2, 0.0)


@nb.jit(nopython=True, cache=True)
def _simulate_one(N, M, d, K, X, N_pi, h, deltati):
    """Generate one trajectory in the SCALED space.

    Parameters
    ----------
    N       : int — intervals to simulate (= T-1)
    M       : int — number of training paths
    d       : int — asset dimension
    K       : int — Markovian order
    X       : (M, N+1, d) float64 — training data, scaled space
    N_pi    : int   — Euler substeps per interval
    h       : float — kernel bandwidth
    deltati : float — dt between observations

    Returns
    -------
    (N+1, d) float64 — trajectory including the initial state
    """
    time_step_Euler = deltati / N_pi
    v_time_step = np.arange(0.0, deltati + 1e-9, time_step_Euler)
    num_brownian = N * (len(v_time_step) - 1)
    Brownian = np.random.normal(0.0, 1.0, (num_brownian, d))

    X_ = X[0, 0].copy()
    timeSeries = np.zeros((N + 1, d))
    timeSeries[0] = X_
    weights = np.ones(M)
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0
    index_ = 0

    for i in range(N):
        if i > 0:
            if index_queue >= K:
                X_oldest = last_K[0]
                kernel_oldest = _kernel(X[:, i - K] - X_oldest, h)
                if np.any(kernel_oldest == 0.0):
                    # cannot divide out a zero factor: rebuild from i-K+1
                    weights = np.ones(M)
                    ind_ref = i - K
                    for j in range(1, K):
                        weights *= _kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                last_K[:-1] = last_K[1:]
                last_K[-1] = X_
            else:
                last_K[index_queue] = X_
            index_queue += 1
            weights *= _kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1.0 / M          # dummy column: equal weights

        diff = X[:, i + 1] - X_
        # (3) correct bridge correction — see module docstring
        weights_tilde = weights * np.exp(np.sum(diff ** 2 / (2.0 * deltati), axis=1))

        for k in range(len(v_time_step) - 1):
            timeprev = v_time_step[k]
            timestep = v_time_step[k + 1] - v_time_step[k]

            if k == 0:
                expec_den = np.sum(weights)
                numerator = np.sum(
                    weights.reshape(M, 1) * (X[:, i + 1] - X_), axis=0
                )
            else:
                diff2 = X[:, i + 1] - X_
                termtoadd = np.sum(diff2 ** 2, axis=1)
                termtoadd = weights_tilde * np.exp(
                    -termtoadd / (2.0 * (deltati - timeprev))
                )
                expec_den = np.sum(termtoadd)
                numerator = np.sum(
                    termtoadd.reshape(M, 1) * (X[:, i + 1] - X_), axis=0
                )

            if expec_den > 0.0:
                drift = (numerator / expec_den) / (deltati - timeprev)
            else:
                drift = np.zeros(d)

            X_ = X_ + drift * timestep + Brownian[index_] * np.sqrt(timestep)
            index_ += 1

        timeSeries[i + 1, :] = X_

    return timeSeries


@nb.jit(nopython=True, cache=True)
def _simulate_one_diag(N, M, d, K, X, N_pi, h, deltati):
    """Same recursion as _simulate_one, but returns weight diagnostics only.

    Returns
    -------
    ess    : (N,) float64 — effective sample size (sum w)^2 / sum w^2 per step
    n_zero : int          — steps where sum(w) == 0 (drift forced to 0)
    """
    time_step_Euler = deltati / N_pi
    v_time_step = np.arange(0.0, deltati + 1e-9, time_step_Euler)
    num_brownian = N * (len(v_time_step) - 1)
    Brownian = np.random.normal(0.0, 1.0, (num_brownian, d))

    X_ = X[0, 0].copy()
    weights = np.ones(M)
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0
    index_ = 0
    ess = np.zeros(N)
    n_zero = 0

    for i in range(N):
        if i > 0:
            if index_queue >= K:
                X_oldest = last_K[0]
                kernel_oldest = _kernel(X[:, i - K] - X_oldest, h)
                if np.any(kernel_oldest == 0.0):
                    weights = np.ones(M)
                    ind_ref = i - K
                    for j in range(1, K):
                        weights *= _kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                last_K[:-1] = last_K[1:]
                last_K[-1] = X_
            else:
                last_K[index_queue] = X_
            index_queue += 1
            weights *= _kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1.0 / M

        s1 = np.sum(weights)
        s2 = np.sum(weights ** 2)
        if s1 <= 0.0 or s2 <= 0.0:
            ess[i] = 0.0
            n_zero += 1
        else:
            ess[i] = s1 * s1 / s2

        diff = X[:, i + 1] - X_
        weights_tilde = weights * np.exp(np.sum(diff ** 2 / (2.0 * deltati), axis=1))

        for k in range(len(v_time_step) - 1):
            timeprev = v_time_step[k]
            timestep = v_time_step[k + 1] - v_time_step[k]
            if k == 0:
                expec_den = np.sum(weights)
                numerator = np.sum(weights.reshape(M, 1) * (X[:, i + 1] - X_), axis=0)
            else:
                diff2 = X[:, i + 1] - X_
                termtoadd = np.sum(diff2 ** 2, axis=1)
                termtoadd = weights_tilde * np.exp(
                    -termtoadd / (2.0 * (deltati - timeprev))
                )
                expec_den = np.sum(termtoadd)
                numerator = np.sum(termtoadd.reshape(M, 1) * (X[:, i + 1] - X_), axis=0)

            if expec_den > 0.0:
                drift = (numerator / expec_den) / (deltati - timeprev)
            else:
                drift = np.zeros(d)
            X_ = X_ + drift * timestep + Brownian[index_] * np.sqrt(timestep)
            index_ += 1

    return ess, n_zero


# ── Top-level worker (module scope, required for multiprocessing pickle) ─────

def _worker(args):
    """Generate chunk_size paths; return scaled increments (chunk, N, d)."""
    chunk_size, X_train, N, M, d, K, N_pi, h, deltati, rng_seed = args
    np.random.seed(rng_seed)

    data = np.zeros((chunk_size, X_train.shape[1], d), dtype=np.float64)
    for k in range(chunk_size):
        data[k] = _simulate_one(N, M, d, K, X_train, N_pi, h, deltati)

    return data[:, 1:, :]          # drop dummy initial row -> (chunk, N, d)


# ── Public API ───────────────────────────────────────────────────────────────

def scale_log_returns(S):
    """Price paths -> per-asset scaled log-returns.

    Parameters
    ----------
    S : (M, T, d) float64

    Returns
    -------
    R_tilde : (M, T-1, d) float64 — R * sqrt(dt) / sigma_j, per asset j
    sigma   : (d,) float64        — per-asset std of raw log-returns
    """
    R = np.log(S[:, 1:, :] / S[:, :-1, :])           # (M, T-1, d)
    sigma = R.std(axis=(0, 1))                       # (d,)
    R_tilde = (R * np.sqrt(DT) / sigma).astype(np.float64)
    return R_tilde, sigma


def build_training_tensor(S_train):
    """S (M,T,d) -> X (M,T,d) scaled with the dummy-0 initial row prepended."""
    R_tilde, sigma = scale_log_returns(S_train)
    M, _, d = R_tilde.shape
    X = np.concatenate(
        [np.zeros((M, 1, d), dtype=np.float64), R_tilde], axis=1
    )
    return np.ascontiguousarray(X), sigma


def ess_diagnostic(X_train, h, K, N_pi=50, n_rep=8, seed=0, deltati=DT):
    """Measure kernel health for a candidate (h, K) on the real recursion.

    Returns a dict with the ESS distribution and the zero-denominator rate.
    An ESS near 1 means the conditional expectation is a single memorised
    training path; an ESS near M means the kernel is conditioning on nothing.
    A zero rate above ~0 means the generator is emitting driftless Brownian
    motion for those steps.
    """
    np.random.seed(seed)
    M, T, d = X_train.shape
    N = T - 1
    all_ess, zeros = [], 0
    for _ in range(n_rep):
        ess, n_zero = _simulate_one_diag(N, M, d, K, X_train, N_pi, h, deltati)
        all_ess.append(ess)
        zeros += n_zero
    ess = np.concatenate(all_ess)
    nz = ess[ess > 0]
    return dict(
        h=float(h), K=int(K), M=int(M),
        zero_frac=float(zeros / (n_rep * N)),
        ess_median=float(np.median(nz)) if nz.size else 0.0,
        ess_p05=float(np.percentile(nz, 5)) if nz.size else 0.0,
        ess_p95=float(np.percentile(nz, 95)) if nz.size else 0.0,
        ess_frac_median=float(np.median(nz) / M) if nz.size else 0.0,
    )


def generate_paths(S_train, M_simu, h, K=20, N_pi=50, n_workers=16, seed=0):
    """Generate M_simu multi-asset price paths with SBTS multivariate Markovian.

    Parameters
    ----------
    S_train   : (M, T, d) float64 — training price paths
    M_simu    : int   — paths to generate
    h         : float — kernel bandwidth (recalibrated for d; see docstring)
    K         : int   — Markovian order
    N_pi      : int   — Euler substeps per interval
    n_workers : int   — CPU workers (hard max 16 on this machine)
    seed      : int   — base random seed

    Returns
    -------
    S_gen : (M_simu, T, d) float64 — price paths anchored at S0
    meta  : dict
    """
    M, T, d = S_train.shape
    N = T - 1

    X_train, sigma = build_training_tensor(S_train)
    print(f"  sigma per asset = {np.array2string(sigma, precision=5)}", flush=True)
    print(f"  scaled std = {float(X_train[:, 1:].std()):.6f} "
          f"(target ~ sqrt(dt) = {np.sqrt(DT):.6f})", flush=True)

    base, rem = divmod(M_simu, n_workers)
    chunks = [base + (1 if i < rem else 0) for i in range(n_workers)]
    chunks = [c for c in chunks if c > 0]

    args_list = [
        (c, X_train, N, M, d, K, N_pi, h, DT, seed * 10_000 + i)
        for i, c in enumerate(chunks)
    ]
    print(f"  Launching {len(args_list)} workers ({chunks[0]} paths each)", flush=True)

    t0 = time.perf_counter()
    with Pool(len(args_list)) as pool:
        results = pool.map(_worker, args_list)
    elapsed = time.perf_counter() - t0

    R_tilde_gen = np.vstack(results)                  # (M_simu, N, d)
    R_gen = R_tilde_gen * sigma / np.sqrt(DT)         # inverse per-asset scaling

    S_gen = np.empty((M_simu, T, d), dtype=np.float64)
    S_gen[:, 0, :] = S0
    S_gen[:, 1:, :] = S0 * np.exp(np.cumsum(R_gen, axis=1))

    meta = dict(
        seed=int(seed), M_train=int(M), M_simu=int(M_simu), T=int(T), d=int(d),
        h=float(h), K=int(K), N_pi=int(N_pi), dt=float(DT),
        sigma=[float(s) for s in sigma],
        scaling="per_asset",
        n_workers=int(len(args_list)), elapsed_sec=round(elapsed, 2),
        shape=[int(M_simu), int(T), int(d)],
        min_val=float(S_gen.min()), max_val=float(S_gen.max()),
    )
    print(f"  Done: {elapsed:.1f}s | S_gen min={meta['min_val']:.2f} "
          f"max={meta['max_val']:.2f}", flush=True)
    return S_gen, meta


def warmup_jit(d=8):
    """Pre-compile the Numba kernels so workers do not pay a cold start."""
    print("  Warming up Numba JIT...", end=" ", flush=True)
    dummy = np.random.randn(5, 4, d).astype(np.float64)
    _simulate_one(3, 5, d, 1, dummy, 2, 1.0, DT)
    _simulate_one_diag(3, 5, d, 1, dummy, 2, 1.0, DT)
    print("done.", flush=True)
