"""
Metrics A1–A12 and A15 for time-series generation benchmarking.

A1   Full joint-path MMD²
A2   Terminal MMD²
A3   Increment MMD²
A4   Volatility-discrepancy MMD
A5   Terminal Sliced Wasserstein Distance
A6   Path Sliced Wasserstein Distance
A7   Terminal Covariance Error
A8   Terminal Mean RMSE
A9   Return Std Error
A10  Return Kurtosis Error
A11  ACF Error (absolute returns)
A12  ACF Error (squared returns)
A15  Teacher-Sigma Correlation + RMSE  (Heston-specific)

A13 and A14 are implemented separately in discriminative_score.py
and predictive_score.py using PyTorch.
"""

import numpy as np
from scipy.stats import wasserstein_distance, kurtosis
from typing import Callable, Tuple


# ======================================================================
# Kernel helpers
# ======================================================================

def rbf_multiscale_kernel(
    u: np.ndarray, v: np.ndarray,
    scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> np.ndarray:
    r"""Multi-scale RBF (Gaussian) kernel.

    k(u, v) = (1/L) * sum_l exp(-||u-v||^2 / (2 * h_l^2 * d))

    Parameters
    ----------
    u : ndarray (N_u, d)
    v : ndarray (N_v, d)
    scales : bandwidths h_l
    """
    d = u.shape[-1]
    diff2 = np.sum((u[:, None, :] - v[None, :, :]) ** 2, axis=-1)
    result = np.zeros_like(diff2)
    for h in scales:
        result += np.exp(-diff2 / (2.0 * h ** 2 * d))
    return result / len(scales)


# ======================================================================
# A1–A4  MMD-based metrics
# ======================================================================

def mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A1. Full joint-path MMD².

    MMD²(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

    Parameters
    ----------
    X : ndarray (N_x, T, d)  — real paths
    Y : ndarray (N_y, T, d)  — generated paths
    """
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)
    kxx = kernel(X_flat, X_flat)
    kyy = kernel(Y_flat, Y_flat)
    kxy = kernel(X_flat, Y_flat)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def terminal_mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A2. Terminal MMD² — applied only to the last time step t=T."""
    return mmd2(X[:, -1:, :], Y[:, -1:, :], kernel)


def increment_mmd2(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A3. Increment MMD² on first differences dX = X_{t+1} - X_t."""
    return mmd2(np.diff(X, axis=1), np.diff(Y, axis=1), kernel)


def volatility_mmd(
    X: np.ndarray, Y: np.ndarray,
    kernel: Callable = rbf_multiscale_kernel,
) -> float:
    """A4. Volatility-discrepancy MMD.

    Weighted sum of MMD² over multiple stylised feature views:
    instantaneous RV, state-RV pairs, global RV mean, terminal return,
    returns, rolling-vol (window=5), absolute returns, squared returns,
    ACF lag-products of |dX| and dX² for lags {1,2,5,10}.
    """
    dX = np.diff(X, axis=1)
    dY = np.diff(Y, axis=1)

    features_X, features_Y = [], []

    # 1. Instantaneous realised variance
    rv_X = dX ** 2
    rv_Y = dY ** 2
    features_X.append(rv_X.reshape(rv_X.shape[0], -1))
    features_Y.append(rv_Y.reshape(rv_Y.shape[0], -1))

    # 2. State-RV pairs (X_{t+1}, RV_t)
    state_rv_X = np.concatenate([X[:, 1:, :], rv_X], axis=-1)
    state_rv_Y = np.concatenate([Y[:, 1:, :], rv_Y], axis=-1)
    features_X.append(state_rv_X.reshape(state_rv_X.shape[0], -1))
    features_Y.append(state_rv_Y.reshape(state_rv_Y.shape[0], -1))

    # 3. Global RV mean per sample
    features_X.append(np.mean(rv_X, axis=1))
    features_Y.append(np.mean(rv_Y, axis=1))

    # 4. Terminal return X_T - X_0
    features_X.append(X[:, -1, :] - X[:, 0, :])
    features_Y.append(Y[:, -1, :] - Y[:, 0, :])

    # 5. Returns dX
    features_X.append(dX.reshape(dX.shape[0], -1))
    features_Y.append(dY.reshape(dY.shape[0], -1))

    # 6. Rolling volatility (window=5)
    def rolling_vol(x_sq, window=5):
        x_pad = np.pad(x_sq, ((0,0),(window-1,0),(0,0)), mode="edge")
        out = np.zeros_like(x_sq)
        for i in range(x_sq.shape[1]):
            out[:,i,:] = np.mean(x_pad[:,i:i+window,:], axis=1)
        return np.sqrt(out**2 + 1e-6)

    rvol_X = rolling_vol(dX**2)
    rvol_Y = rolling_vol(dY**2)
    features_X.append(rvol_X.reshape(rvol_X.shape[0], -1))
    features_Y.append(rvol_Y.reshape(rvol_Y.shape[0], -1))

    # 7. Absolute returns
    features_X.append(np.abs(dX).reshape(dX.shape[0], -1))
    features_Y.append(np.abs(dY).reshape(dY.shape[0], -1))

    # 8. Squared returns
    features_X.append((dX**2).reshape(dX.shape[0], -1))
    features_Y.append((dY**2).reshape(dY.shape[0], -1))

    # 9. ACF lag-products
    for lag in [1, 2, 5, 10]:
        if lag >= dX.shape[1]:
            continue
        for sq in [False, True]:
            series_X = np.abs(dX) if not sq else dX**2
            series_Y = np.abs(dY) if not sq else dY**2
            acf_vals_X = np.array([
                acf(series_X[i,:,f], lag)
                for i in range(series_X.shape[0])
                for f in range(series_X.shape[2])
            ])
            acf_vals_Y = np.array([
                acf(series_Y[i,:,f], lag)
                for i in range(series_Y.shape[0])
                for f in range(series_Y.shape[2])
            ])
            features_X.append(acf_vals_X.reshape(-1, 1))
            features_Y.append(acf_vals_Y.reshape(-1, 1))

    total = 0.0
    for fx, fy in zip(features_X, features_Y):
        total += mmd2(fx, fy, kernel)
    return total


# ======================================================================
# A5–A6  Sliced Wasserstein Distance
# ======================================================================

def slicing_wasserstein(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """Sliced Wasserstein Distance between two empirical distributions.

    Parameters
    ----------
    X : ndarray (N_x, d)
    Y : ndarray (N_y, d)
    n_proj : number of random projections
    """
    rng = np.random.default_rng(seed)
    d = X.shape[-1]
    dists = []
    for _ in range(n_proj):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u) + 1e-16
        dists.append(wasserstein_distance(X @ u, Y @ u))
    return float(np.mean(dists))


def terminal_swd(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """A5. Terminal SWD — at t=T (last time step)."""
    return slicing_wasserstein(X[:, -1, :], Y[:, -1, :], n_proj, seed)


def path_swd(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 50, seed: int = 0,
) -> float:
    """A6. Path SWD — mean of terminal SWD computed at every time step."""
    T = X.shape[1]
    return float(np.mean([
        slicing_wasserstein(X[:, t, :], Y[:, t, :], n_proj, seed)
        for t in range(T)
    ]))


# ======================================================================
# A7–A12  Distributional moment metrics
# ======================================================================

def terminal_cov_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A7. Frobenius norm of covariance matrix difference at t=T."""
    cov_X = np.atleast_2d(np.cov(X[:, -1, :].T))
    cov_Y = np.atleast_2d(np.cov(Y[:, -1, :].T))
    return float(np.linalg.norm(cov_X - cov_Y, ord="fro"))


def terminal_mean_rmse(X: np.ndarray, Y: np.ndarray) -> float:
    """A8. RMSE between mean vectors at t=T."""
    return float(np.linalg.norm(X[:, -1, :].mean(0) - Y[:, -1, :].mean(0)))


def return_std_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A9. Absolute difference of return standard deviations."""
    return float(abs(np.std(np.diff(X, axis=1)) - np.std(np.diff(Y, axis=1))))


def return_kurtosis_error(X: np.ndarray, Y: np.ndarray) -> float:
    """A10. Absolute difference of excess kurtosis of returns (Fisher, unbiased)."""
    dX = np.diff(X, axis=1).ravel()
    dY = np.diff(Y, axis=1).ravel()
    kx = kurtosis(dX, fisher=True, bias=False)
    ky = kurtosis(dY, fisher=True, bias=False)
    if np.isnan(kx) or np.isnan(ky):
        return 0.0
    return float(abs(kx - ky))


def acf(q: np.ndarray, lag: int) -> float:
    """Auto-correlation at a given lag.

    Parameters
    ----------
    q   : 1-D array of length T
    lag : positive integer
    """
    q = q - q.mean()
    denom = np.sum(q**2)
    if denom < 1e-16:
        return 0.0
    return float(np.sum(q[:-lag] * q[lag:]) / denom)


def acf_error(
    q_gen: np.ndarray, q_real: np.ndarray,
    lags: Tuple[int, ...] = (1, 2, 5, 10),
) -> float:
    """A11 / A12. Mean absolute ACF error over multiple lags.

    Pass absolute returns |dX| for A11, squared returns dX² for A12.

    Parameters
    ----------
    q_gen  : ndarray (N, T, d) or (T,)  — generated series
    q_real : ndarray (N, T, d) or (T,)  — real series
    lags   : lags at which ACF is evaluated
    """
    if q_gen.ndim == 3 and q_real.ndim == 3:
        errors = []
        for lag in lags:
            if lag >= q_real.shape[1]:
                continue
            real_acfs = np.array([
                acf(q_real[i, :, f], lag)
                for i in range(q_real.shape[0])
                for f in range(q_real.shape[2])
            ])
            gen_acfs = np.array([
                acf(q_gen[i, :, f], lag)
                for i in range(q_gen.shape[0])
                for f in range(q_gen.shape[2])
            ])
            errors.append(abs(np.mean(real_acfs) - np.mean(gen_acfs)))
        return float(np.mean(errors)) if errors else 0.0
    # 1-D fallback
    if np.std(q_real) < 1e-16 or np.std(q_gen) < 1e-16:
        return 0.0
    return float(np.mean([abs(acf(q_gen, lag) - acf(q_real, lag)) for lag in lags]))


# ======================================================================
# A15  Teacher-Sigma metrics  (Heston-specific)
# ======================================================================

def _rolling_mean_std_5(dX_sq: np.ndarray) -> np.ndarray:
    """Rolling window-5 mean of squared increments, returned as vol."""
    out = np.zeros_like(dX_sq)
    cum = np.cumsum(dX_sq, axis=1)
    for i in range(dX_sq.shape[1]):
        start = max(0, i - 4)
        w = i - start + 1
        out[:, i, :] = (cum[:, i, :] - (cum[:, start-1, :] if start > 0 else 0)) / w
    return np.sqrt(out + 1e-6)


def teacher_sigma_metrics(
    X_gen: np.ndarray, v_true: np.ndarray,
) -> Tuple[float, float]:
    """A15. Teacher-Sigma Correlation and RMSE (Heston-specific).

    Estimates latent vol as rolling window-5 std of returns and compares
    it against the true latent vol sqrt(v_true).

    Parameters
    ----------
    X_gen  : ndarray (N, T, d)  — generated price paths
    v_true : ndarray (N, T)     — true latent variance paths

    Returns
    -------
    corr : float  — Pearson correlation (perfect = 1, higher is better)
    rmse : float  — RMSE (perfect = 0)
    """
    dX = np.diff(X_gen, axis=1)                    # (N, T-1, d)
    sigma_hat = _rolling_mean_std_5(dX**2)          # (N, T-1, d)

    if v_true.shape[1] > dX.shape[1]:
        v_sqrt = np.sqrt(np.maximum(v_true[:, 1:], 0.0))
    else:
        v_sqrt = np.sqrt(np.maximum(v_true, 0.0))

    sh = sigma_hat.ravel()
    vs = (v_sqrt[:, :, None] if v_sqrt.ndim == 2 else v_sqrt).ravel()
    n = min(len(sh), len(vs))
    sh, vs = sh[:n], vs[:n]

    corr = np.corrcoef(sh, vs)[0, 1]
    rmse = np.sqrt(np.mean((sh - vs)**2))
    return (float(corr) if not np.isnan(corr) else 0.0), float(rmse)


# ======================================================================
# A16  Tail Survival Error
# ======================================================================

def tail_survival_error(
    X: np.ndarray,
    Y: np.ndarray,
    quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> float:
    """A16. Tail Survival Error — RMS of survival probability difference.

    For each quantile alpha in {0.90, 0.95, 0.99}:
      - q_alpha = alpha-quantile of real |returns|
      - real_surv(alpha) = P_real(|r| > q_alpha)   (by definition ~= 1-alpha)
      - fake_surv(alpha) = P_gen (|r| > q_alpha)

    Score = sqrt( mean_alpha( (real_surv - fake_surv)^2 ) )

    Tests whether the generator reproduces the fat tail of the return
    distribution at the 90th, 95th, and 99th percentile levels.
    Perfect: 0. Direction: lower is better.

    Parameters
    ----------
    X : ndarray (N, T, d)  — real paths
    Y : ndarray (N, T, d)  — generated paths
    quantiles : tail quantile levels (default: 0.90, 0.95, 0.99)
    """
    real_abs_r = np.abs(np.diff(X, axis=1)).ravel()
    fake_abs_r = np.abs(np.diff(Y, axis=1)).ravel()
    thresholds = np.quantile(real_abs_r, quantiles)
    real_surv = np.array([(real_abs_r > t).mean() for t in thresholds])
    fake_surv = np.array([(fake_abs_r > t).mean() for t in thresholds])
    return float(np.sqrt(np.mean((real_surv - fake_surv) ** 2)))


__all__ = [
    "rbf_multiscale_kernel",
    "mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd",
    "terminal_cov_error", "terminal_mean_rmse",
    "return_std_error", "return_kurtosis_error",
    "acf", "acf_error",
    "teacher_sigma_metrics",
    "tail_survival_error",
]
