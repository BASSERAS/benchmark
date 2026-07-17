"""
Metrics A1–A20 for time-series generation benchmarking.

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
A16  Tail Survival Error (RMS + per-quantile q90/q95/q99)
A17  Oracle Mean
A18  Agent Mean
A19  Oracle-Agent Correlation
A20  Realized Volatility Law Loss

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
        return np.sqrt(np.maximum(out, 0.0) + 1e-6)

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
    dt: float = 1.0 / 250.0,
) -> Tuple[float, float]:
    """A15. Teacher-Sigma Correlation and RMSE (Heston-specific).

    Estimates instantaneous vol as rolling window-5 QV of log-returns
    (annualised by 1/dt) and compares against true latent vol sqrt(v_true).

    Correct formula: sigma_hat = sqrt( rolling_mean(r_t^2) / dt )
    where r_t = log(S_{t+1}/S_t) are log-returns.
    This gives sigma_hat ~ sqrt(v_t), same units as sqrt(v_true).

    IMPORTANT: do NOT use price increments (np.diff on prices) — they produce
    a scale mismatch of factor ~S_t * sqrt(dt) ~ 6x, ruining the metric.
    Always use log-returns normalised by dt.

    Parameters
    ----------
    X_gen  : ndarray (N, T, d)  — generated price paths (must be positive)
    v_true : ndarray (N, T)     — true latent variance paths (annualised)
    dt     : float              — time step in years (default 1/250)

    Returns
    -------
    corr : float  — Pearson correlation (perfect = 1, higher is better)
    rmse : float  — RMSE (perfect = 0)
    """
    # Log-returns: r_t = log(S_{t+1}/S_t),  shape (N, T-1, d)
    log_r = np.diff(np.log(np.maximum(X_gen, 1e-10)), axis=1)
    # _rolling_mean_std_5 returns sqrt(rolling_mean(input)).
    # Passing r_t^2/dt makes it return sqrt(rolling_mean(r_t^2)/dt) ~ sqrt(v_t).
    sigma_hat = _rolling_mean_std_5(log_r ** 2 / dt)   # (N, T-1, d), units: sqrt(annualised var)

    if v_true.shape[1] > log_r.shape[1]:
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
) -> Tuple[float, float, float, float]:
    """A16. Tail Survival Error — RMS of survival probability difference.

    For each quantile alpha in {0.90, 0.95, 0.99}:
      - q_alpha = alpha-quantile of real |returns|
      - real_surv(alpha) = P_real(|r| > q_alpha)   (by definition ~= 1-alpha)
      - fake_surv(alpha) = P_gen (|r| > q_alpha)

    Returns (rms, q90, q95, q99) where:
      - rms = sqrt( mean_alpha( (real_surv - fake_surv)^2 ) )
      - qXX = abs(real_surv - fake_surv) at that quantile level

    Tests whether the generator reproduces the fat tail of the return
    distribution at the 90th, 95th, and 99th percentile levels.
    Perfect: (0, 0, 0, 0). Direction: lower is better.

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
    rms = float(np.sqrt(np.mean((real_surv - fake_surv) ** 2)))
    q90 = float(abs(real_surv[0] - fake_surv[0]))
    q95 = float(abs(real_surv[1] - fake_surv[1]))
    q99 = float(abs(real_surv[2] - fake_surv[2]))
    return rms, q90, q95, q99


# ======================================================================
# A20  Realized Volatility Law Loss
# ======================================================================

def rv_law_loss(X: np.ndarray, Y: np.ndarray, dt: float = 1.0/250.0) -> float:
    """A20. Realized Volatility Law Loss — Wasserstein-1 distance between RV distributions.

    For each path i:
      RV_i = (1/dt) * sum_t r_{i,t}^2   (annualized realized variance)

    where r_{i,t} = log(S_{i,t+1}/S_{i,t}) are log-returns.
    Returns W_1(distribution of RV_real, distribution of RV_gen).
    Perfect: 0. Direction: lower is better.

    Parameters
    ----------
    X : ndarray (N, T, d)  — real paths
    Y : ndarray (N, T, d)  — generated paths
    dt : float  — time step in years (default 1/250)
    """
    log_r_real = np.diff(np.log(np.maximum(X, 1e-10)), axis=1)  # (N, T-1, d)
    log_r_gen  = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)
    rv_real = np.sum(log_r_real ** 2, axis=1) / dt  # (N, d)
    rv_gen  = np.sum(log_r_gen ** 2, axis=1) / dt
    rv_r = rv_real.ravel()
    rv_g = rv_gen.ravel()
    return float(wasserstein_distance(rv_r, rv_g))


# ======================================================================
# A17–A19  Oracle evaluation protocol
# ======================================================================

def oracle_metrics(
    X_real: np.ndarray, X_gen: np.ndarray,
    ar_order: int = 5, test_frac: float = 0.2, seed: int = 0,
) -> Tuple[float, float, float]:
    """A17-A19. Oracle evaluation protocol (per-path AR, no cross-path leakage).

    For each path independently, build AR(p) (lag features, target) pairs.
    Stack all paths' (X_lag, y) into a single matrix, then randomly split
    into train/test sets (shuffled, not temporal — paths are i.i.d. MC draws).

    Train an AR(p) on REAL log-returns -> predict test -> oracle mean.
    Train same AR(p) on GENERATED log-returns -> predict test -> agent mean.
    Compute Pearson correlation between oracle and agent predictions.

    Parameters
    ----------
    X_real : (N, T, d) real paths
    X_gen  : (N, T, d) generated paths
    ar_order : order of AR model (default 5)
    test_frac : fraction of AR transitions held out for testing (default 0.2)
    seed : random seed for reproducibility (shuffle)

    Returns
    -------
    oracle_mean : mean of oracle predictions on test set
    agent_mean  : mean of agent predictions on test set
    correlation : Pearson r between oracle and agent predictions
    """
    rng = np.random.default_rng(seed)

    # Log-returns: r_t = log(S_{t+1}/S_t)
    r_real = np.diff(np.log(np.maximum(X_real, 1e-10)), axis=1)  # (N, T-1, d)
    r_gen  = np.diff(np.log(np.maximum(X_gen,  1e-10)), axis=1)

    # Build per-path AR(p) matrices, then STACK (no cross-path lags)
    def make_ar_stacked(data, order):
        """data: (N, T, d) -> X_lag: (M, order), y: (M,) where M = N*(T-order)*d"""
        X_list, y_list = [], []
        for n in range(data.shape[0]):
            for f in range(data.shape[2]):
                seq = data[n, :, f]
                for t in range(order, len(seq)):
                    X_list.append(seq[t-order:t])
                    y_list.append(seq[t])
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    X_real_all, y_real_all = make_ar_stacked(r_real, ar_order)

    # Guard: AR(p) order must be < available transitions
    if len(X_real_all) < ar_order + 2:
        import warnings as _w
        _w.warn(f"oracle_metrics: ar_order={ar_order} too large for data (T={X_real.shape[1]}, "
                f"N={X_real.shape[0]}). Returning (0, 0, 0).")
        return (0.0, 0.0, 0.0)

    # Shuffled train/test split (paths are i.i.d., no temporal ordering)
    n_total = len(X_real_all)
    n_test = max(1, int(n_total * test_frac))
    idx = rng.permutation(n_total)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]

    # Oracle: train OLS on real data, predict on held-out real features
    X_train_r = X_real_all[idx_train]
    y_train_r = y_real_all[idx_train]
    X_test = X_real_all[idx_test]

    from numpy.linalg import lstsq
    coeff_r = lstsq(X_train_r, y_train_r, rcond=1e-8)[0]
    oracle_pred = X_test @ coeff_r
    oracle_mean = float(np.mean(oracle_pred))

    # Agent: train OLS on generated data, predict on same test features
    X_gen_all, y_gen_all = make_ar_stacked(r_gen, ar_order)
    X_train_g = X_gen_all[idx_train]
    y_train_g = y_gen_all[idx_train]
    coeff_g = lstsq(X_train_g, y_train_g, rcond=1e-8)[0]
    agent_pred = X_test @ coeff_g
    agent_mean = float(np.mean(agent_pred))

    # Pearson correlation between oracle and agent predictions
    with np.errstate(invalid='ignore'):
        corr = float(np.corrcoef(oracle_pred, agent_pred)[0, 1]) if len(oracle_pred) > 2 else 0.0
    if np.isnan(corr):
        import warnings as _w
        _w.warn("oracle_metrics: correlation is NaN (constant input or insufficient variance)")
        corr = 0.0

    return oracle_mean, agent_mean, corr


__all__ = [
    "rbf_multiscale_kernel",
    "mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd",
    "terminal_cov_error", "terminal_mean_rmse",
    "return_std_error", "return_kurtosis_error",
    "acf", "acf_error",
    "teacher_sigma_metrics",
    "tail_survival_error",
    "rv_law_loss",
    "oracle_metrics",
]
