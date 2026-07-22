"""
Metrics A1–A34 for time-series generation benchmarking.

Numbered in category-display order (Fat Tail → Distribution → Adversarial →
Predictive → Temporal → Vol → Heston Spec).

Fat Tail
  A1   Kurtosis Error  (absolute difference on log-returns)
  A2   |r| q95 Error
  A3   |r| q99 Error
  A4   Tail QQ Error
  A5   Hill Tail Index Error
Distribution
  A6   Path MMD²  (full joint-path)
  A7   Terminal MMD²
  A8   Increment MMD²
  A9   Volatility-discrepancy MMD
  A10  Terminal Sliced Wasserstein Distance
  A11  Path Sliced Wasserstein Distance
  A12  Realized Volatility Law Loss  (W₁ on per-path RV)
  A13  Mean Path RMSE
  A14  KS on Log-returns
  A15  Skewness Error
  A16  QQ RMSE  (300-point)
  A17  Terminal Price KS
Adversarial / Predictive  (PyTorch, separate modules)
  A18  Discriminative Score GRU + MLP  (discriminative_score.py)
  A19  Predictive Score GRU + MLP  (predictive_score.py)
Temporal
  A20  Terminal Covariance Error
  A21  ACF Error on |r| over lags 1–20
  A22  ACF Error on r² over lags 1–20
  A23  ACF |r| Lag-1 Error
  A24  ACF r² Lag-1 Error
Vol
  A25  Mean RMSE  (per-step mean price)
  A26  Return Std Error  (price increments)
  A27  Log-Return Std Error
  A28  Kurtosis Ratio  (target / model, excess kurtosis; perfect = 1.0)
  A29  Sigma Mean Error  (mean annualised per-path vol)
  A30  Cross-Sectional Vol Path RMSE
  A31  Rolling Vol KS  (window = 5)
  A32  Vol-of-Vol Error
Heston Spec
  A33  Teacher-Sigma Correlation  (Heston-specific)
  A34  Teacher-Sigma RMSE  (Heston-specific)

B curve-shape metrics  (this same file, bottom section)
  6 diagnostic plots × 3 sub-metrics (funct / der / sec_der) × 2 error variants
  (MSE / %), computed by compute_curve_metrics and aggregated by
  aggregate_curve_metrics. "B" refers exclusively to these stylized-fact CURVES —
  there are no legacy scalar B metrics; the old scalar shape metrics were absorbed
  into A25–A34 above.

  Stylized-facts framework: Cont R. (2001) "Empirical properties of asset
    returns: stylized facts and statistical issues." Quantitative Finance
    1(2), 223–236.
  ACF sub-metrics: Ding Z., Granger C.W.J., Engle R.F. (1993) "A long memory
    property of stock market returns and a new model." Journal of Empirical
    Finance 1(1), 83–106 (|r| autocorrelation); Bollerslev T. (1986)
    "Generalized Autoregressive Conditional Heteroskedasticity." Journal of
    Econometrics 31(3), 307–327 (r² autocorrelation).
"""

import numpy as np
from scipy.stats import wasserstein_distance, kurtosis, ks_2samp, skew as scipy_skew
from typing import Callable, Dict, Tuple


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
# A16  Log-Return Standard Deviation Error
# ======================================================================

def logreturn_std_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A16. Absolute error in log-return standard deviation.

    A_16 = |sigma(r_real) - sigma(r_gen)|

    where r_{i,t} = log(S_{i,t+1} / S_{i,t}) are log-returns
    flattened over all paths i and time steps t.

    Similar to A9 but A9 uses raw price increments; A16 uses log-returns
    (the financially correct convention consistent with A11/A12).
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X : (N, T, d)  real price paths
    Y : (N, T, d)  generated price paths
    """
    lr_r = np.diff(np.log(np.maximum(X, 1e-10)), axis=1).ravel()
    lr_f = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1).ravel()
    return float(abs(lr_r.std() - lr_f.std()))


# ======================================================================
# A17–A18  Absolute log-return quantile errors
# ======================================================================

def abs_return_quantile_error(
    X: np.ndarray, Y: np.ndarray, q: float,
) -> float:
    r"""A17 / A18. Absolute error at quantile q of the |log-return| distribution.

    A_17/18 = |Q_q(|r_real|) - Q_q(|r_gen|)|

    where r is the flattened vector of all log-returns r_{i,t} = log(S_{i,t+1}/S_{i,t}).
    Use q=0.95 for A17, q=0.99 for A18.

    Directly measures the ability of the generator to reproduce the right tail
    of the absolute-return distribution (fat tails, Cont 2001 stylised fact).
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X : (N, T, d)  real price paths
    Y : (N, T, d)  generated price paths
    q : quantile level in (0, 1)
    """
    abs_r = np.abs(np.diff(np.log(np.maximum(X, 1e-10)), axis=1)).ravel()
    abs_g = np.abs(np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)).ravel()
    return float(abs(np.quantile(abs_r, q) - np.quantile(abs_g, q)))


# ======================================================================
# A19  Kurtosis ratio
# ======================================================================

def kurtosis_ratio(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A19. Ratio of excess kurtosis: kappa_target / kappa_model.

    A_19 = kappa_real / kappa_gen

    where kappa = E[(r - mu)^4] / sigma^4 - 3  is Fisher's excess kurtosis
    on log-returns (unbiased, Fisher=True).

    Perfect: 1.0.  Values > 1 mean the model underestimates tail heaviness
    (lighter tails than target); values < 1 mean the model overestimates.
    Unlike A10 (absolute difference), the ratio is scale-invariant and
    directly shows the *fraction* of fat-tailedness reproduced.

    Parameters
    ----------
    X : (N, T, d)  real price paths
    Y : (N, T, d)  generated price paths
    """
    lr_r = np.diff(np.log(np.maximum(X, 1e-10)), axis=1).ravel()
    lr_f = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1).ravel()
    k_r = float(kurtosis(lr_r, fisher=True, bias=False))
    k_g = float(kurtosis(lr_f, fisher=True, bias=False))
    if abs(k_g) < 1e-8 or np.isnan(k_g):
        return float('nan')
    return float(k_r / k_g)


# ======================================================================
# A20  Sigma mean error
# ======================================================================

def sigma_mean_error(
    X: np.ndarray, Y: np.ndarray,
    dt: float = 1.0 / 250.0,
) -> float:
    r"""A20. Absolute error in mean annualised per-path volatility.

    For each path i, annualised vol:

        sigma_i = std_t(r_{i,t}) * sqrt(1/dt)

    where r_{i,t} = log(S_{i,t+1}/S_{i,t}).

    A_20 = |mean_i(sigma_i^real) - mean_i(sigma_i^gen)|

    Measures whether the model reproduces the average level of volatility
    (i.e. the mean of the vol distribution, not just a global std).
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X  : (N, T, d)  real price paths
    Y  : (N, T, d)  generated price paths
    dt : time step in years (default 1/250)
    """
    ann = np.sqrt(1.0 / dt)
    lr_r = np.diff(np.log(np.maximum(X, 1e-10)), axis=1)   # (N, T-1, d)
    lr_f = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)
    N_r, N_f = lr_r.shape[0], lr_f.shape[0]
    # Per-path std over all time steps (and features for d>1)
    sigma_r = lr_r.reshape(N_r, -1).std(axis=1) * ann       # (N_r,)
    sigma_f = lr_f.reshape(N_f, -1).std(axis=1) * ann       # (N_f,)
    return float(abs(sigma_r.mean() - sigma_f.mean()))


# ======================================================================
# ACF lag-1 errors  (Temporal category)
# ======================================================================

def acf_lag1_abs_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A22. ACF lag-1 error on absolute log-returns.

    A_22 = |E_i[ACF(|r_i|, lag=1)]_real - E_i[ACF(|r_i|, lag=1)]_gen|

    where ACF(x, 1) = corr(x_t, x_{t+1}) per path i.

    Captures the ARCH effect: |r_t| is positively autocorrelated in real
    data (volatility clustering, Engle 1982). A perfect generator reproduces
    this first-lag autocorrelation.
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X : (N, T, d)  real price paths
    Y : (N, T, d)  generated price paths
    """
    lr_r = np.diff(np.log(np.maximum(X, 1e-10)), axis=1)
    lr_f = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)
    return float(acf_error(np.abs(lr_f), np.abs(lr_r), lags=(1,)))


def acf_lag1_sq_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A23. ACF lag-1 error on squared log-returns.

    A_23 = |E_i[ACF(r_i^2, lag=1)]_real - E_i[ACF(r_i^2, lag=1)]_gen|

    Captures the GARCH effect: r_t^2 is autocorrelated (variance clustering,
    Bollerslev 1986). Complementary to A22 — squared returns emphasise large
    moves while |r| weights all moves equally.
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X : (N, T, d)  real price paths
    Y : (N, T, d)  generated price paths
    """
    lr_r = np.diff(np.log(np.maximum(X, 1e-10)), axis=1)
    lr_f = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)
    return float(acf_error(lr_f ** 2, lr_r ** 2, lags=(1,)))


# ======================================================================
# A24  Realized Volatility Law Loss
# ======================================================================

def rv_law_loss(X: np.ndarray, Y: np.ndarray, dt: float = 1.0 / 250.0) -> float:
    r"""A24. Realized Volatility Law Loss — W₁ distance between RV distributions.

    For each path i, annualised realised variance:

        RV_i = (1/dt) * sum_t r_{i,t}^2

    where r_{i,t} = log(S_{i,t+1}/S_{i,t}).

    A_24 = W_1( {RV_i}_real, {RV_i}_gen )

    Tests whether the generator reproduces the *distribution* of realised
    volatility across paths (not just the mean captured by A20).
    Perfect: 0.  Direction: lower is better.

    Parameters
    ----------
    X  : (N, T, d)  real price paths
    Y  : (N, T, d)  generated price paths
    dt : time step in years (default 1/250)
    """
    log_r_real = np.diff(np.log(np.maximum(X, 1e-10)), axis=1)
    log_r_gen  = np.diff(np.log(np.maximum(Y, 1e-10)), axis=1)
    rv_real = (np.sum(log_r_real ** 2, axis=1) / dt).ravel()
    rv_gen  = (np.sum(log_r_gen  ** 2, axis=1) / dt).ravel()
    return float(wasserstein_distance(rv_real, rv_gen))


# ======================================================================
# A25–A34  Distributional shape metrics  (absorbed from old B1–B12)
# B7/B8 = A22/A23 already — not duplicated here.
# ======================================================================

def _to2d(X: np.ndarray) -> np.ndarray:
    """Accept (N,T) or (N,T,d) and return (N,T) by squeezing d=1 or flattening."""
    if X.ndim == 3:
        return X[:, :, 0] if X.shape[2] == 1 else X.reshape(X.shape[0], -1)
    return X


def mean_path_rmse(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A25. RMSE of cross-sectional mean paths over time.

    B25 = sqrt( 1/T * sum_t ( mean_i(S_real[i,t]) - mean_i(S_gen[i,t]) )^2 )

    Tests whether the average price trajectory is reproduced.
    Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    return float(np.sqrt(np.mean((Xp.mean(0) - Yp.mean(0)) ** 2)))


def vol_path_rmse(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A26. RMSE of cross-sectional volatility (std) paths over time.

    A26 = sqrt( 1/T * sum_t ( std_i(S_real[i,t]) - std_i(S_gen[i,t]) )^2 )

    Tests whether the cross-sectional dispersion at each time step is reproduced.
    Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    return float(np.sqrt(np.mean((Xp.std(0) - Yp.std(0)) ** 2)))


def ks_logreturns(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A27. Two-sample KS statistic on pooled log-returns.

    A27 = sup_x | F_real(x) - F_gen(x) |   over log-returns r_t = log(S_{t+1}/S_t)

    Ref: Massey (1951). Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    lr_r = np.diff(np.log(np.maximum(Xp, 1e-10)), axis=1).ravel()
    lr_g = np.diff(np.log(np.maximum(Yp, 1e-10)), axis=1).ravel()
    stat, _ = ks_2samp(lr_r, lr_g)
    return float(stat)


def skewness_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A28. Absolute error in log-return skewness.

    A28 = | skew(r_real) - skew(r_gen) |

    where skew uses Fisher's definition (bias=False).
    Heston produces negative skew (rho < 0 leverage). Ref: Cont (2001).
    Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    lr_r = np.diff(np.log(np.maximum(Xp, 1e-10)), axis=1).ravel()
    lr_g = np.diff(np.log(np.maximum(Yp, 1e-10)), axis=1).ravel()
    return float(abs(float(scipy_skew(lr_r, bias=False)) - float(scipy_skew(lr_g, bias=False))))


def qq_rmse(X: np.ndarray, Y: np.ndarray, n_pts: int = 300) -> float:
    r"""A29. RMSE between quantile functions of log-returns (300 pts).

    A29 = sqrt( 1/G * sum_g ( Q_real(p_g) - Q_gen(p_g) )^2 )   p_g in [0.005, 0.995]

    Ref: Wilk & Gnanadesikan (1968). Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    lr_r = np.diff(np.log(np.maximum(Xp, 1e-10)), axis=1).ravel()
    lr_g = np.diff(np.log(np.maximum(Yp, 1e-10)), axis=1).ravel()
    pts = np.linspace(0.005, 0.995, n_pts)
    q_r = np.quantile(lr_r, pts)
    q_g = np.quantile(lr_g, pts)
    return float(np.sqrt(np.mean((q_r - q_g) ** 2)))


def tail_qq_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A30. Mean absolute tail quantile error on log-returns.

    Evaluated at p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99] (10 pts).
    Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    lr_r = np.diff(np.log(np.maximum(Xp, 1e-10)), axis=1).ravel()
    lr_g = np.diff(np.log(np.maximum(Yp, 1e-10)), axis=1).ravel()
    tail_pts = np.concatenate([np.linspace(0.01, 0.05, 5), np.linspace(0.95, 0.99, 5)])
    q_r = np.quantile(lr_r, tail_pts)
    q_g = np.quantile(lr_g, tail_pts)
    return float(np.mean(np.abs(q_r - q_g)))


def _rolling_std(S: np.ndarray, window: int = 5) -> np.ndarray:
    """(N, T) → (N, T-window) rolling std of log-returns."""
    R = np.diff(np.log(np.maximum(S, 1e-10)), axis=1)
    T1 = R.shape[1]
    cols = [R[:, t - window + 1: t + 1].std(axis=1)
            for t in range(window - 1, T1)]
    return np.stack(cols, axis=1)


def rolling_vol_ks(X: np.ndarray, Y: np.ndarray, window: int = 5) -> float:
    r"""A31. Two-sample KS on rolling log-return volatility (window=5).

    A31 = KS( {sigma_real_{i,t}}, {sigma_gen_{i,t}} )
    sigma_{i,t} = std( r_{i,t-4:t} )

    Ref: Cont (2001). Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    rv_r = _rolling_std(Xp, window).ravel()
    rv_g = _rolling_std(Yp, window).ravel()
    stat, _ = ks_2samp(rv_r, rv_g)
    return float(stat)


def vol_of_vol_error(X: np.ndarray, Y: np.ndarray, window: int = 5) -> float:
    r"""A32. Absolute error in vol-of-vol (std of rolling vol distribution).

    A32 = | std( sigma_real ) - std( sigma_gen ) |

    Ref: Hull & White (1987). Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    rv_r = _rolling_std(Xp, window).ravel()
    rv_g = _rolling_std(Yp, window).ravel()
    return float(abs(rv_r.std() - rv_g.std()))


def terminal_ks(X: np.ndarray, Y: np.ndarray) -> float:
    r"""A33. Two-sample KS on terminal price S_T.

    A33 = KS( S_real[:, T], S_gen[:, T] )

    Tests whether the terminal marginal is reproduced (log-normal for GBM).
    Ref: Massey (1951). Perfect: 0. Direction: lower is better.
    """
    Xp, Yp = _to2d(X), _to2d(Y)
    stat, _ = ks_2samp(Xp[:, -1], Yp[:, -1])
    return float(stat)


def hill_tail_index_error(X: np.ndarray, Y: np.ndarray, k_frac: float = 0.10) -> float:
    r"""A34. Absolute error in Hill tail index estimator on terminal prices.

    alpha_hat = 1 / mean_i( log( X_{(n-i+1)} / X_{(n-k)} ) )  for i=1..k, k=10% of n

    A34 = | alpha_real - alpha_gen |

    Ref: Hill (1975). alpha > 4 => finite kurtosis. Perfect: 0. Direction: lower is better.
    """
    def _hill(x: np.ndarray) -> float:
        x_pos = np.sort(x[x > 0])
        n = len(x_pos)
        k = max(10, int(k_frac * n))
        if k >= n:
            return float("nan")
        threshold = x_pos[-(k + 1)]
        exceedances = x_pos[-k:]
        diffs = np.log(exceedances) - np.log(threshold)
        m = diffs.mean()
        return float("nan") if m <= 0 else float(1.0 / m)

    Xp, Yp = _to2d(X), _to2d(Y)
    ar = _hill(Xp[:, -1])
    ag = _hill(Yp[:, -1])
    if not (np.isfinite(ar) and np.isfinite(ag)):
        return float("nan")
    return float(abs(ar - ag))


__all__ = [
    "rbf_multiscale_kernel",
    # Distribution (MMD, SWD)
    "mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd",
    # Statistical moments / Vol
    "terminal_cov_error", "terminal_mean_rmse",
    "return_std_error", "return_kurtosis_error",
    # ACF / Temporal
    "acf", "acf_error",
    "acf_lag1_abs_error", "acf_lag1_sq_error",
    # A33–A34  Heston-specific teacher-sigma (corr + rmse)
    "teacher_sigma_metrics",
    # Log-return vol / distribution / tail
    "logreturn_std_error",
    "abs_return_quantile_error",
    "kurtosis_ratio",
    "sigma_mean_error",
    # Realized vol
    "rv_law_loss",
    # Distributional shape / tail / curve-derived
    "mean_path_rmse",
    "vol_path_rmse",
    "ks_logreturns",
    "skewness_error",
    "qq_rmse",
    "tail_qq_error",
    "rolling_vol_ks",
    "vol_of_vol_error",
    "terminal_ks",
    "hill_tail_index_error",
    # B curve-shape metrics (6 plots × 3 sub-metrics × 2 variants)
    "compute_curve_metrics",
    "aggregate_curve_metrics",
    "CURVE_PLOTS",
]


# ======================================================================
# B curve-shape metrics
# 6 diagnostic plots × 3 sub-metrics (funct / der / sec_der) × 2 error
# variants (MSE / %), computed by compute_curve_metrics and aggregated by
# aggregate_curve_metrics.  Each plot's real vs. generated curve is built on
# shared evaluation points so every number can be visually verified against
# the PNG diagnostic figure.  See the module docstring for paper references
# (Cont 2001; Ding/Granger/Engle 1993; Bollerslev 1986).
# ======================================================================

# ── curve helpers ─────────────────────────────────────────────────────

def _log_returns(S: np.ndarray) -> np.ndarray:
    """(N, T) price paths → (N, T-1) log-returns  r_{i,t} = log(S_{i,t+1}/S_{i,t})."""
    return np.log(np.maximum(S[:, 1:], 1e-10) / np.maximum(S[:, :-1], 1e-10))


def _rolling_vol(S: np.ndarray, window: int = 5) -> np.ndarray:
    """(N, T) → (N, T-window) rolling std of log-returns over a sliding window."""
    R = _log_returns(S)                # (N, T-1)
    T1 = R.shape[1]
    cols = [R[:, t - window + 1: t + 1].std(axis=1)
            for t in range(window - 1, T1)]
    return np.stack(cols, axis=1)     # (N, T-1-(window-1))


def _acf_mean(X: np.ndarray, lag: int) -> float:
    """Mean cross-path ACF at a given lag.  X : (N, T)."""
    X_c = X - X.mean(axis=1, keepdims=True)
    var = (X_c ** 2).mean(axis=1)
    cov = (X_c[:, lag:] * X_c[:, :-lag]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(var > 0, cov / var, 0.0)
    return float(r.mean())


# ── Curve-shape B metrics  (6 plots × 3 sub-metrics × 3 measures) ──────
#
# For each of the 6 diagnostic plots we build a 1-D curve L from the real data and
# a matching curve L_gen from the generated data on the SAME evaluation points, so
# the two are directly comparable point-by-point. From each pair we derive three
# sub-metrics — on the curve itself (funct), its 1st finite difference (der), and
# its 2nd finite difference (sec_der) — under THREE error measures:
#
#   MSE   : mean( (L_gen - L_real)^2 )                              (absolute, units²)
#   %err  : mean( |L_gen - L_real| / (|L_real| + eps) ) * 100       (relative, %)
#           scale-aware epsilon floor  eps = 1e-3 * (|L_real|.max() + 1e-12).
#           The floor is a fixed fraction of the curve's largest magnitude, so the
#           denominator can never collapse to ~0 (as a bare |L_real| would where the
#           curve crosses zero, e.g. on der/sec_der). This makes the percentage
#           meaningful on ALL three sub-metrics, not just the curve itself.
#   NRMSE : sqrt(mean((L_gen - L_real)^2)) / (|L_real|.max() - |L_real|.min()) * 100
#           range-normalised RMSE (%), scale-free and comparable across plots.
#
# All three measures are computed identically point-by-point; only the discrepancy
# functional differs. For every plot the three sub-metrics (funct, der, sec_der)
# are combined into ONE number PER MEASURE by taking their MEAN (mean-of-3).

# Registry of the 6 plots: (key prefix, human-readable name). Used by the
# aggregator and by every downstream README so future methods stay consistent.
CURVE_PLOTS = [
    ("B_log_ret_hist",  "Log-return histogram"),
    ("B_qq_plot",       "QQ plot"),
    ("B_acf_abs_r",     "ACF |r|"),
    ("B_acf_sq_r",      "ACF r²"),
    ("B_roll_vol_hist", "Rolling vol histogram"),
    ("B_tail_surv",     "Tail survival"),
]

_CURVE_SUBS = ("funct", "der", "sec_der")


def _curve_scores(L_r: np.ndarray, L_g: np.ndarray) -> Dict[str, float]:
    """Nine sub-scores for one (real, generated) curve pair.

    Returns a dict with keys
        funct_mse,   der_mse,   sec_der_mse     (MSE measure)
        funct_pct,   der_pct,   sec_der_pct     (scale-aware %-error measure, %)
        funct_nrmse, der_nrmse, sec_der_nrmse   (range-normalised RMSE, %)

    where
        funct   compares L                         (the curve itself)
        der     compares diff(L)                   diff[k]  = L[k+1] - L[k]
        sec_der compares diff2(L)                  diff2[k] = diff[k+1] - diff[k]

    MSE(a, b)   = mean( (b - a)^2 )
    PCT(a, b)   = mean( |b - a| / (|a| + eps) ) * 100,   eps = 1e-3*(|a|.max()+1e-12)
                  Scale-aware epsilon floor (Prop 1): the denominator is floored at a
                  fixed fraction of the curve's largest magnitude, so it can never
                  collapse to ~0 where the real curve crosses zero (der/sec_der).
    NRMSE(a, b) = sqrt(mean((b - a)^2)) / (|a|.max() - |a|.min() + 1e-12) * 100
                  Range-normalised RMSE in % (Prop 2): scale-free, comparable across
                  plots and across the curve / its derivatives.
    """
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((b - a) ** 2))

    def _pct(a: np.ndarray, b: np.ndarray) -> float:
        # scale-aware epsilon floor: eps is a fixed fraction of the curve's peak
        # magnitude, so the relative error never explodes where a ~ 0.
        eps = 1e-3 * (np.abs(a).max() + 1e-12)
        return float(np.mean(np.abs(b - a) / (np.abs(a) + eps)) * 100.0)

    def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
        rng = np.abs(a).max() - np.abs(a).min() + 1e-12
        rmse = np.sqrt(np.mean((b - a) ** 2))
        return float(rmse / rng * 100.0)

    d_r,  d_g  = np.diff(L_r),  np.diff(L_g)
    dd_r, dd_g = np.diff(d_r),  np.diff(d_g)
    return {
        "funct_mse":     _mse(L_r,  L_g),
        "der_mse":       _mse(d_r,  d_g),
        "sec_der_mse":   _mse(dd_r, dd_g),
        "funct_pct":     _pct(L_r,  L_g),
        "der_pct":       _pct(d_r,  d_g),
        "sec_der_pct":   _pct(dd_r, dd_g),
        "funct_nrmse":   _nrmse(L_r,  L_g),
        "der_nrmse":     _nrmse(d_r,  d_g),
        "sec_der_nrmse": _nrmse(dd_r, dd_g),
    }


def _emit(out: Dict[str, float], prefix: str, scores: Dict[str, float]) -> None:
    """Write the 9 sub-scores of one plot into the flat output dict.

    MSE keys keep their historical names (B_<plot>_funct / _der / _sec_der) for
    backward compatibility; the %-error measure appends _pct and the NRMSE
    measure appends _nrmse.
    """
    out[f"{prefix}_funct"]         = scores["funct_mse"]
    out[f"{prefix}_der"]           = scores["der_mse"]
    out[f"{prefix}_sec_der"]       = scores["sec_der_mse"]
    out[f"{prefix}_funct_pct"]     = scores["funct_pct"]
    out[f"{prefix}_der_pct"]       = scores["der_pct"]
    out[f"{prefix}_sec_der_pct"]   = scores["sec_der_pct"]
    out[f"{prefix}_funct_nrmse"]   = scores["funct_nrmse"]
    out[f"{prefix}_der_nrmse"]     = scores["der_nrmse"]
    out[f"{prefix}_sec_der_nrmse"] = scores["sec_der_nrmse"]


def compute_curve_metrics(
    S_real: np.ndarray,
    S_gen:  np.ndarray,
    n_bins: int = 100,
    n_lags: int = 20,
) -> Dict[str, float]:
    """Compute the B curve metrics: 6 plots × 3 sub-metrics × 3 measures (54 keys).

    For each diagnostic plot a 1-D curve L is constructed from real and generated
    data on shared evaluation points, then scored with :func:`_curve_scores`
    (MSE + scale-aware %-error + range-normalised RMSE on the curve, its 1st diff
    and its 2nd diff).

    Output keys per plot ``<prefix>``:
        <prefix>_funct         <prefix>_der         <prefix>_sec_der         (MSE)
        <prefix>_funct_pct     <prefix>_der_pct     <prefix>_sec_der_pct     (%err)
        <prefix>_funct_nrmse   <prefix>_der_nrmse   <prefix>_sec_der_nrmse   (NRMSE)

    6 plots
    -------
    B_log_ret_hist   Empirical histogram density of log-returns at n_bins shared bins
    B_qq_plot        Quantile function Q(p) at n_bins uniform percentile levels
    B_acf_abs_r      Mean per-path ACF(|r|, lag) for lag = 1 .. n_lags
    B_acf_sq_r       Mean per-path ACF(r^2, lag) for lag = 1 .. n_lags
    B_roll_vol_hist  Histogram density of rolling-5 log-return vol at n_bins shared bins
    B_tail_surv      Empirical survival P(|r| > x) at n_bins quantile thresholds of real

    This routine is method-agnostic: it only needs the real and generated price
    matrices, so any future generator can be scored by the same call.
    """
    R_real = _log_returns(S_real)   # (N, T-1)
    R_gen  = _log_returns(S_gen)

    r_r = R_real.ravel()
    r_g = R_gen.ravel()

    out: Dict[str, float] = {}

    # -- Plot 1: Log-return histogram --
    # Bins fixed from real data (0.5–99.5th percentile) so L_real is the same
    # reference curve for every seed regardless of generated distribution width.
    lo_r, hi_r = np.percentile(r_r, 0.5), np.percentile(r_r, 99.5)
    edges = np.linspace(lo_r, hi_r, n_bins + 1)
    density_r, _ = np.histogram(r_r, bins=edges, density=True)
    density_g, _ = np.histogram(r_g, bins=edges, density=True)
    _emit(out, "B_log_ret_hist", _curve_scores(density_r, density_g))

    # -- Plot 2: QQ plot --
    pp = np.linspace(0.005, 0.995, n_bins)
    q_r = np.quantile(r_r, pp)
    q_g = np.quantile(r_g, pp)
    _emit(out, "B_qq_plot", _curve_scores(q_r, q_g))

    # -- Plot 3: ACF of |r| --
    lags = np.arange(1, n_lags + 1)
    acf_abs_r = np.array([_acf_mean(np.abs(R_real), lag=int(l)) for l in lags])
    acf_abs_g = np.array([_acf_mean(np.abs(R_gen),  lag=int(l)) for l in lags])
    _emit(out, "B_acf_abs_r", _curve_scores(acf_abs_r, acf_abs_g))

    # -- Plot 4: ACF of r^2 --
    acf_sq_r = np.array([_acf_mean(R_real ** 2, lag=int(l)) for l in lags])
    acf_sq_g = np.array([_acf_mean(R_gen  ** 2, lag=int(l)) for l in lags])
    _emit(out, "B_acf_sq_r", _curve_scores(acf_sq_r, acf_sq_g))

    # -- Plot 5: Rolling vol histogram --
    # Bins fixed from real data so L_real is the same reference curve every seed.
    rv_r = _rolling_vol(S_real, window=5).ravel()
    rv_g = _rolling_vol(S_gen,  window=5).ravel()
    lo_rv, hi_rv = np.percentile(rv_r, 0.5), np.percentile(rv_r, 99.5)
    edges_rv = np.linspace(lo_rv, hi_rv, n_bins + 1)
    dens_rv_r, _ = np.histogram(rv_r, bins=edges_rv, density=True)
    dens_rv_g, _ = np.histogram(rv_g, bins=edges_rv, density=True)
    _emit(out, "B_roll_vol_hist", _curve_scores(dens_rv_r, dens_rv_g))

    # -- Plot 6: Tail survival --
    abs_r = np.abs(r_r)
    abs_g = np.abs(r_g)
    thresholds = np.quantile(abs_r, np.linspace(0.005, 0.995, n_bins))
    surv_r = np.array([np.mean(abs_r > t) for t in thresholds])
    surv_g = np.array([np.mean(abs_g > t) for t in thresholds])
    _emit(out, "B_tail_surv", _curve_scores(surv_r, surv_g))

    return out


def aggregate_curve_metrics(per_seed: list) -> Dict[str, dict]:
    """Combine per-seed B curve metrics into the per-plot summary shown in READMEs.

    Parameters
    ----------
    per_seed : list of dicts, one per seed, each holding the 54 flat keys produced
               by :func:`compute_curve_metrics`.

    For every plot and every measure (MSE, %err, NRMSE) the three sub-metrics
    (funct, der, sec_der) are combined into a single per-seed score by taking their
    MEAN (mean-of-3), then averaged across seeds:

        combined_per_seed = mean( funct_seed, der_seed, sec_der_seed )
        mean = mean_over_seeds(combined_per_seed)
        std  = std_over_seeds(combined_per_seed)   (direct sample std across seeds)

    The scale-aware %-error floor (eps = 1e-3*|L_real|.max()) keeps the der /
    sec_der relative errors finite, so — unlike the old function-only MAPE — all
    three sub-metrics can now be averaged for the %err measure too.

    EXCEPTION — Tail survival: the B_tail_surv curve is evaluated at the real
    data's own quantile levels, so surv_r is a straight line by construction; its
    1st derivative is a constant and its 2nd derivative is pure numerical noise
    (~1e-6). Those degenerate der / sec_der carry no information and their relative
    errors (%err, NRMSE) explode. For B_tail_surv ONLY, the %err and NRMSE measures
    are therefore reported from the FUNCTION curve alone (funct-only); its MSE
    measure still uses the mean-of-3 (MSE does not explode there).

    Returns
    -------
    dict keyed by plot prefix ->
        {"name": str,
         "mse":   {"mean", "std", "per_seed": [..5..]},
         "pct":   {"mean", "std", "per_seed": [..5..]},
         "nrmse": {"mean", "std", "per_seed": [..5..]}}

    Method-agnostic: pass the seed dicts of any generator to render its B table.
    """
    agg: Dict[str, dict] = {}
    for prefix, name in CURVE_PLOTS:
        row = {"name": name}
        for measure, suffix in (("mse", ""), ("pct", "_pct"), ("nrmse", "_nrmse")):
            sub_arrays = {
                s: np.array([float(d[f"{prefix}_{s}{suffix}"]) for d in per_seed])
                for s in _CURVE_SUBS
            }
            if prefix == "B_tail_surv" and measure in ("pct", "nrmse"):
                # tail_surv's survival curve is linear-by-construction (evaluated at
                # the real data's own quantile levels), so its der/sec_der are
                # degenerate and their relative errors explode. Report the FUNCTION
                # curve only for the %err and NRMSE measures (MSE keeps mean-of-3).
                combined = sub_arrays["funct"]
            else:
                # mean-of-3: combine the three sub-metrics by their arithmetic mean,
                # per seed; std is the direct sample std of that per-seed mean.
                combined = (sub_arrays["funct"] + sub_arrays["der"]
                            + sub_arrays["sec_der"]) / 3.0
            std = float(combined.std())
            row[measure] = {
                "mean": float(combined.mean()),
                "std":  std,
                "per_seed": [float(x) for x in combined],
            }
        agg[prefix] = row
    return agg
