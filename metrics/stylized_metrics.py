"""
stylized_metrics.py — B1–B12: Quantitative metrics extracted from the 8 diagnostic plots.

B8 (ARCH Persistence Error) and B10 (GARCH Persistence Error) removed:
they are redundant with A11 (ACF |r| at lags {1,2,5,10}) and A12 (ACF r² at same lags).
Renumbering: old B9→B8, old B11→B9, old B12→B10, old B13→B11, old B14→B12.

Each metric corresponds directly to one (or two paired) diagnostic panels so that
every number in this table can be visually verified against the PNG diagnostic figure.

References
----------
Cont R. (2001) "Empirical properties of asset returns: stylized facts and
  statistical issues." Quantitative Finance 1(2), 223–236.
Bollerslev T. (1986) "Generalized Autoregressive Conditional Heteroskedasticity."
  Journal of Econometrics 31(3), 307–327.
Ding Z., Granger C.W.J., Engle R.F. (1993) "A long memory property of stock
  market returns and a new model." Journal of Empirical Finance 1(1), 83–106.
Hill B.M. (1975) "A simple general approach to inference about the tail of a
  distribution." Annals of Statistics 3(5), 1163–1174.
Massey F.J. (1951) "The Kolmogorov-Smirnov test for goodness of fit."
  Journal of the American Statistical Association 46(253), 68–78.

Metric map
----------
Plots 1+2  (sample paths)          B1  Mean Path RMSE
                                   B2  Cross-Sectional Vol RMSE
Plot  3    (log-return histogram)  B3  KS Statistic (log-returns)
                                   B4  Skewness Error
Plot  4    (QQ plot)               B5  QQ RMSE
                                   B6  Tail QQ Error (5th/95th extremes)
Plot  5    (ACF |r|)               B7  ACF Lag-1 Error (|r|)
Plot  6    (ACF r²)                B8  ACF Lag-1 Error (r²)
Plot  7    (rolling vol histogram)  B9  Rolling Vol KS Statistic
                                   B10 Vol-of-Vol Error
Plot  8    (tail survival log-log) B11 Terminal Price KS Statistic
                                   B12 Tail Index Error (Hill estimator)
"""

import numpy as np
from scipy.stats import ks_2samp, skew
from typing import Dict


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _hill_estimator(x: np.ndarray, k_frac: float = 0.10) -> float:
    """Hill (1975) tail index α̂.

    Uses the top k = max(10, int(k_frac * n)) order statistics.
    α̂ = 1 / mean_i(log X_{(n-i+1)} − log X_{(n-k)})   for i=1..k

    Returns α̂ (higher = lighter tail; Pareto tail index).
    Returns NaN if insufficient data.
    """
    x_pos = np.sort(x[x > 0])
    n = len(x_pos)
    k = max(10, int(k_frac * n))
    if k >= n:
        return float("nan")
    threshold = x_pos[-(k + 1)]
    exceedances = x_pos[-k:]
    diffs = np.log(exceedances) - np.log(threshold)
    if diffs.mean() <= 0:
        return float("nan")
    return float(1.0 / diffs.mean())


# ── main function ─────────────────────────────────────────────────────────────

def compute_stylized_metrics(
    S_real: np.ndarray,
    S_gen: np.ndarray,
    acf_lags: int = 20,
    vol_window: int = 5,
    qq_grid: int = 300,
) -> Dict[str, float]:
    """Compute B1–B14 stylized metrics.

    Parameters
    ----------
    S_real : (N, T)  real price paths (price space, S0 ≈ 100)
    S_gen  : (N, T)  generated price paths
    acf_lags   : max lag for ACF panels (default 20, matches plot)
    vol_window : rolling vol window in days (default 5, matches plot)
    qq_grid    : number of quantile points for QQ RMSE (default 300, matches plot)

    Returns
    -------
    dict with keys B1..B12 (floats, all ↓ lower is better)
    """
    N_r, T = S_real.shape
    N_g    = S_gen.shape[0]

    # log-returns
    R_real = _log_returns(S_real)   # (N_r, T-1)
    R_gen  = _log_returns(S_gen)    # (N_g, T-1)

    r_flat_r = R_real.ravel()
    r_flat_g = R_gen.ravel()

    out: Dict[str, float] = {}

    # ── B1  Mean Path RMSE (Plots 1+2) ───────────────────────────────────────
    # RMSE between cross-sectional mean paths over time
    # Formula: sqrt(1/T * Σ_t (E_real[S(t)] - E_gen[S(t)])²)
    mu_r = S_real.mean(axis=0)   # (T,)
    mu_g = S_gen.mean(axis=0)
    out["B1_mean_path_rmse"] = float(np.sqrt(np.mean((mu_r - mu_g) ** 2)))

    # ── B2  Cross-Sectional Vol RMSE (Plots 1+2) ─────────────────────────────
    # RMSE between cross-sectional standard deviation at each t
    # Formula: sqrt(1/T * Σ_t (std_real(t) - std_gen(t))²)
    sig_r = S_real.std(axis=0)
    sig_g = S_gen.std(axis=0)
    out["B2_vol_path_rmse"] = float(np.sqrt(np.mean((sig_r - sig_g) ** 2)))

    # ── B3  KS Statistic — log-returns (Plot 3) ──────────────────────────────
    # Two-sample Kolmogorov-Smirnov: D = max_x |F_real(x) - F_gen(x)|
    # Ref: Massey (1951). Perfect: 0. ↓
    ks_stat, _ = ks_2samp(r_flat_r, r_flat_g)
    out["B3_ks_logreturns"] = float(ks_stat)

    # ── B4  Skewness Error (Plot 3) ──────────────────────────────────────────
    # |skew_real - skew_gen|, skew = E[(r-μ)³]/σ³ (Fisher, unbiased)
    # Heston generates negative skew via ρ<0 (leverage effect). Ref: Cont 2001.
    sk_r = float(skew(r_flat_r, bias=False))
    sk_g = float(skew(r_flat_g, bias=False))
    out["B4_skewness_error"] = float(abs(sk_r - sk_g))

    # ── B5  QQ RMSE (Plot 4) ─────────────────────────────────────────────────
    # sqrt(1/G * Σ_g (Q_real(g) - Q_gen(g))²) over uniform quantile grid
    # Ref: Wilk & Gnanadesikan (1968). Perfect: 0. ↓
    qq_grid_pts = np.linspace(0.005, 0.995, qq_grid)
    q_r = np.quantile(r_flat_r, qq_grid_pts)
    q_g = np.quantile(r_flat_g, qq_grid_pts)
    out["B5_qq_rmse"] = float(np.sqrt(np.mean((q_r - q_g) ** 2)))

    # ── B6  Tail QQ Error (Plot 4) ───────────────────────────────────────────
    # Mean absolute deviation at extreme quantile levels [0.01..0.05, 0.95..0.99]
    # Captures how well tails match; complements B5 which is dominated by the bulk.
    tail_levels = np.concatenate([np.linspace(0.01, 0.05, 5), np.linspace(0.95, 0.99, 5)])
    q_r_tail = np.quantile(r_flat_r, tail_levels)
    q_g_tail = np.quantile(r_flat_g, tail_levels)
    out["B6_tail_qq_error"] = float(np.mean(np.abs(q_r_tail - q_g_tail)))

    # ── B7  ACF Lag-1 Error |r| (Plot 5) ─────────────────────────────────────
    # |ACF_real(|r|, lag=1) - ACF_gen(|r|, lag=1)|
    # Lag-1 captures the dominant ARCH signal. Ref: Ding, Granger & Engle (1993).
    acf1_abs_r = _acf_mean(np.abs(R_real), lag=1)
    acf1_abs_g = _acf_mean(np.abs(R_gen),  lag=1)
    out["B7_acf_lag1_abs"] = float(abs(acf1_abs_r - acf1_abs_g))

    # ── B8  ACF Lag-1 Error r² (Plot 6) ──────────────────────────────────────
    # |ACF_real(r², lag=1) - ACF_gen(r², lag=1)|
    # Squared returns ACF captures GARCH effect. Ref: Bollerslev (1986).
    acf1_sq_r = _acf_mean(R_real ** 2, lag=1)
    acf1_sq_g = _acf_mean(R_gen  ** 2, lag=1)
    out["B8_acf_lag1_sq"] = float(abs(acf1_sq_r - acf1_sq_g))

    # ── B9  Rolling Vol KS Statistic (Plot 7) ────────────────────────────────
    # Two-sample KS on rolling volatility distributions (window=vol_window)
    # Ref: Cont (2001) stylized fact: "volatility clusters in time"
    rv_r = _rolling_vol(S_real, vol_window).ravel()
    rv_g = _rolling_vol(S_gen,  vol_window).ravel()
    ks_vol, _ = ks_2samp(rv_r, rv_g)
    out["B9_rolling_vol_ks"] = float(ks_vol)

    # ── B10  Vol-of-Vol Error (Plot 7) ───────────────────────────────────────
    # |std(roll_vol_real) - std(roll_vol_gen)| normalized by real std
    # Measures whether the dispersion of the vol distribution is reproduced.
    # Ref: Hull & White (1987) — vol-of-vol drives option smile.
    std_rv_r = float(rv_r.std())
    std_rv_g = float(rv_g.std())
    out["B10_vol_of_vol_error"] = float(abs(std_rv_r - std_rv_g))

    # ── B11  Terminal Price KS Statistic (Plot 8) ─────────────────────────────
    # Two-sample KS on terminal price S_T distribution
    # Tests whether the generator reproduces the terminal marginal (log-normal for GBM).
    term_r = S_real[:, -1]
    term_g = S_gen[:,  -1]
    ks_term, _ = ks_2samp(term_r, term_g)
    out["B11_terminal_ks"] = float(ks_term)

    # ── B12  Tail Index Error — Hill estimator (Plot 8) ───────────────────────
    # |α̂_real - α̂_gen| where α̂ = Hill(1975) estimator on terminal prices
    # Uses top 10% of terminal prices. α > 4 → finite 4th moment; α > 2 → finite variance.
    # Ref: Hill (1975); Mandelbrot (1963) — empirical tail indices for stocks ≈ 3–5.
    alpha_r = _hill_estimator(term_r, k_frac=0.10)
    alpha_g = _hill_estimator(term_g, k_frac=0.10)
    if np.isfinite(alpha_r) and np.isfinite(alpha_g):
        out["B12_tail_index_error"] = float(abs(alpha_r - alpha_g))
    else:
        out["B12_tail_index_error"] = float("nan")

    return out


# ── Curve-shape B metrics  (6 plots × 3 sub-metrics × 2 variants) ──────────────
#
# For each of the 6 diagnostic plots we build a 1-D curve L from the real data and
# a matching curve L_gen from the generated data on the SAME evaluation points, so
# the two are directly comparable point-by-point. From each pair we derive three
# sub-metrics — on the curve itself, its 1st finite difference, and its 2nd finite
# difference — under two error measures:
#
#   MSE      : mean( (L_real - L_gen)^2 )                        (absolute, units²)
#   % error  : mean( |L_gen - L_real| / (|L_real| + 1e-6) ) * 100   (relative, %)
#
# The two families are computed identically; only the point-wise discrepancy
# differs (squared difference vs. relative absolute difference).

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
    """Six sub-scores for one (real, generated) curve pair.

    Returns a dict with keys
        funct_mse, der_mse, sec_der_mse         (MSE variant)
        funct_pct, der_pct, sec_der_pct         (percentage-error variant, %)

    where
        funct   compares L                         (the curve itself)
        der     compares diff(L)                   diff[k]  = L[k+1] - L[k]
        sec_der compares diff2(L)                  diff2[k] = diff[k+1] - diff[k]

    MSE(a, b)  = mean( (a - b)^2 )
    PCT(a, b)  = mean( |b - a| / (|a| + 1e-6) ) * 100    (mean absolute
                 percentage error over the curve's points; the mean already
                 divides by the number of points — ONE division, a proper MAPE.
                 |·| in the denominator keeps it a magnitude percentage even
                 where the real curve is negative, e.g. an ACF that dips below 0).
    """
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    def _pct(a: np.ndarray, b: np.ndarray) -> float:
        # mean absolute percentage error over the curve's points (ONE division:
        # np.mean already sums and divides by the number of points)
        return float(np.mean(np.abs(b - a) / (np.abs(a) + 1e-6)) * 100.0)

    d_r,  d_g  = np.diff(L_r),  np.diff(L_g)
    dd_r, dd_g = np.diff(d_r),  np.diff(d_g)
    return {
        "funct_mse":   _mse(L_r,  L_g),
        "der_mse":     _mse(d_r,  d_g),
        "sec_der_mse": _mse(dd_r, dd_g),
        "funct_pct":   _pct(L_r,  L_g),
        "der_pct":     _pct(d_r,  d_g),
        "sec_der_pct": _pct(dd_r, dd_g),
    }


def _emit(out: Dict[str, float], prefix: str, scores: Dict[str, float]) -> None:
    """Write the 6 sub-scores of one plot into the flat output dict.

    MSE keys keep their historical names (B_<plot>_funct / _der / _sec_der) for
    backward compatibility; the percentage variant appends _pct.
    """
    out[f"{prefix}_funct"]       = scores["funct_mse"]
    out[f"{prefix}_der"]         = scores["der_mse"]
    out[f"{prefix}_sec_der"]     = scores["sec_der_mse"]
    out[f"{prefix}_funct_pct"]   = scores["funct_pct"]
    out[f"{prefix}_der_pct"]     = scores["der_pct"]
    out[f"{prefix}_sec_der_pct"] = scores["sec_der_pct"]


def compute_curve_metrics(
    S_real: np.ndarray,
    S_gen:  np.ndarray,
    n_bins: int = 100,
    n_lags: int = 20,
) -> Dict[str, float]:
    """Compute the B curve metrics: 6 plots × 3 sub-metrics × 2 variants (36 keys).

    For each diagnostic plot a 1-D curve L is constructed from real and generated
    data on shared evaluation points, then scored with :func:`_curve_scores`
    (MSE + percentage error on the curve, its 1st diff and its 2nd diff).

    Output keys per plot ``<prefix>``:
        <prefix>_funct       <prefix>_der       <prefix>_sec_der        (MSE)
        <prefix>_funct_pct   <prefix>_der_pct   <prefix>_sec_der_pct    (% error)

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
    per_seed : list of dicts, one per seed, each holding the 36 flat keys produced
               by :func:`compute_curve_metrics`.

    For every plot the three sub-metrics (funct, der, sec_der) are combined into a
    single per-seed score, then averaged across seeds. The two variants differ:

    MSE variant (sum of the three sub-metrics, quadrature std):
        combined_per_seed = funct_seed + der_seed + sec_der_seed
        mean = mean_over_seeds(combined_per_seed)           (= sum of the 3 means)
        std  = sqrt( var(funct) + var(der) + var(sec_der) ) (sub-metrics combined
                     in quadrature, per the benchmark spec)

    PCT variant (function-level MAPE on the curve L itself, one division):
        combined_per_seed = funct_pct_seed                 (= 100*mean(|L_g - L_r|
                     / (|L_r| + 1e-6)); the derivative/second-derivative MAPE is
                     NOT averaged in — their near-zero true differences make the
                     relative error explode into meaningless 10^4-% values)
        mean = mean_over_seeds(combined_per_seed)
        std  = std_over_seeds(combined_per_seed)            (direct sample std of the
                     per-seed function MAPE across seeds)

    Returns
    -------
    dict keyed by plot prefix ->
        {"name": str,
         "mse": {"mean", "std", "per_seed": [..5..]},
         "pct": {"mean", "std", "per_seed": [..5..]}}

    Method-agnostic: pass the seed dicts of any generator to render its B table.
    """
    agg: Dict[str, dict] = {}
    for prefix, name in CURVE_PLOTS:
        row = {"name": name}
        for variant, suffix in (("mse", ""), ("pct", "_pct")):
            sub_arrays = {
                s: np.array([float(d[f"{prefix}_{s}{suffix}"]) for d in per_seed])
                for s in _CURVE_SUBS
            }
            if variant == "mse":
                # sum of the three sub-metrics; sub-metric stds combined in quadrature
                combined = (sub_arrays["funct"] + sub_arrays["der"]
                            + sub_arrays["sec_der"])
                std = float(np.sqrt(sum(sub_arrays[s].std() ** 2
                                        for s in _CURVE_SUBS)))
            else:
                # curve-level MAPE on the FUNCTION L itself only:
                #   100 * mean(|L_g - L_r| / (|L_r| + 1e-6))   (one division).
                # The derivative / second-derivative MAPE is intentionally NOT
                # averaged in: diff(L) and diff2(L) have near-zero true values,
                # so their relative error explodes (denominator → 0) and produces
                # meaningless 10^4-% figures. The reported % error is therefore the
                # single meaningful percentage — the deviation of the curve itself.
                combined = sub_arrays["funct"]
                std = float(combined.std())
            row[variant] = {
                "mean": float(combined.mean()),
                "std":  std,
                "per_seed": [float(x) for x in combined],
            }
        agg[prefix] = row
    return agg


# ── public ────────────────────────────────────────────────────────────────────

__all__ = [
    "compute_stylized_metrics",
    "compute_curve_metrics",
    "aggregate_curve_metrics",
    "CURVE_PLOTS",
]
