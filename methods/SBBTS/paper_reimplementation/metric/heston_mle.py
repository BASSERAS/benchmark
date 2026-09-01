"""
The paper's OWN Heston metric (arXiv:2604.07159 section 5.1, Figure 2).

Per-path maximum-likelihood estimation of (kappa, theta, xi, rho, r) on the
bivariate (price, variance) trajectory, followed by a KDE comparison of the
estimated-parameter distributions between real and generated data.

``MLE_Heston_robust`` and the L-BFGS-B settings are reproduced verbatim from the
authors' ``utils/get_params.py`` (vendored at
``../../code/reference/utils/get_params.py``). Only the plotting helper is
rewritten: the reference calls ``sns.kdeplot(shade=...)``, removed in
seaborn >= 0.14, and we additionally report the quantitative summary that the
paper only shows graphically.
"""

import numba as nb
import numpy as np
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

PARAM_NAMES = ["kappa", "theta", "xi", "rho", "r"]
PARAM_LABELS = [r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$", r"$r$"]

# Table 3 priors, keyed by MLE output name.
PRIOR_RANGES = {
    "kappa": (0.5, 4.0),
    "theta": (0.5, 1.5),
    "xi": (0.1, 0.9),
    "rho": (-0.9, 0.9),
    "r": (0.01, 0.1),
}


@nb.jit(nopython=True, cache=True)
def MLE_Heston_robust(params, X, dt):
    """Negative log-likelihood of a bivariate Heston path. Verbatim from the reference."""
    kappa, theta, xi, rho, r = params
    N = len(X)
    logL = 0.0

    S = X[:, 0]
    v = X[:, 1]
    for t in range(N - 1):
        S_t, S_t_next = S[t], S[t + 1]
        v_t = v[t]

        mu_S = np.log(S_t) + (r - 0.5 * v_t) * dt
        mu_v = v_t + kappa * (theta - v_t) * dt

        var_S = v_t * dt
        var_v = xi ** 2 * v_t * dt

        cov_Sv = rho * xi * v_t * dt
        cov_matrix = np.array([[var_S, cov_Sv], [cov_Sv, var_v]])

        if np.linalg.det(cov_matrix) <= 0:
            return 1e10

        inv_cov = np.linalg.inv(cov_matrix)
        det_cov = np.linalg.det(cov_matrix)

        joint_observation = np.array([
            np.log(S_t_next) - mu_S,
            v[t + 1] - mu_v
        ])
        joint_log_pdf = -0.5 * (
            2 * np.log(2 * np.pi) + np.log(det_cov)
            + joint_observation.T @ inv_cov @ joint_observation
        )
        logL -= joint_log_pdf

    return logL


# Reference bounds and initial point, verbatim.
_BOUNDS = [
    (1e-6, None),   # kappa > 0
    (1e-6, None),   # theta > 0
    (1e-6, None),   # xi    > 0
    (-1, 1),        # rho in [-1, 1]
    (0, 1),         # r   in [0, 1]
]
_X0 = np.array([2.0, 0.02, 0.1, -0.5, 0.03])


def _fit_one(path, dt=1 / 252):
    res = minimize(MLE_Heston_robust, x0=_X0,
                   args=(np.ascontiguousarray(path, dtype=np.float64), float(dt)),
                   bounds=_BOUNDS, method="L-BFGS-B")
    return res.x


def get_params_estimation(X, dt=1 / 252, n_jobs=1, verbose=False):
    """Per-path L-BFGS-B MLE. X: (M, N, 2). Returns (M, 5) = [kappa, theta, xi, rho, r]."""
    M = len(X)
    out = np.zeros((M, 5))

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        fit = partial(_fit_one, dt=dt)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for m, res in enumerate(ex.map(fit, (X[i] for i in range(M)), chunksize=16)):
                out[m] = res
        return out

    for m in range(M):
        if verbose and m % 500 == 0:
            print(f"    MLE {m}/{M}", flush=True)
        out[m] = _fit_one(X[m], dt)
    return out


def joint_percentile_filter(params, q1=5, q2=95):
    """The reference's joint percentile filter (all 5 coordinates inside [q1, q2])."""
    lo = np.percentile(params, q1, axis=0)
    hi = np.percentile(params, q2, axis=0)
    keep = (params >= lo).all(axis=1) & (params <= hi).all(axis=1)
    return params[keep], keep


def summarize(params_data, params_gen, q1=5, q2=95):
    """Quantitative version of Figure 2.

    The paper's claim is distributional coverage: SBBTS should span the real
    range of every parameter, whereas SBTS collapses xi and rho onto a spike.
    ``std_ratio`` is the discriminating statistic (about 1 = matched spread,
    much less than 1 = collapsed); ``w1`` is the Wasserstein-1 distance to data.
    """
    fd, _ = joint_percentile_filter(params_data, q1, q2)
    fg, _ = joint_percentile_filter(params_gen, q1, q2)

    rows = []
    for i, name in enumerate(PARAM_NAMES):
        d, g = fd[:, i], fg[:, i]
        rows.append(dict(
            param=name,
            data_mean=float(d.mean()), data_std=float(d.std()),
            data_q05=float(np.percentile(d, 5)), data_q95=float(np.percentile(d, 95)),
            gen_mean=float(g.mean()), gen_std=float(g.std()),
            gen_q05=float(np.percentile(g, 5)), gen_q95=float(np.percentile(g, 95)),
            std_ratio=float(g.std() / d.std()) if d.std() > 0 else float("nan"),
            w1=float(wasserstein_distance(d, g)),
            prior_low=PRIOR_RANGES[name][0], prior_high=PRIOR_RANGES[name][1],
        ))
    return rows


def plot_figure2(params_data, params_gen, out_path, gen_label="SBBTS",
                 params_extra=None, extra_label="SBTS", q1=5, q2=95, show_prior=True):
    """Reproduce Figure 2: 2x3 KDE grid of estimated-parameter distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fd, _ = joint_percentile_filter(params_data, q1, q2)
    fg, _ = joint_percentile_filter(params_gen, q1, q2)
    fe = joint_percentile_filter(params_extra, q1, q2)[0] if params_extra is not None else None

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    # Paper Figure 2 panel order: xi and rho on the top row, then kappa, theta, r.
    order = [2, 3, 0, 1, 4]

    for panel, i in enumerate(order):
        ax = axs[panel // 3, panel % 3]
        sns.kdeplot(ax=ax, data=fd[:, i], fill=True, label="Data")
        if fe is not None:
            sns.kdeplot(ax=ax, data=fe[:, i], fill=True, label=extra_label)
        sns.kdeplot(ax=ax, data=fg[:, i], fill=True, label=gen_label)
        if show_prior:
            lo, hi = PRIOR_RANGES[PARAM_NAMES[i]]
            ax.axvspan(lo, hi, color="grey", alpha=0.12, lw=0, label="Table 3 range")
        ax.set_title(f"Distribution of param {PARAM_LABELS[i]}")
        ax.legend()
        ax.grid(alpha=0.3)

    axs[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
