"""
Multi-asset Heston dataset generator (d = 8 assets, T = 252 daily steps).

Model
-----
Parsimonious multi-asset Heston model of Szimayer, Dimitroff & Lorenz (2011),
IJTAF 14(8), 1299-1333.  For i = 1..d:

    dS_i = mu_i S_i dt + sqrt(v_i) S_i dW_i
    dv_i = kappa_i (theta_i - v_i) dt
           + eta_i sqrt(v_i) [ rho_i dW_i + sqrt(1 - rho_i^2) dW~_i ]

with  corr(W) = Sigma_S = (rho_ij),  corr(W~) = I_d,  W independent of W~
(their Assumption 2.1).  This yields, by their Proposition 2.2,

    corr(dS_i, dS_j)   = rho_ij
    corr(dS_i, dv_j)   = rho_ij rho_j
    corr(dv_i, dv_j)   = rho_ij rho_i rho_j        (i != j)

and is positive semi-definite BY CONSTRUCTION, because the correlation matrix
of (W, W~) is block diagonal, diag(Sigma_S, I_d).  No PSD repair step is needed.

Parameters
----------
Single-asset parameters are centred on the maximum-likelihood estimates of
Ait-Sahalia & Kimmel (2007), JFE 83(2), Table 6, for daily S&P 500 data under
the PHYSICAL measure at dt = 1/252 -- i.e. exactly this sampling frequency and
measure:

    kappa_bar = 5.07,  theta_bar = 0.0457,  eta_bar = 0.48,  rho_bar = -0.767

Per-asset heterogeneity is induced by randomising the parameter vector once
(seed PARAM_SEED), following the randomised-parameter protocol of Alouadi et
al. (2025, ICAIF).  Parameters are drawn ONCE and then held FIXED across all
paths and all five splits -- otherwise the eight "assets" would have no
identity and the splits would not be draws from a common distribution.  Only
the Brownian noise differs between splits.

Storage contract
----------------
Consistent with dataset/Heston: arrays store PRICE paths S_t (and the latent
variance v_t), NOT log returns.  Arithmetic Euler on S directly.  Any
log-return transform belongs to the model-input layer, not to the dataset.

Deviation from Lord et al. (2010) full truncation, kept deliberately for
consistency with dataset/Heston/generate_heston.py: the variance STATE is also
truncated at zero after each step, not only the coefficients.

Outputs (float64)
-----------------
    heston_ma_S_8192x252x8.npy           train    (seed 0)
    heston_ma_v_8192x252x8.npy
    heston_ma_S_test_8192x252x8.npy      test     (seed 1)
    heston_ma_v_test_8192x252x8.npy
    heston_ma_S_disc_8192x252x8.npy      disc     (seed 2)
    heston_ma_v_disc_8192x252x8.npy
    heston_ma_S_val_8192x252x8.npy       val      (seed 3)
    heston_ma_v_val_8192x252x8.npy
    heston_ma_S_valdisc_8192x252x8.npy   valdisc  (seed 4)
    heston_ma_v_valdisc_8192x252x8.npy
    parameters.json                      the frozen per-asset parameters
"""
import json
import os

import numpy as np

# ---------------------------------------------------------------- dimensions
D = 8                    # number of assets
N_SAMPLES = 8192         # paths per split
SEQ_LEN = 252            # daily steps (= 1 year at dt = 1/252)
DT = 1.0 / 252.0
S0 = 100.0
MU = 0.05                # drift, common to all assets (same value as dataset/Heston)

# --------------------- Ait-Sahalia & Kimmel (2007), Table 6, daily SPX, P-measure
KAPPA_BAR = 5.07         # mean-reversion speed  (half-life ln2/kappa ~ 35 trading days)
THETA_BAR = 0.0457       # long-run variance     (~21.4% vol)
ETA_BAR = 0.48           # vol-of-vol
RHO_BAR = -0.767         # leverage correlation
# Feller at the reference point: 2*5.07*0.0457 = 0.4634 > 0.48^2 = 0.2304  -> satisfied

# ------------------------------------------------- cross-sectional dispersion
S_LOG = 0.30             # log-normal sd for (kappa, theta, eta)
S_RHO = 0.25             # sd for leverage in Fisher-z space
BETA_LO, BETA_HI = 0.50, 0.85   # one-factor loadings -> rho_ij in [0.25, 0.72]
PARAM_SEED = 1234        # frozen: the identity of the 8 assets

# Feller safety factor: eta_i <= FELLER_SAFETY * sqrt(2 kappa_i theta_i).
# Capping at the raw boundary (factor 1.0) puts assets exactly ON the Feller
# boundary, where the variance touches zero on a large fraction of paths (up to
# 60% observed at factor 1.0) -- a degenerate marginal, not a modelling choice.
# 0.70 reproduces the Feller ratio of the reference estimate itself,
# 2*kappa_bar*theta_bar / eta_bar^2 = 0.4634 / 0.2304 = 2.01 ~ 1/0.70^2,
# so every asset is at least as Feller-safe as the S&P 500 point estimate.
FELLER_SAFETY = 0.70

SPLITS = {"": 0, "_test": 1, "_disc": 2, "_val": 3, "_valdisc": 4}


def sample_parameters(seed=PARAM_SEED, d=D):
    """Draw the frozen per-asset parameter set. Called once; never per split."""
    rng = np.random.default_rng(seed)

    kappa = np.exp(rng.normal(np.log(KAPPA_BAR), S_LOG, size=d))
    theta = np.exp(rng.normal(np.log(THETA_BAR), S_LOG, size=d))
    eta = np.exp(rng.normal(np.log(ETA_BAR), S_LOG, size=d))
    rho = np.tanh(np.arctanh(RHO_BAR) + S_RHO * rng.normal(size=d))

    # Feller condition 2 kappa theta >= eta^2, enforced by capping vol-of-vol
    # with a safety margin (see FELLER_SAFETY) so no asset sits ON the boundary.
    eta_raw = eta.copy()
    eta = np.minimum(eta, FELLER_SAFETY * np.sqrt(2.0 * kappa * theta))

    v0 = theta.copy()          # start at the long-run level (Andersen 2008 convention)

    # one-factor asset-asset correlation: Sigma_S = beta beta^T + diag(1 - beta^2)
    beta = rng.uniform(BETA_LO, BETA_HI, size=d)
    sigma_s = np.outer(beta, beta) + np.diag(1.0 - beta ** 2)
    np.fill_diagonal(sigma_s, 1.0)          # kill float drift on the diagonal
    chol = np.linalg.cholesky(sigma_s)      # raises if not PD -> fail loudly

    return {
        "kappa": kappa, "theta": theta, "eta": eta, "rho": rho, "v0": v0,
        "beta": beta, "sigma_s": sigma_s, "chol": chol,
        "eta_uncapped": eta_raw,
        "feller_clipped": np.flatnonzero(eta_raw > eta + 1e-15),
    }


def generate_multiasset_heston(seed, params, n_samples=N_SAMPLES, seq_len=SEQ_LEN):
    """One split. Returns S, v each of shape (n_samples, seq_len, d) float64."""
    rng = np.random.default_rng(seed)
    d = params["kappa"].size

    chol = params["chol"]
    kappa, theta, eta, rho = params["kappa"], params["theta"], params["eta"], params["rho"]
    sqrt_1mr2 = np.sqrt(1.0 - rho ** 2)
    sqrt_dt = np.sqrt(DT)

    S = np.empty((n_samples, seq_len, d), dtype=np.float64)
    v = np.empty((n_samples, seq_len, d), dtype=np.float64)
    S[:, 0, :] = S0
    v[:, 0, :] = params["v0"]

    for t in range(1, seq_len):
        # Szimayer et al. (2011) sec. 2.2.2: Cholesky of the d x d matrix only,
        # never the 2d x 2d one.
        z = rng.standard_normal((n_samples, d))
        z_tilde = rng.standard_normal((n_samples, d))
        eps_s = z @ chol.T                              # corr = Sigma_S
        eps_v = rho * eps_s + sqrt_1mr2 * z_tilde       # corr(eps_s_i, eps_v_i) = rho_i

        v_plus = np.maximum(v[:, t - 1, :], 0.0)
        sqrt_v = np.sqrt(v_plus)
        v[:, t, :] = np.maximum(
            v[:, t - 1, :] + kappa * (theta - v_plus) * DT
            + eta * sqrt_v * sqrt_dt * eps_v,
            0.0,
        )
        S[:, t, :] = (
            S[:, t - 1, :] + MU * S[:, t - 1, :] * DT
            + sqrt_v * S[:, t - 1, :] * sqrt_dt * eps_s
        )

    return S, v


def _params_to_json(params):
    return {
        "model": "Szimayer, Dimitroff & Lorenz (2011) parsimonious multi-asset Heston",
        "parameter_source": "Ait-Sahalia & Kimmel (2007) JFE 83(2) Table 6, daily SPX, physical measure",
        "d": D, "n_samples": N_SAMPLES, "seq_len": SEQ_LEN, "dt": DT,
        "S0": S0, "mu": MU, "param_seed": PARAM_SEED,
        "centres": {"kappa_bar": KAPPA_BAR, "theta_bar": THETA_BAR,
                    "eta_bar": ETA_BAR, "rho_bar": RHO_BAR},
        "dispersion": {"s_log": S_LOG, "s_rho": S_RHO,
                       "beta_lo": BETA_LO, "beta_hi": BETA_HI},
        "per_asset": {
            "kappa": params["kappa"].tolist(),
            "theta": params["theta"].tolist(),
            "eta": params["eta"].tolist(),
            "rho": params["rho"].tolist(),
            "v0": params["v0"].tolist(),
            "beta": params["beta"].tolist(),
        },
        "sigma_s": params["sigma_s"].tolist(),
        "feller_2kappatheta": (2.0 * params["kappa"] * params["theta"]).tolist(),
        "feller_eta_squared": (params["eta"] ** 2).tolist(),
        "feller_clipped_assets": params["feller_clipped"].tolist(),
        "splits": {(k or "train").lstrip("_"): s for k, s in SPLITS.items()},
    }


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    params = sample_parameters()

    off = params["sigma_s"][~np.eye(D, dtype=bool)]
    print(f"d={D}  N={N_SAMPLES}  T={SEQ_LEN}  dt=1/{round(1/DT)}")
    print(f"kappa  {np.array2string(params['kappa'], precision=3)}")
    print(f"theta  {np.array2string(params['theta'], precision=4)}")
    print(f"eta    {np.array2string(params['eta'], precision=3)}")
    print(f"rho    {np.array2string(params['rho'], precision=3)}")
    print(f"rho_ij in [{off.min():.3f}, {off.max():.3f}]  mean {off.mean():.3f}")
    print(f"Feller 2*k*th - eta^2  {np.array2string(2*params['kappa']*params['theta'] - params['eta']**2, precision=4)}")
    if params["feller_clipped"].size:
        print(f"  eta capped for assets {params['feller_clipped'].tolist()}")

    with open(os.path.join(out_dir, "parameters.json"), "w") as fh:
        json.dump(_params_to_json(params), fh, indent=2)

    tag = f"{N_SAMPLES}x{SEQ_LEN}x{D}"
    for suffix, seed in SPLITS.items():
        S, v = generate_multiasset_heston(seed, params)
        np.save(os.path.join(out_dir, f"heston_ma_S{suffix}_{tag}.npy"), S)
        np.save(os.path.join(out_dir, f"heston_ma_v{suffix}_{tag}.npy"), v)
        r = np.diff(np.log(S), axis=1)
        ann = r.std(axis=(0, 1)) * np.sqrt(252.0)
        print(
            f"{(suffix or '(train)'):9s} seed={seed}  S{S.shape}  "
            f"S0={S[:, 0, :].mean():.2f}  v0={v[:, 0, :].mean():.5f}  "
            f"ann.vol per asset {np.array2string(ann, precision=3)}"
        )
