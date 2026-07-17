# RW — Calibrated Random Walk / GBM baseline

**Method:** Geometric Brownian Motion (GBM) with empirically calibrated parameters.
No neural network, no training, no temporal structure. Pure i.i.d. Gaussian log-returns.

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Generation:** `R_t ~ N(μ_dt, σ_dt²)` i.i.d., with μ_dt=0.000132 and σ_dt=0.012647 calibrated
from the Heston training set. Price reconstruction: `S_{t+1} = S_t × exp(R_t)`.

**Purpose:** Sanity-check baseline. Shows which benchmark metrics can be passed trivially
by a matched-marginal Gaussian model with no temporal structure.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**. A16 ↓.

---

## Results (mean ± std across 5 seeds)

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|-----------|--------|--------|--------|--------|--------|---------|
| A1  | Path MMD²                   | 0.0049 ± 0.0018 | 0.0067 | 0.0068 | 0.0056 | 0.0026 | 0.0028 | **0** |
| A2  | Terminal MMD²               | 0.0053 ± 0.0022 | 0.0083 | 0.0063 | 0.0062 | 0.0028 | 0.0028 | **0** |
| A3  | Increment MMD²              | 0.0014 ± 0.0000 | 0.0015 | 0.0014 | 0.0015 | 0.0014 | 0.0014 | **0** |
| A4  | Volatility MMD              | 0.054 ± 0.006   | 0.0635 | 0.0542 | 0.0569 | 0.0477 | 0.0489 | **0** |
| A5  | Terminal SWD                | 1.865 ± 0.364   | 2.360  | 2.091  | 1.983  | 1.470  | 1.421  | **0** |
| A6  | Path SWD                    | 1.163 ± 0.254   | 1.439  | 1.395  | 1.260  | 0.851  | 0.870  | **0** |
| A7  | Covariance Error            | 17.47 ± 3.01    | 16.29  | 22.52  | 15.04  | 19.14  | 14.34  | **0** |
| A8  | Mean RMSE                   | 0.090 ± 0.059   | 0.124  | 0.019  | 0.174  | 0.106  | 0.027  | **0** |
| A9  | Return Std Error            | 0.049 ± 0.001   | 0.0494 | 0.0464 | 0.0505 | 0.0492 | 0.0483 | **0** |
| A10 | Return Kurtosis Error       | 0.482 ± 0.008   | 0.484  | 0.470  | 0.479  | 0.482  | 0.494  | **0** |
| A11 | ACF Error (abs returns)     | 0.033 ± 0.000   | 0.0332 | 0.0334 | 0.0331 | 0.0325 | 0.0321 | **0** |
| A12 | ACF Error (sq returns)      | 0.029 ± 0.000   | 0.0288 | 0.0292 | 0.0295 | 0.0285 | 0.0283 | **0** |
| A13 | Discriminative Score (GRU)  | 0.010 ± 0.008   | 0.001  | 0.025  | 0.009  | 0.011  | 0.005  | **0** |
| A13 | Discriminative Score (MLP)  | 0.021 ± 0.018   | 0.024  | 0.005  | 0.000  | 0.027  | 0.051  | **0** |
| A14 | Predictive Score GRU (TSTR) | 0.0083 ± 0.0000 | 0.0083 | 0.0083 | 0.0083 | 0.0083 | 0.0083 | baseline |
| A14 | Predictive Score MLP (TSTR) | 0.0087 ± 0.0001 | 0.0084 | 0.0087 | 0.0086 | 0.0088 | 0.0088 | baseline |
| A15 | Sigma Corr ↑                | −0.003 ± 0.004  | −0.010 | −0.002 | −0.002 | +0.001 | +0.001 | **1** |
| A15 | Sigma RMSE                  | 0.085 ± 0.000   | 0.0850 | 0.0847 | 0.0846 | 0.0845 | 0.0844 | **0** |
| A16 | Tail Survival Error         | 0.0075 ± 0.0003 | 0.0074 | 0.0070 | 0.0077 | 0.0077 | 0.0075 | **0** |

---

## What does the RW reveal?

The RW wins **17/21** benchmark metrics against both SBTS and TimeGAN. The 4 metrics where
RW loses to at least one method:

| Metric | RW score | Best method | Why RW fails |
|--------|:--------:|:-----------:|:-------------|
| A10 Kurtosis Error ↓ | 0.482 | **SBTS** (0.119) | GBM has Gaussian returns → no excess kurtosis; Heston has fat tails via stochastic vol |
| A15 Sigma Corr ↑ | −0.003 | **SBTS** (0.005) | GBM has constant vol → no correlation with Heston's true vₜ path |
| PS-MC CRPS H=32 ↓ | — | **SBTS** (2.761) | Not computed for RW; path shadowing exploits temporal matching |
| PS-MC CRPS H=64 ↓ | — | **SBTS** (3.900) | Not computed for RW |

**Interpretation:** Most standard evaluation metrics (MMD, SWD, discriminative score, predictive
score) can be passed by a calibrated GBM with no temporal structure. Only kurtosis (fat tails),
stochastic volatility (Sigma Corr), and path-shadowing (PS-MC) reliably distinguish the Heston
process from a matched-marginal Gaussian. This is a known limitation of standard metrics for
financial time series benchmarking.

---

→ Cross-method comparison: [`results/README.md`](../../README.md)
