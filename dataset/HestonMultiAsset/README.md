# Multi-Asset Heston Dataset, d = 8, seq_len = 252

## Quick description

Synthetic price/variance paths for **8 correlated assets** generated from the **parsimonious
multi-asset Heston model** of Szimayer, Dimitroff & Lorenz (2011) using an Euler-Maruyama
full-truncation scheme. Every path has **252 time steps** (one trading year, dt = 1/252) and starts
at $S_0 = 100$ for all assets.

Single-asset parameters are centred on the maximum-likelihood estimates of Ait-Sahalia & Kimmel
(2007), Table 6, for daily S&P 500 data under the **physical measure** — i.e. estimated at exactly
this sampling frequency and under exactly the measure we simulate. Per-asset heterogeneity is
induced by randomising the parameter vector once, following the randomised-parameter protocol of
Alouadi et al. (2025).

The benchmark uses **five disjoint 8 192-path splits**, all drawn from the *same* SDE with the
*same* frozen parameters, differing only by RNG seed.

## Stochastic differential equations

For each asset $i = 1, \dots, d$:

$$
dS_i = \mu S_i dt + \sqrt{v_i} S_i dW_i
$$

$$
dv_i = \kappa_i (\theta_i - v_i) dt + \eta_i \sqrt{v_i} \left( \rho_i dW_i + \sqrt{1 - \rho_i^2} d\tilde{W}_i \right)
$$

with $\mathrm{corr}(W) = \Sigma^S = (\rho_{ij})$, $\mathrm{corr}(\tilde{W}) = I_d$, and $W$
independent of $\tilde{W}$ (their Assumption 2.1).

**Why this parameterisation.** The model is *parsimonious*: the $d$ marginals are ordinary
one-dimensional Heston models, and the only coupling is the $d(d-1)/2$ spot correlations
$\rho_{ij}$. By their Proposition 2.2 this induces, for $i \neq j$,

$$
\mathrm{corr}(dS_i, dS_j) = \rho_{ij}
$$

$$
\mathrm{corr}(dS_i, dv_j) = \rho_{ij} \rho_j
$$

$$
\mathrm{corr}(dv_i, dv_j) = \rho_{ij} \rho_i \rho_j
$$

so cross-asset leverage and vol-vol correlation come **for free** from the spot correlations — no
extra parameters. The full $2d \times 2d$ correlation matrix is positive semi-definite **by
construction**, because the correlation matrix of $(W, \tilde{W})$ is block diagonal,
$\mathrm{diag}(\Sigma^S, I_d)$. No nearest-correlation-matrix repair step is needed, and the
simulation only ever factorises the $d \times d$ matrix $\Sigma^S$, never the $2d \times 2d$ one
(their §2.2.2).

## Parameters

### Global

| Symbol | Meaning | Value |
|--------|---------|-------|
| $d$ | Number of assets | 8 |
| $\mu$ | Drift (common to all assets) | 0.05 |
| $S_0$ | Initial price (all assets) | 100.0 |
| $dt$ | Time step | 1/252 (daily) |
| $T$ | Sequence length | 252 |
| $N$ | Paths per split | 8 192 |

### Reference point — Ait-Sahalia & Kimmel (2007), Table 6

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\bar{\kappa}$ | Mean-reversion speed | 5.07 |
| $\bar{\theta}$ | Long-run variance | 0.0457 (≈ 21.4 % vol) |
| $\bar{\eta}$ | Vol-of-vol | 0.48 |
| $\bar{\rho}$ | Leverage correlation | −0.767 |

Half-life $\ln 2 / \bar{\kappa} \approx 35$ trading days, so a 252-step path spans about 7
half-lives — mean reversion is comfortably identifiable at this horizon.

### Cross-sectional randomisation (frozen, `PARAM_SEED = 1234`)

$\kappa_i, \theta_i, \eta_i$ are lognormal around their reference values with log-sd 0.30;
$\rho_i$ is drawn in Fisher-z space with sd 0.25. Asset-asset correlation is one-factor,
$\Sigma^S = \beta \beta^\top + \mathrm{diag}(1 - \beta_i^2)$ with $\beta_i \sim U[0.50, 0.85]$.

**Parameters are drawn ONCE and held FIXED across all paths and all five splits.** Redrawing them
per path would leave the eight "assets" with no identity, and redrawing per split would mean the
splits are not draws from a common distribution. Only the Brownian noise differs between splits.

| Asset | $\kappa_i$ | $\theta_i$ | $\eta_i$ | $\rho_i$ | $v_0^i$ | $\beta_i$ | ann. vol |
|:-----:|-----------:|-----------:|---------:|---------:|--------:|----------:|---------:|
| 0 | 3.134 | 0.0277 | 0.292 | −0.685 | 0.0277 | 0.554 | 0.167 |
| 1 | 5.168 | 0.0507 | 0.507 | −0.728 | 0.0507 | 0.847 | 0.226 |
| 2 | 6.332 | 0.0392 | 0.493 | −0.770 | 0.0392 | 0.564 | 0.199 |
| 3 | 5.308 | 0.0680 | 0.355 | −0.766 | 0.0680 | 0.829 | 0.261 |
| 4 | 6.570 | 0.0353 | 0.477 | −0.828 | 0.0353 | 0.530 | 0.188 |
| 5 | 12.149 | 0.0534 | 0.604 | −0.824 | 0.0534 | 0.664 | 0.231 |
| 6 | 3.253 | 0.0313 | 0.316 | −0.592 | 0.0313 | 0.790 | 0.177 |
| 7 | 6.733 | 0.0239 | 0.339 | −0.739 | 0.0239 | 0.598 | 0.155 |

$v_0^i = \theta_i$: every asset starts at its own long-run level (Andersen 2008 convention).

Off-diagonal $\rho_{ij} \in [0.294, 0.702]$, mean 0.450. The exact $8 \times 8$ matrix $\Sigma^S$
and all per-asset values are dumped to `parameters.json`.

### Feller condition

Enforced by capping vol-of-vol at $\eta_i \leq 0.70 \sqrt{2 \kappa_i \theta_i}$, which binds for
assets 0, 1, 2, 4, 6.

**The 0.70 is not arbitrary and the margin is not cosmetic.** Capping at the raw boundary
(factor 1.0) puts those assets exactly *on* the Feller boundary, where the variance touches zero on
a large fraction of paths — up to **60.6 %** for asset 1 in the first build of this dataset. That is
a degenerate marginal, not a modelling choice. The factor 0.70 reproduces the Feller ratio of the
reference estimate itself,

$$
\frac{2 \bar{\kappa} \bar{\theta}}{\bar{\eta}^2} = \frac{0.4634}{0.2304} = 2.01 \approx 0.70^{-2}
$$

so every asset ends up at least as Feller-safe as the S&P 500 point estimate. With the margin in
place, the worst asset touches $v = 0$ on 7.8 % of paths and only 0.015 % of all cells are zero.

| Asset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-------|---|---|---|---|---|---|---|---|
| $2 \kappa_i \theta_i - \eta_i^2$ | 0.0886 | 0.2671 | 0.2531 | 0.5953 | 0.2366 | 0.9326 | 0.1038 | 0.2069 |
| frac. paths touching $v = 0$ | 1.8 % | 4.9 % | 7.6 % | 0.0 % | 7.8 % | 1.0 % | 2.1 % | 1.0 % |

## The five benchmark splits

Each split is an independent 8 192-path draw (same SDE, same frozen parameters, distinct seed).
Keeping them disjoint is what makes every reported number a genuine **out-of-sample** measurement.

| Split | Seed | Role |
|-------|:----:|------|
| **train** | 0 | The only data a generator ever sees. Every method is fit on this split. |
| **test** | 1 | Held-out **real reference**. Every A/B metric scores generated paths against this split, never against train. The **Perfect floor** is a fresh draw scored against this same test set. |
| **disc** | 2 | A third independent real split used **only** by the discriminative / predictive metric classifiers (A18 / A19) as their real-vs-fake training data, so those classifiers never touch the test split they are evaluated on. |
| **val** | 3 | Validation counterpart of `test`, for model selection and hyperparameter search. Never used for a reported number. |
| **valdisc** | 4 | Validation counterpart of `disc`, for the classifiers during model selection. |

## Storage contract

Consistent with `dataset/Heston`: arrays store **PRICE** paths $S_t$ (and the latent variance
$v_t$), **NOT** log returns. Arithmetic Euler is applied to $S$ directly. Any log-return transform
belongs to the model-input layer, not to the dataset.

**Deviation from Lord, Koekkoek & van Dijk (2010) full truncation**, kept deliberately for
consistency with `dataset/Heston/generate_heston.py`: the variance **state** is also truncated at
zero after each step, not only the coefficients appearing in the drift and diffusion.

## Files

All ten split arrays are **float64, shape (8192, 252, 8)** with axes `(path, time, asset)`.

| File | Split | Field | Description |
|------|-------|-------|-------------|
| `heston_ma_S_8192x252x8.npy` | train (seed 0) | $S_t$ | Price paths |
| `heston_ma_v_8192x252x8.npy` | train (seed 0) | $v_t$ | Variance paths |
| `heston_ma_S_test_8192x252x8.npy` | test (seed 1) | $S_t$ | Price paths, scoring reference |
| `heston_ma_v_test_8192x252x8.npy` | test (seed 1) | $v_t$ | Variance paths |
| `heston_ma_S_disc_8192x252x8.npy` | disc (seed 2) | $S_t$ | Price paths, A18/A19 classifier real data |
| `heston_ma_v_disc_8192x252x8.npy` | disc (seed 2) | $v_t$ | Variance paths |
| `heston_ma_S_val_8192x252x8.npy` | val (seed 3) | $S_t$ | Price paths, model selection |
| `heston_ma_v_val_8192x252x8.npy` | val (seed 3) | $v_t$ | Variance paths |
| `heston_ma_S_valdisc_8192x252x8.npy` | valdisc (seed 4) | $S_t$ | Price paths, classifier model selection |
| `heston_ma_v_valdisc_8192x252x8.npy` | valdisc (seed 4) | $v_t$ | Variance paths |
| `parameters.json` | , | , | The frozen per-asset parameters, $\Sigma^S$, Feller diagnostics, split→seed map. **Tracked.** |
| `generate_heston_multiasset.py` | , | , | CPU (numpy, float64) generator for all five splits |

> ⚠️ **The `.npy` files are NOT committed.** Each is 126 MiB — over GitHub's 100 MiB per-file hard
> limit, 1.26 GiB in total, and too large for the free LFS quota. They are git-ignored and must be
> regenerated locally with the command below. `parameters.json` **is** committed, so the model is
> pinned in the repo even though the samples are not.

## Reproduce

```bash
cd dataset/HestonMultiAsset
OMP_NUM_THREADS=8 taskset -c 0-7 python generate_heston_multiasset.py
```

Takes about 4 minutes on 8 cores and writes all ten arrays plus `parameters.json`.

Regeneration is bitwise-exact on a fixed numpy version (verified against numpy 2.4.6). Across numpy
versions the *model* is still reproduced exactly — the parameters come from `PARAM_SEED = 1234` and
are pinned in `parameters.json` — but `np.random.Generator` streams are not guaranteed stable across
releases (unlike the legacy `RandomState`), so arrays regenerated under a different numpy are
statistically, not bitwise, identical.

## Sanity checks

Verified on the committed build:

| Check | Result |
|-------|--------|
| $S_0$ exact | mean 100.000000, std 0 |
| $S_t > 0$ everywhere | True |
| $\mathrm{corr}(dS_i, dv_i)$ vs $\rho_i$ | max abs diff **0.010** |
| $\mathrm{corr}(dS_i, dS_j)$ vs $\rho_{ij}$ | max abs diff 0.050 (off-diagonal) |
| $\mathrm{corr}(dv_i, dv_j)$ vs $\rho_{ij} \rho_i \rho_j$ | max abs diff 0.022 (off-diagonal) |
| $\mathrm{corr}(dS_i, dv_j)$ vs $\rho_{ij} \rho_j$ | max abs diff 0.038 (off-diagonal) |
| $v = 0$ cells | 0.015 % |

**On the off-diagonal gaps.** The realised-return correlations sit systematically *below* the
Brownian correlations $\rho_{ij}$. This is the model's own behaviour, not an implementation bug:
Remark 2.5(1) of Szimayer et al. derives from Cauchy-Schwarz that the estimated correlation
satisfies $|\hat{\rho}_{ij}| \leq |\rho_{ij}|$ asymptotically, because the stochastic volatility
factors attenuate the realised correlation. The discriminating control is the **leverage**
correlation $\mathrm{corr}(dS_i, dv_i)$, which involves no such attenuation and matches $\rho_i$ to
0.010 — confirming the Cholesky/noise construction is correct. Consistently, halving the effective
vol-of-vol via the Feller margin halved the spot-correlation gap (0.096 → 0.050).

## References

- Szimayer, A., Dimitroff, G. & Lorenz, S. (2011). A parsimonious multi-asset Heston model:
  calibration and derivative pricing. *International Journal of Theoretical and Applied Finance*
  14(8), 1299–1333.
- Aït-Sahalia, Y. & Kimmel, R. (2007). Maximum likelihood estimation of stochastic volatility
  models. *Journal of Financial Economics* 83(2), 413–452.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with
  applications to bond and currency options. *Review of Financial Studies* 6(2), 327–343.
- Lord, R., Koekkoek, R. & van Dijk, D. (2010). A comparison of biased simulation schemes for
  stochastic volatility models. *Quantitative Finance* 10(2), 177–194.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model.
  *Journal of Computational Finance* 11(3), 1–42.
