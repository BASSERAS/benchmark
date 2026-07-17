# Metrics — SBTS on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**. A16 ↓.

---

## Results (mean ± std across 5 seeds)

> Filled in after `metrics/compute_all.py --method SBTS` completes.

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|-----------|--------|--------|--------|--------|--------|---------|
| A1  | Path MMD²                   | TBD | — | — | — | — | — | **0** |
| A2  | Terminal MMD²               | TBD | — | — | — | — | — | **0** |
| A3  | Increment MMD²              | TBD | — | — | — | — | — | **0** |
| A4  | Volatility MMD              | TBD | — | — | — | — | — | **0** |
| A5  | Terminal SWD                | TBD | — | — | — | — | — | **0** |
| A6  | Path SWD                    | TBD | — | — | — | — | — | **0** |
| A7  | Covariance Error            | TBD | — | — | — | — | — | **0** |
| A8  | Mean RMSE                   | TBD | — | — | — | — | — | **0** |
| A9  | Return Std Error            | TBD | — | — | — | — | — | **0** |
| A10 | Return Kurtosis Error       | TBD | — | — | — | — | — | **0** |
| A11 | ACF Error (abs returns)     | TBD | — | — | — | — | — | **0** |
| A12 | ACF Error (sq returns)      | TBD | — | — | — | — | — | **0** |
| A13 | Discriminative Score (GRU)  | TBD | — | — | — | — | — | **0** |
| A13 | Discriminative Score (MLP)  | TBD | — | — | — | — | — | **0** |
| A14 | Predictive Score GRU (TSTR) | TBD | — | — | — | — | — | baseline |
| A14 | Predictive Score MLP (TSTR) | TBD | — | — | — | — | — | baseline |
| A15 | Sigma Corr ↑                | TBD | — | — | — | — | — | **1** |
| A15 | Sigma RMSE                  | TBD | — | — | — | — | — | **0** |
| A16 | Tail Survival Error         | TBD | — | — | — | — | — | **0** |

---

## Comparison with TimeGAN (same dataset, same metrics)

> Filled in once both methods have results.

| Metric | SBTS (ours) | TimeGAN (ours) | Better |
|--------|:-----------:|:--------------:|:------:|
| A1 Path MMD²       | TBD | 0.0180 | — |
| A3 Increment MMD²  | TBD | 0.0078 | — |
| A7 Cov Error       | TBD | 17.75  | — |
| A13 Disc (GRU)     | TBD | 0.0499 | — |
| A14 Pred GRU       | TBD | 0.0087 | — |
| A15 Sigma Corr ↑   | TBD | 0.0031 | — |
| A16 Tail Error     | TBD | 0.0216 | — |

---

## Metric Definitions

**Notation.**
$X \sim P$ = real paths $(N \times T \times d)$,
$\tilde{X} \sim Q$ = generated paths,
$r_t = \log X_{t+1} - \log X_t$ = log-returns,
$k(x, x') = \exp(-\|x-x'\|^2 / 2\sigma^2)$ = RBF kernel.

### A1 — Path MMD²

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(X,X')] - 2\mathbb{E}[k(X,\tilde{X})] + \mathbb{E}[k(\tilde{X},\tilde{X}')]$$

Computed on flattened paths $(N, T)$. Measures full trajectory distribution match.

### A2 — Terminal MMD²

Same MMD² kernel, applied to terminal values $X_T$ only. Measures marginal distribution at T.

### A3 — Increment MMD²

MMD² on increments $r_t = X_{t+1} - X_t$, averaged across timesteps. Measures step distribution.

### A4 — Volatility MMD

MMD distance between rolling volatility distributions (window=5). Measures vol clustering.

### A5 — Terminal SWD

Sliced Wasserstein distance on terminal distribution. Projection-based, robust to outliers.

### A6 — Path SWD

Sliced Wasserstein distance on full path. Captures global trajectory shape.

### A7 — Covariance Error (%)

$$\text{Cov Error} = \frac{\|\Sigma_{\text{real}} - \Sigma_{\text{gen}}\|_F}{\|\Sigma_{\text{real}}\|_F} \times 100$$

Measures temporal covariance structure preservation.

### A8 — Mean RMSE

$$\text{Mean RMSE} = \sqrt{\frac{1}{T}\sum_t (\mu_t^{\text{real}} - \mu_t^{\text{gen}})^2}$$

### A9 — Return Std Error

$|\sigma(r^{\text{real}}) - \sigma(r^{\text{gen}})|$ — global return volatility match.

### A10 — Return Kurtosis Error

$|\kappa(r^{\text{real}}) - \kappa(r^{\text{gen}})|$ — excess kurtosis (fat-tail) match.

### A11 — ACF Error (|returns|)

$$\text{ACF Error} = \frac{1}{L}\sum_{\ell=1}^{L}|\hat{\rho}_{|r|}(\ell) - \hat{\rho}_{|\tilde{r}|}(\ell)|, \quad L=20$$

Volatility clustering (leverage effect proxy).

### A12 — ACF Error (squared returns)

Same as A11 with $r_t^2$ instead of $|r_t|$. GARCH-effect proxy.

### A13 — Discriminative Score

Train a classifier to separate real from fake paths. Score = $|\text{accuracy} - 0.5|$.
0 = indistinguishable. 0.5 = perfectly separable (bad generator).
Two variants: **GRU** (2-layer, hidden=8) and **MLP** (flatten → 128 → 64 → 1).

### A14 — Predictive Score (TSTR)

Train-on-Synthetic, Test-on-Real MAE for next-step prediction.

$$\text{Pred Score} = \frac{1}{N_{\text{real}}(T-1)}\sum_{i,t}|\hat{X}_{i,t+1} - X_{i,t+1}|$$

Two variants: **GRU** and **MLP** predictor.

### A15 — Sigma (Heston-specific)

Infer instantaneous variance from generated paths via quadratic variation:
$\hat{v}_t = (\log S_{t+1} - \log S_t)^2 / \Delta t$.
Compare to true $v_t$ from `dataset/Heston/heston_v_8192x128.npy`.

- **Sigma Corr**: Pearson correlation between $\hat{v}_t$ and $v_t$ (↑, perfect=1).
- **Sigma RMSE**: $\sqrt{\text{mean}(\hat{v}_t - v_t)^2}$ (↓, perfect=0).

### A16 — Tail Survival Error

$$\text{Tail Error} = \sqrt{\frac{1}{3}\sum_{q \in \{0.90,0.95,0.99\}} (S^{\text{real}}(q) - S^{\text{gen}}(q))^2}$$

where $S(q)$ is the survival probability (fraction of paths exceeding the $q$-quantile of the
real terminal distribution). Tests fat-tail reproduction.

---

## CRPS, MAE, RMSE (Path Shadowing context)

These three metrics are used in the **Path Shadowing MC** evaluation
(`results/Heston/SBTS/path_shadowing/`):

**CRPS** (Continuous Ranked Probability Score):

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}_{z \ge y})^2 \, dz = \mathbb{E}_{X \sim F}|X - y| - \frac{1}{2}\mathbb{E}_{X,X' \sim F}|X - X'|$$

Lower is better. A proper scoring rule: minimised in expectation only by the true distribution.
Reduces to MAE when $F$ is a point mass.

**MAE** (Mean Absolute Error):

$$\text{MAE} = \frac{1}{H}\sum_{u=t+1}^{t+H} |\bar{X}(u) - X_{\text{real}}(u)|$$

where $\bar{X}$ is the ensemble mean forecast.

**RMSE** (Root Mean Squared Error):

$$\text{RMSE} = \sqrt{\frac{1}{H}\sum_{u=t+1}^{t+H} (\bar{X}(u) - X_{\text{real}}(u))^2}$$
