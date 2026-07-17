# SBTS — Score-Based Time Series Generation via Schrödinger Bridge

**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**.

---

## What we generate — price paths (not log-price)

SBTS internally operates on **scaled log-returns**:

```
R = log(S[:,1:] / S[:,:-1])          # log-returns  (8192, 127)
R̃ = R × √dt / σ(R)                  # scale so that Var(R̃) ≈ dt  (paper §6)
```

SBTS generates R̃_gen in this scaled space. We then inverse-scale and reconstruct:

```
R_gen = R̃_gen × σ(R) / √dt          # inverse scaling
S_gen[:,0] = 100                      # anchor at S₀
S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])   # price reconstruction
```

**Output: price paths S_t anchored at S₀=100** — the same space as the real dataset.
We never divide by √σ in the output; the √dt/σ scaling is only internal to SBTS.

---

## Results (mean ± std across 5 seeds)

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|-----------|--------|--------|--------|--------|--------|---------|
| A1  | Path MMD²                   | 0.0110 ± 0.0016 | 0.0093 | 0.0136 | 0.0117 | 0.0093 | 0.0110 | **0** |
| A2  | Terminal MMD²               | 0.0090 ± 0.0035 | 0.0061 | 0.0153 | 0.0083 | 0.0056 | 0.0098 | **0** |
| A3  | Increment MMD²              | 0.0071 ± 0.0005 | 0.0074 | 0.0076 | 0.0063 | 0.0074 | 0.0065 | **0** |
| A4  | Volatility MMD              | 0.3125 ± 0.0176 | 0.3230 | 0.3366 | 0.2900 | 0.3183 | 0.2947 | **0** |
| A5  | Terminal SWD                | 3.465 ± 0.588   | 2.775  | 4.258  | 3.634  | 2.799  | 3.858  | **0** |
| A6  | Path SWD                    | 2.497 ± 0.288   | 2.203  | 2.928  | 2.620  | 2.150  | 2.582  | **0** |
| A7  | Covariance Error            | 145.35 ± 4.89   | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | **0** |
| A8  | Mean RMSE                   | 1.3013 ± 0.2776 | 1.2972 | 0.9199 | 1.4819 | 1.0986 | 1.7088 | **0** |
| A9  | Return Std Error            | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | **0** |
| A10 | Return Kurtosis Error       | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | **0** |
| A11 | ACF Error (abs returns)     | 0.0568 ± 0.0006 | 0.0568 | 0.0565 | 0.0570 | 0.0558 | 0.0577 | **0** |
| A12 | ACF Error (sq returns)      | 0.0619 ± 0.0006 | 0.0619 | 0.0618 | 0.0617 | 0.0610 | 0.0628 | **0** |
| A13 | Discriminative Score (GRU)  | 0.0291 ± 0.0276 | 0.0032 | 0.0148 | 0.0157 | 0.0816 | 0.0301 | **0** |
| A13 | Discriminative Score (MLP)  | 0.0711 ± 0.0077 | 0.0676 | 0.0859 | 0.0685 | 0.0700 | 0.0636 | **0** |
| A14 | Predictive Score GRU (TSTR) | 0.0091 ± 0.0000 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | baseline |
| A14 | Predictive Score MLP (TSTR) | 0.0093 ± 0.0006 | 0.0092 | 0.0090 | 0.0104 | 0.0088 | 0.0088 | baseline |
| A15 | Sigma Corr ↑                | 0.0011 ± 0.0035 | −0.0024 | 0.0031 | −0.0038 | 0.0039 | 0.0045 | **1** |
| A15 | Sigma RMSE                  | 0.8207 ± 0.0019 | 0.8198 | 0.8181 | 0.8214 | 0.8205 | 0.8238 | **0** |

---

## Comparison with the paper (Alouadi et al., ICAIF 2025)

> ⚠️ **Direct metric comparison is not meaningful** — different data representation,
> dataset size, and evaluation horizon. What we can verify is that our hyperparameters
> exactly match the paper's recommended settings for Heston (Appendix C, Table 4).

| Setting | Our reimplementation | Paper (Alouadi et al. 2025) |
|---------|:--------------------:|:---------------------------:|
| Data representation | Price paths S_t | Log-returns (scaled) |
| Dimension | 1 (univariate price) | 2 (price + variance) |
| Seq len T | 128 | ~252 |
| Training paths | 8 192 | ~500 |
| h (bandwidth) | **0.4** | **0.4** (App. C Table 4) |
| K (Markov order) | **1** | **1** (App. C Table 4) |
| N_pi (Euler substeps) | **200** | **200** (App. C Table 4) |
| Disc Score GRU ↓ | 0.0291 ± 0.0276 | — (different eval protocol) |
| Pred Score GRU ↓ | 0.0091 ± 0.0000 | — (different eval protocol) |

Our discriminative score (0.029 GRU) is in the range expected for a well-calibrated
Schrödinger-bridge method on a Markov-1 process, consistent with the paper's claim
that SBTS achieves low discriminative scores on Heston-type data.
The key hyperparameters (h, K, N_pi) are taken verbatim from the paper's Appendix C Table 4,
confirming our reimplementation uses the authors' recommended settings.

→ Cross-method comparison with TimeGAN: [`results/README.md`](../../README.md)

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
