# Metrics Results — TimeGAN on Heston (5 Seeds)

Dataset: 8192 Heston price paths, seq\_len=128, canonical parameters
(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, S0=100, v0=0.04, dt=1/250).  
Model: PyTorch TimeGAN, 20k steps (5k embed + 5k supervised + 10k adversarial), 2×A100 GPUs.  
Convention: lower is better for all metrics except A15 Corr (↑).

---

## Results Table (mean ± std across 5 seeds)

| ID | Metric | Mean | Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----:|----:|-------:|-------:|-------:|-------:|-------:|
| A1  | Path MMD²                   | 0.7549 | ± 0.0075 | 0.7513 | 0.7687 | 0.7468 | 0.7521 | 0.7555 |
| A2  | Terminal MMD²               | 0.7412 | ± 0.0117 | 0.7420 | 0.7544 | 0.7245 | 0.7319 | 0.7532 |
| A3  | Increment MMD²              | 0.2329 | ± 0.0011 | 0.2328 | 0.2311 | 0.2339 | 0.2342 | 0.2326 |
| A4  | Volatility MMD              | 4.2441 | ± 0.1689 | 4.2447 | 4.1419 | 4.0930 | 4.1736 | 4.5673 |
| A5  | Terminal SWD                | 11.254 | ± 0.194  | 11.284 | 11.426 | 10.941 | 11.147 | 11.473 |
| A6  | Path SWD                    | 7.4373 | ± 0.1306 | 7.4264 | 7.6016 | 7.2561 | 7.3395 | 7.5627 |
| A7  | Covariance Error            | 199.60 | ± 0.00   | 199.60 | 199.60 | 199.60 | 199.60 | 199.60 |
| A8  | Mean RMSE (terminal)        | 0.4353 | ± 0.3231 | 0.6477 | 0.7213 | 0.7241 | 0.0700 | 0.0132 |
| A9  | Return Std Error            | 1.2239 | ± 0.0061 | 1.2274 | 1.2221 | 1.2136 | 1.2320 | 1.2245 |
| A10 | Return Kurtosis Error       | 71.44  | ± 31.88  | 56.37  | 103.66 | 108.98 | 22.73  | 65.45  |
| A11 | ACF Error (abs returns)     | 0.0967 | ± 0.0510 | 0.1171 | 0.0769 | 0.0167 | 0.1000 | 0.1728 |
| A12 | ACF Error (sq returns)      | 0.0682 | ± 0.0478 | 0.0599 | 0.0374 | 0.0223 | 0.0622 | 0.1592 |
| A13 | Discriminative Score (GRU)  | 0.0782 | ± 0.0652 | 0.1741 | 0.0002 | 0.1048 | 0.0099 | 0.1021 |
| A13 | Discriminative Score (MLP)  | 0.4682 | ± 0.0100 | 0.4823 | 0.4731 | 0.4716 | 0.4548 | 0.4591 |
| A14 | Predictive Score TSTR (GRU) | 0.0526 | ± 0.0046 | 0.0610 | 0.0521 | 0.0525 | 0.0472 | 0.0500 |
| A14 | Predictive Score TSTR (MLP) | 0.0169 | ± 0.0018 | 0.0159 | 0.0163 | 0.0175 | 0.0147 | 0.0201 |
| A15 | Teacher-Sigma Corr (↑)      | 0.0291 | ± 0.0085 | 0.0380 | 0.0289 | 0.0313 | 0.0133 | 0.0342 |
| A15 | Teacher-Sigma RMSE          | 0.1765 | ± 0.0023 | 0.1749 | 0.1767 | 0.1733 | 0.1801 | 0.1773 |

---

## Metric Definitions, Formulas & Perfect Scores

Notation: $X \sim P$ = real paths $(N, T, d)$, $\tilde{X} \sim Q$ = generated paths, $dX_t = X_{t+1} - X_t$ = increments/returns, $k(\cdot,\cdot)$ = RBF kernel.

---

### A1 — Path MMD²  |  perfect **0**

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(x,x')] - 2\,\mathbb{E}[k(x,\tilde{x})] + \mathbb{E}[k(\tilde{x},\tilde{x}')]$$

where $x, x' \sim P$ and $\tilde{x}, \tilde{x}' \sim Q$. Applied to **full paths** (all $T$ time steps concatenated). Measures whether the joint temporal distribution is matched.

---

### A2 — Terminal MMD²  |  perfect **0**

Same MMD² formula applied only to the **terminal values** $X_T \in \mathbb{R}^d$. Tests whether the model reproduces the correct final price distribution.

---

### A3 — Increment MMD²  |  perfect **0**

MMD² applied to **first-order increments** (returns):

$$\text{MMD}^2\!\left(\{dX_t\}, \{d\tilde{X}_t\}\right), \quad dX_t = X_{t+1} - X_t$$

Measures whether the return distribution is correctly reproduced.

---

### A4 — Volatility MMD  |  perfect **0**

MMD applied to the **rolling realised volatility** (window $w=5$):

$$\hat{\sigma}_t = \sqrt{\frac{1}{w}\sum_{s=t-w}^{t-1}(dX_s)^2}$$

Tests whether volatility clustering and the stylised vol distribution are captured.

---

### A5 — Terminal SWD  |  perfect **0**

$$\text{SWD}(P_T, Q_T) = \mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}\!\left[W_1\!\left(\theta_\sharp P_T,\, \theta_\sharp Q_T\right)\right]$$

Sliced Wasserstein Distance on terminal values: average 1-Wasserstein distance over random 1D projections $\theta$. More robust than MMD to heavy tails.

---

### A6 — Path SWD  |  perfect **0**

Same SWD formula applied to **full paths** (path treated as a $T \cdot d$-dimensional point). Captures global path geometry, sensitive to drift and curvature.

---

### A7 — Covariance Error  |  perfect **0**

$$\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F, \quad \Sigma = \text{Cov}(X_T)$$

Frobenius norm of the difference of terminal covariance matrices. For $d=1$ (Heston price), the covariance is a scalar variance — the high value here signals that TimeGAN's terminal variance is systematically wrong.

---

### A8 — Mean RMSE  |  perfect **0**

$$\sqrt{\frac{1}{d}\left\|\mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T]\right\|^2}$$

RMSE between the mean terminal vectors of real and generated paths. Measures systematic bias in the generated final price level.

---

### A9 — Return Std Error  |  perfect **0**

$$\left|\,\sigma(dX) - \sigma(d\tilde{X})\,\right|$$

Absolute error on the standard deviation of returns. Measures whether the overall volatility level is correctly reproduced.

---

### A10 — Return Kurtosis Error  |  perfect **0**

$$\left|\,\kappa(dX) - \kappa(d\tilde{X})\,\right|, \quad \kappa(Z) = \frac{\mathbb{E}[(Z-\mu)^4]}{\sigma^4} - 3$$

Absolute error on excess kurtosis of returns. Measures whether fat tails and leptokurticity of financial returns are reproduced.

---

### A11 — ACF Error (abs returns)  |  perfect **0**

$$\frac{1}{|L|}\sum_{\ell \in L} \left|\,\text{ACF}(|dX|, \ell) - \text{ACF}(|d\tilde{X}|, \ell)\,\right|, \quad L = \{1, 2, 5, 10\}$$

Mean absolute ACF error on **absolute returns** at lags 1,2,5,10. Tests volatility clustering (absolute returns exhibit positive autocorrelation in real financial data).

---

### A12 — ACF Error (sq returns)  |  perfect **0**

Same formula as A11 but applied to **squared returns** $(dX)^2$. Tests the ARCH effect — squared returns are autocorrelated in real markets.

---

### A13 — Discriminative Score  |  perfect **0**

$$\text{DS} = \left|\,\text{acc}_{\text{test}} - 0.5\,\right|$$

A post-hoc classifier (GRU or MLP) is trained on 80% of $\{$real$\} \cup \{$fake$\}$ with labels 1/0, then accuracy is measured on the held-out 20%. **Score 0 = classifier at chance = cannot distinguish real from fake (perfect generator). Score 0.5 = perfect separation (bad generator).** Two architectures are reported: GRU (temporal, same family as TimeGAN's discriminator) and MLP (flattened, architecture-agnostic).

---

### A14 — Predictive Score TSTR  |  perfect **0**

$$\text{PS} = \frac{1}{N \cdot T}\sum_{i,t}\left|\hat{X}_{i,t+1} - X_{i,t+1}\right|$$

Train-on-Synthetic, Test-on-Real MAE. A next-step predictor is trained on **synthetic** data only, then its MAE is evaluated on **real** data. Tests whether temporal dynamics of the synthetic data are useful for real prediction. Two predictors reported: GRU (sequence-to-sequence) and MLP (sliding window of 8 steps).

---

### A15 — Teacher-Sigma Correlation (↑)  |  perfect **1**

$$\rho = \text{Corr}\!\left(\hat{\sigma}_{\text{gen}},\, \sqrt{v_{\text{true}}}\right)$$

where $\hat{\sigma}_{\text{gen},t}$ is the rolling window-5 realised vol estimated from generated paths and $\sqrt{v_{\text{true}}}$ is the true Heston latent volatility. **Higher is better (↑), perfect = 1.** Heston-specific bonus metric.

---

### A15 — Teacher-Sigma RMSE  |  perfect **0**

$$\text{RMSE} = \sqrt{\frac{1}{N \cdot T}\sum_{i,t}\!\left(\hat{\sigma}_{\text{gen},i,t} - \sqrt{v_{\text{true},i,t}}\right)^2}$$

Companion to the correlation; measures absolute accuracy of the latent vol proxy estimated from generated paths.

---

## Files

| File | Description |
|------|-------------|
| `seed_{i}_metrics.json`   | Full per-seed metric dict |
| `metrics_summary.csv`     | Mean ± std across seeds (machine-readable) |
| `plots/seed_{i}_pca.png`  | PCA 2D projection (real vs fake, 500-sample subset) |
| `plots/seed_{i}_tsne.png` | t-SNE 2D projection (real vs fake, 500-sample subset) |
