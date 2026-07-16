# Metrics Results — TimeGAN on Heston (5 Seeds, Fixed Implementation)

Dataset: 8192 Heston price paths, seq\_len=128, canonical parameters
(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, S0=100, v0=0.04, dt=1/250).  
Model: PyTorch TimeGAN, 20k steps (5k embed + 5k supervised + 10k adversarial), 2×A100 GPUs.  
Convention: lower is better for all metrics except A15 Corr (↑).

**v2 — 5 bugs fixed vs v1:** Recovery sigmoid, Phase 1 loss scaling (10·√MSE),
Generator supervised coeff (100× not 10×), Embedder loss form, moment-matching gradient.

---

## Results Table (mean ± std across 5 seeds)

| ID | Metric | Mean | Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----:|----:|-------:|-------:|-------:|-------:|-------:|
| A1  | Path MMD²                   | 0.0174 | ± 0.0142 | 0.0091 | 0.0037 | 0.0319 | 0.0050 | 0.0372 |
| A2  | Terminal MMD²               | 0.0285 | ± 0.0239 | 0.0176 | 0.0078 | 0.0694 | 0.0067 | 0.0409 |
| A3  | Increment MMD²              | 0.0081 | ± 0.0041 | 0.0054 | 0.0072 | 0.0130 | 0.0024 | 0.0125 |
| A4  | Volatility MMD              | 0.3822 | ± 0.2397 | 0.1710 | 0.3442 | 0.6722 | 0.0823 | 0.6413 |
| A5  | Terminal SWD                | 2.731  | ± 0.957  | 2.592  | 1.635  | 4.133  | 1.816  | 3.480  |
| A6  | Path SWD                    | 1.4513 | ± 0.5447 | 1.2560 | 0.8224 | 2.1548 | 0.9886 | 2.0349 |
| A7  | Covariance Error            | 17.751 | ± 6.707  | 8.830  | 18.765 | 14.807 | 29.373 | 16.980 |
| A8  | Mean RMSE (terminal)        | 0.7385 | ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 |
| A9  | Return Std Error            | 0.1519 | ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 |
| A10 | Return Kurtosis Error       | 2.955  | ± 2.099  | 0.015  | 5.360  | 3.768  | 0.958  | 4.672  |
| A11 | ACF Error (abs returns)     | 0.1339 | ± 0.0728 | 0.0821 | 0.1065 | 0.2184 | 0.0421 | 0.2203 |
| A12 | ACF Error (sq returns)      | 0.0919 | ± 0.0386 | 0.0588 | 0.0833 | 0.1318 | 0.0445 | 0.1412 |
| A13 | Discriminative Score (GRU)  | 0.0342 | ± 0.0219 | 0.0325 | 0.0157 | 0.0749 | 0.0145 | 0.0334 |
| A13 | Discriminative Score (MLP)  | 0.1556 | ± 0.1263 | 0.1161 | 0.0417 | 0.2345 | 0.0252 | 0.3605 |
| A14 | Predictive Score TSTR (GRU) | 0.0086 | ± 0.0002 | 0.0086 | 0.0089 | 0.0084 | 0.0087 | 0.0086 |
| A14 | Predictive Score TSTR (MLP) | 0.0091 | ± 0.0004 | 0.0097 | 0.0088 | 0.0094 | 0.0084 | 0.0091 |
| A15 | Teacher-Sigma Corr (↑)      | 0.0031 | ± 0.0101 | 0.0008 | 0.0079 | -0.0100 | -0.0029 | 0.0196 |
| A15 | Teacher-Sigma RMSE          | 0.9659 | ± 0.1237 | 0.9279 | 0.8392 | 1.0714 | 1.1474 | 0.8436 |

---

## Improvement vs v1 (before bug fixes)

| Metric | v1 (buggy) | v2 (fixed) | Improvement |
|--------|:----------:|:----------:|:-----------:|
| A1 Path MMD²         | 0.755 | **0.017** | 44× better |
| A2 Terminal MMD²     | 0.741 | **0.028** | 26× better |
| A3 Increment MMD²    | 0.233 | **0.008** | 29× better |
| A7 Cov Error         | 199.6 | **17.75** | 11× better |
| A9 Return Std Error  | 1.224 | **0.152** | 8× better  |
| A10 Kurtosis Error   | 71.44 | **2.955** | 24× better |
| A13 Disc (MLP)       | 0.468 | **0.156** | 3× better  |
| A14 Pred TSTR (GRU)  | 0.053 | **0.009** | 6× better  |

The five fixes (Recovery sigmoid, loss scaling, supervised coefficient 100×, embedder loss form, moment-matching gradient) collectively reduced distributional error by 10–44× across most metrics.

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

Frobenius norm of the difference of terminal covariance matrices.

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

Absolute error on excess kurtosis of returns. Measures whether fat tails are reproduced.

---

### A11 — ACF Error (abs returns)  |  perfect **0**

$$\frac{1}{|L|}\sum_{\ell \in L} \left|\,\text{ACF}(|dX|, \ell) - \text{ACF}(|d\tilde{X}|, \ell)\,\right|, \quad L = \{1, 2, 5, 10\}$$

Mean absolute ACF error on **absolute returns** at lags 1,2,5,10. Tests volatility clustering.

---

### A12 — ACF Error (sq returns)  |  perfect **0**

Same formula as A11 but applied to **squared returns** $(dX)^2$. Tests the ARCH effect.

---

### A13 — Discriminative Score  |  perfect **0**

$$\text{DS} = \left|\,\text{acc}_{\text{test}} - 0.5\,\right|$$

A post-hoc classifier (GRU or MLP) is trained on 80% of $\{$real$\} \cup \{$fake$\}$ with labels 1/0, then accuracy is measured on the held-out 20%. **Score 0 = classifier at chance = cannot distinguish real from fake (perfect generator). Score 0.5 = perfect separation (bad generator).** Two architectures: GRU (temporal) and MLP (flattened, architecture-agnostic).

---

### A14 — Predictive Score TSTR  |  perfect **0**

$$\text{PS} = \frac{1}{N \cdot T}\sum_{i,t}\left|\hat{X}_{i,t+1} - X_{i,t+1}\right|$$

Train-on-Synthetic, Test-on-Real MAE. A next-step predictor is trained on **synthetic** data only, then its MAE is evaluated on **real** data. Two predictors: GRU and MLP.

---

### A15 — Teacher-Sigma Correlation (↑)  |  perfect **1**

$$\rho = \text{Corr}\!\left(\hat{\sigma}_{\text{gen}},\, \sqrt{v_{\text{true}}}\right)$$

where $\hat{\sigma}_{\text{gen},t}$ is the rolling window-5 realised vol from generated paths and $\sqrt{v_{\text{true}}}$ is the true Heston latent volatility. **Higher is better (↑), perfect = 1.** Heston-specific bonus metric.

---

### A15 — Teacher-Sigma RMSE  |  perfect **0**

$$\text{RMSE} = \sqrt{\frac{1}{N \cdot T}\sum_{i,t}\!\left(\hat{\sigma}_{\text{gen},i,t} - \sqrt{v_{\text{true},i,t}}\right)^2}$$

Companion to the correlation; measures absolute accuracy of the latent vol proxy.

---

## Files

| File | Description |
|------|-------------|
| `seed_{i}_metrics.json`   | Full per-seed metric dict |
| `metrics_summary.csv`     | Mean ± std across seeds (machine-readable) |
| `plots/seed_{i}_pca.png`  | PCA 2D projection (real vs fake, 500-sample subset) |
| `plots/seed_{i}_tsne.png` | t-SNE 2D projection (real vs fake, 500-sample subset) |
