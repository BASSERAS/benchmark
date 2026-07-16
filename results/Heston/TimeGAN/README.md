# Metrics Results — TimeGAN on Heston (5 Seeds, v2 Fixed Implementation)

Dataset: 8192 Heston price paths, seq\_len=128, canonical parameters
(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, S0=100, v0=0.04, dt=1/250).  
Model: PyTorch TimeGAN v2, 20k steps (5k embed + 5k supervised + 10k adversarial), 2×A100 GPUs.  
Convention: lower is better for all metrics **except A15 Corr (↑)**.

---

## Results Table (mean ± std across 5 seeds)

| ID | Metric | Mean | Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----:|----:|-------:|-------:|-------:|-------:|-------:|
| A1  | Path MMD²                   | 0.0180 | ± 0.0147 | 0.0095 | 0.0035 | 0.0345 | 0.0054 | 0.0373 |
| A2  | Terminal MMD²               | 0.0296 | ± 0.0235 | 0.0202 | 0.0086 | 0.0646 | 0.0051 | 0.0494 |
| A3  | Increment MMD²              | 0.0078 | ± 0.0037 | 0.0054 | 0.0076 | 0.0117 | 0.0023 | 0.0121 |
| A4  | Volatility MMD              | 0.3798 | ± 0.2351 | 0.1746 | 0.3673 | 0.6468 | 0.0709 | 0.6394 |
| A5  | Terminal SWD                | 2.850  | ± 1.079  | 2.765  | 1.820  | 4.339  | 1.552  | 3.772  |
| A6  | Path SWD                    | 1.5006 | ± 0.5834 | 1.2791 | 0.8390 | 2.3493 | 1.0207 | 2.0149 |
| A7  | Covariance Error            | 17.751 | ± 6.707  | 8.830  | 18.765 | 14.807 | 29.373 | 16.980 |
| A8  | Mean RMSE (terminal)        | 0.7385 | ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 |
| A9  | Return Std Error            | 0.1519 | ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 |
| A10 | Return Kurtosis Error       | 2.955  | ± 2.099  | 0.015  | 5.360  | 3.768  | 0.958  | 4.672  |
| A11 | ACF Error (abs returns)     | 0.1339 | ± 0.0728 | 0.0821 | 0.1065 | 0.2184 | 0.0421 | 0.2203 |
| A12 | ACF Error (sq returns)      | 0.0919 | ± 0.0386 | 0.0588 | 0.0833 | 0.1318 | 0.0445 | 0.1412 |
| A13 | Discriminative Score (GRU)  | 0.0499 | ± 0.0336 | 0.0468 | 0.0224 | 0.1088 | 0.0133 | 0.0581 |
| A13 | Discriminative Score (MLP)  | 0.1508 | ± 0.1415 | 0.0368 | 0.0352 | 0.2901 | 0.0374 | 0.3544 |
| A14 | Predictive Score TSTR (GRU) | 0.0087 | ± 0.0002 | 0.0085 | 0.0090 | 0.0085 | 0.0088 | 0.0085 |
| A14 | Predictive Score TSTR (MLP) | 0.0090 | ± 0.0005 | 0.0090 | 0.0087 | 0.0090 | 0.0084 | 0.0099 |
| A15 | Teacher-Sigma Corr (↑)      | 0.0031 | ± 0.0101 | 0.0008 | 0.0079 | -0.010 | -0.003 | 0.0196 |
| A15 | Teacher-Sigma RMSE          | 0.9659 | ± 0.1237 | 0.9279 | 0.8392 | 1.0714 | 1.1474 | 0.8436 |

---

## Comparison with Yoon et al. NeurIPS 2019 (Table 2)

> ⚠️ **Not a direct comparison.** The paper uses different data (Sines d=5 T=24, Stocks d=6) and a **2-layer LSTM** post-hoc classifier. We use Heston (d=1, T=128) with a GRU and MLP. Numbers are shown for orientation only.

| Metric | Paper — Sines (LSTM) | Paper — Stocks (LSTM) | Ours — Heston (GRU) | Ours — Heston (MLP) |
|--------|:-------------------:|:--------------------:|:-------------------:|:-------------------:|
| **Discriminative Score** ↓ | **0.011 ± 0.008** | **0.102 ± 0.021** | 0.050 ± 0.034 | 0.151 ± 0.142 |
| **Predictive Score (TSTR)** ↓ | **0.093 ± 0.019** | **0.038 ± 0.001** | 0.009 ± 0.000 | 0.009 ± 0.001 |

**Key differences:**
- **Discriminative score convention**: `|accuracy − 0.5|`. Score 0 = indistinguishable (perfect). Score 0.5 = trivially separated (bad). Paper uses a 2-layer LSTM; we report both GRU and MLP.
- **Predictive score convention**: Train-on-Synthetic, Test-on-Real MAE (TSTR). Paper uses a 2-layer LSTM predictor; we use GRU and MLP.
- **Our predictive score is much lower** (0.009 vs 0.093 on Sines) because Heston is a 1D geometric process — next-step prediction is easier than 5D Sines, and our generated paths closely match the real distribution (A1 MMD = 0.018 vs original TF1 run of 0.755).
- **Our discriminative GRU score (0.050) is between** the paper's Sines (0.011) and Stocks (0.102) results, consistent with Heston being a moderately challenging 1D financial process.

---

## Improvement: v1 (buggy) → v2 (fixed)

5 bugs were fixed in `TimeGan/timegan_torch.py` (Recovery sigmoid, Phase-1 loss scaling,
Generator supervised coeff 100×, Embedder loss form, moment-matching gradient).

| Metric | v1 (buggy) | v2 (fixed) | Factor |
|--------|:----------:|:----------:|:------:|
| A1 Path MMD²        | 0.755 | **0.018** | 42× |
| A2 Terminal MMD²    | 0.741 | **0.030** | 25× |
| A3 Increment MMD²   | 0.233 | **0.008** | 29× |
| A7 Cov Error        | 199.6 | **17.75** | 11× |
| A9 Std Error        | 1.224 | **0.152** |  8× |
| A10 Kurtosis Error  | 71.44 | **2.955** | 24× |
| A13 Disc (GRU)      | 0.078 | **0.050** | 1.6× |
| A14 Pred TSTR (GRU) | 0.053 | **0.009** |  6× |

---

## Metric Definitions, Formulas & Perfect Scores

Notation: $X \sim P$ = real paths $(N, T, d)$, $\tilde{X} \sim Q$ = generated paths,
$dX_t = X_{t+1} - X_t$ = increments/returns, $k(\cdot,\cdot)$ = RBF kernel.

---

### A1 — Path MMD²  |  perfect **0**

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(x,x')] - 2\,\mathbb{E}[k(x,\tilde{x})] + \mathbb{E}[k(\tilde{x},\tilde{x}')]$$

Applied to **full paths** (all $T$ steps concatenated). Measures whether the joint temporal distribution is matched.

---

### A2 — Terminal MMD²  |  perfect **0**

Same MMD² applied only to terminal values $X_T$. Tests whether the final price distribution is correct.

---

### A3 — Increment MMD²  |  perfect **0**

$$\text{MMD}^2\!\left(\{dX_t\}, \{d\tilde{X}_t\}\right), \quad dX_t = X_{t+1} - X_t$$

Measures whether the return distribution is correctly reproduced.

---

### A4 — Volatility MMD  |  perfect **0**

MMD on rolling realised volatility (window $w=5$):

$$\hat{\sigma}_t = \sqrt{\frac{1}{w}\sum_{s=t-w}^{t-1}(dX_s)^2}$$

Tests volatility clustering and the stylised vol distribution.

---

### A5 — Terminal SWD  |  perfect **0**

$$\text{SWD}(P_T, Q_T) = \mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}\!\left[W_1\!\left(\theta_\sharp P_T,\, \theta_\sharp Q_T\right)\right]$$

Sliced Wasserstein Distance on terminal values. More robust to heavy tails than MMD.

---

### A6 — Path SWD  |  perfect **0**

Same SWD on full paths (path as $T \cdot d$-dimensional point). Captures global path geometry.

---

### A7 — Covariance Error  |  perfect **0**

$$\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F, \quad \Sigma = \text{Cov}(X_T)$$

Frobenius norm of terminal covariance difference.

---

### A8 — Mean RMSE  |  perfect **0**

$$\sqrt{\frac{1}{d}\left\|\mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T]\right\|^2}$$

Measures systematic bias in generated terminal price level.

---

### A9 — Return Std Error  |  perfect **0**

$$\left|\,\sigma(dX) - \sigma(d\tilde{X})\,\right|$$

Absolute error on return standard deviation (volatility level).

---

### A10 — Return Kurtosis Error  |  perfect **0**

$$\left|\,\kappa(dX) - \kappa(d\tilde{X})\,\right|, \quad \kappa(Z) = \frac{\mathbb{E}[(Z-\mu)^4]}{\sigma^4} - 3$$

Absolute error on excess kurtosis (fat-tail matching).

---

### A11 — ACF Error (abs returns)  |  perfect **0**

$$\frac{1}{|L|}\sum_{\ell \in L} \left|\,\text{ACF}(|dX|, \ell) - \text{ACF}(|d\tilde{X}|, \ell)\,\right|, \quad L = \{1, 2, 5, 10\}$$

Volatility clustering test via absolute-return autocorrelation.

---

### A12 — ACF Error (sq returns)  |  perfect **0**

Same as A11 on squared returns $(dX)^2$. Tests the ARCH effect.

---

### A13 — Discriminative Score  |  perfect **0**

$$\text{DS} = \left|\,\text{acc}_{\text{test}} - 0.5\,\right|$$

Post-hoc classifier (GRU or MLP) trained on 80% of $\{\text{real}\} \cup \{\text{fake}\}$,
evaluated on held-out 20%. **Score 0 = random guessing = indistinguishable (perfect).
Score 0.5 = perfect separation (bad).** BCE training loss saved per seed in
`seed_{i}_disc_gru_loss.csv` / `seed_{i}_disc_mlp_loss.csv`.

Paper (Yoon et al.) uses a 2-layer LSTM on Sines: **0.011 ± 0.008**.

---

### A14 — Predictive Score TSTR  |  perfect **0**

$$\text{PS} = \frac{1}{N \cdot T}\sum_{i,t}\left|\hat{X}_{i,t+1} - X_{i,t+1}\right|$$

Train-on-Synthetic, Test-on-Real MAE. Predictor trained on fake, evaluated on real.

Paper (Yoon et al.) uses a 2-layer LSTM on Sines: **0.093 ± 0.019**.

---

### A15 — Teacher-Sigma Correlation (↑)  |  perfect **1**

$$\rho = \text{Corr}\!\left(\hat{\sigma}_{\text{gen}},\, \sqrt{v_{\text{true}}}\right)$$

Pearson correlation between rolling realised vol from generated paths and true Heston latent vol.
**Higher is better (↑).** Heston-specific bonus metric (not in paper).

---

### A15 — Teacher-Sigma RMSE  |  perfect **0**

$$\text{RMSE} = \sqrt{\frac{1}{N \cdot T}\sum_{i,t}\!\left(\hat{\sigma}_{\text{gen},i,t} - \sqrt{v_{\text{true},i,t}}\right)^2}$$

---

## Files

| File | Description |
|------|-------------|
| `seed_{i}_metrics.json`          | Full per-seed metric dict |
| `metrics_summary.csv`            | Mean ± std across seeds |
| `seed_{i}_disc_gru_loss.csv`     | GRU discriminative classifier BCE loss per training step |
| `seed_{i}_disc_mlp_loss.csv`     | MLP discriminative classifier BCE loss per training step |
| `plots/seed_{i}_pca.png`         | PCA 2D projection (real vs fake) |
| `plots/seed_{i}_tsne.png`        | t-SNE 2D projection (real vs fake) |
