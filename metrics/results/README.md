# Metrics Results — TimeGAN on Heston (5 Seeds)

Dataset: 8192 Heston price paths, seq_len=128, canonical parameters
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

## Metric Definitions & Perfect Scores

| ID | Name | Perfect score | Explanation |
|----|------|:-------------:|-------------|
| A1  | Path MMD²                | **0** | Maximum Mean Discrepancy between full generated and real path distributions in a kernel RKHS. Measures whether the joint temporal distribution is matched across all time steps. |
| A2  | Terminal MMD²            | **0** | MMD applied only to the terminal (last-step) value distributions. Tests whether the model reproduces the correct final price distribution. |
| A3  | Increment MMD²           | **0** | MMD on first-order increments (returns dX = X[t+1]−X[t]). Measures whether the return distribution is correctly reproduced. |
| A4  | Volatility MMD           | **0** | MMD on rolling standard deviation of returns (realized volatility proxy). Tests whether stylised volatility clustering facts are captured. |
| A5  | Terminal SWD             | **0** | Sliced Wasserstein Distance on terminal values. More robust than MMD to heavy tails; measures transport cost between terminal distributions. |
| A6  | Path SWD                 | **0** | Sliced Wasserstein Distance on full paths. Captures global path geometry; sensitive to drift and curvature differences. |
| A7  | Covariance Error         | **0** | Frobenius norm of the difference between real and generated terminal covariance matrices. Tests cross-sectional correlation structure (for d>1; trivially non-zero for 1D data). |
| A8  | Mean RMSE                | **0** | RMSE between mean vectors of real and generated terminal distributions. Measures systematic bias in the generated final price level. |
| A9  | Return Std Error         | **0** | Absolute error on the standard deviation of returns. Measures whether the overall volatility level is correct. |
| A10 | Return Kurtosis Error    | **0** | Absolute error on excess kurtosis of returns. Measures whether fat tails and leptokurticity of financial returns are reproduced. |
| A11 | ACF Error (abs returns)  | **0** | Mean absolute ACF error on \|returns\| at lags 1,2,5,10. Tests volatility clustering (absolute returns have positive autocorrelation in real data). |
| A12 | ACF Error (sq returns)   | **0** | Mean absolute ACF error on returns² at lags 1,2,5,10. Also tests volatility clustering via squared returns (ARCH effect). |
| A13 | Discriminative Score     | **0** | \|accuracy − 0.5\| of a post-hoc classifier (GRU or MLP) trained on 80% of real+fake data and evaluated on held-out 20%. **Score 0 = classifier at chance = cannot distinguish real from fake (perfect generator). Score 0.5 = perfect separation (bad generator).** |
| A14 | Predictive Score (TSTR)  | **0** | Train-on-Synthetic, Test-on-Real MAE. A predictor (GRU or MLP) is trained to forecast the next step on synthetic data, then its MAE is measured on real data. Measures whether the synthetic temporal dynamics are useful for prediction. |
| A15 | Teacher-Sigma Corr (↑)   | **1** | Pearson correlation between the true Heston latent volatility √v and the realized volatility estimated from generated paths (rolling window std of returns). Heston-specific bonus metric; higher means the generator implicitly captures the correct stochastic vol process. |
| A15 | Teacher-Sigma RMSE       | **0** | RMSE between true latent √v and estimated realized volatility from generated paths. Companion to the correlation; measures absolute accuracy of the latent vol proxy. |

---

## Files

| File | Description |
|------|-------------|
| `seed_{i}_metrics.json`   | Full per-seed metric dict |
| `metrics_summary.csv`     | Mean ± std across seeds (machine-readable) |
| `plots/seed_{i}_pca.png`  | PCA 2D projection (real vs fake, 500-sample subset) |
| `plots/seed_{i}_tsne.png` | t-SNE 2D projection (real vs fake, 500-sample subset) |
