# Metrics Results — TimeGAN on Heston (5 Seeds)

Dataset: 8192 Heston price paths, seq_len=128, canonical parameters.  
Model: PyTorch TimeGAN, 20k training steps (5k embed + 5k supervised + 10k adversarial).  
All metrics: **lower is better**.

---

## Summary Table (mean ± std across 5 seeds)

| ID | Metric | Mean | Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----:|----:|-------:|-------:|-------:|-------:|-------:|
| A1  | Path MMD²                  | 0.7549 | ± 0.0075 | 0.7513 | 0.7687 | 0.7468 | 0.7521 | 0.7555 |
| A2  | Terminal MMD²              | 0.7412 | ± 0.0117 | 0.7420 | 0.7544 | 0.7245 | 0.7319 | 0.7532 |
| A3  | Increment MMD²             | 0.2329 | ± 0.0011 | 0.2328 | 0.2311 | 0.2339 | 0.2342 | 0.2326 |
| A4  | Volatility MMD             | 4.2441 | ± 0.1689 | 4.2447 | 4.1419 | 4.0930 | 4.1736 | 4.5673 |
| A5  | Terminal SWD               | 11.254 | ± 0.194  | 11.284 | 11.426 | 10.941 | 11.147 | 11.473 |
| A6  | Path SWD                   | 7.4373 | ± 0.1306 | 7.4264 | 7.6016 | 7.2561 | 7.3395 | 7.5627 |
| A7  | Covariance Error (Frob.)   | 199.60 | ± 0.00   | 199.60 | 199.60 | 199.60 | 199.60 | 199.60 |
| A8  | Mean RMSE (terminal)       | 0.4353 | ± 0.3231 | 0.6477 | 0.7213 | 0.7241 | 0.0700 | 0.0132 |
| A9  | Return Std Error           | 1.2239 | ± 0.0061 | 1.2274 | 1.2221 | 1.2136 | 1.2320 | 1.2245 |
| A10 | Return Kurtosis Error      | 71.44  | ± 31.88  | 56.37  | 103.66 | 108.98 | 22.73  | 65.45  |
| A11 | ACF Error (abs returns)    | 0.0967 | ± 0.0510 | 0.1171 | 0.0769 | 0.0167 | 0.1000 | 0.1728 |
| A12 | ACF Error (sq returns)     | 0.0682 | ± 0.0478 | 0.0599 | 0.0374 | 0.0223 | 0.0622 | 0.1592 |
| A13 | Discriminative Score (GRU) | 0.0782 | ± 0.0652 | 0.1741 | 0.0002 | 0.1048 | 0.0099 | 0.1021 |
| A13 | Discriminative Score (MLP) | 0.4682 | ± 0.0100 | 0.4823 | 0.4731 | 0.4716 | 0.4548 | 0.4591 |
| A14 | Predictive Score TSTR (GRU)| 0.0526 | ± 0.0046 | 0.0610 | 0.0521 | 0.0525 | 0.0472 | 0.0500 |
| A14 | Predictive Score TSTR (MLP)| 0.0169 | ± 0.0018 | 0.0159 | 0.0163 | 0.0175 | 0.0147 | 0.0201 |
| A15 | Teacher-Sigma Corr (↑)     | 0.0291 | ± 0.0085 | 0.0380 | 0.0289 | 0.0313 | 0.0133 | 0.0342 |
| A15 | Teacher-Sigma RMSE         | 0.1765 | ± 0.0023 | 0.1749 | 0.1767 | 0.1733 | 0.1801 | 0.1773 |

> A13 Discriminative Score = |accuracy − 0.5| (0 = indistinguishable from real).  
> A14 Predictive Score = MAE on real data after training on synthetic (TSTR protocol).  
> A15 Teacher-Sigma Corr: higher is better (↑); all others lower is better.

---

## Metric Definitions

| ID | Name | Description |
|----|------|-------------|
| A1  | Path MMD²                | Maximum Mean Discrepancy on full joint paths |
| A2  | Terminal MMD²            | MMD on terminal (last step) distribution |
| A3  | Increment MMD²           | MMD on first-order increments (returns) |
| A4  | Volatility MMD           | MMD on rolling volatility (stylised facts) |
| A5  | Terminal SWD             | Sliced Wasserstein Distance, terminal |
| A6  | Path SWD                 | Sliced Wasserstein Distance, full path |
| A7  | Covariance Error         | Frobenius norm of terminal covariance difference |
| A8  | Mean RMSE                | RMSE of terminal mean vector |
| A9  | Return Std Error         | Absolute error on return standard deviation |
| A10 | Return Kurtosis Error    | Absolute error on return excess kurtosis |
| A11 | ACF Error (abs returns)  | Mean ACF error on absolute returns (lags 1,2,5,10) |
| A12 | ACF Error (sq returns)   | Mean ACF error on squared returns (lags 1,2,5,10) |
| A13 | Discriminative Score     | Post-hoc RNN/MLP classifier accuracy gap from 0.5 |
| A14 | Predictive Score (TSTR)  | Train-on-Synthetic-Test-on-Real MAE |
| A15 | Teacher-Sigma            | Heston-specific: correlation & RMSE vs true latent vol |

---

## Files

| File | Description |
|------|-------------|
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `metrics_summary.csv`   | Mean ± std across seeds |
| `plots/seed_{i}_pca.png`  | PCA 2D projection (real vs fake) |
| `plots/seed_{i}_tsne.png` | t-SNE 2D projection (real vs fake) |
