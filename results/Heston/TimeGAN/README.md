# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A15 Corr ↑ and A21 Corr ↑**. A19 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----------|--------|--------|--------|--------|--------|
| | **— Fat Tail —** | | | | | | |
| A10 | Kurtosis Error         | 2.9545 ± 2.0988 | 0.0148 | 5.3599 | 3.7677 | 0.9581 | 4.6722 |
| A17 | \|r\| q95 Error        | 0.0032 ± 0.0018 | 0.0042 | 0.0056 | 0.0016 | 0.0005 | 0.0040 |
| A18 | \|r\| q99 Error        | 0.0043 ± 0.0028 | 0.0074 | 0.0069 | 0.0052 | 0.0017 | 0.0004 |
| A30 | Tail QQ Error          | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 |
| A34 | Hill Tail Index Error  | 36.88 ± 17.05   | 40.70  | 18.78  | 51.75  | 15.49  | 57.70  |
| | **— Distribution —** | | | | | | |
| A1  | Path MMD²              | 0.0181 ± 0.0147 | 0.0093 | 0.0046 | 0.0322 | 0.0051 | 0.0393 |
| A2  | Terminal MMD²          | 0.0308 ± 0.0229 | 0.0256 | 0.0103 | 0.0681 | 0.0060 | 0.0439 |
| A3  | Increment MMD²         | 0.0077 ± 0.0039 | 0.0048 | 0.0070 | 0.0112 | 0.0023 | 0.0129 |
| A4  | Volatility MMD         | 0.3933 ± 0.2553 | 0.1700 | 0.3416 | 0.6572 | 0.0797 | 0.7179 |
| A5  | Terminal SWD           | 3.1284 ± 0.9227 | 2.9505 | 2.0579 | 4.5309 | 2.3097 | 3.7931 |
| A6  | Path SWD               | 1.6343 ± 0.5763 | 1.2793 | 0.9696 | 2.4624 | 1.2892 | 2.1711 |
| A24 | RV Law Loss (W₁ on ann. RV) | 1.5512 ± 0.3788 | 1.4914 | 1.7536 | 1.8266 | 0.8373 | 1.8470 |
| A25 | Mean Path RMSE         | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 |
| A27 | KS on Log-returns      | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 |
| A28 | Skewness Error         | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 |
| A29 | QQ RMSE (300-pt)       | 0.0025 ± 0.0006 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 |
| A33 | Terminal Price KS      | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 |
| | **— Adversarial —** | | | | | | |
| A13 | Disc Score GRU        | 0.0099 ± 0.0084 | 0.0035 | 0.0124 | 0.0023 | 0.0252 | 0.0063 |
| A13 | Disc Score MLP        | 0.0921 ± 0.0463 | 0.1277 | 0.0053 | 0.0832 | 0.1182 | 0.1262 |
| | **— Predictive —** | | | | | | |
| A14 | Pred Score GRU (TSTR) | 0.0570 ± 0.0013 | 0.0553 | 0.0591 | 0.0575 | 0.0561 | 0.0570 |
| A14 | Pred Score MLP (TSTR) | 0.0573 ± 0.0015 | 0.0556 | 0.0593 | 0.0570 | 0.0559 | 0.0588 |
| | **— Temporal —** | | | | | | |
| A7  | Covariance Error       | 17.75 ± 6.71    | 8.83   | 18.76  | 14.81  | 29.37  | 16.98  |
| A11 | ACF \|r\| Error        | 0.1252 ± 0.0674 | 0.0648 | 0.1046 | 0.2011 | 0.0477 | 0.2080 |
| A12 | ACF r² Error           | 0.0839 ± 0.0348 | 0.0450 | 0.0793 | 0.1172 | 0.0479 | 0.1300 |
| A22 | ACF \|r\| Lag-1 Error  | 0.2264 ± 0.1034 | 0.1537 | 0.2120 | 0.3669 | 0.0840 | 0.3155 |
| A23 | ACF r² Lag-1 Error     | 0.1719 ± 0.0626 | 0.1177 | 0.2000 | 0.2634 | 0.0874 | 0.1908 |
| | **— Vol —** | | | | | | |
| A8  | Mean RMSE              | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 |
| A9  | Return Std Error       | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 |
| A16 | Log-Return Std Error   | 0.0017 ± 0.0008 | 0.0020 | 0.0023 | 0.0006 | 0.0010 | 0.0025 |
| A19 | Kurtosis Ratio (→ 1)   | −1.0947 ± 3.5247 | 1.9788 | 0.1360 | 0.2451 | −8.0053 | 0.1718 |
| A20 | Sigma Mean Error       | 0.0307 ± 0.0089 | 0.0301 | 0.0373 | 0.0273 | 0.0164 | 0.0422 |
| A26 | Cross-Sect. Vol Path RMSE | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 |
| A31 | Rolling Vol KS (window=5) | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 |
| A32 | Vol-of-Vol Error       | 0.0009 ± 0.0009 | 0.0004 | 0.0003 | 0.0025 | 0.0003 | 0.0011 |
| | **— Heston Spec —** | | | | | | |
| A15 | Sigma Corr ↑          | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | −0.0082 | −0.0057 | 0.0166 |
| A15 | Sigma RMSE            | 0.1183 ± 0.0184 | 0.1016 | 0.1108 | 0.1479  | 0.1004  | 0.1308 |
| A21 | Oracle Sigma Corr ↑   | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | −0.0082 | −0.0057 | 0.0166 |

### B — Curve-shape metrics (6 plots × 3 sub-metrics)

Each cell = MSE between the real and generated **curve** (not a scalar).
- **funct**: MSE(L\_real, L\_gen) between curve values
- **der**: MSE of first finite difference — L\_der\[k\] = L\[k+1\] − L\[k\]
- **sec\_der**: MSE of second finite difference — L\_sec\[k\] = L\_der\[k+1\] − L\_der\[k\]

All values: mean ± std across 5 seeds. Perfect floor = 0 for all (row-shuffled real data has identical distribution curves). Full formulas: [`metrics/README.md`](../../../metrics/README.md).

| Plot | funct | der | sec\_der |
|------|-------|-----|----------|
| Log-return histogram | 89.43 ± 101.01 | 24.19 ± 38.76 | 30.59 ± 53.31 |
| QQ plot              | 6.90e-6 ± 3.3e-6 | 1.60e-7 ± 5.9e-8 | 2.67e-8 ± 1.1e-8 |
| ACF \|r\|            | 9.13e-3 ± 8.5e-3 | 7.10e-4 ± 4.6e-4 | 6.32e-4 ± 7.1e-4 |
| ACF r²               | 3.76e-3 ± 2.9e-3 | 8.42e-4 ± 5.98e-4 | 1.17e-3 ± 1.5e-3 |
| Rolling vol hist.    | 430.4 ± 216.7 | 5.756 ± 4.211 | 3.147 ± 1.498 |
| Tail survival        | 1.169e-2 ± 9.2e-3 | 1.857e-5 ± 1.7e-5 | 3.34e-7 ± 4.4e-7 |

---

## Comparison with Yoon et al. NeurIPS 2019 (Table 2)

> ⚠️ **Not a direct comparison.** The paper evaluates on Sines (d=5, T=24) and Stocks (d=6)
> with a **2-layer LSTM** classifier. We evaluate on Heston (d=1, T=128) with GRU and MLP.

| Metric | Paper — Sines | Paper — Stocks | Ours — Heston GRU | Ours — Heston MLP |
|--------|:------------:|:-------------:|:-----------------:|:-----------------:|
| Disc Score ↓  | 0.011 ± 0.008 | 0.102 ± 0.021 | 0.050 ± 0.034 | 0.151 ± 0.142 |
| Pred Score ↓  | 0.093 ± 0.019 | 0.038 ± 0.001 | 0.009 ± 0.000 | 0.009 ± 0.001 |

Our GRU discriminative score (0.050) sits between the paper's Sines (0.011) and Stocks (0.102),
consistent with Heston being a moderately challenging 1-D financial process.
Our predictive score is lower than the paper's Sines result because Heston is 1-D and
next-step prediction is inherently simpler than 5-D Sines.

---

## B — Curve-shape metrics: plot mapping

Each B metric measures the shape of a diagnostic curve, not a single scalar statistic.
For each of the 6 plots, three sub-metrics are computed:

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Formulas: [`metrics/README.md`](../../../metrics/README.md)

---

## Stylised Facts Diagnostic (Heston vs TimeGAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step (every 100 steps) |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step (every 100 steps) |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
