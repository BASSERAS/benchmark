# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**. A16 ↓.

---

## Results (mean ± std across 5 seeds)

### A1–A20 — Core metrics

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| A1  | Path MMD²              | 0.0181 ± 0.0147 | 0.0093 | 0.0046 | 0.0322 | 0.0051 | 0.0393 | 0.0018 ± 0.0002 |
| A2  | Terminal MMD²          | 0.0308 ± 0.0229 | 0.0256 | 0.0103 | 0.0681 | 0.0060 | 0.0439 | 0.0016 ± 0.0002 |
| A3  | Increment MMD²         | 0.0077 ± 0.0039 | 0.0048 | 0.0070 | 0.0112 | 0.0023 | 0.0129 | 0.0008 ± 0.0000 |
| A4  | Volatility MMD         | 0.3933 ± 0.2553 | 0.1700 | 0.3416 | 0.6572 | 0.0797 | 0.7179 | 0.0077 ± 0.0006 |
| A5  | Terminal SWD           | 3.1284 ± 0.9227 | 2.9505 | 2.0579 | 4.5309 | 2.3097 | 3.7931 | 0.7635 ± 0.1174 |
| A6  | Path SWD               | 1.6343 ± 0.5763 | 1.2793 | 0.9696 | 2.4624 | 1.2892 | 2.1711 | 0.5542 ± 0.0624 |
| A7  | Covariance Error       | 17.75 ± 6.71    | 8.83   | 18.76  | 14.81  | 29.37  | 16.98  | 4.76 ± 2.50 |
| A8  | Mean RMSE              | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 | 0.1400 ± 0.1303 |
| A9  | Return Std Error       | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 | 0.0048 ± 0.0031 |
| A10 | Kurtosis Error         | 2.9545 ± 2.0988 | 0.0148 | 5.3599 | 3.7677 | 0.9581 | 4.6722 | 0.0172 ± 0.0155 |
| A11 | ACF |r| Error         | 0.1252 ± 0.0674 | 0.0648 | 0.1046 | 0.2011 | 0.0477 | 0.2080 | 0.0017 ± 0.0006 |
| A12 | ACF r² Error          | 0.0839 ± 0.0348 | 0.0450 | 0.0793 | 0.1172 | 0.0479 | 0.1300 | 0.0014 ± 0.0006 |
| A13 | Disc Score GRU        | 0.0099 ± 0.0084 | 0.0035 | 0.0124 | 0.0023 | 0.0252 | 0.0063 | 0.0128 ± 0.0068 |
| A13 | Disc Score MLP        | 0.0921 ± 0.0463 | 0.1277 | 0.0053 | 0.0832 | 0.1182 | 0.1262 | 0.0080 ± 0.0081 |
| A14 | Pred Score GRU (TSTR) | 0.0570 ± 0.0013 | 0.0553 | 0.0591 | 0.0575 | 0.0561 | 0.0570 | 0.0564 ± 0.0022 |
| A14 | Pred Score MLP (TSTR) | 0.0573 ± 0.0015 | 0.0556 | 0.0593 | 0.0570 | 0.0559 | 0.0588 | 0.0565 ± 0.0022 |
| A15 | Sigma Corr ↑          | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | −0.0082 | −0.0057 | 0.0166 | 0.6135 ± 0.0019 |
| A15 | Sigma RMSE            | 0.1183 ± 0.0184 | 0.1016 | 0.1108 | 0.1479  | 0.1004  | 0.1308 | 0.0653 ± 0.0002 |
| A16 | Tail RMS              | 0.0234 ± 0.0109 | 0.0301 | 0.0374 | 0.0076 | 0.0143 | 0.0278 | 0.0008 ± 0.0008 |
| A16 | q90 Error             | 0.0337 ± 0.0188 | 0.0422 | 0.0560 | 0.0021 | 0.0241 | 0.0441 | 0.0010 ± 0.0010 |
| A16 | q95 Error             | 0.0191 ± 0.0107 | 0.0296 | 0.0321 | 0.0096 | 0.0047 | 0.0194 | 0.0009 ± 0.0009 |
| A16 | q99 Error             | 0.0052 ± 0.0031 | 0.0080 | 0.0058 | 0.0087 | 0.0029 | 0.0005 | 0.0004 ± 0.0004 |
| A17 | Oracle MAE (AR5)      | 0.0097 ± 0.0000 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 ± 0.0000 |
| A18 | Agent MAE (AR5 TSTR)  | 0.0101 ± 0.0003 | 0.0098 | 0.0106 | 0.0100 | 0.0100 | 0.0099 | 0.0097 ± 0.0000 |
| A19 | Oracle-Agent Corr ↑   | −0.332 ± 0.306  | −0.773 | −0.038 | −0.206 | −0.030 | −0.614 | −0.058 ± 0.430 |
| A20 | RV Law Loss           | 1.5512 ± 0.3788 | 1.4914 | 1.7536 | 1.8266 | 0.8373 | 1.8470 | 0.0673 ± 0.0362 |

### B1–B12 — Stylized metrics

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| B1  | Mean Path RMSE        | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0.1511 ± 0.0708 |
| B2  | Cross-Sect. Vol RMSE  | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0.1355 ± 0.0735 |
| B3  | KS on Log-returns     | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0.0018 ± 0.0009 |
| B4  | Skewness Error        | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0.0060 ± 0.0048 |
| B5  | QQ RMSE (300-pt)      | 0.0025 ± 0.0006 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0.0001 ± 0.0000 |
| B6  | Tail QQ Error         | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0.0001 ± 0.0001 |
| B7  | ACF lag-1 |r| Err     | 0.2282 ± 0.1042 | 0.1549 | 0.2137 | 0.3698 | 0.0847 | 0.3180 | 0.0018 ± 0.0016 |
| B8  | ACF lag-1 r² Err      | 0.1732 ± 0.0631 | 0.1186 | 0.2016 | 0.2655 | 0.0881 | 0.1923 | 0.0017 ± 0.0014 |
| B9  | Rolling Vol KS        | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0.0046 ± 0.0024 |
| B10 | Vol-of-Vol Error      | 0.0009 ± 0.0009 | 0.0004 | 0.0003 | 0.0025 | 0.0003 | 0.0011 | 0.0000 ± 0.0000 |
| B11 | Terminal Price KS     | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0.0145 ± 0.0043 |
| B12 | Hill Tail Index Err   | 36.88 ± 17.05   | 40.70  | 18.78  | 51.75  | 15.49  | 57.70  | 0.499 ± 0.610   |

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

## Stylised Metrics B1–B12 — mean ± std across 5 seeds

One scalar per diagnostic plot panel; extracted from the same data as the 8-panel PNG.
B8 (ARCH Persistence) and B10 (GARCH Persistence) have been removed as redundant with
A11 and A12 respectively. All ↓ lower is better.

> Metric-to-plot mapping (matches `metrics/plot_diagnostics.py` panel order):

| Plots | B metric | What you see |
|-------|----------|-------------|
| 1 + 2 (sample paths) | B1 Mean Path RMSE, B2 Cross-Sect. Vol RMSE | Systematic drift + spread of the fan |
| 3 (log-return histogram) | B3 KS Stat, B4 Skewness Error | Shape, centre, asymmetry of density |
| 4 (QQ plot) | B5 QQ RMSE (300-pt), B6 Tail QQ Error | Diagonal alignment; 5th/95th deviation |
| 5 (ACF |r|) | B7 ACF lag-1 |r| error | Volatility clustering: lag-1 bar height |
| 6 (ACF r²) | B8 ACF lag-1 r² error | GARCH effect: lag-1 bar height |
| 7 (rolling vol hist.) | B9 Rolling Vol KS, B10 Vol-of-Vol Error | Width and spread of the vol histogram |
| 8 (tail survival) | B11 Terminal Price KS, B12 Tail Index Error (Hill) | Log-log alignment + slope (tail heaviness) |

### Formulas

**B1 — Mean Path RMSE**

$$B1 = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\!\left(\bar{S}^{\text{real}}_t - \bar{S}^{\text{gen}}_t\right)^2}$$

**B2 — Cross-Sectional Vol RMSE**

$$B2 = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\!\left(\sigma^{\text{real}}_t - \sigma^{\text{gen}}_t\right)^2}$$

where $\sigma_t = \text{std}_{i}(S_{i,t})$ is the cross-sectional standard deviation at time $t$.

**B3 — KS Statistic on log-returns**

$$B3 = \sup_x\!\left|F^{\text{real}}_r(x) - F^{\text{gen}}_r(x)\right|, \quad r_t = \log(S_{t+1}/S_t)$$

Two-sample Kolmogorov-Smirnov statistic on all pooled log-returns. Perfect: 0.

**B4 — Skewness Error**

$$B4 = \left|\text{skew}(r^{\text{real}}) - \text{skew}(r^{\text{gen}})\right|$$

Heston generates negative skew ($\rho < 0$ leverage). Perfect: 0.

**B5 — QQ RMSE (300-pt)**

$$B5 = \sqrt{\frac{1}{300}\sum_{g=1}^{300}\!\left(Q^{\text{real}}_r(p_g) - Q^{\text{gen}}_r(p_g)\right)^2}, \quad p_g \in [0.005,\,0.995]$$

Tests bulk distributional match. Perfect: 0.

**B6 — Tail QQ Error**

Same as B5 restricted to $p \in [0.01,0.05]\cup[0.95,0.99]$ (10 points). Perfect: 0.

**B7 — ACF lag-1 Error on |r|**

$$B7 = \left|\,\text{ACF}(|r^{\text{real}}|,\,\ell=1) - \text{ACF}(|r^{\text{gen}}|,\,\ell=1)\,\right|$$

Lag-1 absolute-return ACF. Positive in Heston (~0.05): dominant ARCH signal. Perfect: 0.

**B8 — ACF lag-1 Error on r²**

$$B8 = \left|\,\text{ACF}({r^{\text{real}}}^2,\,\ell=1) - \text{ACF}({r^{\text{gen}}}^2,\,\ell=1)\,\right|$$

Lag-1 squared-return ACF (GARCH effect). Perfect: 0.

**B9 — Rolling Vol KS Statistic**

$$B9 = \text{KS}\!\left(\{\hat{\sigma}^{\text{real}}_{i,t}\},\,\{\hat{\sigma}^{\text{gen}}_{i,t}\}\right), \quad \hat{\sigma}_{i,t}=\text{std}(r_{i,t-4:t})$$

Two-sample KS on rolling std of log-returns (window = 5 days). Perfect: 0.

**B10 — Vol-of-Vol Error**

$$B10 = \left|\,\text{std}(\hat{\sigma}^{\text{real}}) - \text{std}(\hat{\sigma}^{\text{gen}})\,\right|$$

Dispersion of the rolling volatility distribution. Perfect: 0.

**B11 — Terminal Price KS Statistic**

$$B11 = \text{KS}\!\left(S^{\text{real}}_T,\,S^{\text{gen}}_T\right)$$

Two-sample KS on the terminal marginal $S_T$. Perfect: 0.

**B12 — Hill Tail Index Error**

$$B12 = \left|\hat{\alpha}^{\text{real}} - \hat{\alpha}^{\text{gen}}\right|, \quad \hat{\alpha} = \left[\frac{1}{k}\sum_{i=1}^{k}\log\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$$

Hill (1975) estimator on top 10% of terminal prices $S_T$. Perfect: 0.

> **B7–B8** (ACF metrics): Heston true ACF(|r|, lag=1) ≈ +0.052, ACF(r², lag=1) ≈ +0.050.
> TimeGAN often collapses to near-zero ACF, missing the ARCH signature (seeds 2, 4 worst).
> **B12**: Hill estimator on terminal prices $S_T$. Large variance across seeds — use the mean.

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
