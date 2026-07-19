# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case a perfect generator reaches with finite
samples, from a row-shuffled copy of the real data (see
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/)). Most floors are 0 because a
permutation preserves every column-wise marginal; the residual non-zero floors are pure finite-sample noise.

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | |
| A1 | Kurtosis Error | 2.955 ± 2.099 | 0.0148 | 5.360 | 3.768 | 0.9581 | 4.672 | 0 |
| A2 | \|r\| q95 Error | 0.0032 ± 0.0018 | 0.0042 | 0.0056 | 0.0016 | 5.06e-04 | 0.0040 | 0 |
| A3 | \|r\| q99 Error | 0.0043 ± 0.0028 | 0.0074 | 0.0069 | 0.0052 | 0.0017 | 4.49e-04 | 0 |
| A4 | Tail QQ Error | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0 |
| A5 | Hill Tail Index Error | 36.885 ± 17.053 | 40.695 | 18.783 | 51.751 | 15.495 | 57.699 | 0 |
| | **— Distribution —** | | | | | | | |
| A6 | Path MMD² | 0.0165 ± 0.0127 | 0.0103 | 0.0048 | 0.0274 | 0.0042 | 0.0355 | 0.0015 |
| A7 | Terminal MMD² | 0.0267 ± 0.0192 | 0.0168 | 0.0105 | 0.0537 | 0.0066 | 0.0455 | 0.0016 |
| A8 | Increment MMD² | 0.0077 ± 0.0041 | 0.0048 | 0.0072 | 0.0114 | 0.0022 | 0.0131 | 7.45e-04 |
| A9 | Volatility MMD | 0.3789 ± 0.2430 | 0.1645 | 0.3504 | 0.6260 | 0.0708 | 0.6828 | 0.0071 |
| A10 | Terminal SWD | 2.658 ± 0.8567 | 2.303 | 2.004 | 3.774 | 1.638 | 3.570 | 0.6873 |
| A11 | Path SWD | 1.417 ± 0.4914 | 1.406 | 0.9015 | 2.039 | 0.8440 | 1.893 | 0.4381 |
| A12 | RV Law Loss | 1.551 ± 0.3788 | 1.491 | 1.754 | 1.827 | 0.8373 | 1.847 | 0 |
| A13 | Mean Path RMSE | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0 |
| A14 | KS Log-returns | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0 |
| A15 | Skewness Error | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0 |
| A16 | QQ RMSE (300-pt) | 0.0025 ± 6.43e-04 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0 |
| A17 | Terminal Price KS | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0 |
| | **— Adversarial —** | | | | | | | |
| A18 GRU | Discriminative Score GRU | 0.0077 ± 0.0050 | 0.0081 | 7.63e-04 | 0.0038 | 0.0148 | 0.0111 | 0.0071 |
| A18 MLP | Discriminative Score MLP | 0.1031 ± 0.0395 | 0.1366 | 0.0346 | 0.0822 | 0.1326 | 0.1295 | 0.0033 |
| | **— Predictive —** | | | | | | | |
| A19 GRU | Predictive Score GRU | 0.0574 ± 0.0019 | 0.0553 | 0.0600 | 0.0592 | 0.0560 | 0.0564 | 0.0537 |
| A19 MLP | Predictive Score MLP | 0.0570 ± 0.0012 | 0.0553 | 0.0586 | 0.0571 | 0.0560 | 0.0578 | 0.0537 |
| | **— Temporal —** | | | | | | | |
| A20 | Covariance Error | 17.751 ± 6.707 | 8.830 | 18.765 | 14.807 | 29.373 | 16.980 | 0 |
| A21 | ACF \|r\| Error (lags) | 0.1252 ± 0.0674 | 0.0648 | 0.1046 | 0.2011 | 0.0477 | 0.2080 | 0 |
| A22 | ACF r² Error (lags) | 0.0839 ± 0.0348 | 0.0450 | 0.0793 | 0.1172 | 0.0479 | 0.1300 | 0 |
| A23 | ACF \|r\| Lag-1 Error | 0.2264 ± 0.1034 | 0.1537 | 0.2120 | 0.3669 | 0.0840 | 0.3155 | 0 |
| A24 | ACF r² Lag-1 Error | 0.1719 ± 0.0626 | 0.1177 | 0.2000 | 0.2634 | 0.0874 | 0.1908 | 0 |
| | **— Vol —** | | | | | | | |
| A25 | Mean RMSE | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.056 | 1.341 | 0.0743 | 0 |
| A26 | Return Std Error | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 | 0 |
| A27 | Log-Return Std Error | 0.0017 ± 7.78e-04 | 0.0020 | 0.0023 | 5.63e-04 | 9.72e-04 | 0.0025 | 0 |
| A28 | Kurtosis Ratio | -1.095 ± 3.525 | 1.979 | 0.1360 | 0.2451 | -8.005 | 0.1718 | 1.000 |
| A29 | Sigma Mean Error | 0.0307 ± 0.0089 | 0.0301 | 0.0373 | 0.0273 | 0.0164 | 0.0422 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0 |
| A31 | Rolling Vol KS (w=5) | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0 |
| A32 | Vol-of-Vol Error | 8.97e-04 ± 8.69e-04 | 3.54e-04 | 2.75e-04 | 0.0025 | 2.63e-04 | 0.0011 | 0 |
| | **— Heston Spec —** | | | | | | | |
| A33 | Teacher-Sigma Corr ↑ | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | -0.0082 | -0.0057 | 0.0166 | 0.6143 |
| A34 | Teacher-Sigma RMSE | 0.1183 ± 0.0184 | 0.1016 | 0.1108 | 0.1479 | 0.1004 | 0.1308 | 0.0654 |

---

## Stylised Facts Diagnostic (Heston vs TimeGAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means;
  combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one division. the **function-level MAPE on the curve L itself** —
  the derivative / 2nd-derivative MAPE is **excluded** (near-zero true diffs make it explode). Combined mean/std = mean and sample std across the 5 seeds.

↓ lower is better for both rows. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals).
TimeGAN's large MSE std on the log-return histogram (±120.61) is a genuine seed-2 collapse
(504.48 vs 11–170 for other seeds).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 144.21 ± 120.61 | 11.054 | 20.679 | 504.48 | 14.578 | 170.28 | 0 |
| | % err | 33.42% ± 6.512% | 26.16% | 36.78% | 32.86% | 27.35% | 43.95% | 0 |
| **QQ plot** | MSE | 7.09e-06 ± 3.34e-06 | 4.16e-06 | 7.22e-06 | 8.07e-06 | 3.19e-06 | 1.28e-05 | 0 |
| | % err | 34.29% ± 11.19% | 20.18% | 23.68% | 47.70% | 33.99% | 45.88% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.0105 ± 0.0085 | 0.0029 | 0.0053 | 0.0182 | 0.0010 | 0.0250 | 0 |
| | % err | 164% ± 101% | 87.82% | 111% | 237% | 59.02% | 324% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0058 ± 0.0033 | 0.0020 | 0.0043 | 0.0072 | 0.0010 | 0.0143 | 0 |
| | % err | 110% ± 60.72% | 72.18% | 70.40% | 105% | 72.05% | 228% | 0 |
| **Rolling vol histogram** | MSE | 439.33 ± 216.74 | 280.40 | 633.95 | 604.18 | 82.528 | 595.57 | 0 |
| | % err | 56.06% ± 20.98% | 52.54% | 71.71% | 84.04% | 22.66% | 49.36% | 0 |
| **Tail survival** | MSE | 0.0117 ± 0.0092 | 0.0027 | 0.0061 | 0.0219 | 0.0041 | 0.0238 | 0 |
| | % err | 23.60% ± 6.040% | 18.29% | 24.46% | 26.00% | 16.10% | 33.16% | 0 |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

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

## Paper reproduction on Stocks (our port vs Yoon et al. Table 1)

Before running TimeGAN on Heston we reproduced the **original TimeGAN paper result on the
Stocks dataset** with the same PyTorch port, using the paper's own hyperparameters
(seq_len 24, hidden 24, 3 layers, 50 000 iterations/phase, batch 128). This validates the
generator port independently of Heston. Full write-up:
[`../../../methods/TimeGAN/paper_reimplementation/`](../../../methods/TimeGAN/paper_reimplementation/).

| Dataset | Metric | Ours — 2-layer judge, 1 seed | Ours — 1-layer judge, 5 seeds | Paper (Table 1) | Verdict |
|---------|--------|:----------------------------:|:-----------------------------:|:---------------:|---------|
| Stocks | Discriminative ↓ | 0.219 ± 0.066 | **0.119 ± 0.036** | 0.102 ± 0.031 | **matches** ✓ (within 0.5σ) |
| Stocks | Predictive ↓ | 0.039 ± 0.000 | **0.042 ± 0.002** | 0.038 ± 0.001 | **matches** ✓ |

Once the discriminative judge is matched to the paper's own depth (1-layer GRU, hidden = ⌊d/2⌋)
across 5 training seeds, the score drops from 0.219 → **0.119 ± 0.036**, overlapping the paper's
**0.102 ± 0.031** (gap < 0.5σ). The predictive score matched all along (0.042 vs 0.038). The
original 2× discrepancy was an artefact of the scoring judge's depth, not the generator.

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B two-subline aggregates (MSE + % err) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step (every 100 steps) |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step (every 100 steps) |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts |

→ Cross-method comparison with SBTS: [`results/README.md`](../../README.md)
