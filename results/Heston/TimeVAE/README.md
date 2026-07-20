# Metrics — TimeVAE on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** TimeVAE-Base (Desai, Freeman, Beaver, Wang, arXiv:2111.08095v3), a
convolutional variational auto-encoder. Encoder = 3× strided Conv1d
(50/100/200 filters) → dense z_mean/z_log_var (latent 8); decoder = level model +
residual dense→reshape→3× transposed-conv. Loss = 3.0·reconstruction (SSE +
feature-mean SSE) + KL. PyTorch port, Adam lr 1e-3, batch 16, EarlyStopping
(all 5 seeds stop 230–340 epochs, 247 340 params). See
[`../../../methods/TimeVAE/code/README.md`](../../../methods/TimeVAE/code/README.md).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case a perfect generator reaches with finite
samples, from a row-shuffled copy of the real data (see
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/)). Most floors are 0 because a
permutation preserves every column-wise marginal; the residual non-zero floors are pure finite-sample
noise, and are **identical across methods** (same real data, same permutation).

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | |
| A1 | Kurtosis Error | 2.258 ± 0.5719 | 1.368 | 2.662 | 1.791 | 2.668 | 2.799 | 0 |
| A2 | \|r\| q95 Error | 0.0222 ± 1.22e-04 | 0.0220 | 0.0223 | 0.0222 | 0.0223 | 0.0223 | 0 |
| A3 | \|r\| q99 Error | 0.0308 ± 1.05e-04 | 0.0306 | 0.0309 | 0.0308 | 0.0309 | 0.0308 | 0 |
| A4 | Tail QQ Error | 0.0219 ± 1.17e-04 | 0.0216 | 0.0219 | 0.0218 | 0.0220 | 0.0219 | 0 |
| A5 | Hill Tail Index Error | 2.396 ± 0.6794 | 2.056 | 1.263 | 2.541 | 3.110 | 3.011 | 0 |
| | **— Distribution —** | | | | | | | |
| A6 | Path MMD² | 0.0184 ± 9.55e-04 | 0.0180 | 0.0190 | 0.0178 | 0.0199 | 0.0172 | 0.0015 |
| A7 | Terminal MMD² | 0.0042 ± 0.0011 | 0.0032 | 0.0060 | 0.0041 | 0.0046 | 0.0029 | 0.0016 |
| A8 | Increment MMD² | 0.2134 ± 0.0012 | 0.2131 | 0.2118 | 0.2130 | 0.2155 | 0.2136 | 7.45e-04 |
| A9 | Volatility MMD | 3.591 ± 0.4563 | 2.941 | 3.812 | 3.153 | 3.999 | 4.049 | 0.0071 |
| A10 | Terminal SWD | 1.798 ± 0.2603 | 1.523 | 2.057 | 1.819 | 2.108 | 1.483 | 0.6873 |
| A11 | Path SWD | 0.9882 ± 0.2052 | 0.8341 | 1.202 | 0.8952 | 1.262 | 0.7480 | 0.4381 |
| A12 | RV Law Loss | 4.986 ± 0.0084 | 4.971 | 4.991 | 4.982 | 4.994 | 4.992 | 0 |
| A13 | Mean Path RMSE | 0.2981 ± 0.2172 | 0.4999 | 0.1395 | 0.6196 | 0.1259 | 0.1058 | 0 |
| A14 | KS Log-returns | 0.3673 ± 0.0046 | 0.3612 | 0.3709 | 0.3622 | 0.3720 | 0.3701 | 0 |
| A15 | Skewness Error | 0.5568 ± 0.0984 | 0.4720 | 0.6895 | 0.4188 | 0.6172 | 0.5867 | 0 |
| A16 | QQ RMSE (300-pt) | 0.0105 ± 8.40e-05 | 0.0104 | 0.0106 | 0.0105 | 0.0106 | 0.0106 | 0 |
| A17 | Terminal Price KS | 0.0478 ± 0.0099 | 0.0664 | 0.0468 | 0.0449 | 0.0439 | 0.0370 | 0 |
| | **— Adversarial —** | | | | | | | |
| A18 GRU | Discriminative Score GRU | 0.2035 ± 0.1934 | 0.0072 | 0.1414 | 0.4576 | 0.0056 | 0.4057 | 0.0042 |
| A18 MLP | Discriminative Score MLP | 0.1756 ± 0.1354 | 0.3392 | 0.1216 | 0.3300 | 0.0841 | 0.0032 | 0.0067 |
| | **— Predictive —** | | | | | | | |
| A19 GRU | Predictive Score GRU | 0.0577 ± 8.53e-04 | 0.0566 | 0.0576 | 0.0571 | 0.0584 | 0.0589 | 0.0537 |
| A19 MLP | Predictive Score MLP | 0.0564 ± 2.75e-04 | 0.0568 | 0.0561 | 0.0561 | 0.0566 | 0.0565 | 0.0539 |
| | **— Temporal —** | | | | | | | |
| A20 | Covariance Error | 51.272 ± 1.758 | 52.105 | 51.964 | 53.760 | 49.103 | 49.430 | 0 |
| A21 | ACF \|r\| Error (lags) | 0.3865 ± 0.1057 | 0.2211 | 0.4488 | 0.3024 | 0.4799 | 0.4803 | 0 |
| A22 | ACF r² Error (lags) | 0.3580 ± 0.0885 | 0.2253 | 0.4130 | 0.2788 | 0.4382 | 0.4344 | 0 |
| A23 | ACF \|r\| Lag-1 Error | 0.4637 ± 0.1346 | 0.2784 | 0.5216 | 0.3283 | 0.5937 | 0.5965 | 0 |
| A24 | ACF r² Lag-1 Error | 0.4589 ± 0.1189 | 0.3016 | 0.5167 | 0.3301 | 0.5764 | 0.5697 | 0 |
| | **— Vol —** | | | | | | | |
| A25 | Mean RMSE | 0.3396 ± 0.2710 | 0.8152 | 0.0701 | 0.4608 | 0.1752 | 0.1766 | 0 |
| A26 | Return Std Error | 1.073 ± 0.0078 | 1.060 | 1.078 | 1.068 | 1.080 | 1.078 | 0 |
| A27 | Log-Return Std Error | 0.0109 ± 7.80e-05 | 0.0108 | 0.0110 | 0.0109 | 0.0110 | 0.0110 | 0 |
| A28 | Kurtosis Ratio | 0.2780 ± 0.0467 | 0.3487 | 0.2346 | 0.3189 | 0.2485 | 0.2393 | 1.000 |
| A29 | Sigma Mean Error | 0.1741 ± 0.0018 | 0.1712 | 0.1752 | 0.1729 | 0.1757 | 0.1754 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | 1.122 ± 0.0447 | 1.128 | 1.146 | 1.188 | 1.090 | 1.059 | 0 |
| A31 | Rolling Vol KS (w=5) | 0.9871 ± 0.0045 | 0.9785 | 0.9872 | 0.9885 | 0.9913 | 0.9900 | 0 |
| A32 | Vol-of-Vol Error | 0.0046 ± 5.60e-05 | 0.0045 | 0.0046 | 0.0046 | 0.0046 | 0.0046 | 0 |
| | **— Heston Spec —** | | | | | | | |
| A33 | Teacher-Sigma Corr ↑ | 0.0273 ± 0.0050 | 0.0241 | 0.0264 | 0.0261 | 0.0228 | 0.0370 | 0.6143 |
| A34 | Teacher-Sigma RMSE | 0.1793 ± 0.0016 | 0.1766 | 0.1803 | 0.1782 | 0.1808 | 0.1806 | 0.0654 |

**Reading the table.** TimeVAE reproduces the **marginal centre** of the Heston log-return
distribution but fails its **tails, volatility structure and autocorrelation**. Two numbers dominate.
(1) **A28 kurtosis ratio 0.278** — TimeVAE's returns are *under-dispersed* (0.28× the true excess
kurtosis; the VAE's Gaussian-decoder output is too thin-tailed), the opposite failure mode to
Diffusion-TS (1.87, closest to the ideal 1.0). This propagates to A1 kurtosis error 2.26, A14 KS on
log-returns 0.367 (high), and A9 volatility MMD **3.59** — an order of magnitude worse than
Diffusion-TS (0.38), the single clearest sign TimeVAE does not capture Heston's stochastic-vol
signature. (2) **A31 rolling-vol KS 0.987** — the rolling-5 volatility distribution is almost entirely
disjoint from real, and A26 return-std error 1.07 confirms the generated paths are the wrong scale in
volatility. The autocorrelation (ARCH) structure is only weakly present: A21 ACF-|r| error 0.39, A23
lag-1 0.46 — much larger than Diffusion-TS (~0.02–0.004). A18 GRU discriminative is high-variance
(0.20 ± 0.19: seeds 2/4 fully separable, seeds 0/3 near-fooled). As with every method, A33
teacher-sigma correlation ≈ 0.027 — no generator recovers the latent Heston variance process from
prices alone (perfect floor 0.614, unreachable without the hidden state). This is an **honest weak
result**: TimeVAE is a strong general-purpose TS-VAE but a poor fit for heavy-tailed stochastic-vol
price paths.

---

## Stylised Facts Diagnostic (Heston vs TimeVAE, seed 0)

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
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one
  division: the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is
  **excluded** (near-zero true diffs make it explode). Combined mean/std = mean and sample std across the 5 seeds.

↓ lower is better for both rows. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals).
The very large ACF %err (~890–900%) and rolling-vol-histogram MSE (47 159) quantify the same failure the A
table shows: TimeVAE misses Heston's autocorrelation and volatility shape by a wide margin.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 2887.3 ± 311.0 | 2082.3 | 3156.6 | 2445.7 | 3386.1 | 3365.6 | 0 |
| | % err | 114.83% ± 0.588% | 113.75% | 115.20% | 114.64% | 115.33% | 115.21% | 0 |
| **QQ plot** | MSE | 1.191e-04 ± 1.80e-06 | 1.160e-04 | 1.203e-04 | 1.183e-04 | 1.207e-04 | 1.204e-04 | 0 |
| | % err | 90.291% ± 1.536% | 87.504% | 91.291% | 89.758% | 91.291% | 91.610% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.1005 ± 0.0453 | 0.0399 | 0.1207 | 0.0600 | 0.1389 | 0.1428 | 0 |
| | % err | 891.10% ± 249.26% | 519.20% | 1042.70% | 666.70% | 1105.25% | 1121.65% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0798 ± 0.0330 | 0.0360 | 0.0942 | 0.0502 | 0.1082 | 0.1102 | 0 |
| | % err | 903.15% ± 234.33% | 558.59% | 1048.82% | 685.03% | 1105.74% | 1117.59% | 0 |
| **Rolling vol histogram** | MSE | 47159 ± 4926.1 | 56934 | 35526 | 50667 | 48666 | 44004 | 0 |
| | % err | 334.81% ± 11.759% | 347.74% | 312.84% | 340.85% | 335.87% | 336.75% | 0 |
| **Tail survival** | MSE | 0.2168 ± 0.0057 | 0.2071 | 0.2206 | 0.2135 | 0.2219 | 0.2210 | 0 |
| | % err | 90.076% ± 0.638% | 88.993% | 90.482% | 89.704% | 90.652% | 90.550% | 0 |

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

## Discriminative & Predictive Classifier Losses (A18 / A19)

BCE loss during GRU/MLP discriminator training (A18) and MAE loss during GRU/MLP predictor training on
*synthetic* data (A19, TSTR), 5 seeds. A discriminator BCE near ln(2) ≈ 0.693 means real and generated
are indistinguishable.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (TimeVAE, disc / pred metrics)

> ⚠️ **Not a direct comparison.** The TimeVAE paper evaluates the *TimeGAN suite* on Sines (d=5, T=24)
> with GRU judges. We evaluate on Heston (d=1, T=128) with the benchmark's own GRU **and** MLP judges
> (hidden dim floored at `max(8, n_features·8)` to stay defined on univariate data).

| Metric | Paper — Sines | Ours — Heston GRU | Ours — Heston MLP |
|--------|:------------:|:-----------------:|:-----------------:|
| Disc Score ↓ | 0.021 ± 0.040 | 0.204 ± 0.193 | 0.176 ± 0.135 |
| Pred Score ↓ | 0.213 ± 0.000 | 0.058 ± 0.001 | 0.056 ± 0.000 |

Our Heston discriminative scores (0.20 GRU / 0.18 MLP) are far above the paper's near-perfect Sines
0.021 — Heston's heavy tails give the judges real signal to separate on, which TimeVAE-Base does not
match (consistent with A9/A28 above). The predictive scores are **not** comparable in magnitude: the
paper's 0.213 is the *sine "Original" LSTM floor* (5-D, harder), whereas our 0.058 is a 1-D next-step
MAE (inherently smaller). Same-metric reproduction on the paper's own sine dataset is in the next
section.

---

## Paper reproduction on Sine (our port vs TimeVAE paper)

Before running TimeVAE on Heston we reproduced the **TimeVAE paper result on the Sine dataset** with the
same PyTorch port and the paper's hyperparameters (latent 8, hidden 50/100/200), with `reconstruction_wt`
**tuned to 8.0** per the paper's per-dataset weight-tuning protocol (Sec. 4.2), scored with the
Yoon-et-al. discriminative/predictive metrics. This validates the generator port independently of Heston.
(The Heston benchmark above uses the released default `reconstruction_wt=3.0` — the tuning is specific to
the sine reproduction.) Full write-up:
[`../../../methods/TimeVAE/paper_reimplementation/`](../../../methods/TimeVAE/paper_reimplementation/).

| Dataset | Metric | Ours (PyTorch port, 5 seeds, recon_wt=8) | Paper (Table) | Verdict |
|---------|--------|:----------------------------------------:|:-------------:|---------|
| Sine | Discriminative ↓ | 0.073 ± 0.024 | 0.021 ± 0.040 | close — bands overlap |
| Sine | Predictive ↓ | 0.2133 ± 0.0001 | 0.213 ± 0.000 | **matches** ✓ |

The predictive score **reproduces the paper exactly** (0.2133 vs 0.213 — both sit on the "Original" LSTM
floor, i.e. synthetic sine is predictively indistinguishable from real). For the discriminative score, a
diagnostic isolated the cause of the initial gap as **KL over-regularization** at the released default
weight 3.0 (disc 0.114) — not the discriminator architecture (identical verbatim 1-layer Yoon GRU judge)
nor a prior hole. Tuning `reconstruction_wt` to 8.0 as the paper describes lets the decoder recover the
full signal dispersion and drops disc to **0.073 ± 0.024**, bringing our band [0.050, 0.097] into overlap
with the paper's [−0.019, 0.061]. ⚠️ **Honest caveat:** 8.0 is *above* the paper's stated tuning range of
[0.5, 3.5], so this is protocol-faithful but not range-faithful; the weight-3.0 result is preserved in the
JSON (`released_default_wt3_result`). See the paper-reimplementation README §4 for the full analysis.
Source: [`../../../methods/TimeVAE/paper_reimplementation/results/sine_paper_metrics.json`](../../../methods/TimeVAE/paper_reimplementation/results/sine_paper_metrics.json).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K=77 nearest TimeVAE paths by L2 in z-scored space, forecast with their price-anchored futures.

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 3.855 ± 0.070 | 3.856 ± 0.070 | 5.634 ± 0.124 | 5.634 ± 0.124 | 3.73 / 5.30 |
| MAE    | 4.414 ± 0.045 | 4.414 ± 0.045 | 6.549 ± 0.086 | 6.549 ± 0.086 | 3.73 / 5.30 |
| RMSE   | 6.064 ± 0.061 | 6.065 ± 0.061 | 8.947 ± 0.111 | 8.948 ± 0.111 | 5.07 / 7.18 |

PS-MC over the TimeVAE pool **does not beat the naive random walk**: CRPS 3.855 > 3.73 at H=32 and
5.634 > 5.30 at H=64, on all 5 seeds. Because the TimeVAE paths do not carry Heston's volatility
structure (A9/A31 above), their nearest-neighbour futures are no more informative than a driftless RW —
in fact slightly worse. This contrasts with Diffusion-TS, whose pool **beats** RW on CRPS (2.72 / 3.85).
Uniform ≈ Gaussian: Heston is time-homogeneous.

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B two-subline aggregates (MSE + % err) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts (summary.json + per-seed + plots) |

→ Cross-method comparison with TimeGAN, SBTS, Fourier Flow, Diffusion-TS & CSDI: [`results/README.md`](../../README.md)
