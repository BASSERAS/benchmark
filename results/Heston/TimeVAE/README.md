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

**Evaluation protocol (test set everywhere).** Generators were trained on the **train** split (seed 0) and
are **never scored on it**. Every metric below compares the 8 192 generated paths against the **held-out
test set** (an independent 8 192-path Heston draw, seed 1) — with one deliberate exception: the two
adversarial/predictive metrics A18 (discriminative) and A19 (predictive-TSTR) draw their *real* class from a
**third** Heston split (seed 2), so the judge never sees the same real data used everywhere else. This is the
protocol applied identically to all nine methods.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the best value a *perfect* generator can reach at this sample size. It is
measured by scoring an **independent Heston draw** (fresh seeds, identical parameters) against the same test
set — i.e. real-vs-real finite-sample noise. It is **non-zero** (finite samples never match exactly) and
**identical across all methods**, because it depends only on the test set and the protocol, not on the
generator. See [`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

<!-- ===== PER-METHOD A TABLE ===== -->
| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 2.257 ± 0.5719 | 1.367 | 2.662 | 1.791 | 2.668 | 2.799 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.02227 ± 1.22e-04 | 0.02204 | 0.02234 | 0.02223 | 0.02238 | 0.02234 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.03082 ± 1.05e-04 | 0.03062 | 0.03087 | 0.03084 | 0.03094 | 0.03082 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.02191 ± 1.17e-04 | 0.02170 | 0.02198 | 0.02187 | 0.02202 | 0.02198 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 1.831 ± 0.6794 | 1.491 | 0.6971 | 1.975 | 2.544 | 2.445 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.01914 ± 0.001334 | 0.02143 | 0.01928 | 0.01931 | 0.01751 | 0.01817 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.004951 ± 0.001715 | 0.007400 | 0.005300 | 0.005501 | 0.002120 | 0.004436 | 0.001983 |
| A8 Increment MMD² ↓ | 0.2130 ± 0.001204 | 0.2145 | 0.2134 | 0.2124 | 0.2109 | 0.2135 | 8.69e-04 |
| A9 Volatility MMD ↓ | 3.575 ± 0.4476 | 2.936 | 3.844 | 3.135 | 3.979 | 3.980 | 0.008554 |
| A10 Terminal SWD ↓ | 1.947 ± 0.3598 | 2.386 | 1.934 | 1.816 | 1.356 | 2.242 | 1.151 |
| A11 Path SWD ↓ | 1.167 ± 0.1135 | 1.358 | 1.036 | 1.135 | 1.085 | 1.223 | 0.6191 |
| A12 RV Law Loss ↓ | 5.010 ± 0.008395 | 4.995 | 5.015 | 5.006 | 5.017 | 5.015 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.3196 ± 0.2225 | 0.4066 | 0.2254 | 0.7117 | 0.08735 | 0.1669 | 0.1205 |
| A14 KS Log-returns ↓ | 0.3670 ± 0.004602 | 0.3610 | 0.3706 | 0.3619 | 0.3718 | 0.3699 | 0.001491 |
| A15 Skewness Error ↓ | 0.5479 ± 0.09837 | 0.4631 | 0.6806 | 0.4099 | 0.6083 | 0.5778 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.01057 ± 8.40e-05 | 0.01043 | 0.01063 | 0.01053 | 0.01064 | 0.01063 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.05127 ± 0.007848 | 0.06628 | 0.04895 | 0.04907 | 0.04895 | 0.04309 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.4272 ± 0.08815 | 0.3258 | 0.3129 | 0.5000 | 0.4994 | 0.4979 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.1358 ± 0.1503 | 0.3206 | 0.01999 | 0.3187 | 0.01846 | 0.001068 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05385 ± 7.71e-04 | 0.05289 | 0.05420 | 0.05298 | 0.05444 | 0.05476 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05243 ± 1.91e-04 | 0.05267 | 0.05225 | 0.05216 | 0.05253 | 0.05253 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 57.28 ± 1.758 | 58.11 | 57.97 | 59.77 | 55.11 | 55.44 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.3890 ± 0.1057 | 0.2236 | 0.4513 | 0.3049 | 0.4824 | 0.4828 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.3609 ± 0.08849 | 0.2283 | 0.4159 | 0.2818 | 0.4411 | 0.4374 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.4674 ± 0.1346 | 0.2821 | 0.5252 | 0.3320 | 0.5974 | 0.6002 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.4630 ± 0.1189 | 0.3057 | 0.5208 | 0.3342 | 0.5805 | 0.5738 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.3883 ± 0.2340 | 0.6172 | 0.2680 | 0.6588 | 0.02275 | 0.3746 | 0.1392 |
| A26 Return Std Error ↓ | 1.074 ± 0.007809 | 1.061 | 1.079 | 1.069 | 1.081 | 1.079 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.01098 ± 7.75e-05 | 0.01084 | 0.01103 | 0.01094 | 0.01105 | 0.01103 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 0.2834 ± 0.04765 | 0.3555 | 0.2392 | 0.3251 | 0.2533 | 0.2440 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.1745 ± 0.001776 | 0.1715 | 0.1756 | 0.1733 | 0.1761 | 0.1758 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.325 ± 0.04564 | 1.331 | 1.349 | 1.392 | 1.292 | 1.260 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.9869 ± 0.004527 | 0.9783 | 0.9870 | 0.9882 | 0.9911 | 0.9898 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.004576 ± 5.62e-05 | 0.004471 | 0.004579 | 0.004581 | 0.004629 | 0.004621 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.02254 ± 0.003796 | 0.01590 | 0.02234 | 0.02283 | 0.02402 | 0.02760 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.1803 ± 0.001643 | 0.1776 | 0.1813 | 0.1792 | 0.1818 | 0.1816 | 0.06559 |

**Reading the table.** TimeVAE reproduces the **marginal centre** of the Heston log-return distribution but
fails its **tails, volatility structure and autocorrelation**. Its single "win" is **A33 teacher-sigma
correlation (0.02254)** — the *least-bad* of all nine methods on a metric **every** generator fails: no model
recovers the latent Heston variance process from prices alone (floor 0.616, unreachable without the hidden
state), so 0.0225 vs the field's ≈0 is a technical win on a metric where the whole field is far from the floor,
not a sign of genuine variance-process recovery. Everywhere else TimeVAE is weak. Two numbers dominate.
(1) **A28 kurtosis ratio 0.283** — TimeVAE's returns are *under-dispersed* (0.28× the true excess kurtosis;
the Gaussian-decoder output is too thin-tailed), the opposite failure mode to CSDI (0.871, the A28 winner).
This propagates to A1 kurtosis error 2.26, A14 KS on log-returns 0.367 (high), and A9 volatility MMD **3.58**
— an order of magnitude worse than the diffusion models, the clearest sign TimeVAE does not capture Heston's
stochastic-vol signature. (2) **A31 rolling-vol KS 0.987** — the rolling-5 volatility distribution is almost
entirely disjoint from real, and A26 return-std error 1.07 confirms the generated paths are the wrong scale in
volatility. The autocorrelation (ARCH) structure is only weakly present: A21 ACF-|r| error 0.39, A23 lag-1
0.47 — far larger than the leaders (~0.002–0.02). A18 GRU discriminative is high (0.427 ± 0.088: seeds 2/3/4
fully separable at ≈0.50). This is an **honest weak result**: TimeVAE is a strong general-purpose TS-VAE but a
poor fit for heavy-tailed stochastic-vol price paths.

---

## Stylised Facts Diagnostic (Heston vs TimeVAE, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L′ (der), and its second finite difference
L″ (sec\_der) — then combine them into **three sub-scores per plot**:

- **MSE row** (decides the winner): for each list, mean((L\_gen − L\_real)²), averaged over the three lists
  (funct / der / sec\_der). This is the headline curve-fit error.
- **% err row** (function-level MAPE): mean(|L\_gen − L\_real| / (|L\_real| + 1e-6)) × 100 on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE row**: sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100 on the curve L
  only (funct-only).

↓ lower is better for all three rows. **Perfect floor** is the non-zero real-vs-test value an independent
Heston draw reaches — identical across methods. The enormous log-return-histogram MSE (968), ACF %err
(~2100–4600 %) and rolling-vol-histogram MSE (16 019) quantify the same failure the A table shows: TimeVAE
misses Heston's tails, autocorrelation and volatility shape by a wide margin.

<!-- ===== PER-METHOD B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 968.0 ± 183.1 | 693.2 | 1055 | 810.6 | 1152 | 1129 | 0.1098 |
|  | % err | 114.9% ± 0.6458% | 113.7% | 115.3% | 114.6% | 115.4% | 115.3% | 1.799% |
|  | NRMSE | 123.7% ± 6.783% | 112.7% | 128.1% | 118.8% | 129.6% | 129.4% | 0.5328% |
| **QQ plot** | MSE | 3.99e-05 ± 5.99e-07 | 3.88e-05 | 4.03e-05 | 3.96e-05 | 4.04e-05 | 4.03e-05 | 1.09e-09 |
|  | % err | 90.53% ± 1.555% | 87.70% | 91.54% | 90.01% | 91.53% | 91.87% | 0.4629% |
|  | NRMSE | 29.57% ± 0.2260% | 29.17% | 29.71% | 29.46% | 29.76% | 29.73% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.03390 ± 0.01422 | 0.01361 | 0.04073 | 0.02028 | 0.04678 | 0.04810 | 9.61e-06 |
|  | % err | 983.6% ± 273.1% | 577.3% | 1151% | 736.0% | 1218% | 1236% | 8.724% |
|  | NRMSE | 795.3% ± 212.4% | 476.6% | 910.4% | 609.1% | 982.9% | 997.7% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.02694 ± 0.01034 | 0.01224 | 0.03187 | 0.01699 | 0.03645 | 0.03716 | 9.17e-06 |
|  | % err | 1026% ± 265.1% | 637.2% | 1195% | 778.4% | 1254% | 1268% | 11.34% |
|  | NRMSE | 782.1% ± 188.7% | 505.1% | 881.9% | 609.1% | 952.2% | 962.3% | 6.486% |
| **Rolling vol histogram** | MSE | 16019 ± 2352 | 19160 | 12099 | 17212 | 16492 | 15132 | 1.372 |
|  | % err | 340.0% ± 11.74% | 352.8% | 318.0% | 345.9% | 341.3% | 341.9% | 2.264% |
|  | NRMSE | 221.5% ± 13.05% | 237.6% | 198.3% | 228.5% | 223.2% | 219.7% | 0.8688% |
| **Tail survival** | MSE | 0.07224 ± 0.001903 | 0.06900 | 0.07349 | 0.07113 | 0.07393 | 0.07364 | 5.22e-07 |
|  | % err | 90.06% ± 0.6385% | 88.98% | 90.47% | 89.69% | 90.64% | 90.54% | 0.3302% |
|  | NRMSE | 46.97% ± 0.6196% | 45.92% | 47.38% | 46.62% | 47.52% | 47.43% | 0.1050% |

TimeVAE wins **none of the 6 B-plots** — every curve-fit MSE is far above the leaders. Its relatively least-bad
curve is the QQ plot (MSE 3.99e-05), but even that NRMSE (18.7 %) is well above the perfect floor (0.34 %).

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
are indistinguishable. The real class for A18/A19 is drawn from the **disc split (seed 2)**, never the test set.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (TimeVAE, disc / pred metrics)

> ⚠️ **Not a direct comparison.** The TimeVAE paper evaluates the *TimeGAN suite* on Sines (d=5, T=24)
> with GRU judges. We evaluate on Heston (d=1, T=128) with the benchmark's own GRU **and** MLP judges
> (hidden dim floored at `max(8, n_features·8)` to stay defined on univariate data).

| Metric | Paper — Sines | Ours — Heston GRU | Ours — Heston MLP |
|--------|:------------:|:-----------------:|:-----------------:|
| Disc Score ↓ | 0.021 ± 0.040 | 0.427 ± 0.088 | 0.136 ± 0.150 |
| Pred Score ↓ | 0.213 ± 0.000 | 0.054 ± 0.001 | 0.052 ± 0.000 |

Our Heston discriminative scores (0.43 GRU / 0.14 MLP) are far above the paper's near-perfect Sines
0.021 — Heston's heavy tails give the judges real signal to separate on, which TimeVAE-Base does not
match (consistent with A9/A28 above). The predictive scores are **not** comparable in magnitude: the
paper's 0.213 is the *sine "Original" LSTM floor* (5-D, harder), whereas our 0.054 is a 1-D next-step
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
retrieve K nearest TimeVAE paths by L2 in z-scored space, forecast with their price-anchored futures.
CRPS is scored against the test set at two horizons; the naive random-walk (RW) baseline is
3.738 (H=32) / 5.246 (H=64). Full analysis: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 3.912 ± 0.07154 | 3.738 |
| PS-MC CRPS H=64 ↓ | 5.670 ± 0.1222 | 5.246 |

PS-MC over the TimeVAE pool **does not beat the naive random walk**: CRPS 3.912 > 3.738 at H=32 and
5.670 > 5.246 at H=64, on all 5 seeds — one of only two pools (with COSCI-GAN) that fail RW. Because the
TimeVAE paths do not carry Heston's volatility structure (A9/A31 above), their nearest-neighbour futures are
no more informative than a driftless RW — in fact slightly worse. This contrasts with the diffusion/SSM pools
(LS4, CSDI, Diffusion-TS), which all beat RW comfortably. Heston is time-homogeneous, so the uniform and
Gaussian prefix weightings coincide.

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B three-subline aggregates (MSE + % err + NRMSE) |
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

→ Cross-method comparison with all nine generators: [`results/README.md`](../../README.md)
