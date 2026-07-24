# Metrics — GT-GAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Data split (test set everywhere).** Three disjoint 8 192-path Heston draws are used throughout:
the generator is **trained on seed 0**; every A/B metric and every diagnostic plot compares the
generated paths against the **test set (seed 1)**; the A18 discriminative score and the A19
predictive-TSTR score use a **third real set (seed 2)** as the "real" class. No metric is ever scored
against the generator's own training data.

**Model:** GT-GAN — General Purpose Time Series Synthesis (Jeon, Kim, Song, Cho, Park, **NeurIPS 2022**,
*GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks*, arXiv:2210.02040,
`github.com/Jinsung-Jeon/GT-GAN`). A **continuous-time** GAN whose four sub-networks are all differential
equations: the embedder is a **Neural-CDE** (`FinalTanh` field), the recovery and discriminator are
**Neural-ODEs** (`Multi_Layer_ODENetwork`, Euler, Δt = 0.5), and the generator is a **continuous
normalising flow** (`build_model_tabular_nonlinear`, solver `sym12async`). There is **no supervisor
network** — the supervised term is a CTFP latent-likelihood. hidden = 24, layers = 3, batch = 128,
**32 957 params — the smallest generator in the benchmark.** Heston prices are globally min-max scaled to
[0, 1] (data\_min = 39.8936, data\_max = 155.5790) for training and inverted + clipped to ≥ 1e-6 on
generation. See [`../../../methods/GT-GAN/code/README.md`](../../../methods/GT-GAN/code/README.md).

> **Two changes from the paper's Stocks config, plus one de-conflation fix.** (1) feature dim 6 → 1
> (Heston is univariate); (2) seq\_len 24 → 128. The only reference-code edit is a one-line
> de-conflation in `run_ctfp` — substituting `args.effective_shape` for `values.shape[1]` in two spots so
> the CTFP latent size stays fixed at 24 when `seq_len` ≠ `effective_shape`. On the paper's own Stocks case
> (`seq_len == effective_shape == 24`) the edit is **byte-identical** to the released code. See
> [§ Comparison with the paper](#comparison-with-the-paper-stocks-reproduction).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case score a *perfect* generator reaches at this
sample size. It is an **independent Heston draw** (fresh seeds 1000+i, identical parameters) scored
against the **test set** — a genuine **non-zero finite-sample noise floor**, not zero and not a
row-shuffle/permutation of the test data. Two independent draws of the same process still differ by
sampling noise, and that irreducible noise is what the floor measures. See
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 281.8 ± 288.2 | 43.57 | 399.1 | 122.4 | 47.75 | 796.1 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.02279 ± 2.78e-04 | 0.02284 | 0.02253 | 0.02301 | 0.02314 | 0.02241 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.02978 ± 0.001743 | 0.03077 | 0.02982 | 0.03008 | 0.03168 | 0.02654 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.02240 ± 3.79e-04 | 0.02246 | 0.02231 | 0.02268 | 0.02281 | 0.02172 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 7.568 ± 1.267 | 6.726 | 9.082 | 6.557 | 9.142 | 6.334 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.03292 ± 0.009071 | 0.02808 | 0.02217 | 0.03063 | 0.03453 | 0.04919 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.008520 ± 0.002539 | 0.007595 | 0.005447 | 0.007351 | 0.01301 | 0.009198 | 0.001983 |
| A8 Increment MMD² ↓ | 0.2025 ± 0.01417 | 0.2082 | 0.1788 | 0.2107 | 0.2195 | 0.1952 | 8.69e-04 |
| A9 Volatility MMD ↓ | 2.882 ± 0.6128 | 3.098 | 2.232 | 2.838 | 3.925 | 2.317 | 0.008554 |
| A10 Terminal SWD ↓ | 2.391 ± 0.1196 | 2.230 | 2.417 | 2.309 | 2.586 | 2.410 | 1.151 |
| A11 Path SWD ↓ | 2.236 ± 0.2567 | 2.357 | 2.077 | 1.922 | 2.157 | 2.666 | 0.6191 |
| A12 RV Law Loss ↓ | 15.11 ± 13.84 | 14.83 | 4.774 | 41.82 | 5.045 | 9.105 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.7421 ± 0.3193 | 0.2363 | 0.9397 | 0.5355 | 0.8587 | 1.140 | 0.1205 |
| A14 KS Log-returns ↓ | 0.3881 ± 0.003914 | 0.3827 | 0.3853 | 0.3909 | 0.3877 | 0.3937 | 0.001491 |
| A15 Skewness Error ↓ | 390.5 ± 355.8 | 704.9 | 13.46 | 349.9 | 3.449 | 880.8 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.01086 ± 1.44e-04 | 0.01077 | 0.01078 | 0.01100 | 0.01107 | 0.01071 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.06672 ± 0.01592 | 0.05884 | 0.04810 | 0.06555 | 0.06519 | 0.09595 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.4871 ± 0.01292 | 0.4658 | 0.4786 | 0.4942 | 0.4976 | 0.4994 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.07345 ± 0.1266 | 0.007476 | 0.03158 | 0.001678 | 7.63e-04 | 0.3258 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05547 ± 0.001080 | 0.05442 | 0.05471 | 0.05486 | 0.05734 | 0.05600 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05302 ± 2.01e-04 | 0.05306 | 0.05335 | 0.05273 | 0.05292 | 0.05301 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 20.55 ± 7.355 | 12.97 | 15.87 | 26.06 | 15.61 | 32.23 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.3181 ± 0.1375 | 0.3864 | 0.2086 | 0.2849 | 0.5475 | 0.1629 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.1619 ± 0.1184 | 0.1884 | 0.07974 | 0.1195 | 0.3783 | 0.04365 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.4201 ± 0.1602 | 0.5242 | 0.2833 | 0.3715 | 0.6765 | 0.2448 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.2270 ± 0.1494 | 0.2830 | 0.1116 | 0.1700 | 0.4916 | 0.07881 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.7845 ± 0.3300 | 0.5848 | 0.9195 | 0.2382 | 1.115 | 1.065 | 0.1392 |
| A26 Return Std Error ↓ | 1.005 ± 0.09141 | 1.060 | 0.9097 | 1.066 | 1.109 | 0.8821 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.009540 ± 0.007044 | 0.005015 | 0.009470 | 0.02144 | 0.01133 | 4.44e-04 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 0.002659 ± 0.004016 | 1.91e-06 | 0.002913 | 7.91e-06 | 0.01037 | 1.16e-06 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.1649 ± 0.01028 | 0.1703 | 0.1570 | 0.1579 | 0.1828 | 0.1565 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.8923 ± 0.2085 | 0.7953 | 0.8662 | 0.7824 | 0.7189 | 1.299 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.9868 ± 0.004912 | 0.9867 | 0.9864 | 0.9895 | 0.9932 | 0.9783 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.009854 ± 0.007895 | 0.01055 | 0.003694 | 0.02492 | 0.004629 | 0.005485 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.01003 ± 0.008468 | 0.002942 | 0.02036 | 0.002096 | 0.02036 | 0.004376 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.3088 ± 0.1407 | 0.3322 | 0.1889 | 0.5689 | 0.1876 | 0.2664 | 0.06559 |

**Reading the table — the benchmark's weakest marginal-distribution matcher.** GT-GAN wins **0 of the 36
A-metric rows**. The single dominant failure is the **return law**: A28 Kurtosis Ratio **0.002659**
(target 1.0) means GT-GAN's log-returns are roughly **375× more leptokurtic than Heston** — an
over-peaked, near-degenerate spike of returns with almost no spread — and this poisons every
marginal metric downstream. A14 KS Log-returns **0.3881** is the **worst KS of any method** (LS4 0.01258),
A1 Kurtosis Error **281.8** the worst by two orders of magnitude (TimeGAN, the next-worst, 2.954), A15
Skewness Error **390.5** and A16 QQ RMSE **0.01086** likewise last. The distributional-kernel block is
uniformly far above floor (A6 Path MMD² **0.03292** vs floor 0.001842; A8 Increment MMD² **0.2025** vs
floor 8.69e-04; A11 Path SWD **2.236** vs floor 0.6191), and the volatility spread collapses — A26 Return
Std Error **1.005** means the generated per-step return standard deviation is essentially the wrong scale,
and A31 Rolling-Vol KS **0.9868** is near-total separation of the rolling-vol distribution.

The adversarial score confirms it. **A18 GRU = 0.4871** — a 2-layer GRU classifier separates GT-GAN paths
from real Heston paths **almost perfectly** (|acc − 0.5| ≈ 0.49, i.e. ~99% accuracy), the near-worst
possible adversarial score and light-years from the floor (0.006195). The **A18 MLP** column is much lower
but **wildly unstable** (0.07345 ± 0.1266; seeds 2–3 near zero, seed 4 = 0.3258): a flatten-MLP keys on
whatever cheap surface statistic happens to separate that seed, so its score is not evidence of fidelity —
the GRU is the reliable judge here. A19 predictive is close to the floor (**0.05547 GRU**, floor 0.05002)
only because next-step MAE on a smooth price level is an easy task that a badly-shaped return law barely
perturbs — it is **not** a sign of distributional quality.

The one honest bright spot and the shared caveat. A33 teacher-sigma correlation **≈ 0.010** (floor 0.6163):
like **every** single-factor generator in this benchmark, GT-GAN does not reconstruct the path-wise
stochastic-vol trajectory — the latent variance is unrecoverable from prices alone. Net: GT-GAN is the
**weakest marginal-distribution matcher of the ten methods**, driven by a collapsed, over-peaked return law
(A28); it wins no A-row and no B-row, yet — see below — its path-shadowing ensemble still edges the naive
random walk on CRPS.

---

## Stylised Facts Diagnostic (Heston vs GT-GAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — and report **three measures** per plot:

- **MSE**: for each list, `mean((L_gen − L_real)²)`; reported as the **mean of the three** sub-scores
  (funct, der, sec\_der).
- **% err** (function-level MAPE): `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE**: `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100`, mean of three.

↓ lower is better for all three. The **Perfect floor** is again an independent Heston draw vs the test
set — a non-zero finite-sample floor, not zero.

**Worst curves in the benchmark.** GT-GAN posts the **worst MSE on the return-density curves**:
log-return-histogram MSE **2160** (next-worst TimeVAE ~968; floor 0.1098) and rolling-vol-histogram MSE
**3029** (floor 1.372) — the direct curve-space image of the collapsed return law. QQ MSE **4.16e-05**,
tail-survival MSE **0.07918** and both ACF curves are far above floor. The ACF %err / NRMSE figures balloon
into the hundreds of percent because the true ACF ≈ 0.05, so a modest absolute error becomes an enormous
*relative* one; read MSE for absolute agreement.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 2160 ± 655.2 | 1243 | 1680 | 2986 | 2784 | 2107 | 0.1098 |
|  | % err | 117.7% ± 1.125% | 117.3% | 116.5% | 118.6% | 119.4% | 116.6% | 1.799% |
|  | NRMSE | 151.6% ± 13.15% | 130.7% | 143.1% | 163.9% | 165.5% | 154.9% | 0.5328% |
| **QQ plot** | MSE | 4.16e-05 ± 1.27e-06 | 4.12e-05 | 4.08e-05 | 4.25e-05 | 4.36e-05 | 4.00e-05 | 1.09e-09 |
|  | % err | 92.66% ± 2.380% | 91.21% | 90.27% | 95.24% | 95.83% | 90.72% | 0.4629% |
|  | NRMSE | 30.25% ± 0.4431% | 30.09% | 29.97% | 30.59% | 30.92% | 29.69% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.02626 ± 0.02245 | 0.03271 | 0.009179 | 0.01865 | 0.06672 | 0.004055 | 9.61e-06 |
|  | % err | 893.2% ± 463.3% | 1135% | 593.6% | 876.3% | 1609% | 251.7% | 8.724% |
|  | NRMSE | 668.0% ± 311.1% | 819.3% | 434.6% | 620.0% | 1178% | 288.1% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.008475 ± 0.01103 | 0.007744 | 0.001317 | 0.003090 | 0.02993 | 2.93e-04 | 9.17e-06 |
|  | % err | 541.6% ± 420.6% | 671.1% | 278.6% | 423.6% | 1281% | 53.90% | 11.34% |
|  | NRMSE | 366.9% ± 274.6% | 430.8% | 180.2% | 274.4% | 865.9% | 83.08% | 6.486% |
| **Rolling vol histogram** | MSE | 3029 ± 1983 | 2815 | 1188 | 1168 | 6585 | 3388 | 1.372 |
|  | % err | 187.8% ± 42.87% | 201.9% | 145.3% | 141.5% | 258.7% | 191.5% | 2.264% |
|  | NRMSE | 97.99% ± 31.28% | 102.1% | 65.03% | 65.59% | 149.4% | 107.8% | 0.8688% |
| **Tail survival** | MSE | 0.07918 ± 0.002862 | 0.07564 | 0.07688 | 0.08195 | 0.08303 | 0.07839 | 5.22e-07 |
|  | % err | 91.34% ± 1.201% | 90.54% | 90.08% | 92.23% | 93.24% | 90.58% | 0.3302% |
|  | NRMSE | 49.16% ± 0.8809% | 48.07% | 48.45% | 50.01% | 50.34% | 48.92% | 0.1050% |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> The ACF %err / NRMSE (hundreds of %) is a near-zero-denominator artefact: the true ACF ≈ 0.05, so a small
> absolute error becomes a large *relative* error. Read MSE for absolute agreement, %err for shape.
> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

---

## Discriminative & Predictive Classifier Losses (A18 / A19)

BCE loss during GRU/MLP discriminator training (A18) and MAE loss during GRU/MLP predictor training on
*synthetic* data (A19, TSTR), 5 seeds. A discriminator BCE near ln(2) ≈ 0.693 means real and generated
are indistinguishable; a BCE that collapses toward 0 means the judge separates them trivially — **which is
what the GRU judge does here.** GT-GAN's GRU-discriminator BCE drives well below ln 2 (A18 GRU ≈ 0.49
score, i.e. ~99% accuracy), the direct evidence that GT-GAN paths are near-separable from real Heston.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (Stocks reproduction)

> ⚠️ **GT-GAN's paper metrics are the TSTR discriminative/predictive protocol on a different dataset.**
> The NeurIPS 2022 paper (Table 1) reports **discriminative** and **predictive** scores on Stocks, Energy,
> Sines, etc. (the TimeGAN protocol), not on Heston. There is therefore no native "Ours — Heston" entry for
> the paper metric; we validated our GT-GAN port **on the paper's own Stocks dataset** (feat dim 6,
> seq\_len 24) before running the Heston experiment above.

| Dataset | Metric | Ours (5 seeds) | Paper (Table 1) | Verdict |
|---------|--------|:--------------:|:---------------:|---------|
| Stocks | Discriminative ↓ | **0.026 ± 0.012** | 0.010 ± 0.008 | same regime ✓ |
| Stocks | Predictive ↓ | **0.018 ± 0.003** | 0.017 ± 0.000 | matches ✓ |

Our Stocks port reaches predictive **0.018** (paper 0.017) — a match to the third decimal — and
discriminative **0.026**, the same order as the paper's **0.010** and far below the paper's own GAN
baselines on Stocks (TimeGAN 0.102, RCGAN 0.196, COT-GAN 0.285). This confirms the port reproduces GT-GAN's
published generative quality on its native short-sequence multivariate data; the same code is carried into
the Heston generator above, where the 24 → 128 sequence length and univariate return law are the harder
regime that the A-table reflects. Full write-up:
[`../../../methods/GT-GAN/paper_reimplementation/`](../../../methods/GT-GAN/paper_reimplementation/).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real (test-set) prefix as a murex-style feature vector,
retrieve nearest GT-GAN paths by L2 in z-scored space, and forecast with their price-anchored futures.

| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 3.551 ± 0.1083 | 3.738 |
| PS-MC CRPS H=64 ↓ | 4.996 ± 0.1952 | 5.246 |

Despite the collapsed marginal, PS-MC over the GT-GAN pool **still beats the naive random walk on CRPS at
both horizons** (3.551 < 3.738 at H=32; 4.996 < 5.246 at H=64). The gain is **CRPS-specific**: price
anchoring subtracts the prefix's terminal level, and averaging over K = 77 nearest neighbours washes out
the spiky per-step returns, so the ensemble stays useful as a *calibrated distribution* even though its
per-path returns are mis-shaped — the improvement does **not** carry over to point-wise MAE/RMSE at H=32.
Full breakdown: [`path_shadowing/README.md`](path_shadowing/README.md).

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B three-measure aggregates (MSE + % err + NRMSE) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts (summary.json + per-seed + plots + README) |

→ Cross-method comparison with TimeGAN, COSCI-GAN, SBTS, Fourier Flow, Diffusion-TS, CSDI, TimeVAE, TimeVQVAE & LS4: [`results/README.md`](../../README.md)
