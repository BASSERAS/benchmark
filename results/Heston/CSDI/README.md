# Metrics — CSDI on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** CSDI (Tashiro, Song, Song, Ermon, NeurIPS 2021 — *Conditional Score-based Diffusion
Models for Probabilistic Time Series Imputation*, arXiv:2107.03502), adapted as an **unconditional**
score-based diffusion generator. 4 residual layers, 64 channels, 8 heads, 50 diffusion steps
(quadratic β schedule 1e-4 → 0.5), 200 epochs, Adam (lr 1e-3, wd 1e-6) + MultiStepLR ×0.1 @ 150/180.
Unconditional generation is obtained by forcing the conditioning mask to zero (`cond_mask ≡ 0`), so
every point becomes a diffusion target — a pure DDPM sampler. Base config is the released `base.yaml`,
verbatim. See [`../../../methods/CSDI/code/README.md`](../../../methods/CSDI/code/README.md).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Data split — train / test / disc

Every number on this page is an **out-of-sample** score. The benchmark uses three disjoint Heston
draws of 8 192 paths each:

- **Train (seed 0)** — the paths the generator was fitted on. Never scored here.
- **Test (seed 1)** — the held-out real reference. All A1–A17, A20–A34, every B curve, the diagnostic
  plots and PS-MC are computed **generated-vs-test**.
- **Disc (seed 2)** — a third independent real draw, used only as the "real" class for the A18
  discriminative and A19 predictive-TSTR classifiers, so the adversary never sees the test set.

---

## What we generate — price paths from the Heston SDE

The benchmark target is 8 192 Heston paths $dS_t = \mu S_t dt + \sqrt{v_t} S_t dW^S_t$,
$dv_t = \kappa(\theta - v_t) dt + \xi\sqrt{v_t} dW^v_t$ with $\mathrm{corr}(dW^S, dW^v) = \rho$.
Before training, prices are **z-scored** by the global real-data statistics (mean **101.3255**,
std **9.9717**); the diffusion model operates on standardized paths and outputs are mapped back to the
original price scale for metric scoring. All A/B metrics below are computed on the de-standardized
generated paths against the **test** Heston paths.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the non-zero finite-sample noise floor a perfect generator reaches.
It is measured by scoring an **independent Heston draw** (fresh seeds, identical parameters) against the
test set — the same real-vs-real comparison every generated batch faces, so the floor is the best score
attainable when the model *is* the true process. Floors are **non-zero** because two independent finite
samples of the same law never coincide exactly; they are identical across methods (same test set, same
independent-draw protocol). See [`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

<!-- ===== PER-METHOD A TABLE ===== -->
| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.09543 ± 0.02623 | 0.1177 | 0.1035 | 0.05046 | 0.08336 | 0.1220 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.005393 ± 1.50e-04 | 0.005323 | 0.005456 | 0.005281 | 0.005248 | 0.005658 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.007327 ± 2.29e-04 | 0.007126 | 0.007489 | 0.007149 | 0.007169 | 0.007699 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.005296 ± 1.50e-04 | 0.005211 | 0.005359 | 0.005186 | 0.005164 | 0.005563 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 1.426 ± 0.5856 | 1.737 | 1.047 | 0.6381 | 1.361 | 2.348 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.003646 ± 4.16e-04 | 0.003767 | 0.003480 | 0.003718 | 0.004272 | 0.002992 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.003605 ± 8.41e-04 | 0.002982 | 0.004824 | 0.004161 | 0.003617 | 0.002441 | 0.001983 |
| A8 Increment MMD² ↓ | 0.008062 ± 7.11e-04 | 0.007420 | 0.008333 | 0.007393 | 0.007855 | 0.009308 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.2498 ± 0.01607 | 0.2275 | 0.2532 | 0.2397 | 0.2527 | 0.2757 | 0.008554 |
| A10 Terminal SWD ↓ | 1.618 ± 0.2760 | 1.446 | 1.794 | 2.063 | 1.498 | 1.290 | 1.151 |
| A11 Path SWD ↓ | 1.069 ± 0.1305 | 1.079 | 0.9828 | 1.231 | 1.181 | 0.8722 | 0.6191 |
| A12 RV Law Loss ↓ | 1.920 ± 0.05633 | 1.892 | 1.947 | 1.869 | 1.875 | 2.018 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.3654 ± 0.3226 | 0.1979 | 0.1927 | 0.9994 | 0.1257 | 0.3114 | 0.1205 |
| A14 KS Log-returns ↓ | 0.05391 ± 0.001972 | 0.05283 | 0.05419 | 0.05293 | 0.05200 | 0.05760 | 0.001491 |
| A15 Skewness Error ↓ | 0.03681 ± 0.002124 | 0.03865 | 0.03802 | 0.03396 | 0.03453 | 0.03891 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.002576 ± 8.57e-05 | 0.002538 | 0.002612 | 0.002499 | 0.002502 | 0.002726 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.03667 ± 0.004476 | 0.03088 | 0.03357 | 0.04333 | 0.04004 | 0.03552 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.06302 ± 0.1056 | 0.002289 | 0.002289 | 0.2736 | 0.02640 | 0.01053 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.01138 ± 0.002541 | 0.01083 | 0.01327 | 0.01175 | 0.01419 | 0.006866 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05024 ± 1.88e-05 | 0.05024 | 0.05022 | 0.05023 | 0.05023 | 0.05028 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05025 ± 1.43e-04 | 0.05015 | 0.05014 | 0.05014 | 0.05050 | 0.05033 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 41.55 ± 5.776 | 41.81 | 46.67 | 31.79 | 47.96 | 39.51 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.01126 ± 0.003095 | 0.009799 | 0.01173 | 0.006607 | 0.01209 | 0.01608 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.01124 ± 0.002605 | 0.01018 | 0.01175 | 0.007159 | 0.01196 | 0.01516 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.02252 ± 0.004755 | 0.02227 | 0.02184 | 0.01572 | 0.02212 | 0.03065 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.02168 ± 0.003561 | 0.02186 | 0.02003 | 0.01708 | 0.02147 | 0.02796 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.5139 ± 0.4595 | 0.3424 | 0.3112 | 1.393 | 0.05548 | 0.4674 | 0.1392 |
| A26 Return Std Error ↓ | 0.2580 ± 0.009849 | 0.2552 | 0.2621 | 0.2446 | 0.2537 | 0.2742 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.002667 ± 8.89e-05 | 0.002621 | 0.002708 | 0.002587 | 0.002595 | 0.002822 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 0.8706 ± 0.03043 | 0.8257 | 0.8626 | 0.9004 | 0.9085 | 0.8557 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.04078 ± 0.001489 | 0.04017 | 0.04140 | 0.03941 | 0.03953 | 0.04340 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.134 ± 0.1303 | 1.160 | 1.261 | 0.9401 | 1.277 | 1.034 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.2202 ± 0.008329 | 0.2166 | 0.2217 | 0.2131 | 0.2139 | 0.2357 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.001048 ± 2.14e-05 | 0.001016 | 0.001053 | 0.001060 | 0.001034 | 0.001078 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.003948 ± 0.003596 | 9.19e-04 | 0.006939 | 0.005420 | -0.001423 | 0.007886 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.09917 ± 6.44e-04 | 0.09932 | 0.09921 | 0.09814 | 0.09902 | 0.1002 | 0.06559 |

**Footnotes.**
- **A18** — discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). Seed 2's GRU outlier (0.274) drives the wide A18-GRU std; the other four seeds sit at or near the perfect floor (≤ 0.026), and the MLP variant is a stable 0.011.
- **A19** — TSTR predictive MAE; all methods cluster near 0.050 (irreducible log-return floor) (GRU + MLP).
- **A33** — Teacher-Sigma correlation (Heston-recovered vol vs teacher σ), **higher is better**; perfect floor ≈ 0.616 (unreachable from prices alone — the latent variance process is hidden).
- **A34** — Teacher-Sigma RMSE, perfect floor ≈ 0.066.

**Reading the table.** CSDI **wins 3 of the 36 A-metric rows** in the cross-method comparison: **A1 kurtosis
error (0.09543)** — the tightest fourth-moment match of any method; **A21 ACF |r| error (0.01126)** — the
best autocorrelation-magnitude fit; and **A28 kurtosis ratio (0.8706)** — the closest to the perfect 1.0
(it slightly *under*-disperses the tails rather than over-dispersing, where most methods sit further off).
Its distributional block is strong throughout: A6/A7 path & terminal MMD² (0.0036 / 0.0036) sit close to
their floors, and A16 QQ RMSE (0.00258) is competitive. Where it is **weak** is where every price-only
generator is weak on Heston: A33 teacher-sigma correlation ≈ 0 (no method recovers the latent variance
path; floor 0.616), and A9 volatility MMD / A26 return-std error / A31 rolling-vol KS all remain well above
floor. A20–A24 land near the true Heston ACF magnitude, so the ARCH signature is roughly matched.

---

## Stylised Facts Diagnostic

Eight-panel comparison (seed 0): sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — and report **three rows per plot**:

- **MSE** — `mean((L_gen − L_real)²)`, averaged over the three lists (funct/der/sec\_der). This is the
  quantity that decides the cross-method winner.
- **% err** — function-level MAPE `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE** — `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100` on the curve L
  only (funct-only).

*Special case:* for **Tail survival** the % err and NRMSE rows use the **function-level curve only**
(funct), because the near-zero true differences of the survival derivative make a relative error explode;
its MSE row stays the mean-of-three.

↓ lower is better for all three rows. **Perfect floor** = the same independent-draw-vs-test floor as the A
table (non-zero, finite-sample). Winner between methods is decided by the **MSE** row.

<!-- ===== PER-METHOD B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 4.644 ± 0.4940 | 4.521 | 4.808 | 4.128 | 4.249 | 5.513 | 0.1098 |
|  | % err | 35.27% ± 1.063% | 34.79% | 35.69% | 34.36% | 34.35% | 37.16% | 1.799% |
|  | NRMSE | 9.998% ± 0.5467% | 9.889% | 10.19% | 9.368% | 9.601% | 10.94% | 0.5328% |
| **QQ plot** | MSE | 2.36e-06 ± 1.57e-07 | 2.28e-06 | 2.43e-06 | 2.22e-06 | 2.23e-06 | 2.63e-06 | 1.09e-09 |
|  | % err | 24.22% ± 1.083% | 23.73% | 24.23% | 24.45% | 22.67% | 26.00% | 0.4629% |
|  | NRMSE | 7.188% ± 0.2370% | 7.077% | 7.296% | 6.977% | 6.987% | 7.603% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 3.02e-05 ± 1.61e-05 | 2.16e-05 | 2.98e-05 | 1.42e-05 | 2.48e-05 | 6.08e-05 | 9.61e-06 |
|  | % err | 19.26% ± 8.314% | 11.94% | 22.67% | 16.14% | 11.69% | 33.87% | 8.724% |
|  | NRMSE | 19.33% ± 5.196% | 16.17% | 19.90% | 13.31% | 18.57% | 28.70% | 6.071% |
| **ACF r² lags 1–20** | MSE | 2.71e-05 ± 1.16e-05 | 2.18e-05 | 2.72e-05 | 1.36e-05 | 2.42e-05 | 4.85e-05 | 9.17e-06 |
|  | % err | 21.75% ± 10.67% | 14.05% | 28.03% | 13.10% | 13.65% | 39.93% | 11.34% |
|  | NRMSE | 20.43% ± 5.060% | 17.81% | 20.99% | 13.87% | 20.22% | 29.25% | 6.486% |
| **Rolling vol histogram** | MSE | 157.5 ± 12.45 | 152.3 | 157.9 | 148.6 | 147.4 | 181.3 | 1.372 |
|  | % err | 61.91% ± 2.364% | 60.96% | 62.71% | 59.19% | 60.62% | 66.08% | 2.264% |
|  | NRMSE | 24.39% ± 0.9523% | 23.99% | 24.45% | 23.72% | 23.61% | 26.21% | 0.8688% |
| **Tail survival** | MSE | 0.001960 ± 1.85e-04 | 0.001909 | 0.002027 | 0.001757 | 0.001824 | 0.002283 | 5.22e-07 |
|  | % err | 24.78% ± 0.8772% | 24.46% | 25.16% | 23.89% | 24.09% | 26.31% | 0.3302% |
|  | NRMSE | 7.733% ± 0.3598% | 7.641% | 7.873% | 7.329% | 7.468% | 8.354% | 0.1050% |

**CSDI's one B-win:** it takes the **ACF |r|** plot on MSE (3.02e-05), the tightest autocorrelation-magnitude
curve of any method — consistent with its A21 win above. On the other five plots it trails LS4, though its
QQ (2.36e-06) and tail-survival (0.00196) curves remain among the better fits.

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

## Comparison with the paper (paper's own metric: CRPS)

This section uses **the paper's own headline metric** — the **CRPS** (Continuous Ranked Probability Score,
approximated over 19 quantile levels from `nsample=100` imputations, normalised by the sum of ground-truth
absolute values; `utils.calc_quantile_CRPS`), computed by the authors' `utils.evaluate` — unchanged.

⚠️ **Direct comparison is not meaningful.** The paper's CRPS is measured on **sparse multivariate**
data — PhysioNet (35 clinical variables, T=48, self-supervised random missingness) and PM2.5 (36 stations,
T=36, real structured missingness). Our **Heston** column applies the **same conditional CSDI** (`is_unconditional=0`,
`target_strategy='random'`, `nsample=100`, scored by the identical `calc_quantile_CRPS`) to **univariate**
(d=1), **T=128**, **z-scored** Heston prices. Heston paths are smooth and low-dimensional, so their
imputation CRPS is *expected* to be far below the clinical/air-quality numbers — the lower value reflects an
**easier task**, not a better model. The missing ratios r0.1 / r0.5 / r0.9 map to the paper's PhysioNet
10% / 50% / 90% held-out fractions.

### A — Hyperparameter verification

| Setting | Our reimplementation (Heston imputation) | Paper (source) |
|---------|------------------------------------------|----------------|
| `model.is_unconditional` | 0 (conditional imputation) | 0 — Table 2 headline (`base.yaml`) |
| `model.target_strategy` | `random` | `random` (PhysioNet) (`base.yaml`) |
| `diffusion.num_steps` | 50 | 50 (`base.yaml`) |
| `diffusion.schedule` | `quad` (β 1e-4 → 0.5) | `quad`, 1e-4 → 0.5 (`base.yaml`) |
| `diffusion.layers / channels / nheads` | 4 / 64 / 8 | 4 / 64 / 8 (`base.yaml`) |
| `diffusion.diffusion_embedding_dim` | 128 | 128 (`base.yaml`) |
| `model.timeemb / featureemb` | 128 / 16 | 128 / 16 (`base.yaml`) |
| `train.epochs` | 200 | 200 (`base.yaml`) |
| `train.batch_size` | 16 | 16 (`base.yaml`) |
| `train.lr` (Adam, wd 1e-6, MultiStepLR ×0.1) | 1e-3 @ 150/180 | 1e-3 @ 0.75/0.9 (`base.yaml`) |
| `nsample` (CRPS) | 100 | 100 (paper §Experiments) |
| target_dim | 1 (Heston, univariate) | 35 (PhysioNet) / 36 (PM2.5) |

### B — CRPS: paper vs our reproduction vs Heston

| Metric (paper's own) | Paper (Table 2, PhysioNet/PM2.5) | Ours — PhysioNet/PM2.5 (paper dataset) | Ours — Heston |
|----------------------|:--------------------------------:|:--------------------------------------:|:-------------:|
| CRPS ↓ — 10% missing | 0.238 (0.001) | **0.2344** | **0.0717** |
| CRPS ↓ — 50% missing | 0.330 (0.002) | **0.3364** | **0.0861** |
| CRPS ↓ — 90% missing | 0.522 (0.002) | **0.5256** | **0.1656** |
| CRPS ↓ — PM2.5 | 0.108 (0.001) | **0.1064** | — |

The **Ours — Heston** column is scored by the same authors' evaluator (`evaluate(..., scaler=1,
mean_scaler=0)`) on z-scored Heston data, so the CRPS is directly comparable in magnitude across the three
missing ratios. It rises monotonically with the held-out fraction (0.0717 → 0.0861 → 0.1656), exactly the
paper's ordering (harder imputation ⇒ higher CRPS), and stays well below the clinical numbers because a
smooth univariate diffusion target is far easier to reconstruct than 35-variable ICU records. The
**Ours — PhysioNet/PM2.5** column is the authors' released code run verbatim on the paper's own datasets;
full write-up (with per-trial σ verdicts, all four within ±1.6 σ of the paper mean):
[`../../../methods/CSDI/paper_reimplementation/`](../../../methods/CSDI/paper_reimplementation/).

**Paper reproduction on PhysioNet + PM2.5** (the split-out **Ours — <PaperDataset>** column, with RMSE/MAE
secondaries and single-trial verdicts):

| Dataset / setting | **Ours CRPS** | Paper CRPS (Table 2) | Ours RMSE | Ours MAE | Verdict |
|---|:---:|:---:|:---:|:---:|:---|
| Healthcare — 10% missing | **0.2344** | 0.238 (0.001) | 0.520 | 0.216 | matches ✓ (−1.6 σ) |
| Healthcare — 50% missing | **0.3364** | 0.330 (0.002) | 0.676 | 0.309 | matches ✓ (+1.4 σ) |
| Healthcare — 90% missing | **0.5256** | 0.522 (0.002) | 0.837 | 0.484 | matches ✓ (+0.8 σ) |
| Air quality (PM2.5) | **0.1064** | 0.108 (0.001) | 18.59 | 9.57 | matches ✓ (−0.7 σ) |

The paper's parenthesised value is the **standard error over 5 trials**; we ran one trial per setting, so
the honest comparison is our single draw against the paper's 5-trial mean in per-trial σ (SE·√5). All four
land inside ±1.6 per-trial σ — seed/sampling variance of one diffusion draw, not a hyperparameter error.

**Heston imputation-CRPS source.** The three Heston values come from committed JSONs produced by the
conditional-imputation driver on the benchmark Heston dataset (no hand-typing):
`methods/CSDI/paper_reimplementation/results/heston_impute_crps_r{0.1,0.5,0.9}.json` (each `nsample=100`,
`epochs=200`, seed 1), scored by the authors' `evaluate` in
`methods/CSDI/paper_reimplementation/heston_imputation/`.

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K nearest CSDI paths by L2 in z-scored space, forecast with their price-anchored futures.
Full analysis: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 2.718 ± 0.003646 | 3.738 |
| PS-MC CRPS H=64 ↓ | 3.776 ± 0.005153 | 5.246 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.718 < 3.738 at H=32; 3.776 < 5.246 at H=64), on all
5 seeds. CSDI's CRPS is the **second-best of any method benchmarked**, just behind LS4 (2.704 / 3.763) and
ahead of the rest of the field — its diffusion-generated pool provides one of the tightest, best-calibrated
nearest-neighbour shadowing sets. Heston is time-homogeneous, so the uniform and Gaussian shadow-weightings
coincide.

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
| `path_shadowing/` | Path-shadowing MC forecasts |

→ Cross-method comparison with TimeGAN, SBTS, Fourier Flow & Diffusion-TS: [`results/README.md`](../../README.md)
