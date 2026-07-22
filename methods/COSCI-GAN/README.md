# COSCI-GAN on Heston

PyTorch reimplementation of **COSCI-GAN** (Seyfi, Rajotte & Ng, NeurIPS 2022 —
*Generating multivariate time series with COmmon Source CoordInated GAN*)
trained on 8 192 Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for the source, the original paper/GitHub, the 3-player
architecture (per-channel LSTM Generator + LSTM Discriminator, one MLP Central Discriminator),
the hyperparameters (`gamma` = 5, `noise_len` = 32, `hidden_dim` = 256, Adam betas (0.5, 0.9)),
and the scalar MinMax normalisation chain applied to fit the price-scale Heston data into the
model's `[0, 1]` space.

> **⚠️ C = 1 degeneracy (documented honestly).** COSCI-GAN's contribution is *cross-channel*
> coordination: C univariate "Channel GANs" driven by a **shared noise vector**, coupled by one
> **Central Discriminator** (CD) that sees the concatenation of all channels. Heston here is
> **price-only, so C = 1**. With a single channel the CD receives the *same* 128-dim vector as the
> single channel discriminator — it becomes a redundant second critic with nothing cross-channel to
> coordinate. The healthy-equilibrium signature is therefore **`loss_CD ≈ ln 2 ≈ 0.693` (CD stuck at
> chance)**, which is exactly what we observe (see Training Loss below). The paper's own metric
> (Table 4 cross-channel correlation-MAE) is **structurally undefined at C = 1** and is reproduced
> separately on the multi-channel EEG dataset in
> [`paper_reimplementation/`](paper_reimplementation/README.md) (ours 0.1085 ± 0.0066 vs paper
> 0.111 ± 0.005). The Heston numbers below still exercise the full 3-player training loop; they just
> cannot reward the CD, so read them as "single-channel GAN with a spectating CD".

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.5615 ± 0.1128 | 0.5689 | 0.6521 | 0.4439 | 0.4277 | 0.7150 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.09711 ± 0.003466 | 0.1026 | 0.09618 | 0.09329 | 0.09939 | 0.09408 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.1240 ± 0.005959 | 0.1261 | 0.1213 | 0.1201 | 0.1346 | 0.1178 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.09566 ± 0.003535 | 0.1011 | 0.09464 | 0.09185 | 0.09820 | 0.09244 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 1.614 ± 1.128 | 1.057 | 3.477 | 2.314 | 0.8120 | 0.4107 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.04686 ± 0.004162 | 0.05151 | 0.04119 | 0.05157 | 0.04643 | 0.04361 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.01623 ± 0.01333 | 0.006198 | 0.002270 | 0.009863 | 0.02459 | 0.03820 | 0.001983 |
| A8 Increment MMD² ↓ | 0.4788 ± 0.01185 | 0.4989 | 0.4774 | 0.4617 | 0.4786 | 0.4775 | 8.69e-04 |
| A9 Volatility MMD ↓ | 3.955 ± 0.04883 | 3.995 | 3.988 | 3.875 | 3.997 | 3.923 | 0.008554 |
| A10 Terminal SWD ↓ | 4.756 ± 3.118 | 3.349 | 0.8315 | 2.764 | 8.893 | 7.942 | 1.151 |
| A11 Path SWD ↓ | 3.505 ± 0.1711 | 3.739 | 3.288 | 3.672 | 3.405 | 3.422 | 0.6191 |
| A12 RV Law Loss ↓ | 118.7 ± 7.929 | 129.8 | 118.3 | 107.8 | 124.9 | 112.9 | 0.05202 |
| A13 Mean Path RMSE ↓ | 3.995 ± 0.1803 | 4.250 | 3.744 | 4.145 | 3.926 | 3.909 | 0.1205 |
| A14 KS Log-returns ↓ | 0.3206 ± 0.007269 | 0.3155 | 0.3234 | 0.3121 | 0.3331 | 0.3190 | 0.001491 |
| A15 Skewness Error ↓ | 0.04981 ± 0.04124 | 0.1217 | 0.02345 | 0.07044 | 0.01282 | 0.02066 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.04857 ± 0.001967 | 0.05140 | 0.04864 | 0.04575 | 0.04982 | 0.04722 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.1473 ± 0.09804 | 0.1035 | 0.02246 | 0.08972 | 0.2312 | 0.2894 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.4999 ± 1.22e-04 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4997 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.5000 ± 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.1331 ± 0.01808 | 0.1455 | 0.09753 | 0.1417 | 0.1359 | 0.1446 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.09591 ± 0.006992 | 0.1043 | 0.1019 | 0.08746 | 0.09785 | 0.08800 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 30.59 ± 29.16 | 4.828 | 1.233 | 73.77 | 16.66 | 56.45 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.08056 ± 0.02054 | 0.07869 | 0.1098 | 0.06495 | 0.09630 | 0.05303 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.09004 ± 0.02156 | 0.08482 | 0.1217 | 0.07282 | 0.1073 | 0.06357 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1700 ± 0.04930 | 0.1795 | 0.1992 | 0.1451 | 0.2355 | 0.09066 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.1957 ± 0.05105 | 0.2150 | 0.2368 | 0.1714 | 0.2473 | 0.1079 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 4.539 ± 3.359 | 3.477 | 0.9836 | 1.212 | 8.430 | 8.593 | 0.1392 |
| A26 Return Std Error ↓ | 5.032 ± 0.2229 | 5.299 | 5.082 | 4.683 | 5.211 | 4.887 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.04975 ± 0.002001 | 0.05250 | 0.04968 | 0.04694 | 0.05132 | 0.04829 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | -8.150 ± 12.11 | -19.44 | -9.203 | 6.342 | 4.788 | -23.24 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.7871 ± 0.03094 | 0.8305 | 0.7877 | 0.7436 | 0.8094 | 0.7644 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.155 ± 0.3231 | 1.097 | 0.8276 | 1.757 | 1.159 | 0.9320 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.9371 ± 0.007667 | 0.9435 | 0.9452 | 0.9234 | 0.9370 | 0.9363 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.01806 ± 0.001147 | 0.01915 | 0.01685 | 0.01809 | 0.01950 | 0.01670 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -0.005511 ± 0.008042 | -0.02120 | -0.003004 | 3.30e-04 | 3.78e-04 | -0.004059 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.8087 ± 0.02874 | 0.8544 | 0.8096 | 0.7713 | 0.8213 | 0.7868 | 0.06559 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (an independent Heston draw scored against the test set — finite-sample noise).
> **A12**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002). **A13**: path-level RMSE between real/gen mean trajectories. **A14**: KS statistic on pooled log-returns. **A15**: |skew_real − skew_gen|, Heston true skew ≈ −0.45. **A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS statistic on terminal prices S_T.
> **A18**: Discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable, 0.5 = perfectly separable (GRU + MLP). **A19**: TSTR predictive MAE (GRU + MLP).
> **A20**: covariance-matrix error (%). **A21–A22**: ACF error on |r| and r² across lags 1–20. ARCH signal: |r_t| has positive lag-1 ACF ~0.05 in Heston. **A23–A24**: ACF lag-1 error on |r| and r². Heston true values ≈ +0.052 / +0.050.
> **A25**: mean-path RMSE. **A26**: return std error, uses price increments $\Delta S_t$. **A27**: log-return std error, uses $r_t = \log(S_{t+1}/S_t)$. **A28**: kurtosis ratio real/gen, perfect = 1.0. **A29**: sigma mean error — annualized per-path vol. **A30**: cross-sectional vol-path RMSE. **A31**: KS statistic on rolling-5 vol histograms. **A32**: |vol-of-vol_real − vol-of-vol_gen|.
> **A33**: Teacher-sigma correlation (Heston-recovered vol vs teacher σ), higher is better, perfect ≈ 0.614. **A34**: Teacher-sigma RMSE, perfect ≈ 0.065.

---

## B — Curve-Shape Metrics — mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. For the real
data (L_r) and generated data (L_g) we build three lists — the curve L, its first finite
difference L' (der), and its second finite difference L'' (sec\_der) — then combine the three
sub-scores into **one number per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\_der)/3; std = the sample std of that per-seed combined score across the 5 seeds. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE — one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the 5 seeds** of that per-seed function MAPE.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)** — the ill-posed derivative / 2nd-derivative curves are excluded for the same reason as the % err row.

All ↓ lower is better. The perfect floor is **non-zero** for all six plots — it is the residual finite-sample error of an independent Heston draw scored against the test set, identical across methods.
Three sublines per plot: **MSE**, **% error** and **NRMSE** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 42.66 ± 1.999 | 41.16 | 40.58 | 41.39 | 45.11 | 45.07 | 0.1098 |
|  | % err | 246.6% ± 7.987% | 239.4% | 240.4% | 240.3% | 257.2% | 255.4% | 1.799% |
|  | NRMSE | 30.81% ± 0.7154% | 30.27% | 30.07% | 30.36% | 31.68% | 31.68% | 0.5328% |
| **QQ plot** | MSE | 8.25e-04 ± 6.60e-05 | 9.18e-04 | 8.24e-04 | 7.33e-04 | 8.74e-04 | 7.76e-04 | 1.09e-09 |
|  | % err | 437.1% ± 19.17% | 447.9% | 436.8% | 401.3% | 457.4% | 442.2% | 0.4629% |
|  | NRMSE | 134.7% ± 5.407% | 142.3% | 134.8% | 127.1% | 138.7% | 130.8% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.008548 ± 0.003519 | 0.008189 | 0.01521 | 0.007491 | 0.007038 | 0.004809 | 9.61e-06 |
|  | % err | 230.0% ± 48.05% | 212.8% | 322.3% | 198.8% | 227.8% | 188.2% | 8.724% |
|  | NRMSE | 198.2% ± 35.47% | 173.9% | 253.7% | 167.7% | 227.0% | 168.8% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.008781 ± 0.003516 | 0.006998 | 0.01505 | 0.009647 | 0.007563 | 0.004648 | 9.17e-06 |
|  | % err | 287.8% ± 57.85% | 233.8% | 398.9% | 271.6% | 253.8% | 281.0% | 11.34% |
|  | NRMSE | 221.1% ± 36.09% | 194.4% | 282.2% | 196.8% | 242.8% | 189.4% | 6.486% |
| **Rolling vol histogram** | MSE | 1398 ± 34.29 | 1390 | 1444 | 1360 | 1365 | 1431 | 1.372 |
|  | % err | 799.2% ± 14.12% | 796.1% | 817.9% | 781.4% | 787.6% | 812.9% | 2.264% |
|  | NRMSE | 73.06% ± 0.8956% | 72.83% | 74.25% | 72.08% | 72.17% | 73.95% | 0.8688% |
| **Tail survival** | MSE | 0.05973 ± 0.001991 | 0.05945 | 0.05962 | 0.05642 | 0.06257 | 0.06057 | 5.22e-07 |
|  | % err | 342.3% ± 8.331% | 343.2% | 344.5% | 327.4% | 353.2% | 343.2% | 0.3302% |
|  | NRMSE | 42.74% ± 0.7148% | 42.64% | 42.71% | 41.54% | 43.75% | 43.04% | 0.1050% |

> **Log-ret histogram**: MSE 42.66 — comparable to TimeGAN (45.40) and far better than TimeVAE (968), but well behind Fourier Flow (0.92), LS4 (0.45) and the diffusion/bridge cluster (SBTS 4.08, CSDI 4.64, Diffusion-TS 4.88, TimeVQVAE 4.39). The histogram shape is only mid-pack: the central mass is roughly right but the **tails are badly mismatched** (A28 Kurtosis Ratio −8.15, wrong-signed excess kurtosis), which the curve metric penalises.
> **ACF \|r\|, ACF r²**: MSE small (0.0085 / 0.0088) because the true ACF ≈ 0.05 sits near zero; the **% error** (function-level MAPE) is 230% / 288% for that same near-zero-denominator reason. COSCI-GAN reproduces only part of the ARCH autocorrelation (A21 0.081, A23 0.170 error) — better than TimeVAE (0.39 / 0.47) but far from Diffusion-TS (0.018 / 0.0024). Read MSE for absolute agreement, % error for relative shape.
> **Rolling vol histogram**: MSE 1398 — like every VAE/GAN baseline here, COSCI-GAN fails to reproduce the Heston rolling-volatility distribution (A31 rolling-vol KS 0.937, near-disjoint supports).

---

## Reading the table — honest mixed result

COSCI-GAN has the **best scalar low-order moments** of the VAE/GAN family here, but its **full-density
curves rank near the bottom of the benchmark** and it suffers a **hard adversarial failure**:

- **Good — scalar moments.** A1 Kurtosis Error **0.56** (vs TimeVAE 2.26, TimeGAN 2.95) and A15
  Skewness Error **0.050** (vs TimeVAE 0.55) are the best of the VAE/GAN family, and **A5 Hill tail-index
  error 1.61** is a strong third (behind only LS4 1.23 and CSDI 1.43). But these are *scalar*
  summaries — they do **not** carry over to the full-density *curve* diagnostics (B section above),
  where COSCI-GAN ranks **near the bottom (7th–9th of 9) on every plot** (dead-last on QQ, log-ret-histogram MSE 42.66
  behind all but TimeGAN and TimeVAE). Good moments, weak curves.
- **Decent — ARCH autocorrelation.** A21 0.081 / A23 0.170 beat TimeVAE (0.39 / 0.47); COSCI-GAN
  captures *some* volatility clustering, though not at Diffusion-TS level.
- **Hard fail — A18 = 0.50 (both GRU and MLP).** The discriminative score is at its **maximum**
  (`score = |accuracy − 0.5|`, so 0.50 = the classifier separates real from fake with ~100%
  accuracy). This is a **genuine result, not a metric bug**: the classifier's hidden dim is floored
  to `max(8, n_features·8) = 8` (never the degenerate 0 that crashed on 1-D data elsewhere), and its
  BCE loss collapses to ~1e-6 (MLP) / ~1e-3 (GRU) during training (see A18 loss plot), whereas the
  same pipeline leaves TimeVAE at ~0.69 (chance). COSCI-GAN paths carry a fingerprint the classifier
  learns almost perfectly.
- **Fat tails too thin — A28 = −8.15 (sign-flipping across seeds).** The generated log-returns have
  near-zero-to-slightly-negative *excess* kurtosis (Gaussian-ish tails) against Heston's mildly
  fat tails, so the ratio κ_real/κ_gen straddles zero (seeds range −23.2 … +6.3). The marginal
  *centre* is right (A1 good) but the *tail* is not.
- **No latent-vol recovery — A33 ≈ −0.006 (perfect 0.616), A34 0.81.** As with every single-factor
  generator here, the recovered instantaneous vol is uncorrelated with the teacher σ.

Net: a good density-matcher that a modern sequence classifier nonetheless nails, with thin tails and
no stochastic-vol structure.

---

## Stylised Facts Diagnostic (Heston vs COSCI-GAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/COSCI-GAN/plots/heston_diagnostics.png)

---

## COSCI-GAN Training Loss (5 seeds)

COSCI-GAN is a **3-player** game, so each epoch logs three losses (`epoch, loss_D_0, loss_G_0,
loss_CD`):

- **`loss_D_0`** — channel-0 discriminator BCE, hovers at **~0.69–0.75** (critic near chance = the
  GAN equilibrium).
- **`loss_G_0`** — channel-0 generator loss = local adversarial − γ·loss_CD (γ = 5), sits around
  **−2.8**, dominated by the −γ·loss_CD coupling term.
- **`loss_CD`** — central discriminator BCE, pinned at **ln 2 ≈ 0.693** for the entire run — the
  **C = 1 degeneracy signature**: with one channel the CD has nothing cross-channel to coordinate
  and stays exactly at chance.

Training runs Adam(betas (0.5, 0.9)), batch 32, 120 epochs; generator/discriminator LR 1e-3, CD LR
1e-4 (10× smaller so the CD cannot overpower the channel GAN). All 5 seeds converge to this
equilibrium. See [`code/README.md`](code/README.md) for the loss definitions and the MinMax
normalisation chain.

![COSCI-GAN Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake; here the loss
**collapses toward zero** (MLP → ~1e-6, GRU → ~1e-3), the direct cause of the A18 = 0.50 score —
COSCI-GAN paths are near-perfectly separable from real Heston paths.

![Discriminative Classifier Loss](../../results/Heston/COSCI-GAN/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/COSCI-GAN/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest COSCI-GAN paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to the other methods'.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/COSCI-GAN/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/COSCI-GAN/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 4.657 ± 0.775 | 4.656 ± 0.773 | 5.834 ± 0.763 | 5.834 ± 0.764 | 3.73 / 5.30 |
| MAE    | 6.030 ± 0.891 | 6.027 ± 0.888 | 7.674 ± 0.866 | 7.673 ± 0.866 | 3.73 / 5.30 |
| RMSE   | 7.613 ± 0.940 | 7.610 ± 0.938 | 9.782 ± 0.944 | 9.780 ± 0.944 | 5.07 / 7.18 |

PS-MC does **not** beat the naive RW on CRPS at either horizon (4.66 > 3.73 at H=32; 5.83 > 5.30 at
H=64); only **1 of 5 seeds** (seed 2, CRPS 3.54) edges below the RW at H=32. COSCI-GAN's
prior-generated pool does not contain price-anchored futures close enough to the real prefixes to
form a well-calibrated nearest-neighbour ensemble — consistent with its A18 = 0.50 separability and
weak stylised-facts fit (A9 Volatility MMD 3.96, A31 Rolling-Vol KS 0.937). Uniform ≈ Gaussian:
Heston is time-homogeneous, so the K nearest neighbours are roughly equally predictive.

Full analysis: [`../../results/Heston/COSCI-GAN/path_shadowing/README.md`](../../results/Heston/COSCI-GAN/path_shadowing/README.md)

---

## File layout

```
methods/COSCI-GAN/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time, params (799 618)
├── weights/
│   ├── seed_{i}_model.pt              per-channel G/D + CD state_dict
│   └── seed_{i}_config.json           full hyperparameters + MinMax constants
├── losses/
│   ├── seed_{i}_losses.csv            epoch, loss_D_0, loss_G_0, loss_CD
│   └── loss_convergence.png           convergence plot (5 seeds, 3 loss panels + CD-vs-ln2 overlay)
├── code/
│   ├── train_heston.py                Heston training driver (3-player COSCI-GAN, C=1)
│   ├── plot_losses.py                 loss-convergence plot generator
│   ├── reference/                     verbatim released code (aliseyfi75/COSCI-GAN)
│   └── README.md                      paper, GitHub, architecture, C=1 note, hyperparameters
├── paper_reimplementation/            EEG eye-state Table-4 correlation-MAE reproduction
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/COSCI-GAN/code
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 0

# Compute all metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method COSCI-GAN --dataset Heston

# Run Path Shadowing MC
cd methods/COSCI-GAN/path_shadowing
/home/tbasseras/gpu-venv/bin/python run_eval.py
```
