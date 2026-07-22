# Fourier Flow on Heston

PyTorch/NumPy reimplementation of **Fourier Flow** (Alaa, Chan, van der Schaar, ICLR 2021 —
*Generative Time-series Modeling with Fourier Flows*) trained on 8 192 Heston stochastic-volatility
price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, the original paper/GitHub, and the numerical guards
applied to make the frequency-domain flow finite on Heston data.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.5761 ± 0.008273 | 0.5672 | 0.5903 | 0.5732 | 0.5699 | 0.5798 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 7.21e-04 ± 2.10e-04 | 7.09e-04 | 9.82e-04 | 3.43e-04 | 7.93e-04 | 7.80e-04 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.002325 ± 5.06e-04 | 0.002328 | 0.003091 | 0.001496 | 0.002402 | 0.002309 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 7.42e-04 ± 1.38e-04 | 7.02e-04 | 9.80e-04 | 5.53e-04 | 7.55e-04 | 7.22e-04 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 5.802 ± 2.000 | 7.516 | 2.953 | 3.881 | 7.872 | 6.789 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.005527 ± 0.002289 | 0.003506 | 0.005485 | 0.009694 | 0.003346 | 0.005602 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.01105 ± 0.006414 | 0.007151 | 0.005616 | 0.02248 | 0.006248 | 0.01376 | 0.001983 |
| A8 Increment MMD² ↓ | 0.001124 ± 6.46e-05 | 0.001188 | 0.001160 | 0.001086 | 0.001015 | 0.001171 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.05871 ± 0.007003 | 0.05246 | 0.05759 | 0.07123 | 0.05196 | 0.06029 | 0.008554 |
| A10 Terminal SWD ↓ | 2.710 ± 1.034 | 1.918 | 2.075 | 4.630 | 1.962 | 2.967 | 1.151 |
| A11 Path SWD ↓ | 1.334 ± 0.3806 | 1.097 | 1.728 | 1.847 | 0.8888 | 1.107 | 0.6191 |
| A12 RV Law Loss ↓ | 0.5397 ± 0.1300 | 0.5313 | 0.7595 | 0.3519 | 0.5447 | 0.5112 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.4336 ± 0.3651 | 0.3360 | 1.049 | 0.6146 | 0.05405 | 0.1147 | 0.1205 |
| A14 KS Log-returns ↓ | 0.01895 ± 0.002028 | 0.01726 | 0.02286 | 0.01874 | 0.01756 | 0.01830 | 0.001491 |
| A15 Skewness Error ↓ | 0.02288 ± 0.01115 | 0.01933 | 0.01509 | 0.04385 | 0.01250 | 0.02363 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 5.81e-04 ± 4.14e-05 | 5.63e-04 | 6.62e-04 | 5.74e-04 | 5.50e-04 | 5.56e-04 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.08098 ± 0.01617 | 0.07373 | 0.09021 | 0.1046 | 0.05627 | 0.08008 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.009185 ± 0.009209 | 0.005951 | 0.02670 | 0.003204 | 0.009307 | 7.63e-04 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.005951 ± 0.002921 | 0.008087 | 4.58e-04 | 0.005951 | 0.008697 | 0.006561 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05004 ± 2.00e-05 | 0.05004 | 0.05002 | 0.05007 | 0.05001 | 0.05004 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05032 ± 3.48e-04 | 0.04992 | 0.05094 | 0.05040 | 0.05027 | 0.05009 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 60.80 ± 36.58 | 39.62 | 39.52 | 132.3 | 35.01 | 57.55 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.04095 ± 5.50e-04 | 0.04127 | 0.04149 | 0.03990 | 0.04096 | 0.04111 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.03498 ± 5.56e-04 | 0.03542 | 0.03546 | 0.03395 | 0.03486 | 0.03520 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.04897 ± 7.04e-04 | 0.04807 | 0.04982 | 0.04827 | 0.04965 | 0.04902 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.04195 ± 7.01e-04 | 0.04063 | 0.04261 | 0.04194 | 0.04243 | 0.04214 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.7990 ± 0.7970 | 0.5667 | 2.290 | 0.8638 | 0.04447 | 0.2298 | 0.1392 |
| A26 Return Std Error ↓ | 0.004832 ± 0.002757 | 0.005448 | 1.17e-04 | 0.008623 | 0.004291 | 0.005681 | 0.002523 |
| A27 Log-Return Std Error ↓ | 7.64e-05 ± 5.51e-05 | 1.00e-05 | 1.32e-04 | 1.50e-04 | 5.86e-05 | 3.18e-05 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 3.098 ± 0.7754 | 2.875 | 4.578 | 2.295 | 2.961 | 2.781 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.002245 ± 8.77e-04 | 0.002362 | 0.001509 | 0.003899 | 0.001631 | 0.001823 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.381 ± 0.4336 | 0.9429 | 1.876 | 1.935 | 1.012 | 1.138 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.07213 ± 0.001372 | 0.07279 | 0.07440 | 0.07190 | 0.07054 | 0.07103 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 6.89e-04 ± 9.20e-05 | 6.78e-04 | 8.41e-04 | 5.52e-04 | 6.99e-04 | 6.77e-04 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -0.002564 ± 0.002730 | -0.001968 | -0.003338 | -0.005218 | -0.004700 | 0.002406 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.08963 ± 0.001225 | 0.08969 | 0.08773 | 0.09160 | 0.08957 | 0.08954 | 0.06559 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (an independent Heston draw scored against the test set — finite-sample noise).
> **A12**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002). **A13**: path-level RMSE between real/gen mean trajectories. **A14**: KS statistic on pooled log-returns. **A15**: |skew_real − skew_gen|, Heston true skew ≈ −0.45. **A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS statistic on terminal prices S_T.
> **A18**: Discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). **A19**: TSTR predictive MAE; all methods cluster near 0.054–0.059 (irreducible log-return floor) (GRU + MLP).
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
| **Log-return histogram** | MSE | 0.9211 ± 0.02370 | 0.9371 | 0.9394 | 0.9444 | 0.8956 | 0.8891 | 0.1098 |
|  | % err | 9.167% ± 0.5606% | 8.987% | 10.28% | 8.876% | 8.912% | 8.782% | 1.799% |
|  | NRMSE | 4.186% ± 0.1102% | 4.142% | 4.342% | 4.272% | 4.027% | 4.147% | 0.5328% |
| **QQ plot** | MSE | 1.45e-07 ± 2.63e-08 | 1.37e-07 | 1.97e-07 | 1.25e-07 | 1.33e-07 | 1.32e-07 | 1.09e-09 |
|  | % err | 9.342% ± 2.293% | 7.914% | 13.86% | 8.822% | 7.678% | 8.438% | 0.4629% |
|  | NRMSE | 1.687% ± 0.1351% | 1.638% | 1.956% | 1.608% | 1.611% | 1.620% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 3.83e-04 ± 1.20e-05 | 3.85e-04 | 3.95e-04 | 3.61e-04 | 3.91e-04 | 3.81e-04 | 9.61e-06 |
|  | % err | 117.2% ± 2.149% | 119.0% | 119.3% | 113.5% | 116.0% | 118.1% | 8.724% |
|  | NRMSE | 88.45% ± 1.425% | 88.99% | 90.06% | 85.78% | 88.81% | 88.64% | 6.071% |
| **ACF r² lags 1–20** | MSE | 2.80e-04 ± 1.13e-05 | 2.80e-04 | 2.94e-04 | 2.61e-04 | 2.89e-04 | 2.78e-04 | 9.17e-06 |
|  | % err | 120.8% ± 3.065% | 123.1% | 124.1% | 115.4% | 119.8% | 121.8% | 11.34% |
|  | NRMSE | 82.92% ± 1.680% | 83.36% | 85.06% | 79.89% | 83.33% | 82.98% | 6.486% |
| **Rolling vol histogram** | MSE | 29.88 ± 2.639 | 29.80 | 34.64 | 26.52 | 29.36 | 29.07 | 1.372 |
|  | % err | 25.42% ± 3.199% | 25.19% | 30.77% | 20.71% | 25.49% | 24.94% | 2.264% |
|  | NRMSE | 10.43% ± 0.4823% | 10.43% | 11.31% | 9.838% | 10.29% | 10.29% | 0.8688% |
| **Tail survival** | MSE | 1.71e-04 ± 1.49e-05 | 1.69e-04 | 1.72e-04 | 1.98e-04 | 1.53e-04 | 1.65e-04 | 5.22e-07 |
|  | % err | 5.711% ± 0.2437% | 5.659% | 6.185% | 5.629% | 5.495% | 5.584% | 0.3302% |
|  | NRMSE | 2.287% ± 0.09795% | 2.272% | 2.295% | 2.461% | 2.163% | 2.244% | 0.1050% |

> **Log-ret histogram**: Fourier Flow is far more stable than TimeGAN here (MSE 0.921 ± 0.024 vs 45.40 ± 57.91) — explicit likelihood avoids the seed-to-seed collapse of adversarial training.
> **ACF \|r\|, ACF r²**: the MSE is tiny (~3.8e-4) because the true ACF ≈ 0.05 sits near zero, but the **% error** (function-level MAPE) is large (117% / 121%) for exactly that reason — near-zero denominators amplify any deviation. Read MSE for absolute agreement, % error for relative shape.
> **Rolling vol histogram**: MSE 29.88 from vol-distribution mismatch; still one of the weaker B panels alongside the log-return histogram.

---

## Stylised Facts Diagnostic (Heston vs Fourier Flow, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/FourierFlow/plots/heston_diagnostics.png)

---

## Fourier Flow Training Loss (5 seeds)

Negative-log-likelihood loss `(-log_pz - log_jacob).mean()` over 1000 full-batch epochs, all 5 seeds.
Gradient clipping at 1.0 keeps the loss finite on Heston (see `code/README.md`).

![Fourier Flow Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/FourierFlow/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/FourierFlow/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest Fourier Flow paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to TimeGAN's.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/FourierFlow/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/FourierFlow/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **2.742 ± 0.027** | 2.743 ± 0.027 | **3.992 ± 0.106** | 3.992 ± 0.106 | 3.73 / 5.30 |
| MAE    | 3.774 ± 0.028 | 3.774 ± 0.028 | 5.461 ± 0.120 | 5.461 ± 0.121 | 3.73 / 5.30 |
| RMSE   | 5.185 ± 0.055 | 5.185 ± 0.055 | 7.547 ± 0.197 | 7.548 ± 0.197 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.74 < 3.73 at H=32; 3.99 < 5.30 at H=64), on all
5 seeds, and its CRPS is lower than TimeGAN's (2.74 vs 3.09 at H=32) — Fourier Flow's pool gives tighter,
better-calibrated nearest-neighbour futures. Uniform ≈ Gaussian: Heston is time-homogeneous, so the K
nearest neighbours are roughly equally predictive.

Full analysis: [`../../results/Heston/FourierFlow/path_shadowing/README.md`](../../results/Heston/FourierFlow/path_shadowing/README.md)

---

## File layout

```
methods/FourierFlow/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time
├── weights/
│   ├── seed_{i}_model.pt              full PyTorch state_dict
│   └── seed_{i}_config.json           hyperparameters
├── losses/
│   ├── seed_{i}_losses.csv            epoch, loss (NLL per full-batch epoch)
│   └── loss_convergence.png           convergence plot (5 seeds overlaid)
├── code/
│   ├── train_heston.py                Heston training driver (imports reference FourierFlow) + guards
│   ├── train_all.sh                   orchestrator — 5 seeds (CPU, core-pinned)
│   ├── plot_losses.py                 loss_convergence.png generator
│   ├── reference/                     verbatim released code (ahmedmalaa/Fourier-flows)
│   └── README.md                      paper, GitHub, numerical guards
├── paper_reimplementation/            Stocks Table-2 reproduction (F-score + MAE)
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (CPU, numpy.fft — Fourier Flow has no GPU path)
cd methods/FourierFlow/code
./train_all.sh

# Compute metrics
cd metrics
python compute_all.py --method FourierFlow --dataset Heston
```
