# TimeVAE on Heston

PyTorch reimplementation of **TimeVAE** (Desai, Freeman, Beaver & Wang, 2021 —
*TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation*, arXiv:2111.08095v3)
trained on 8 192 Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for the source, the original paper/GitHub, the TimeVAE-Base
architecture (latent\_dim = 8, hidden = 50/100/200, `reconstruction_wt` = 3.0), and the per-`(t, feature)`
MinMax normalisation chain applied to fit the price-scale Heston data into the model's `[0, 1]` space.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

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

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (an independent Heston draw scored against the test set — finite-sample noise).
> **A12**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002). **A13**: path-level RMSE between real/gen mean trajectories. **A14**: KS statistic on pooled log-returns. **A15**: |skew_real − skew_gen|, Heston true skew ≈ −0.45. **A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS statistic on terminal prices S_T.
> **A18**: Discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). **A19**: TSTR predictive MAE; all methods cluster near 0.056–0.059 (irreducible log-return floor) (GRU + MLP).
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

> **Log-ret histogram**: MSE 968.0 is by far the weakest B panel — TimeVAE collapses the log-return density into a too-narrow central peak (A28 Kurtosis Ratio 0.28 ≪ 1.0, i.e. generated returns are ~3.6× more peaked than Heston), so the histogram bins mismatch strongly in absolute terms.
> **ACF \|r\|, ACF r²**: the MSE is small (0.034 / 0.027) because the true ACF ≈ 0.05 sits near zero, but the **% error** (function-level MAPE) blows up (984% / 1026%) for exactly that reason — near-zero denominators amplify any deviation. TimeVAE reproduces almost none of the ARCH autocorrelation (A21 0.39, A23 0.47 error vs Diffusion-TS 0.018 / 0.0024). Read MSE for absolute agreement, % error for relative shape.
> **Rolling vol histogram**: MSE 16019 — TimeVAE fails to reproduce the Heston rolling-volatility distribution (A31 rolling-vol KS 0.987, essentially disjoint supports).

---

## Stylised Facts Diagnostic (Heston vs TimeVAE, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/TimeVAE/plots/heston_diagnostics.png)

---

## TimeVAE Training Loss (5 seeds)

TimeVAE is a variational auto-encoder, so each epoch logs three additive components:
**Total = 3.0·Reconstruction + KL** (the `reconstruction_wt` = 3.0 preset from the paper),
plus the `ReduceLROnPlateau` learning rate. Training runs Adam(lr = 1e-3), batch 16, up to 1 000
epochs with `EarlyStopping(monitor = total_loss, min_delta = 1e-2, patience = 50)`. All 5 seeds
early-stop between **230 and 340 epochs**, with the total loss falling from ~450 to a plateau of
**~83** (per-seed final total: seed 0 83.19 @ 247 ep, seed 1 83.13 @ 230, seed 2 82.99 @ 340,
seed 3 82.81 @ 298, seed 4 82.76 @ 278). See [`code/README.md`](code/README.md) for the loss
definition and the MinMax normalisation chain.

![TimeVAE Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/TimeVAE/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/TimeVAE/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest TimeVAE paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to the other methods'.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/TimeVAE/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/TimeVAE/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 3.855 ± 0.070 | 3.856 ± 0.070 | 5.634 ± 0.124 | 5.634 ± 0.124 | 3.73 / 5.30 |
| MAE    | 4.414 ± 0.045 | 4.414 ± 0.045 | 6.549 ± 0.086 | 6.549 ± 0.086 | 3.73 / 5.30 |
| RMSE   | 6.064 ± 0.061 | 6.065 ± 0.061 | 8.947 ± 0.111 | 8.948 ± 0.111 | 5.07 / 7.18 |

PS-MC does **not** beat the naive RW on CRPS at either horizon (3.86 > 3.73 at H=32; 5.63 > 5.30 at
H=64). TimeVAE's prior-generated pool does not contain price-anchored futures that are close enough to
the real prefixes to form a well-calibrated nearest-neighbour ensemble — consistent with its weak
stylised-facts fit (A9 Volatility MMD 3.59, A31 Rolling-Vol KS 0.987). Uniform ≈ Gaussian: Heston is
time-homogeneous, so the K nearest neighbours are roughly equally predictive.

Full analysis: [`../../results/Heston/TimeVAE/path_shadowing/README.md`](../../results/Heston/TimeVAE/path_shadowing/README.md)

---

## File layout

```
methods/TimeVAE/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time, params
├── weights/
│   ├── seed_{i}_model.pt              VAE encoder/decoder state_dict
│   └── seed_{i}_config.json           full hyperparameters + MinMax constants
├── losses/
│   ├── seed_{i}_losses.csv            epoch, total_loss, reconstruction_loss, kl_loss, lr
│   └── loss_convergence.png           convergence plot (5 seeds, Total/Recon/KL/LR panels)
├── code/
│   ├── train_heston.py                Heston training driver (imports timevae_torch)
│   ├── timevae_torch.py               PyTorch TimeVAE-Base port
│   ├── plot_losses.py                 loss-convergence plot generator
│   ├── reference/                     verbatim released code (abudesai/timeVAE)
│   └── README.md                      paper, GitHub, architecture, fixes, hyperparameters
├── paper_reimplementation/            sine len-24 paper reproduction (disc/pred score)
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/TimeVAE/code
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 0

# Compute all metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method TimeVAE --dataset Heston

# Run Path Shadowing MC
cd methods/TimeVAE/path_shadowing
/home/tbasseras/gpu-venv/bin/python run_eval.py
```
