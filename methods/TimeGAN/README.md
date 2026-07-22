# TimeGAN on Heston

PyTorch reimplementation of **TimeGAN** (Yoon et al., NeurIPS 2019) trained on 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, original paper, and the 5 fixes applied
to the TF1 reference implementation.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 2.954 ± 2.098 | 0.01519 | 5.359 | 3.767 | 0.9585 | 4.672 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.003196 ± 0.001907 | 0.004248 | 0.005644 | 0.001564 | 4.36e-04 | 0.004088 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.004342 ± 0.002767 | 0.007384 | 0.006912 | 0.005192 | 0.001759 | 4.64e-04 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.003401 ± 0.001522 | 0.004204 | 0.005410 | 0.001582 | 0.001658 | 0.004149 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 36.32 ± 17.05 | 40.13 | 18.22 | 51.19 | 14.93 | 57.13 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.01866 ± 0.01472 | 0.01025 | 0.003450 | 0.03561 | 0.006646 | 0.03735 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.03072 ± 0.02472 | 0.01905 | 0.005805 | 0.06974 | 0.009835 | 0.04915 | 0.001983 |
| A8 Increment MMD² ↓ | 0.008280 ± 0.004303 | 0.005390 | 0.007798 | 0.01429 | 0.002251 | 0.01167 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.3975 ± 0.2486 | 0.1864 | 0.3624 | 0.7126 | 0.07878 | 0.6473 | 0.008554 |
| A10 Terminal SWD ↓ | 2.917 ± 1.131 | 2.517 | 1.402 | 4.550 | 2.274 | 3.840 | 1.151 |
| A11 Path SWD ↓ | 1.678 ± 0.5770 | 1.689 | 0.7691 | 2.383 | 1.371 | 2.176 | 0.6191 |
| A12 RV Law Loss ↓ | 1.558 ± 0.3879 | 1.515 | 1.776 | 1.816 | 0.8208 | 1.863 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.5356 ± 0.2514 | 0.4458 | 0.3018 | 0.9468 | 0.6922 | 0.2913 | 0.1205 |
| A14 KS Log-returns ↓ | 0.08474 ± 0.03769 | 0.03948 | 0.06328 | 0.1259 | 0.06167 | 0.1334 | 0.001491 |
| A15 Skewness Error ↓ | 0.3412 ± 0.3279 | 0.006432 | 0.4384 | 0.09798 | 0.2341 | 0.9290 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.002506 ± 6.49e-04 | 0.001915 | 0.002621 | 0.002769 | 0.001705 | 0.003521 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.1109 ± 0.05875 | 0.1039 | 0.05347 | 0.2140 | 0.05603 | 0.1273 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.03305 ± 0.05328 | 0.003814 | 0.001984 | 0.01755 | 0.002899 | 0.1390 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.08792 ± 0.04703 | 0.1384 | 0.06210 | 0.008697 | 0.1088 | 0.1216 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05277 ± 0.001115 | 0.05154 | 0.05456 | 0.05353 | 0.05221 | 0.05199 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05322 ± 0.001031 | 0.05136 | 0.05452 | 0.05350 | 0.05318 | 0.05352 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 21.36 ± 9.068 | 14.84 | 24.77 | 8.799 | 35.38 | 22.99 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.1278 ± 0.06738 | 0.06730 | 0.1072 | 0.2036 | 0.05023 | 0.2106 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.08676 ± 0.03470 | 0.04795 | 0.08225 | 0.1202 | 0.05091 | 0.1325 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.2301 ± 0.1034 | 0.1574 | 0.2157 | 0.3706 | 0.08767 | 0.3192 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.1760 ± 0.06259 | 0.1218 | 0.2041 | 0.2675 | 0.09154 | 0.1949 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.7781 ± 0.3669 | 0.6340 | 0.5870 | 1.254 | 1.143 | 0.2722 | 0.1392 |
| A26 Return Std Error ↓ | 0.1525 ± 0.08911 | 0.1529 | 0.2389 | 0.03119 | 0.07782 | 0.2618 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.001703 ± 7.89e-04 | 0.002047 | 0.002359 | 5.93e-04 | 9.43e-04 | 0.002573 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | -1.116 ± 3.593 | 2.017 | 0.1387 | 0.2498 | -8.162 | 0.1752 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.03089 ± 0.009106 | 0.03044 | 0.03768 | 0.02770 | 0.01602 | 0.04260 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.4742 ± 0.2079 | 0.4317 | 0.6883 | 0.1711 | 0.7242 | 0.3557 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.2552 ± 0.1101 | 0.1895 | 0.2727 | 0.3634 | 0.07956 | 0.3709 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 8.96e-04 ± 8.69e-04 | 3.55e-04 | 2.76e-04 | 0.002530 | 2.62e-04 | 0.001060 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.002745 ± 0.01354 | -0.003191 | 0.009072 | -0.01798 | 0.002783 | 0.02304 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.1186 ± 0.01863 | 0.1022 | 0.1112 | 0.1488 | 0.1000 | 0.1310 | 0.06559 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (an independent Heston draw scored against the test set — finite-sample noise).
> **A12**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002). **A13**: path-level RMSE between real/gen mean trajectories. **A14**: KS statistic on pooled log-returns. **A15**: |skew_real − skew_gen|, Heston true skew ≈ −0.45. **A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS statistic on terminal prices S_T.
> **A18**: Discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). **A19**: TSTR predictive MAE; all methods cluster near 0.056–0.059 (irreducible log-return floor) (GRU + MLP).
> **A20**: covariance-matrix error (%). **A21–A22**: ACF error on |r| and r² across lags 1–20. ARCH signal: |r_t| has positive lag-1 ACF ~0.05 in Heston. **A23–A24**: ACF lag-1 error on |r| and r². Heston true values ≈ +0.052 / +0.050.
> **A25**: mean-path RMSE. **A26**: return std error, uses price increments $\Delta S_t$. **A27**: log-return std error, uses $r_t = \log(S_{t+1}/S_t)$. **A28**: kurtosis ratio real/gen, perfect = 1.0; negative means gen has negative excess kurtosis. **A29**: sigma mean error — annualized per-path vol. **A30**: cross-sectional vol-path RMSE. **A31**: KS statistic on rolling-5 vol histograms. **A32**: |vol-of-vol_real − vol-of-vol_gen|.
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
| **Log-return histogram** | MSE | 45.40 ± 57.91 | 3.579 | 6.973 | 153.7 | 4.778 | 57.94 | 0.1098 |
|  | % err | 33.41% ± 6.533% | 26.37% | 36.99% | 32.64% | 27.13% | 43.94% | 1.799% |
|  | NRMSE | 21.38% ± 14.34% | 8.194% | 12.08% | 44.34% | 10.17% | 32.12% | 0.5328% |
| **QQ plot** | MSE | 2.38e-06 ± 1.14e-06 | 1.41e-06 | 2.44e-06 | 2.71e-06 | 1.05e-06 | 4.29e-06 | 1.09e-09 |
|  | % err | 34.50% ± 11.22% | 20.51% | 23.70% | 47.99% | 34.21% | 46.12% | 0.4629% |
|  | NRMSE | 6.960% ± 1.738% | 5.488% | 7.318% | 7.627% | 4.709% | 9.660% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.003597 ± 0.003199 | 0.001026 | 0.001841 | 0.006211 | 3.79e-04 | 0.008526 | 9.61e-06 |
|  | % err | 186.2% ± 107.8% | 107.0% | 128.7% | 264.2% | 72.79% | 358.2% | 8.724% |
|  | NRMSE | 224.6% ± 123.4% | 116.9% | 181.2% | 346.7% | 85.32% | 392.9% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.001982 ± 0.001602 | 7.02e-04 | 0.001482 | 0.002501 | 3.90e-04 | 0.004835 | 9.17e-06 |
|  | % err | 130.0% ± 65.84% | 95.37% | 88.69% | 113.6% | 91.94% | 260.6% | 11.34% |
|  | NRMSE | 168.2% ± 70.21% | 94.68% | 157.1% | 224.2% | 94.36% | 270.8% | 6.486% |
| **Rolling vol histogram** | MSE | 150.2 ± 75.22 | 96.14 | 215.1 | 207.3 | 27.97 | 204.5 | 1.372 |
|  | % err | 56.76% ± 21.18% | 53.37% | 72.71% | 84.37% | 22.36% | 51.01% | 2.264% |
|  | NRMSE | 22.64% ± 7.203% | 19.01% | 28.61% | 27.91% | 10.07% | 27.61% | 0.8688% |
| **Tail survival** | MSE | 0.003912 ± 0.003064 | 9.04e-04 | 0.002042 | 0.007307 | 0.001354 | 0.007952 | 5.22e-07 |
|  | % err | 23.64% ± 6.097% | 18.46% | 24.62% | 25.99% | 15.89% | 33.26% | 0.3302% |
|  | NRMSE | 10.02% ± 4.365% | 5.257% | 7.902% | 14.94% | 6.434% | 15.59% | 0.1050% |

> **Log-ret histogram**: MSE std (57.91) approaches the mean (45.40), driven by seed 2 near-collapse (153.7 vs 3.6–58 for the other seeds). A std near the mean is expected for a non-negative combined score over 5 skewed samples — it is not a bug. SBTS is far more stable here (4.082 ± 0.048).
> **ACF \|r\|, ACF r²**: TimeGAN misses the ARCH signature — near-zero generated ACF vs Heston ≈ +0.05 (A21 error 0.128, A23 error 0.230) — so both MSE and % error are larger than SBTS.
> **Rolling vol histogram**: High MSE (150.2) from vol-distribution mismatch across most seeds.
> **% error reading**: the **function-level MAPE** of the curve itself (mean |gen−real|/(|real|+1e-6) × 100); the derivative / 2nd-derivative MAPE is excluded as ill-posed (near-zero true differences blow up the denominator). ACF |r| (186%) and ACF r² (130%) are the largest — the true ACF ≈ 0.05 sits near zero, so any deviation is a big *relative* error; log-ret hist (33%), QQ (35%), rolling vol (57%) and tail survival (24%) are all modest. Read MSE for absolute agreement, % error for relative shape deviation.

---

## Stylised Facts Diagnostic (Heston vs TimeGAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/TimeGAN/plots/heston_diagnostics.png)

---

## TimeGAN Training Loss (5 seeds)

Loss curves across all three training phases (embedding 0–5k, supervisor 5–10k, joint 10–20k).

![TimeGAN Training Loss](losses/loss_convergence.png)

---

## A13 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/TimeGAN/plots/disc_classifier_loss.png)

---

## A14 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/TimeGAN/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest TimeGAN paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data).

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/TimeGAN/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/TimeGAN/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

Embedding: **65D murex-style prefix features** — 63 log-returns + 1 terminal return + 1 realized vol,
z-scored per dimension using the generated pool. Adaptive Gaussian bandwidth: η = η̃·‖z(x̃)‖, η̃ = median(dist)/median(‖z‖).

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **3.087 ± 0.340** | 3.087 ± 0.341 | **4.372 ± 0.431** | 4.373 ± 0.432 | 3.73 / 5.30 |
| MAE    | 4.039 ± 0.228 | 4.039 ± 0.229 | 5.680 ± 0.178 | 5.681 ± 0.179 | 3.73 / 5.30 |
| RMSE   | 5.452 ± 0.293 | 5.452 ± 0.293 | 7.667 ± 0.203 | 7.668 ± 0.203 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (3.09 < 3.73 at H=32; 4.37 < 5.30 at H=64).
Uniform ≈ Gaussian: Heston is time-homogeneous, so K nearest neighbours are roughly equally predictive.
The 65D embedding selects paths by full trajectory shape (vs 22D eq.13 which captures only market regime).

Full analysis: [`results/Heston/TimeGAN/path_shadowing/README.md`](../../results/Heston/TimeGAN/path_shadowing/README.md)

---

## File layout

```
methods/TimeGAN/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time
├── weights/
│   ├── seed_{i}_model.pt              full PyTorch state_dict
│   └── seed_{i}_config.json           hyperparameters
├── losses/
│   ├── seed_{i}_losses.csv            step, phase, e_loss, s_loss, g_loss, d_loss
│   └── loss_convergence.png           convergence plot (5 seeds overlaid)
└── code/
    ├── timegan_torch.py               PyTorch TimeGAN implementation
    ├── train.py                       orchestrator — 5 seeds on 2 GPUs in pairs
    ├── train_seed.py                  single-seed worker
    ├── reference/                     verbatim TF1 code (jsyoon0823/TimeGAN)
    └── README.md                      paper, GitHub, list of 5 fixes vs TF1
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/TimeGAN/code
python train.py --gpu0 0 --gpu1 3

# Compute metrics
cd metrics
python compute_all.py --method TimeGAN --dataset Heston
```
