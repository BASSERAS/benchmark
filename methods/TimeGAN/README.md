# TimeGAN on Heston

PyTorch reimplementation of **TimeGAN** (Yoon et al., NeurIPS 2019) trained on 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, original paper, and the 5 fixes applied
to the TF1 reference implementation.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A1 | Kurtosis Error | Fat Tail | ↓ | 2.955 ± 2.099 | 0.0148 | 5.360 | 3.768 | 0.9581 | 4.672 | 0 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | 0.0032 ± 0.0018 | 0.0042 | 0.0056 | 0.0016 | 5.06e-04 | 0.0040 | 0 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | 0.0043 ± 0.0028 | 0.0074 | 0.0069 | 0.0052 | 0.0017 | 4.49e-04 | 0 |
| A4 | Tail QQ Error | Fat Tail | ↓ | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0 |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | 36.885 ± 17.053 | 40.695 | 18.783 | 51.751 | 15.495 | 57.699 | 0 |
| | **— Distribution —** | | | | | | | | | |
| A6 | Path MMD² | Distribution | ↓ | 0.0165 ± 0.0127 | 0.0103 | 0.0048 | 0.0274 | 0.0042 | 0.0355 | 0.0015 |
| A7 | Terminal MMD² | Distribution | ↓ | 0.0267 ± 0.0192 | 0.0168 | 0.0105 | 0.0537 | 0.0066 | 0.0455 | 0.0016 |
| A8 | Increment MMD² | Distribution | ↓ | 0.0077 ± 0.0041 | 0.0048 | 0.0072 | 0.0114 | 0.0022 | 0.0131 | 7.45e-04 |
| A9 | Volatility MMD | Distribution | ↓ | 0.3789 ± 0.2430 | 0.1645 | 0.3504 | 0.6260 | 0.0708 | 0.6828 | 0.0071 |
| A10 | Terminal SWD | Distribution | ↓ | 2.658 ± 0.8567 | 2.303 | 2.004 | 3.774 | 1.638 | 3.570 | 0.6873 |
| A11 | Path SWD | Distribution | ↓ | 1.417 ± 0.4914 | 1.406 | 0.9015 | 2.039 | 0.8440 | 1.893 | 0.4381 |
| A12 | RV Law Loss | Distribution | ↓ | 1.551 ± 0.3788 | 1.491 | 1.754 | 1.827 | 0.8373 | 1.847 | 0 |
| A13 | Mean Path RMSE | Distribution | ↓ | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0 |
| A14 | KS Log-returns | Distribution | ↓ | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0 |
| A15 | Skewness Error | Distribution | ↓ | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0 |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0025 ± 6.43e-04 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0 |
| A17 | Terminal Price KS | Distribution | ↓ | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0 |
| | **— Adversarial —** | | | | | | | | | |
| A18 GRU | Discriminative Score GRU | Adversarial | ↓ | 0.0077 ± 0.0050 | 0.0081 | 7.63e-04 | 0.0038 | 0.0148 | 0.0111 | 0.0071 |
| A18 MLP | Discriminative Score MLP | Adversarial | ↓ | 0.1031 ± 0.0395 | 0.1366 | 0.0346 | 0.0822 | 0.1326 | 0.1295 | 0.0033 |
| | **— Predictive —** | | | | | | | | | |
| A19 GRU | Predictive Score GRU | Predictive | ↓ | 0.0574 ± 0.0019 | 0.0553 | 0.0600 | 0.0592 | 0.0560 | 0.0564 | 0.0537 |
| A19 MLP | Predictive Score MLP | Predictive | ↓ | 0.0570 ± 0.0012 | 0.0553 | 0.0586 | 0.0571 | 0.0560 | 0.0578 | 0.0537 |
| | **— Temporal —** | | | | | | | | | |
| A20 | Covariance Error | Temporal | ↓ | 17.751 ± 6.707 | 8.830 | 18.765 | 14.807 | 29.373 | 16.980 | 0 |
| A21 | ACF \|r\| Error (lags) | Temporal | ↓ | 0.1252 ± 0.0674 | 0.0648 | 0.1046 | 0.2011 | 0.0477 | 0.2080 | 0 |
| A22 | ACF r² Error (lags) | Temporal | ↓ | 0.0839 ± 0.0348 | 0.0450 | 0.0793 | 0.1172 | 0.0479 | 0.1300 | 0 |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0.2264 ± 0.1034 | 0.1537 | 0.2120 | 0.3669 | 0.0840 | 0.3155 | 0 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | 0.1719 ± 0.0626 | 0.1177 | 0.2000 | 0.2634 | 0.0874 | 0.1908 | 0 |
| | **— Vol —** | | | | | | | | | |
| A25 | Mean RMSE | Vol | ↓ | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.056 | 1.341 | 0.0743 | 0 |
| A26 | Return Std Error | Vol | ↓ | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 | 0 |
| A27 | Log-Return Std Error | Vol | ↓ | 0.0017 ± 7.78e-04 | 0.0020 | 0.0023 | 5.63e-04 | 9.72e-04 | 0.0025 | 0 |
| A28 | Kurtosis Ratio | Vol | — | -1.095 ± 3.525 | 1.979 | 0.1360 | 0.2451 | -8.005 | 0.1718 | 1.000 |
| A29 | Sigma Mean Error | Vol | ↓ | 0.0307 ± 0.0089 | 0.0301 | 0.0373 | 0.0273 | 0.0164 | 0.0422 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0 |
| A31 | Rolling Vol KS (w=5) | Vol | ↓ | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 8.97e-04 ± 8.69e-04 | 3.54e-04 | 2.75e-04 | 0.0025 | 2.63e-04 | 0.0011 | 0 |
| | **— Heston Spec —** | | | | | | | | | |
| A33 | Teacher-Sigma Corr | Heston Spec | ↑ | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | -0.0082 | -0.0057 | 0.0166 | 0.6143 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | 0.1183 ± 0.0184 | 0.1016 | 0.1108 | 0.1479 | 0.1004 | 0.1308 | 0.0654 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (row-shuffle keeps joint path structure).
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

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = m_funct + m_der + m_sec\_der (**sum** of the three seed-means); std = sqrt(s_funct² + s_der² + s_sec\_der²) (**quadrature**).
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE — one division (the mean already averages over the curve's points). Reported mean = **mean** of the three sub-scores; std = **sample std across the 5 seeds** (not quadrature).

All ↓ lower is better. Perfect floor = 0 for all six plots (row-shuffled real data has identical curves).
Two sublines per plot: **MSE** and **% error** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|------|---------|-----------|--------|--------|--------|--------|--------|:------:|
| **Log-return histogram** | MSE | 144.21 ± 120.61 | 11.054 | 20.679 | 504.48 | 14.578 | 170.28 | 0 |
| | % err | 25147% ± 8785% | 27938% | 12081% | 37201% | 29756% | 18757% | 0 |
| **QQ plot** | MSE | 7.09e-06 ± 3.34e-06 | 4.16e-06 | 7.22e-06 | 8.07e-06 | 3.19e-06 | 1.28e-05 | 0 |
| | % err | 52.65% ± 13.63% | 41.38% | 40.64% | 75.77% | 44.86% | 60.57% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.0105 ± 0.0085 | 0.0029 | 0.0053 | 0.0182 | 0.0010 | 0.0250 | 0 |
| | % err | 780.7% ± 602.8% | 949.7% | 324.3% | 537.1% | 215.8% | 1877% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0058 ± 0.0033 | 0.0020 | 0.0043 | 0.0072 | 0.0010 | 0.0143 | 0 |
| | % err | 1526% ± 1625% | 1912% | 385.1% | 507.9% | 274.8% | 4550% | 0 |
| **Rolling vol histogram** | MSE | 439.33 ± 216.74 | 280.40 | 633.95 | 604.18 | 82.528 | 595.57 | 0 |
| | % err | 294.3% ± 137.0% | 211.2% | 206.2% | 321.6% | 181.6% | 550.7% | 0 |
| **Tail survival** | MSE | 0.0117 ± 0.0092 | 0.0027 | 0.0061 | 0.0219 | 0.0041 | 0.0238 | 0 |
| | % err | 8109% ± 3779% | 5891% | 5375% | 15274% | 5385% | 8622% | 0 |

> **Log-ret histogram**: MSE std (120.6) approaches the mean (144.2), driven by seed 2 near-collapse (504.5 vs 11–170 for the other seeds). A std near the mean is expected for a non-negative combined score over 5 skewed samples — it is not a bug. SBTS is far more stable here (12.1 ± 0.16).
> **ACF \|r\|, ACF r²**: TimeGAN misses the ARCH signature — near-zero ACF vs Heston ≈ +0.05 — so both MSE and % error are larger than SBTS.
> **Rolling vol histogram**: High MSE (439) from vol-distribution mismatch across most seeds.
> **% error reading**: a proper MAPE (mean |gen−real|/|real| × 100) averaged over the three sub-metrics (funct/der/sec_der). Log-ret hist (25147%) and tail survival (8109%) are the hardest — deep-tail and empty-bin points inflate the relative error. Read MSE for absolute agreement, % error for relative shape deviation.

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
