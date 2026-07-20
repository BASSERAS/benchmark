# Diffusion-TS on Heston

PyTorch reimplementation of **Diffusion-TS** (Yuan & Qiao, ICLR 2024 —
*Diffusion-TS: Interpretable Diffusion for General Time Series Generation*) trained on 8 192 Heston
stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, the original paper/GitHub, the `mujoco` architecture
choice (with the three smoke Context-FID results), and the normalisation chain applied to fit the
price-scale Heston data into the model's `[-1, 1]` space.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A1 | Kurtosis Error | Fat Tail | ↓ | 0.4238 ± 0.0230 | 0.4232 | 0.3932 | 0.4646 | 0.4173 | 0.4210 | 0 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | 0.0068 ± 1.57e-04 | 0.0069 | 0.0070 | 0.0066 | 0.0069 | 0.0067 | 0 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | 0.0103 ± 1.75e-04 | 0.0103 | 0.0104 | 0.0101 | 0.0106 | 0.0101 | 0 |
| A4 | Tail QQ Error | Fat Tail | ↓ | 0.0067 ± 1.50e-04 | 0.0068 | 0.0069 | 0.0065 | 0.0068 | 0.0066 | 0 |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | 3.613 ± 0.2789 | 3.409 | 3.967 | 3.560 | 3.236 | 3.890 | 0 |
| | **— Distribution —** | | | | | | | | | |
| A6 | Path MMD² | Distribution | ↓ | 0.0033 ± 6.56e-04 | 0.0037 | 0.0033 | 0.0021 | 0.0035 | 0.0040 | 0.0015 |
| A7 | Terminal MMD² | Distribution | ↓ | 0.0021 ± 3.92e-04 | 0.0025 | 0.0022 | 0.0014 | 0.0021 | 0.0023 | 0.0016 |
| A8 | Increment MMD² | Distribution | ↓ | 0.0112 ± 9.37e-04 | 0.0117 | 0.0123 | 0.0095 | 0.0117 | 0.0111 | 7.45e-04 |
| A9 | Volatility MMD | Distribution | ↓ | 0.3840 ± 0.0314 | 0.4085 | 0.4072 | 0.3251 | 0.4012 | 0.3783 | 0.0071 |
| A10 | Terminal SWD | Distribution | ↓ | 1.358 ± 0.2152 | 1.329 | 1.701 | 1.041 | 1.281 | 1.441 | 0.6873 |
| A11 | Path SWD | Distribution | ↓ | 0.9838 ± 0.1107 | 1.028 | 1.005 | 0.7761 | 1.002 | 1.108 | 0.4381 |
| A12 | RV Law Loss | Distribution | ↓ | 2.250 ± 0.0491 | 2.268 | 2.319 | 2.181 | 2.273 | 2.210 | 0 |
| A13 | Mean Path RMSE | Distribution | ↓ | 0.3615 ± 0.2364 | 0.2165 | 0.5903 | 0.7015 | 0.1661 | 0.1330 | 0 |
| A14 | KS Log-returns | Distribution | ↓ | 0.0600 ± 0.0019 | 0.0605 | 0.0627 | 0.0569 | 0.0606 | 0.0595 | 0 |
| A15 | Skewness Error | Distribution | ↓ | 0.0698 ± 0.0358 | 0.0799 | 0.0302 | 0.0296 | 0.0860 | 0.1232 | 0 |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0030 ± 8.30e-05 | 0.0031 | 0.0032 | 0.0029 | 0.0031 | 0.0030 | 0 |
| A17 | Terminal Price KS | Distribution | ↓ | 0.0400 ± 0.0073 | 0.0294 | 0.0480 | 0.0431 | 0.0461 | 0.0336 | 0 |
| | **— Adversarial —** | | | | | | | | | |
| A18 GRU | Discriminative Score GRU | Adversarial | ↓ | 0.2621 ± 0.1578 | 0.3700 | 0.4042 | 0.0282 | 0.1164 | 0.3917 | 0.0042 |
| A18 MLP | Discriminative Score MLP | Adversarial | ↓ | 0.0554 ± 0.0396 | 0.0163 | 0.0041 | 0.0700 | 0.1106 | 0.0761 | 0.0067 |
| | **— Predictive —** | | | | | | | | | |
| A19 GRU | Predictive Score GRU | Predictive | ↓ | 0.0549 ± 1.59e-04 | 0.05503 | 0.05467 | 0.05496 | 0.05479 | 0.05510 | 0.0537 |
| A19 MLP | Predictive Score MLP | Predictive | ↓ | 0.0551 ± 3.72e-04 | 0.05500 | 0.05458 | 0.05539 | 0.05566 | 0.05497 | 0.0539 |
| | **— Temporal —** | | | | | | | | | |
| A20 | Covariance Error | Temporal | ↓ | 38.172 ± 10.637 | 27.684 | 35.689 | 30.357 | 57.851 | 39.279 | 0 |
| A21 | ACF \|r\| Error (lags) | Temporal | ↓ | 0.0201 ± 0.0030 | 0.0204 | 0.0146 | 0.0203 | 0.0234 | 0.0219 | 0 |
| A22 | ACF r² Error (lags) | Temporal | ↓ | 0.0168 ± 0.0027 | 0.0157 | 0.0120 | 0.0182 | 0.0190 | 0.0190 | 0 |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0.0039 ± 0.0022 | 0.0066 | 0.0010 | 0.0023 | 0.0064 | 0.0034 | 0 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | 0.0038 ± 0.0026 | 4.20e-05 | 0.0069 | 0.0063 | 0.0018 | 0.0039 | 0 |
| | **— Vol —** | | | | | | | | | |
| A25 | Mean RMSE | Vol | ↓ | 0.5767 ± 0.4444 | 0.3298 | 1.028 | 1.177 | 0.3149 | 0.0343 | 0 |
| A26 | Return Std Error | Vol | ↓ | 0.3098 ± 0.0093 | 0.3132 | 0.3235 | 0.2956 | 0.3118 | 0.3048 | 0 |
| A27 | Log-Return Std Error | Vol | ↓ | 0.0032 ± 8.20e-05 | 0.0032 | 0.0033 | 0.0031 | 0.0032 | 0.0031 | 0 |
| A28 | Kurtosis Ratio | Vol | — | 1.866 ± 0.2509 | 1.865 | 1.434 | 2.129 | 2.101 | 1.803 | 1.000 |
| A29 | Sigma Mean Error | Vol | ↓ | 0.0485 ± 0.0013 | 0.0490 | 0.0504 | 0.0467 | 0.0488 | 0.0476 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 1.154 ± 0.2019 | 1.137 | 0.8906 | 1.007 | 1.470 | 1.267 | 0 |
| A31 | Rolling Vol KS (w=5) | Vol | ↓ | 0.2558 ± 0.0078 | 0.2540 | 0.2696 | 0.2481 | 0.2584 | 0.2490 | 0 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 0.0016 ± 3.80e-05 | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0015 | 0 |
| | **— Heston Spec —** | | | | | | | | | |
| A33 | Teacher-Sigma Corr | Heston Spec | ↑ | -0.0036 ± 0.0032 | -0.0035 | -0.0032 | -0.0089 | 0.0010 | -0.0034 | 0.6143 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | 0.0960 ± 7.41e-04 | 0.0961 | 0.0974 | 0.0953 | 0.0955 | 0.0957 | 0.0654 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (row-shuffle keeps joint path structure).
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

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = m_funct + m_der + m_sec\_der (**sum** of the three seed-means); std = sqrt(s_funct² + s_der² + s_sec\_der²) (**quadrature**).
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE — one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the 5 seeds** of that per-seed function MAPE.

All ↓ lower is better. Perfect floor = 0 for all six plots (row-shuffled real data has identical curves).
Two sublines per plot: **MSE** and **% error** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|------|---------|-----------|--------|--------|--------|--------|--------|:------:|
| **Log-return histogram** | MSE | 14.505 ± 1.469 | 14.898 | 16.807 | 12.225 | 14.788 | 13.810 | 0 |
| | % err | 41.943% ± 0.996% | 42.311% | 43.375% | 40.470% | 42.303% | 41.255% | 0 |
| **QQ plot** | MSE | 1.028e-05 ± 5.24e-07 | 1.03e-05 | 1.11e-05 | 9.35e-06 | 1.04e-05 | 1.01e-05 | 0 |
| | % err | 25.392% ± 1.704% | 25.641% | 26.438% | 22.044% | 26.578% | 26.258% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 5.76e-04 ± 1.26e-04 | 5.32e-04 | 3.07e-04 | 6.05e-04 | 7.67e-04 | 6.68e-04 | 0 |
| | % err | 74.755% ± 12.017% | 77.249% | 52.700% | 73.708% | 87.881% | 82.238% | 0 |
| **ACF r² lags 1–20** | MSE | 4.34e-04 ± 1.07e-04 | 3.83e-04 | 1.83e-04 | 4.72e-04 | 5.95e-04 | 5.32e-04 | 0 |
| | % err | 73.899% ± 14.289% | 76.727% | 47.460% | 73.138% | 88.936% | 83.235% | 0 |
| **Rolling vol histogram** | MSE | 652.353 ± 44.946 | 643.547 | 735.677 | 609.589 | 658.960 | 613.991 | 0 |
| | % err | 68.609% ± 1.420% | 68.127% | 70.702% | 67.183% | 69.825% | 67.208% | 0 |
| **Tail survival** | MSE | 6.70e-03 ± 5.97e-04 | 6.82e-03 | 7.71e-03 | 5.94e-03 | 6.75e-03 | 6.29e-03 | 0 |
| | % err | 28.247% ± 0.842% | 28.493% | 29.553% | 27.092% | 28.475% | 27.620% | 0 |

> **Log-ret histogram**: MSE 14.5 is the weakest B panel in absolute terms — the diffusion sampler slightly over-disperses the central log-return bins relative to Heston.
> **ACF \|r\|, ACF r²**: the MSE is tiny (~5e-4) because the true ACF ≈ 0.05 sits near zero, but the **% error** (function-level MAPE) is large (75% / 74%) for exactly that reason — near-zero denominators amplify any deviation. Read MSE for absolute agreement, % error for relative shape.
> **QQ plot**: MSE 1.0e-5 — the return-quantile curve is reproduced tightly, consistent with the low A16 QQ RMSE (0.0030).

---

## Stylised Facts Diagnostic (Heston vs Diffusion-TS, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/DiffusionTS/plots/heston_diagnostics.png)

---

## Diffusion-TS Training Loss (5 seeds)

Reweighted L1 + Fourier-FFT reconstruction loss (direct $\hat{x}_0$ prediction) over 12 000 steps
(`mujoco` preset), all 5 seeds. The loss decreases smoothly from ~3.9 to ~0.07 and plateaus — see
[`code/README.md`](code/README.md) for the training signals and normalisation chain.

![Diffusion-TS Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/DiffusionTS/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/DiffusionTS/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest Diffusion-TS paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to FourierFlow's and TimeGAN's.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/DiffusionTS/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/DiffusionTS/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **2.717 ± 0.003** | 2.717 ± 0.003 | **3.845 ± 0.005** | 3.845 ± 0.005 | 3.73 / 5.30 |
| MAE    | 3.718 ± 0.004 | 3.718 ± 0.004 | 5.259 ± 0.011 | 5.259 ± 0.011 | 3.73 / 5.30 |
| RMSE   | 5.084 ± 0.006 | 5.084 ± 0.006 | 7.196 ± 0.007 | 7.196 ± 0.007 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.72 < 3.73 at H=32; 3.85 < 5.30 at H=64), on all
5 seeds, and its CRPS is the **lowest of the three methods** (2.72 vs FourierFlow 2.74, TimeGAN 3.09 at H=32) —
the Diffusion-TS pool gives the tightest, best-calibrated nearest-neighbour futures. Uniform ≈ Gaussian:
Heston is time-homogeneous, so the K nearest neighbours are roughly equally predictive.

Full analysis: [`../../results/Heston/DiffusionTS/path_shadowing/README.md`](../../results/Heston/DiffusionTS/path_shadowing/README.md)

---

## File layout

```
methods/DiffusionTS/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time, params
├── weights/
│   ├── seed_{i}_model.pt              model + EMA state_dict, arch, minmax
│   └── seed_{i}_config.json           full hyperparameters + scaling constants
├── losses/
│   ├── seed_{i}_losses.csv            step, loss (reweighted L1 + Fourier)
│   └── loss_convergence.png           convergence plot (5 seeds overlaid)
├── code/
│   ├── train_heston.py                Heston training driver (imports reference Diffusion-TS)
│   ├── reference/                     verbatim released code (Y-debug-sys/Diffusion-TS)
│   └── README.md                      paper, GitHub, mujoco arch choice + 3 smoke results
├── paper_reimplementation/            Stocks len-24 Table-1 reproduction (paper's own 4 metrics)
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (mujoco, 2 GPUs in parallel — see code/README.md)
cd methods/DiffusionTS/code
PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py --arch mujoco --seed 0

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method DiffusionTS --dataset Heston
```
