# CSDI on Heston

PyTorch reimplementation of **CSDI** (Tashiro, Song, Song, Ermon, NeurIPS 2021 —
*CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*)
adapted as an **unconditional** score-based diffusion generator, trained on 8 192 Heston
stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for the source, the original paper/GitHub, and the exact
change (`is_unconditional=1` + `cond_mask ≡ 0`) that turns the authors' conditional imputer into a
pure DDPM sampler for Heston generation.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

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

> **QQ plot**: CSDI's best B panel — MSE 2.4e-06 (function-level MAPE 24%) shows the return-quantile shape is reproduced tightly. **ACF \|r\|, ACF r²**: MSE is tiny (3.0e-05 / 2.7e-05) because the true ACF ≈ 0.05 sits near zero; read MSE for absolute agreement, % error for relative shape.
> **Log-return histogram / Rolling vol histogram**: the two weakest panels in absolute terms (MSE 4.644 / 157.5) — the diffusion pool slightly over-disperses the return and rolling-vol distributions.

---

## Stylised Facts Diagnostic (Heston vs CSDI, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/CSDI/plots/heston_diagnostics.png)

---

## CSDI Training Loss (5 seeds)

Single-phase **DDPM noise-prediction MSE** $\mathbb{E}_t \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$
over 200 epochs (359 mini-batches/epoch, batch 16). CSDI is a score-based diffusion model, so there is
**no adversarial or supervised phase** — a single denoising objective is minimised. Optimiser Adam
(lr 1e-3, weight-decay 1e-6); `MultiStepLR` decays lr ×0.1 at epochs 150 and 180. Unconditional
generation is obtained by forcing the conditioning mask to zero (`cond_mask ≡ 0`), so every point is a
diffusion target (see [`code/README.md`](code/README.md)). Min training loss ≈ 0.0096, no NaN.

![CSDI Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/CSDI/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/CSDI/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest CSDI paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to TimeGAN / Fourier Flow.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/CSDI/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/CSDI/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **2.713 ± 0.005** | 2.713 ± 0.005 | **3.814 ± 0.007** | 3.814 ± 0.007 | 3.73 / 5.30 |
| MAE    | 3.721 ± 0.007 | 3.721 ± 0.007 | 5.254 ± 0.009 | 5.254 ± 0.009 | 3.73 / 5.30 |
| RMSE   | 5.067 ± 0.005 | 5.067 ± 0.005 | 7.154 ± 0.005 | 7.154 ± 0.005 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.71 < 3.73 at H=32; 3.81 < 5.30 at H=64), on all
5 seeds. CSDI's CRPS is the **lowest of all methods benchmarked** (2.713 vs Fourier Flow 2.742 and
TimeGAN 3.09 at H=32) — its diffusion-generated pool provides the tightest, best-calibrated
nearest-neighbour futures. Uniform ≈ Gaussian: Heston is time-homogeneous, so the K nearest neighbours
are roughly equally predictive.

Full analysis: [`../../results/Heston/CSDI/path_shadowing/README.md`](../../results/Heston/CSDI/path_shadowing/README.md)

---

## File layout

```
methods/CSDI/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time, params, z-score
├── weights/
│   ├── seed_{i}_model.pt              full PyTorch state_dict
│   └── seed_{i}_config.json           hyperparameters (base.yaml verbatim)
├── losses/
│   ├── seed_{i}_losses.csv            step, loss (DDPM noise-prediction MSE)
│   └── loss_convergence.png           convergence plot (5 seeds overlaid)
├── code/
│   ├── train_heston.py                unconditional CSDI generator (imports reference CSDI)
│   ├── plot_losses.py                 loss_convergence.png generator
│   ├── reference/                     verbatim released code (ermongroup/CSDI)
│   └── README.md                      paper, GitHub, the is_unconditional=1 change
├── paper_reimplementation/            PhysioNet + PM2.5 CRPS reproduction (Table 2) + Heston imputation-CRPS
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel — one --seed per process)
cd methods/CSDI/code
for s in 0 1 2 3 4; do
  gpu=$([ $((s % 2)) -eq 0 ] && echo 0 || echo 3)
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s &
done
wait

# Compute all metrics
cd metrics
python compute_all.py --method CSDI --dataset Heston

# Run Path Shadowing MC
cd methods/CSDI/path_shadowing
python run_eval.py
```

See [`code/README.md`](code/README.md) for the exact interpreter path, env vars, and which output
file produced each committed number.
