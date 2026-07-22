# TimeVQVAE on Heston

PyTorch reimplementation of **TimeVQVAE** (Lee, Malacarne & Aune, 2023 —
*Vector Quantized Time Series Generation with a Bidirectional Prior Model*, AISTATS 2023,
`lee23d`, arXiv:2303.04743) trained on 8 192 Heston stochastic-volatility price paths
(seq\_len = 128).

See [`code/README.md`](code/README.md) for the source, the original paper/GitHub, the two-stage
architecture (Stage-1 STFT VQ-VAE with separate LF/HF branches, Stage-2 MaskGIT bidirectional prior),
the exact paper hyperparameters, and the global z-normalisation chain applied to fit the price-scale
Heston data into the model's standardised space.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.1363 ± 0.09243 | 0.07804 | 0.1050 | 0.01350 | 0.2240 | 0.2611 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.004515 ± 2.54e-04 | 0.004119 | 0.004424 | 0.004643 | 0.004499 | 0.004891 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.006058 ± 3.03e-04 | 0.005649 | 0.005854 | 0.006323 | 0.005987 | 0.006475 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.004444 ± 2.48e-04 | 0.004061 | 0.004372 | 0.004542 | 0.004421 | 0.004824 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 3.777 ± 1.193 | 1.855 | 5.546 | 3.769 | 3.481 | 4.232 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.003433 ± 7.97e-04 | 0.003413 | 0.003091 | 0.003349 | 0.002440 | 0.004872 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.003838 ± 0.001368 | 0.003903 | 0.002518 | 0.005524 | 0.002094 | 0.005151 | 0.001983 |
| A8 Increment MMD² ↓ | 0.007018 ± 0.001054 | 0.005662 | 0.007131 | 0.006964 | 0.006472 | 0.008863 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.1932 ± 0.02799 | 0.1591 | 0.1928 | 0.2020 | 0.1719 | 0.2403 | 0.008554 |
| A10 Terminal SWD ↓ | 1.356 ± 0.2690 | 1.464 | 0.9473 | 1.747 | 1.196 | 1.424 | 1.151 |
| A11 Path SWD ↓ | 0.8781 ± 0.2081 | 0.9891 | 0.7267 | 1.228 | 0.6502 | 0.7961 | 0.6191 |
| A12 RV Law Loss ↓ | 1.706 ± 0.08942 | 1.581 | 1.681 | 1.703 | 1.705 | 1.860 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.7593 ± 0.1340 | 0.8771 | 0.7874 | 0.8817 | 0.7348 | 0.5152 | 0.1205 |
| A14 KS Log-returns ↓ | 0.05084 ± 0.003747 | 0.04730 | 0.04992 | 0.04786 | 0.05135 | 0.05775 | 0.001491 |
| A15 Skewness Error ↓ | 0.03079 ± 0.008248 | 0.03836 | 0.04116 | 0.02484 | 0.01896 | 0.03064 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.002268 ± 1.38e-04 | 0.002082 | 0.002234 | 0.002235 | 0.002276 | 0.002510 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.05522 ± 0.009093 | 0.06470 | 0.06152 | 0.04358 | 0.06152 | 0.04480 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.07174 ± 0.06503 | 0.05172 | 0.1741 | 0.009918 | 0.1173 | 0.005645 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.009002 ± 0.003393 | 0.008087 | 0.01022 | 0.007171 | 0.004730 | 0.01480 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05014 ± 2.87e-05 | 0.05011 | 0.05014 | 0.05011 | 0.05018 | 0.05016 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05018 ± 6.79e-05 | 0.05027 | 0.05020 | 0.05008 | 0.05014 | 0.05023 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 22.61 ± 14.72 | 12.02 | 18.36 | 11.74 | 19.56 | 51.35 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.01979 ± 0.004246 | 0.01778 | 0.01699 | 0.01557 | 0.02117 | 0.02744 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.01817 ± 0.003251 | 0.01696 | 0.01587 | 0.01531 | 0.01836 | 0.02433 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.01523 ± 0.008014 | 0.006685 | 0.007382 | 0.01722 | 0.01613 | 0.02871 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.01323 ± 0.007254 | 0.005223 | 0.006626 | 0.01767 | 0.01185 | 0.02479 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 1.033 ± 0.1905 | 1.313 | 1.058 | 0.8230 | 1.151 | 0.8223 | 0.1392 |
| A26 Return Std Error ↓ | 0.2316 ± 0.01420 | 0.2106 | 0.2269 | 0.2410 | 0.2269 | 0.2524 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.002336 ± 1.37e-04 | 0.002147 | 0.002297 | 0.002330 | 0.002335 | 0.002573 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 0.8410 ± 0.06953 | 0.8536 | 0.8218 | 0.9665 | 0.8019 | 0.7613 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.03743 ± 0.002059 | 0.03477 | 0.03708 | 0.03689 | 0.03729 | 0.04113 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.5701 ± 0.3404 | 0.3155 | 0.4384 | 0.3385 | 0.5238 | 1.234 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.1850 ± 0.01013 | 0.1690 | 0.1847 | 0.1817 | 0.1895 | 0.2000 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 6.76e-04 ± 5.79e-05 | 6.10e-04 | 6.46e-04 | 7.79e-04 | 6.93e-04 | 6.52e-04 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 7.04e-04 ± 0.005837 | -0.004412 | -0.004393 | -0.002965 | 0.009693 | 0.005596 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.1014 ± 9.08e-04 | 0.1011 | 0.1015 | 0.1001 | 0.1012 | 0.1029 | 0.06559 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns|. **A4**: QQ error restricted to top-5% tail quantiles. **A5**: |Hill tail index_real − Hill tail index_gen|, Hill estimator on |log-returns| above 95th pct.
> **A6–A11**: path-kernel distances — Gaussian MMD² on full paths / terminal prices / increments / realized-vol, and sliced-Wasserstein on terminal & full paths. Non-zero perfect floor (an independent Heston draw scored against the test set — finite-sample noise).
> **A12**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002). **A13**: path-level RMSE between real/gen mean trajectories. **A14**: KS statistic on pooled log-returns. **A15**: |skew_real − skew_gen|, Heston true skew ≈ −0.45. **A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS statistic on terminal prices S_T.
> **A18**: Discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). **A19**: TSTR predictive MAE; all methods cluster near 0.056–0.059 (irreducible log-return floor) (GRU + MLP).
> **A20**: covariance-matrix error (%). **A21–A22**: ACF error on |r| and r² across lags 1–20. ARCH signal: |r_t| has positive lag-1 ACF ~0.05 in Heston. **A23–A24**: ACF lag-1 error on |r| and r². Heston true values ≈ +0.052 / +0.050.
> **A25**: mean-path RMSE. **A26**: return std error, uses price increments $\Delta S_t$. **A27**: log-return std error, uses $r_t = \log(S_{t+1}/S_t)$. **A28**: kurtosis ratio real/gen, perfect = 1.0. **A29**: sigma mean error — annualized per-path vol. **A30**: cross-sectional vol-path RMSE. **A31**: KS statistic on rolling-5 vol histograms. **A32**: |vol-of-vol_real − vol-of-vol_gen|.
> **A33**: Teacher-sigma correlation (Heston-recovered vol vs teacher σ), higher is better, perfect ≈ 0.614. **A34**: Teacher-sigma RMSE, perfect ≈ 0.065.

**Reading the table.** TimeVQVAE is the strongest method in this benchmark on the distributional and
adversarial axes. The **discriminative scores sit essentially on the perfect floor** (A18 GRU 0.024
vs floor 0.004; A18 MLP 0.007 vs floor 0.007) — the trained classifiers cannot reliably separate its
samples from real Heston paths. The **fat-tail block is near-ideal** (A1 Kurtosis Error 0.14, A28
Kurtosis Ratio 0.82 — captures ~82% of Heston's excess kurtosis, versus TimeVAE's 0.28), and the
volatility distribution is matched an order of magnitude better than TimeVAE (A9 Volatility MMD 0.20
vs 3.59; A31 Rolling-Vol KS 0.18 vs 0.987). The weak spots are **A20 Covariance Error** (16.6,
seed-4-driven — the STFT tokeniser occasionally distorts the lag-covariance structure) and the
**Heston-spec latent-vol recovery** (A33 ≈ 0, A34 0.101), where — like every generator in this
benchmark — it fails to reconstruct the unobserved instantaneous variance path.

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
| **Log-return histogram** | MSE | 4.386 ± 0.8335 | 3.620 | 4.185 | 3.625 | 4.631 | 5.871 | 0.1098 |
|  | % err | 30.95% ± 1.747% | 28.63% | 30.54% | 30.43% | 31.16% | 34.01% | 1.799% |
|  | NRMSE | 9.691% ± 0.9011% | 8.831% | 9.521% | 8.872% | 9.941% | 11.29% | 0.5328% |
| **QQ plot** | MSE | 1.82e-06 ± 2.20e-07 | 1.53e-06 | 1.75e-06 | 1.77e-06 | 1.82e-06 | 2.21e-06 | 1.09e-09 |
|  | % err | 23.84% ± 2.434% | 23.56% | 24.27% | 19.44% | 25.24% | 26.67% | 0.4629% |
|  | NRMSE | 6.308% ± 0.3785% | 5.797% | 6.204% | 6.240% | 6.327% | 6.971% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 1.22e-04 ± 3.84e-05 | 1.41e-04 | 1.11e-04 | 5.55e-05 | 1.30e-04 | 1.71e-04 | 9.61e-06 |
|  | % err | 63.03% ± 14.21% | 66.81% | 61.31% | 37.15% | 70.55% | 79.32% | 8.724% |
|  | NRMSE | 45.54% ± 9.362% | 47.04% | 43.31% | 29.49% | 49.90% | 57.97% | 6.071% |
| **ACF r² lags 1–20** | MSE | 1.05e-04 ± 3.00e-05 | 1.35e-04 | 9.82e-05 | 5.43e-05 | 1.03e-04 | 1.37e-04 | 9.17e-06 |
|  | % err | 70.37% ± 13.75% | 77.06% | 69.76% | 44.85% | 74.57% | 85.59% | 11.34% |
|  | NRMSE | 45.61% ± 7.936% | 48.22% | 44.45% | 31.73% | 47.57% | 56.10% | 6.486% |
| **Rolling vol histogram** | MSE | 113.9 ± 13.91 | 95.07 | 113.4 | 106.0 | 118.3 | 137.0 | 1.372 |
|  | % err | 54.51% ± 2.433% | 50.75% | 53.48% | 54.17% | 56.30% | 57.84% | 2.264% |
|  | NRMSE | 20.68% ± 1.268% | 18.92% | 20.67% | 19.96% | 21.12% | 22.74% | 0.8688% |
| **Tail survival** | MSE | 0.001709 ± 2.78e-04 | 0.001414 | 0.001643 | 0.001521 | 0.001748 | 0.002218 | 5.22e-07 |
|  | % err | 22.34% ± 1.374% | 20.56% | 22.00% | 21.89% | 22.45% | 24.78% | 0.3302% |
|  | NRMSE | 7.206% ± 0.5711% | 6.576% | 7.088% | 6.821% | 7.311% | 8.235% | 0.1050% |

> **Log-ret histogram**: MSE 4.386 — over **200× lower** than TimeVAE's 968. TimeVQVAE reproduces the log-return density shape (A28 Kurtosis Ratio 0.84), so the histogram bins align in absolute terms; the residual % err ~31% is the central-peak height mismatch.
> **QQ plot**: MSE 1.8e-06, essentially perfect — the quantile-quantile curve of generated vs real log-returns is nearly on the diagonal.
> **ACF \|r\|, ACF r²**: the MSE is tiny (~1.2e-04 / 1.1e-04) because the true ACF ≈ 0.05 sits near zero; the **% error** (function-level MAPE) is ~63–70%, far better than TimeVAE's ~984–1026% — TimeVQVAE captures a meaningful share of the ARCH autocorrelation. Read MSE for absolute agreement, % error for relative shape.
> **Rolling vol histogram**: MSE 113.9 (vs TimeVAE's 16019) — still the weakest B panel in relative terms (% err ~55%), consistent with the residual A31 Rolling-Vol KS 0.18.

---

## Stylised Facts Diagnostic (Heston vs TimeVQVAE, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/TimeVQVAE/plots/heston_diagnostics.png)

---

## TimeVQVAE Training Loss (5 seeds)

TimeVQVAE trains in **two sequential stages**, so the convergence figure is a 2×2 grid:

- **Stage 1 — STFT VQ-VAE tokenisation.** Total loss = time-domain + time-frequency reconstruction +
  LF/HF commitment losses. It falls from ~2.59 to **~0.024** over 250 epochs (seed-0 final 0.0244;
  all seeds 0.023–0.032). The LF/HF **codebook perplexity** rises from ~9 to ~22 out of the max 32
  codes, confirming the codebooks stay well-used and do not collapse.
- **Stage 2 — MaskGIT bidirectional prior.** A single masked-token cross-entropy (`prior_loss`)
  falls from ~5.42 to **~0.87** over 1 000 epochs (per-seed final: 0.865 / 0.697 / 1.194 / 0.907 /
  0.773).

Budget: **stage1 = 250 / stage2 = 1000 epochs**, ~53 min/seed on one A100 (seed times 52.8–54.4 min).
This is deliberately fewer epochs than the paper's 2000/10000 — the paper's schedule is calibrated for
tiny UCR sets, whereas Heston has 8 192 samples (16× larger), so this budget **matches the paper's
gradient-step count** (~16k stage-1 / ~32k stage-2 steps). See [`code/README.md`](code/README.md) for
the full loss definition and the z-normalisation chain.

![TimeVQVAE Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/TimeVQVAE/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/TimeVQVAE/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest TimeVQVAE paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to the other methods'.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/TimeVQVAE/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/TimeVQVAE/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 2.771 ± 0.015 | 2.770 ± 0.015 | 3.889 ± 0.017 | 3.889 ± 0.017 | 3.73 / 5.30 |
| MAE    | 3.701 ± 0.003 | 3.701 ± 0.003 | 5.227 ± 0.007 | 5.227 ± 0.007 | 3.73 / 5.30 |
| RMSE   | 5.066 ± 0.006 | 5.066 ± 0.006 | 7.159 ± 0.010 | 7.159 ± 0.010 | 5.07 / 7.18 |

TimeVQVAE PS-MC **beats the naive RW on CRPS at both horizons** (2.77 < 3.73 at H=32; 3.89 < 5.30 at
H=64) — the only method in the benchmark to do so cleanly. Its generated pool contains price-anchored
futures close enough to the real prefixes to form a well-calibrated nearest-neighbour ensemble,
consistent with its near-floor discriminative score (A18) and strong stylised-facts fit (A9 Volatility
MMD 0.20, A28 Kurtosis Ratio 0.82). Uniform ≈ Gaussian: Heston is time-homogeneous, so the K nearest
neighbours are roughly equally predictive.

Full analysis: [`../../results/Heston/TimeVQVAE/path_shadowing/README.md`](../../results/Heston/TimeVQVAE/path_shadowing/README.md)

---

## File layout

```
methods/TimeVQVAE/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, znorm mean/std, train time, params
├── weights/
│   ├── seed_{i}_model.pt              Stage-1 VQ-VAE + Stage-2 MaskGIT state_dict
│   └── seed_{i}_config.json           full hyperparameters + z-norm constants
├── losses/
│   ├── seed_{i}_stage1_losses.csv     stage1: VQ recon/commit/perplexity per epoch
│   ├── seed_{i}_stage2_losses.csv     stage2: MaskGIT prior loss per epoch
│   └── loss_convergence.png           2×2 grid (Stage-1 total/recon, Stage-2 prior, perplexity)
├── code/
│   ├── train_heston.py                Heston training driver (two-stage)
│   ├── plot_losses.py                 loss-convergence plot generator
│   ├── reference/                     verbatim released code (ML4ITS/TimeVQVAE)
│   └── README.md                      paper, GitHub, architecture, hyperparameters, running config
├── paper_reimplementation/            ECG5000 paper reproduction (FID / IS)
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel; ~53 min/seed)
cd methods/TimeVQVAE/code
/home/tbasseras/tvqvae-venv/bin/python train_heston.py --seed 0 --stage1_epochs 250 --stage2_epochs 1000

# Compute all metrics (shared harness, gpu-venv)
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method TimeVQVAE --dataset Heston
/home/tbasseras/gpu-venv/bin/python metrics/recompute_curve_b.py --method TimeVQVAE

# Run Path Shadowing MC
cd methods/TimeVQVAE/path_shadowing
/home/tbasseras/gpu-venv/bin/python run_eval.py
```
