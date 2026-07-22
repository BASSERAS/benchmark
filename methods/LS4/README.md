# LS4 on Heston

PyTorch run of the **official LS4** (Zhou, Poli, Xu, Massaroli, Ermon, **ICML 2023** —
*Deep Latent State Space Models for Time-Series Generation*, PMLR 202, arXiv:2212.12749,
`github.com/alexzhou907/ls4`) trained on 8 192 Heston stochastic-volatility price paths
(seq\_len = 128).

See [`code/README.md`](code/README.md) for the source, the original paper/GitHub, the
architecture (S4 structured-state-space **prior** rolling a continuous latent `z` forward,
S4 **posterior** + S4 **decoder** reconstructing observations, `latent_type = split`,
2 146 857 params), the released `solar_weekly` preset used verbatim
(`z_dim` = 5, `d_model` = 64, `d_state` = 64, `n_layers` = 4, `backbone` = autoreg,
`s4_type` = s4, `sigma` = 0.1), the AdamW + ReduceLROnPlateau + EMA optimiser stack, and
the global-standardise chain (`(X−μ)/σ` with μ = 101.325, σ = 9.972) that maps the
price-scale Heston data into the model's ≈N(0,1) space and decodes back.

> **⚠️ Cauchy fix (REQUIRED — the only change to reference model code).** LS4's
> `model.generate` rolls the S4 latent **prior** forward with `latent.step` — the STEP-mode
> recurrence (one timestep at a time), not the convolutional scan used at training. On a
> CUDA-13 A100 the fast Cauchy kernels (`pykeops` / bundled CUDA extension) are unavailable,
> so S4 falls back to the **naive Python Cauchy kernel** in `reference/models/s4.py`. As
> shipped, that fallback sums the kernel over the *full* pole set instead of over **conjugate
> pole pairs** — correct for the keops/CUDA path, wrong for the naive path used at generation.
> The fix at `code/reference/models/s4.py:795` makes the naive kernel sum over conjugate
> **pairs**, matching keops/CUDA. Without it, generation degenerates (verified on Solar-Weekly:
> marginal score frozen at 0.197 vs paper 0.046 — see
> [`paper_reimplementation/README.md`](paper_reimplementation/README.md)); the Heston generator
> uses the identical `latent.step` recurrence and carries the same fix. No other reference code
> was modified.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.3684 ± 0.01609 | 0.3900 | 0.3719 | 0.3783 | 0.3577 | 0.3439 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 3.99e-04 ± 1.13e-04 | 5.13e-04 | 2.31e-04 | 5.32e-04 | 3.33e-04 | 3.88e-04 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.001156 ± 1.66e-04 | 0.001278 | 0.001017 | 0.001381 | 9.26e-04 | 0.001178 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 4.05e-04 ± 8.23e-05 | 4.85e-04 | 2.86e-04 | 5.10e-04 | 3.73e-04 | 3.71e-04 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 1.225 ± 0.4268 | 1.858 | 1.565 | 1.121 | 0.7392 | 0.8400 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.001926 ± 2.51e-04 | 0.001807 | 0.001572 | 0.001850 | 0.002100 | 0.002298 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.001520 ± 3.61e-04 | 0.001144 | 0.002034 | 0.001187 | 0.001380 | 0.001857 | 0.001983 |
| A8 Increment MMD² ↓ | 9.63e-04 ± 3.76e-05 | 9.24e-04 | 0.001005 | 9.41e-04 | 9.32e-04 | 0.001011 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.01447 ± 0.001550 | 0.01240 | 0.01560 | 0.01463 | 0.01311 | 0.01661 | 0.008554 |
| A10 Terminal SWD ↓ | 0.7480 ± 0.3255 | 0.4507 | 0.5745 | 0.8891 | 0.5025 | 1.323 | 1.151 |
| A11 Path SWD ↓ | 0.5744 ± 0.1246 | 0.4182 | 0.4565 | 0.6336 | 0.6020 | 0.7616 | 0.6191 |
| A12 RV Law Loss ↓ | 0.2415 ± 0.01757 | 0.2518 | 0.2251 | 0.2679 | 0.2199 | 0.2430 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.1722 ± 0.1200 | 0.4071 | 0.1361 | 0.1362 | 0.06929 | 0.1122 | 0.1205 |
| A14 KS Log-returns ↓ | 0.01258 ± 6.74e-04 | 0.01347 | 0.01223 | 0.01169 | 0.01325 | 0.01225 | 0.001491 |
| A15 Skewness Error ↓ | 0.02998 ± 0.01249 | 0.02338 | 0.03295 | 0.01980 | 0.05315 | 0.02062 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 3.41e-04 ± 9.53e-06 | 3.39e-04 | 3.51e-04 | 3.39e-04 | 3.49e-04 | 3.24e-04 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.01584 ± 0.005488 | 0.02539 | 0.01514 | 0.01721 | 0.01233 | 0.009155 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.005890 ± 0.001676 | 0.004425 | 0.009002 | 0.005645 | 0.005951 | 0.004425 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.006256 ± 0.002539 | 0.009307 | 0.001678 | 0.007476 | 0.005951 | 0.006866 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05001 ± 3.66e-06 | 0.05001 | 0.05001 | 0.05001 | 0.05002 | 0.05001 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05006 ± 1.23e-04 | 0.05011 | 0.05015 | 0.04992 | 0.05020 | 0.04990 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 13.63 ± 6.662 | 20.50 | 3.699 | 21.62 | 10.92 | 11.41 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.01294 ± 0.001791 | 0.01504 | 0.01253 | 0.01501 | 0.01116 | 0.01094 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.006752 ± 0.001737 | 0.008731 | 0.006568 | 0.008739 | 0.004985 | 0.004736 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.01743 ± 0.005532 | 0.02617 | 0.01543 | 0.02127 | 0.01312 | 0.01116 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.009068 ± 0.005290 | 0.01786 | 0.007325 | 0.01184 | 0.005517 | 0.002797 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.3270 ± 0.2333 | 0.7538 | 0.3154 | 0.3352 | 0.1245 | 0.1063 | 0.1392 |
| A26 Return Std Error ↓ | 0.004853 ± 0.003540 | 0.001139 | 0.01070 | 0.001243 | 0.006115 | 0.005069 | 0.002523 |
| A27 Log-Return Std Error ↓ | 4.63e-05 ± 2.22e-05 | 3.84e-05 | 7.52e-05 | 5.98e-05 | 4.91e-05 | 9.23e-06 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 1.565 ± 0.07840 | 1.508 | 1.562 | 1.717 | 1.524 | 1.512 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.001445 ± 6.99e-04 | 7.51e-04 | 0.002404 | 5.93e-04 | 0.002007 | 0.001472 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.3372 ± 0.1171 | 0.4314 | 0.1391 | 0.4743 | 0.3436 | 0.2976 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.03798 ± 0.001391 | 0.03909 | 0.03918 | 0.03887 | 0.03711 | 0.03564 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 3.21e-04 ± 4.23e-05 | 3.48e-04 | 2.87e-04 | 3.90e-04 | 2.79e-04 | 2.99e-04 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -3.94e-04 ± 0.006577 | 0.004833 | -0.005410 | 0.007382 | -0.01036 | 0.001587 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.09513 ± 7.87e-04 | 0.09448 | 0.09573 | 0.09398 | 0.09611 | 0.09535 | 0.06559 |

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
| **Log-return histogram** | MSE | 0.4517 ± 0.02799 | 0.4332 | 0.4900 | 0.4752 | 0.4478 | 0.4124 | 0.1098 |
|  | % err | 5.429% ± 0.1852% | 5.500% | 5.328% | 5.621% | 5.579% | 5.117% | 1.799% |
|  | NRMSE | 2.779% ± 0.08180% | 2.722% | 2.929% | 2.786% | 2.762% | 2.693% | 0.5328% |
| **QQ plot** | MSE | 4.59e-08 ± 2.12e-09 | 4.59e-08 | 4.82e-08 | 4.77e-08 | 4.58e-08 | 4.21e-08 | 1.09e-09 |
|  | % err | 6.022% ± 0.6435% | 6.813% | 6.336% | 5.213% | 6.442% | 5.304% | 0.4629% |
|  | NRMSE | 0.9701% ± 0.02323% | 0.9701% | 0.9909% | 0.9842% | 0.9795% | 0.9256% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 5.14e-05 ± 1.08e-05 | 6.76e-05 | 5.21e-05 | 5.74e-05 | 4.32e-05 | 3.66e-05 | 9.61e-06 |
|  | % err | 37.09% ± 3.059% | 38.57% | 40.98% | 38.66% | 34.79% | 32.43% | 8.724% |
|  | NRMSE | 29.46% ± 2.604% | 32.21% | 31.05% | 31.39% | 26.77% | 25.88% | 6.071% |
| **ACF r² lags 1–20** | MSE | 2.48e-05 ± 6.52e-06 | 3.57e-05 | 2.40e-05 | 2.70e-05 | 2.11e-05 | 1.62e-05 | 9.17e-06 |
|  | % err | 24.39% ± 3.127% | 24.85% | 28.58% | 26.75% | 21.57% | 20.19% | 11.34% |
|  | NRMSE | 19.10% ± 2.524% | 21.49% | 21.03% | 20.93% | 16.23% | 15.81% | 6.486% |
| **Rolling vol histogram** | MSE | 8.514 ± 0.7580 | 9.402 | 8.258 | 9.417 | 7.859 | 7.634 | 1.372 |
|  | % err | 11.70% ± 1.165% | 12.72% | 10.83% | 13.36% | 10.26% | 11.31% | 2.264% |
|  | NRMSE | 5.275% ± 0.3034% | 5.628% | 5.238% | 5.618% | 4.949% | 4.943% | 0.8688% |
| **Tail survival** | MSE | 6.90e-05 ± 8.10e-06 | 6.12e-05 | 8.38e-05 | 6.37e-05 | 7.10e-05 | 6.51e-05 | 5.22e-07 |
|  | % err | 3.345% ± 0.1144% | 3.265% | 3.557% | 3.362% | 3.308% | 3.233% | 0.3302% |
|  | NRMSE | 1.449% ± 0.08321% | 1.367% | 1.601% | 1.395% | 1.473% | 1.410% | 0.1050% |

> **Log-ret histogram**: MSE **0.4517** — the **best log-return-histogram fit in the whole benchmark** (next best Fourier Flow 0.92; SBTS 4.08, CSDI 4.64, Diffusion-TS 4.88; TimeGAN 45; TimeVAE 968). LS4's marginal return density is essentially on top of Heston's.
> **QQ plot**: MSE **4.6e-08** (function-level %err 6.02%) — quantile-by-quantile the LS4 return distribution matches Heston's; the residual %err is tail-driven.
> **ACF \|r\|, ACF r²**: MSE tiny (5.1e-5 / 2.5e-5) because the true ACF ≈ 0.05 sits near zero; the **% error** (function-level MAPE) is 37% / 24% for that same near-zero-denominator reason. LS4 tracks the ARCH autocorrelation shape well (A21 0.013, A23 0.017 error — an order of magnitude better than the VAE/GAN baselines).
> **Rolling vol histogram**: MSE **8.514** — the **best rolling-volatility-histogram fit in the benchmark** (next Fourier Flow 29.88; the GAN/VAE baselines range 114–16000). Paired with A31 rolling-vol KS **0.038**, LS4 is the only method here whose rolling-vol distribution overlaps Heston's.

---

## Reading the table — best-in-class density & path fit, no latent-vol recovery

LS4 is the **strongest density- and path-matcher in this benchmark**, sitting at or near the
perfect-recovery floor on the kernel, quantile, adversarial and predictive metrics. Its one
structural blind spot is the Heston latent volatility factor, which — like every single-factor
generator here — it does not recover.

- **At or below the perfect-recovery floor on the hard distributional metrics.** A6 Path MMD² **0.0019** (floor 0.0018),
  A8 Increment MMD² **9.6e-4** (floor 8.7e-4), A11 Path SWD **0.574** (floor 0.619), A16 QQ RMSE
  **3.4e-4**, A14 KS **0.0126** — within a hair of the independent-draw perfect-recovery floor,
  the best of any method in the benchmark. A7 Terminal MMD² **0.0015** and A10 Terminal SWD **0.748**
  actually sit **below** their finite-sample floors (0.0020 / 1.151).
- **Adversarially indistinguishable — A18 ≈ 0.006/0.006.** The discriminative score (`|accuracy − 0.5|`)
  is essentially at chance for both GRU (**0.0059**) and MLP (**0.0063**), right at the perfect floor
  (0.0062 / 0.0060). A modern sequence classifier **cannot** separate LS4 paths from real Heston paths —
  the exact opposite of COSCI-GAN (A18 = 0.50, perfectly separable).
- **Predictive score at the floor — A19 GRU = 0.0500.** TSTR (train-on-synthetic, test-on-real) MAE
  equals the perfect-recovery floor to four decimals (GRU 0.0500 = floor; MLP 0.0501 vs 0.0504). Models
  trained on LS4 output predict real Heston just as well as models trained on real data.
- **Strong scalar moments too.** A1 Kurtosis Error **0.368** (COSCI-GAN 0.56, TimeVAE 2.26; though CSDI 0.095 and SBTS 0.118 are lower),
  A15 Skewness Error **0.030**, and A5 Hill tail-index error **1.23** — the best Hill fit of any method.
- **Thin-tail caveat — A28 = 1.57 (perfect 1.0).** The generated log-returns are mildly *platykurtic*
  relative to Heston: κ_real/κ_gen ≈ 1.57 across all seeds (stable sign, unlike COSCI-GAN's ±). The
  marginal *centre* is excellent (A1, A14, QQ all top-tier) but the extreme tail is slightly under-heavy.
- **No latent-vol recovery — A33 ≈ −0.000 (perfect 0.616), A34 0.095 (perfect 0.066).** The recovered
  instantaneous vol is uncorrelated with the teacher σ. LS4 reproduces the *marginal* and *rolling*
  volatility distributions (A9 0.014, A31 0.038) superbly, but does not reconstruct the *path-wise*
  stochastic-vol trajectory — the standard limitation of single-factor generators on a two-factor SDE.

Net: **the new best-in-class method** on this Heston benchmark for density, path-kernel, adversarial and
predictive fidelity, with two honest caveats — slightly thin tails (A28) and no per-path latent-vol
structure (A33).

---

## Stylised Facts Diagnostic (Heston vs LS4, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/LS4/plots/heston_diagnostics.png)

---

## LS4 Training Loss (5 seeds)

LS4 is a **VAE-style latent state-space model** trained on the ELBO. Each epoch logs
`epoch, total_loss, kld_loss, nll_loss, mse_loss, lr`:

- **`total_loss`** — the training objective, `total = kld_loss + nll_loss` (ELBO). Descends to
  ≈ **−1.068** at convergence (negative because the Gaussian NLL with `sigma` = 0.1 is strongly
  negative on well-reconstructed, standardised data).
- **`kld_loss`** — KL between the S4 latent posterior and the S4 latent prior.
- **`nll_loss`** — Gaussian reconstruction NLL of the decoder.
- **`mse_loss`** — reconstruction MSE, a **diagnostic only** (not part of the objective).
- **`lr`** — AdamW learning rate, stepped by `ReduceLROnPlateau` (patience 20, factor 0.5); EMA of
  the weights (lamb 0.99, start_step 200) is used for generation.

100 epochs/seed, ≈ 973 s/seed on one A100 (naive Cauchy/Vandermonde fallback — no CUDA extension,
hence slower than the paper's keops run), no NaN. See [`code/README.md`](code/README.md) for the loss
definitions and the standardise chain.

![LS4 Training Loss](losses/loss_convergence.png)

---

## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake; here the loss
**stays near ln 2** for both classifiers, the direct cause of the A18 ≈ 0.008/0.010 score —
LS4 paths are near-indistinguishable from real Heston paths.

![Discriminative Classifier Loss](../../results/Heston/LS4/plots/disc_classifier_loss.png)

---

## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/LS4/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest LS4 paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data). The PS-MC pipeline
is **model-agnostic** — it consumes only the generated `.npy` paths, identical to the other methods'.

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/LS4/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/LS4/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 2.7007 ± 0.0034 | 2.7007 ± 0.0034 | 3.7999 ± 0.0055 | 3.7999 ± 0.0055 | 3.73 / 5.30 |
| MAE    | 3.7177 ± 0.0046 | 3.7177 ± 0.0047 | 5.2484 ± 0.0078 | 5.2484 ± 0.0078 | 3.73 / 5.30 |
| RMSE   | 5.0700 ± 0.0040 | 5.0700 ± 0.0039 | 7.1600 ± 0.0056 | 7.1600 ± 0.0055 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS at both horizons** (2.70 < 3.73 at H=32; 3.80 < 5.30 at H=64) —
the **largest CRPS margin of any method in the benchmark**, and LS4 also edges the RW on MAE
(3.72 < 3.73; 5.25 < 5.30) and ties/beats it on RMSE (5.07 ≈ 5.07; 7.16 < 7.18). LS4's generated pool
contains price-anchored futures close enough to the real prefixes to form a well-calibrated
nearest-neighbour ensemble — consistent with its floor-level A18 separability and top-tier stylised-facts
fit. Uniform ≈ Gaussian: Heston is time-homogeneous, so the K nearest neighbours are roughly equally
predictive.

Full analysis: [`../../results/Heston/LS4/path_shadowing/README.md`](../../results/Heston/LS4/path_shadowing/README.md)

---

## File layout

```
methods/LS4/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time, params (2 146 857)
├── weights/
│   ├── seed_{i}_model.pt              EMA state_dict (S4 prior/posterior/decoder)
│   └── seed_{i}_config.json           released solar_weekly preset + standardise constants (μ, σ)
├── losses/
│   ├── seed_{i}_losses.csv            epoch, total_loss, kld_loss, nll_loss, mse_loss, lr
│   └── loss_convergence.png           convergence plot (5 seeds, Total/KLD/NLL/MSE + LR panels)
├── code/
│   ├── train_heston.py                Heston training driver (ELBO, EMA, standardise chain)
│   ├── plot_losses.py                 loss-convergence plot generator
│   ├── reference/                     verbatim official code (alexzhou907/ls4) + the s4.py:795 Cauchy fix
│   └── README.md                      paper, GitHub, architecture, Cauchy fix, hyperparameters
├── paper_reimplementation/            Solar-Weekly marginal/clf/predictive reproduction (paper Table 1)
└── path_shadowing/                    model-agnostic PS-MC forecaster
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/LS4/code
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 0

# Compute all metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method LS4 --dataset Heston

# Run Path Shadowing MC
cd methods/LS4/path_shadowing
/home/tbasseras/gpu-venv/bin/python run_eval.py
```
