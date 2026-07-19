# Results — Methods Comparison on Heston

All methods are evaluated on the same dataset:
**8 192 Heston price paths, seq\_len = 128**
(μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250)

---

## Cross-Method Comparison A1–A34 — Heston (mean ± std, 5 seeds)

↓ = lower is better. ↑ = higher is better. (→ 1) = ratio, target = 1.0. **Bold** = best method per metric.
Perfect-recovery floors (the empirical best-case score a perfect generator would achieve with finite
samples) are method-independent and documented once, reproducibly, in
[`methods/perfect_recovery/README.md`](../methods/perfect_recovery/README.md).

| Metric | SBTS | TimeGAN | Fourier Flow | Winner |
|--------|:----:|:-------:|:------------:|:------:|
| **— Fat Tail —** | | | | |
| A1 Kurtosis Error ↓ | **0.1187 ± 0.0060** | 2.955 ± 2.099 | 0.5757 ± 0.0083 | **SBTS** |
| A2 \|r\| q95 Error ↓ | 0.0063 ± 3.00e-05 | 0.0032 ± 0.0018 | **6.52e-04 ± 2.10e-04** | **Fourier Flow** |
| A3 \|r\| q99 Error ↓ | 0.0098 ± 4.80e-05 | 0.0043 ± 0.0028 | **0.0023 ± 5.06e-04** | **Fourier Flow** |
| A4 Tail QQ Error ↓ | 0.0062 ± 2.60e-05 | 0.0034 ± 0.0015 | **7.15e-04 ± 1.23e-04** | **Fourier Flow** |
| A5 Hill Tail Index Error ↓ | 9.499 ± 0.3457 | 36.885 ± 17.053 | **6.368 ± 2.000** | **Fourier Flow** |
| **— Distribution —** | | | | |
| A6 Path MMD² ↓ | 0.0110 ± 0.0026 | 0.0165 ± 0.0127 | **0.0052 ± 0.0019** | **Fourier Flow** |
| A7 Terminal MMD² ↓ | **0.0100 ± 0.0036** | 0.0267 ± 0.0192 | 0.0106 ± 0.0051 | **SBTS** |
| A8 Increment MMD² ↓ | 0.0071 ± 2.47e-04 | 0.0077 ± 0.0041 | **0.0011 ± 7.70e-05** | **Fourier Flow** |
| A9 Volatility MMD ↓ | 0.3038 ± 0.0071 | 0.3789 ± 0.2430 | **0.0596 ± 0.0086** | **Fourier Flow** |
| A10 Terminal SWD ↓ | 3.539 ± 0.7368 | **2.658 ± 0.8567** | 2.815 ± 0.9433 | **TimeGAN** |
| A11 Path SWD ↓ | 2.415 ± 0.4104 | 1.417 ± 0.4914 | **1.289 ± 0.4198** | **Fourier Flow** |
| A12 RV Law Loss ↓ | 2.148 ± 0.0074 | 1.551 ± 0.3788 | **0.5291 ± 0.1299** | **Fourier Flow** |
| A13 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | 0.5289 ± 0.2624 | **0.4910 ± 0.4022** | **Fourier Flow** |
| A14 KS Log-returns ↓ | 0.0534 ± 3.62e-04 | 0.0848 ± 0.0374 | **0.0191 ± 0.0024** | **Fourier Flow** |
| A15 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | 0.0282 ± 0.0152 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 0.0028 ± 1.20e-05 | 0.0025 ± 6.43e-04 | **5.86e-04 ± 4.20e-05** | **Fourier Flow** |
| A17 Terminal Price KS ↓ | 0.0921 ± 0.0051 | 0.1121 ± 0.0556 | **0.0848 ± 0.0166** | **Fourier Flow** |
| **— Adversarial —** | | | | |
| A18 Disc Score GRU ↓ | 0.2755 ± 0.2166 | **0.0077 ± 0.0050** | 0.0094 ± 0.0097 | **TimeGAN** |
| A18 Disc Score MLP ↓ | 0.0079 ± 0.0049 | 0.1031 ± 0.0395 | **0.0053 ± 0.0041** | **Fourier Flow** |
| **— Predictive —** | | | | |
| A19 Pred Score GRU ↓ | 0.0586 ± 5.90e-05 | 0.0574 ± 0.0019 | **0.0537 ± 7.60e-06** | **Fourier Flow** |
| A19 Pred Score MLP ↓ | 0.0583 ± 2.55e-04 | 0.0570 ± 0.0012 | **0.0540 ± 4.90e-04** | **Fourier Flow** |
| **— Temporal —** | | | | |
| A20 Covariance Error ↓ | 145.35 ± 4.886 | **17.751 ± 6.707** | 64.406 ± 38.255 | **TimeGAN** |
| A21 ACF \|r\| Error (lags) ↓ | 0.0596 ± 4.70e-04 | 0.1252 ± 0.0674 | **0.0435 ± 5.50e-04** | **Fourier Flow** |
| A22 ACF r² Error (lags) ↓ | 0.0619 ± 5.08e-04 | 0.0839 ± 0.0348 | **0.0379 ± 5.56e-04** | **Fourier Flow** |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1437 ± 0.0012 | 0.2264 ± 0.1034 | **0.0526 ± 7.04e-04** | **Fourier Flow** |
| A24 ACF r² Lag-1 Error ↓ | 0.1665 ± 0.0017 | 0.1719 ± 0.0626 | **0.0461 ± 7.01e-04** | **Fourier Flow** |
| **— Vol —** | | | | |
| A25 Mean RMSE ↓ | 1.301 ± 0.2776 | **0.7385 ± 0.4552** | 0.9000 ± 0.8807 | **TimeGAN** |
| A26 Return Std Error ↓ | 0.2492 ± 0.0018 | 0.1519 ± 0.0888 | **0.0058 ± 0.0028** | **Fourier Flow** |
| A27 Log-Return Std Error ↓ | 0.0030 ± 1.20e-05 | 0.0017 ± 7.78e-04 | **6.70e-05 ± 6.60e-05** | **Fourier Flow** |
| A28 Kurtosis Ratio (→ 1) | **1.989 ± 0.0182** | −1.095 ± 3.525 | 3.039 ± 0.7605 | **SBTS** |
| A29 Sigma Mean Error ↓ | 0.0440 ± 1.84e-04 | 0.0307 ± 0.0089 | **0.0026 ± 8.77e-04** | **Fourier Flow** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 3.276 ± 0.0637 | **0.3534 ± 0.1253** | 1.367 ± 0.4499 | **TimeGAN** |
| A31 Rolling Vol KS (w=5) ↓ | 0.3435 ± 6.43e-04 | 0.2540 ± 0.1093 | **0.0740 ± 0.0014** | **Fourier Flow** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 6.00e-06 | 8.97e-04 ± 8.69e-04 | **6.88e-04 ± 9.20e-05** | **Fourier Flow** |
| **— Heston Spec —** | | | | |
| A33 Teacher-Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | 7.85e-04 ± 0.0038 | **SBTS** |
| A34 Teacher-Sigma RMSE ↓ | 0.0955 ± 9.10e-05 | 0.1183 ± 0.0184 | **0.0894 ± 0.0013** | **Fourier Flow** |
| PS-MC CRPS H=32 ↓ | 2.761 ± 0.004 | 3.087 ± 0.340 | **2.742 ± 0.027** | **Fourier Flow** |
| PS-MC CRPS H=64 ↓ | **3.900 ± 0.008** | 4.372 ± 0.431 | 3.992 ± 0.106 | **SBTS** |
| Training (8 192×128) | — (no training) | ~6.5 min / A100 | ~8.2 min / CPU | **SBTS** |
| Generation (8 192×128) | ~6.3 min / 64 CPUs | **<1 s / A100** | ~1.5 s / CPU | **TimeGAN** |

> **A33 Teacher-Sigma Corr**: floor = 0.614 (not 1.0) — 5-step rolling QV is a noisy estimator of
> instantaneous variance vₜ. Neither SBTS (0.005) nor TimeGAN (0.002) preserves stochastic volatility.
>
> **A28 Kurtosis Ratio**: target = 1.0. SBTS (1.989) closer to 1 than TimeGAN (−1.095, bad seeds collapse
> kurtosis sign). |SBTS−1| = 0.989 vs |TimeGAN−1| = 2.095.

**Fourier Flow wins 27/38, SBTS wins 6/38, TimeGAN wins 5/38** (on A1-A34 + PS-MC, excluding training/generation rows).

**Interpretation:**
- **Fourier Flow dominates marginal, distributional and temporal-ACF fidelity** (A2–A6, A8, A9, A11–A14, A16, A17, A21–A24, A26–A27, A29, A31, A32, A34): the explicit-likelihood flow trained in the frequency domain fits the full spectral covariance, so log-return tails (A2/A3/A4), MMD (A6/A8/A9), autocorrelation (A21–A24), and vol moments (A26/A27/A29/A32) are all tighter than either baseline — often by an order of magnitude (A27: 6.7e-05 vs 1.7e-03; A26: 0.006 vs 0.15).
- **SBTS keeps the fat-tail extremes and the Heston teacher signal** (A1 kurtosis, A15 skew, A28 kurtosis ratio, A33 teacher-sigma corr): its kernel construction preserves the sign and magnitude of higher moments that the flow slightly over-disperses (FF kurtosis ratio 3.04 vs SBTS 1.99, target 1.0).
- **TimeGAN keeps three structural metrics and the GRU discriminator** (A10 terminal SWD, A18-GRU, A20 covariance, A25 mean-price RMSE, A30 cross-sectional vol): the recurrent generator still captures multi-step price-level covariance (A20: 18 vs FF 64 vs SBTS 145) better than the per-frequency flow.
- **A18 discriminative**: FF is least separable by the MLP (0.005) — its marginals are near-perfect — while TimeGAN edges the GRU discriminator (0.008 vs FF 0.009), i.e. the two are within noise on the temporal probe.
- **Path Shadowing MC**: FF wins at H=32 (CRPS 2.74 vs SBTS 2.76 vs TimeGAN 3.09) — its tighter, better-calibrated pool yields marginally better short-horizon nearest-neighbour forecasts; SBTS reclaims H=64 (3.90 vs FF 3.99) at the longer horizon.
- **Cross-seed stability**: FF is extremely stable on the moment metrics (A19-GRU std=7.6e-06, A21 std=5.5e-04) but shows real spread on A20 covariance (std=38, seed-2 outlier 138) and A25 mean RMSE (std=0.88); SBTS remains the most deterministic overall (A1 std=0.006).

---

## B — Curve-Shape Metrics Cross-Method Comparison — Heston (mean ± std, 5 seeds)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we
build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = **sum** of the three
  seed-means; combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (**quadrature**).
- **% err row**: the **function-level MAPE on the curve L itself**, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 — one division. The
  derivative / 2nd-derivative MAPE is **excluded** (near-zero true differences make it explode into meaningless 10⁴-% values); combined std = **sample std across the 5 seeds**. Bold marks the lower % error.

↓ lower is better for both rows. Histogram bins use real-data [0.5th, 99.5th]-percentile edges.
**Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Fourier Flow | Perfect | Winner |
|------|---------|:----:|:-------:|:------------:|:------:|:------:|
| **Log-return histogram** | MSE   | 12.14 ± 0.16 | 144.2 ± 120.6 | **2.847 ± 0.141** | 0 | **Fourier Flow** |
|                          | % err | 38.98% ± 0.132% | 33.42% ± 6.512% | **9.072% ± 0.571%** | 0 | |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | 7.09e-6 ± 3.3e-6 | **4.43e-7 ± 6.6e-8** | 0 | **Fourier Flow** |
|                          | % err | 21.27% ± 0.364% | 34.29% ± 11.19% | **9.363% ± 2.272%** | 0 | |
| **ACF \|r\| lags 1–20**  | MSE   | 4.57e-3 ± 3.7e-5 | 1.05e-2 ± 8.5e-3 | **1.30e-3 ± 3.8e-5** | 0 | **Fourier Flow** |
|                          | % err | 143% ± 1.580% | 164% ± 101% | **115.19% ± 1.926%** | 0 | |
| **ACF r² lags 1–20**     | MSE   | 5.17e-3 ± 5.7e-5 | 5.77e-3 ± 3.3e-3 | **9.43e-4 ± 3.5e-5** | 0 | **Fourier Flow** |
|                          | % err | 160% ± 1.615% | **110% ± 60.72%** | 117.36% ± 2.638% | 0 | |
| **Rolling vol histogram**| MSE   | 1227.3 ± 5.1 | 439.3 ± 216.7 | **92.44 ± 8.16** | 0 | **Fourier Flow** |
|                          | % err | 84.04% ± 0.124% | 56.06% ± 20.98% | **25.29% ± 3.210%** | 0 | |
| **Tail survival**        | MSE   | 5.74e-3 ± 6.6e-5 | 1.17e-2 ± 9.2e-3 | **5.30e-4 ± 4.6e-5** | 0 | **Fourier Flow** |
|                          | % err | 26.48% ± 0.114% | 23.60% ± 6.040% | **5.759% ± 0.237%** | 0 | |

> **Reading the two rows**: the **MSE** row is an absolute squared-error on the curve (+ its slope +
> its curvature); the **% err** row is the function-level MAPE of the curve L only. It stays in a sane
> range (≈ 21–164%): the ACF plots are largest because the true ACF ≈ 0.05 sits near zero, so any
> deviation is a big *relative* error — a property of the curve, not a bug. The ill-posed derivative
> MAPE is excluded, so the figures no longer explode to 10⁴-%.
>
> **Log-return histogram**: SBTS (MSE 12.1) much smaller than TimeGAN (144.2) — kernel smoothing closely
> preserves marginal returns. TimeGAN std=120.6 is driven by a seed-2 collapse (504.5 vs 11–170).
>
> **QQ plot**: TimeGAN wins on MSE (7.1e-6 vs 8.9e-6); SBTS wins on % err (both reproduce QQ shape closely).
>
> **Rolling vol histogram**: with real-data bins, SBTS's near-constant vol produces high MSE (1227 vs 439);
> TimeGAN wins here.
>
> **Tail survival**: SBTS wins decisively on both rows.

**Fourier Flow wins B: 6/6 plots on MSE.** The spectral objective fits every curve-shape diagnostic
tighter than either baseline. On the relative (% err) measure Fourier Flow is also lowest on 5 of 6
plots; TimeGAN edges it only on ACF r² (110% < 117.36%), where near-zero true autocorrelation inflates
the relative error. Each value is computed over the same **5 seeds** per method.

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
| Fourier Flow | [`Heston/FourierFlow/`](Heston/FourierFlow/) | [`../methods/FourierFlow/`](../methods/FourierFlow/) |
| Perfect recovery (floor) | — | [`../methods/perfect_recovery/`](../methods/perfect_recovery/) |

---

## Methods

### TimeGAN — Time-series Generative Adversarial Network
**Paper:** Yoon, Jarrett, van der Schaar — *Time-series GAN* — NeurIPS 2019, [arXiv:2010.00782](https://arxiv.org/abs/2010.00782)
**Code:** [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) — PyTorch reimplementation in this repo

TimeGAN is a **neural GAN** with five interacting GRU components:
- **Embedder + Recovery** (3-layer GRU, hidden=24): maps price paths ↔ latent embedding space
- **Generator** (3-layer GRU): generates latent sequences from Gaussian noise
- **Supervisor** (2-layer GRU): enforces step-by-step temporal consistency in latent space
- **Discriminator** (3-layer GRU): distinguishes real from generated latent sequences

**Training**: 3-phase adversarial, 20 000 steps (5 k embed → 5 k supervisor → 10 k joint).
**Hardware**: GPU required (A100 80 GB). ~6–8 min/seed.
**Generation**: Milliseconds (GRU forward pass). Sequences start near S₀=100 via internal min-max denorm.

### SBTS — Schrödinger Bridge Time Series
**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)
**Code:** [alexouadi/SBTS](https://github.com/alexouadi/SBTS) — Numba-accelerated reimplementation in this repo

SBTS is a **non-parametric kernel method** with no neural network and no training:
- Estimates the Schrödinger-bridge drift from training data using a compact quartic kernel K_h
- Simulates paths via Euler-Maruyama with the estimated drift (N_pi=200 substeps per interval)
- Markovian order K=1: weight of path m depends only on its most recent state X_i^m (valid for Heston)
- Internally operates on **scaled log-returns** R̃ = R × √Δt / σ(R) — not on prices or log-prices — then reconstructs prices: S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])

**Generation**: No training phase. ~6.3 min/seed with 64 CPU workers.
**Hardware**: CPU-only (Numba JIT). GPUs only used for A13/A14 metric evaluation.

### Fourier Flow — Generative Time-series Modeling with Fourier Flows
**Paper:** Alaa, Chan, van der Schaar — *Generative Time-series Modeling with Fourier Flows* — ICLR 2021, [OpenReview](https://openreview.net/forum?id=PpshD0AXfA)
**Code:** [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) — released-code-as-is reimplementation in this repo

Fourier Flow is an **explicit-likelihood normalizing flow that operates in the frequency domain**:
- Applies a **Discrete Fourier Transform** to each path, then a chain of invertible spectral filters (3 flows)
- Each **SpectralFilter** is an MLP (hidden=200) coupling layer acting on the real/imaginary spectral bins
- Trained by **direct negative-log-likelihood** minimisation (loss `(−log_pz − log_jacob).mean()`), full-batch Adam + ExponentialLR (γ=0.999), **1000 epochs**, **CPU-only** (numpy.fft)
- Inverts the flow and applies the **inverse DFT** to sample new price paths

**Two numerical guards** make training finite on Heston (paths start at a deterministic S₀=100, so the spectral covariance is near-singular at the DC bin): a **zero-std spectral-bin clamp** (necessary but not sufficient) and a **gradient clip = 1.0** (the actual fix that removes the NaN blow-up). See [`Heston/FourierFlow/README.md`](Heston/FourierFlow/README.md).

**Training**: ~8.2 min/seed (490 s, CPU). **Generation**: ~1.5 s/seed. **Hardware**: CPU-only; GPUs only used for A13/A14 metric evaluation.

### Perfect recovery — reproducible floor
A row-shuffled copy of the real dataset (`S_real[rng.permutation(N)]`, one permutation per seed). Because a
permutation preserves every column-wise marginal exactly, most A-metrics and all B-metrics hit **0**; the
residual non-zero floors (A1–A6 MMD/SWD, A13/A14 learned scores, A15/A21 = 0.614) are pure finite-sample
noise. This is the single source of truth for every "Perfect floor" column in the repo —
see [`../methods/perfect_recovery/`](../methods/perfect_recovery/).

---

## Key differences

| Aspect | TimeGAN | SBTS | Fourier Flow |
|--------|:-------:|:----:|:------------:|
| **Type** | Neural GAN (5 GRU components) | Non-parametric kernel estimator | Explicit-likelihood normalizing flow (frequency domain) |
| **Learnable parameters** | ~120 k (GRU weights) | **0** (no parameters) | ~360 k (3 spectral-filter MLPs, hidden=200) |
| **Training time / seed** | ~6–8 min (A100 GPU) | No training | ~8.2 min (CPU, 1000 epochs) |
| **Generation time / seed** | <1 s (GPU inference) | ~6.3 min (64 CPU workers) | ~1.5 s (CPU inverse flow + iDFT) |
| **Temporal memory** | Full (GRU sees all past steps) | **Markov-1 only** | Global (per-frequency spectral coupling) |
| **Internal representation** | Latent embeddings (min-max) | Scaled log-returns R̃ | DFT spectral bins (real/imag) |
| **Final output** | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) |
| **Cross-seed stability** | Moderate (GAN variance) | **High** (deterministic kernel) | High on moments, moderate on covariance |
| **Scales to long T** | Well (RNN) | Degrades (K=1 insufficient) | Well (fixed spectral size) |
| **Hyperparameter sensitivity** | Many (arch, lr, steps) | One critical: h (bandwidth) | Few (n_flows, hidden, grad-clip guard) |
| **Training objective** | Adversarial + supervised | Schrödinger-bridge drift (closed-form) | **Exact negative log-likelihood** |
