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

| Metric | SBTS | TimeGAN | Winner |
|--------|:----:|:-------:|:------:|
| **— Fat Tail —** | | | |
| A1 Kurtosis Error ↓ | **0.1187 ± 0.0060** | 2.955 ± 2.099 | **SBTS** |
| A2 \|r\| q95 Error ↓ | 0.0063 ± 3.00e-05 | **0.0032 ± 0.0018** | **TimeGAN** |
| A3 \|r\| q99 Error ↓ | 0.0098 ± 4.80e-05 | **0.0043 ± 0.0028** | **TimeGAN** |
| A4 Tail QQ Error ↓ | 0.0062 ± 2.60e-05 | **0.0034 ± 0.0015** | **TimeGAN** |
| A5 Hill Tail Index Error ↓ | **9.499 ± 0.3457** | 36.885 ± 17.053 | **SBTS** |
| **— Distribution —** | | | |
| A6 Path MMD² ↓ | **0.0110 ± 0.0026** | 0.0165 ± 0.0127 | **SBTS** |
| A7 Terminal MMD² ↓ | **0.0100 ± 0.0036** | 0.0267 ± 0.0192 | **SBTS** |
| A8 Increment MMD² ↓ | **0.0071 ± 2.47e-04** | 0.0077 ± 0.0041 | **SBTS** |
| A9 Volatility MMD ↓ | **0.3038 ± 0.0071** | 0.3789 ± 0.2430 | **SBTS** |
| A10 Terminal SWD ↓ | 3.539 ± 0.7368 | **2.658 ± 0.8567** | **TimeGAN** |
| A11 Path SWD ↓ | 2.415 ± 0.4104 | **1.417 ± 0.4914** | **TimeGAN** |
| A12 RV Law Loss ↓ | 2.148 ± 0.0074 | **1.551 ± 0.3788** | **TimeGAN** |
| A13 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | **TimeGAN** |
| A14 KS Log-returns ↓ | **0.0534 ± 3.62e-04** | 0.0848 ± 0.0374 | **SBTS** |
| A15 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 0.0028 ± 1.20e-05 | **0.0025 ± 6.43e-04** | **TimeGAN** |
| A17 Terminal Price KS ↓ | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | **SBTS** |
| **— Adversarial —** | | | |
| A18 Disc Score GRU ↓ | 0.2755 ± 0.2166 | **0.0077 ± 0.0050** | **TimeGAN** |
| A18 Disc Score MLP ↓ | **0.0079 ± 0.0049** | 0.1031 ± 0.0395 | **SBTS** |
| **— Predictive —** | | | |
| A19 Pred Score GRU ↓ | 0.0586 ± 5.90e-05 | **0.0574 ± 0.0019** | **TimeGAN** |
| A19 Pred Score MLP ↓ | 0.0583 ± 2.55e-04 | **0.0570 ± 0.0012** | **TimeGAN** |
| **— Temporal —** | | | |
| A20 Covariance Error ↓ | 145.35 ± 4.886 | **17.751 ± 6.707** | **TimeGAN** |
| A21 ACF \|r\| Error (lags) ↓ | **0.0596 ± 4.70e-04** | 0.1252 ± 0.0674 | **SBTS** |
| A22 ACF r² Error (lags) ↓ | **0.0619 ± 5.08e-04** | 0.0839 ± 0.0348 | **SBTS** |
| A23 ACF \|r\| Lag-1 Error ↓ | **0.1437 ± 0.0012** | 0.2264 ± 0.1034 | **SBTS** |
| A24 ACF r² Lag-1 Error ↓ | **0.1665 ± 0.0017** | 0.1719 ± 0.0626 | **SBTS** |
| **— Vol —** | | | |
| A25 Mean RMSE ↓ | 1.301 ± 0.2776 | **0.7385 ± 0.4552** | **TimeGAN** |
| A26 Return Std Error ↓ | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | **TimeGAN** |
| A27 Log-Return Std Error ↓ | 0.0030 ± 1.20e-05 | **0.0017 ± 7.78e-04** | **TimeGAN** |
| A28 Kurtosis Ratio (→ 1) | **1.989 ± 0.0182** | −1.095 ± 3.525 | **SBTS** |
| A29 Sigma Mean Error ↓ | 0.0440 ± 1.84e-04 | **0.0307 ± 0.0089** | **TimeGAN** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 3.276 ± 0.0637 | **0.3534 ± 0.1253** | **TimeGAN** |
| A31 Rolling Vol KS (w=5) ↓ | 0.3435 ± 6.43e-04 | **0.2540 ± 0.1093** | **TimeGAN** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 6.00e-06 | **8.97e-04 ± 8.69e-04** | **TimeGAN** |
| **— Heston Spec —** | | | |
| A33 Teacher-Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** |
| A34 Teacher-Sigma RMSE ↓ | **0.0955 ± 9.10e-05** | 0.1183 ± 0.0184 | **SBTS** |
| PS-MC CRPS H=32 ↓ | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓ | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |
| Training (8 192×128) | — (no training) | ~6.5 min / A100 | **SBTS** |
| Generation (8 192×128) | ~6.3 min / 64 CPUs | **<1 s / A100** | **TimeGAN** |

> **A33 Teacher-Sigma Corr**: floor = 0.614 (not 1.0) — 5-step rolling QV is a noisy estimator of
> instantaneous variance vₜ. Neither SBTS (0.005) nor TimeGAN (0.002) preserves stochastic volatility.
>
> **A28 Kurtosis Ratio**: target = 1.0. SBTS (1.989) closer to 1 than TimeGAN (−1.095, bad seeds collapse
> kurtosis sign). |SBTS−1| = 0.989 vs |TimeGAN−1| = 2.095.

**SBTS wins 19/38, TimeGAN wins 19/38** (on A1-A34 + PS-MC, excluding training/generation rows).

**Interpretation:**
- **SBTS wins on distributional fidelity** (A1, A5–A9, A14–A15, A17): the kernel method accurately matches marginal return distributions. MMD, kurtosis, KS, skewness, and ACF errors are consistently lower.
- **TimeGAN wins on temporal structure** (A10–A13, A20, A25–A27, A29–A32): the GRU captures multi-step covariance that Markov-1 SBTS cannot. A20 (covariance error) is the starkest gap: **145 vs 18**. A12 (RV law loss) also strongly favours TimeGAN (1.55 vs 2.15).
- **A18 GRU discriminative**: TimeGAN paths are harder to separate by the GRU (0.008 vs 0.276). SBTS is more separable because its near-constant rolling vol is a clear distinguishing feature (see A31).
- **A18 MLP discriminative**: SBTS wins (0.008 vs 0.103) — MLP without temporal context can't exploit the vol clustering signal, so SBTS looks more realistic at the moment-matching level.
- **Path Shadowing MC**: SBTS wins decisively (CRPS 2.76 vs 3.09 at H=32) — richer and more faithful retrieval pool → better nearest-neighbour forecasts.
- **Cross-seed stability**: SBTS is very stable (A1 std=0.006, A20 std=4.9) vs TimeGAN which can vary widely (A1 std=2.1, A20 std=6.7, A10 std=0.86).

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

| Plot | Measure | SBTS | TimeGAN | Perfect | Winner |
|------|---------|:----:|:-------:|:------:|:------:|
| **Log-return histogram** | MSE   | **12.14 ± 0.16** | 144.2 ± 120.6 | 0 | **SBTS** |
|                          | % err | 38.98% ± 0.132% | **33.42% ± 6.512%** | 0 | |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | **7.09e-6 ± 3.3e-6** | 0 | **TimeGAN** |
|                          | % err | **21.27% ± 0.364%** | 34.29% ± 11.19% | 0 | |
| **ACF \|r\| lags 1–20**  | MSE   | **4.57e-3 ± 3.7e-5** | 1.05e-2 ± 8.5e-3 | 0 | **SBTS** |
|                          | % err | **143% ± 1.580%** | 164% ± 101% | 0 | |
| **ACF r² lags 1–20**     | MSE   | **5.17e-3 ± 5.7e-5** | 5.77e-3 ± 3.3e-3 | 0 | **SBTS** |
|                          | % err | 160% ± 1.615% | **110% ± 60.72%** | 0 | |
| **Rolling vol histogram**| MSE   | 1227.3 ± 5.1 | **439.3 ± 216.7** | 0 | **TimeGAN** |
|                          | % err | 84.04% ± 0.124% | **56.06% ± 20.98%** | 0 | |
| **Tail survival**        | MSE   | **5.74e-3 ± 6.6e-5** | 1.17e-2 ± 9.2e-3 | 0 | **SBTS** |
|                          | % err | 26.48% ± 0.114% | **23.60% ± 6.040%** | 0 | |

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

**SBTS wins B: 4/6 plots on MSE. TimeGAN wins 2/6** (QQ plot, rolling-vol histogram). Each value is
computed over the same **5 seeds** per method.

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
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

### Perfect recovery — reproducible floor
A row-shuffled copy of the real dataset (`S_real[rng.permutation(N)]`, one permutation per seed). Because a
permutation preserves every column-wise marginal exactly, most A-metrics and all B-metrics hit **0**; the
residual non-zero floors (A1–A6 MMD/SWD, A13/A14 learned scores, A15/A21 = 0.614) are pure finite-sample
noise. This is the single source of truth for every "Perfect floor" column in the repo —
see [`../methods/perfect_recovery/`](../methods/perfect_recovery/).

---

## Key differences

| Aspect | TimeGAN | SBTS |
|--------|:-------:|:----:|
| **Type** | Neural GAN (5 GRU components) | Non-parametric kernel estimator |
| **Learnable parameters** | ~120 k (GRU weights) | **0** (no parameters) |
| **Training time / seed** | ~6–8 min (A100 GPU) | No training |
| **Generation time / seed** | <1 s (GPU inference) | ~6.3 min (64 CPU workers) |
| **Temporal memory** | Full (GRU sees all past steps) | **Markov-1 only** |
| **Internal representation** | Latent embeddings (min-max) | Scaled log-returns R̃ |
| **Final output** | Price paths (S_t) | Price paths (S_t) |
| **Cross-seed stability** | Moderate (GAN variance) | **High** (deterministic kernel) |
| **Scales to long T** | Well (RNN) | Degrades (K=1 insufficient) |
| **Hyperparameter sensitivity** | Many (arch, lr, steps) | One critical: h (bandwidth) |
