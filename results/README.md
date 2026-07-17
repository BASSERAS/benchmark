# Results — Methods Comparison on Heston

All methods are evaluated on the same dataset:
**8 192 Heston price paths, seq\_len = 128**
(μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250)

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

---

## Cross-Method Comparison A1–A20 — Heston (mean ± std, 5 seeds)

↓ = lower is better. ↑ = higher is better. **Bold** = best method per metric.
"Perfect recovery" = two independent halves of the real dataset evaluated against each other
(empirical floor — what a perfect generator would achieve with finite samples).

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner |
|--------|:---------------:|:----:|:-------:|:------:|
| A1  Path MMD² ↓ | 0.0018 ± 0.0002 | **0.0112 ± 0.0011** | 0.0181 ± 0.0147 | **SBTS** |
| A2  Terminal MMD² ↓ | 0.0016 ± 0.0002 | **0.0102 ± 0.0014** | 0.0308 ± 0.0229 | **SBTS** |
| A3  Increment MMD² ↓ | 0.0008 ± 0.0000 | **0.0069 ± 0.0005** | 0.0077 ± 0.0039 | **SBTS** |
| A4  Volatility MMD ↓ | 0.0077 ± 0.0006 | **0.2964 ± 0.0126** | 0.3933 ± 0.2553 | **SBTS** |
| A5  Terminal SWD ↓ | 0.7635 ± 0.1174 | 3.7097 ± 0.3209 | **3.1284 ± 0.9227** | **TimeGAN** |
| A6  Path SWD ↓ | 0.5542 ± 0.0624 | 2.5335 ± 0.2212 | **1.6343 ± 0.5763** | **TimeGAN** |
| A7  Cov Error ↓ | 4.76 ± 2.50 | 145.35 ± 4.89 | **17.75 ± 6.71** | **TimeGAN** |
| A8  Mean RMSE ↓ | 0.1400 ± 0.1303 | 1.3013 ± 0.2776 | **0.7385 ± 0.4552** | **TimeGAN** |
| A9  Std Error ↓ | 0.0048 ± 0.0031 | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | **TimeGAN** |
| A10 Kurtosis Error ↓ | 0.0172 ± 0.0155 | **0.1187 ± 0.0060** | 2.9545 ± 2.0988 | **SBTS** |
| A11 ACF \|r\| Error ↓ | 0.0017 ± 0.0006 | **0.0596 ± 0.0005** | 0.1252 ± 0.0674 | **SBTS** |
| A12 ACF r² Error ↓ | 0.0014 ± 0.0006 | **0.0619 ± 0.0005** | 0.0839 ± 0.0348 | **SBTS** |
| A13 Disc GRU ↓ | 0.0128 ± 0.0068 | 0.2740 ± 0.2208 | **0.0099 ± 0.0084** | **TimeGAN** |
| A13 Disc MLP ↓ | 0.0080 ± 0.0081 | **0.0063 ± 0.0038** | 0.0921 ± 0.0463 | **SBTS** |
| A14 Pred GRU ↓ | 0.0564 ± 0.0022 | 0.0586 ± 0.0000 | **0.0570 ± 0.0013** | **TimeGAN** |
| A14 Pred MLP ↓ | 0.0565 ± 0.0022 | 0.0582 ± 0.0002 | **0.0573 ± 0.0015** | **TimeGAN** |
| A15 Sigma Corr ↑ | 0.6135 ± 0.0019 ⁽¹⁾ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** |
| A15 Sigma RMSE ↓ | 0.0653 ± 0.0002 ⁽²⁾ | **0.0955 ± 0.0001** | 0.1183 ± 0.0184 | **SBTS** |
| A16 Tail RMS ↓ | 0.0008 ± 0.0008 | 0.0428 ± 0.0002 | **0.0234 ± 0.0109** | **TimeGAN** |
| A17 Oracle MAE ↓ | 0.0097 ± 0.0000 | 0.0097 ± 0.0000 | 0.0097 ± 0.0000 | Tie |
| A18 Agent MAE ↓ | 0.0097 ± 0.0000 | 0.0106 ± 0.0000 | **0.0101 ± 0.0003** | **TimeGAN** |
| A19 Oracle-Agent Corr ↑ | −0.058 ± 0.430 ⁽³⁾ | −0.342 ± 0.171 | **−0.332 ± 0.306** | **TimeGAN** |
| A20 RV Law Loss ↓ | 0.0673 ± 0.0362 | 2.1482 ± 0.0074 | **1.5512 ± 0.3788** | **TimeGAN** |
| PS-MC CRPS H=32 ↓ | — | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓ | — | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |
| Training (8 192×128) | — | — (no training) | ~6.5 min / A100 | **SBTS** |
| Generation (8 192×128) | — | ~6.3 min / 64 CPUs | **<1 s / A100** | **TimeGAN** |

> ⁽¹⁾ **A15 Sigma Corr floor = 0.614** (not 1.0): 5-step rolling QV is a noisy estimator
> of instantaneous variance vₜ. Even for real Heston paths vs their own true variance, Pearson ρ ≈ 0.614.
> Neither SBTS (0.005) nor TimeGAN (0.002) preserves stochastic volatility — both are near zero.
>
> ⁽²⁾ **A15 Sigma RMSE floor = 0.065** (not 0): measurement noise from rolling QV creates an irreducible
> baseline even for real paths. Both SBTS (0.096) and TimeGAN (0.118) score **above** the floor,
> confirming the metric is well-calibrated. SBTS wins (lower RMSE = less vol mis-estimation).
>
> ⁽³⁾ **A19 Oracle-Agent Corr floor ≈ −0.06 ± 0.43**: Heston log-returns have near-zero autocorrelation,
> so AR(5) predictions are essentially white noise for both oracle and agent. The metric degenerates for
> Heston (both methods score near floor) but is meaningful for datasets with genuine temporal structure.

**SBTS wins 10/23, TimeGAN wins 12/23, Tie 1/23** (on A1-A20 + PS-MC, excluding training/generation rows).

**Interpretation:**
- **SBTS wins on distributional fidelity** (A1–A4, A10–A12): the kernel method accurately matches marginal return distributions. MMD, kurtosis, and ACF errors are consistently lower.
- **TimeGAN wins on temporal structure** (A5–A9, A20): the GRU captures multi-step covariance that Markov-1 SBTS cannot. A7 (covariance error) is the starkest gap: **145% vs 18%**. A20 (RV law loss) also strongly favours TimeGAN (1.55 vs 2.15).
- **A13 GRU discriminative**: TimeGAN paths are harder to separate by the GRU (0.010 vs 0.274). SBTS is more separable because its near-constant rolling vol is a clear distinguishing feature (see B11).
- **A13 MLP discriminative**: SBTS wins (0.006 vs 0.092) — MLP without temporal context can't exploit the vol clustering signal, so SBTS looks more realistic at the moment-matching level.
- **Path Shadowing MC**: SBTS wins decisively (CRPS 2.76 vs 3.09 at H=32) — richer and more faithful retrieval pool → better nearest-neighbour forecasts.
- **Cross-seed stability**: SBTS is very stable (A1 std=0.001, A7 std=4.9%) vs TimeGAN which can vary widely (A1 std=0.015, A7 std=6.7%, A10 std=2.1).

---

## Cross-Method Comparison B1–B14 Stylized Metrics — Heston (mean ± std, 5 seeds)

Extracted from the 8 diagnostic plot panels. Each metric quantifies a known stylized fact (Cont 2001).
↓ = lower is better (all B metrics). **Bold** = best method.

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner |
|--------|:---------------:|:----:|:-------:|:------:|
| B1  Mean Path RMSE | 0.1511 ± 0.0708 | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | **TimeGAN** |
| B2  Cross-Sect. Vol RMSE | 0.1355 ± 0.0735 | 3.2760 ± 0.0637 | **0.3534 ± 0.1253** | **TimeGAN** |
| B3  KS on Log-returns | 0.0018 ± 0.0009 | **0.0534 ± 0.0004** | 0.0848 ± 0.0374 | **SBTS** |
| B4  Skewness Error | 0.0060 ± 0.0048 | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | **SBTS** |
| B5  QQ RMSE (300-pt) | 0.0001 ± 0.0000 | 0.0028 ± 0.0000 | **0.0025 ± 0.0006** | **TimeGAN** |
| B6  Tail QQ Error | 0.0001 ± 0.0001 | 0.0062 ± 0.0000 | **0.0034 ± 0.0015** | **TimeGAN** |
| B7  ACF lag-1 \|r\| Err | 0.0018 ± 0.0016 | **0.1449 ± 0.0012** | 0.2282 ± 0.1042 | **SBTS** |
| B8  ARCH Persistence Err | 0.0011 ± 0.0005 | **0.0274 ± 0.0004** | 0.0591 ± 0.0359 | **SBTS** |
| B9  ACF lag-1 r² Err | 0.0017 ± 0.0014 | **0.1678 ± 0.0017** | 0.1732 ± 0.0631 | **SBTS** |
| B10 GARCH Persistence Err | 0.0010 ± 0.0006 | **0.0227 ± 0.0004** | 0.0328 ± 0.0151 | **SBTS** |
| B11 Rolling Vol KS | 0.0046 ± 0.0024 | 0.3435 ± 0.0006 | **0.2540 ± 0.1093** | **TimeGAN** |
| B12 Vol-of-Vol Error | 0.0000 ± 0.0000 | 0.0021 ± 0.0000 | **0.0009 ± 0.0009** | **TimeGAN** |
| B13 Terminal Price KS | 0.0145 ± 0.0043 | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | **SBTS** |
| B14 Hill Tail Index Err | 0.499 ± 0.610 | **9.499 ± 0.346** | 36.88 ± 17.05 | **SBTS** |

> **B2 Cross-Sect. Vol RMSE**: SBTS 3.28 vs TimeGAN 0.35. SBTS uses kernel bootstrap which inherits
> the full cross-sectional variance from training data — diverse paths but all with similar realised vols.
> TimeGAN latent space compresses vol variability, resulting in lower B2 but higher A7 Cov Error.
>
> **B7–B10** (ACF metrics): Heston true ACF(|r|, lag=1) ≈ +0.052, ACF(r², lag=1) ≈ +0.050.
> SBTS consistently wins because the kernel method preserves marginal return autocorrelations.
> TimeGAN often collapses to near-zero ACF in bad seeds (seeds 2, 4), missing the ARCH signature.
>
> **B11 Rolling Vol KS**: SBTS 0.344 vs TimeGAN 0.254 (floor 0.005). Both fail badly, but for different reasons.
> SBTS: kernel smoothing produces near-constant rolling vol. TimeGAN: vol clustering is erratic across seeds.
>
> **B14 Hill Tail Index Error**: SBTS 9.50 (stable ± 0.35) vs TimeGAN 36.88 (noisy ± 17.05).
> SBTS systematically underestimates tail heaviness (kernel smoothing attenuates extremes).
> TimeGAN seed-to-seed blowup (paths collapse or explode in bad seeds) creates extreme Hill estimates.

**SBTS wins B1-B14: 8/14. TimeGAN wins 6/14.**

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
