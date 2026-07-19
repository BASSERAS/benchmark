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
| A10 Kurtosis Error ↓ | **0.1187 ± 0.0060** | 2.9545 ± 2.0988 | **SBTS** |
| A17 Tail \|r\| Q95 Error ↓ | 0.0063 ± 0.0000 | **0.0032 ± 0.0018** | **TimeGAN** |
| A18 Tail \|r\| Q99 Error ↓ | 0.0098 ± 0.0000 | **0.0043 ± 0.0028** | **TimeGAN** |
| A30 Tail QQ Error ↓ | 0.0062 ± 0.0000 | **0.0034 ± 0.0015** | **TimeGAN** |
| A34 Hill Tail Index Err ↓ | **9.499 ± 0.346** | 36.88 ± 17.05 | **SBTS** |
| **— Distribution —** | | | |
| A1  Path MMD² ↓ | **0.0112 ± 0.0011** | 0.0181 ± 0.0147 | **SBTS** |
| A2  Terminal MMD² ↓ | **0.0102 ± 0.0014** | 0.0308 ± 0.0229 | **SBTS** |
| A3  Increment MMD² ↓ | **0.0069 ± 0.0005** | 0.0077 ± 0.0039 | **SBTS** |
| A4  Volatility MMD ↓ | **0.2964 ± 0.0126** | 0.3933 ± 0.2553 | **SBTS** |
| A5  Terminal SWD ↓ | 3.7097 ± 0.3209 | **3.1284 ± 0.9227** | **TimeGAN** |
| A6  Path SWD ↓ | 2.5335 ± 0.2212 | **1.6343 ± 0.5763** | **TimeGAN** |
| A24 RV Law Loss ↓ | 2.1482 ± 0.0074 | **1.5512 ± 0.3788** | **TimeGAN** |
| A25 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | **TimeGAN** |
| A27 KS Log-returns ↓ | **0.0534 ± 0.0004** | 0.0848 ± 0.0374 | **SBTS** |
| A28 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | **SBTS** |
| A29 QQ RMSE ↓ | 0.0028 ± 0.0000 | **0.0025 ± 0.0006** | **TimeGAN** |
| A33 Terminal Price KS ↓ | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | **SBTS** |
| **— Adversarial —** | | | |
| A13 Disc GRU ↓ | 0.2740 ± 0.2208 | **0.0099 ± 0.0084** | **TimeGAN** |
| A13 Disc MLP ↓ | **0.0063 ± 0.0038** | 0.0921 ± 0.0463 | **SBTS** |
| **— Predictive —** | | | |
| A14 Pred GRU ↓ | 0.0586 ± 0.0000 | **0.0570 ± 0.0013** | **TimeGAN** |
| A14 Pred MLP ↓ | 0.0582 ± 0.0002 | **0.0573 ± 0.0015** | **TimeGAN** |
| **— Temporal —** | | | |
| A7  Cov Error ↓ | 145.35 ± 4.89 | **17.75 ± 6.71** | **TimeGAN** |
| A11 ACF \|r\| Error ↓ | **0.0596 ± 0.0005** | 0.1252 ± 0.0674 | **SBTS** |
| A12 ACF r² Error ↓ | **0.0619 ± 0.0005** | 0.0839 ± 0.0348 | **SBTS** |
| A22 ACF lag-1 \|r\| Err ↓ | **0.1437 ± 0.0012** | 0.2264 ± 0.1034 | **SBTS** |
| A23 ACF lag-1 r² Err ↓ | **0.1665 ± 0.0017** | 0.1719 ± 0.0626 | **SBTS** |
| **— Vol —** | | | |
| A8  Mean RMSE ↓ | 1.3013 ± 0.2776 | **0.7385 ± 0.4552** | **TimeGAN** |
| A9  Std Error ↓ | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | **TimeGAN** |
| A16 Log-ret Std Error ↓ | 0.0030 ± 0.0000 | **0.0017 ± 0.0008** | **TimeGAN** |
| A19 Kurtosis Ratio (→ 1) | **1.989 ± 0.018** | −1.095 ± 3.525 | **SBTS** |
| A20 Sigma Mean Error ↓ | 0.0440 ± 0.0002 | **0.0307 ± 0.0089** | **TimeGAN** |
| A26 Vol Path RMSE ↓ | 3.2760 ± 0.0637 | **0.3534 ± 0.1253** | **TimeGAN** |
| A31 Rolling Vol KS ↓ | 0.3435 ± 0.0006 | **0.2540 ± 0.1093** | **TimeGAN** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 0.0000 | **0.0009 ± 0.0009** | **TimeGAN** |
| **— Heston Spec —** | | | |
| A15 Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** |
| A15 Sigma RMSE ↓ | **0.0955 ± 0.0001** | 0.1183 ± 0.0184 | **SBTS** |
| A21 Oracle Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** |
| PS-MC CRPS H=32 ↓ | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓ | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |
| Training (8 192×128) | — (no training) | ~6.5 min / A100 | **SBTS** |
| Generation (8 192×128) | ~6.3 min / 64 CPUs | **<1 s / A100** | **TimeGAN** |

> **A15/A21 Sigma Corr**: floor = 0.614 (not 1.0) — 5-step rolling QV is a noisy estimator of
> instantaneous variance vₜ. Neither SBTS (0.005) nor TimeGAN (0.002) preserves stochastic volatility.
>
> **A19 Kurtosis Ratio**: target = 1.0. SBTS (1.989) closer to 1 than TimeGAN (−1.095, bad seeds collapse
> kurtosis sign). |SBTS−1| = 0.989 vs |TimeGAN−1| = 2.095.

**SBTS wins 20/39, TimeGAN wins 19/39** (on A1-A34 + PS-MC, excluding training/generation rows).

**Interpretation:**
- **SBTS wins on distributional fidelity** (A1–A4, A10–A12, A27–A28): the kernel method accurately matches marginal return distributions. MMD, kurtosis, KS, skewness, and ACF errors are consistently lower.
- **TimeGAN wins on temporal structure** (A5–A9, A24–A26, A29–A32): the GRU captures multi-step covariance that Markov-1 SBTS cannot. A7 (covariance error) is the starkest gap: **145% vs 18%**. A24 (RV law loss) also strongly favours TimeGAN (1.55 vs 2.15).
- **A13 GRU discriminative**: TimeGAN paths are harder to separate by the GRU (0.010 vs 0.274). SBTS is more separable because its near-constant rolling vol is a clear distinguishing feature (see A31).
- **A13 MLP discriminative**: SBTS wins (0.006 vs 0.092) — MLP without temporal context can't exploit the vol clustering signal, so SBTS looks more realistic at the moment-matching level.
- **Path Shadowing MC**: SBTS wins decisively (CRPS 2.76 vs 3.09 at H=32) — richer and more faithful retrieval pool → better nearest-neighbour forecasts.
- **Cross-seed stability**: SBTS is very stable (A1 std=0.001, A7 std=4.9%) vs TimeGAN which can vary widely (A1 std=0.015, A7 std=6.7%, A10 std=2.1).

---

## B — Curve-Shape Metrics Cross-Method Comparison — Heston (mean ± std, 5 seeds)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we
build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means;
  combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100. Combined the same way.

↓ lower is better for both rows. Histogram bins use real-data [0.5th, 99.5th]-percentile edges.
**Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Winner |
|------|---------|:----:|:-------:|:------:|
| **Log-return histogram** | MSE   | **12.14 ± 0.16** | 144.2 ± 120.6 | **SBTS** |
|                          | % err | **14264% ± 8212%** | 75440% ± 24681% | |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | **7.09e-6 ± 3.3e-6** | **TimeGAN** |
|                          | % err | **113.1% ± 5.8%** | 157.9% ± 27.2% | |
| **ACF \|r\| lags 1–20**  | MSE   | **4.57e-3 ± 3.7e-5** | 1.05e-2 ± 8.5e-3 | **SBTS** |
|                          | % err | **1269.9% ± 40.0%** | 2342.1% ± 1351.8% | |
| **ACF r² lags 1–20**     | MSE   | **5.17e-3 ± 5.7e-5** | 5.77e-3 ± 3.3e-3 | **SBTS** |
|                          | % err | **1603.5% ± 142.8%** | 4578.1% ± 4266.4% | |
| **Rolling vol histogram**| MSE   | 1227.3 ± 5.1 | **439.3 ± 216.7** | **TimeGAN** |
|                          | % err | **715.9% ± 20.2%** | 882.8% ± 410.8% | |
| **Tail survival**        | MSE   | **5.74e-3 ± 6.6e-5** | 1.17e-2 ± 9.2e-3 | **SBTS** |
|                          | % err | **17151% ± 910%** | 24327% ± 11324% | |

> **Reading the two rows**: the **MSE** row is an absolute squared-error on the curve (+ its slope +
> its curvature); the **% err** row is a relative error and blows up when the real curve passes through
> near-zero values (empty histogram bins, tail-survival ≈ 0, near-zero ACF lags), so triple-digit-plus
> percentages are expected and are a property of the curve, not a bug.
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
