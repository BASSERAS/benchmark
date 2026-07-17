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

### SBTS — Score-Based Time Series (Schrödinger Bridge)
**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)
**Code:** [alexouadi/SBTS](https://github.com/alexouadi/SBTS) — Numba-accelerated reimplementation in this repo

SBTS is a **non-parametric kernel method** with no neural network and no training:
- Estimates the Schrödinger-bridge drift from training data using a compact quartic kernel K_h
- Simulates paths via Euler-Maruyama with the estimated drift (N_pi=200 substeps per interval)
- Markovian order K=1: weight of path m depends only on its most recent state X_i^m (valid for Heston)
- Internally operates on **scaled log-returns** R̃ = R × √Δt / σ(R) — not on prices or log-prices — then reconstructs prices: S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])

**Generation**: No training phase. ~6.3 min/seed with 64 CPU workers (one path per worker task).
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

## Cross-Method Comparison — Heston (mean ± std, 5 seeds)

↓ = lower is better. ↑ = higher is better. **Bold** = best method for that metric.
"Perfect recovery" = two independent halves of the real dataset evaluated against each other
(i.e., what you'd see if the generator perfectly recovered the true distribution).

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner |
|--------|:---------------:|:----:|:-------:|:------:|
| A1  Path MMD² ↓      | 0.0018 ± 0.0002 | **0.0110 ± 0.0016** | 0.0180 ± 0.0147 | **SBTS** |
| A2  Terminal MMD² ↓  | 0.0016 ± 0.0002 | **0.0090 ± 0.0035** | 0.0296 ± 0.0235 | **SBTS** |
| A3  Increment MMD² ↓ | 0.0008 ± 0.0000 | **0.0071 ± 0.0005** | 0.0078 ± 0.0037 | **SBTS** |
| A4  Volatility MMD ↓ | 0.0082 ± 0.0006 | **0.3125 ± 0.0176** | 0.3798 ± 0.2351 | **SBTS** |
| A5  Terminal SWD ↓   | 0.763 ± 0.117   | 3.465 ± 0.588 | **2.850 ± 1.079** | **TimeGAN** |
| A6  Path SWD ↓       | 0.554 ± 0.062   | 2.497 ± 0.288 | **1.501 ± 0.583** | **TimeGAN** |
| A7  Cov Error (%) ↓  | 4.76 ± 2.50     | 145.35 ± 4.89 | **17.75 ± 6.71** | **TimeGAN** |
| A8  Mean RMSE ↓      | 0.140 ± 0.130   | 1.301 ± 0.278 | **0.739 ± 0.455** | **TimeGAN** |
| A9  Std Error ↓      | 0.0048 ± 0.0031 | 0.249 ± 0.002 | **0.152 ± 0.089** | **TimeGAN** |
| A10 Kurtosis Error ↓ | 0.017 ± 0.016   | **0.119 ± 0.006** | 2.955 ± 2.099 | **SBTS** |
| A11 ACF Abs Error ↓  | 0.0017 ± 0.0007 | **0.057 ± 0.001** | 0.134 ± 0.073 | **SBTS** |
| A12 ACF Sq Error ↓   | 0.0015 ± 0.0006 | **0.062 ± 0.001** | 0.092 ± 0.039 | **SBTS** |
| A13 Disc GRU ↓       | 0.0042 ± 0.0048 | **0.029 ± 0.028** | 0.050 ± 0.034 | **SBTS** |
| A13 Disc MLP ↓       | 0.0112 ± 0.0079 | **0.071 ± 0.008** | 0.151 ± 0.142 | **SBTS** |
| A14 Pred GRU ↓       | 0.0085 ± 0.0001 | 0.0091 ± 0.0000 | **0.0087 ± 0.0002** | ≈ tie |
| A14 Pred MLP ↓       | 0.0087 ± 0.0002 | 0.0093 ± 0.0006 | **0.0090 ± 0.0005** | ≈ tie |
| A15 Sigma Corr ↑     | 0.505 ± 0.001 ⁽¹⁾ | 0.0011 ± 0.0035 | **0.0031 ± 0.0101** | ≈ tie |
| A15 Sigma RMSE ↓     | 1.054 ± 0.002 ⁽²⁾ | **0.821 ± 0.002** | 0.966 ± 0.124 | **SBTS** |
| PS-MC CRPS H=32 ↓    | — | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓    | — | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |
| A16 Tail Survival ↓  | 0.0009 ± 0.0005 | 0.0367 ± 0.0002 | **0.0216 ± 0.0111** | **TimeGAN** |
| Training (8 192×128) | — | — (no training) | **~6.5 min / A100** | **SBTS** |
| Generation (8 192×128) | — | ~6.3 min / 64 CPUs | **<1 s / A100** | **TimeGAN** |

> ⁽¹⁾ **A15 Sigma Corr floor = 0.505** (not 1.0): rolling-window QV (5 steps) is a noisy estimator
> of instantaneous variance. Even for real Heston paths vs their own true variance path, Pearson ρ ≈ 0.5
> due to measurement noise. Neither SBTS (0.001) nor TimeGAN (0.003) preserves instantaneous variance.
>
> ⁽²⁾ **A15 Sigma RMSE floor = 1.054** — paradoxically, SBTS (0.821) and TimeGAN (0.966) both score
> **below** the floor. This is due to **variance compression**: generated paths have narrower return
> distributions than real Heston paths, making the RMSE of the inferred variance artificially small.
> A lower-than-floor Sigma RMSE is a warning sign, not a win.

**SBTS wins 12/21, TimeGAN wins 7/21, 2 ties.**

**Interpretation:**
- **SBTS wins on distribution matching** (A1–A4, A10–A12): the kernel method matches marginal and return distributions. MMD and ACF errors are consistently lower.
- **TimeGAN wins on temporal structure** (A5–A9): the GRU captures multi-step covariance that Markov-1 SBTS cannot. A7 (covariance error) is the starkest gap: **145% vs 18%**.
- **Discriminative scores** (A13): SBTS paths are harder to tell from real (0.029 vs 0.050 GRU) — better statistical fidelity at sample level.
- **Path Shadowing MC**: SBTS wins decisively (CRPS 2.76 vs 3.09 at H=32) — richer kernel pool → better nearest-neighbour forecasts.
- **Cross-seed stability**: SBTS is very stable (A1 std=0.0016, A7 std=4.89%) vs TimeGAN which can vary (A1 std=0.0147, A7 std=6.71%).

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
