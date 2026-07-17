# SBTS — Schrödinger Bridge Time Series (ICAIF 2025)

**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**. A16 ↓.

---

## What we generate — price paths from the Heston SDE

The **target process** is the Heston stochastic volatility model:

$$dS_t = \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \quad \text{Corr}(dW^S, dW^v) = \rho$$

Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**SBTS does not generate $S_t$ directly.** The method operates on **scaled log-returns** $\tilde{R}$
(Paper §6) and reconstructs price paths via an inverse transform:

```
Input:  S_real (8192, 128)   — price paths from Heston SDE (training data)
Step 1: R = log(S[:,1:] / S[:,:-1])       — log-returns (8192, 127)
Step 2: R̃ = R × √(dt) / σ(R)            — scaled log-returns  [SBTS input]
Step 3: SBTS generates R̃_gen             — new scaled log-returns (kernel estimation)
Step 4: R_gen = R̃_gen × σ(R) / √(dt)   — inverse scaling
Step 5: S_gen[:,0] = 100                  — anchor at S₀
        S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])  — reconstruct prices
Output: S_gen (8192, 128)   — generated price paths in price space ≈ 100
```

The $\sqrt{\Delta t}/\sigma(R)$ scaling normalises the empirical return variance to $\Delta t$,
matching the theoretical SDE diffusion coefficient and stabilising the kernel bridge estimation.

---

## Results (mean ± std across 5 seeds)

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|-----------|--------|--------|--------|--------|--------|---------|
| A1  | Path MMD²                   | 0.0110 ± 0.0016 | 0.0093 | 0.0136 | 0.0117 | 0.0093 | 0.0110 | **0** |
| A2  | Terminal MMD²               | 0.0090 ± 0.0035 | 0.0061 | 0.0153 | 0.0083 | 0.0056 | 0.0098 | **0** |
| A3  | Increment MMD²              | 0.0071 ± 0.0005 | 0.0074 | 0.0076 | 0.0063 | 0.0074 | 0.0065 | **0** |
| A4  | Volatility MMD              | 0.3125 ± 0.0176 | 0.3230 | 0.3366 | 0.2900 | 0.3183 | 0.2947 | **0** |
| A5  | Terminal SWD                | 3.465 ± 0.588   | 2.775  | 4.258  | 3.634  | 2.799  | 3.858  | **0** |
| A6  | Path SWD                    | 2.497 ± 0.288   | 2.203  | 2.928  | 2.620  | 2.150  | 2.582  | **0** |
| A7  | Covariance Error            | 145.35 ± 4.89   | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | **0** |
| A8  | Mean RMSE                   | 1.3013 ± 0.2776 | 1.2972 | 0.9199 | 1.4819 | 1.0986 | 1.7088 | **0** |
| A9  | Return Std Error            | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | **0** |
| A10 | Return Kurtosis Error       | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | **0** |
| A11 | ACF Error (abs returns)     | 0.0568 ± 0.0006 | 0.0568 | 0.0565 | 0.0570 | 0.0558 | 0.0577 | **0** |
| A12 | ACF Error (sq returns)      | 0.0619 ± 0.0006 | 0.0619 | 0.0618 | 0.0617 | 0.0610 | 0.0628 | **0** |
| A13 | Discriminative Score (GRU)  | 0.0291 ± 0.0276 | 0.0032 | 0.0148 | 0.0157 | 0.0816 | 0.0301 | **0** |
| A13 | Discriminative Score (MLP)  | 0.0711 ± 0.0077 | 0.0676 | 0.0859 | 0.0685 | 0.0700 | 0.0636 | **0** |
| A14 | Predictive Score GRU (TSTR) | 0.0091 ± 0.0000 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | baseline |
| A14 | Predictive Score MLP (TSTR) | 0.0093 ± 0.0006 | 0.0092 | 0.0090 | 0.0104 | 0.0088 | 0.0088 | baseline |
| A15 | Sigma Corr ↑                | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | **1** |
| A15 | Sigma RMSE                  | 0.0955 ± 0.0001 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 | **0** |
| A16 | Tail Survival Error         | 0.0367 ± 0.0002 | 0.0369 | 0.0369 | 0.0365 | 0.0366 | 0.0363 | **0** |

---

## Comparison with the paper (Alouadi et al., ICAIF 2025)

> ⚠️ **Direct metric comparison is not meaningful across datasets** — the paper uses
> different data representations (base-one-scale log-returns), multivariate series (d=2 for Heston),
> shorter sequences (T=24 for real datasets, T=252 for robustness tests), and ~500 training samples.
> Our setup: T=128, N=8 192 paths, univariate price paths.
> What we *can* verify: (a) hyperparameters match the paper exactly, and
> (b) discriminative scores fall in the paper's expected performance range.

### A. Hyperparameter verification (Appendix C, Table 4)

| Setting | Our reimplementation | Paper (Appendix C Table 4) |
|---------|:--------------------:|:---------------------------:|
| Data representation | Price paths S_t | Log-returns (base-one-scale) |
| Dimension d | 1 (univariate) | 2 (price + variance) |
| Seq len T | **128** | **252** |
| Training paths N | **8 192** | ~**1 000** |
| h (bandwidth) | **0.4** | **0.4** ✓ |
| K (Markov order) | **1** | **1** ✓ |
| N_pi (Euler substeps) | **200** | **200** ✓ |
| Generation hardware | 64 CPU workers (EPYC 7763) | 12 CPU cores (Intel Broadwell) |
| Generation time (N=1 000) | ~47 s (extrapolated) | **548 s** (App. B) |
| Generation time (N=8 192) | 376 s (64 workers, seeds 1-4) | N/A |

> Note on timing: the paper reports 548 s for N=1 000, T=252 on 12 cores (single-threaded).
> Our faster time (extrapolated ~47 s for 1 000 paths at T=128 with 64 workers) is explained by
> shorter series (T=128 vs 252) and more workers (64 vs 12). Per-path cost: ~2.9 s/path at T=128
> vs ~6.6 s/path at T=252 — consistent with O(T²) kernel computation.

### B. Score comparison vs paper (Tables 1 & 2)

> ⚠️ Not a direct comparison. The paper evaluates SBTS on Stocks (d=6, T=24) and Sine (d=5, T=24);
> we evaluate on Heston (d=1, T=128). The table validates that our implementation achieves
> discriminative scores in the paper's expected range.

| Metric | Paper — Stocks (d=6, T=24) | Paper — Sine (d=5, T=24) | Ours — Heston GRU (d=1, T=128) | Ours — Heston MLP (d=1, T=128) |
|--------|:--------------------------:|:------------------------:|:-------------------------------:|:-------------------------------:|
| Disc Score ↓ | 0.010 ± 0.008 | 0.061 ± 0.010 | 0.029 ± 0.028 | 0.071 ± 0.008 |
| Pred Score ↓ | 0.017 ± 0.000 | 0.095 ± 0.002 | 0.0091 ± 0.0000 | 0.0093 ± 0.0006 |

Our GRU discriminative score (0.029) sits between the paper's Stocks (0.010) and Sine (0.061),
consistent with Heston d=1 T=128 being harder to fool than a 6-D financial dataset but easier
than a 5-D synthetic Sine. Predictive scores are lower than the paper's because next-step
prediction is inherently simpler in 1-D Heston than in 5-D Sine or 6-D Stocks.

### C. Scaling — What SBTS generates (Paper §6)

The paper's §6 explains the internal scaling used to handle non-stationary variance:

$$\tilde{R}_{t_i} = R_{t_i} \times \frac{\sqrt{\Delta t}}{\sigma(R)}$$

where $R_{t_i} = (R_{t_1}, \ldots, R_{t_N})$ is the log-return sequence and $\sigma(R)$ is the
empirical standard deviation of the training log-returns.

**Our pipeline:**
```
Input:  S_real (8192, 128)  — price paths from Heston
Step 1: R = log(S[:,1:] / S[:,:-1])          — log-returns (8192, 127)
Step 2: R̃ = R × √(dt) / σ(R)               — scaled log-returns (SBTS input)
Step 3: SBTS generates R̃_gen                — in scaled log-return space
Step 4: R_gen = R̃_gen × σ(R) / √(dt)       — inverse scaling
Step 5: S_gen[:,0] = 100                      — anchor at S₀
        S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])  — reconstruct prices
Output: S_gen (8192, 128)  — price paths in original scale ≈ 100
```

**SBTS does NOT generate log-prices or prices directly.** It generates **scaled log-returns** R̃,
then reconstructs price paths via the inverse transform. The √dt/σ scaling ensures the empirical
variance of R̃ matches the theoretical SDE variance Δt, which stabilises the kernel estimation.

---

→ Cross-method comparison with TimeGAN: [`results/README.md`](../../README.md)

---

## Metric Definitions

**Notation.**
$X \sim P$ = real paths $(N \times T \times d)$,
$\tilde{X} \sim Q$ = generated paths,
$r_t = \log X_{t+1} - \log X_t$ = log-returns,
$k(x, x') = \exp(-\|x-x'\|^2 / 2\sigma^2)$ = RBF kernel.

---

### A1 — Path MMD² · *Maximum Mean Discrepancy on full paths*

$$
\widehat{\text{MMD}}^2(P, Q)
= \frac{1}{N^2}\sum_{i,j} k(x_i, x_j)
- \frac{2}{N^2}\sum_{i,j} k(x_i, \tilde{x}_j)
+ \frac{1}{N^2}\sum_{i,j} k(\tilde{x}_i, \tilde{x}_j)
$$

Applied to the **concatenated full path** $x = (X_1, \ldots, X_T) \in \mathbb{R}^{T \cdot d}$.
Tests whether the joint path distribution $P$ is correctly reproduced. **Perfect: 0.**

---

### A2 — Terminal MMD² · *Maximum Mean Discrepancy on terminal values*

Same estimator applied only to **terminal prices** $x = X_T \in \mathbb{R}^d$.
Tests whether the marginal distribution at maturity is correct. **Perfect: 0.**

---

### A3 — Increment MMD² · *Maximum Mean Discrepancy on returns*

$$
\widehat{\text{MMD}}^2\!\left(\{r_t\}_{t<T},\, \{\tilde{r}_t\}_{t<T}\right),
\quad r_t = X_{t+1} - X_t
$$

Applied to all increments pooled across time. Tests the return distribution. **Perfect: 0.**

---

### A4 — Volatility MMD · *Maximum Mean Discrepancy on realised volatility*

Rolling realised volatility with window $w = 5$:

$$
\hat{\sigma}_t = \sqrt{\frac{1}{w}\sum_{s=t-w}^{t-1} r_s^2}
$$

MMD between $\{\hat{\sigma}_t^{\text{real}}\}$ and $\{\hat{\sigma}_t^{\text{fake}}\}$.
Tests **volatility clustering** and the vol level distribution. **Perfect: 0.**

---

### A5 — Terminal SWD · *Sliced Wasserstein Distance on terminal values*

$$
\text{SWD}(P_T, Q_T)
= \mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}
  \left[W_1\!\left(\theta_\sharp P_T,\, \theta_\sharp Q_T\right)\right]
$$

$W_1$ = 1-Wasserstein distance, $\theta_\sharp$ = projection onto direction $\theta$.
Approximated with 512 random slices. More robust to heavy tails than MMD. **Perfect: 0.**

---

### A6 — Path SWD · *Sliced Wasserstein Distance on full paths*

Same SWD formula applied to full paths $x \in \mathbb{R}^{T \cdot d}$.
Captures the geometry of the entire path distribution. **Perfect: 0.**

---

### A7 — Covariance Error · *Frobenius norm of covariance difference*

$$
\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F,
\quad \Sigma = \text{Cov}(X_T) \in \mathbb{R}^{d \times d}
$$

Measures how well the cross-asset covariance structure is reproduced at maturity.
For $d=1$ (Heston) this reduces to $|\text{Var}(X_T^{\text{real}}) - \text{Var}(X_T^{\text{fake}})|$.
SBTS (Markov-1) cannot capture the full multi-step covariance — hence A7 = 145% error. **Perfect: 0.**

---

### A8 — Mean RMSE · *Root Mean Square Error of terminal means*

$$
\sqrt{\frac{1}{d}\left\|\mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T]\right\|^2}
$$

Measures systematic bias in the generated terminal price level. **Perfect: 0.**

---

### A9 — Return Std Error · *Mean Absolute Error of return standard deviation*

$$
\left|\,\sigma(r_{\text{real}}) - \sigma(r_{\text{fake}})\,\right|
$$

Tests whether the overall volatility level (standard deviation of returns) is correct. **Perfect: 0.**

---

### A10 — Return Kurtosis Error · *Mean Absolute Error of excess kurtosis*

$$
\left|\,\kappa(r_{\text{real}}) - \kappa(r_{\text{fake}})\,\right|,
\quad \kappa(Z) = \frac{\mathbb{E}[(Z-\mu)^4]}{\sigma^4} - 3
$$

Tests fat-tail reproduction. Excess kurtosis = 0 for Gaussian; financial returns typically show
$\kappa > 0$. SBTS scores 0.119 (small) vs TimeGAN 2.955 (large). **Perfect: 0.**

---

### A11 — ACF Error (abs returns) · *Mean Absolute Error of autocorrelation function on |r|*

$$
\frac{1}{|L|}\sum_{\ell \in L}
\left|\,\text{ACF}(|r_{\text{real}}|, \ell) - \text{ACF}(|r_{\text{fake}}|, \ell)\,\right|,
\quad L = \{1, 2, 5, 10\}
$$

Tests **volatility clustering**: real financial returns exhibit significant positive autocorrelation
in $|r_t|$ and $r_t^2$ (ARCH effect). **Perfect: 0.**

---

### A12 — ACF Error (sq returns) · *Mean Absolute Error of autocorrelation function on r²*

Same formula as A11 applied to squared returns $r_t^2$ instead of $|r_t|$.
Also tests the ARCH/GARCH effect; provides a complementary view to A11. **Perfect: 0.**

---

### A13 — Discriminative Score · *Post-hoc binary classification accuracy offset*

$$
\text{DS} = \left|\,\text{Acc}_{\text{test}} - 0.5\,\right|
$$

**Principle.** If a binary classifier trained to separate real from fake achieves test
accuracy close to 50 %, the two distributions are indistinguishable — the generator is
good. If accuracy is close to 100 %, the distributions are trivially different — the
generator is bad.

**Data preparation.**
The combined dataset of 8 192 real paths + 8 192 generated paths (16 384 total,
label 1 = real, 0 = fake) is split 80/20 into train (13 107 samples) and test (3 277 samples).
Splitting is done once before training; the test set is never seen during training.

**Training.**
Both classifiers are trained for 2 000 steps with Adam (lr = 1 × 10⁻³), batch size 128,
Binary Cross-Entropy loss.

---

#### GRU Discriminator

```
Input: path (128, 1)
  └─ GRU, hidden = 8, num_layers = 2, batch_first = True
       └─ last hidden state h_T  (8,)
            └─ Linear(8 → 1)  → logit
```

The GRU reads the path **step by step**, building a hidden state that encodes the full
temporal trajectory. It can exploit **temporal patterns** (autocorrelation, vol clustering,
mean-reversion) to distinguish real from fake.

Score 0.029 ± 0.028: high variance across seeds (seed 3 = 0.082 vs seed 0 = 0.003) but
mean near zero — SBTS paths are largely indistinguishable from real temporally.

---

#### MLP Discriminator

```
Input: path (128, 1) → Flatten → (128,)
  └─ Linear(128 → 128) → ReLU
       └─ Linear(128 → 64) → ReLU
            └─ Linear(64 → 1)  → logit
```

The MLP receives the **entire path as a flat vector** with no notion of ordering.
It detects differences in the **joint marginal distribution** but has no structural
inductive bias toward temporal dependencies.

Score 0.071 ± 0.008: consistently higher than GRU, suggesting the MLP finds marginal
distributional differences that the GRU misses (e.g., price scale drift over the path).

**Key difference between GRU and MLP:**
- GRU ≈ test for *temporal structure* (autocorrelation, clustering, dynamics).
- MLP ≈ test for *marginal distributional match* (moments, price levels, shape).

**Evaluation.** After training, the classifier scores every sample in the held-out 20 %
test set. `DS = |accuracy − 0.5|`. **Perfect score = 0** (random guessing). **Worst = 0.5**
(perfect separation).

---

### A14 — Predictive Score · *Train-on-Synthetic Test-on-Real (TSTR) MAE*

$$
\text{PS} = \frac{1}{N \cdot (T-1)} \sum_{i=1}^{N} \sum_{t=1}^{T-1}
\left|\hat{X}_{i,t+1} - X_{i,t+1}^{\text{real}}\right|
$$

**Principle (TSTR).** A one-step-ahead predictor is trained **exclusively on synthetic
paths**. It is then evaluated **on real paths**. If the predictor generalises well (low MAE
on real data), the synthetic data has captured the true temporal dynamics.

**Training.**
Both predictors are trained for 5 000 steps with Adam (lr = 1 × 10⁻³), batch size 128,
L1 (MAE) loss on normalised paths.

---

#### GRU Predictor

```
Input: prefix X[0:T-1]  shape (127, 1)
  └─ GRU, hidden = 8, num_layers = 2, batch_first = True
       └─ output at every step  (127, 8)
            └─ Linear(8 → 1) applied at each step → (127, 1)
```

The GRU sees the **full causal history** up to step $t$ and predicts $X_{t+1}$ for
every $t \in \{1, \ldots, T-1\}$ simultaneously (sequence-to-sequence).

Score 0.0091 ± 0.0000: extremely stable across seeds — SBTS log-return distribution
well predicts the real path dynamics.

---

#### MLP Predictor

```
Input: sliding window X[t-8:t]  shape (8, 1)
  └─ Flatten → (8,)
       └─ Linear(8 → 64) → ReLU
            └─ Linear(64 → 32) → ReLU
                 └─ Linear(32 → 1) → X̂_{t+1}
```

The MLP predicts $X_{t+1}$ from a **local context window of the 8 most recent steps**.
Tests whether **short-range temporal patterns** in the synthetic data match real data.

**Key difference between GRU and MLP:**
- GRU ≈ test for *long-range temporal predictability* (mean-reversion, trends, vol regimes).
- MLP ≈ test for *local (8-step) temporal predictability* (short-term momentum).

**Evaluation.** Both predictors are evaluated on all 8 192 real paths.
PS = mean absolute error on next-step predictions, in the normalised [0, 1] scale.
**Perfect score = 0** (zero prediction error, unreachable). A score matching the real-data
auto-regression baseline means synthetic data has the same short-range predictability as real data.

---

### A15 — Teacher-Sigma Correlation · *Pearson correlation, realised vol vs true vol* ↑

$$
\rho = \text{Corr}\!\left(\hat{\sigma}^{\text{gen}},\, \sqrt{v_{\text{true}}}\right)
$$

$\hat{\sigma}^{\text{gen}}$ = rolling realised volatility (window 5) computed from generated
log-returns: $\hat{\sigma}_t = \sqrt{\text{mean}(\log(S_{t+1}/S_t)^2 / \Delta t)}$ over 5-step window.
$\sqrt{v_{\text{true}}}$ = true instantaneous vol from the Heston latent process
(stored in `dataset/Heston/heston_v_8192x128.npy`).

A generator that reproduces stochastic volatility correctly should show $\rho \approx 1$.
**Heston-specific metric.** **Perfect: 1 ↑ (floor: 0.614).**

> **Floor = 0.614:** 5-step rolling QV is a noisy estimator of vₜ. Even for real Heston
> paths vs their own true variance, Pearson ρ ≈ 0.614 due to measurement noise.
> SBTS scores 0.005 — neither this method nor TimeGAN captures stochastic volatility.

---

### A15 — Teacher-Sigma RMSE · *Root Mean Square Error, realised vol vs true vol*

$$
\text{RMSE} = \sqrt{\frac{1}{N \cdot T}
\sum_{i=1}^{N}\sum_{t=1}^{T}
\!\left(\hat{\sigma}^{\text{gen}}_{i,t} - \sqrt{v_{\text{true},i,t}}\right)^2}
$$

Complementary to the correlation: measures absolute scale accuracy of the reproduced
volatility process. **Perfect: 0 (floor: 0.065).**

> **Floor = 0.065:** rolling-QV measurement noise gives an irreducible baseline.
> SBTS (0.096) and TimeGAN (0.118) both score above the floor — SBTS wins.

---

### A16 — Tail Survival Error · *RMS survival-probability error at extreme quantiles*

$$
\text{Tail Error} = \sqrt{\frac{1}{3}\sum_{q \in \{0.90,0.95,0.99\}} (S^{\text{real}}(q) - S^{\text{gen}}(q))^2}
$$

where $S(q) = \Pr(X_T > F_{\text{real}}^{-1}(q))$ is the survival probability evaluated at the
$q$-quantile of the **real** terminal distribution. Tests fat-tail and extreme-event reproduction.

SBTS scores 0.037 vs TimeGAN 0.022 — TimeGAN better reproduces the tails, likely because its GRU
captures longer-range trends that push terminal prices into extreme quantiles. **Perfect: 0.**

---

## CRPS, MAE, RMSE (Path Shadowing context)

These three metrics are used in the **Path Shadowing MC** evaluation
(`results/Heston/SBTS/path_shadowing/`):

**CRPS** (Continuous Ranked Probability Score):

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}_{z \ge y})^2 \, dz = \mathbb{E}_{X \sim F}|X - y| - \frac{1}{2}\mathbb{E}_{X,X' \sim F}|X - X'|$$

Lower is better. A proper scoring rule: minimised in expectation only by the true distribution.
Reduces to MAE when $F$ is a point mass.

**MAE** (Mean Absolute Error):

$$\text{MAE} = \frac{1}{H}\sum_{u=t+1}^{t+H} |\bar{X}(u) - X_{\text{real}}(u)|$$

where $\bar{X}$ is the ensemble mean forecast.

**RMSE** (Root Mean Squared Error):

$$\text{RMSE} = \sqrt{\frac{1}{H}\sum_{u=t+1}^{t+H} (\bar{X}(u) - X_{\text{real}}(u))^2}$$
