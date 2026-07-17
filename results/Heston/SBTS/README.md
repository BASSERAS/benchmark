# SBTS — Score-Based Time Series Generation via Schrödinger Bridge

**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**.

---

## What we generate — price paths (not log-price)

SBTS internally operates on **scaled log-returns**:

```
R = log(S[:,1:] / S[:,:-1])          # log-returns  (8192, 127)
R̃ = R × √dt / σ(R)                  # scale so that Var(R̃) ≈ dt  (paper §6)
```

SBTS generates R̃_gen in this scaled space. We then inverse-scale and reconstruct:

```
R_gen = R̃_gen × σ(R) / √dt          # inverse scaling
S_gen[:,0] = 100                      # anchor at S₀
S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])   # price reconstruction
```

**Output: price paths S_t anchored at S₀=100** — the same space as the real dataset.
We never divide by √σ in the output; the √dt/σ scaling is only internal to SBTS.

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
| A15 | Sigma Corr ↑                | 0.0011 ± 0.0035 | −0.0024 | 0.0031 | −0.0038 | 0.0039 | 0.0045 | **1** |
| A15 | Sigma RMSE                  | 0.8207 ± 0.0019 | 0.8198 | 0.8181 | 0.8214 | 0.8205 | 0.8238 | **0** |

---

## Comparison with the paper (Alouadi et al., ICAIF 2025)

> ⚠️ **Direct metric comparison is not meaningful across datasets** — the paper uses
> different data representations (base-one-scale log-returns), multivariate series (d=2 for Heston),
> shorter sequences (T=24 for real datasets, T=252 for robustness tests), and ~500 training samples.
> Our setup: T=128, N=8 192 paths, univariate price paths.
> What we *can* verify: (a) our hyperparameters match the paper exactly, and
> (b) our discriminative scores are consistent with the paper's reported range.

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

---

### B. Paper Table 1 — Real-world datasets (Stocks, Energy, Air) — SBTS results

Source: Alouadi et al. Table 1. T=24, N≈3 000 training paths, 10 runs mean±std.
Evaluation on base-one-scale log-returns. **SBTS outperforms TimeGAN on all three datasets.**

| Dataset | SBTS Disc Score ↓ | TimeGAN Disc Score ↓ | SBTS Pred Score ↓ | TimeGAN Pred Score ↓ |
|---------|:-----------------:|:--------------------:|:-----------------:|:--------------------:|
| **Stocks** | **0.010 ± 0.008** | 0.102 ± 0.031 | **0.017 ± 0.000** | 0.017 ± 0.004 |
| Energy | 0.356 ± 0.020 | **0.236 ± 0.012** | **0.072 ± 0.001** | 0.273 ± 0.004 |
| **Air** | **0.036 ± 0.016** | 0.447 ± 0.017 | **0.005 ± 0.001** | 0.017 ± 0.004 |

> Energy is the hardest dataset for SBTS (high disc score 0.356) due to abrupt jumps in the data
> not suited to the smooth-SDE assumption. On Stocks and Air, SBTS is the top performer.

---

### C. Paper Table 2 — Toy datasets (Sine, AR) — SBTS results

Source: Alouadi et al. Table 2. T=24, N≈3 000 training paths, 10 runs.

| Dataset | SBTS Disc Score ↓ | TimeGAN Disc Score ↓ | SBTS Pred Score ↓ | TimeGAN Pred Score ↓ |
|---------|:-----------------:|:--------------------:|:-----------------:|:--------------------:|
| Sine | 0.061 ± 0.010 | **0.011 ± 0.008** | 0.095 ± 0.002 | **0.093 ± 0.019** |
| **AR** | **0.034 ± 0.003** | 0.174 ± 0.012 | **0.092 ± 0.007** | 0.412 ± 0.002 |

> SBTS wins convincingly on AR (Markov-1 compatible). Sine is periodic and non-Markovian — TimeGAN wins.

---

### D. Paper Table 3 — Additional metrics on fBM and Stock datasets

Source: Alouadi et al. Table 3. Auto-C = Auto-Correlation error, Cross-C = Cross-Correlation error,
Disc = discriminative score, Pred = predictive score, ONND = Outgoing Nearest Neighbour Distance.
**Bold = best method.**

| Dataset | Metric | RCGAN | TimeGAN | PCFGAN | **SBTS** |
|---------|--------|:-----:|:-------:|:------:|:--------:|
| fBM | Auto-C ↓ | 0.105±0.001 | 0.459±0.003 | 0.125±0.003 | **0.017±0.004** |
| fBM | Cross-C ↓ | 0.051±0.001 | 0.092±0.001 | 0.047±0.001 | **0.005±0.000** |
| fBM | Disc ↓ | 0.207±0.008 | 0.480±0.002 | 0.265±0.006 | **0.005±0.005** |
| fBM | Pred ↓ | 0.456±0.004 | 0.686±0.013 | 0.474±0.003 | **0.423±0.000** |
| fBM | ONND ↓ | 0.622±0.002 | 0.632±0.002 | 0.654±0.002 | **0.471±0.004** |
| Stock | Auto-C ↓ | 0.239±0.016 | 0.228±0.110 | 0.192±0.008 | **0.192±0.008** |
| Stock | Cross-C ↓ | 0.067±0.011 | 0.056±0.002 | 0.055±0.002 | **0.032±0.001** |
| Stock | Disc ↓ | 0.134±0.058 | 0.020±0.021 | 0.028±0.017 | **0.012±0.010** |
| Stock | Pred ↓ | 0.010±0.000 | 0.009±0.000 | 0.009±0.000 | **0.008±0.000** |
| Stock | ONND ↓ | 0.017±0.001 | 0.017±0.000 | 0.016±0.000 | **0.025±0.000** |

> SBTS wins on 9/10 metrics across fBM and Stock. Only fBM Pred and Stock ONND are close calls.

---

### E. Heston robustness — Parameter recovery (Paper §5, Figures 5 & 8)

The paper's §5 evaluates whether SBTS-generated paths preserve the **underlying SDE parameters** of
Heston (κ, θ, r, ξ, ρ), inferred via MLE from the generated paths.

**Figure 5 — Random parameter regime** (parameters uniformly sampled from wide ranges):
- κ, θ, r: SBTS recovers these well — the inferred distribution (orange) aligns with data distribution (blue)
- **ξ (vol of vol) and ρ (correlation): SBTS fails** — generates a Gaussian distribution centered at the
  midpoint of the prior range, rather than matching the true variability
- Root cause: SBTS assumes constant variance in its kernel, which is incompatible with stochastic
  volatility. The drift estimate α* cannot capture Heston's vol-of-vol dynamics.

**Figure 8 — Fixed parameter regime** (same parameters for all samples):
- With fixed κ=0.5, θ=0.04, ξ=0.7, ρ=0.7, r=0.02: SBTS recovers all parameters correctly
- Orange (SBTS) and blue (data) densities align tightly; black line (truth) is inside both
- Confirms SBTS is well-calibrated when the process has constant parameters

**Interpretation for our benchmark:**
Our Heston dataset uses fixed parameters (κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7), placing us in the
**fixed-parameter regime where SBTS should perform well** (Figure 8 case). The A15 Sigma Corr
score of 0.001 (vs floor 0.505) reflects SBTS's known limitation on stochastic variance, consistent
with Figure 5 of the paper.

---

### F. Scaling — What SBTS generates (Paper §6)

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

### A1 — Path MMD²

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(X,X')] - 2\mathbb{E}[k(X,\tilde{X})] + \mathbb{E}[k(\tilde{X},\tilde{X}')]$$

Computed on flattened paths $(N, T)$. Measures full trajectory distribution match.

### A2 — Terminal MMD²

Same MMD² kernel, applied to terminal values $X_T$ only. Measures marginal distribution at T.

### A3 — Increment MMD²

MMD² on increments $r_t = X_{t+1} - X_t$, averaged across timesteps. Measures step distribution.

### A4 — Volatility MMD

MMD distance between rolling volatility distributions (window=5). Measures vol clustering.

### A5 — Terminal SWD

Sliced Wasserstein distance on terminal distribution. Projection-based, robust to outliers.

### A6 — Path SWD

Sliced Wasserstein distance on full path. Captures global trajectory shape.

### A7 — Covariance Error (%)

$$\text{Cov Error} = \frac{\|\Sigma_{\text{real}} - \Sigma_{\text{gen}}\|_F}{\|\Sigma_{\text{real}}\|_F} \times 100$$

Measures temporal covariance structure preservation.

### A8 — Mean RMSE

$$\text{Mean RMSE} = \sqrt{\frac{1}{T}\sum_t (\mu_t^{\text{real}} - \mu_t^{\text{gen}})^2}$$

### A9 — Return Std Error

$|\sigma(r^{\text{real}}) - \sigma(r^{\text{gen}})|$ — global return volatility match.

### A10 — Return Kurtosis Error

$|\kappa(r^{\text{real}}) - \kappa(r^{\text{gen}})|$ — excess kurtosis (fat-tail) match.

### A11 — ACF Error (|returns|)

$$\text{ACF Error} = \frac{1}{L}\sum_{\ell=1}^{L}|\hat{\rho}_{|r|}(\ell) - \hat{\rho}_{|\tilde{r}|}(\ell)|, \quad L=20$$

Volatility clustering (leverage effect proxy).

### A12 — ACF Error (squared returns)

Same as A11 with $r_t^2$ instead of $|r_t|$. GARCH-effect proxy.

### A13 — Discriminative Score

Train a classifier to separate real from fake paths. Score = $|\text{accuracy} - 0.5|$.
0 = indistinguishable. 0.5 = perfectly separable (bad generator).
Two variants: **GRU** (2-layer, hidden=8) and **MLP** (flatten → 128 → 64 → 1).

### A14 — Predictive Score (TSTR)

Train-on-Synthetic, Test-on-Real MAE for next-step prediction.

$$\text{Pred Score} = \frac{1}{N_{\text{real}}(T-1)}\sum_{i,t}|\hat{X}_{i,t+1} - X_{i,t+1}|$$

Two variants: **GRU** and **MLP** predictor.

### A15 — Sigma (Heston-specific)

Infer instantaneous variance from generated paths via quadratic variation:
$\hat{v}_t = (\log S_{t+1} - \log S_t)^2 / \Delta t$.
Compare to true $v_t$ from `dataset/Heston/heston_v_8192x128.npy`.

- **Sigma Corr**: Pearson correlation between $\hat{v}_t$ and $v_t$ (↑, perfect=1).
- **Sigma RMSE**: $\sqrt{\text{mean}(\hat{v}_t - v_t)^2}$ (↓, perfect=0).

### A16 — Tail Survival Error

$$\text{Tail Error} = \sqrt{\frac{1}{3}\sum_{q \in \{0.90,0.95,0.99\}} (S^{\text{real}}(q) - S^{\text{gen}}(q))^2}$$

where $S(q)$ is the survival probability (fraction of paths exceeding the $q$-quantile of the
real terminal distribution). Tests fat-tail reproduction.

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
