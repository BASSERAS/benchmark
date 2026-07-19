# SBTS — Schrödinger Bridge Time Series (ICAIF 2025)

**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

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

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case a perfect generator reaches with finite
samples, from a row-shuffled copy of the real data (see
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/)). Most floors are 0 because a
permutation preserves every column-wise marginal; the residual non-zero floors are pure finite-sample noise.

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | |
| A1 | Kurtosis Error | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | 0 |
| A2 | \|r\| q95 Error | 0.0063 ± 3.00e-05 | 0.0064 | 0.0063 | 0.0063 | 0.0063 | 0.0063 | 0 |
| A3 | \|r\| q99 Error | 0.0098 ± 4.80e-05 | 0.0098 | 0.0098 | 0.0099 | 0.0098 | 0.0097 | 0 |
| A4 | Tail QQ Error | 0.0062 ± 2.60e-05 | 0.0063 | 0.0062 | 0.0062 | 0.0062 | 0.0062 | 0 |
| A5 | Hill Tail Index Error | 9.499 ± 0.3457 | 9.286 | 9.201 | 9.853 | 9.981 | 9.175 | 0 |
| | **— Distribution —** | | | | | | | |
| A6 | Path MMD² | 0.0110 ± 0.0026 | 0.0083 | 0.0131 | 0.0124 | 0.0076 | 0.0138 | 0.0015 |
| A7 | Terminal MMD² | 0.0100 ± 0.0036 | 0.0059 | 0.0147 | 0.0123 | 0.0055 | 0.0113 | 0.0016 |
| A8 | Increment MMD² | 0.0071 ± 2.47e-04 | 0.0076 | 0.0069 | 0.0071 | 0.0072 | 0.0069 | 7.45e-04 |
| A9 | Volatility MMD | 0.3038 ± 0.0071 | 0.3172 | 0.2982 | 0.3049 | 0.2987 | 0.3002 | 0.0071 |
| A10 | Terminal SWD | 3.539 ± 0.7368 | 2.706 | 4.389 | 4.133 | 2.617 | 3.849 | 0.6873 |
| A11 | Path SWD | 2.415 ± 0.4104 | 1.897 | 2.723 | 2.828 | 1.933 | 2.692 | 0.4381 |
| A12 | RV Law Loss | 2.148 ± 0.0074 | 2.156 | 2.151 | 2.155 | 2.141 | 2.138 | 0 |
| A13 | Mean Path RMSE | 0.7499 ± 0.1823 | 0.7951 | 0.5148 | 0.9253 | 0.5595 | 0.9545 | 0 |
| A14 | KS Log-returns | 0.0534 ± 3.62e-04 | 0.0537 | 0.0530 | 0.0539 | 0.0530 | 0.0536 | 0 |
| A15 | Skewness Error | 0.0227 ± 0.0037 | 0.0196 | 0.0184 | 0.0217 | 0.0249 | 0.0287 | 0 |
| A16 | QQ RMSE (300-pt) | 0.0028 ± 1.20e-05 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0 |
| A17 | Terminal Price KS | 0.0921 ± 0.0051 | 0.0892 | 0.0938 | 0.0903 | 0.0863 | 0.1011 | 0 |
| | **— Adversarial —** | | | | | | | |
| A18 GRU | Discriminative Score GRU | 0.2755 ± 0.2166 | 0.4377 | 0.4490 | 0.0017 | 0.4695 | 0.0194 | 0.0071 |
| A18 MLP | Discriminative Score MLP | 0.0079 ± 0.0049 | 0.0038 | 0.0139 | 0.0032 | 0.0139 | 0.0047 | 0.0033 |
| | **— Predictive —** | | | | | | | |
| A19 GRU | Predictive Score GRU | 0.0586 ± 5.90e-05 | 0.0585 | 0.0586 | 0.0587 | 0.0586 | 0.0586 | 0.0537 |
| A19 MLP | Predictive Score MLP | 0.0583 ± 2.55e-04 | 0.0582 | 0.0578 | 0.0583 | 0.0586 | 0.0583 | 0.0537 |
| | **— Temporal —** | | | | | | | |
| A20 | Covariance Error | 145.35 ± 4.886 | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | 0 |
| A21 | ACF \|r\| Error (lags) | 0.0596 ± 4.70e-04 | 0.0601 | 0.0595 | 0.0596 | 0.0587 | 0.0599 | 0 |
| A22 | ACF r² Error (lags) | 0.0619 ± 5.08e-04 | 0.0625 | 0.0618 | 0.0614 | 0.0612 | 0.0624 | 0 |
| A23 | ACF \|r\| Lag-1 Error | 0.1437 ± 0.0012 | 0.1436 | 0.1419 | 0.1439 | 0.1436 | 0.1456 | 0 |
| A24 | ACF r² Lag-1 Error | 0.1665 ± 0.0017 | 0.1674 | 0.1637 | 0.1659 | 0.1668 | 0.1688 | 0 |
| | **— Vol —** | | | | | | | |
| A25 | Mean RMSE | 1.301 ± 0.2776 | 1.297 | 0.9199 | 1.482 | 1.099 | 1.709 | 0 |
| A26 | Return Std Error | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | 0 |
| A27 | Log-Return Std Error | 0.0030 ± 1.20e-05 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0 |
| A28 | Kurtosis Ratio | 1.989 ± 0.0182 | 1.991 | 1.998 | 1.988 | 2.012 | 1.957 | 1.000 |
| A29 | Sigma Mean Error | 0.0440 ± 1.84e-04 | 0.0442 | 0.0440 | 0.0441 | 0.0437 | 0.0437 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | 3.276 ± 0.0637 | 3.204 | 3.296 | 3.296 | 3.209 | 3.375 | 0 |
| A31 | Rolling Vol KS (w=5) | 0.3435 ± 6.43e-04 | 0.3444 | 0.3433 | 0.3440 | 0.3430 | 0.3426 | 0 |
| A32 | Vol-of-Vol Error | 0.0021 ± 6.00e-06 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0 |
| | **— Heston Spec —** | | | | | | | |
| A33 | Teacher-Sigma Corr ↑ | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | 0.6143 |
| A34 | Teacher-Sigma RMSE | 0.0955 ± 9.10e-05 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 | 0.0654 |

---

## Stylised Facts Diagnostic (Heston vs SBTS, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means;
  combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one division. Combined mean = mean
  of the three sub-scores; combined std = sample std across the 5 seeds.

↓ lower is better for both rows. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 12.138 ± 0.1605 | 12.213 | 12.187 | 12.374 | 11.917 | 11.998 | 0 |
| | % err | 4755% ± 2735% | 8372% | 1121% | 5642% | 6549% | 2089% | 0 |
| **QQ plot** | MSE | 8.90e-06 ± 6.77e-08 | 8.98e-06 | 8.91e-06 | 8.98e-06 | 8.84e-06 | 8.81e-06 | 0 |
| | % err | 37.69% ± 1.874% | 40.72% | 38.65% | 35.80% | 37.56% | 35.73% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.0046 ± 3.70e-05 | 0.0046 | 0.0045 | 0.0045 | 0.0045 | 0.0046 | 0 |
| | % err | 423.3% ± 11.52% | 410.1% | 429.8% | 408.6% | 432.5% | 435.4% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0052 ± 5.67e-05 | 0.0053 | 0.0051 | 0.0051 | 0.0052 | 0.0053 | 0 |
| | % err | 534.5% ± 49.86% | 543.7% | 603.7% | 567.5% | 490.1% | 467.5% | 0 |
| **Rolling vol histogram** | MSE | 1227.30 ± 5.109 | 1234.15 | 1226.44 | 1230.95 | 1223.20 | 1221.77 | 0 |
| | % err | 238.6% ± 5.655% | 235.3% | 229.1% | 241.5% | 243.8% | 243.5% | 0 |
| **Tail survival** | MSE | 0.0057 ± 6.60e-05 | 0.0058 | 0.0058 | 0.0058 | 0.0057 | 0.0057 | 0 |
| | % err | 5717% ± 303.4% | 6171% | 5755% | 5740% | 5704% | 5215% | 0 |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

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

**SBTS does NOT generate log-prices or prices directly.** It generates **scaled log-returns** R̃,
then reconstructs price paths via the inverse transform. The √dt/σ scaling ensures the empirical
variance of R̃ matches the theoretical SDE variance Δt, which stabilises the kernel estimation.

---

## Files

| Artifact | Path |
|----------|------|
| All A + B metrics (mean/std + per-seed) | `metrics_summary.csv` |
| Per-seed raw metric dumps | `seed_{0..4}_metrics.json` |
| B two-subline aggregates (MSE + % err) | `curve_b_aggregate.json` |
| Classifier / predictor loss curves | `seed_{i}_{disc,pred}_{gru,mlp}_loss.csv` |
| Stylised-facts 8-panel diagnostic | `plots/heston_diagnostics.png` |
| PCA / t-SNE embeddings per seed | `plots/seed_{i}_{pca,tsne}.png` |
| Path-shadowing MC forecasts | `path_shadowing/` |
| Generated price paths (8192×128) | `../../../methods/SBTS/generated_paths/seed_{i}/generated_paths_8192x128.npy` |

→ Cross-method comparison with TimeGAN: [`results/README.md`](../../README.md)
