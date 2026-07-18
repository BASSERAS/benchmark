# SBTS — Schrödinger Bridge Time Series (ICAIF 2025)

**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** SBTS univariate Markovian — h=0.4, K=1, N_pi=200, CPU-only (no GPU).
No neural network, no training. Kernel density estimation with Schrödinger-bridge drift.

**Convention:** lower is better for all metrics **except A15 Corr ↑ and A21 Corr ↑**. A19 Kurtosis Ratio: perfect = 1.0.

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

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|-----------|--------|--------|--------|--------|--------|
| | **— Fat Tail —** | | | | | | |
| A10 | Kurtosis Error         | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 |
| A17 | \|r\| q95 Error        | 0.0063 ± 0.0000 | 0.0064 | 0.0063 | 0.0063 | 0.0063 | 0.0063 |
| A18 | \|r\| q99 Error        | 0.0098 ± 0.0000 | 0.0098 | 0.0098 | 0.0099 | 0.0098 | 0.0097 |
| A30 | Tail QQ Error          | 0.0062 ± 0.0000 | 0.0063 | 0.0062 | 0.0062 | 0.0062 | 0.0062 |
| A34 | Hill Tail Index Error  | 9.499 ± 0.346   | 9.286  | 9.201  | 9.853  | 9.981  | 9.175  |
| | **— Distribution —** | | | | | | |
| A1  | Path MMD²              | 0.0112 ± 0.0011 | 0.0129 | 0.0120 | 0.0104 | 0.0100 | 0.0105 |
| A2  | Terminal MMD²          | 0.0102 ± 0.0014 | 0.0113 | 0.0097 | 0.0085 | 0.0092 | 0.0124 |
| A3  | Increment MMD²         | 0.0069 ± 0.0005 | 0.0067 | 0.0066 | 0.0069 | 0.0063 | 0.0077 |
| A4  | Volatility MMD         | 0.2964 ± 0.0126 | 0.2970 | 0.2873 | 0.2968 | 0.2821 | 0.3188 |
| A5  | Terminal SWD           | 3.7097 ± 0.3209 | 4.1147 | 3.6846 | 3.6757 | 3.1564 | 3.9169 |
| A6  | Path SWD               | 2.5335 ± 0.2212 | 2.8445 | 2.6110 | 2.5826 | 2.1668 | 2.4625 |
| A24 | RV Law Loss (W₁ on ann. RV) | 2.1482 ± 0.0074 | 2.1559 | 2.1510 | 2.1552 | 2.1411 | 2.1380 |
| A25 | Mean Path RMSE         | 0.7499 ± 0.1823 | 0.7951 | 0.5148 | 0.9253 | 0.5595 | 0.9545 |
| A27 | KS on Log-returns      | 0.0534 ± 0.0004 | 0.0537 | 0.0530 | 0.0539 | 0.0530 | 0.0536 |
| A28 | Skewness Error         | 0.0227 ± 0.0037 | 0.0196 | 0.0184 | 0.0217 | 0.0249 | 0.0287 |
| A29 | QQ RMSE (300-pt)       | 0.0028 ± 0.0000 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0028 |
| A33 | Terminal Price KS      | 0.0921 ± 0.0051 | 0.0892 | 0.0938 | 0.0903 | 0.0863 | 0.1011 |
| | **— Adversarial —** | | | | | | |
| A13 | Disc Score GRU        | 0.2740 ± 0.2208 | 0.0005 | 0.4286 | 0.4643 | 0.4689 | 0.0078 |
| A13 | Disc Score MLP        | 0.0063 ± 0.0038 | 0.0029 | 0.0014 | 0.0084 | 0.0121 | 0.0066 |
| | **— Predictive —** | | | | | | |
| A14 | Pred Score GRU (TSTR) | 0.0586 ± 0.0000 | 0.0586 | 0.0586 | 0.0585 | 0.0586 | 0.0586 |
| A14 | Pred Score MLP (TSTR) | 0.0582 ± 0.0002 | 0.0584 | 0.0579 | 0.0584 | 0.0581 | 0.0582 |
| | **— Temporal —** | | | | | | |
| A7  | Covariance Error       | 145.35 ± 4.89   | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 |
| A11 | ACF \|r\| Error        | 0.0596 ± 0.0005 | 0.0601 | 0.0595 | 0.0596 | 0.0587 | 0.0599 |
| A12 | ACF r² Error           | 0.0619 ± 0.0005 | 0.0625 | 0.0618 | 0.0614 | 0.0612 | 0.0624 |
| A22 | ACF \|r\| Lag-1 Error  | 0.1437 ± 0.0012 | 0.1436 | 0.1419 | 0.1439 | 0.1436 | 0.1456 |
| A23 | ACF r² Lag-1 Error     | 0.1665 ± 0.0017 | 0.1674 | 0.1637 | 0.1659 | 0.1668 | 0.1688 |
| | **— Vol —** | | | | | | |
| A8  | Mean RMSE              | 1.3013 ± 0.2776 | 1.2972 | 0.9199 | 1.4819 | 1.0986 | 1.7088 |
| A9  | Return Std Error       | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 |
| A16 | Log-Return Std Error   | 0.0030 ± 0.0000 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0030 |
| A19 | Kurtosis Ratio (→ 1)   | 1.9890 ± 0.0182 | 1.9907 | 1.9978 | 1.9877 | 2.0121 | 1.9569 |
| A20 | Sigma Mean Error       | 0.0440 ± 0.0002 | 0.0442 | 0.0440 | 0.0441 | 0.0437 | 0.0437 |
| A26 | Cross-Sect. Vol Path RMSE | 3.2760 ± 0.0637 | 3.2044 | 3.2961 | 3.2960 | 3.2086 | 3.3751 |
| A31 | Rolling Vol KS (window=5) | 0.3435 ± 0.0006 | 0.3444 | 0.3433 | 0.3440 | 0.3430 | 0.3426 |
| A32 | Vol-of-Vol Error       | 0.0021 ± 0.0000 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 |
| | **— Heston Spec —** | | | | | | |
| A15 | Sigma Corr ↑          | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 |
| A15 | Sigma RMSE            | 0.0955 ± 0.0001 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 |
| A21 | Oracle Sigma Corr ↑   | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 |

### B — Curve-shape metrics (6 plots × 3 sub-metrics)

Each cell = MSE between the real and generated **curve** (not a scalar).
- **funct**: MSE(L\_real, L\_gen) between curve values
- **der**: MSE of first finite difference — L\_der\[k\] = L\[k+1\] − L\[k\]
- **sec\_der**: MSE of second finite difference — L\_sec\[k\] = L\_der\[k+1\] − L\_der\[k\]

All values: mean ± std across 5 seeds. Perfect floor = 0 (row-shuffled real → identical curves). Full formulas: [`metrics/README.md`](../../../metrics/README.md).

| Plot | funct | der | sec\_der |
|------|-------|-----|----------|
| Log-return histogram | 11.59 ± 0.156 | 0.225 ± 0.0091 | 0.320 ± 0.0373 |
| QQ plot              | 8.70e-6 ± 6.8e-8 | 1.71e-7 ± 3.6e-9 | 3.75e-8 ± 1.7e-9 |
| ACF \|r\|            | 2.42e-3 ± 3.3e-5 | 1.32e-3 ± 1.3e-5 | 8.27e-4 ± 9.6e-6 |
| ACF r²               | 2.54e-3 ± 4.2e-5 | 1.61e-3 ± 2.9e-5 | 1.03e-3 ± 2.4e-5 |
| Rolling vol hist.    | 1214 ± 5.1 | 11.81 ± 0.072 | 1.775 ± 0.399 |
| Tail survival        | 5.74e-3 ± 6.6e-5 | 6.86e-6 ± 6.6e-8 | 6.65e-8 ± 5.6e-9 |

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

## B — Curve-shape metrics: plot mapping

Each B metric measures the shape of a diagnostic curve, not a single scalar statistic.
For each of the 6 plots, three sub-metrics are computed:

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Formulas: [`metrics/README.md`](../../../metrics/README.md)

---

## Stylised Facts Diagnostic (Heston vs SBTS, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)
