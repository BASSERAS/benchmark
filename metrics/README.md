# Metrics — A1 to A34 + B Curve Metrics

Evaluation suite for generative models of financial time series.
All metrics compare **real paths** $X \sim P$ against **generated paths** $\tilde{X} \sim Q$.

Metrics are numbered in **category display order**: Fat Tail (A1–A5), Distribution
(A6–A17), Adversarial (A18), Predictive (A19), Temporal (A20–A24), Volatility (A25–A32),
Heston-specific (A33–A34).

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | All NumPy/SciPy metric functions (A1–A17, A20–A34) **and** the B stylized-fact curve metrics — `compute_curve_metrics()` / `aggregate_curve_metrics()` (A18/A19 live in the two PyTorch scorer files below) |
| `discriminative_score.py` | A18 — post-hoc GRU + MLP classifiers (PyTorch) |
| `predictive_score.py` | A19 — GRU + MLP predictors, TSTR protocol (PyTorch) |
| `compute_all.py` | Orchestrator: A1–A34 + B metrics × N seeds → JSON + CSV + plots |
| `compute_perfect_recovery.py` | Independent-draw floor: all metrics on a fresh Heston draw (seed 1000+i) scored against the test set |
| `recompute_curve_b.py` | Recomputes `curve_b_aggregate.json` (per-plot combined B scores) |

## Run

```bash
# Default: TimeGAN on Heston, 5 seeds
python compute_all.py

# Generic
python compute_all.py --method <MethodName> --dataset <DatasetName> --seeds 5
```

Outputs land in `results/<dataset>/<method>/`.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $X \sim P$ | Real paths, shape $(N, T, d)$ |
| $\tilde{X} \sim Q$ | Generated paths, shape $(N, T, d)$ |
| $r_t = X_{t+1} - X_t$ | Increments / returns |
| $k(x,x') = \frac{1}{L}\sum_{l} \exp\left(-\frac{\|x-x'\|^2}{2 h_l^2 d}\right)$ | Multi-scale RBF kernel, bandwidths $h_l \in \{0.5,1,2,4,8\}$ |
| $W_1$ | 1-Wasserstein distance |
| $\theta_{\sharp} P$ | Push-forward of $P$ along direction $\theta$ (1-D projection) |

---

# Fat-Tail Metrics (A1–A5)

## A1 — Return Kurtosis Error · *Target and generated excess kurtosis*

$$
\kappa_{\text{real}} = \dfrac{\mathbb{E}[(r-\mu)^4]}{\sigma^4} - 3, \qquad
\kappa_{\text{fake}} = \dfrac{\mathbb{E}[(\tilde{r}-\mu)^4]}{\sigma^4} - 3
$$

Output: **target κ** (κ of real returns) and **model κ** (κ of generated returns).
Excess kurtosis = 0 for a Gaussian. Financial returns typically show $\kappa > 0$ (**fat tails**).
Computed using Fisher's definition with bias correction (`scipy.stats.kurtosis(fisher=True, bias=False)`).
**Perfect: target κ ≈ model κ.**

---

## A2 — |r| q95 Error · *Absolute error at the 95th percentile of absolute log-returns*

$$
\left| Q_{0.95}(|r_{\text{real}}|) - Q_{0.95}(|r_{\text{gen}}|) \right|
$$

where $Q_p$ denotes the $p$-th quantile and $r_t = \log(S_{t+1}/S_t)$.
Tests whether the **upper tail (95th percentile)** of the absolute log-return distribution is reproduced.
**Perfect: 0. ↓**

---

## A3 — |r| q99 Error · *Absolute error at the 99th percentile of absolute log-returns*

$$
\left| Q_{0.99}(|r_{\text{real}}|) - Q_{0.99}(|r_{\text{gen}}|) \right|
$$

Same as A2 at the more extreme **99th percentile**. Tests deeper tail reproduction.
**Perfect: 0. ↓**

---

## A4 — Tail QQ Error · *QQ RMSE restricted to the extreme percentiles*

QQ RMSE restricted to $p \in [0.01,0.05] \cup [0.95,0.99]$ (10 extreme percentile points).
Tests **tail alignment** in both tails. **Perfect: 0. ↓**

---

## A5 — Hill Tail Index Error · *Absolute error of the Hill tail-index estimator*

$$\left|\hat{\alpha}^{\text{real}} - \hat{\alpha}^{\text{gen}}\right|, \quad \hat{\alpha} = \left[\frac{1}{k}\sum_{i=1}^{k}\log\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$$

Hill (1975) estimator on top 10% of terminal prices $S_T$. $\alpha > 4$ implies finite kurtosis.
**Perfect: 0. ↓**

---

# Distribution Metrics (A6–A17)

## A6 — Path MMD² · *Maximum Mean Discrepancy on full paths*

$$\text{MMD}^2(P, Q) = \frac{1}{N^2}\sum_{i,j} k(x_i, x_j) - \frac{2}{N^2}\sum_{i,j} k(x_i, \tilde{x}_j) + \frac{1}{N^2}\sum_{i,j} k(\tilde{x}_i, \tilde{x}_j)$$

Each path is flattened to $x = (X_1,\ldots,X_T) \in \mathbb{R}^{T \cdot d}$ before applying the kernel.
Tests whether the **joint temporal distribution** is reproduced. **Perfect: 0. Direction: ↓**

---

## A7 — Terminal MMD² · *Maximum Mean Discrepancy on terminal values*

Same biased estimator as A6, applied only to the terminal value $x = X_T \in \mathbb{R}^d$.
Tests whether the **marginal distribution at maturity** is correct. **Perfect: 0. ↓**

---

## A8 — Increment MMD² · *Maximum Mean Discrepancy on returns*

$$
\text{MMD}^2\left(\{r_t\}_{t < T}, \{\tilde{r}_t\}_{t < T}\right)
$$

where $r_t = X_{t+1} - X_t$. All increments pooled across time before computing MMD.
Tests the **return distribution** (mean, variance, tail shape). **Perfect: 0. ↓**

---

## A9 — Volatility MMD · *Sum of MMD² over volatility-related feature groups*

Biased MMD² is computed **independently for each of the following feature groups**
derived from $r_t = X_{t+1} - X_t$, then summed:

| # | Feature | Shape per sample |
|---|---------|-----------------|
| 1 | Instantaneous RV: $r_t^2$ | $(T-1, d)$ |
| 2 | State-RV pairs: $(X_{t+1}, r_t^2)$ | $(T-1, 2d)$ |
| 3 | Global RV mean per path: $\frac{1}{T}\sum_t r_t^2$ | $(d,)$ |
| 4 | Terminal return: $X_T - X_0$ | $(d,)$ |
| 5 | Returns: $r_t$ | $(T-1, d)$ |
| 6 | Rolling vol (window $w=5$): $\sqrt{\frac{1}{w}\sum_{s=t-w}^{t} r_s^2}$ | $(T-1, d)$ |
| 7 | Absolute returns: $\|r_t\|$ | $(T-1, d)$ |
| 8 | Squared returns: $r_t^2$ | $(T-1, d)$ |
| 9 | ACF lag-products of $|r_t|$ and $r_t^2$ at lags $\{1,2,5,10\}$ (8 sub-groups) | $(N(T-1)d, 1)$ each |

$$
\text{VolMMD}(P, Q) = \sum_{g=1}^{16} \text{MMD}^2(F_g(X), F_g(\tilde{X}))
$$

where $F_g$ extracts feature group $g$ (groups 1–8 contribute one term each; group 9 expands
into 8 ACF sub-groups, giving 16 terms total).
This comprehensively tests volatility level, clustering, tail behaviour, and autocorrelation
structure. **Perfect: 0. ↓**

---

## A10 — Terminal SWD · *Sliced Wasserstein Distance on terminal values*

$$
\text{SWD}(P_T, Q_T)
= \mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}
  \left[ W_1\left(\theta_{\sharp} P_T, \theta_{\sharp} Q_T \right) \right]
$$

Approximated with **50 random projections**.
More robust to heavy tails and high dimensionality than MMD. **Perfect: 0. ↓**

---

## A11 — Path SWD · *Time-averaged Sliced Wasserstein Distance*

$$
\text{SWD}^{\text{path}}(P, Q)
= \frac{1}{T} \sum_{t=1}^{T} \text{SWD}(P_t, Q_t)
$$

where $P_t$ and $Q_t$ are the marginal distributions of real and generated paths at time $t$,
each estimated with **50 random projections**.
Tests whether the **marginal distribution is correct at every time step**, not just at maturity.
**Perfect: 0. ↓**

---

## A12 — Realized-Vol Law Loss · *Wasserstein-1 on the distribution of annualized realized variance*

$$
\text{RV}_i = \frac{1}{\Delta t} \sum_{t=1}^{T-1} r_{i,t}^2 \qquad \text{(annualized realized variance, path }i\text{)}
$$

$$
\mathcal{L}_{\text{RV}} = W_1\!\left(P_{\text{real}}(\text{RV}),\, P_{\text{gen}}(\text{RV})\right)
= \int_{-\infty}^{\infty} \left| F_{\text{real}}^{\text{RV}}(x) - F_{\text{gen}}^{\text{RV}}(x) \right| dx
$$

where $W_1$ is the 1-Wasserstein (Earth Mover's) distance and $F$ is the empirical CDF.
Tests whether the full **volatility regime distribution** is reproduced — not just its mean.
**Perfect: 0. ↓**

Ref: Barndorff-Nielsen & Shephard (2002), *Econometric Analysis of Realized Volatility*.

---

## A13 — Mean Path RMSE

$$A_{13} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(\bar{S}^{\text{real}}_t - \bar{S}^{\text{gen}}_t\right)^2}$$

where $\bar{S}_t = \frac{1}{N}\sum_i S_{i,t}$ is the cross-sectional mean at time $t$.
Tests whether the generator reproduces the **ensemble-mean trajectory**. **Perfect: 0. ↓**

---

## A14 — KS Statistic on Log-returns

$$A_{14} = \sup_x \left|F^{\text{real}}_r(x) - F^{\text{gen}}_r(x)\right|, \quad r_t = \log(S_{t+1}/S_t)$$

Two-sample Kolmogorov-Smirnov statistic on all pooled log-returns.
**Perfect: 0. ↓**

---

## A15 — Skewness Error

$$A_{15} = \left|\text{skew}(r^{\text{real}}) - \text{skew}(r^{\text{gen}})\right|$$

Heston generates negative skew ($\rho < 0$ leverage effect). **Perfect: 0. ↓**

---

## A16 — QQ RMSE (300-pt)

$$A_{16} = \sqrt{\frac{1}{300}\sum_{g=1}^{300}\left(Q^{\text{real}}_r(p_g) - Q^{\text{gen}}_r(p_g)\right)^2}, \quad p_g \in [0.005,\,0.995]$$

Tests **bulk distributional match** across 300 quantile levels. **Perfect: 0. ↓**

---

## A17 — Terminal Price KS

$$A_{17} = \text{KS}\!\left(S^{\text{real}}_T,\,S^{\text{gen}}_T\right)$$

Two-sample KS on the terminal marginal $S_T$. Tests terminal price distribution. **Perfect: 0. ↓**

---

# Adversarial Metric (A18)

## A18 — Discriminative Score · *Post-hoc binary classification accuracy offset*

$$
\text{DS} = \left| \text{Acc}_{\text{test}} - 0.5 \right|
$$

**Principle.** Train a binary classifier to separate real paths (label 1) from generated
paths (label 0). If the classifier cannot do better than random guessing (50 % accuracy),
the two distributions are indistinguishable and the generator is good.
Score 0 = perfect generator. Score 0.5 = trivially separated = bad generator.

### Data preparation

The full set of 8 192 real paths and 8 192 generated paths (16 384 total) is split
**80 % train / 20 % test** before any training. The test set is held out and never
seen during training.

### Training

Both classifiers are trained for **2 000 steps**, Adam (lr = 1×10⁻³), batch size 128,
Binary Cross-Entropy loss. Training loss is logged every 50 steps.

### GRU Discriminator — architecture

```
Input  : path  (T=128, d=1)
  └─ GRU(input=1, hidden=8, num_layers=2, batch_first=True)
       └─ last hidden state h_T  shape (8,)
            └─ Linear(8 → 1)  →  logit
```

The GRU processes the path **step by step**, accumulating a hidden state that encodes
the full temporal trajectory. Because of its recurrent structure it can detect
**temporal patterns** — autocorrelation, volatility clustering, mean-reversion speed —
that differ between real and generated data.

A GRU score close to 0 means the generated paths have **correct temporal dynamics**.

### MLP Discriminator — architecture

```
Input  : path  (T=128, d=1)  →  Flatten  →  (128,)
  └─ Linear(128 → 128)  →  ReLU
       └─ Linear(128 → 64)  →  ReLU
            └─ Linear(64 → 1)  →  logit
```

The MLP receives the **entire path flattened to a vector** with no notion of temporal
ordering. It can detect differences in the **joint marginal distribution** of all 128
values (price levels, global shape, statistical moments) but has no structural inductive
bias toward sequential patterns.

A MLP score close to 0 means the generated paths have **correct marginal statistics**.

### GRU vs MLP — what each tests

| | GRU | MLP |
|--|-----|-----|
| Inductive bias | Temporal / sequential | None (flat vector) |
| Sensitive to | Autocorrelation, clustering, dynamics | Price levels, moments, shape |
| Interpretation | Temporal structure correct | Marginal distribution correct |

A generator that passes GRU but fails MLP has good dynamics but wrong marginals.
A generator that passes MLP but fails GRU has correct marginals but wrong dynamics.

---

# Predictive Metric (A19)

## A19 — Predictive Score · *Train-on-Synthetic Test-on-Real (TSTR) MAE*

$$
\text{PS} = \frac{1}{N(T-1)}
\sum_{i=1}^{N} \sum_{t=1}^{T-1}
\left| \hat{X}_{i,t+1} - X_{i,t+1}^{\text{real}} \right|
$$

**Principle (TSTR).** A one-step-ahead predictor is trained **exclusively on generated
paths** (normalised to [0, 1]). It is then evaluated on **real paths** (same normalisation).
If the predictor generalises well — low MAE on real data — the synthetic data has captured
the true temporal dynamics. If it fails on real data, there is a distributional gap.

### Training

Both predictors are trained for **5 000 steps**, Adam (lr = 1×10⁻³), batch size 128,
L1 (MAE) loss. Training loss is logged every 100 steps.

### GRU Predictor — architecture

```
Input  : prefix  X[0:T-1]  shape (127, 1)
  └─ GRU(input=1, hidden=8, num_layers=2, batch_first=True)
       └─ output at every step  shape (127, 8)
            └─ Linear(8 → 1) applied at each step  →  (127, 1)
```

Sequence-to-sequence: predicts $\hat{X}_{t+1}$ for every $t$ simultaneously using the
**full causal history** $X_1,\ldots,X_t$. Can capture long-range dependencies
(mean-reversion, trend, volatility regime).

### MLP Predictor — architecture

```
Input  : local window  X[t-8:t]  shape (8, 1)  →  Flatten  →  (8,)
  └─ Linear(8 → 64)  →  ReLU
       └─ Linear(64 → 32)  →  ReLU
            └─ Linear(32 → 1)  →  X̂_{t+1}
```

Predicts $\hat{X}_{t+1}$ from only the **8 most recent steps**. Tests whether
short-range local patterns in the synthetic data match those in real data.

### GRU vs MLP — what each tests

| | GRU | MLP |
|--|-----|-----|
| Context used | Full history $X_1,\ldots,X_t$ | Last 8 steps only |
| Sensitive to | Long-range structure, regimes | Short-range momentum / micro-structure |
| Interpretation | Long-range temporal dynamics | Local (8-step) predictability |

**Evaluation.** After training on fake paths, both predictors are scored on all real paths.
PS = mean absolute error on normalised scale. **Perfect: 0. Direction: ↓**

---

# Temporal / Autocorrelation Metrics (A20–A24)

## A20 — Covariance Error · *Frobenius norm of terminal covariance difference*

$$
\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F
$$

where $\Sigma = \text{Cov}(X_T) \in \mathbb{R}^{d \times d}$.
For $d=1$ (Heston) reduces to $|\text{Var}(X_T^{\text{real}}) - \text{Var}(X_T^{\text{fake}})|$.
Tests the cross-asset covariance structure at maturity. **Perfect: 0. ↓**

---

## A21 — ACF |r| error (across lags) · *Mean absolute error of sample-mean ACF on |r|*

$$
\frac{1}{|L|} \sum_{\ell \in L}
\left|
  \frac{1}{N}\sum_{i=1}^{N} \text{ACF}(|r_i|, \ell)
  -
  \frac{1}{N}\sum_{i=1}^{N} \text{ACF}(|\tilde{r}_i|, \ell)
\right| \quad (L = \{1, 2, 5, 10\})
$$

where $\text{ACF}(q, \ell) = \dfrac{\sum_t (q_t - \bar{q})(q_{t+\ell} - \bar{q})}{\sum_t (q_t - \bar{q})^2}$.

Real financial returns have near-zero autocorrelation but $|r_t|$ and $r_t^2$ show
significant positive autocorrelation — the **ARCH / volatility clustering** stylised fact.
A21 tests this via absolute returns. **Perfect: 0. ↓**

---

## A22 — ACF r² error (across lags) · *Mean absolute error of sample-mean ACF on r²*

Same formula as A21 applied to squared returns $r_t^2$ instead of $|r_t|$.
Complementary to A21: also tests the ARCH effect but is more sensitive to large moves.
**Perfect: 0. ↓**

---

## A23 — ACF |r| Lag-1 Error · *Absolute error at lag 1 of the ACF of absolute log-returns*

$$
\left| \text{ACF}(|r_{\text{real}}|,\,\ell=1) - \text{ACF}(|r_{\text{gen}}|,\,\ell=1) \right|
$$

where $\text{ACF}(q,\ell)$ is the sample autocorrelation at lag $\ell$, averaged over paths.
Lag-1 absolute-return ACF is the dominant **ARCH signal** (volatility clustering).
Heston true value ≈ +0.052. **Perfect: 0. ↓**

Complements A21 (which averages over lags $\{1,2,5,10\}$): A23 isolates the single most
important lag.

---

## A24 — ACF r² Lag-1 Error · *Absolute error at lag 1 of the ACF of squared log-returns*

$$
\left| \text{ACF}(r_{\text{real}}^2,\,\ell=1) - \text{ACF}(r_{\text{gen}}^2,\,\ell=1) \right|
$$

Squared-return lag-1 ACF is the dominant **GARCH signal**.
Heston true value ≈ +0.050. **Perfect: 0. ↓**

Complements A22 (which averages over lags $\{1,2,5,10\}$): A24 isolates the single most
important lag.

---

# Volatility Metrics (A25–A32)

## A25 — Mean RMSE · *L2 norm of terminal mean difference*

$$
\left\| \mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T] \right\|_2
$$

Measures **systematic bias** in the generated terminal price level.
For $d=1$ this is simply $|\mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T]|$. **Perfect: 0. ↓**

---

## A26 — Return std. error · *Absolute difference of return standard deviations*

$$
\left| \sigma(r_{\text{real}}) - \sigma(r_{\text{fake}}) \right|
$$

Tests whether the overall **volatility level** is correctly reproduced. **Perfect: 0. ↓**

---

## A27 — Log-Return Std Error · *Absolute difference of log-return standard deviations*

$$
\left| \sigma(\log\text{-returns}_{\text{real}}) - \sigma(\log\text{-returns}_{\text{gen}}) \right|
$$

where $r_t = \log(S_{t+1}/S_t)$ are log-returns pooled across all paths and time steps.
Unlike A26 (which uses price increments $\Delta S_t$), A27 uses log-returns — the standard
volatility representation in quantitative finance.
Tests whether the overall **log-return volatility level** is correctly reproduced. **Perfect: 0. ↓**

---

## A28 — Kurtosis Ratio (target/model) · *Ratio of excess kurtosis values*

$$
\text{KR} = \frac{\kappa_{\text{real}}}{\kappa_{\text{gen}}}
$$

where $\kappa = \dfrac{\mathbb{E}[(r-\mu)^4]}{\sigma^4} - 3$ is the excess kurtosis (Fisher definition,
bias-corrected) of all pooled log-returns $r_t = \log(S_{t+1}/S_t)$.
A ratio of **1.0** means the generator reproduces the exact tail heaviness. Ratio > 1 means
the generated distribution is lighter-tailed than real; < 1 means heavier-tailed.
**Perfect: 1.0. Closer to 1 = better.**

---

## A29 — Sigma Mean Error · *Absolute error of mean per-path annualized volatility*

$$
\left| \frac{1}{N}\sum_{i=1}^{N} \hat{\sigma}_i^{\text{real}} - \frac{1}{N}\sum_{i=1}^{N} \hat{\sigma}_i^{\text{gen}} \right|
$$

where $\hat{\sigma}_i = \text{std}(r_{i,\cdot}) \times \sqrt{1/\Delta t}$ is the annualized realized
volatility of path $i$ (log-returns $r_{i,t} = \log(S_{i,t+1}/S_{i,t})$, $\Delta t = 1/250$).
Tests whether the **unconditional mean volatility** is correctly reproduced across paths.
**Perfect: 0. ↓**

---

## A30 — Cross-Sectional Vol Path RMSE

$$A_{30} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(\sigma^{\text{real}}_t - \sigma^{\text{gen}}_t\right)^2}$$

where $\sigma_t = \text{std}_i(S_{i,t})$ is the cross-sectional standard deviation at time $t$.
Tests whether the **spread of the path fan** is correctly reproduced over time. **Perfect: 0. ↓**

---

## A31 — Rolling Vol KS (window=5)

$$A_{31} = \text{KS}\!\left(\{\hat{\sigma}^{\text{real}}_{i,t}\},\,\{\hat{\sigma}^{\text{gen}}_{i,t}\}\right), \quad \hat{\sigma}_{i,t}=\text{std}(r_{i,t-4:t})$$

Two-sample KS on rolling log-return std (window=5). Tests the **volatility distribution**. **Perfect: 0. ↓**

---

## A32 — Vol-of-Vol Error

$$A_{32} = \left|\text{std}(\hat{\sigma}^{\text{real}}) - \text{std}(\hat{\sigma}^{\text{gen}})\right|$$

Dispersion of the rolling vol distribution — measures **volatility-of-volatility** reproduction. **Perfect: 0. ↓**

---

# Heston-Specific Metrics (A33–A34)

## A33 — Teacher-Sigma Correlation · *Pearson correlation, realised vol vs true vol*

$$
\rho = \text{Corr}\left( \hat{\sigma}^{\text{gen}}, \sqrt{v_{\text{true}}} \right)
$$

$\hat{\sigma}^{\text{gen}}_{i,t}$ = rolling realised volatility (window $w=5$) computed
from generated price increments:

$$
\hat{\sigma}_{i,t} = \sqrt{ \frac{1}{w} \sum_{s=t-w}^{t} r_{i,s}^2 + \varepsilon }
$$

$\sqrt{v_{\text{true},i,t}}$ = true instantaneous vol from the Heston latent variance
process (stored in `dataset/Heston/heston_v_8192x128.npy`).

**Heston-specific bonus metric.** Tests whether the generator reproduces the latent
stochastic volatility process, not just the price paths.
**Perfect: 1. Direction: ↑** (higher is better)

---

## A34 — Teacher-Sigma RMSE · *RMSE of realised vol vs true vol*

$$
\text{RMSE} = \sqrt{
  \frac{1}{N \cdot T}
  \sum_{i=1}^{N} \sum_{t=1}^{T}
  \left( \hat{\sigma}^{\text{gen}}_{i,t} - \sqrt{v_{\text{true},i,t}} \right)^2
}
$$

Complementary to the correlation: measures the absolute scale accuracy of the reproduced
volatility process. **Perfect: 0. Direction: ↓**

---

## B — Curve-Shape Metrics (6 plots × 3 sub-metrics)

The B metrics move beyond scalars: for each of 6 diagnostic plots, the **full shape of the
real and generated curves** is compared, including their first and second finite differences.
Three measures are reported per plot: an **MSE** variant (averaged over the three
sub-metrics) and a **% error** and **NRMSE** variant (both on the curve L only, funct-only).

### Derivative definition

Starting from a curve $L = [L_0, L_1, \ldots, L_K]$:

$$L^{\text{der}}_k = L_{k+1} - L_k \qquad \text{(first finite difference)}$$
$$L^{\text{sec}}_k = L^{\text{der}}_{k+1} - L^{\text{der}}_k \qquad \text{(second finite difference)}$$

For each plot, three sub-metrics are reported:

| Sub-metric | Curve compared | Interpretation |
|------------|----------------|----------------|
| `_funct` | $L^{\text{real}}$ vs $L^{\text{gen}}$ | Overall curve alignment |
| `_der` | $L^{\text{real,der}}$ vs $L^{\text{gen,der}}$ | First-derivative shape (slope match) |
| `_sec_der` | $L^{\text{real,sec}}$ vs $L^{\text{gen,sec}}$ | Second-derivative shape (curvature match) |

### The three measures

**MSE measure** (raw squared error):

$$\text{MSE}(a, b) = \frac{1}{K}\sum_{k} (b_k - a_k)^2$$

**% error measure** (mean absolute percentage error, MAPE):

$$\text{pcterr}(a, b) = \frac{1}{K}\sum_{k=1}^{K} \frac{|b_k - a_k|}{|a_k| + 10^{-6}} \times 100$$

i.e. a proper MAPE — the mean relative error (%) across the $K$ curve points. **One**
division only: the $\frac{1}{K}$ already averages over the curve's points (a prior version
divided a second time by $K$, collapsing the value ~$K\times$ into a sub-1% artefact).

**NRMSE measure** (range-normalised RMSE):

$$\text{NRMSE}(a, b) = \frac{\sqrt{\frac{1}{K}\sum_k (b_k - a_k)^2}}{\max_k a_k - \min_k a_k + 10^{-12}} \times 100$$

### Per-plot combined scores

For each plot the sub-metrics are combined into a single number per measure:

- **MSE combined** = $\frac{1}{3}(\text{funct} + \text{der} + \text{sec})$ — the **mean** of the
  three sub-metrics; cross-seed std = sample std of the per-seed combined values.
- **% err combined** and **NRMSE combined** = the **funct** sub-metric only (funct-only). The
  first and second differences of these curves are near-zero, so their relative error is
  ill-posed and would explode into meaningless $10^4$-% figures; only the MSE averages all three.

**Perfect floor: non-zero** — the residual finite-sample error of an independent Heston draw
scored against the test set, identical across methods. Direction: ↓

### The 6 plots

| Plot | Key prefix | Curve $L$ |
|------|-----------|-----------|
| Log-return histogram | `B_log_ret_hist_*` | Density of pooled log-returns over shared bins |
| QQ plot | `B_qq_plot_*` | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*` | Mean per-path $\text{ACF}(\|r\|, \ell)$ at each lag $\ell$ |
| ACF r² (lags 1–20) | `B_acf_sq_r_*` | Mean per-path $\text{ACF}(r^2, \ell)$ at each lag $\ell$ |
| Rolling vol hist. | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival | `B_tail_surv_*` | $P(\|r\| > x)$ at thresholds of real $\|r\|$ |

All 36 keys (18 MSE + 18 %, i.e. 6 plots × 3 sub-metrics × 2 measures) are computed by `metrics.compute_curve_metrics(S_real, S_gen)`.
`recompute_curve_b.py` aggregates them into the per-plot combined scores read by the READMEs.
The perfect-recovery floor for all B metrics is **non-zero** — the residual finite-sample error of an independent Heston draw scored against the test set, identical across methods.

> Ref: Cont (2001), *Quantitative Finance* 1, 223–236.

---

## Forecast Metrics — MAE, RMSE, CRPS

Used in the Path Shadowing Monte-Carlo evaluation.
Let $\{f_{i,k,h}\}_{k=1}^K$ be a weighted ensemble of $K$ forecasts with weights
$w_k \geq 0$, $\sum_k w_k = 1$, and $y_{i,h}$ the true future value.
$N$ = number of query paths, $H$ = forecast horizon.

### MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{N \cdot H} \sum_{i=1}^{N} \sum_{h=1}^{H} \left| \bar{f}_{i,h} - y_{i,h} \right|$$

where $\bar{f}_{i,h} = \sum_{k=1}^{K} w_k\, f_{i,k,h}$ is the weighted ensemble mean.

MAE measures **point forecast accuracy** of the ensemble mean. It does not reward
calibrated uncertainty — a wider ensemble is penalised if its mean drifts from $y$.

---

### RMSE — Root Mean Square Error

$$\text{RMSE} = \sqrt{\frac{1}{N \cdot H} \sum_{i=1}^{N} \sum_{h=1}^{H} \left( \bar{f}_{i,h} - y_{i,h} \right)^2}$$

Same structure as MAE but with squared errors — penalises large deviations more
strongly. Also applied to the ensemble mean only (not the full distribution).

---

### CRPS — Continuous Ranked Probability Score (energy form)

For a single (path $i$, step $h$) pair, the CRPS is:

$$\text{CRPS}_{i,h} = \underbrace{\sum_{k=1}^{K} w_k \left| f_{i,k,h} - y_{i,h} \right|}_{\text{accuracy}} - \underbrace{\frac{1}{2} \sum_{j=1}^{K} \sum_{k=1}^{K} w_j\, w_k \left| f_{i,j,h} - f_{i,k,h} \right|}_{\text{sharpness penalty}}$$

The reported scalar CRPS is the mean over all $N$ paths and $H$ steps:

$$\text{CRPS} = \frac{1}{N \cdot H} \sum_{i=1}^{N} \sum_{h=1}^{H} \text{CRPS}_{i,h}$$

**Interpretation of the two terms:**

| Term | Role |
|------|------|
| $\sum_k w_k \|f_k - y\|$ | Penalises ensemble members far from the truth |
| $-\tfrac{1}{2}\sum_{j,k} w_j w_k \|f_j - f_k\|$ | Rewards ensemble spread (sharpness) |

**Key properties:**
- **Proper scoring rule**: minimised in expectation when the ensemble exactly represents
  the true predictive distribution $p(y \mid x_{\text{past}})$.
- **Generalises MAE**: for a deterministic forecast ($K=1$), CRPS $=$ MAE.
- **Rewards calibrated uncertainty**: a wider ensemble is not penalised per se; an
  inaccurately wide one is (via the accuracy term).
- **Perfect score: 0. Direction: ↓**

**Why CRPS < MAE for PS-MC:** the ensemble mean has higher MAE than the naive
random-walk (deterministic) baseline, but CRPS is lower because the K-path fan-out
provides a correctly calibrated spread around the truth, which the RW cannot.
