# Metrics — A1 to A20

Evaluation suite for generative models of financial time series.
All metrics compare **real paths** $X \sim P$ against **generated paths** $\tilde{X} \sim Q$.

## Files

| File | Purpose |
|------|---------|
| `metrics_np.py` | A1–A12, A15–A16 — NumPy/SciPy implementations |
| `discriminative_score.py` | A13 — post-hoc GRU + MLP classifiers (PyTorch) |
| `predictive_score.py` | A14 — GRU + MLP predictors, TSTR protocol (PyTorch) |
| `compute_all.py` | Orchestrator: all metrics × N seeds → JSON + CSV + plots |

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

## A1 — Path MMD² · *Maximum Mean Discrepancy on full paths*

$$\text{MMD}^2(P, Q) = \frac{1}{N^2}\sum_{i,j} k(x_i, x_j) - \frac{2}{N^2}\sum_{i,j} k(x_i, \tilde{x}_j) + \frac{1}{N^2}\sum_{i,j} k(\tilde{x}_i, \tilde{x}_j)$$

Each path is flattened to $x = (X_1,\ldots,X_T) \in \mathbb{R}^{T \cdot d}$ before applying the kernel.
Tests whether the **joint temporal distribution** is reproduced. **Perfect: 0. Direction: ↓**

---

## A2 — Terminal MMD² · *Maximum Mean Discrepancy on terminal values*

Same biased estimator as A1, applied only to the terminal value $x = X_T \in \mathbb{R}^d$.
Tests whether the **marginal distribution at maturity** is correct. **Perfect: 0. ↓**

---

## A3 — Increment MMD² · *Maximum Mean Discrepancy on returns*

$$
\text{MMD}^2\left(\{r_t\}_{t < T}, \{\tilde{r}_t\}_{t < T}\right)
$$

where $r_t = X_{t+1} - X_t$. All increments pooled across time before computing MMD.
Tests the **return distribution** (mean, variance, tail shape). **Perfect: 0. ↓**

---

## A4 — Volatility MMD · *Sum of MMD² over volatility-related feature groups*

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

## A5 — Terminal SWD · *Sliced Wasserstein Distance on terminal values*

$$
\text{SWD}(P_T, Q_T)
= \mathbb{E}_{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}
  \left[ W_1\left(\theta_{\sharp} P_T, \theta_{\sharp} Q_T \right) \right]
$$

Approximated with **50 random projections**.
More robust to heavy tails and high dimensionality than MMD. **Perfect: 0. ↓**

---

## A6 — Path SWD · *Time-averaged Sliced Wasserstein Distance*

$$
\text{SWD}^{\text{path}}(P, Q)
= \frac{1}{T} \sum_{t=1}^{T} \text{SWD}(P_t, Q_t)
$$

where $P_t$ and $Q_t$ are the marginal distributions of real and generated paths at time $t$,
each estimated with **50 random projections**.
Tests whether the **marginal distribution is correct at every time step**, not just at maturity.
**Perfect: 0. ↓**

---

## A7 — Covariance Error · *Frobenius norm of terminal covariance difference*

$$
\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F
$$

where $\Sigma = \text{Cov}(X_T) \in \mathbb{R}^{d \times d}$.
For $d=1$ (Heston) reduces to $|\text{Var}(X_T^{\text{real}}) - \text{Var}(X_T^{\text{fake}})|$.
Tests the cross-asset covariance structure at maturity. **Perfect: 0. ↓**

---

## A8 — Mean RMSE · *L2 norm of terminal mean difference*

$$
\left\| \mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T] \right\|_2
$$

Measures **systematic bias** in the generated terminal price level.
For $d=1$ this is simply $|\mathbb{E}[X_T] - \mathbb{E}[\tilde{X}_T]|$. **Perfect: 0. ↓**

---

## A9 — Return std. error · *Absolute difference of return standard deviations*

$$
\left| \sigma(r_{\text{real}}) - \sigma(r_{\text{fake}}) \right|
$$

Tests whether the overall **volatility level** is correctly reproduced. **Perfect: 0. ↓**

---

## A10 — Return Kurtosis (target κ, model κ) · *Target and generated excess kurtosis*

$$
\kappa_{\text{real}} = \dfrac{\mathbb{E}[(r-\mu)^4]}{\sigma^4} - 3, \qquad
\kappa_{\text{fake}} = \dfrac{\mathbb{E}[(\tilde{r}-\mu)^4]}{\sigma^4} - 3
$$

Output: **target κ** (κ of real returns) and **model κ** (κ of generated returns).
Excess kurtosis = 0 for a Gaussian. Financial returns typically show $\kappa > 0$ (**fat tails**).
Computed using Fisher's definition with bias correction (`scipy.stats.kurtosis(fisher=True, bias=False)`).
**Perfect: target κ ≈ model κ.**

---

## A11 — ACF |r| error (across lags) · *Mean absolute error of sample-mean ACF on |r|*

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
A11 tests this via absolute returns. **Perfect: 0. ↓**

---

## A12 — ACF r² error (across lags) · *Mean absolute error of sample-mean ACF on r²*

Same formula as A11 applied to squared returns $r_t^2$ instead of $|r_t|$.
Complementary to A11: also tests the ARCH effect but is more sensitive to large moves.
**Perfect: 0. ↓**

---

## A13 — Discriminative Score · *Post-hoc binary classification accuracy offset*

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

## A14 — Predictive Score · *Train-on-Synthetic Test-on-Real (TSTR) MAE*

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

## A15 — Teacher-Sigma Correlation · *Pearson correlation, realised vol vs true vol*

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

## A15 — Teacher-Sigma RMSE · *RMSE of realised vol vs true vol*

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

## A16 — Tail Survival Error · *RMS of survival probability difference*

For each tail quantile level $\alpha \in \{0.90, 0.95, 0.99\}$:

- $q_\alpha$ = $\alpha$-quantile of real absolute returns $|r_t|$
- $\text{surv}_{\text{real}}(\alpha) = P_{\text{real}}(|r| > q_\alpha)$ (by construction $\approx 1 - \alpha$)
- $\text{surv}_{\text{gen}}(\alpha) = P_{\text{gen}}(|r| > q_\alpha)$

$$
\text{TSE}(P, Q)
= \sqrt{
  \frac{1}{|\mathcal{A}|}
  \sum_{\alpha \in \mathcal{A}}
  \left( \text{surv}_{\text{real}}(\alpha) - \text{surv}_{\text{gen}}(\alpha) \right)^2
}
\quad \mathcal{A} = \{0.90,\, 0.95,\, 0.99\}
$$

Tests whether the generator reproduces the **fat tail** of the return distribution at the
90th, 95th, and 99th percentile levels.
A score of 0 means survival probabilities match exactly at all three quantile levels.
**Perfect: 0. Direction: ↓**

Per-level outputs:

| Output | Description |
|--------|-------------|
| `rms` (A16_tail_survival) | RMS of survival probability differences across all levels |
| `r q90 error` | Survival probability error at the 90th percentile |
| `r q95 error` | Survival probability error at the 95th percentile |
| `ir q99 error` | Survival probability error at the 99th percentile |

---

## A17-A18 — Target oracle o mean · Generated a mean · *AR(p) prediction means*

Train an AR(p) predictor on **real** log-returns, predict held-out test returns $\rightarrow$ **oracle mean** $\mu_o$.
Train the same AR(p) on **generated** log-returns, predict the same test set $\rightarrow$ **agent mean** $\mu_g$.

$$
\mu_o = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \hat{r}_i^{(o)}, \qquad
\mu_g = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \hat{r}_i^{(g)}
$$

where $\hat{r}_i^{(o)} = \sum_{k=1}^p \beta_k^{(o)} r_{i-k}$ (AR(p) fitted on real data)
and $\hat{r}_i^{(g)}$ from AR(p) fitted on generated data.

Default: $p=5$, $20\%$ held out. **Perfect: match real mean. Direction: closer to real mean = better.**

---

## A19 — Learned oracle a corr. · *Pearson correlation*

$$
\rho_{o,g} = \frac{\sum_i (\hat{r}_i^{(o)} - \mu_o)(\hat{r}_i^{(g)} - \mu_g)}
{\sqrt{\sum_i (\hat{r}_i^{(o)} - \mu_o)^2} \sqrt{\sum_i (\hat{r}_i^{(g)} - \mu_g)^2}}
$$

Measures whether the generated data preserves the **same predictive relationships** as real data.
$\rho \approx 1$ = perfect preservation. $\rho \approx 0$ = signal destroyed.
**Perfect: 1. Direction: $\uparrow$**

---

## A20 — Realized-vol law loss · *Wasserstein-1 on RV distribution*

$$
\text{RV}_i = \frac{1}{\Delta t} \sum_{t=1}^{T-1} r_{i,t}^2 \quad \text{(annualized realized variance)}
$$

$$
\mathcal{L}_{\text{RV}} = W_1\big(P_{\text{real}}(\text{RV}), P_{\text{gen}}(\text{RV})\big)
= \int_{-\infty}^{\infty} \big| F_{\text{real}}^{\text{RV}}(x) - F_{\text{gen}}^{\text{RV}}(x) \big| \, dx
$$

where $W_1$ is the 1-Wasserstein (Earth Mover's) distance and $F$ the empirical CDF.
Tests whether the unconditional volatility regime distribution is reproduced.
**Perfect: 0. $\downarrow$**

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
