# Metrics — A1 to A15

Evaluation suite for generative models of financial time series.
All metrics compare a set of **real paths** $X \sim P$ against **generated paths** $\tilde{X} \sim Q$.

## Files

| File | Purpose |
|------|---------|
| `metrics_np.py` | A1–A12, A15 — NumPy/SciPy implementations |
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
| $r_t = X_{t+1} - X_t$ | Log-returns / increments |
| $k(x,x') = \exp\!\left(-\|x-x'\|^2/2\sigma^2\right)$ | RBF kernel |
| $W_1$ | 1-Wasserstein distance |
| $\theta_\sharp P$ | Projection of $P$ onto direction $\theta$ |

---

## A1 — Path MMD² · *Maximum Mean Discrepancy on full paths*

$$
\widehat{\text{MMD}}^2(P, Q)
= \frac{1}{N^2}\sum_{i,j} k(x_i,x_j)
- \frac{2}{N^2}\sum_{i,j} k(x_i,\tilde{x}_j)
+ \frac{1}{N^2}\sum_{i,j} k(\tilde{x}_i,\tilde{x}_j)
$$

Each path is flattened to $x = (X_1,\ldots,X_T)\in\mathbb{R}^{T\cdot d}$ before applying the kernel.
Tests whether the **joint temporal distribution** is reproduced. **Perfect: 0. Direction: ↓**

---

## A2 — Terminal MMD² · *Maximum Mean Discrepancy on terminal values*

Same estimator, applied only to the terminal value $x = X_T \in \mathbb{R}^d$.
Tests whether the **marginal at maturity** is correct. **Perfect: 0. ↓**

---

## A3 — Increment MMD² · *Maximum Mean Discrepancy on returns*

$$
\widehat{\text{MMD}}^2\!\left(\{r_t\}_{t<T},\,\{\tilde{r}_t\}_{t<T}\right),
\quad r_t = X_{t+1} - X_t
$$

All increments pooled across time before computing MMD.
Tests the **return distribution** (mean, variance, shape). **Perfect: 0. ↓**

---

## A4 — Volatility MMD · *Maximum Mean Discrepancy on realised volatility*

Rolling realised volatility with window $w = 5$:

$$
\hat{\sigma}_t = \sqrt{\frac{1}{w}\sum_{s=t-w}^{t-1} r_s^2}
$$

MMD applied to $\{\hat{\sigma}_t^{\text{real}}\}$ vs $\{\hat{\sigma}_t^{\text{fake}}\}$.
Tests **volatility clustering** and the distribution of the vol process. **Perfect: 0. ↓**

---

## A5 — Terminal SWD · *Sliced Wasserstein Distance on terminal values*

$$
\text{SWD}(P_T, Q_T)
= \mathbb{E}_{\theta\sim\mathcal{U}(\mathbb{S}^{d-1})}
  \!\left[W_1\!\left(\theta_\sharp P_T,\,\theta_\sharp Q_T\right)\right]
$$

Approximated with 512 random projections.
More robust to heavy tails and high dimensionality than MMD. **Perfect: 0. ↓**

---

## A6 — Path SWD · *Sliced Wasserstein Distance on full paths*

Same SWD formula with paths flattened to $x\in\mathbb{R}^{T\cdot d}$.
Captures the **geometry of the full path distribution**. **Perfect: 0. ↓**

---

## A7 — Covariance Error · *Frobenius norm of terminal covariance difference*

$$
\|\Sigma_{\text{real}} - \Sigma_{\text{fake}}\|_F,
\quad \Sigma = \text{Cov}(X_T)\in\mathbb{R}^{d\times d}
$$

For $d=1$ reduces to $|\text{Var}(X_T^{\text{real}}) - \text{Var}(X_T^{\text{fake}})|$.
Tests cross-asset covariance structure at maturity. **Perfect: 0. ↓**

---

## A8 — Mean RMSE · *Root Mean Square Error of terminal means*

$$
\sqrt{\frac{1}{d}\left\|\mathbb{E}[X_T]-\mathbb{E}[\tilde{X}_T]\right\|^2}
$$

Measures **systematic bias** in the generated terminal price level. **Perfect: 0. ↓**

---

## A9 — Return Std Error · *Mean Absolute Error of return standard deviation*

$$
\left|\,\sigma(r_{\text{real}}) - \sigma(r_{\text{fake}})\,\right|
$$

Tests whether the overall **volatility level** is correctly reproduced. **Perfect: 0. ↓**

---

## A10 — Return Kurtosis Error · *Mean Absolute Error of excess kurtosis*

$$
\left|\,\kappa(r_{\text{real}}) - \kappa(r_{\text{fake}})\,\right|,
\quad \kappa(Z) = \frac{\mathbb{E}[(Z-\mu)^4]}{\sigma^4} - 3
$$

Excess kurtosis = 0 for Gaussian; financial returns typically show $\kappa > 0$ (**fat tails**).
Tests whether the generator reproduces the tail behaviour. **Perfect: 0. ↓**

---

## A11 — ACF Error (abs returns) · *Mean Absolute Error of autocorrelation on |r|*

$$
\frac{1}{|L|}\sum_{\ell\in L}
\left|\,\text{ACF}(|r_{\text{real}}|,\ell) - \text{ACF}(|r_{\text{fake}}|,\ell)\,\right|,
\quad L = \{1,2,5,10\}
$$

Real financial returns have near-zero autocorrelation but $|r_t|$ and $r_t^2$ show
significant positive autocorrelation — the **ARCH / volatility clustering** stylised fact.
A11 tests this via absolute returns. **Perfect: 0. ↓**

---

## A12 — ACF Error (sq returns) · *Mean Absolute Error of autocorrelation on r²*

Same formula as A11 applied to squared returns $r_t^2$.
Complementary to A11: also tests the ARCH effect but is more sensitive to large moves.
**Perfect: 0. ↓**

---

## A13 — Discriminative Score · *Post-hoc binary classification accuracy offset*

$$
\text{DS} = \left|\,\text{Acc}_{\text{test}} - 0.5\,\right|
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
the full temporal trajectory. Because of its recurrent structure, it can detect
**temporal patterns** — autocorrelation, volatility clustering, mean-reversion speed —
that differ between real and generated data.

A GRU score close to 0 means the generated paths have **correct temporal dynamics**.

### MLP Discriminator — architecture

```
Input  : path  (T=128, d=1)  →  Flatten  →  (128,)
  └─ Linear(128 → 128)  →  ReLU
       └─ Linear(128 → 64)   →  ReLU
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
\text{PS} = \frac{1}{N\cdot(T-1)}
\sum_{i=1}^{N}\sum_{t=1}^{T-1}
\left|\hat{X}_{i,t+1} - X_{i,t+1}^{\text{real}}\right|
$$

**Principle (TSTR).** A one-step-ahead predictor is trained **exclusively on generated
paths**. It is then evaluated on **real paths**. If the predictor generalises well —
low MAE on real data — the synthetic data has captured the true temporal dynamics and
the generator is good. If the predictor trained on fake fails on real, there is a
distributional gap in the temporal structure.

### Training

Both predictors are trained for **5 000 steps**, Adam (lr = 1×10⁻³), batch size 128,
L1 (MAE) loss on min-max normalised paths. Training loss is logged every 100 steps.

### GRU Predictor — architecture

```
Input  : prefix  X[0:T-1]  shape (127, 1)
  └─ GRU(input=1, hidden=8, num_layers=2, batch_first=True)
       └─ output at every step  shape (127, 8)
            └─ Linear(8 → 1) applied at each step  →  (127, 1)
```

Sequence-to-sequence: predicts $\hat{X}_{t+1}$ for every $t$ simultaneously, using the
**full causal history** $X_1,\ldots,X_t$. Can capture long-range dependencies
(mean-reversion, trend, volatility regime).

### MLP Predictor — architecture

```
Input  : local window  X[t-8:t]  shape (8, 1)  →  Flatten  →  (8,)
  └─ Linear(8 → 64)   →  ReLU
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

### Evaluation

After training on fake, both predictors score all real paths.
PS = mean absolute error (normalised scale). **Perfect: 0.** In practice, compare against
a naïve baseline (e.g., last-value predictor) to contextualise the scale. **Direction: ↓**

---

## A15 — Teacher-Sigma Correlation · *Pearson correlation, realised vol vs true vol* ↑

$$
\rho = \text{Corr}\!\left(\hat{\sigma}^{\text{gen}},\,\sqrt{v_{\text{true}}}\right)
$$

$\hat{\sigma}^{\text{gen}}$ = rolling realised vol (window 5) from generated price paths.
$\sqrt{v_{\text{true}}}$ = true instantaneous vol from the Heston latent variance process.

**Heston-specific bonus metric.** Tests whether the generator reproduces the latent
stochastic volatility process, not just the price paths. **Perfect: 1. Direction: ↑**

---

## A15 — Teacher-Sigma RMSE · *RMSE of realised vol vs true vol*

$$
\text{RMSE} = \sqrt{
  \frac{1}{N\cdot T}\sum_{i,t}
  \!\left(\hat{\sigma}^{\text{gen}}_{i,t} - \sqrt{v_{\text{true},i,t}}\right)^2
}
$$

Complementary to the correlation: measures absolute scale accuracy of the reproduced
volatility process. **Perfect: 0. Direction: ↓**
