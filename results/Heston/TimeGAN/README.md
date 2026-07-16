# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**.

---

## Results (mean ± std across 5 seeds)

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|-----------|--------|--------|--------|--------|--------|---------|
| A1  | Path MMD²                   | 0.0180 ± 0.0147 | 0.0095 | 0.0035 | 0.0345 | 0.0054 | 0.0373 | **0** |
| A2  | Terminal MMD²               | 0.0296 ± 0.0235 | 0.0202 | 0.0086 | 0.0646 | 0.0051 | 0.0494 | **0** |
| A3  | Increment MMD²              | 0.0078 ± 0.0037 | 0.0054 | 0.0076 | 0.0117 | 0.0023 | 0.0121 | **0** |
| A4  | Volatility MMD              | 0.3798 ± 0.2351 | 0.1746 | 0.3673 | 0.6468 | 0.0709 | 0.6394 | **0** |
| A5  | Terminal SWD                | 2.850  ± 1.079  | 2.765  | 1.820  | 4.339  | 1.552  | 3.772  | **0** |
| A6  | Path SWD                    | 1.501  ± 0.583  | 1.279  | 0.839  | 2.349  | 1.021  | 2.015  | **0** |
| A7  | Covariance Error            | 17.75  ± 6.71   | 8.830  | 18.76  | 14.81  | 29.37  | 16.98  | **0** |
| A8  | Mean RMSE                   | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 | **0** |
| A9  | Return Std Error            | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 | **0** |
| A10 | Return Kurtosis Error       | 2.955  ± 2.099  | 0.015  | 5.360  | 3.768  | 0.958  | 4.672  | **0** |
| A11 | ACF Error (abs returns)     | 0.1339 ± 0.0728 | 0.0821 | 0.1065 | 0.2184 | 0.0421 | 0.2203 | **0** |
| A12 | ACF Error (sq returns)      | 0.0919 ± 0.0386 | 0.0588 | 0.0833 | 0.1318 | 0.0445 | 0.1412 | **0** |
| A13 | Discriminative Score (GRU)  | 0.0499 ± 0.0336 | 0.0468 | 0.0224 | 0.1088 | 0.0133 | 0.0581 | **0** |
| A13 | Discriminative Score (MLP)  | 0.1508 ± 0.1415 | 0.0368 | 0.0352 | 0.2901 | 0.0374 | 0.3544 | **0** |
| A14 | Predictive Score GRU (TSTR) | 0.0087 ± 0.0002 | 0.0085 | 0.0090 | 0.0085 | 0.0088 | 0.0085 | baseline |
| A14 | Predictive Score MLP (TSTR) | 0.0090 ± 0.0005 | 0.0090 | 0.0087 | 0.0090 | 0.0084 | 0.0099 | baseline |
| A15 | Sigma Corr ↑                | 0.0031 ± 0.0101 | 0.0008 | 0.0079 | −0.010 | −0.003 | 0.0196 | **1** |
| A15 | Sigma RMSE                  | 0.9659 ± 0.1237 | 0.9279 | 0.8392 | 1.0714 | 1.1474 | 0.8436 | **0** |

---

## Comparison with Yoon et al. NeurIPS 2019 (Table 2)

> ⚠️ **Not a direct comparison.** The paper evaluates on Sines (d=5, T=24) and Stocks (d=6)
> with a **2-layer LSTM** classifier. We evaluate on Heston (d=1, T=128) with GRU and MLP.

| Metric | Paper — Sines | Paper — Stocks | Ours — Heston GRU | Ours — Heston MLP |
|--------|:------------:|:-------------:|:-----------------:|:-----------------:|
| Disc Score ↓  | 0.011 ± 0.008 | 0.102 ± 0.021 | 0.050 ± 0.034 | 0.151 ± 0.142 |
| Pred Score ↓  | 0.093 ± 0.019 | 0.038 ± 0.001 | 0.009 ± 0.000 | 0.009 ± 0.001 |

Our GRU discriminative score (0.050) sits between the paper's Sines (0.011) and Stocks (0.102),
consistent with Heston being a moderately challenging 1-D financial process.
Our predictive score is lower than the paper's Sines result because Heston is 1-D and
next-step prediction is inherently simpler than 5-D Sines.

---

## v1 (buggy) → v2 (fixed) improvement

5 bugs fixed in `timegan_torch.py`: Recovery sigmoid, Phase-1 loss scaling,
Generator supervised coeff ×100, Embedder loss form, moment-matching gradient flow.

| Metric | v1 | v2 | Factor |
|--------|:--:|:--:|:------:|
| A1 Path MMD²        | 0.755 | **0.018** | 42× |
| A2 Terminal MMD²    | 0.741 | **0.030** | 25× |
| A3 Increment MMD²   | 0.233 | **0.008** | 29× |
| A7 Cov Error        | 199.6 | **17.75** | 11× |
| A9 Std Error        | 1.224 | **0.152** |  8× |
| A10 Kurtosis Error  | 71.44 | **2.955** | 24× |
| A13 Disc (GRU)      | 0.078 | **0.050** | 1.6× |
| A14 Pred GRU        | 0.053 | **0.009** |  6× |

---

## Metric Definitions

**Notation.**
$X \sim P$ = real paths $(N \times T \times d)$,
$\tilde{X} \sim Q$ = generated paths,
$r_t = X_{t+1} - X_t$ = log-returns / increments,
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
For $d=1$ (Heston) this reduces to $|\text{Var}(X_T^{\text{real}}) - \text{Var}(X_T^{\text{fake}})|$. **Perfect: 0.**

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

Tests whether the overall volatility level (annualised standard deviation of returns) is correct. **Perfect: 0.**

---

### A10 — Return Kurtosis Error · *Mean Absolute Error of excess kurtosis*

$$
\left|\,\kappa(r_{\text{real}}) - \kappa(r_{\text{fake}})\,\right|,
\quad \kappa(Z) = \frac{\mathbb{E}[(Z-\mu)^4]}{\sigma^4} - 3
$$

Tests fat-tail reproduction. Excess kurtosis = 0 for Gaussian; financial returns typically show $\kappa > 0$. **Perfect: 0.**

---

### A11 — ACF Error (abs returns) · *Mean Absolute Error of autocorrelation function on |r|*

$$
\frac{1}{|L|}\sum_{\ell \in L}
\left|\,\text{ACF}(|r_{\text{real}}|, \ell) - \text{ACF}(|r_{\text{fake}}|, \ell)\,\right|,
\quad L = \{1, 2, 5, 10\}
$$

Tests **volatility clustering**: real financial returns have near-zero autocorrelation but
$|r_t|$ and $r_t^2$ exhibit significant positive autocorrelation (ARCH effect). **Perfect: 0.**

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
Binary Cross-Entropy loss. Loss is logged every 50 steps → `seed_{i}_disc_{arch}_loss.csv`.

---

#### GRU Discriminator

```
Input: path (128, 1)
  └─ GRU, hidden = 8, num_layers = 2, batch_first = True
       └─ last hidden state h_T  (8,)
            └─ Linear(8 → 1)  → logit
```

The GRU reads the path **step by step**, building a hidden state that encodes the full
temporal trajectory. Because GRUs have an explicit recurrence, the classifier can exploit
**temporal patterns** — autocorrelation structure, volatility clustering, mean-reversion —
to distinguish real from fake.

Score 0.050 ± 0.034: the GRU finds generated paths nearly indistinguishable, meaning
**temporal dynamics are well reproduced**.

---

#### MLP Discriminator

```
Input: path (128, 1) → Flatten → (128,)
  └─ Linear(128 → 128) → ReLU
       └─ Linear(128 → 64) → ReLU
            └─ Linear(64 → 1)  → logit
```

The MLP receives the **entire path as a flat vector** with no notion of ordering or
recurrence. It can only detect differences in the **joint marginal distribution** of all
128 values simultaneously (price levels, global shape, statistical moments) but has no
structural inductive bias toward temporal dependencies.

Score 0.151 ± 0.142: higher than GRU, meaning the MLP occasionally finds distributional
differences that the GRU misses. This is expected — the flat representation makes it
sensitive to any non-temporal statistical mismatch (e.g., price scale, overall trend).

**Key difference between GRU and MLP:**
- GRU ≈ test for *temporal structure* (autocorrelation, clustering, dynamics).
- MLP ≈ test for *marginal distributional match* (moments, price levels, shape).
A generator that passes GRU but fails MLP has correct dynamics but wrong marginals; vice
versa means correct marginals but wrong dynamics.

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
on real data), the synthetic data has captured the true temporal dynamics — the generator
is good. If the predictor fails on real data despite fitting synthetic data, there is a
distributional gap.

**Training.**
Both predictors are trained for 5 000 steps with Adam (lr = 1 × 10⁻³), batch size 128,
L1 (MAE) loss on normalised paths. Loss is logged every 100 steps → `seed_{i}_pred_{arch}_loss.csv`.

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
It can exploit **long-range dependencies** across the entire sequence.

---

#### MLP Predictor

```
Input: sliding window X[t-8:t]  shape (8, 1)
  └─ Flatten → (8,)
       └─ Linear(8 → 64) → ReLU
            └─ Linear(64 → 32) → ReLU
                 └─ Linear(32 → 1) → X̂_{t+1}
```

The MLP predicts $X_{t+1}$ from a **local context window of the 8 most recent steps**
only. It tests whether **short-range temporal patterns** in the synthetic data match
those in real data. It has no memory beyond the 8-step window.

**Key difference between GRU and MLP:**
- GRU ≈ test for *long-range temporal predictability* (mean-reversion, trends, vol regimes).
- MLP ≈ test for *local (8-step) temporal predictability* (short-term momentum, micro-structure).

**Evaluation.** Both predictors are evaluated on all 8 192 real paths.
PS = mean absolute error on next-step predictions, in the normalised [0, 1] scale.
**Perfect score = 0** (zero prediction error, unreachable in practice). A score matching
the real-data auto-regression baseline means synthetic data has the same short-range
predictability structure as real data.

---

### A15 — Teacher-Sigma Correlation · *Pearson correlation, realised vol vs true vol* ↑

$$
\rho = \text{Corr}\!\left(\hat{\sigma}^{\text{gen}},\, \sqrt{v_{\text{true}}}\right)
$$

$\hat{\sigma}^{\text{gen}}$ = rolling realised volatility (window 5) computed from generated
price paths. $\sqrt{v_{\text{true}}}$ = true instantaneous vol from the Heston latent process
(stored in `dataset/Heston/heston_v_8192x128.npy`).

A generator that reproduces stochastic volatility correctly should show $\rho \approx 1$.
**Heston-specific metric — not in Yoon et al.** **Perfect: 1 ↑.**

---

### A15 — Teacher-Sigma RMSE · *Root Mean Square Error, realised vol vs true vol*

$$
\text{RMSE} = \sqrt{\frac{1}{N \cdot T}
\sum_{i=1}^{N}\sum_{t=1}^{T}
\!\left(\hat{\sigma}^{\text{gen}}_{i,t} - \sqrt{v_{\text{true},i,t}}\right)^2}
$$

Complementary to the correlation: measures absolute scale accuracy of the reproduced
volatility process. **Perfect: 0.**

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step (every 100 steps) |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step (every 100 steps) |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
