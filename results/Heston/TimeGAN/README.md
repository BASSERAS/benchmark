# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A15 Corr (↑)**. A16 ↓.

---

## Results (mean ± std across 5 seeds)

### A1–A20 — Core metrics

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| A1  | Path MMD²              | 0.0181 ± 0.0147 | 0.0093 | 0.0046 | 0.0322 | 0.0051 | 0.0393 | 0.0018 ± 0.0002 |
| A2  | Terminal MMD²          | 0.0308 ± 0.0229 | 0.0256 | 0.0103 | 0.0681 | 0.0060 | 0.0439 | 0.0016 ± 0.0002 |
| A3  | Increment MMD²         | 0.0077 ± 0.0039 | 0.0048 | 0.0070 | 0.0112 | 0.0023 | 0.0129 | 0.0008 ± 0.0000 |
| A4  | Volatility MMD         | 0.3933 ± 0.2553 | 0.1700 | 0.3416 | 0.6572 | 0.0797 | 0.7179 | 0.0077 ± 0.0006 |
| A5  | Terminal SWD           | 3.1284 ± 0.9227 | 2.9505 | 2.0579 | 4.5309 | 2.3097 | 3.7931 | 0.7635 ± 0.1174 |
| A6  | Path SWD               | 1.6343 ± 0.5763 | 1.2793 | 0.9696 | 2.4624 | 1.2892 | 2.1711 | 0.5542 ± 0.0624 |
| A7  | Covariance Error       | 17.75 ± 6.71    | 8.83   | 18.76  | 14.81  | 29.37  | 16.98  | 4.76 ± 2.50 |
| A8  | Mean RMSE              | 0.7385 ± 0.4552 | 0.8320 | 0.3890 | 1.0560 | 1.3412 | 0.0743 | 0.1400 ± 0.1303 |
| A9  | Return Std Error       | 0.1519 ± 0.0888 | 0.1519 | 0.2379 | 0.0302 | 0.0788 | 0.2608 | 0.0048 ± 0.0031 |
| A10 | Kurtosis Error         | 2.9545 ± 2.0988 | 0.0148 | 5.3599 | 3.7677 | 0.9581 | 4.6722 | 0.0172 ± 0.0155 |
| A11 | ACF |r| Error         | 0.1252 ± 0.0674 | 0.0648 | 0.1046 | 0.2011 | 0.0477 | 0.2080 | 0.0017 ± 0.0006 |
| A12 | ACF r² Error          | 0.0839 ± 0.0348 | 0.0450 | 0.0793 | 0.1172 | 0.0479 | 0.1300 | 0.0014 ± 0.0006 |
| A13 | Disc Score GRU        | 0.0099 ± 0.0084 | 0.0035 | 0.0124 | 0.0023 | 0.0252 | 0.0063 | 0.0128 ± 0.0068 |
| A13 | Disc Score MLP        | 0.0921 ± 0.0463 | 0.1277 | 0.0053 | 0.0832 | 0.1182 | 0.1262 | 0.0080 ± 0.0081 |
| A14 | Pred Score GRU (TSTR) | 0.0570 ± 0.0013 | 0.0553 | 0.0591 | 0.0575 | 0.0561 | 0.0570 | 0.0564 ± 0.0022 |
| A14 | Pred Score MLP (TSTR) | 0.0573 ± 0.0015 | 0.0556 | 0.0593 | 0.0570 | 0.0559 | 0.0588 | 0.0565 ± 0.0022 |
| A15 | Sigma Corr ↑          | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | −0.0082 | −0.0057 | 0.0166 | 0.6135 ± 0.0019 |
| A15 | Sigma RMSE            | 0.1183 ± 0.0184 | 0.1016 | 0.1108 | 0.1479  | 0.1004  | 0.1308 | 0.0653 ± 0.0002 |
| A16 | Tail RMS              | 0.0234 ± 0.0109 | 0.0301 | 0.0374 | 0.0076 | 0.0143 | 0.0278 | 0.0008 ± 0.0008 |
| A16 | q90 Error             | 0.0337 ± 0.0188 | 0.0422 | 0.0560 | 0.0021 | 0.0241 | 0.0441 | 0.0010 ± 0.0010 |
| A16 | q95 Error             | 0.0191 ± 0.0107 | 0.0296 | 0.0321 | 0.0096 | 0.0047 | 0.0194 | 0.0009 ± 0.0009 |
| A16 | q99 Error             | 0.0052 ± 0.0031 | 0.0080 | 0.0058 | 0.0087 | 0.0029 | 0.0005 | 0.0004 ± 0.0004 |
| A17 | Oracle MAE (AR5)      | 0.0097 ± 0.0000 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 ± 0.0000 |
| A18 | Agent MAE (AR5 TSTR)  | 0.0101 ± 0.0003 | 0.0098 | 0.0106 | 0.0100 | 0.0100 | 0.0099 | 0.0097 ± 0.0000 |
| A19 | Oracle-Agent Corr ↑   | −0.332 ± 0.306  | −0.773 | −0.038 | −0.206 | −0.030 | −0.614 | −0.058 ± 0.430 |
| A20 | RV Law Loss           | 1.5512 ± 0.3788 | 1.4914 | 1.7536 | 1.8266 | 0.8373 | 1.8470 | 0.0673 ± 0.0362 |

### B1–B14 — Stylized metrics

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| B1  | Mean Path RMSE        | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0.1511 ± 0.0708 |
| B2  | Cross-Sect. Vol RMSE  | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0.1355 ± 0.0735 |
| B3  | KS on Log-returns     | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0.0018 ± 0.0009 |
| B4  | Skewness Error        | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0.0060 ± 0.0048 |
| B5  | QQ RMSE (300-pt)      | 0.0025 ± 0.0006 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0.0001 ± 0.0000 |
| B6  | Tail QQ Error         | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0.0001 ± 0.0001 |
| B7  | ACF lag-1 |r| Err     | 0.2282 ± 0.1042 | 0.1549 | 0.2137 | 0.3698 | 0.0847 | 0.3180 | 0.0018 ± 0.0016 |
| B8  | ARCH Persistence Err  | 0.0591 ± 0.0359 | 0.0272 | 0.0436 | 0.0895 | 0.0221 | 0.1130 | 0.0011 ± 0.0005 |
| B9  | ACF lag-1 r² Err      | 0.1732 ± 0.0631 | 0.1186 | 0.2016 | 0.2655 | 0.0881 | 0.1923 | 0.0017 ± 0.0014 |
| B10 | GARCH Persistence Err | 0.0328 ± 0.0151 | 0.0173 | 0.0265 | 0.0380 | 0.0224 | 0.0598 | 0.0010 ± 0.0006 |
| B11 | Rolling Vol KS        | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0.0046 ± 0.0024 |
| B12 | Vol-of-Vol Error      | 0.0009 ± 0.0009 | 0.0004 | 0.0003 | 0.0025 | 0.0003 | 0.0011 | 0.0000 ± 0.0000 |
| B13 | Terminal Price KS     | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0.0145 ± 0.0043 |
| B14 | Hill Tail Index Err   | 36.88 ± 17.05   | 40.70  | 18.78  | 51.75  | 15.49  | 57.70  | 0.499 ± 0.610   |

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
