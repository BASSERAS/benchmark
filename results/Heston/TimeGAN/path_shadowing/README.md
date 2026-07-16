# Path Shadowing MC — TimeGAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

---

## Method

### Step 1 — Multi-scale log-return embedding (eq. 13)

Given a path prefix of `prefix_len` price steps, we embed it as a vector of
**multi-scale cumulative log-returns**, normalised by a power-law:

$$h_{\alpha,\beta}(x)[m] = \frac{\log S_t - \log S_{t-\ell_m}}{\ell_m^\beta}, \quad \ell_m = \lfloor \alpha^m \rfloor, \quad m = 1, 2, \ldots$$

**Optimal parameters** (calibrated on S&P data in Table I of the paper):
`α = 1.15`, `β = 0.9`.

With `prefix_len = 64` (63 available log-returns), this yields **M = 22 unique lags**:

```
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 28, 32, 37, 43, 50, 57}
```

producing a **22-dimensional embedding** (the paper achieves dim=34 with its longer
126-day prefix; we are limited by the Heston path length of 128 steps).

Why this embedding beats flat log-returns:
- **Recency bias**: small lags ℓ are divided by ℓ^0.9 ≈ small → recent dynamics weighted more
- **Lower dimension**: 22D vs 63D → less curse of dimensionality for KNN
- **Logarithmic spacing**: coarser sampling of the distant past, scale-invariant

### Step 2 — KNN retrieval (NOT combinatorial)

For every real query path `x̃_past`:

```python
# O(N × M) scan — sklearn NearestNeighbors, L2 metric
distances, indices = nn.kneighbors(h(x̃_past))  # returns K smallest distances
```

`K = 77` generated paths with the smallest L2 distance in embedding space are
selected. **No subset enumeration, no combinatorial search** — just a vectorised
distance computation over the pool of 8 192 generated paths.

### Step 3 — Price anchoring

Each retrieved fake future is multiplicatively scaled so it starts at the real
path's last prefix price, removing the price-level offset:

$$\tilde{S}^{(k)}_{\text{anchored}}(u) = S^{(k)}_{\text{fake}}(u) \times \frac{S_{\text{real}}(t)}{S^{(k)}_{\text{fake}}(t)}, \quad u > t$$

### Step 4 — Two weighting variants

**Uniform:** flat weight `1/K` on all K retrieved futures.

**Gaussian:** distance-weighted with per-query bandwidth:

$$w_k \propto \exp\!\left(-\frac{\|h(x^k_{\text{past}}) - h(\tilde{x}_{\text{past}})\|^2}{2\,\eta^2}\right), \quad \eta = \tilde{\eta} \cdot \|h(\tilde{x}_{\text{past}})\|$$

The paper proposes `η̃ = 0.075` calibrated on S&P data. **We re-calibrate η̃ for
our Heston data** using:

$$\tilde{\eta}_{\text{adapt}} = \frac{\text{median}(\text{distances})}{\text{median}(\|h\|)}$$

This preserves the per-query scaling idea while being dataset-neutral.
Using the paper's raw `η̃ = 0.075` collapses weights onto a single nearest
neighbour (embedding norms differ between S&P and Heston), which gives
CRPS worse than the random-walk baseline.

### Step 5 — Evaluation

Forecast = weighted average of the K anchored futures.
Evaluated at two horizons **H=32** (steps 64–95) and **H=64** (steps 64–127)
using CRPS (proper scoring rule), MAE, and RMSE.

---

## Results (mean ± std across 5 seeds) — multi-scale embedding, α=1.15, β=0.9

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline |
|--------|---------|---------|----------------------|-------------------|
| **CRPS** | H=32 | **3.134 ± 0.450** | 3.137 ± 0.451 | 3.732 |
| MAE    | H=32 | 4.059 ± 0.222 | 4.062 ± 0.222 | 3.732 |
| RMSE   | H=32 | 5.481 ± 0.293 | 5.484 ± 0.293 | 5.068 |
| **CRPS** | H=64 | **4.410 ± 0.560** | 4.415 ± 0.561 | 5.301 |
| MAE    | H=64 | 5.669 ± 0.216 | 5.673 ± 0.215 | 5.301 |
| RMSE   | H=64 | 7.662 ± 0.291 | 7.667 ± 0.291 | 7.181 |

**PS-MC beats the naive RW on CRPS** at both horizons (3.13 < 3.73 at H=32; 4.41 < 5.30 at H=64).

**Uniform ≈ Gaussian** for simple Heston paths: Heston is time-homogeneous with
constant parameters, so all K near neighbours are roughly equally good predictors.
The adaptive Gaussian provides no meaningful gain over uniform on this synthetic data
(unlike on real S&P, where the paper reports gains from the Gaussian weighting).

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE.
PS-MC improves CRPS because it provides a *calibrated ensemble* rather than a point.
The ensemble MAE is higher than RW-MAE by construction (the ensemble mean is a shrunk
version of each scenario), but CRPS rewards proper uncertainty quantification.

---

## Comparison: flat log-returns vs multi-scale embedding

| Embedding | CRPS h32 uniform | CRPS h32 gaussian | Notes |
|-----------|:----------------:|:-----------------:|-------|
| Flat (63D, previous) | 3.097 ± 0.296 | 3.098 ± 0.297 | simple, works with median η |
| **Multi-scale (22D, eq.13)** | **3.134 ± 0.450** | **3.137 ± 0.451** | paper-correct; higher variance |

The multi-scale embedding gives nearly identical CRPS on Heston. The S&P-calibrated
embedding was designed to capture non-stationary, multi-scale dynamics absent in the
Heston model (constant parameters, time-homogeneous). On more realistic datasets the
multi-scale embedding would be expected to outperform.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 2.907  | 2.990  | 2.980  | 2.772  | 4.020  |

Seed 4 is again the weakest (same as with flat embedding): highest disc score,
highest tail error — indicating generated paths of lower distributional quality.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston paths `heston_S_8192x128.npy` |
| Pool | TimeGAN generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | Multi-scale log-returns, α=1.15, β=0.9, dim=22 |
| K | 77 nearest neighbours (L2 in embedding space) |
| η̃ | Adaptive: median(dist) / median(‖h‖) |
| Horizons | H=32 (steps 64–95), H=64 (steps 64–127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved TimeGAN futures (anchored). Bold red = ensemble mean.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

PS-MC stays below the RW baseline at both H=32 and H=64.

---

## Files

| File | Contents |
|------|---------|
| `seed_{0..4}_results.json` | Per-seed metrics (eta, CRPS/MAE/RMSE at h32/h64, both variants) |
| `summary.json` | Mean ± std across seeds + baseline + per_seed array |
| `plots/ps_mc_example.png` | Fan-out illustration (seed 0) |
| `plots/crps_per_step.png` | CRPS per forecast step (5 seeds, mean ± std) |
