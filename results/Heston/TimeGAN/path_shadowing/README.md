# Path Shadowing MC — TimeGAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

---

## Method

### Step 1 — 65D murex-style prefix embedding

Given a path prefix of `prefix_len = 64` price steps, we embed it as a
**65-dimensional feature vector** adapted from Murex's internal implementation
(`deep-mkv-gen-path-dt/experiments/path_dt_experiments/shadowing.py`):

| Component | Dimension | Formula |
|-----------|-----------|---------|
| Full log-return trajectory | 63 | `r_t = log S_t − log S_{t−1}`, t = 1…63 |
| Terminal cumulative return | 1 | `R = log S_63 − log S_0` |
| Realized volatility | 1 | `σ = sqrt(mean(r_t²))` |
| **Total** | **65** | — |

Each dimension is **z-scored** using the mean and std of the generated pool,
making distances scale-invariant across features:

```python
mean = fake_emb.mean(axis=0)   # per-dimension, from generated pool
std  = fake_emb.std(axis=0)
z_real = (real_emb - mean) / std
z_fake = (fake_emb - mean) / std
```

**Why 65D outperforms the paper's 22D eq.(13) embedding:**
The eq.(13) embedding captures only endpoint-relative multi-scale differences
(market regime), not the full path shape. Two paths with different trajectories
can share the same 22D vector if their endpoint-to-lag differences coincide.
The 65D embedding retains the full return trajectory, so KNN selects paths
whose **prefix shape** is closest — directly relevant to forecasting.

### Step 2 — KNN retrieval (NOT combinatorial)

For every real query path `x̃_past`:

```python
# O(N × D) scan — sklearn NearestNeighbors, L2 metric in z-scored space
distances, indices = nn.kneighbors(z_real)  # returns K smallest distances
```

`K = 77` generated paths with the smallest L2 distance in the z-scored
65D space are selected. **No subset enumeration, no combinatorial search.**

### Step 3 — Price anchoring

Each retrieved fake future is multiplicatively scaled so it starts at the real
path's last prefix price, removing the price-level offset:

$$\tilde{S}^{(k)}_{\text{anchored}}(u) = S^{(k)}_{\text{fake}}(u) \times \frac{S_{\text{real}}(t)}{S^{(k)}_{\text{fake}}(t)}, \quad u > t$$

### Step 4 — Two weighting variants

**Uniform:** flat weight `1/K` on all K retrieved futures.

**Gaussian:** distance-weighted with per-query adaptive bandwidth:

$$w_k \propto \exp\!\left(-\frac{\|z^k_{\text{past}} - z_{\text{query}}\|^2}{2\,\eta_i^2}\right), \quad \eta_i = \tilde{\eta} \cdot \|z(\tilde{x}_{\text{past},i})\|$$

Adaptive calibration (dataset-neutral):

$$\tilde{\eta}_{\text{adapt}} = \frac{\text{median}(\text{distances})}{\text{median}(\|z\|)}$$

Using the paper's raw `η̃ = 0.075` (calibrated on S&P) collapses weights onto
a single nearest neighbour on Heston data, giving CRPS worse than the
random-walk baseline. The adaptive η̃ preserves the per-query scaling idea
while being dataset-neutral.

### Step 5 — Evaluation

Forecast = weighted average of the K anchored futures.
Evaluated at two horizons **H=32** (steps 64–95) and **H=64** (steps 64–127)
using CRPS (proper scoring rule), MAE, and RMSE.

---

## Results (mean ± std across 5 seeds) — 65D murex embedding

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline |
|--------|---------|---------|----------------------|-------------------|
| **CRPS** | H=32 | **3.087 ± 0.340** | 3.087 ± 0.341 | 3.732 |
| MAE    | H=32 | 4.039 ± 0.228 | 4.039 ± 0.229 | 3.732 |
| RMSE   | H=32 | 5.452 ± 0.293 | 5.452 ± 0.293 | 5.068 |
| **CRPS** | H=64 | **4.372 ± 0.431** | 4.373 ± 0.432 | 5.301 |
| MAE    | H=64 | 5.680 ± 0.178 | 5.681 ± 0.179 | 5.301 |
| RMSE   | H=64 | 7.667 ± 0.203 | 7.668 ± 0.203 | 7.181 |

**PS-MC beats the naive RW on CRPS** at both horizons (3.09 < 3.73 at H=32; 4.37 < 5.30 at H=64).

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good predictors.
The adaptive Gaussian provides no meaningful gain over uniform on this
synthetic data (unlike real S&P, where the paper reports gains from Gaussian weighting).

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE.
PS-MC improves CRPS because it provides a *calibrated ensemble* rather than a
point forecast. The ensemble mean has higher MAE than RW by construction (shrinkage),
but CRPS rewards proper uncertainty quantification.

---

## Embedding comparison

| Embedding | CRPS h32 uniform | Std | Notes |
|-----------|:----------------:|:---:|-------|
| Multi-scale 22D eq.(13) | 3.134 | 0.450 | Paper's S&P-calibrated, high variance |
| **Murex 65D (current)** | **3.087** | **0.340** | Full trajectory, z-scored; lower CRPS and variance |

The 65D embedding gives both lower mean CRPS (−1.5%) and substantially lower
cross-seed variance (std −24%). The lower variance reflects more stable KNN
matches: the full trajectory captures path shape rather than only endpoint-regime.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 3.021  | 2.795  | 3.093  | 2.800  | 3.725  |

Seed 4 remains the weakest (highest disc score, highest tail error — generated
paths of lower distributional quality). The gap to other seeds is smaller than
with the 22D embedding (seed 4 was 4.020 there), confirming that the 65D
embedding is more robust to generator quality.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston paths `heston_S_8192x128.npy` |
| Pool | TimeGAN generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) |
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
