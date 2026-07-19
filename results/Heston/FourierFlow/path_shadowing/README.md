# Path Shadowing MC — FourierFlow on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

> The PS-MC evaluation is **model-agnostic**: it consumes only the generated
> `.npy` paths, so the embedding, retrieval and scoring are identical to the
> TimeGAN pipeline ([`../../TimeGAN/path_shadowing/README.md`](../../TimeGAN/path_shadowing/README.md)).
> Only the generated pool differs (FourierFlow instead of TimeGAN).

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
| **CRPS** | H=32 | **2.742 ± 0.027** | 2.743 ± 0.027 | 3.732 |
| MAE    | H=32 | 3.774 ± 0.028 | 3.774 ± 0.028 | 3.732 |
| RMSE   | H=32 | 5.185 ± 0.055 | 5.185 ± 0.055 | 5.068 |
| **CRPS** | H=64 | **3.992 ± 0.106** | 3.992 ± 0.106 | 5.301 |
| MAE    | H=64 | 5.461 ± 0.120 | 5.461 ± 0.121 | 5.301 |
| RMSE   | H=64 | 7.547 ± 0.197 | 7.548 ± 0.197 | 7.181 |

**PS-MC beats the naive RW on CRPS** at both horizons by a wide margin
(2.74 < 3.73 at H=32; 3.99 < 5.30 at H=64). FourierFlow's CRPS is lower than
TimeGAN's (2.74 vs 3.09 at H=32) — its generated pool provides tighter,
better-calibrated nearest-neighbour futures.

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good predictors.
The adaptive Gaussian provides no meaningful gain over uniform on this
synthetic data (unlike real S&P, where the paper reports gains from Gaussian weighting).

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE.
PS-MC improves CRPS because it provides a *calibrated ensemble* rather than a
point forecast. The ensemble MAE sits near the RW MAE by construction (shrinkage),
but CRPS rewards proper uncertainty quantification.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 2.728  | 2.697  | 2.773  | 2.756  | 2.758  |

The cross-seed spread is very tight (2.70–2.77, std 0.027) — every FourierFlow
training seed yields a pool of comparable forecasting quality, and all five
clear the RW baseline (3.732) comfortably.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston paths `heston_S_8192x128.npy` |
| Pool | FourierFlow generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) ≈ 9.0 |
| Horizons | H=32 (steps 64–95), H=64 (steps 64–127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved FourierFlow futures (anchored). Bold red = ensemble mean.

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

---

## Reproduce

```bash
cd methods/FourierFlow/path_shadowing
python run_eval.py     # 5 seeds in parallel → results/Heston/FourierFlow/path_shadowing/
```
