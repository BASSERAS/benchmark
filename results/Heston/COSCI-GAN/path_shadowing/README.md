# Path Shadowing MC — COSCI-GAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

> The PS-MC evaluation is **model-agnostic**: it consumes only the generated
> `.npy` paths, so the embedding, retrieval and scoring are identical to the
> Diffusion-TS, FourierFlow, TimeGAN and TimeVQVAE pipelines
> ([`../../DiffusionTS/path_shadowing/README.md`](../../DiffusionTS/path_shadowing/README.md)).
> Only the generated pool differs (COSCI-GAN instead of the others).

---

## Method

### Step 1 — 65D murex-style prefix embedding

Given a path prefix of `prefix_len = 64` price steps, we embed it as a
**65-dimensional feature vector** adapted from Murex's internal implementation:

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
a single nearest neighbour on Heston data. The adaptive η̃ preserves the
per-query scaling idea while being dataset-neutral.

### Step 5 — Evaluation

Forecast = weighted average of the K anchored futures.
Evaluated at two horizons **H=32** (steps 64–95) and **H=64** (steps 64–127)
using CRPS (proper scoring rule), MAE, and RMSE.

---

## Results (mean ± std across 5 seeds) — 65D murex embedding

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline |
|--------|---------|---------|----------------------|-------------------|
| **CRPS** | H=32 | 4.657 ± 0.775 | 4.656 ± 0.773 | **3.732** |
| MAE    | H=32 | 6.030 ± 0.891 | 6.027 ± 0.888 | 3.732 |
| RMSE   | H=32 | 7.613 ± 0.940 | 7.610 ± 0.938 | 5.068 |
| **CRPS** | H=64 | 5.834 ± 0.763 | 5.834 ± 0.764 | **5.301** |
| MAE    | H=64 | 7.674 ± 0.866 | 7.673 ± 0.866 | 5.301 |
| RMSE   | H=64 | 9.782 ± 0.944 | 9.780 ± 0.944 | 7.181 |

**PS-MC over the COSCI-GAN pool does NOT beat the naive RW.** CRPS 4.657 > 3.732
at H=32 and 5.834 > 5.301 at H=64 — the naive random walk is a *better* forecaster
than the COSCI-GAN nearest-neighbour ensemble at both horizons. This is the direct
downstream consequence of the metric table in
[`../README.md`](../README.md): COSCI-GAN reproduces the marginal centre but carries
**no Heston volatility structure** (A9 vol-MMD 3.96, A31 rolling-vol KS 0.937,
A33 σ-corr ≈ 0). Its generated paths therefore supply **uninformative** nearest
neighbours — retrieval finds prefix-similar paths whose futures diverge no better
than a driftless RW. This contrasts sharply with Diffusion-TS and TimeVQVAE, whose
pools clear RW at both horizons (CRPS ≈ 2.72–2.77 / 3.85–3.89).

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good (equally poor,
here) predictors. The adaptive Gaussian provides no meaningful gain over uniform.

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE.
For a pool with genuine structure PS-MC improves CRPS via a *calibrated ensemble*;
for COSCI-GAN the ensemble is mis-calibrated relative to the true dynamics, so it
falls behind the point-forecast RW.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 4.188  | 4.673  | 3.538  | 5.063  | 5.825  |

Wide cross-seed spread (3.538–5.825, std 0.775). **Only seed 2** (CRPS 3.538) dips
below the RW floor (3.732); the other four seeds are all worse than RW. No seed
delivers a robust improvement — consistent with COSCI-GAN's absent volatility
dynamics rather than a single unlucky training run.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston paths `heston_S_8192x128.npy` |
| Pool | COSCI-GAN generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) ≈ 13.9 (12.47–14.93 across seeds) |
| Horizons | H=32 (steps 64–95), H=64 (steps 64–127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved COSCI-GAN futures (anchored). Bold red = ensemble mean.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

PS-MC stays **above** the RW baseline across the forecast horizon — the COSCI-GAN
pool does not improve on the random walk.

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
cd methods/DiffusionTS/path_shadowing
python run_eval.py --method COSCI-GAN   # 5 seeds → results/Heston/COSCI-GAN/path_shadowing/
```
