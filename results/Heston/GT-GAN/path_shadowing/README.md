# Path Shadowing MC — GT-GAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

> The PS-MC evaluation is **model-agnostic**: it consumes only the generated
> `.npy` paths, so the embedding, retrieval and scoring are identical to the
> Diffusion-TS, FourierFlow, TimeGAN, TimeVQVAE, COSCI-GAN and LS4 pipelines
> ([`../../DiffusionTS/path_shadowing/README.md`](../../DiffusionTS/path_shadowing/README.md)).
> Only the generated pool differs (GT-GAN instead of the others).

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
| **CRPS** | H=32 | 3.551 ± 0.108 | 3.552 ± 0.108 | **3.738** |
| MAE    | H=32 | 4.133 ± 0.087 | 4.134 ± 0.087 | 3.738 |
| RMSE   | H=32 | 5.646 ± 0.129 | 5.646 ± 0.129 | 5.040 |
| **CRPS** | H=64 | 4.996 ± 0.195 | 4.996 ± 0.195 | **5.246** |
| MAE    | H=64 | 6.002 ± 0.219 | 6.003 ± 0.220 | 5.246 |
| RMSE   | H=64 | 8.222 ± 0.324 | 8.223 ± 0.325 | 7.066 |

**PS-MC over the GT-GAN pool beats the naive RW on CRPS at both horizons** — CRPS
3.551 < 3.738 at H=32 and 4.996 < 5.246 at H=64 — even though GT-GAN is the
benchmark's **weakest marginal-distribution matcher** (see [`../README.md`](../README.md):
A28 kurtosis ratio 0.002659, A14 KS 0.3881, A18 GRU 0.4871). The win is
**CRPS-specific and does NOT extend to point error**: MAE 4.133 > 3.738 and RMSE
5.646 > 5.040 at H=32 — the random walk is a *better point* forecaster, but a
*worse distributional* one.

Why the collapsed marginal still helps CRPS. CRPS rewards a **calibrated ensemble
spread**, not a single point. Price anchoring subtracts each prefix's terminal
level, and averaging over K = 77 nearest neighbours washes out GT-GAN's spiky,
over-peaked per-step returns, leaving an ensemble whose *dispersion* brackets the
real future better than the RW's degenerate point mass — hence CRPS improves while
MAE/RMSE (which the anchored-mean cannot beat, given the mis-shaped returns) do not.

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good predictors. The
adaptive Gaussian provides no meaningful gain over uniform.

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE. The
GT-GAN ensemble's edge over RW is purely the value of a calibrated spread.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 3.723  | 3.393  | 3.506  | 3.538  | 3.597  |

Tight cross-seed spread (3.393–3.723, std 0.108). **All five seeds sit below the RW
floor (3.738)** — a robust, consistent improvement rather than a single lucky run,
in contrast to COSCI-GAN (only 1/5 seeds beat RW). The gain is small but stable.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston test paths `heston_S_test_8192x128.npy` |
| Pool | GT-GAN generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) ≈ 55.5 (40.64–69.21 across seeds) |
| Horizons | H=32 (steps 64–95), H=64 (steps 64–127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved GT-GAN futures (anchored). Bold red = ensemble mean.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

PS-MC stays **at or below** the RW baseline across the forecast horizon — the GT-GAN
ensemble's calibrated spread edges the random walk on CRPS.

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
python run_eval.py --method GT-GAN   # 5 seeds → results/Heston/GT-GAN/path_shadowing/
```
