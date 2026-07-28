# Path Shadowing MC, TimeMoDE on Heston

**Reference:** Morel, Mallat, Bouchaud (2023), arXiv:2308.01486

> The PS-MC evaluation is **model-agnostic**: it consumes only the generated
> `.npy` paths, so the embedding, retrieval and scoring are identical to the
> Diffusion-TS and TimeGAN pipelines
> ([`../../DiffusionTS/path_shadowing/README.md`](../../DiffusionTS/path_shadowing/README.md)).
> Only the generated pool differs (TimeMoDE instead of Diffusion-TS / TimeGAN).

---

## Method

### Step 1, 65D murex-style prefix embedding

Given a path prefix of `prefix_len = 64` price steps, we embed it as a
**65-dimensional feature vector** adapted from Murex's internal implementation
(`deep-mkv-gen-path-dt/experiments/path_dt_experiments/shadowing.py`):

| Component | Dimension | Formula |
|-----------|-----------|---------|
| Full log-return trajectory | 63 | `r_t = log S_t − log S_{t−1}`, t = 1…63 |
| Terminal cumulative return | 1 | `R = log S_63 − log S_0` |
| Realized volatility | 1 | `σ = sqrt(mean(r_t²))` |
| **Total** | **65** |, |

Each dimension is **z-scored** using the mean and std of the generated pool,
making distances scale-invariant across features:

```python
mean = fake_emb.mean(axis=0)   # per-dimension, from generated pool
std  = fake_emb.std(axis=0)
z_real = (real_emb - mean) / std
z_fake = (fake_emb - mean) / std
```

### Step 2, KNN retrieval (NOT combinatorial)

For every real query path `x̃_past`:

```python
# O(N × D) scan — sklearn NearestNeighbors, L2 metric in z-scored space
distances, indices = nn.kneighbors(z_real)  # returns K smallest distances
```

`K = 77` generated paths with the smallest L2 distance in the z-scored
65D space are selected. **No subset enumeration, no combinatorial search.**

### Step 3, Price anchoring

Each retrieved fake future is multiplicatively scaled so it starts at the real
path's last prefix price, removing the price-level offset:

$$\tilde{S}^{(k)}_{\text{anchored}}(u) = S^{(k)}_{\text{fake}}(u) \times \frac{S_{\text{real}}(t)}{S^{(k)}_{\text{fake}}(t)}, \quad u > t$$

### Step 4, Two weighting variants

**Uniform:** flat weight `1/K` on all K retrieved futures.

**Gaussian:** distance-weighted with per-query adaptive bandwidth:

$$w_k \propto \exp\!\left(-\frac{\|z^k_{\text{past}} - z_{\text{query}}\|^2}{2\,\eta_i^2}\right), \quad \eta_i = \tilde{\eta} \cdot \|z(\tilde{x}_{\text{past},i})\|$$

Adaptive calibration (dataset-neutral):

$$\tilde{\eta}_{\text{adapt}} = \frac{\text{median}(\text{distances})}{\text{median}(\|z\|)}$$

Using the paper's raw `η̃ = 0.075` (calibrated on S&P) collapses weights onto
a single nearest neighbour on Heston data, giving CRPS worse than the
random-walk baseline. The adaptive η̃ preserves the per-query scaling idea
while being dataset-neutral.

### Step 5, Evaluation

Forecast = weighted average of the K anchored futures.
Evaluated at two horizons **H=32** (steps 64-95) and **H=64** (steps 64-127)
using CRPS (proper scoring rule), MAE, and RMSE.

---

## Results (mean ± std across 5 seeds), 65D murex embedding

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline |
|--------|---------|---------|----------------------|-------------------|
| **CRPS** | H=32 | **3.196 ± 0.139** | 3.197 ± 0.139 | 3.738 |
| MAE    | H=32 | 4.002 ± 0.149 | 4.002 ± 0.149 | 3.738 |
| RMSE   | H=32 | 5.454 ± 0.202 | 5.454 ± 0.202 | 5.040 |
| **CRPS** | H=64 | **4.601 ± 0.290** | 4.601 ± 0.290 | 5.246 |
| MAE    | H=64 | 5.694 ± 0.276 | 5.694 ± 0.276 | 5.246 |
| RMSE   | H=64 | 7.792 ± 0.385 | 7.792 ± 0.385 | 7.066 |

**PS-MC beats the naive RW on CRPS** at both horizons (3.196 < 3.738 at H=32;
4.601 < 5.246 at H=64), even a weak TimeMoDE pool still supplies a calibrated
nearest-neighbour ensemble that improves on the point forecast. But TimeMoDE's
CRPS is the **weakest of the diffusion/state-space methods** (3.196 vs
Diffusion-TS 2.717, CSDI 2.718, LS4 2.704 at H=32): the poorer generated pool
gives looser, less-informative neighbours, and the spread across seeds (±0.139)
is an order of magnitude larger than Diffusion-TS's (±0.003), the same
seed-instability that dominates the A/B metric tables.

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good predictors.
The adaptive Gaussian provides no meaningful gain over uniform on this
synthetic data (unlike real S&P, where the paper reports gains from Gaussian weighting).

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE.
PS-MC improves CRPS because it provides a *calibrated ensemble* rather than a
point forecast. Note the ensemble MAE (4.002) sits slightly *above* the RW MAE
(3.738) at H=32, the weak pool's neighbours are biased, yet CRPS still wins
because proper scoring rewards the ensemble's uncertainty quantification.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 3.242  | 3.328  | 2.932  | 3.284  | 3.196  |

The cross-seed spread (2.93-3.33, std 0.14) is far wider than Diffusion-TS's
(2.714-2.721): seed 2 gives the tightest pool (2.93, near Diffusion-TS quality)
while seeds 0/1/3 are looser, mirroring the seed-2-is-best pattern seen in the
A/B metric tables. All five still clear the RW baseline (3.738).

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston paths `heston_S_8192x128.npy` |
| Pool | TimeMoDE generated paths per seed (8 192 paths) |
| Prefix | Steps 0-63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) ≈ 5.3-6.9 per seed |
| Horizons | H=32 (steps 64-95), H=64 (steps 64-127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0-63). Blue dashed = real future. Red fan = K=77 retrieved TimeMoDE futures (anchored). Bold red = ensemble mean.

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
cd methods/TimeMoDE/path_shadowing
python run_eval.py     # 5 seeds in parallel → results/Heston/TimeMoDE/path_shadowing/
```
