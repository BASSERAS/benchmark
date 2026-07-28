# Path Shadowing MC — TimeDiT on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

> The PS-MC evaluation is **model-agnostic**: it consumes only the generated
> `.npy` paths, so the embedding, retrieval and scoring are identical to the
> Diffusion-TS, FourierFlow, TimeGAN, TimeVQVAE, COSCI-GAN, GT-GAN and LS4
> pipelines
> ([`../../DiffusionTS/path_shadowing/README.md`](../../DiffusionTS/path_shadowing/README.md)).
> Only the generated pool differs (TimeDiT instead of the others).

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

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline | Perfect floor |
|--------|---------|---------|----------------------|-------------------|:-------------:|
| **CRPS** | H=32 | **2.724 ± 0.01941** | 2.724 ± 0.01938 | 3.738 | 2.721 |
| MAE    | H=32 | 3.731 ± 0.01014 | 3.731 ± 0.01013 | 3.738 | 3.728 |
| RMSE   | H=32 | 5.052 ± 0.004140 | 5.052 ± 0.004145 | 5.040 | 5.040 |
| **CRPS** | H=64 | **3.786 ± 0.01767** | 3.786 ± 0.01762 | 5.246 | 3.788 |
| MAE    | H=64 | 5.207 ± 0.009637 | 5.207 ± 0.009612 | 5.246 | 5.204 |
| RMSE   | H=64 | 7.065 ± 0.01254 | 7.065 ± 0.01256 | 7.066 | 7.052 |

**PS-MC over the TimeDiT pool sits in the top tier, essentially on the perfect floor**
— CRPS **2.724 < 3.738 at H=32 (−27 %)** and **3.786 < 5.246 at H=64 (−28 %)**,
within ~0.003 of the finite-sample floor (2.721 / 3.788). It is **not the row winner**:
**LS4 wins both horizons** (2.704 / 3.763), with Diffusion-TS (2.717) and CSDI (2.718)
also edging TimeDiT at H=32, so TimeDiT ranks **4th at H=32 and 3rd at H=64** — inside
a tight cluster of strong diffusion/latent methods that all bunch against the floor.
Unlike the GAN pools, whose PS-MC edge is CRPS-specific, TimeDiT's advantage **extends
to point error**: MAE **3.731 < 3.738** and **5.207 < 5.246** beat the random walk at
both horizons, and RMSE **ties** it (5.052 vs 5.040; 7.065 vs 7.066). This is the same
excellent marginal (A15 skewness 0.01515, A28 kurtosis ratio 0.9925 — see
[`../README.md`](../README.md)) paying off: the retrieved neighbours are both
well-calibrated *and* well-located, even if LS4's are fractionally better.

**Uniform ≈ Gaussian** for Heston: Heston is time-homogeneous with constant
parameters, so all K nearest neighbours are roughly equally good predictors. The
adaptive Gaussian provides no meaningful gain over uniform.

**Naive RW**: deterministic forecast (last prefix value repeated) → CRPS = MAE. The
TimeDiT ensemble beats RW on CRPS, MAE **and** matches RMSE — a genuinely better
forecaster, not merely a better-calibrated spread.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 2.718  | 2.713  | 2.736  | 2.699  | 2.755  |

Extremely tight cross-seed spread (2.699–2.755, std 0.019). **All five seeds sit
far below the RW floor (3.738) and within ~0.03 of the perfect floor (2.721)** — a
robust, consistent, near-optimal improvement.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Query set | 8 192 real Heston test paths `heston_S_test_8192x128.npy` |
| Pool | TimeDiT generated paths per seed (8 192 paths) |
| Prefix | Steps 0–63 (64 steps) |
| Embedding | 65D murex-style (63 log-returns + terminal return + realized vol), z-scored |
| K | 77 nearest neighbours (L2 in z-scored embedding space) |
| η̃ | Adaptive: median(dist) / median(‖z‖) ≈ 9.09 (8.58–9.62 across seeds) |
| Horizons | H=32 (steps 64–95), H=64 (steps 64–127) |

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved TimeDiT futures (anchored). Bold red = ensemble mean.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

PS-MC stays **well below** the RW baseline across the entire forecast horizon — the
TimeDiT ensemble's calibrated, well-located spread beats the random walk on CRPS.

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
python run_eval.py --method TimeDiT   # 5 seeds → results/Heston/TimeDiT/path_shadowing/
```
