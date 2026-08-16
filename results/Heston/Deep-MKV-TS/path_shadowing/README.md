# Path Shadowing MC, Deep-MKV-TS on Heston

**Reference:** Morel, Mallat, Bouchaud (2023), arXiv:2308.01486

---

## Method

Identical to the TimeGAN path shadowing evaluation, only the generated path pool differs.
Full method description: [`results/Heston/TimeGAN/path_shadowing/README.md`](../../TimeGAN/path_shadowing/README.md)

### Step 1, 65D murex-style prefix embedding

Given a path prefix of `prefix_len = 64` price steps, embed it as a
**65-dimensional feature vector**:

| Component | Dimension | Formula |
|-----------|-----------|---------|
| Full log-return trajectory | 63 | `r_t = log S_t − log S_{t−1}`, t = 1…63 |
| Terminal cumulative return | 1 | `R = log S_63 − log S_0` |
| Realized volatility | 1 | `σ = sqrt(mean(r_t²))` |
| **Total** | **65** | |

Each dimension is **z-scored** using the mean and std of the Deep-MKV-TS generated pool.

### Step 2, KNN retrieval

For every real query path: retrieve K=77 Deep-MKV-TS generated paths with smallest L2
distance in the z-scored 65D space.

The pool is one seed's `generated_paths/seed_i/generated_paths_8192x128.npy`, shape `(8192, 128)`,
produced by rolling the step-2500 checkpoint forward with `bank_seed = 70000 + i`.
K=77 neighbours drawn from 8192 candidates is the same K used for every other method
in this benchmark, so the retrieval budget is held fixed and only the pool quality varies.

### Step 3, Price anchoring

Each retrieved fake future is multiplicatively scaled to start at the real path's
last prefix price:

$$\tilde{S}^{(k)}_{\text{anchored}}(u) = S^{(k)}_{\text{fake}}(u) \times \frac{S_{\text{real}}(t)}{S^{(k)}_{\text{fake}}(t)}, \quad u > t$$

### Step 4, Two weighting variants

**Uniform:** flat weight `1/K` on all K retrieved futures.

**Gaussian:** distance-weighted with per-query adaptive bandwidth:

$$w_k \propto \exp\!\left(-\frac{\|z^k_{\text{past}} - z_{\text{query}}\|^2}{2\,\eta_i^2}\right), \quad \eta_i = \tilde{\eta} \cdot \|z(\tilde{x}_{\text{past},i})\|$$

Adaptive calibration:

$$\tilde{\eta}_{\text{adapt}} = \frac{\text{median}(\text{distances})}{\text{median}(\|z\|)}$$

### Step 5, Evaluation

Forecast = weighted average of the K anchored futures.
Two horizons: **H=32** (steps 64-95) and **H=64** (steps 64-127).
Metrics: CRPS, MAE, RMSE.

---

## Results (mean ± std across 5 seeds)

| Metric | Horizon | Uniform | Gaussian (adaptive η̃) | Naive RW baseline | Perfect-Recovery Floor |
|--------|---------|---------|----------------------|-------------------|------------------------|
| **CRPS** | H=32 | **2.696 ± 0.004060** | 2.696 ± 0.004043 | 3.738 | 2.721 ± 0.004183 |
| MAE    | H=32 | 3.729 ± 0.005325 | 3.729 ± 0.005253 | 3.738 | — |
| RMSE   | H=32 | 5.047 ± 0.004908 | 5.047 ± 0.004880 | 5.040 | — |
| **CRPS** | H=64 | **3.758 ± 0.004762** | 3.758 ± 0.004747 | 5.246 | 3.788 ± 0.006463 |
| MAE    | H=64 | 5.207 ± 0.007615 | 5.207 ± 0.007605 | 5.246 | — |
| RMSE   | H=64 | 7.055 ± 0.005291 | 7.055 ± 0.005234 | 7.066 | — |

PS-MC **beats the naive random walk on CRPS** at both horizons:
2.696 < 3.738 at H=32 and 3.758 < 5.246 at H=64.

### Reading the point-forecast rows honestly

CRPS is the ranked metric; MAE and RMSE are reported for completeness and they are
**much less flattering**:

- H=32 MAE 3.729 vs RW 3.738 — a 0.2 % improvement, i.e. essentially a tie.
- H=32 RMSE 5.047 vs RW 5.040 — **worse than the random walk.**
- H=64 MAE 5.207 vs RW 5.246 and RMSE 7.055 vs RW 7.066 — small wins.

This is not a Deep-MKV-TS pathology, it is what path shadowing does. The ensemble mean
of K=77 anchored futures is a shrunk forecast: shrinkage helps a proper scoring rule that
rewards a calibrated predictive *distribution* (CRPS) and does almost nothing for a
point-error metric on a near-martingale price. Any method whose pool is a decent Heston
sample lands in the same place. Do not read the MAE/RMSE rows as evidence that the
generator forecasts prices — it does not, and neither does any other entry in this table.

### Uniform ≡ Gaussian

The two weighting variants agree to 4 significant digits at both horizons
(2.6957 vs 2.6957, 3.7575 vs 3.7575). This is expected, not a bug: with K=77 neighbours
retrieved from an 8192-path pool the retained distances are tightly clustered, so the
adaptive bandwidth η̃ produces a near-flat kernel and the Gaussian weights collapse onto
the uniform ones. The same collapse is visible in every other method's PS-MC table.

### Per-seed adaptive bandwidth

| Seed | η̃ (adaptive) | CRPS H=32 (uniform) | CRPS H=64 (uniform) |
|:----:|:------------:|:-------------------:|:-------------------:|
| 0 | 8.7467 | 2.6905 | 3.7526 |
| 1 | 8.7001 | 2.6911 | 3.7516 |
| 2 | 8.6326 | 2.6979 | 3.7640 |
| 3 | 8.7216 | 2.6996 | 3.7607 |
| 4 | 8.7253 | 2.6995 | 3.7588 |

η̃ is calibrated per seed from the retrieval geometry of that seed's own pool
(median distance / median embedding norm), never tuned. Its spread across seeds is
1.3 % — the five pools are geometrically interchangeable, which is the expected
consequence of only 47 330 parameters differing between them while the frozen
Guyon–Lekeufack drift is shared.

---

## Comparison with every other method (same real data, same protocol, K=77)

| Metric | Horizon | Deep-MKV-TS | LS4 | Diffusion-TS | CSDI | TimeDiT | SBTS | Fourier Flow | Naive RW | **Perfect** |
|--------|---------|:-----------:|:---:|:------------:|:----:|:-------:|:----:|:------------:|:--------:|:-----------:|
| CRPS | H=32 | **2.696 ± 0.004060** | 2.704 ± 0.002510 | 2.717 ± 0.002200 | 2.718 ± 0.003646 | 2.724 ± 0.01941 | 2.777 ± 0.005721 | 2.744 ± 0.03009 | 3.738 | 2.721 ± 0.004183 |
| CRPS | H=64 | **3.758 ± 0.004762** | 3.763 ± 0.005851 | 3.804 ± 0.007848 | 3.776 ± 0.005153 | 3.786 ± 0.01767 | 3.858 ± 0.008858 | 3.961 ± 0.1098 | 5.246 | 3.788 ± 0.006463 |

Remaining methods, both horizons (H=32 / H=64):
TimeVQVAE 2.779 / 3.851 · TimeGAN 3.085 / 4.337 · TimeMoDE 3.196 / 4.601 ·
GT-GAN 3.551 / 4.996 · TimeVAE 3.912 / 5.670 · COSCI-GAN 4.657 / 5.789.

**Deep-MKV-TS wins 2 / 2 PS-MC contests.** It is the only method that takes both
horizons, and the margin over the runner-up (LS4) is 0.008 at H=32 and 0.005 at H=64 —
roughly 2× and 1× the cross-seed std respectively, so H=32 is a real separation and
H=64 is a photo finish.

### ⚠️ The win is partly a variance deficit, not purely a modelling win

Deep-MKV-TS scores **below the Perfect-Recovery Floor** at both horizons
(2.696 < 2.721 and 3.758 < 3.788). The floor is an *independent true Heston draw*
scored through the identical pipeline. A generator cannot legitimately beat the true
process at forecasting the true process. Scoring below it therefore diagnoses an
**under-dispersed pool**, not superiority:

- Retrieval + ensemble-averaging rewards a pool whose paths are too tightly clustered
  around the conditional mean, because the resulting ensemble mean is over-shrunk and
  CRPS on a near-martingale target is minimised by shrinkage.
- The same deficit is visible independently in the A table:
  **A5 = 5.110 vs floor 0.5266** (kurtosis under-shoot) and
  **A28 = 1.182 vs floor 1.006** — the generator does not put enough mass in the tails.
- LS4, the runner-up, sits below the floor for the same reason.

The honest reading: Deep-MKV-TS produces the retrieval pool best matched to this
particular task, and part of that match is a genuine strength (the frozen
Guyon–Lekeufack drift pins the conditional level correctly) and part of it is a known
weakness (insufficient tail mass) that happens to be rewarded here. Both statements
are true; neither cancels the other.

---

## Figures

### Example ensemble fan-out (seed 0)

![PS-MC Example](plots/ps_mc_example.png)

The grey fan is the K=77 anchored futures, the blue line their weighted mean, the
black line the realised path. The fan visibly widens with horizon, which is the
qualitative behaviour the CRPS numbers quantify.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

CRPS grows monotonically with forecast step, as it must for a diffusive price. The
Deep-MKV-TS curve sits below the RW baseline over the whole horizon rather than only
at the endpoints, so the H=32 / H=64 aggregates are not an artefact of the two
reporting points.

---

## Reproduce

```bash
/home/tbasseras/gpu-venv/bin/python methods/Deep-MKV-TS/path_shadowing/run_eval.py

# Results saved to:
#   results/Heston/Deep-MKV-TS/path_shadowing/seed_{0..4}_results.json
#   results/Heston/Deep-MKV-TS/path_shadowing/summary.json
#   results/Heston/Deep-MKV-TS/path_shadowing/plots/ps_mc_example.png
#   results/Heston/Deep-MKV-TS/path_shadowing/plots/crps_per_step.png
```

This stage is also step 4 of the one-command driver
[`methods/Deep-MKV-TS/code/run_benchmark_pipeline.sh`](../../../../methods/Deep-MKV-TS/code/run_benchmark_pipeline.sh),
so if you ran that you already have these files. The evaluation is CPU-only, reads
the five exported pools under `methods/Deep-MKV-TS/generated_paths/seed_{0..4}/`, and
takes a few minutes on 8 cores.
