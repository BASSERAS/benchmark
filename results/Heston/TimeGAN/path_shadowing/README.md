# Path Shadowing MC — TimeGAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

**Setup:**
- Query set: 8 192 real Heston paths `heston_S_8192x128.npy`
- Pool: TimeGAN generated paths per seed (8 192 paths)
- Prefix used for retrieval: first **64 steps** (steps 0–63)
- K = **77** nearest neighbours (L2 on raw prefix)
- Two variants: **Uniform** (flat 1/K) and **Gaussian** (w ∝ exp(−d²/2η²))
- Evaluation horizons: **H=32** (steps 64–95) and **H=64** (steps 64–127)

---

## Results (mean ± std across 5 seeds)

| Metric | Horizon | Uniform | Gaussian | Naive RW baseline |
|--------|---------|---------|----------|-------------------|
| CRPS   | H=32    | 4.159 ± 0.982 | 4.154 ± 0.984 | 3.732 |
| MAE    | H=32    | 5.144 ± 0.732 | 5.138 ± 0.735 | 3.732 |
| RMSE   | H=32    | 6.778 ± 0.855 | 6.770 ± 0.860 | 5.068 |
| CRPS   | H=64    | 5.385 ± 0.990 | 5.384 ± 0.989 | 5.301 |
| MAE    | H=64    | 6.595 ± 0.605 | 6.593 ± 0.605 | 5.301 |
| RMSE   | H=64    | 8.746 ± 0.709 | 8.746 ± 0.707 | 7.181 |

**Naive RW baseline:** forecast = last prefix value repeated. CRPS of a deterministic forecast = MAE.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 3.987  | 3.191  | 4.504  | 3.249  | 5.863  |

---

## Interpretation

The PS-MC CRPS (4.16 at H=32) is slightly **above** the naive random-walk baseline (3.73),
meaning the retrieved synthetic ensemble does not outperform simply repeating the last known value.

Two likely causes:

1. **Scale mismatch:** Retrieval is done on raw price levels. TimeGAN seeds with different
   price-level drift produce paths whose L2 prefix distance is dominated by level effects,
   not by dynamics. A normalised retrieval (z-score or log-return prefix) would likely improve results.

2. **Dynamics gap:** TimeGAN (A11 ACF error = 0.134, A12 = 0.092) does not fully
   reproduce volatility clustering. Retrieved synthetic futures therefore don't accurately
   predict the real path's future volatility regime.

The Gaussian variant offers negligible improvement over uniform (Δ CRPS ≈ 0.004),
suggesting the distance-weighted emphasis on closest neighbours adds little when the
pool itself has the above limitations.

This is a **valid scientific finding**: path shadowing quality directly depends on the
generator's fidelity. A generator that scores well on A1–A16 would be expected to produce
a PS-MC CRPS below the naive baseline.

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (steps 0–63). Blue dashed = real future. Red fan = K=77 retrieved
TimeGAN futures. Bold red = ensemble mean.

### CRPS decay over forecast horizon

![CRPS per step](plots/crps_per_step.png)

Both variants (uniform / Gaussian). Gray dashed lines = naive random-walk baseline at H=32 and H=64.

---

## Files

| File | Contents |
|------|---------|
| `seed_{0..4}_results.json` | Per-seed metrics (eta, CRPS/MAE/RMSE at h32 and h64, both variants) |
| `summary.json` | Mean ± std across seeds + baseline + per_seed array |
| `plots/ps_mc_example.png` | Fan-out illustration (seed 0) |
| `plots/crps_per_step.png` | CRPS per forecast step (5 seeds) |
