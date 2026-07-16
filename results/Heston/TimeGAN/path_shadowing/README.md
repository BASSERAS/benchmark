# Path Shadowing MC — TimeGAN on Heston

**Reference:** Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486

**Setup:**
- Query set: 8 192 real Heston paths `heston_S_8192x128.npy`
- Pool: TimeGAN generated paths per seed (8 192 paths)
- Prefix for retrieval: first **64 steps** (steps 0–63)
- Embedding: **log-returns** `r_t = log(S_{t+1}/S_t)` on the prefix (removes price-level effect)
- Future anchoring: each retrieved future multiplied by `S_real_63 / S_fake_63` so all forecasts start at the real path's last known price
- K = **77** nearest neighbours (L2 in log-return space)
- Two weighting variants: **Uniform** (flat 1/K) and **Gaussian** (w ∝ exp(−d²/2η²))
- Evaluation horizons: **H=32** (steps 64–95) and **H=64** (steps 64–127)

---

## Results (mean ± std across 5 seeds)

| Metric | Horizon | Uniform | Gaussian | Naive RW baseline |
|--------|---------|---------|----------|-------------------|
| **CRPS** | H=32 | **3.097 ± 0.296** | 3.098 ± 0.297 | 3.732 |
| MAE    | H=32 | 4.036 ± 0.192 | 4.037 ± 0.192 | 3.732 |
| RMSE   | H=32 | 5.450 ± 0.248 | 5.451 ± 0.249 | 5.068 |
| **CRPS** | H=64 | **4.412 ± 0.380** | 4.414 ± 0.381 | 5.301 |
| MAE    | H=64 | 5.720 ± 0.171 | 5.721 ± 0.172 | 5.301 |
| RMSE   | H=64 | 7.706 ± 0.182 | 7.708 ± 0.183 | 7.181 |

**Naive RW baseline:** forecast = last prefix value repeated for all steps; CRPS = MAE (deterministic).

PS-MC **beats the naive baseline on CRPS** at both horizons (3.10 < 3.73 at H=32; 4.41 < 5.30 at H=64).
MAE is above the RW because the ensemble mean incorporates scenario diversity, while RW is a deterministic point forecast.

---

## Per-seed CRPS (H=32, uniform)

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| 3.103  | 2.816  | 3.108  | 2.826  | 3.630  |

Seed 4 is the weakest (highest A16 tail error = 0.027, highest disc score).

---

## Embedding note

The paper's eq. (13) proposes a multi-scale power-law log-return embedding
with optimal α=1.15, β=0.9. We use flat log-returns on all 63 prefix steps —
a simpler approximation. Implementing the full embedding would likely improve further.

---

## Figures

### Ensemble fan-out (seed 0, 4 example paths)

![PS-MC Example](plots/ps_mc_example.png)

Blue solid = real prefix (0–63). Blue dashed = real future. Red fan = K=77 retrieved TimeGAN futures (anchored). Bold red = ensemble mean.

### CRPS per forecast step

![CRPS per step](plots/crps_per_step.png)

PS-MC stays below the RW baseline line at both H=32 and H=64.

---

## Files

| File | Contents |
|------|---------|
| `seed_{0..4}_results.json` | Per-seed metrics (eta, CRPS/MAE/RMSE at h32/h64, both variants) |
| `summary.json` | Mean ± std across seeds + baseline + per_seed array |
| `plots/ps_mc_example.png` | Fan-out illustration (seed 0) |
| `plots/crps_per_step.png` | CRPS per forecast step (5 seeds, mean ± std) |
