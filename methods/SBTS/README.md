# SBTS on Heston

**Schrödinger Bridge Time Series generation** (Principato et al., arXiv 2025) applied to 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent. It estimates the Schrödinger-bridge drift directly from training data
using a kernel density estimator, then simulates paths via Euler–Maruyama.

See [`code/README.md`](code/README.md) for source, original paper, and implementation details.

---

## Metrics — mean ± std across 5 seeds

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------|
| A1 | Path MMD² | Distribution | ↓ | 0.0110 ± 0.0016 | 0.0093 | 0.0136 | 0.0117 | 0.0093 | 0.0110 | 0 |
| A2 | Terminal MMD² | Distribution | ↓ | 0.0090 ± 0.0035 | 0.0061 | 0.0153 | 0.0083 | 0.0056 | 0.0098 | 0 |
| A3 | Increment MMD² | Distribution | ↓ | 0.0071 ± 0.0005 | 0.0074 | 0.0076 | 0.0063 | 0.0074 | 0.0065 | 0 |
| A4 | Volatility MMD | Distribution | ↓ | 0.3125 ± 0.0176 | 0.3230 | 0.3366 | 0.2900 | 0.3183 | 0.2947 | 0 |
| A5 | Terminal SWD | Distribution | ↓ | 3.465 ± 0.588 | 2.775 | 4.258 | 3.634 | 2.799 | 3.858 | 0 |
| A6 | Path SWD | Distribution | ↓ | 2.497 ± 0.288 | 2.203 | 2.928 | 2.620 | 2.150 | 2.582 | 0 |
| A7 | Cov Error (%) | Statistics | ↓ | 145.35 ± 4.89 | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | 0 |
| A8 | Mean RMSE | Statistics | ↓ | 1.3013 ± 0.2776 | 1.2972 | 0.9199 | 1.4819 | 1.0986 | 1.7088 | 0 |
| A9 | Std Error | Statistics | ↓ | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | 0 |
| A10 | Kurtosis Error | Statistics | ↓ | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | 0 |
| A11 | ACF Abs Error | Temporal | ↓ | 0.0568 ± 0.0006 | 0.0568 | 0.0565 | 0.0570 | 0.0558 | 0.0577 | 0 |
| A12 | ACF Sq Error | Temporal | ↓ | 0.0619 ± 0.0006 | 0.0619 | 0.0618 | 0.0617 | 0.0610 | 0.0628 | 0 |
| A13 | Disc Score GRU | Adversarial | ↓ | 0.0291 ± 0.0276 | 0.0032 | 0.0148 | 0.0157 | 0.0816 | 0.0301 | **0** |
| A13 | Disc Score MLP | Adversarial | ↓ | 0.0711 ± 0.0077 | 0.0676 | 0.0859 | 0.0685 | 0.0700 | 0.0636 | **0** |
| A14 | Pred Score GRU (TSTR) | Predictive | ↓ | 0.0091 ± 0.0000 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | 0.0091 | baseline |
| A14 | Pred Score MLP (TSTR) | Predictive | ↓ | 0.0093 ± 0.0006 | 0.0092 | 0.0090 | 0.0104 | 0.0088 | 0.0088 | baseline |
| A15 | Sigma Corr | Heston-specific | ↑ | 0.0011 ± 0.0035 | −0.0024 | 0.0031 | −0.0038 | 0.0039 | 0.0045 | **1** |
| A15 | Sigma RMSE | Heston-specific | ↓ | 0.8207 ± 0.0019 | 0.8198 | 0.8181 | 0.8214 | 0.8205 | 0.8238 | 0 |

> **A13 discriminative score**: `|accuracy − 0.5|` on a held-out test set (80/20 split).
> 0 = indistinguishable from real. 0.5 = perfect separation (bad generator).
>
> **A14 predictive score**: TSTR MAE — predictor trained on *synthetic*, evaluated on *real*.
>
> **A15 sigma**: Heston-specific. Compares inferred instantaneous vol from generated paths
> against the true variance paths.

---

## Stylised Facts Diagnostic (Heston vs SBTS, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/SBTS/plots/heston_diagnostics.png)

---

## SBTS has no training loss

SBTS is kernel-based — there is no loss curve. Instead, the bandwidth `h` is a
hyperparameter chosen from the paper (h=0.4, Appendix C Table 4 for Heston T=100).
The `losses/` directory stores per-seed bandwidth JSON records for reproducibility.

Generation wall-clock times (64 workers, sequential seeds):

| Seed | Workers | Elapsed |
|------|---------|---------|
| 0 | 16 | 23.4 min |
| 1 | 64 | 6.3 min |
| 2 | 64 | 6.2 min |
| 3 | 64 | 6.3 min |
| 4 | 64 | 6.4 min |

---

## A13 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/SBTS/plots/disc_classifier_loss.png)

---

## A14 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/SBTS/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest SBTS paths by L2 distance,
then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data).

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/SBTS/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/SBTS/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

Embedding: **65D murex-style prefix features** — 63 log-returns + 1 terminal return + 1 realized vol,
z-scored per dimension using the generated pool. Adaptive Gaussian bandwidth: η = η̃·‖z(x̃)‖, η̃ = median(dist)/median(‖z‖).

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **2.761 ± 0.004** | 2.762 ± 0.004 | **3.900 ± 0.008** | 3.900 ± 0.008 | 3.73 / 5.30 |
| MAE    | 3.746 ± 0.003 | 3.746 ± 0.003 | 5.288 ± 0.004 | 5.288 ± 0.004 | 3.73 / 5.30 |
| RMSE   | 5.112 ± 0.007 | 5.112 ± 0.007 | 7.227 ± 0.007 | 7.227 ± 0.007 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.76 < 3.73 at H=32; 3.90 < 5.30 at H=64).
**SBTS outperforms TimeGAN** on PS-MC (2.76 vs 3.09 at H=32; 3.90 vs 4.37 at H=64):
the kernel method faithfully reproduces the training distribution, providing a richer and
more diverse retrieval pool than a GAN.

Full analysis: [`results/Heston/SBTS/path_shadowing/README.md`](../../results/Heston/SBTS/path_shadowing/README.md)

---

## File layout

```
methods/SBTS/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, sigma, elapsed_sec
├── losses/
│   ├── seed_{i}_bandwidth.json        h, K, N_pi, dt — no loss (kernel method)
│   └── generation_time.csv            wall-clock time per seed
├── weights/                           (empty — SBTS has no model weights)
└── code/
    ├── sbts_generate.py               core module: generate_paths(), Numba kernels
    ├── small_test.py                  sanity test (N_train=200, M=20, T=32)
    ├── run_all.py                     full run: 5 seeds × 8 192 paths × 128 steps
    ├── reference/                     verbatim SBTS repo (g-principato/SBTS)
    └── README.md                      paper, architecture, diff vs reference
```

## Reproduce

```bash
# Generate paths — 5 seeds (CPU only, no GPU needed)
cd methods/SBTS/code
source /home/tbasseras/sbts-venv/bin/activate
SBTS_NWORK=64 python run_all.py

# Compute metrics
cd /home/tbasseras/benchmark
CUDA_VISIBLE_DEVICES=0 \
    /home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method SBTS --dataset Heston

# Path Shadowing MC
/home/tbasseras/gpu-venv/bin/python methods/SBTS/path_shadowing/run_eval.py
```
