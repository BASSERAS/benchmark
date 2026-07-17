# SBTS on Heston

**Score-Based Time Series generation** (Principato et al., arXiv 2025) applied to 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent. It estimates the Schrödinger-bridge drift directly from training data
using a kernel density estimator, then simulates paths via Euler–Maruyama.

See [`code/README.md`](code/README.md) for source, original paper, and implementation details.

---

## Metrics — mean ± std across 5 seeds

> Results filled in after `metrics/compute_all.py --method SBTS` completes.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------|
| A1 | Path MMD² | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A2 | Terminal MMD² | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A3 | Increment MMD² | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A4 | Volatility MMD | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A5 | Terminal SWD | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A6 | Path SWD | Distribution | ↓ | TBD | — | — | — | — | — | 0 |
| A7 | Cov Error (%) | Statistics | ↓ | TBD | — | — | — | — | — | 0 |
| A8 | Mean RMSE | Statistics | ↓ | TBD | — | — | — | — | — | 0 |
| A9 | Std Error | Statistics | ↓ | TBD | — | — | — | — | — | 0 |
| A10 | Kurtosis Error | Statistics | ↓ | TBD | — | — | — | — | — | 0 |
| A11 | ACF Abs Error | Temporal | ↓ | TBD | — | — | — | — | — | 0 |
| A12 | ACF Sq Error | Temporal | ↓ | TBD | — | — | — | — | — | 0 |
| A13 | Disc Score GRU | Adversarial | ↓ | TBD | — | — | — | — | — | **0** |
| A13 | Disc Score MLP | Adversarial | ↓ | TBD | — | — | — | — | — | **0** |
| A14 | Pred Score GRU (TSTR) | Predictive | ↓ | TBD | — | — | — | — | — | baseline |
| A14 | Pred Score MLP (TSTR) | Predictive | ↓ | TBD | — | — | — | — | — | baseline |
| A15 | Sigma Corr | Heston-specific | ↑ | TBD | — | — | — | — | — | **1** |
| A15 | Sigma RMSE | Heston-specific | ↓ | TBD | — | — | — | — | — | 0 |
| A16 | Tail Survival Error | Fat-tail | ↓ | TBD | — | — | — | — | — | **0** |

> **A13 discriminative score**: `|accuracy − 0.5|` on a held-out test set (80/20 split).
> 0 = indistinguishable from real. 0.5 = perfect separation (bad generator).
>
> **A14 predictive score**: TSTR MAE — predictor trained on *synthetic*, evaluated on *real*.
>
> **A15 sigma**: Heston-specific. Compares inferred instantaneous vol from generated paths
> against the true variance paths.
>
> **A16 tail survival error**: RMS of survival probability difference at quantiles {0.90, 0.95, 0.99}.

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

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest SBTS paths by L2 distance,
then use their price-anchored futures (steps 64–127) as a forecast ensemble.

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
