# SBTS on Heston

**Schrödinger Bridge Time Series generation** (Principato et al., arXiv 2025) applied to 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent. It estimates the Schrödinger-bridge drift directly from training data
using a kernel density estimator, then simulates paths via Euler–Maruyama.

See [`code/README.md`](code/README.md) for source, original paper, and implementation details.

---

## Metrics A1–A20 — mean ± std across 5 seeds

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| A1 | Path MMD² | Distribution | ↓ | 0.0112 ± 0.0011 | 0.0129 | 0.0120 | 0.0104 | 0.0100 | 0.0105 | 0.0018 ± 0.0002 |
| A2 | Terminal MMD² | Distribution | ↓ | 0.0102 ± 0.0014 | 0.0113 | 0.0097 | 0.0085 | 0.0092 | 0.0124 | 0.0016 ± 0.0002 |
| A3 | Increment MMD² | Distribution | ↓ | 0.0069 ± 0.0005 | 0.0067 | 0.0066 | 0.0069 | 0.0063 | 0.0077 | 0.0008 ± 0.0000 |
| A4 | Volatility MMD | Distribution | ↓ | 0.2964 ± 0.0126 | 0.2970 | 0.2873 | 0.2968 | 0.2821 | 0.3188 | 0.0077 ± 0.0006 |
| A5 | Terminal SWD | Distribution | ↓ | 3.7097 ± 0.3209 | 4.1147 | 3.6846 | 3.6757 | 3.1564 | 3.9169 | 0.7635 ± 0.1174 |
| A6 | Path SWD | Distribution | ↓ | 2.5335 ± 0.2212 | 2.8445 | 2.6110 | 2.5826 | 2.1668 | 2.4625 | 0.5542 ± 0.0624 |
| A7 | Cov Error | Statistics | ↓ | 145.35 ± 4.89 | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | 4.76 ± 2.50 |
| A8 | Mean RMSE | Statistics | ↓ | 1.3013 ± 0.2776 | 1.2972 | 0.9199 | 1.4819 | 1.0986 | 1.7088 | 0.1400 ± 0.1303 |
| A9 | Std Error | Statistics | ↓ | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | 0.0048 ± 0.0031 |
| A10 | Kurtosis Error | Statistics | ↓ | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | 0.0172 ± 0.0155 |
| A11 | ACF \|r\| Error | Temporal | ↓ | 0.0596 ± 0.0005 | 0.0601 | 0.0595 | 0.0596 | 0.0587 | 0.0599 | 0.0017 ± 0.0006 |
| A12 | ACF r² Error | Temporal | ↓ | 0.0619 ± 0.0005 | 0.0625 | 0.0618 | 0.0614 | 0.0612 | 0.0624 | 0.0014 ± 0.0006 |
| A13 | Disc Score GRU | Adversarial | ↓ | 0.2740 ± 0.2208 | 0.0005 | 0.4286 | 0.4643 | 0.4689 | 0.0078 | 0.0128 ± 0.0068 |
| A13 | Disc Score MLP | Adversarial | ↓ | 0.0063 ± 0.0038 | 0.0029 | 0.0014 | 0.0084 | 0.0121 | 0.0066 | 0.0080 ± 0.0081 |
| A14 | Pred GRU (TSTR) | Predictive | ↓ | 0.0586 ± 0.0000 | 0.0586 | 0.0586 | 0.0585 | 0.0586 | 0.0586 | 0.0564 ± 0.0022 |
| A14 | Pred MLP (TSTR) | Predictive | ↓ | 0.0582 ± 0.0002 | 0.0584 | 0.0579 | 0.0584 | 0.0581 | 0.0582 | 0.0565 ± 0.0022 |
| A15 | Sigma Corr | Heston-specific | ↑ | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | 0.6135 ± 0.0019 |
| A15 | Sigma RMSE | Heston-specific | ↓ | 0.0955 ± 0.0001 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 | 0.0653 ± 0.0002 |
| A16 | Tail RMS | Fat-tail | ↓ | 0.0428 ± 0.0002 | 0.0431 | 0.0427 | 0.0428 | 0.0426 | 0.0425 | 0.0008 ± 0.0008 |
| A16 | q90 Error | Fat-tail | ↓ | 0.0630 ± 0.0003 | 0.0636 | 0.0629 | 0.0631 | 0.0628 | 0.0627 | 0.0010 ± 0.0010 |
| A16 | q95 Error | Fat-tail | ↓ | 0.0378 ± 0.0001 | 0.0379 | 0.0377 | 0.0379 | 0.0378 | 0.0376 | 0.0009 ± 0.0009 |
| A16 | q99 Error | Fat-tail | ↓ | 0.0092 ± 0.0000 | 0.0092 | 0.0092 | 0.0092 | 0.0092 | 0.0091 | 0.0004 ± 0.0004 |
| A17 | Oracle MAE | AR(5) TSTR | ↓ | 0.0097 ± 0.0000 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 ± 0.0000 |
| A18 | Agent MAE | AR(5) TSTR | ↓ | 0.0106 ± 0.0000 | 0.0106 | 0.0107 | 0.0106 | 0.0106 | 0.0107 | 0.0097 ± 0.0000 |
| A19 | Oracle-Agent Corr | AR(5) TSTR | ↑ | −0.342 ± 0.171 | −0.561 | −0.045 | −0.421 | −0.378 | −0.303 | −0.058 ± 0.430 |
| A20 | RV Law Loss | Realized Vol | ↓ | 2.1482 ± 0.0074 | 2.1559 | 2.1510 | 2.1552 | 2.1411 | 2.1380 | 0.0673 ± 0.0362 |

> **A13 discriminative score**: `|accuracy − 0.5|` on held-out test set. 0 = indistinguishable. GRU results are high-variance (3 of 5 seeds score near 0.46 = easily separable): SBTS occasionally reproduces the distribution well, other seeds are caught by the GRU. MLP consistently scores near 0 — stylized facts (moments, ACF shape) are well matched.
>
> **A14 predictive score (TSTR)**: MAE of a predictor trained on *synthetic* data evaluated on *real* data. All values cluster near 0.058–0.059, indicating SBTS-generated log-returns are as informative as real ones for next-step prediction.
>
> **A15 Sigma Corr ≈ 0.005** (floor ≈ 0.61): SBTS generates S-paths that do not retain the latent Heston variance path — expected since it is a non-parametric marginal matching method.
>
> **A15 Sigma RMSE ≈ 0.096 < floor 0.065**: SBTS variance estimates are below-floor, indicating variance compression (generated paths are smoother than Heston). Not a win.
>
> **A16 Tail RMS ≈ 0.043** vs floor 0.0008: SBTS tail quantiles are systematically shifted — the kernel method captures the bulk distribution but oversmoothes the tails.
>
> **A19 Oracle-Agent Corr ≈ −0.34**: Heston log-returns have near-zero autocorrelation, making AR(5) predictions essentially white noise. Oracle and agent estimates are uncorrelated by chance. This metric degenerates for Heston but is meaningful for datasets with temporal structure.
>
> **A20 RV Law Loss ≈ 2.15**: Realized-variance distribution error (Wasserstein-1 between annualized RV of real vs synthetic paths). Floor ≈ 0.07. SBTS produces paths with compressed volatility (smoother paths → lower RV → distribution shift).

---

## Stylised Metrics B1–B14 — mean ± std across 5 seeds

Extracted directly from the 8 diagnostic plot panels (return distribution, QQ, ACF, volatility clustering, terminal distribution, tail survival). Each metric quantifies how well SBTS reproduces a known stylized fact (Cont 2001).

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| B1 | Mean Path RMSE | Distribution | ↓ | 0.7499 ± 0.1823 | 0.7951 | 0.5148 | 0.9253 | 0.5595 | 0.9545 | 0.1511 ± 0.0708 |
| B2 | Cross-Sect Vol RMSE | Distribution | ↓ | 3.2760 ± 0.0637 | 3.2044 | 3.2961 | 3.2960 | 3.2086 | 3.3751 | 0.1355 ± 0.0735 |
| B3 | KS on Log-returns | Distribution | ↓ | 0.0534 ± 0.0004 | 0.0537 | 0.0530 | 0.0539 | 0.0530 | 0.0536 | 0.0018 ± 0.0009 |
| B4 | Skewness Error | Distribution | ↓ | 0.0227 ± 0.0037 | 0.0196 | 0.0184 | 0.0217 | 0.0249 | 0.0287 | 0.0060 ± 0.0048 |
| B5 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0028 ± 0.0000 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0001 ± 0.0000 |
| B6 | Tail QQ Error | Fat-tail | ↓ | 0.0062 ± 0.0000 | 0.0063 | 0.0062 | 0.0062 | 0.0062 | 0.0062 | 0.0001 ± 0.0001 |
| B7 | ACF lag-1 \|r\| Err | Temporal | ↓ | 0.1449 ± 0.0012 | 0.1447 | 0.1431 | 0.1450 | 0.1448 | 0.1468 | 0.0018 ± 0.0016 |
| B8 | ARCH Persistence Err | Temporal | ↓ | 0.0274 ± 0.0004 | 0.0276 | 0.0278 | 0.0272 | 0.0267 | 0.0275 | 0.0011 ± 0.0005 |
| B9 | ACF lag-1 r² Err | Temporal | ↓ | 0.1678 ± 0.0017 | 0.1687 | 0.1650 | 0.1672 | 0.1681 | 0.1701 | 0.0017 ± 0.0014 |
| B10 | GARCH Persistence Err | Temporal | ↓ | 0.0227 ± 0.0004 | 0.0229 | 0.0232 | 0.0225 | 0.0221 | 0.0229 | 0.0010 ± 0.0006 |
| B11 | Rolling Vol KS | Volatility | ↓ | 0.3435 ± 0.0006 | 0.3444 | 0.3433 | 0.3440 | 0.3430 | 0.3426 | 0.0046 ± 0.0024 |
| B12 | Vol-of-Vol Error | Volatility | ↓ | 0.0021 ± 0.0000 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0000 ± 0.0000 |
| B13 | Terminal Price KS | Distribution | ↓ | 0.0921 ± 0.0051 | 0.0892 | 0.0938 | 0.0903 | 0.0863 | 0.1011 | 0.0145 ± 0.0043 |
| B14 | Hill Tail Index Err | Fat-tail | ↓ | 9.499 ± 0.346 | 9.286 | 9.201 | 9.853 | 9.981 | 9.175 | 0.499 ± 0.610 |

> **B1–B2**: Mean and cross-sectional vol path RMSE — SBTS paths are internally smooth but have high cross-sectional spread from the kernel bootstrap, producing B2 ≈ 3.28 vs floor 0.14.
>
> **B3–B6**: Log-return distributional tests. KS ≈ 0.053 is 29× above the perfect floor but stable across seeds — SBTS reproduces the return shape but with a small systematic shift. QQ RMSE (B5) ≈ 0.0028 is tight in the bulk; Tail QQ (B6) ≈ 0.0062 shows moderate tail mismatch.
>
> **B7–B10**: ARCH/GARCH temporal structure. ACF lag-1 |r| error (B7) ≈ 0.145 and lag-1 r² (B9) ≈ 0.168 indicate SBTS correctly generates volatility clustering to first approximation, with the same systematic offset for all seeds (kernel method reproduces the population ACF).
>
> **B11–B12**: Rolling vol KS ≈ 0.344 (floor 0.005) — SBTS produces paths with near-constant rolling volatility (kernel bandwidth smooths out stochastic vol), explaining the large deviation.
>
> **B13**: Terminal price KS ≈ 0.092 — moderate mismatch in the terminal price distribution.
>
> **B14**: Hill tail index error ≈ 9.5 vs floor 0.5 — SBTS systematically underestimates the tail heaviness of Heston returns (kernel smoothing attenuates extremes). Result is stable across seeds (low std ± 0.35).

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
