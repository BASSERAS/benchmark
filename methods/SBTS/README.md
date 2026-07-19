# SBTS on Heston

**Schrödinger Bridge Time Series generation** (Principato et al., arXiv 2025) applied to 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent. It estimates the Schrödinger-bridge drift directly from training data
using a kernel density estimator, then simulates paths via Euler–Maruyama.

See [`code/README.md`](code/README.md) for source, original paper, and implementation details.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A1 | Kurtosis Error | Fat Tail | ↓ | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | 0 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | 0.0063 ± 3.00e-05 | 0.0064 | 0.0063 | 0.0063 | 0.0063 | 0.0063 | 0 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | 0.0098 ± 4.80e-05 | 0.0098 | 0.0098 | 0.0099 | 0.0098 | 0.0097 | 0 |
| A4 | Tail QQ Error | Fat Tail | ↓ | 0.0062 ± 2.60e-05 | 0.0063 | 0.0062 | 0.0062 | 0.0062 | 0.0062 | 0 |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | 9.499 ± 0.3457 | 9.286 | 9.201 | 9.853 | 9.981 | 9.175 | 0 |
| | **— Distribution —** | | | | | | | | | |
| A6 | Path MMD² | Distribution | ↓ | 0.0110 ± 0.0026 | 0.0083 | 0.0131 | 0.0124 | 0.0076 | 0.0138 | 0.0015 |
| A7 | Terminal MMD² | Distribution | ↓ | 0.0100 ± 0.0036 | 0.0059 | 0.0147 | 0.0123 | 0.0055 | 0.0113 | 0.0016 |
| A8 | Increment MMD² | Distribution | ↓ | 0.0071 ± 2.47e-04 | 0.0076 | 0.0069 | 0.0071 | 0.0072 | 0.0069 | 7.45e-04 |
| A9 | Volatility MMD | Distribution | ↓ | 0.3038 ± 0.0071 | 0.3172 | 0.2982 | 0.3049 | 0.2987 | 0.3002 | 0.0071 |
| A10 | Terminal SWD | Distribution | ↓ | 3.539 ± 0.7368 | 2.706 | 4.389 | 4.133 | 2.617 | 3.849 | 0.6873 |
| A11 | Path SWD | Distribution | ↓ | 2.415 ± 0.4104 | 1.897 | 2.723 | 2.828 | 1.933 | 2.692 | 0.4381 |
| A12 | RV Law Loss | Distribution | ↓ | 2.148 ± 0.0074 | 2.156 | 2.151 | 2.155 | 2.141 | 2.138 | 0 |
| A13 | Mean Path RMSE | Distribution | ↓ | 0.7499 ± 0.1823 | 0.7951 | 0.5148 | 0.9253 | 0.5595 | 0.9545 | 0 |
| A14 | KS Log-returns | Distribution | ↓ | 0.0534 ± 3.62e-04 | 0.0537 | 0.0530 | 0.0539 | 0.0530 | 0.0536 | 0 |
| A15 | Skewness Error | Distribution | ↓ | 0.0227 ± 0.0037 | 0.0196 | 0.0184 | 0.0217 | 0.0249 | 0.0287 | 0 |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0028 ± 1.20e-05 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0 |
| A17 | Terminal Price KS | Distribution | ↓ | 0.0921 ± 0.0051 | 0.0892 | 0.0938 | 0.0903 | 0.0863 | 0.1011 | 0 |
| | **— Adversarial —** | | | | | | | | | |
| A18 GRU | Discriminative Score GRU | Adversarial | ↓ | 0.2755 ± 0.2166 | 0.4377 | 0.4490 | 0.0017 | 0.4695 | 0.0194 | 0.0071 |
| A18 MLP | Discriminative Score MLP | Adversarial | ↓ | 0.0079 ± 0.0049 | 0.0038 | 0.0139 | 0.0032 | 0.0139 | 0.0047 | 0.0033 |
| | **— Predictive —** | | | | | | | | | |
| A19 GRU | Predictive Score GRU | Predictive | ↓ | 0.0586 ± 5.90e-05 | 0.0585 | 0.0586 | 0.0587 | 0.0586 | 0.0586 | 0.0537 |
| A19 MLP | Predictive Score MLP | Predictive | ↓ | 0.0583 ± 2.55e-04 | 0.0582 | 0.0578 | 0.0583 | 0.0586 | 0.0583 | 0.0537 |
| | **— Temporal —** | | | | | | | | | |
| A20 | Covariance Error | Temporal | ↓ | 145.35 ± 4.886 | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | 0 |
| A21 | ACF \|r\| Error (lags) | Temporal | ↓ | 0.0596 ± 4.70e-04 | 0.0601 | 0.0595 | 0.0596 | 0.0587 | 0.0599 | 0 |
| A22 | ACF r² Error (lags) | Temporal | ↓ | 0.0619 ± 5.08e-04 | 0.0625 | 0.0618 | 0.0614 | 0.0612 | 0.0624 | 0 |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0.1437 ± 0.0012 | 0.1436 | 0.1419 | 0.1439 | 0.1436 | 0.1456 | 0 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | 0.1665 ± 0.0017 | 0.1674 | 0.1637 | 0.1659 | 0.1668 | 0.1688 | 0 |
| | **— Vol —** | | | | | | | | | |
| A25 | Mean RMSE | Vol | ↓ | 1.301 ± 0.2776 | 1.297 | 0.9199 | 1.482 | 1.099 | 1.709 | 0 |
| A26 | Return Std Error | Vol | ↓ | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | 0 |
| A27 | Log-Return Std Error | Vol | ↓ | 0.0030 ± 1.20e-05 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0 |
| A28 | Kurtosis Ratio | Vol | — | 1.989 ± 0.0182 | 1.991 | 1.998 | 1.988 | 2.012 | 1.957 | 1.000 |
| A29 | Sigma Mean Error | Vol | ↓ | 0.0440 ± 1.84e-04 | 0.0442 | 0.0440 | 0.0441 | 0.0437 | 0.0437 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 3.276 ± 0.0637 | 3.204 | 3.296 | 3.296 | 3.209 | 3.375 | 0 |
| A31 | Rolling Vol KS (w=5) | Vol | ↓ | 0.3435 ± 6.43e-04 | 0.3444 | 0.3433 | 0.3440 | 0.3430 | 0.3426 | 0 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 0.0021 ± 6.00e-06 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0 |
| | **— Heston Spec —** | | | | | | | | | |
| A33 | Teacher-Sigma Corr | Heston Spec | ↑ | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | 0.6143 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | 0.0955 ± 9.10e-05 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 | 0.0654 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **A1**: |kurt_real − kurt_gen| on log-returns. **A2–A3**: 95th/99th quantile error on |log-returns| — near-floor (SBTS reproduces marginal tail quantiles well; the deficiency is in tail *shape*, see A5). **A4**: QQ error restricted to top-5% tail quantiles — near-floor. **A5 Hill ≈ 9.5** (floor 0): kernel smoothing systematically attenuates tail heaviness — SBTS's main fat-tail weakness.
> **A6–A11**: path-kernel distances (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths). Non-zero perfect floor (row-shuffle keeps joint path structure).
> **A12**: W₁(RV_real, RV_gen) — SBTS produces compressed volatility (smoother → lower RV → distribution shift). **A13**: mean-path RMSE. **A14**: KS on pooled log-returns — small (0.053), stable. **A15**: |skew_real − skew_gen| — small (0.023), SBTS reproduces skew well. **A16**: QQ RMSE (bulk, 300-pt) — near-floor. **A17**: KS on terminal prices S_T — moderate mismatch (0.092).
> **A18**: Discriminative classifier on log-returns; score = |accuracy − 0.5|. GRU high-variance — 3 of 5 seeds near 0.44 (temporal structure the Markov-1 kernel can't reproduce), 2 seeds near-perfect. MLP (no temporal context) near 0 — marginal moments well matched. **A19**: TSTR MAE; all methods cluster near 0.056–0.059 (irreducible floor).
> **A20 Cov Error ≈ 145%** (floor 0): SBTS is **Markov-1** — each step only sees the previous state, so multi-step covariance is far weaker than real Heston. Single largest SBTS weakness vs TimeGAN (17.75%). **A21–A22**: ACF error on |r| and r² across lags — close population shape, small kernel-smoothing offset. **A23–A24**: ACF lag-1 error on |r| and r².
> **A25**: mean-path RMSE. **A26**: return std error, uses price increments $\Delta S_t$. **A27**: log-return std error. **A28** Kurtosis Ratio ≈ 1.99: generated kurtosis roughly half real — smoothing attenuates extremes. **A29**: sigma mean error. **A30** Cross-Sect Vol RMSE ≈ 3.28 (floor 0): kernel bootstrap gives high cross-sectional vol spread. **A31** Rolling Vol KS ≈ 0.344 (floor 0): bandwidth smooths stochastic vol → near-constant rolling vol. **A32**: vol-of-vol error, near-floor.
> **A33 Teacher-Sigma Corr ≈ 0.005** (floor 0.614): SBTS S-paths don't retain the latent Heston variance path — expected for a marginal-matching method with no latent state. **A34**: Teacher-sigma RMSE ≈ 0.096 (floor 0.065).

---

## B — Curve-Shape Metrics — mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. For the real
data (L_r) and generated data (L_g) we build three lists — the curve L, its first finite
difference L' (der), and its second finite difference L'' (sec\_der) — then combine the three
sub-scores into **one number per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = m_funct + m_der + m_sec\_der (**sum** of the three seed-means); std = sqrt(s_funct² + s_der² + s_sec\_der²) (**quadrature**).
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE — one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the 5 seeds** of that per-seed function MAPE.

All ↓ lower is better. Perfect floor = 0 for all six plots (row-shuffled real data has identical curves).
Two sublines per plot: **MSE** and **% error** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|------|---------|-----------|--------|--------|--------|--------|--------|:------:|
| **Log-return histogram** | MSE | 12.138 ± 0.1605 | 12.213 | 12.187 | 12.374 | 11.917 | 11.998 | 0 |
| | % err | 38.98% ± 0.132% | 39.11% | 39.05% | 39.11% | 38.83% | 38.82% | 0 |
| **QQ plot** | MSE | 8.90e-06 ± 6.77e-08 | 8.98e-06 | 8.91e-06 | 8.98e-06 | 8.84e-06 | 8.81e-06 | 0 |
| | % err | 21.27% ± 0.364% | 21.31% | 20.87% | 21.64% | 20.84% | 21.69% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.0046 ± 3.70e-05 | 0.0046 | 0.0045 | 0.0045 | 0.0045 | 0.0046 | 0 |
| | % err | 143% ± 1.580% | 144% | 144% | 143% | 140% | 144% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0052 ± 5.67e-05 | 0.0053 | 0.0051 | 0.0051 | 0.0052 | 0.0053 | 0 |
| | % err | 160% ± 1.615% | 161% | 160% | 159% | 157% | 161% | 0 |
| **Rolling vol histogram** | MSE | 1227.30 ± 5.109 | 1234.15 | 1226.44 | 1230.95 | 1223.20 | 1221.77 | 0 |
| | % err | 84.04% ± 0.124% | 84.23% | 84.09% | 84.08% | 83.91% | 83.90% | 0 |
| **Tail survival** | MSE | 0.0057 ± 6.60e-05 | 0.0058 | 0.0058 | 0.0058 | 0.0057 | 0.0057 | 0 |
| | % err | 26.48% ± 0.114% | 26.62% | 26.50% | 26.58% | 26.35% | 26.34% | 0 |

> **Log-ret histogram**: SBTS wins decisively on MSE (12.1 vs TimeGAN 144.2) — kernel smoothing closely preserves marginal returns, and unlike TimeGAN has no seed-collapse events (MSE std 0.16 vs mean 12.1). On the function-level % error the two are close (SBTS 38.98% vs TimeGAN 33.42%).
> **Rolling vol histogram**: SBTS's near-constant rolling vol (see A31) produces the highest MSE of any plot (1227) — the clearest visual signature of the Markov-1 vol-smoothing weakness — and the highest function-level % error too (84.04% vs TimeGAN 56.06%).
> **Tail survival, ACF |r|/r²**: SBTS wins on MSE — the kernel method reproduces the population curve shape; its function-level % errors (tail 26.48%, ACF |r| 143%, ACF r² 160%) are dominated by deep-tail and near-zero-ACF points, where the true curve ≈ 0 makes any deviation a large relative error.
> **Cross-seed stability**: SBTS MSE std is tiny relative to mean for every plot (deterministic kernel, no seed-collapse) — contrast with TimeGAN where MSE std can approach the mean (log-return histogram 144.2 ± 120.6) driven by GAN seed-collapse events.

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
