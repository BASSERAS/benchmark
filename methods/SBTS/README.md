# SBTS on Heston

**Schrödinger Bridge Time Series generation** (Principato et al., arXiv 2025) applied to 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent. It estimates the Schrödinger-bridge drift directly from training data
using a kernel density estimator, then simulates paths via Euler–Maruyama.

See [`code/README.md`](code/README.md) for source, original paper, and implementation details.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A9 uses price increments $\Delta S_t$; A11/A12/A16–A34 use log-returns.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A10 | Kurtosis Error                  | Fat Tail      | ↓ | 0.1187 ± 0.0060 | 0.1156 | 0.1116 | 0.1165 | 0.1293 | 0.1203 | 0.0000 |
| A17 | \|r\| q95 Error                 | Fat Tail      | ↓ | 0.0063 ± 0.0000 | 0.0064 | 0.0063 | 0.0063 | 0.0063 | 0.0063 | 0.0000 |
| A18 | \|r\| q99 Error                 | Fat Tail      | ↓ | 0.0098 ± 0.0000 | 0.0098 | 0.0098 | 0.0099 | 0.0098 | 0.0097 | 0.0000 |
| A30 | Tail QQ Error                   | Fat Tail      | ↓ | 0.0062 ± 0.0000 | 0.0063 | 0.0062 | 0.0062 | 0.0062 | 0.0062 | 0.0000 |
| A34 | Hill Tail Index Error           | Fat Tail      | ↓ | 9.499 ± 0.346   | 9.286  | 9.201  | 9.853  | 9.981  | 9.175  | 0.0000 |
| | **— Distribution —** | | | | | | | | | |
| A1  | Path MMD²                       | Distribution  | ↓ | 0.0112 ± 0.0011 | 0.0129 | 0.0120 | 0.0104 | 0.0100 | 0.0105 | 0.0015 |
| A2  | Terminal MMD²                   | Distribution  | ↓ | 0.0102 ± 0.0014 | 0.0113 | 0.0097 | 0.0085 | 0.0092 | 0.0124 | 0.0016 |
| A3  | Increment MMD²                  | Distribution  | ↓ | 0.0069 ± 0.0005 | 0.0067 | 0.0066 | 0.0069 | 0.0063 | 0.0077 | 0.0007 |
| A4  | Volatility MMD                  | Distribution  | ↓ | 0.2964 ± 0.0126 | 0.2970 | 0.2873 | 0.2968 | 0.2821 | 0.3188 | 0.0071 |
| A5  | Terminal SWD                    | Distribution  | ↓ | 3.7097 ± 0.3209 | 4.1147 | 3.6846 | 3.6757 | 3.1564 | 3.9169 | 0.687  |
| A6  | Path SWD                        | Distribution  | ↓ | 2.5335 ± 0.2212 | 2.8445 | 2.6110 | 2.5826 | 2.1668 | 2.4625 | 0.438  |
| A24 | RV Law Loss (W₁ on ann. RV)     | Distribution  | ↓ | 2.1482 ± 0.0074 | 2.1559 | 2.1510 | 2.1552 | 2.1411 | 2.1380 | 0.0000 |
| A25 | Mean Path RMSE                  | Distribution  | ↓ | 0.7499 ± 0.1823 | 0.7951 | 0.5148 | 0.9253 | 0.5595 | 0.9545 | 0.0000 |
| A27 | KS on Log-returns               | Distribution  | ↓ | 0.0534 ± 0.0004 | 0.0537 | 0.0530 | 0.0539 | 0.0530 | 0.0536 | 0.0000 |
| A28 | Skewness Error                  | Distribution  | ↓ | 0.0227 ± 0.0037 | 0.0196 | 0.0184 | 0.0217 | 0.0249 | 0.0287 | 0.0000 |
| A29 | QQ RMSE (300-pt)                | Distribution  | ↓ | 0.0028 ± 0.0000 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0028 | 0.0000 |
| A33 | Terminal Price KS               | Distribution  | ↓ | 0.0921 ± 0.0051 | 0.0892 | 0.0938 | 0.0903 | 0.0863 | 0.1011 | 0.0000 |
| | **— Adversarial —** | | | | | | | | | |
| A13 | Disc Score GRU (log-ret.)       | Adversarial   | ↓ | 0.274 ± 0.221   | 0.001  | 0.429  | 0.464  | 0.469  | 0.008  | 0.008  |
| A13 | Disc Score MLP (log-ret.)       | Adversarial   | ↓ | 0.006 ± 0.004   | 0.003  | 0.001  | 0.008  | 0.012  | 0.007  | 0.010  |
| | **— Predictive —** | | | | | | | | | |
| A14 | Pred Score GRU — TSTR           | Predictive    | ↓ | 0.0586 ± 0.0000 | 0.0586 | 0.0586 | 0.0585 | 0.0586 | 0.0586 | 0.054  |
| A14 | Pred Score MLP — TSTR           | Predictive    | ↓ | 0.0582 ± 0.0002 | 0.0584 | 0.0579 | 0.0584 | 0.0581 | 0.0582 | 0.054  |
| | **— Temporal —** | | | | | | | | | |
| A7  | Cov Error (%)                   | Temporal      | ↓ | 145.35 ± 4.89   | 143.74 | 145.83 | 142.90 | 139.93 | 154.36 | 0.00   |
| A11 | ACF Error \|log-returns\|       | Temporal      | ↓ | 0.0596 ± 0.0005 | 0.0601 | 0.0595 | 0.0596 | 0.0587 | 0.0599 | 0.0000 |
| A12 | ACF Error log-returns²          | Temporal      | ↓ | 0.0619 ± 0.0005 | 0.0625 | 0.0618 | 0.0614 | 0.0612 | 0.0624 | 0.0000 |
| A22 | ACF \|r\| Lag-1 Error           | Temporal      | ↓ | 0.1437 ± 0.0012 | 0.1436 | 0.1419 | 0.1439 | 0.1436 | 0.1456 | 0.0000 |
| A23 | ACF r² Lag-1 Error              | Temporal      | ↓ | 0.1665 ± 0.0017 | 0.1674 | 0.1637 | 0.1659 | 0.1668 | 0.1688 | 0.0000 |
| | **— Vol —** | | | | | | | | | |
| A8  | Mean RMSE                       | Vol           | ↓ | 1.301 ± 0.278   | 1.297  | 0.920  | 1.482  | 1.099  | 1.709  | 0.000  |
| A9  | Return Std Error                | Vol           | ↓ | 0.2492 ± 0.0018 | 0.2503 | 0.2517 | 0.2485 | 0.2491 | 0.2462 | 0.0000 |
| A16 | Log-Return Std Error            | Vol           | ↓ | 0.0030 ± 0.0000 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0000 |
| A19 | Kurtosis Ratio (target/model)   | Vol           | — | 1.989 ± 0.018   | 1.991  | 1.998  | 1.988  | 2.012  | 1.957  | 1.0000 |
| A20 | Sigma Mean Error                | Vol           | ↓ | 0.0440 ± 0.0002 | 0.0442 | 0.0440 | 0.0441 | 0.0437 | 0.0437 | 0.0000 |
| A26 | Cross-Sect. Vol Path RMSE       | Vol           | ↓ | 3.2760 ± 0.0637 | 3.2044 | 3.2961 | 3.2960 | 3.2086 | 3.3751 | 0.0000 |
| A31 | Rolling Vol KS (window=5)       | Vol           | ↓ | 0.3435 ± 0.0006 | 0.3444 | 0.3433 | 0.3440 | 0.3430 | 0.3426 | 0.0000 |
| A32 | Vol-of-Vol Error                | Vol           | ↓ | 0.0021 ± 0.0000 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0000 |
| | **— Heston Spec —** | | | | | | | | | |
| A15 | Sigma Corr (vol recovery)       | Heston Spec   | ↑ | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | 0.614  |
| A15 | Sigma RMSE                      | Heston Spec   | ↓ | 0.0955 ± 0.0001 | 0.0955 | 0.0955 | 0.0957 | 0.0954 | 0.0954 | 0.065  |
| A21 | Learned/Oracle Sigma Corr       | Heston Spec   | ↑ | 0.0046 ± 0.0019 | 0.0046 | 0.0045 | 0.0016 | 0.0048 | 0.0074 | 0.614  |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A19 Kurtosis Ratio: perfect = 1.0.
> **A7 Cov Error ≈ 145%** (floor 0.00%): SBTS is a **Markov-1** method — each generated step only sees the previous state, so multi-step covariance structure is far weaker than real Heston paths. This is the single largest SBTS weakness vs TimeGAN (17.75%).
> **A11–A12**: ACF on log-returns r_t = log(S_{t+1}/S_t). SBTS reproduces the population ACF shape closely (both floors ≈0.001–0.002) but with a small systematic offset from kernel smoothing.
> **A13**: Discriminative classifier trained on log-returns (not raw prices). Score = |accuracy − 0.5|; 0 = indistinguishable. GRU is high-variance — 3 of 5 seeds score near 0.46 (easily separable via temporal structure the Markov-1 kernel can't reproduce), 2 seeds near-perfect. MLP (no temporal context) consistently scores near 0 — marginal moments are well matched.
> **A14**: TSTR MAE; all methods cluster near 0.056–0.059 (irreducible log-return noise floor). Differences are small.
> **A15/A21 Sigma Corr ≈ 0.005** (floor 0.614): SBTS generates S-paths that do not retain the latent Heston variance path — expected for a non-parametric marginal-matching method with no latent state.
> **A16**: Log-return std error. Uses $r_t = \log(S_{t+1}/S_t)$ (vs price increments in A9).
> **A17–A18**: 95th/99th quantile error on |log-returns| — near-floor (SBTS reproduces marginal tail quantiles well; the deficiency is in tail *shape*, see A34).
> **A19**: Kurtosis ratio real/gen, perfect = 1.0. SBTS ≈ 1.99 means generated kurtosis is roughly half real kurtosis — smoothing attenuates extremes.
> **A20**: Sigma mean error — annualized per-path vol, averaged over paths.
> **A22–A23**: ACF lag-1 error on |r| and r². Single-lag version of A11/A12.
> **A24**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt — SBTS produces paths with compressed volatility (smoother → lower RV → distribution shift).
> **A25–A26**: Path-level RMSE between real and generated mean/vol trajectories. A26 (3.28 vs floor 0.0) is large: SBTS's kernel bootstrap gives high cross-sectional spread in vol paths.
> **A27**: Kolmogorov–Smirnov statistic on pooled log-returns — small (0.053) and stable across seeds, a systematic shift rather than seed noise.
> **A28**: |skew_real − skew_gen| — small (0.023), SBTS reproduces skew well.
> **A29–A30**: QQ RMSE (bulk, 300-pt) and Tail QQ (top-5%) — both near-floor; SBTS matches the return quantile function closely in the bulk and moderately in the tail.
> **A31**: KS statistic on rolling-5 volatility histograms — large (0.344 vs floor 0.0): kernel bandwidth smooths out stochastic vol, producing near-constant rolling volatility.
> **A32**: |vol-of-vol_real − vol-of-vol_gen| — small, near-floor.
> **A33**: KS statistic on terminal prices S_T — moderate mismatch (0.092).
> **A34**: |Hill tail index_real − Hill tail index_gen| — large (9.5 vs floor 0.0): kernel smoothing systematically attenuates tail heaviness, SBTS's main fat-tail weakness.

---

## B — Curve-Shape Metrics — mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. For the real
data (L_r) and generated data (L_g) we build three lists — the curve L, its first finite
difference L' (der), and its second finite difference L'' (sec\_der) — then combine the three
sub-scores into **one number per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = m_funct + m_der + m_sec\_der (sum of the three seed-means); std = sqrt(s_funct² + s_der² + s_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100. Combined the same way.

All ↓ lower is better. Perfect floor = 0 for all six plots (row-shuffled real data has identical curves).
Two sublines per plot: **MSE** and **% error** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE   | 12.14 ± 0.16 | 12.21 | 12.19 | 12.37 | 11.92 | 12.00 | 0.0 |
|                          | % err | 14264% ± 8212% | 25115% | 3364% | 16927% | 19648% | 6267% | 0% |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | 8.98e-6 | 8.91e-6 | 8.98e-6 | 8.84e-6 | 8.81e-6 | 0.0 |
|                          | % err | 113.1% ± 5.8% | 122.2% | 116.0% | 107.4% | 112.7% | 107.2% | 0% |
| **ACF \|r\|**            | MSE   | 4.57e-3 ± 3.7e-5 | 4.60e-3 | 4.54e-3 | 4.55e-3 | 4.54e-3 | 4.64e-3 | 0.0 |
|                          | % err | 1269.9% ± 40.0% | 1230.3% | 1289.4% | 1225.9% | 1297.6% | 1306.2% | 0% |
| **ACF r²**               | MSE   | 5.17e-3 ± 5.7e-5 | 5.25e-3 | 5.07e-3 | 5.09e-3 | 5.16e-3 | 5.28e-3 | 0.0 |
|                          | % err | 1603.5% ± 142.8% | 1631.0% | 1811.2% | 1702.5% | 1470.2% | 1402.6% | 0% |
| **Rolling vol hist.**    | MSE   | 1227.3 ± 5.1 | 1234.2 | 1226.4 | 1231.0 | 1223.2 | 1221.8 | 0.0 |
|                          | % err | 715.9% ± 20.2% | 706.0% | 687.3% | 724.6% | 731.3% | 730.5% | 0% |
| **Tail survival**        | MSE   | 5.74e-3 ± 6.6e-5 | 5.82e-3 | 5.75e-3 | 5.81e-3 | 5.66e-3 | 5.67e-3 | 0.0 |
|                          | % err | 17151% ± 910% | 18514% | 17266% | 17220% | 17111% | 15645% | 0% |

> **Log-ret histogram**: SBTS wins decisively on MSE (12.1 vs TimeGAN 144.2) — kernel smoothing closely preserves marginal returns, and unlike TimeGAN has no seed-collapse events (MSE std 0.16 vs mean 12.1). The large % error (14264%) comes from division by near-zero real-curve values in the empty histogram bins, not from a curve-level mismatch.
> **Rolling vol histogram**: SBTS's near-constant rolling vol (see A31) produces the highest MSE of any plot (1227) — the clearest visual signature of the Markov-1 vol-smoothing weakness.
> **Tail survival, ACF |r|/r²**: SBTS wins on MSE — the kernel method reproduces the population curve shape. The % error is inflated (division by the tiny survival/ACF values in the tail), so read MSE for these plots.
> **Cross-seed stability**: SBTS MSE std is tiny relative to mean for every plot (deterministic kernel, no seed-collapse) — contrast with TimeGAN where MSE std can approach the mean (log-return histogram 144.2 ± 120.6) driven by GAN seed-collapse events. A std near the mean is expected and not an error: each cell is a non-negative combined score, so "mean ± std" is a compact summary over 5 skewed samples, not a claim of negative values.

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
