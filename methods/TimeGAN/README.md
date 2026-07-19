# TimeGAN on Heston

PyTorch reimplementation of **TimeGAN** (Yoon et al., NeurIPS 2019) trained on 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, original paper, and the 5 fixes applied
to the TF1 reference implementation.

---

## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A9 uses price increments $\Delta S_t$; A11/A12/A16–A34 use log-returns.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A10 | Kurtosis Error                  | Fat Tail      | ↓ | 2.955 ± 2.099   | 0.015  | 5.360  | 3.768  | 0.958  | 4.672  | 0.0000 |
| A17 | \|r\| q95 Error                 | Fat Tail      | ↓ | 0.0032 ± 0.0018 | 0.0042 | 0.0056 | 0.0016 | 0.0005 | 0.0040 | 0.0000 |
| A18 | \|r\| q99 Error                 | Fat Tail      | ↓ | 0.0043 ± 0.0028 | 0.0074 | 0.0069 | 0.0052 | 0.0017 | 0.0004 | 0.0000 |
| A30 | Tail QQ Error                   | Fat Tail      | ↓ | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0.0000 |
| A34 | Hill Tail Index Error           | Fat Tail      | ↓ | 36.88 ± 17.05   | 40.70  | 18.78  | 51.75  | 15.49  | 57.70  | 0.0000 |
| | **— Distribution —** | | | | | | | | | |
| A1  | Path MMD²                       | Distribution  | ↓ | 0.0181 ± 0.0147 | 0.0093 | 0.0046 | 0.0322 | 0.0051 | 0.0393 | 0.0015 |
| A2  | Terminal MMD²                   | Distribution  | ↓ | 0.0308 ± 0.0229 | 0.0256 | 0.0103 | 0.0681 | 0.0060 | 0.0439 | 0.0016 |
| A3  | Increment MMD²                  | Distribution  | ↓ | 0.0077 ± 0.0039 | 0.0048 | 0.0070 | 0.0112 | 0.0023 | 0.0129 | 0.0007 |
| A4  | Volatility MMD                  | Distribution  | ↓ | 0.3933 ± 0.2553 | 0.1700 | 0.3416 | 0.6572 | 0.0797 | 0.7179 | 0.0071 |
| A5  | Terminal SWD                    | Distribution  | ↓ | 3.128 ± 0.923   | 2.951  | 2.058  | 4.531  | 2.310  | 3.793  | 0.687  |
| A6  | Path SWD                        | Distribution  | ↓ | 1.634 ± 0.576   | 1.279  | 0.970  | 2.462  | 1.289  | 2.171  | 0.438  |
| A24 | RV Law Loss (W₁ on ann. RV)     | Distribution  | ↓ | 1.5512 ± 0.3788 | 1.4914 | 1.7536 | 1.8266 | 0.8373 | 1.8470 | 0.0000 |
| A25 | Mean Path RMSE                  | Distribution  | ↓ | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0.0000 |
| A27 | KS on Log-returns               | Distribution  | ↓ | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0.0000 |
| A28 | Skewness Error                  | Distribution  | ↓ | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0.0000 |
| A29 | QQ RMSE (300-pt)                | Distribution  | ↓ | 0.0025 ± 0.0006 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0.0000 |
| A33 | Terminal Price KS               | Distribution  | ↓ | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0.0000 |
| | **— Adversarial —** | | | | | | | | | |
| A13 | Disc Score GRU (log-ret.)       | Adversarial   | ↓ | 0.010 ± 0.008   | 0.004  | 0.012  | 0.002  | 0.025  | 0.006  | 0.008  |
| A13 | Disc Score MLP (log-ret.)       | Adversarial   | ↓ | 0.092 ± 0.046   | 0.128  | 0.005  | 0.083  | 0.118  | 0.126  | 0.010  |
| | **— Predictive —** | | | | | | | | | |
| A14 | Pred Score GRU — TSTR           | Predictive    | ↓ | 0.057 ± 0.001   | 0.055  | 0.059  | 0.058  | 0.056  | 0.057  | 0.054  |
| A14 | Pred Score MLP — TSTR           | Predictive    | ↓ | 0.057 ± 0.002   | 0.056  | 0.059  | 0.057  | 0.056  | 0.059  | 0.054  |
| | **— Temporal —** | | | | | | | | | |
| A7  | Cov Error (%)                   | Temporal      | ↓ | 17.75 ± 6.71    | 8.83   | 18.76  | 14.81  | 29.37  | 16.98  | 0.00   |
| A11 | ACF Error \|log-returns\|       | Temporal      | ↓ | 0.125 ± 0.067   | 0.065  | 0.105  | 0.201  | 0.048  | 0.208  | 0.0000 |
| A12 | ACF Error log-returns²          | Temporal      | ↓ | 0.084 ± 0.035   | 0.045  | 0.079  | 0.117  | 0.048  | 0.130  | 0.0000 |
| A22 | ACF \|r\| Lag-1 Error           | Temporal      | ↓ | 0.2264 ± 0.1034 | 0.1537 | 0.2120 | 0.3669 | 0.0840 | 0.3155 | 0.0000 |
| A23 | ACF r² Lag-1 Error              | Temporal      | ↓ | 0.1719 ± 0.0626 | 0.1177 | 0.2000 | 0.2634 | 0.0874 | 0.1908 | 0.0000 |
| | **— Vol —** | | | | | | | | | |
| A8  | Mean RMSE                       | Vol           | ↓ | 0.739 ± 0.455   | 0.832  | 0.389  | 1.056  | 1.341  | 0.074  | 0.000  |
| A9  | Return Std Error                | Vol           | ↓ | 0.152 ± 0.089   | 0.152  | 0.238  | 0.030  | 0.079  | 0.261  | 0.0000 |
| A16 | Log-Return Std Error            | Vol           | ↓ | 0.0017 ± 0.0008 | 0.0020 | 0.0023 | 0.0006 | 0.0010 | 0.0025 | 0.0000 |
| A19 | Kurtosis Ratio (target/model)   | Vol           | — | -1.095 ± 3.525  | 1.979  | 0.136  | 0.245  | -8.005 | 0.172  | 1.0000 |
| A20 | Sigma Mean Error                | Vol           | ↓ | 0.0307 ± 0.0089 | 0.0301 | 0.0373 | 0.0273 | 0.0164 | 0.0422 | 0.0000 |
| A26 | Cross-Sect. Vol Path RMSE       | Vol           | ↓ | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0.0000 |
| A31 | Rolling Vol KS (window=5)       | Vol           | ↓ | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0.0000 |
| A32 | Vol-of-Vol Error                | Vol           | ↓ | 0.0009 ± 0.0009 | 0.0004 | 0.0003 | 0.0025 | 0.0003 | 0.0011 | 0.0000 |
| | **— Heston Spec —** | | | | | | | | | |
| A15 | Sigma Corr (vol recovery)       | Heston Spec   | ↑ | 0.002 ± 0.009   | 0.001  | 0.007  | −0.008 | −0.006 | 0.017  | 0.614  |
| A15 | Sigma RMSE                      | Heston Spec   | ↓ | 0.118 ± 0.018   | 0.102  | 0.111  | 0.148  | 0.100  | 0.131  | 0.065  |
| A21 | Learned/Oracle Sigma Corr       | Heston Spec   | ↑ | 0.0021 ± 0.0090 | 0.0010 | 0.0069 | -0.0082| -0.0057| 0.0166 | 0.614  |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction. A19 Kurtosis Ratio: perfect = 1.0.
> **A11–A12**: ACF on log-returns r_t = log(S_{t+1}/S_t). ARCH signal: |r_t| has positive lag-1 ACF ~0.05 in Heston.
> **A13**: Discriminative classifier trained on log-returns (not raw prices). Score = |accuracy − 0.5|; 0 = indistinguishable.
> **A14**: TSTR MAE; all methods cluster near 0.056–0.059 (irreducible log-return noise floor). Differences are small.
> **A16**: Log-return std error. Uses $r_t = \log(S_{t+1}/S_t)$ (vs price increments in A9).
> **A17–A18**: 95th/99th quantile error on |log-returns|. Measures tail reproduction accuracy.
> **A19**: Kurtosis ratio real/gen. Perfect = 1.0. Negative value means gen has negative excess kurtosis (lighter-than-Gaussian tails).
> **A20**: Sigma mean error — annualized per-path vol, averaged over paths.
> **A21**: Learned/oracle sigma corr — identical to A15 Corr; included to group all vol metrics.
> **A22–A23**: ACF lag-1 error on |r| and r². Single-lag version of A11/A12. Heston true values ≈ +0.052 / +0.050.
> **A24**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt. Ref: Barndorff-Nielsen & Shephard (2002).
> **A25–A26**: Path-level RMSE between real and generated mean/vol trajectories (matched by time).
> **A27**: Kolmogorov-Smirnov statistic on pooled log-returns.
> **A28**: |skew_real − skew_gen|. Heston true skew ≈ −0.45.
> **A29**: QQ RMSE over 300 uniform quantile levels.
> **A30**: QQ error restricted to top-5% tail quantiles.
> **A31**: KS statistic on rolling-5 volatility histograms.
> **A32**: |vol-of-vol_real − vol-of-vol_gen|.
> **A33**: KS statistic on terminal prices S_T (= S_128).
> **A34**: |Hill tail index_real − Hill tail index_gen|. Hill estimator on |log-returns| above 95th pct.

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
| **Log-return histogram** | MSE   | 144.2 ± 120.6 | 11.05 | 20.68 | 504.5 | 14.58 | 170.3 | 0.0 |
|                          | % err | 75440% ± 24681% | 83815% | 36244% | 111604% | 89267% | 56270% | 0% |
| **QQ plot**              | MSE   | 7.09e-6 ± 3.3e-6 | 4.16e-6 | 7.22e-6 | 8.07e-6 | 3.19e-6 | 1.28e-5 | 0.0 |
|                          | % err | 157.9% ± 27.2% | 124.1% | 121.9% | 227.3% | 134.6% | 181.7% | 0% |
| **ACF \|r\|**            | MSE   | 1.05e-2 ± 8.5e-3 | 2.90e-3 | 5.28e-3 | 1.82e-2 | 1.01e-3 | 2.50e-2 | 0.0 |
|                          | % err | 2342.1% ± 1351.8% | 2849.1% | 972.8% | 1611.4% | 647.3% | 5630.1% | 0% |
| **ACF r²**               | MSE   | 5.77e-3 ± 3.3e-3 | 1.96e-3 | 4.27e-3 | 7.24e-3 | 1.04e-3 | 1.43e-2 | 0.0 |
|                          | % err | 4578.1% ± 4266.4% | 5737.1% | 1155.4% | 1523.8% | 824.4% | 13649.7% | 0% |
| **Rolling vol hist.**    | MSE   | 439.3 ± 216.7 | 280.4 | 633.9 | 604.2 | 82.5 | 595.6 | 0.0 |
|                          | % err | 882.8% ± 410.8% | 633.7% | 618.7% | 964.8% | 544.9% | 1652.0% | 0% |
| **Tail survival**        | MSE   | 1.17e-2 ± 9.2e-3 | 2.67e-3 | 6.07e-3 | 2.19e-2 | 4.12e-3 | 2.38e-2 | 0.0 |
|                          | % err | 24327% ± 11324% | 17672% | 16124% | 45821% | 16154% | 25867% | 0% |

> **Log-ret histogram**: MSE std (120.6) approaches the mean (144.2), driven by seed 2 near-collapse (504.5 vs 11–170 for the other seeds). A std near the mean is expected for a non-negative combined score over 5 skewed samples — it is not a bug. SBTS is far more stable here (12.1 ± 0.16).
> **ACF \|r\|, ACF r²**: TimeGAN misses the ARCH signature — near-zero ACF vs Heston ≈ +0.05 — so both MSE and % error are larger than SBTS.
> **Rolling vol histogram**: High MSE (439) from vol-distribution mismatch across most seeds.
> **% error caveat**: the very large % values (log-ret hist 75440%, tail survival 24327%) come from dividing by near-zero real-curve values in empty histogram bins / deep tails — read MSE for the level of agreement, % error for relative shape deviation.

---

## Stylised Facts Diagnostic (Heston vs TimeGAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/TimeGAN/plots/heston_diagnostics.png)

---

## TimeGAN Training Loss (5 seeds)

Loss curves across all three training phases (embedding 0–5k, supervisor 5–10k, joint 10–20k).

![TimeGAN Training Loss](losses/loss_convergence.png)

---

## A13 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/TimeGAN/plots/disc_classifier_loss.png)

---

## A14 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/TimeGAN/plots/pred_score_loss.png)

---

## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed it as a **65D murex-style feature vector**
(63 step-by-step log-returns + terminal cumulative return + realized volatility, z-scored
using the generated pool distribution), retrieve K=77 nearest TimeGAN paths by L2 distance
in that space, then use their price-anchored futures (steps 64–127) as a forecast ensemble.
Two variants: flat average (**Uniform**) and distance-weighted (**Gaussian**,
per-query η = η̃·‖z(x̃)‖ with η̃ = median(dist)/median(‖z‖) calibrated from data).

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/TimeGAN/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/TimeGAN/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

Embedding: **65D murex-style prefix features** — 63 log-returns + 1 terminal return + 1 realized vol,
z-scored per dimension using the generated pool. Adaptive Gaussian bandwidth: η = η̃·‖z(x̃)‖, η̃ = median(dist)/median(‖z‖).

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | **3.087 ± 0.340** | 3.087 ± 0.341 | **4.372 ± 0.431** | 4.373 ± 0.432 | 3.73 / 5.30 |
| MAE    | 4.039 ± 0.228 | 4.039 ± 0.229 | 5.680 ± 0.178 | 5.681 ± 0.179 | 3.73 / 5.30 |
| RMSE   | 5.452 ± 0.293 | 5.452 ± 0.293 | 7.667 ± 0.203 | 7.668 ± 0.203 | 5.07 / 7.18 |

PS-MC **beats the naive RW on CRPS** at both horizons (3.09 < 3.73 at H=32; 4.37 < 5.30 at H=64).
Uniform ≈ Gaussian: Heston is time-homogeneous, so K nearest neighbours are roughly equally predictive.
The 65D embedding selects paths by full trajectory shape (vs 22D eq.13 which captures only market regime).

Full analysis: [`results/Heston/TimeGAN/path_shadowing/README.md`](../../results/Heston/TimeGAN/path_shadowing/README.md)

---

## File layout

```
methods/TimeGAN/
├── README.md                          ← this file
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json                  seed, shape, min/max, train time
├── weights/
│   ├── seed_{i}_model.pt              full PyTorch state_dict
│   └── seed_{i}_config.json           hyperparameters
├── losses/
│   ├── seed_{i}_losses.csv            step, phase, e_loss, s_loss, g_loss, d_loss
│   └── loss_convergence.png           convergence plot (5 seeds overlaid)
└── code/
    ├── timegan_torch.py               PyTorch TimeGAN implementation
    ├── train.py                       orchestrator — 5 seeds on 2 GPUs in pairs
    ├── train_seed.py                  single-seed worker
    ├── reference/                     verbatim TF1 code (jsyoon0823/TimeGAN)
    └── README.md                      paper, GitHub, list of 5 fixes vs TF1
```

## Reproduce

```bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/TimeGAN/code
python train.py --gpu0 0 --gpu1 3

# Compute metrics
cd metrics
python compute_all.py --method TimeGAN --dataset Heston
```
