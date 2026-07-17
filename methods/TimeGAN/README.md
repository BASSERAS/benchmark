# TimeGAN on Heston

PyTorch reimplementation of **TimeGAN** (Yoon et al., NeurIPS 2019) trained on 8 192
Heston stochastic-volatility price paths (seq\_len = 128).

See [`code/README.md`](code/README.md) for source, original paper, and the 5 fixes applied
to the TF1 reference implementation.

---

## Metrics A1–A20 — mean ± std across 5 seeds

> All metrics on **log-returns** r_t = log(S_{t+1}/S_t). A11/A12/A13/A14/A16 were fixed from
> price increments ΔS_t after validation against the ACF/tail-survival stylized-facts literature.

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|--------|
| A1  | Path MMD²                       | Distribution  | ↓ | 0.0181 ± 0.0147 | 0.0093 | 0.0046 | 0.0322 | 0.0051 | 0.0393 | 0.0018 |
| A2  | Terminal MMD²                   | Distribution  | ↓ | 0.0308 ± 0.0229 | 0.0256 | 0.0103 | 0.0681 | 0.0060 | 0.0439 | 0.0016 |
| A3  | Increment MMD²                  | Distribution  | ↓ | 0.0077 ± 0.0039 | 0.0048 | 0.0070 | 0.0112 | 0.0023 | 0.0129 | 0.0008 |
| A4  | Volatility MMD                  | Distribution  | ↓ | 0.3933 ± 0.2553 | 0.1700 | 0.3416 | 0.6572 | 0.0797 | 0.7179 | 0.0082 |
| A5  | Terminal SWD                    | Distribution  | ↓ | 3.128 ± 0.923   | 2.951  | 2.058  | 4.531  | 2.310  | 3.793  | 0.763  |
| A6  | Path SWD                        | Distribution  | ↓ | 1.634 ± 0.576   | 1.279  | 0.970  | 2.462  | 1.289  | 2.171  | 0.554  |
| A7  | Cov Error (%)                   | Statistics    | ↓ | 17.75 ± 6.71    | 8.83   | 18.76  | 14.81  | 29.37  | 16.98  | 4.76   |
| A8  | Mean RMSE                       | Statistics    | ↓ | 0.739 ± 0.455   | 0.832  | 0.389  | 1.056  | 1.341  | 0.074  | 0.140  |
| A9  | Return Std Error                | Statistics    | ↓ | 0.152 ± 0.089   | 0.152  | 0.238  | 0.030  | 0.079  | 0.261  | 0.0048 |
| A10 | Kurtosis Error                  | Statistics    | ↓ | 2.955 ± 2.099   | 0.015  | 5.360  | 3.768  | 0.958  | 4.672  | 0.017  |
| A11 | ACF Error \|log-returns\|       | Temporal      | ↓ | 0.125 ± 0.067   | 0.065  | 0.105  | 0.201  | 0.048  | 0.208  | 0.0017 |
| A12 | ACF Error log-returns²          | Temporal      | ↓ | 0.084 ± 0.035   | 0.045  | 0.079  | 0.117  | 0.048  | 0.130  | 0.0014 |
| A13 | Disc Score GRU (log-ret.)       | Adversarial   | ↓ | 0.010 ± 0.008   | 0.004  | 0.012  | 0.002  | 0.025  | 0.006  | 0.012  |
| A13 | Disc Score MLP (log-ret.)       | Adversarial   | ↓ | 0.092 ± 0.046   | 0.128  | 0.005  | 0.083  | 0.118  | 0.126  | 0.008  |
| A14 | Pred Score GRU — TSTR           | Predictive    | ↓ | 0.057 ± 0.001   | 0.055  | 0.059  | 0.058  | 0.056  | 0.057  | 0.056  |
| A14 | Pred Score MLP — TSTR           | Predictive    | ↓ | 0.057 ± 0.002   | 0.056  | 0.059  | 0.057  | 0.056  | 0.059  | 0.057  |
| A15 | Sigma Corr (vol recovery)       | Heston-specif.| ↑ | 0.002 ± 0.009   | 0.001  | 0.007  | −0.008 | −0.006 | 0.017  | 0.614  |
| A15 | Sigma RMSE                      | Heston-specif.| ↓ | 0.118 ± 0.018   | 0.102  | 0.111  | 0.148  | 0.100  | 0.131  | 0.065  |
| A16 | Tail Survival (log-ret.)        | Fat-tail      | ↓ | 0.023 ± 0.011   | 0.030  | 0.037  | 0.008  | 0.014  | 0.028  | ≈0     |
| A17 | Oracle MAE — AR(5) on real      | Predictive    | ↓ | 0.0097 ± 0.0000 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | ≈0.0097|
| A18 | Agent MAE — AR(5) TSTR          | Predictive    | ↓ | 0.0101 ± 0.0003 | 0.0098 | 0.0106 | 0.0100 | 0.0100 | 0.0099 | =A17   |
| A19 | Oracle-Agent Corr               | Predictive    | ↑ | −0.33 ± 0.31    | −0.77  | −0.04  | −0.21  | −0.03  | −0.61  | ≈1     |
| A20 | RV Law Loss (W₁ on ann. RV)     | Distribution  | ↓ | 1.551 ± 0.379   | 1.491  | 1.754  | 1.827  | 0.837  | 1.847  | ≈0     |

> **A11–A12**: ACF on log-returns r_t = log(S_{t+1}/S_t). ARCH signal: |r_t| has positive lag-1 ACF ~0.05 in Heston.
> **A13**: Discriminative classifier trained on log-returns (not raw prices). Score = |accuracy − 0.5|; 0 = indistinguishable.
> **A14**: TSTR MAE; all methods cluster near 0.056–0.059 (irreducible log-return noise floor). Differences are small.
> **A16**: Tail survival on |log-returns|. Quantiles {0.90, 0.95, 0.99} of real used as thresholds.
> **A17–A19**: OLS AR(5) oracle/agent protocol on log-returns. A19 negative = Heston has near-zero autocorrelation, so predictions are white noise — metric is near floor.
> **A20**: W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t}/dt (ann. realized variance per path). Ref: Barndorff-Nielsen & Shephard (2002).

---

## Stylized Metrics B1–B14 — mean ± std across 5 seeds

> One scalar per diagnostic plot panel. Extracted from the same data as the 8-panel PNG.
> All ↓ lower is better.

| ID  | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|-----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| B1  | Mean Path RMSE        | Distribution | ↓ | 0.5289 ± 0.2624 | 0.5327 | 0.2184 | 0.8536 | 0.7866 | 0.2533 | 0.1511 ± 0.0708 |
| B2  | Cross-Sect. Vol RMSE  | Distribution | ↓ | 0.3534 ± 0.1253 | 0.2220 | 0.4752 | 0.2585 | 0.5320 | 0.2790 | 0.1355 ± 0.0735 |
| B3  | KS on Log-returns     | Distribution | ↓ | 0.0848 ± 0.0374 | 0.0400 | 0.0627 | 0.1259 | 0.0628 | 0.1329 | 0.0018 ± 0.0009 |
| B4  | Skewness Error        | Distribution | ↓ | 0.3404 ± 0.3344 | 0.0025 | 0.4473 | 0.0891 | 0.2252 | 0.9379 | 0.0060 ± 0.0048 |
| B5  | QQ RMSE (300-pt)      | Distribution | ↓ | 0.0025 ± 0.0006 | 0.0019 | 0.0026 | 0.0028 | 0.0017 | 0.0035 | 0.0001 ± 0.0000 |
| B6  | Tail QQ Error         | Fat-tail     | ↓ | 0.0034 ± 0.0015 | 0.0042 | 0.0054 | 0.0016 | 0.0017 | 0.0041 | 0.0001 ± 0.0001 |
| B7  | ACF lag-1 |r| Err     | Temporal     | ↓ | 0.2282 ± 0.1042 | 0.1549 | 0.2137 | 0.3698 | 0.0847 | 0.3180 | 0.0018 ± 0.0016 |
| B8  | ARCH Persistence Err  | Temporal     | ↓ | 0.0591 ± 0.0359 | 0.0272 | 0.0436 | 0.0895 | 0.0221 | 0.1130 | 0.0011 ± 0.0005 |
| B9  | ACF lag-1 r² Err      | Temporal     | ↓ | 0.1732 ± 0.0631 | 0.1186 | 0.2016 | 0.2655 | 0.0881 | 0.1923 | 0.0017 ± 0.0014 |
| B10 | GARCH Persistence Err | Temporal     | ↓ | 0.0328 ± 0.0151 | 0.0173 | 0.0265 | 0.0380 | 0.0224 | 0.0598 | 0.0010 ± 0.0006 |
| B11 | Rolling Vol KS        | Volatility   | ↓ | 0.2540 ± 0.1093 | 0.1877 | 0.2705 | 0.3619 | 0.0805 | 0.3695 | 0.0046 ± 0.0024 |
| B12 | Vol-of-Vol Error      | Volatility   | ↓ | 0.0009 ± 0.0009 | 0.0004 | 0.0003 | 0.0025 | 0.0003 | 0.0011 | 0.0000 ± 0.0000 |
| B13 | Terminal Price KS     | Distribution | ↓ | 0.1121 ± 0.0556 | 0.1077 | 0.0573 | 0.2074 | 0.0574 | 0.1307 | 0.0145 ± 0.0043 |
| B14 | Hill Tail Index Err   | Fat-tail     | ↓ | 36.88 ± 17.05   | 40.70  | 18.78  | 51.75  | 15.49  | 57.70  | 0.499 ± 0.610   |

> **B7–B10** (ACF metrics): Heston true ACF(|r|, lag=1) ≈ +0.052, ACF(r², lag=1) ≈ +0.050.
> TimeGAN often collapses to near-zero ACF, missing the ARCH signature (seeds 2, 4 worst).
> **B14**: Hill estimator on terminal prices S_T. Large variance across seeds → noisy estimate.
> This metric is informative across methods but noisy per seed. Use the mean.

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
