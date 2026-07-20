# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with **34 metrics (A1–A34)**
plus **6 curve-shape diagnostics (B)** — each scored by both **MSE** and **% error** — across 5 random seeds.
Every table carries a **Perfect floor** column: the reproducible best-case score from a row-shuffled copy
of the real data (see [`methods/perfect_recovery/`](methods/perfect_recovery/)).

---

## Results — Heston (mean ± std, 5 seeds)

Cross-method comparison on 8 192 Heston price paths (seq\_len=128).
↓ = lower is better. ↑ = higher is better. **Bold** = best across methods.

### A1–A34 — Metrics by category

| Metric | SBTS | TimeGAN | Fourier Flow | Diffusion-TS | CSDI | TimeVAE | TimeVQVAE | Perfect | Winner |
|--------|:----:|:-------:|:------------:|:------------:|:----:|:-------:| :---------: |:-------:|:------:|
| **— Fat Tail —** | | | | | | |  | | |
| A1 Kurtosis Error ↓ | 0.1187 ± 0.0060 | 2.955 ± 2.099 | 0.5757 ± 0.0083 | 0.4238 ± 0.0230 | **0.0958 ± 0.0262** | 2.258 ± 0.5719 | 0.1367 ± 0.0924 | 0 | **CSDI** |
| A2 \|r\| q95 Error ↓ | 0.0063 ± 3.00e-05 | 0.0032 ± 0.0018 | **6.52e-04 ± 2.10e-04** | 0.0068 ± 1.57e-04 | 0.0053 ± 1.50e-04 | 0.0222 ± 1.22e-04 | 0.0044 ± 2.54e-04 | 0 | **Fourier Flow** |
| A3 \|r\| q99 Error ↓ | 0.0098 ± 4.80e-05 | 0.0043 ± 0.0028 | **0.0023 ± 5.06e-04** | 0.0103 ± 1.75e-04 | 0.0073 ± 2.29e-04 | 0.0308 ± 1.05e-04 | 0.0060 ± 3.03e-04 | 0 | **Fourier Flow** |
| A4 Tail QQ Error ↓ | 0.0062 ± 2.60e-05 | 0.0034 ± 0.0015 | **7.15e-04 ± 1.23e-04** | 0.0067 ± 1.50e-04 | 0.0052 ± 1.50e-04 | 0.0219 ± 1.17e-04 | 0.0044 ± 2.48e-04 | 0 | **Fourier Flow** |
| A5 Hill Tail Index Error ↓ | 9.499 ± 0.3457 | 36.885 ± 17.053 | 6.368 ± 2.000 | 3.613 ± 0.2789 | **1.992 ± 0.5856** | 2.396 ± 0.6794 | 4.342 ± 1.193 | 0 | **CSDI** |
| **— Distribution —** | | | | | | |  | | |
| A6 Path MMD² ↓ | 0.0110 ± 0.0026 | 0.0165 ± 0.0127 | 0.0052 ± 0.0019 | 0.0033 ± 6.56e-04 | **0.0027 ± 6.16e-04** | 0.0184 ± 9.55e-04 | 0.0039 ± 7.71e-04 | 0.0015 | **CSDI** |
| A7 Terminal MMD² ↓ | 0.0100 ± 0.0036 | 0.0267 ± 0.0192 | 0.0106 ± 0.0051 | **0.0021 ± 3.92e-04** | 0.0028 ± 0.0011 | 0.0042 ± 0.0011 | 0.0046 ± 9.75e-04 | 0.0016 | **Diffusion-TS** |
| A8 Increment MMD² ↓ | 0.0071 ± 2.47e-04 | 0.0077 ± 0.0041 | **0.0011 ± 7.70e-05** | 0.0112 ± 9.37e-04 | 0.0079 ± 8.54e-04 | 0.2134 ± 0.0012 | 0.0071 ± 9.95e-04 | 7.45e-04 | **Fourier Flow** |
| A9 Volatility MMD ↓ | 0.3038 ± 0.0071 | 0.3789 ± 0.2430 | **0.0596 ± 0.0086** | 0.3840 ± 0.0314 | 0.2448 ± 0.0206 | 3.591 ± 0.4563 | 0.1966 ± 0.0273 | 0.0071 | **Fourier Flow** |
| A10 Terminal SWD ↓ | 3.539 ± 0.7368 | 2.658 ± 0.8567 | 2.815 ± 0.9433 | 1.358 ± 0.2152 | **1.303 ± 0.2465** | 1.798 ± 0.2603 | 1.504 ± 0.4262 | 0.6873 | **CSDI** |
| A11 Path SWD ↓ | 2.415 ± 0.4104 | 1.417 ± 0.4914 | 1.289 ± 0.4198 | 0.9838 ± 0.1107 | **0.7712 ± 0.1581** | 0.9882 ± 0.2052 | 0.9413 ± 0.2060 | 0.4381 | **CSDI** |
| A12 RV Law Loss ↓ | 2.148 ± 0.0074 | 1.551 ± 0.3788 | **0.5291 ± 0.1299** | 2.250 ± 0.0491 | 1.897 ± 0.0563 | 4.986 ± 0.0084 | 1.682 ± 0.0893 | 0 | **Fourier Flow** |
| A13 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | 0.5289 ± 0.2624 | 0.4910 ± 0.4022 | 0.3615 ± 0.2364 | 0.3101 ± 0.3036 | **0.2981 ± 0.2172** | 0.6974 ± 0.1790 | 0 | **TimeVAE** |
| A14 KS Log-returns ↓ | 0.0534 ± 3.62e-04 | 0.0848 ± 0.0374 | **0.0191 ± 0.0024** | 0.0600 ± 0.0019 | 0.0539 ± 0.0021 | 0.3673 ± 0.0046 | 0.0501 ± 0.0036 | 0 | **Fourier Flow** |
| A15 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | 0.0282 ± 0.0152 | 0.0698 ± 0.0358 | 0.0457 ± 0.0021 | 0.5568 ± 0.0984 | 0.0397 ± 0.0082 | 0 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 0.0028 ± 1.20e-05 | 0.0025 ± 6.43e-04 | **5.86e-04 ± 4.20e-05** | 0.0030 ± 8.30e-05 | 0.0026 ± 8.60e-05 | 0.0105 ± 8.40e-05 | 0.0022 ± 1.38e-04 | 0 | **Fourier Flow** |
| A17 Terminal Price KS ↓ | 0.0921 ± 0.0051 | 0.1121 ± 0.0556 | 0.0848 ± 0.0166 | 0.0400 ± 0.0073 | **0.0321 ± 0.0053** | 0.0478 ± 0.0099 | 0.0531 ± 0.0067 | 0 | **CSDI** |
| **— Adversarial —** | | | | | | |  | | |
| A18 Disc Score GRU ↓ | 0.2755 ± 0.2166 | **0.0077 ± 0.0050** | 0.0094 ± 0.0097 | 0.2621 ± 0.1578 | 0.0470 ± 0.0901 | 0.2035 ± 0.1934 | 0.0237 ± 0.0198 | 0.0071 | **TimeGAN** |
| A18 Disc Score MLP ↓ | 0.0079 ± 0.0049 | 0.1031 ± 0.0395 | 0.0053 ± 0.0041 | 0.0554 ± 0.0396 | **0.0046 ± 0.0058** | 0.1756 ± 0.1354 | 0.0074 ± 0.0033 | 0.0033 | **CSDI** |
| **— Predictive —** | | | | | | |  | | |
| A19 Pred Score GRU ↓ | 0.0586 ± 5.90e-05 | 0.0574 ± 0.0019 | **0.0537 ± 7.60e-06** | 0.0549 ± 1.59e-04 | 0.0539 ± 3.20e-05 | 0.0577 ± 8.53e-04 | 0.0539 ± 4.70e-05 | 0.0537 | **Fourier Flow** |
| A19 Pred Score MLP ↓ | 0.0583 ± 2.55e-04 | 0.0570 ± 0.0012 | **0.0540 ± 4.90e-04** | 0.0551 ± 3.72e-04 | 0.0541 ± 2.47e-04 | 0.0564 ± 2.75e-04 | 0.0540 ± 3.45e-04 | 0.0537 | **Fourier Flow** |
| **— Temporal —** | | | | | | |  | | |
| A20 Covariance Error ↓ | 145.35 ± 4.886 | 17.751 ± 6.707 | 64.406 ± 38.255 | 38.172 ± 10.637 | 35.538 ± 5.776 | 51.272 ± 1.758 | **16.599 ± 14.720** | 0 | **TimeVQVAE** |
| A21 ACF \|r\| Error (lags) ↓ | 0.0596 ± 4.70e-04 | 0.1252 ± 0.0674 | 0.0435 ± 5.50e-04 | 0.0201 ± 0.0030 | **0.0091 ± 0.0026** | 0.3865 ± 0.1057 | 0.0172 ± 0.0042 | 0 | **CSDI** |
| A22 ACF r² Error (lags) ↓ | 0.0619 ± 5.08e-04 | 0.0839 ± 0.0348 | 0.0379 ± 5.56e-04 | 0.0168 ± 0.0027 | **0.0086 ± 0.0021** | 0.3580 ± 0.0885 | 0.0152 ± 0.0033 | 0 | **CSDI** |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1437 ± 0.0012 | 0.2264 ± 0.1034 | 0.0526 ± 7.04e-04 | **0.0039 ± 0.0022** | 0.0188 ± 0.0048 | 0.4637 ± 0.1346 | 0.0115 ± 0.0080 | 0 | **Diffusion-TS** |
| A24 ACF r² Lag-1 Error ↓ | 0.1665 ± 0.0017 | 0.1719 ± 0.0626 | 0.0461 ± 7.01e-04 | **0.0038 ± 0.0026** | 0.0176 ± 0.0036 | 0.4589 ± 0.1189 | 0.0091 ± 0.0073 | 0 | **Diffusion-TS** |
| **— Vol —** | | | | | | |  | | |
| A25 Mean RMSE ↓ | 1.301 ± 0.2776 | 0.7385 ± 0.4552 | 0.9000 ± 0.8807 | 0.5767 ± 0.4444 | 0.3729 ± 0.4145 | **0.3396 ± 0.2710** | 0.9147 ± 0.1675 | 0 | **TimeVAE** |
| A26 Return Std Error ↓ | 0.2492 ± 0.0018 | 0.1519 ± 0.0888 | **0.0058 ± 0.0028** | 0.3098 ± 0.0093 | 0.2570 ± 0.0098 | 1.073 ± 0.0078 | 0.2306 ± 0.0142 | 0 | **Fourier Flow** |
| A27 Log-Return Std Error ↓ | 0.0030 ± 1.20e-05 | 0.0017 ± 7.78e-04 | **6.70e-05 ± 6.60e-05** | 0.0032 ± 8.20e-05 | 0.0026 ± 8.90e-05 | 0.0109 ± 7.80e-05 | 0.0023 ± 1.37e-04 | 0 | **Fourier Flow** |
| A28 Kurtosis Ratio (→ 1) | 1.989 ± 0.0182 | −1.095 ± 3.525 | 3.039 ± 0.7605 | 1.866 ± 0.2509 | **0.8539 ± 0.0298** | 0.2780 ± 0.0467 | 0.8249 ± 0.0682 | 1.000 | **CSDI** |
| A29 Sigma Mean Error ↓ | 0.0440 ± 1.84e-04 | 0.0307 ± 0.0089 | **0.0026 ± 8.77e-04** | 0.0485 ± 0.0013 | 0.0404 ± 0.0015 | 0.1741 ± 0.0018 | 0.0371 ± 0.0021 | 0 | **Fourier Flow** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 3.276 ± 0.0637 | **0.3534 ± 0.1253** | 1.367 ± 0.4499 | 1.154 ± 0.2019 | 0.9262 ± 0.1315 | 1.122 ± 0.0447 | 0.3781 ± 0.3311 | 0 | **TimeGAN** |
| A31 Rolling Vol KS (w=5) ↓ | 0.3435 ± 6.43e-04 | 0.2540 ± 0.1093 | **0.0740 ± 0.0014** | 0.2558 ± 0.0078 | 0.2180 ± 0.0082 | 0.9871 ± 0.0045 | 0.1828 ± 0.0102 | 0 | **Fourier Flow** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 6.00e-06 | 8.97e-04 ± 8.69e-04 | 6.88e-04 ± 9.20e-05 | 0.0016 ± 3.80e-05 | 0.0010 ± 2.10e-05 | 0.0046 ± 5.60e-05 | **6.75e-04 ± 5.80e-05** | 0 | **TimeVQVAE** |
| **— Heston Spec —** | | | | | | |  | | |
| A33 Teacher-Sigma Corr ↑ | 0.0046 ± 0.0019 | 0.0021 ± 0.0090 | 7.85e-04 ± 0.0038 | −0.0036 ± 0.0032 | 0.0084 ± 0.0040 | **0.0273 ± 0.0050** | 0.0037 ± 0.0036 | 0.6143 | **TimeVAE** |
| A34 Teacher-Sigma RMSE ↓ | 0.0955 ± 9.10e-05 | 0.1183 ± 0.0184 | **0.0894 ± 0.0013** | 0.0960 ± 7.41e-04 | 0.0985 ± 6.61e-04 | 0.1793 ± 0.0016 | 0.1008 ± 0.0010 | 0.0654 | **Fourier Flow** |
| PS-MC CRPS H=32 ↓ | 2.761 ± 0.004 | 3.087 ± 0.340 | 2.742 ± 0.027 | 2.717 ± 0.003 | **2.713 ± 0.005** | 3.855 ± 0.070 | 2.771 ± 0.015 | — | **CSDI** |
| PS-MC CRPS H=64 ↓ | 3.900 ± 0.008 | 4.372 ± 0.431 | 3.992 ± 0.106 | 3.845 ± 0.005 | **3.814 ± 0.007** | 5.634 ± 0.124 | 3.889 ± 0.017 | — | **CSDI** |

**Fourier Flow wins A: 15/38. CSDI wins 12/38. Diffusion-TS wins 3/38. TimeVAE wins 3/38. TimeVQVAE wins 2/38. TimeGAN wins 2/38. SBTS wins 1/38.**
Adding **TimeVQVAE** takes exactly two structural metrics away from the incumbents: the covariance error
(A20, 16.60 — the lowest of any method, edging out TimeGAN's 17.75) and the vol-of-vol error (A32,
6.75e-04 — narrowly under Fourier Flow's 6.88e-04). More important than its two outright wins is its
**overall balance**: TimeVQVAE never produces a catastrophic column — its fat-tail kurtosis ratio (A28
0.82, second only to CSDI's 0.85 and far above TimeVAE's 0.28), its volatility MMD (A9 0.20, second to
Fourier Flow) and its GRU discriminative score (A18-GRU 0.024, near the 0.004 floor) are all among the best
two or three methods, and it is one of only two pools (with Diffusion-TS) whose Path-Shadowing ensemble
beats the naive random-walk at *both* horizons. **TimeVAE** wins three *averaging* metrics — the mean-path
RMSE (A13), the mean vol-path RMSE (A25), and the teacher-sigma correlation (A33) — but produces the worst
tails and vol shapes of any method (A1, A9, A31) and loses all six B curve-shape plots. CSDI keeps the
best-calibrated marginals of any method, taking the fat-tail kurtosis (A1) and Hill index (A5), the
distributional MMD/SWD family (A6, A10, A11), the terminal-price KS (A17), the MLP discriminative score
(A18-MLP), the ACF-lag-average errors (A21, A22), the kurtosis ratio (A28, the only method within 0.15 of
the ideal 1.0), and **both** Path-Shadowing horizons. Fourier Flow keeps the marginal/spectral-fidelity
crown on the log-return tails (A2–A4), MMD/increment (A8, A9), the RV-law (A12), KS (A14), QQ (A16), vol
moments (A26, A27, A29), rolling-vol KS (A31) and the teacher RMSE (A34) — 15 metrics where its
explicit-likelihood spectral training is tightest. Diffusion-TS retains the terminal MMD (A7) and the two
lag-1 ACF metrics (A23, A24) where its seasonal-trend decoder is sharpest; TimeGAN keeps the GRU
discriminator (A18-GRU) and the cross-sectional vol metric (A30); SBTS retains the skewness moment (A15).

### B — Curve-shape metrics (6 diagnostic plots)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means; combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: the **function-level MAPE on the curve L itself**, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 — one division. The derivative / 2nd-derivative MAPE is **excluded** (near-zero true differences make it explode); combined std = sample std across the 5 seeds. Bold marks the lower % error.

↓ lower is better. Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, so the reference curve is fixed. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Fourier Flow | Diffusion-TS | CSDI | TimeVAE | TimeVQVAE | Perfect | Winner |
|------|---------|:----:|:-------:|:------------:|:------------:|:----:|:-------:| :---------: |:------:|:------:|
| **Log-return histogram** | MSE | 12.138 ± 0.1605 | 144.21 ± 120.61 | **2.847 ± 0.1405** | 14.505 ± 1.469 | 13.854 ± 1.500 | 2887.3 ± 311.0 | 13.198 ± 2.419 | 0 | **Fourier Flow** |
| | % err | 38.98% ± 0.132% | 33.42% ± 6.512% | **9.072% ± 0.571%** | 41.94% ± 0.996% | 35.03% ± 1.059% | 114.83% ± 0.588% | 30.69% ± 1.773% | 0 | |
| **QQ plot** | MSE | 8.90e-06 ± 6.77e-08 | 7.09e-06 ± 3.34e-06 | **4.43e-07 ± 6.56e-08** | 1.03e-05 ± 5.24e-07 | 6.94e-06 ± 4.63e-07 | 1.19e-04 ± 1.8e-06 | 5.34e-06 ± 6.51e-07 | 0 | **Fourier Flow** |
| | % err | 21.27% ± 0.364% | 34.29% ± 11.19% | **9.363% ± 2.272%** | 25.39% ± 1.704% | 23.93% ± 1.070% | 90.29% ± 1.536% | 23.54% ± 2.378% | 0 | |
| **ACF \|r\| lags 1–20** | MSE | 0.0046 ± 3.70e-05 | 0.0105 ± 0.0085 | 0.0013 ± 3.81e-05 | 5.76e-04 ± 1.26e-04 | **6.27e-05 ± 2.10e-05** | 1.00e-1 ± 4.53e-2 | 3.04e-04 ± 1.05e-04 | 0 | **CSDI** |
| | % err | 143% ± 1.580% | 164% ± 101% | 115.19% ± 1.926% | 74.76% ± 12.02% | **15.15% ± 5.425%** | 891% ± 249% | 48.69% ± 12.79% | 0 | |
| **ACF r² lags 1–20** | MSE | 0.0052 ± 5.67e-05 | 0.0058 ± 0.0033 | 9.43e-04 ± 3.51e-05 | 4.34e-04 ± 1.07e-04 | **5.59e-05 ± 1.61e-05** | 7.98e-2 ± 3.30e-2 | 2.65e-04 ± 7.78e-05 | 0 | **CSDI** |
| | % err | 160% ± 1.615% | 110% ± 60.72% | 117.36% ± 2.638% | 73.90% ± 14.29% | **16.27% ± 4.883%** | 903% ± 234% | 50.69% ± 11.90% | 0 | |
| **Rolling vol histogram** | MSE | 1227.30 ± 5.109 | 439.33 ± 216.74 | **92.44 ± 8.157** | 652.35 ± 44.95 | 463.79 ± 36.61 | 47159 ± 4926 | 332.85 ± 40.73 | 0 | **Fourier Flow** |
| | % err | 84.04% ± 0.124% | 56.06% ± 20.98% | **25.29% ± 3.210%** | 68.61% ± 1.420% | 61.28% ± 2.323% | 334.81% ± 11.76% | 53.75% ± 2.409% | 0 | |
| **Tail survival** | MSE | 0.0057 ± 6.60e-05 | 0.0117 ± 0.0092 | **5.30e-04 ± 4.58e-05** | 0.0067 ± 5.97e-04 | 0.0058 ± 5.52e-04 | 2.17e-1 ± 5.7e-3 | 0.0051 ± 8.31e-04 | 0 | **Fourier Flow** |
| | % err | 26.48% ± 0.114% | 23.60% ± 6.040% | **5.759% ± 0.237%** | 28.25% ± 0.842% | 24.63% ± 0.880% | 90.08% ± 0.638% | 22.18% ± 1.379% | 0 | |

The function-level % err stays in a sane range (≈ 5–164%): the largest values are the ACF plots, where the true ACF ≈ 0.05 sits near zero so any deviation is a big *relative* error. It no longer explodes to 10⁴-% now that the ill-posed derivative MAPE is excluded. TimeGAN's log-return-histogram MSE std (±120.61) is driven by a genuine seed-2 collapse (504.48 vs 11–170 for the other seeds).

**Fourier Flow wins B: 4/6 plots on MSE; CSDI wins the two ACF plots.** Fourier Flow's spectral objective
fits the marginal-shape diagnostics (log-return histogram, QQ, rolling-vol, tail survival) tighter than any
method, while **CSDI** wins both autocorrelation curves (ACF \|r\| and ACF r²) on both MSE and % err by a
full order of magnitude (6.27e-05 vs Diffusion-TS's 5.76e-04 on ACF \|r\|) — its score-based diffusion
reproduces Heston's weak vol-clustering autocorrelation that the spectral flow over-smooths. **TimeVAE
loses all six B plots** — its posterior-mean decoder collapses the marginal shape (log-return histogram MSE
2887 vs FF 2.8), consistent with its heavily under-dispersed samples. **TimeVQVAE** wins no B plot but is
consistently mid-pack — third-best log-return histogram (MSE 13.2, behind Fourier Flow 2.8 and SBTS 12.1)
and better than Diffusion-TS on both ACF curves — a full 200× tighter marginal shape than TimeVAE.
Each value is computed over the same **5 seeds** per method.

Detailed per-seed results and plots:
→ [`results/Heston/SBTS/`](results/Heston/SBTS/) — SBTS metrics, diagnostics, PS-MC
→ [`results/Heston/TimeGAN/`](results/Heston/TimeGAN/) — TimeGAN metrics, diagnostics, PS-MC
→ [`results/Heston/FourierFlow/`](results/Heston/FourierFlow/) — Fourier Flow metrics, diagnostics, PS-MC
→ [`results/Heston/DiffusionTS/`](results/Heston/DiffusionTS/) — Diffusion-TS metrics, diagnostics, PS-MC
→ [`results/Heston/CSDI/`](results/Heston/CSDI/) — CSDI metrics, diagnostics, PS-MC
→ [`results/Heston/TimeVAE/`](results/Heston/TimeVAE/) — TimeVAE metrics, diagnostics, PS-MC
→ [`results/Heston/TimeVQVAE/`](results/Heston/TimeVQVAE/) — TimeVQVAE metrics, diagnostics, PS-MC

---

## Datasets

| Dataset | Paths | Seq len | Description |
|---------|-------|---------|-------------|
| [Heston](dataset/Heston/) | 8 192 | 128 | Heston stochastic volatility model, daily prices (~6 months) |

→ [`dataset/Heston/README.md`](dataset/Heston/README.md) — parameters, SDE formula, reproduce instructions.

---

## Methods

| Method | Full name | Paper | Authors | Year | Venue | Original code |
|--------|-----------|-------|---------|------|-------|---------------|
| [TimeGAN](methods/TimeGAN/) | Time-series GAN | [arXiv:2010.00782](https://arxiv.org/abs/2010.00782) | Yoon, Jarrett, van der Schaar | 2019 | NeurIPS | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) |
| [SBTS](methods/SBTS/) | Schrödinger Bridge Time Series | [arXiv:2503.02943](https://arxiv.org/abs/2503.02943) | Alouadi, Barreau, Carlier, Pham | 2025 | ICAIF | [alexouadi/SBTS](https://github.com/alexouadi/SBTS) |
| [FourierFlow](methods/FourierFlow/) | Fourier Flows | [OpenReview](https://openreview.net/forum?id=PpshD0AXfA) | Alaa, Chan, van der Schaar | 2021 | ICLR | [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) |
| [DiffusionTS](methods/DiffusionTS/) | Diffusion-TS | [arXiv:2403.01742](https://arxiv.org/abs/2403.01742) | Yuan, Qiao | 2024 | ICLR | [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) |
| [CSDI](methods/CSDI/) | Conditional Score-based Diffusion (Imputation) | [arXiv:2107.03502](https://arxiv.org/abs/2107.03502) | Tashiro, Song, Song, Ermon | 2021 | NeurIPS | [ermongroup/CSDI](https://github.com/ermongroup/CSDI) |
| [TimeVAE](methods/TimeVAE/) | Variational Auto-Encoder for Time Series | [arXiv:2111.08095](https://arxiv.org/abs/2111.08095) | Desai, Freeman, Beaver, Wang | 2021 | arXiv | [abudesai/timeVAE](https://github.com/abudesai/timeVAE) |
| [TimeVQVAE](methods/TimeVQVAE/) | Vector Quantized Time Series Generation | [arXiv:2303.04743](https://arxiv.org/abs/2303.04743) | Lee, Malacarne, Aune | 2023 | AISTATS | [ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) |

---

## Metrics (A1–A34 + B)

### A1–A34 — Core Metrics by category

| ID | Name | Category | Dir | Formula reference |
|----|------|----------|-----|------------------|
| A1 | Kurtosis Error | Fat Tail | ↓ | \|κ_real − κ_gen\| on log-returns |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | \|q_0.95(\|r_real\|) − q_0.95(\|r_gen\|)\| |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | \|q_0.99(\|r_real\|) − q_0.99(\|r_gen\|)\| |
| A4 | Tail QQ Error | Fat Tail | ↓ | QQ RMSE restricted to top-5% tail quantiles |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | \|Hill_real − Hill_gen\|; Hill (1975), top 5% threshold |
| A6 | Path MMD² | Distribution | ↓ | RBF kernel on full paths; Gretton et al. (2012) |
| A7 | Terminal MMD² | Distribution | ↓ | RBF kernel on terminal prices S_T |
| A8 | Increment MMD² | Distribution | ↓ | RBF kernel on log-return increments |
| A9 | Volatility MMD | Distribution | ↓ | RBF kernel on rolling 5-step realized vol |
| A10 | Terminal SWD | Distribution | ↓ | Sliced Wasserstein on S_T; Rabin et al. (2012) |
| A11 | Path SWD | Distribution | ↓ | Sliced Wasserstein on full paths |
| A12 | RV Law Loss | Distribution | ↓ | W₁(RV_real, RV_gen); RV_i=Σ_t r²_{i,t}/dt; Barndorff-Nielsen & Shephard (2002) |
| A13 | Mean Path RMSE | Distribution | ↓ | RMSE between real/gen mean trajectories |
| A14 | KS Log-returns | Distribution | ↓ | KS statistic on pooled log-returns; Massey (1951) |
| A15 | Skewness Error | Distribution | ↓ | \|skew_real − skew_gen\| on log-returns; Cont (2001) |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | QQ RMSE over 300 uniform quantile levels |
| A17 | Terminal Price KS | Distribution | ↓ | KS statistic on terminal prices S_T |
| A18 | Disc Score GRU / MLP | Adversarial | ↓ | \|accuracy − 0.5\| on log-returns; Esteban et al. (2017) |
| A19 | Pred Score GRU / MLP | Predictive | ↓ | TSTR MAE on log-returns; Esteban et al. (2017) |
| A20 | Covariance Error | Temporal | ↓ | ‖Σ_real − Σ_gen‖_F / ‖Σ_real‖_F × 100% |
| A21 | ACF \|r\| Error (lags) | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on \|r\| |
| A22 | ACF r² Error (lags) | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on r² |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on \|r\|; Heston ≈ +0.052 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on r²; Heston ≈ +0.050 |
| A25 | Mean RMSE | Vol | ↓ | RMSE of per-step mean price E[S_t] |
| A26 | Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on price increments ΔS_t |
| A27 | Log-Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on log-returns |
| A28 | Kurtosis Ratio (→ 1) | Vol | — | κ_real / κ_gen; perfect = 1.0 |
| A29 | Sigma Mean Error | Vol | ↓ | \|mean(σ_real) − mean(σ_gen)\| annualized per-path vol |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | RMSE of cross-sectional vol trajectory |
| A31 | Rolling Vol KS (w=5) | Vol | ↓ | KS on rolling-5 vol histograms; Mandelbrot (1963) |
| A32 | Vol-of-Vol Error | Vol | ↓ | \|std(rolling-vol_real) − std(rolling-vol_gen)\| |
| A33 | Teacher-Sigma Corr | Heston Spec | ↑ | Pearson ρ of QV-estimated vol vs true teacher v_t |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | RMSE of QV-estimated vol vs true teacher v_t |

### B — Curve-shape metrics (6 diagnostic plots)

For each of 6 diagnostic plots we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — and score each list under **two measures**: MSE (absolute squared error) and % err (relative error). For MSE the three sub-scores are summed (std in quadrature); the **% err reports the function-level MAPE of the curve L only** (the derivative MAPE is ill-posed near zero). Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, making the reference curve fixed across seeds.

| Plot | Key | What the curve represents |
|------|-----|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) |
| QQ plot | `B_qq_plot_*` | Quantile function at 100 uniform levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*` | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20) | `B_acf_sq_r_*` | Mean per-path ACF of r² at each lag |
| Rolling vol hist. | `B_roll_vol_hist_*` | Density of rolling-5 vol |
| Tail survival | `B_tail_surv_*` | P(\|r\|>x) at thresholds of real \|r\| |

Full formulas and per-seed results:
→ [`results/Heston/SBTS/README.md`](results/Heston/SBTS/README.md)
→ [`results/Heston/TimeGAN/README.md`](results/Heston/TimeGAN/README.md)
→ [`results/Heston/FourierFlow/README.md`](results/Heston/FourierFlow/README.md)
→ [`results/Heston/DiffusionTS/README.md`](results/Heston/DiffusionTS/README.md)
→ [`results/Heston/CSDI/README.md`](results/Heston/CSDI/README.md)
→ [`results/Heston/TimeVAE/README.md`](results/Heston/TimeVAE/README.md)

---

## Reproducing

```bash
# 1. Generate target dataset
cd dataset/Heston && python generate_heston.py

# 2a. Train TimeGAN (5 seeds, 2 A100 GPUs, ~45 min)
cd methods/TimeGAN/code && python train.py --gpu0 0 --gpu1 3

# 2b. Generate SBTS paths (5 seeds, CPU, 64 workers, ~30 min)
source /path/to/sbts-venv/bin/activate
cd methods/SBTS/code && SBTS_NWORK=64 python run_all.py

# 2c. Train Fourier Flow (5 seeds, CPU-only numpy.fft, grad-clip=1.0)
cd methods/FourierFlow/code && ./train_all.sh

# 2d. Train Diffusion-TS (5 seeds, 2 A100 GPUs, mujoco arch, ~15 min/seed)
cd methods/DiffusionTS/code
for s in 0 1 2 3 4; do g=$((s%2+1)); c=$((g*8)); \
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
  PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py --arch mujoco --seed $s & done; wait

# 2e. Train CSDI (5 seeds, 2 A100 GPUs, unconditional DDPM, ~30 min/seed)
cd methods/CSDI/code
for s in 0 1 2 3 4; do gpu=$([ $((s % 2)) -eq 0 ] && echo 0 || echo 3); c=$(( (s%2)*8 )); \
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
  /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s & done; wait

# 2f. Train TimeVAE (5 seeds, 2 A100 GPUs, conv VAE + EarlyStopping, ~13 min/seed)
cd methods/TimeVAE/code
for s in 0 1 2 3 4; do gpu=$([ $((s % 2)) -eq 0 ] && echo 0 || echo 3); c=$(( (s%2)*8 )); \
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
  /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s & done; wait

# 3. Compute all metrics (GPU for A13/A14)
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method TimeGAN     --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method SBTS        --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method FourierFlow --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method DiffusionTS --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method CSDI        --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method TimeVAE     --dataset Heston

# 4. Compute perfect-recovery floor
CUDA_VISIBLE_DEVICES=0 python metrics/perfect_recovery.py --dataset Heston
```

See [`GUIDELINE.md`](GUIDELINE.md) for the full reproducibility protocol.

---

## Adding a new method

1. Create `methods/<NewMethod>/` with subfolders `generated_paths/`, `weights/`, `losses/`, `code/`
2. Implement generation code — save paths as `generated_paths/seed_{i}/generated_paths_NxT.npy` (price space, S₀≈100)
3. Run `python metrics/compute_all.py --method NewMethod --dataset Heston`
4. Results appear in `results/Heston/NewMethod/` with full A1–A34 + B tables
</content>
