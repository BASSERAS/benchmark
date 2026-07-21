# Results — Methods Comparison on Heston

All methods are evaluated on the same dataset:
**8 192 Heston price paths, seq\_len = 128**
(μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250)

---

## Cross-Method Comparison A1–A34 — Heston (mean ± std, 5 seeds)

↓ = lower is better. ↑ = higher is better. (→ 1) = ratio, target = 1.0. **Bold** = best method per metric.
Perfect-recovery floors (the empirical best-case score a perfect generator would achieve with finite
samples) are method-independent and documented once, reproducibly, in
[`methods/perfect_recovery/README.md`](../methods/perfect_recovery/README.md).

| Metric | SBTS | TimeGAN | Fourier Flow | Diffusion-TS | CSDI | TimeVAE | TimeVQVAE | COSCI-GAN | Winner |
|--------|:----:|:-------:|:------------:|:------------:|:----:|:-------:| :------: |:--------:|:------:|
| **— Fat Tail —** | | | | | | |  | | |
| A1 Kurtosis Error ↓ | 0.1187 ± 0.0060 | 2.955 ± 2.099 | 0.5757 ± 0.0083 | 0.4238 ± 0.0230 | **0.0958 ± 0.0262** | 2.258 ± 0.5719 | 0.1367 ± 0.0924 | 0.5612 ± 0.1128 | **CSDI** |
| A2 \|r\| q95 Error ↓ | 0.0063 ± 3.00e-05 | 0.0032 ± 0.0018 | **6.52e-04 ± 2.10e-04** | 0.0068 ± 1.57e-04 | 0.0053 ± 1.50e-04 | 0.0222 ± 1.22e-04 | 0.0044 ± 2.54e-04 | 0.0972 ± 0.0035 | **Fourier Flow** |
| A3 \|r\| q99 Error ↓ | 0.0098 ± 4.80e-05 | 0.0043 ± 0.0028 | **0.0023 ± 5.06e-04** | 0.0103 ± 1.75e-04 | 0.0073 ± 2.29e-04 | 0.0308 ± 1.05e-04 | 0.0060 ± 3.03e-04 | 0.1240 ± 0.0060 | **Fourier Flow** |
| A4 Tail QQ Error ↓ | 0.0062 ± 2.60e-05 | 0.0034 ± 0.0015 | **7.15e-04 ± 1.23e-04** | 0.0067 ± 1.50e-04 | 0.0052 ± 1.50e-04 | 0.0219 ± 1.17e-04 | 0.0044 ± 2.48e-04 | 0.0957 ± 0.0035 | **Fourier Flow** |
| A5 Hill Tail Index Error ↓ | 9.499 ± 0.3457 | 36.885 ± 17.053 | 6.368 ± 2.000 | 3.613 ± 0.2789 | 1.992 ± 0.5856 | 2.396 ± 0.6794 | 4.342 ± 1.193 | **1.563 ± 1.206** | **COSCI-GAN** |
| **— Distribution —** | | | | | | |  | | |
| A6 Path MMD² ↓ | 0.0110 ± 0.0026 | 0.0165 ± 0.0127 | 0.0052 ± 0.0019 | 0.0033 ± 6.56e-04 | **0.0027 ± 6.16e-04** | 0.0184 ± 9.55e-04 | 0.0039 ± 7.71e-04 | 0.0467 ± 0.0038 | **CSDI** |
| A7 Terminal MMD² ↓ | 0.0100 ± 0.0036 | 0.0267 ± 0.0192 | 0.0106 ± 0.0051 | **0.0021 ± 3.92e-04** | 0.0028 ± 0.0011 | 0.0042 ± 0.0011 | 0.0046 ± 9.75e-04 | 0.0138 ± 0.0137 | **Diffusion-TS** |
| A8 Increment MMD² ↓ | 0.0071 ± 2.47e-04 | 0.0077 ± 0.0041 | **0.0011 ± 7.70e-05** | 0.0112 ± 9.37e-04 | 0.0079 ± 8.54e-04 | 0.2134 ± 0.0012 | 0.0071 ± 9.95e-04 | 0.4784 ± 0.0108 | **Fourier Flow** |
| A9 Volatility MMD ↓ | 0.3038 ± 0.0071 | 0.3789 ± 0.2430 | **0.0596 ± 0.0086** | 0.3840 ± 0.0314 | 0.2448 ± 0.0206 | 3.591 ± 0.4563 | 0.1966 ± 0.0273 | 3.960 ± 0.043 | **Fourier Flow** |
| A10 Terminal SWD ↓ | 3.539 ± 0.7368 | 2.658 ± 0.8567 | 2.815 ± 0.9433 | 1.358 ± 0.2152 | **1.303 ± 0.2465** | 1.798 ± 0.2603 | 1.504 ± 0.4262 | 4.550 ± 3.115 | **CSDI** |
| A11 Path SWD ↓ | 2.415 ± 0.4104 | 1.417 ± 0.4914 | 1.289 ± 0.4198 | 0.9838 ± 0.1107 | **0.7712 ± 0.1581** | 0.9882 ± 0.2052 | 0.9413 ± 0.2060 | 3.486 ± 0.187 | **CSDI** |
| A12 RV Law Loss ↓ | 2.148 ± 0.0074 | 1.551 ± 0.3788 | **0.5291 ± 0.1299** | 2.250 ± 0.0491 | 1.897 ± 0.0563 | 4.986 ± 0.0084 | 1.682 ± 0.0893 | 118.8 ± 7.9 | **Fourier Flow** |
| A13 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | 0.5289 ± 0.2624 | 0.4910 ± 0.4022 | 0.3615 ± 0.2364 | 0.3101 ± 0.3036 | **0.2981 ± 0.2172** | 0.6974 ± 0.1790 | 4.007 ± 0.194 | **TimeVAE** |
| A14 KS Log-returns ↓ | 0.0534 ± 3.62e-04 | 0.0848 ± 0.0374 | **0.0191 ± 0.0024** | 0.0600 ± 0.0019 | 0.0539 ± 0.0021 | 0.3673 ± 0.0046 | 0.0501 ± 0.0036 | 0.3208 ± 0.0073 | **Fourier Flow** |
| A15 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | 0.0282 ± 0.0152 | 0.0698 ± 0.0358 | 0.0457 ± 0.0021 | 0.5568 ± 0.0984 | 0.0397 ± 0.0082 | 0.0445 ± 0.0386 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 0.0028 ± 1.20e-05 | 0.0025 ± 6.43e-04 | **5.86e-04 ± 4.20e-05** | 0.0030 ± 8.30e-05 | 0.0026 ± 8.60e-05 | 0.0105 ± 8.40e-05 | 0.0022 ± 1.38e-04 | 0.0486 ± 0.0020 | **Fourier Flow** |
| A17 Terminal Price KS ↓ | 0.0921 ± 0.0051 | 0.1121 ± 0.0556 | 0.0848 ± 0.0166 | 0.0400 ± 0.0073 | **0.0321 ± 0.0053** | 0.0478 ± 0.0099 | 0.0531 ± 0.0067 | 0.1481 ± 0.0985 | **CSDI** |
| **— Adversarial —** | | | | | | |  | | |
| A18 Disc Score GRU ↓ | 0.2755 ± 0.2166 | **0.0077 ± 0.0050** | 0.0094 ± 0.0097 | 0.2621 ± 0.1578 | 0.0470 ± 0.0901 | 0.2035 ± 0.1934 | 0.0237 ± 0.0198 | 0.4998 ± 0.0002 | **TimeGAN** |
| A18 Disc Score MLP ↓ | 0.0079 ± 0.0049 | 0.1031 ± 0.0395 | 0.0053 ± 0.0041 | 0.0554 ± 0.0396 | **0.0046 ± 0.0058** | 0.1756 ± 0.1354 | 0.0074 ± 0.0033 | 0.5000 ± 0.0000 | **CSDI** |
| **— Predictive —** | | | | | | |  | | |
| A19 Pred Score GRU ↓ | 0.0586 ± 5.90e-05 | 0.0574 ± 0.0019 | **0.0537 ± 7.60e-06** | 0.0549 ± 1.59e-04 | 0.0539 ± 3.20e-05 | 0.0577 ± 8.53e-04 | 0.0539 ± 4.70e-05 | 0.1308 ± 0.0177 | **Fourier Flow** |
| A19 Pred Score MLP ↓ | 0.0583 ± 2.55e-04 | 0.0570 ± 0.0012 | **0.0540 ± 4.90e-04** | 0.0551 ± 3.72e-04 | 0.0541 ± 2.47e-04 | 0.0564 ± 2.75e-04 | 0.0540 ± 3.45e-04 | 0.1089 ± 0.0071 | **Fourier Flow** |
| **— Temporal —** | | | | | | |  | | |
| A20 Covariance Error ↓ | 145.35 ± 4.886 | 17.751 ± 6.707 | 64.406 ± 38.255 | 38.172 ± 10.637 | 35.538 ± 5.776 | 51.272 ± 1.758 | **16.599 ± 14.720** | 28.90 ± 25.36 | **TimeVQVAE** |
| A21 ACF \|r\| Error (lags) ↓ | 0.0596 ± 4.70e-04 | 0.1252 ± 0.0674 | 0.0435 ± 5.50e-04 | 0.0201 ± 0.0030 | **0.0091 ± 0.0026** | 0.3865 ± 0.1057 | 0.0172 ± 0.0042 | 0.0806 ± 0.0206 | **CSDI** |
| A22 ACF r² Error (lags) ↓ | 0.0619 ± 5.08e-04 | 0.0839 ± 0.0348 | 0.0379 ± 5.56e-04 | 0.0168 ± 0.0027 | **0.0086 ± 0.0021** | 0.3580 ± 0.0885 | 0.0152 ± 0.0033 | 0.0902 ± 0.0214 | **CSDI** |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1437 ± 0.0012 | 0.2264 ± 0.1034 | 0.0526 ± 7.04e-04 | **0.0039 ± 0.0022** | 0.0188 ± 0.0048 | 0.4637 ± 0.1346 | 0.0115 ± 0.0080 | 0.1663 ± 0.0493 | **Diffusion-TS** |
| A24 ACF r² Lag-1 Error ↓ | 0.1665 ± 0.0017 | 0.1719 ± 0.0626 | 0.0461 ± 7.01e-04 | **0.0038 ± 0.0026** | 0.0176 ± 0.0036 | 0.4589 ± 0.1189 | 0.0091 ± 0.0073 | 0.1916 ± 0.0511 | **Diffusion-TS** |
| **— Vol —** | | | | | | |  | | |
| A25 Mean RMSE ↓ | 1.301 ± 0.2776 | 0.7385 ± 0.4552 | 0.9000 ± 0.8807 | 0.5767 ± 0.4444 | 0.3729 ± 0.4145 | **0.3396 ± 0.2710** | 0.9147 ± 0.1675 | 4.500 ± 3.382 | **TimeVAE** |
| A26 Return Std Error ↓ | 0.2492 ± 0.0018 | 0.1519 ± 0.0888 | **0.0058 ± 0.0028** | 0.3098 ± 0.0093 | 0.2570 ± 0.0098 | 1.073 ± 0.0078 | 0.2306 ± 0.0142 | 5.033 ± 0.223 | **Fourier Flow** |
| A27 Log-Return Std Error ↓ | 0.0030 ± 1.20e-05 | 0.0017 ± 7.78e-04 | **6.70e-05 ± 6.60e-05** | 0.0032 ± 8.20e-05 | 0.0026 ± 8.90e-05 | 0.0109 ± 7.80e-05 | 0.0023 ± 1.37e-04 | 0.0498 ± 0.0020 | **Fourier Flow** |
| A28 Kurtosis Ratio (→ 1) | 1.989 ± 0.0182 | −1.095 ± 3.525 | 3.039 ± 0.7605 | 1.866 ± 0.2509 | **0.8539 ± 0.0298** | 0.2780 ± 0.0467 | 0.8249 ± 0.0682 | −7.994 ± 11.881 | **CSDI** |
| A29 Sigma Mean Error ↓ | 0.0440 ± 1.84e-04 | 0.0307 ± 0.0089 | **0.0026 ± 8.77e-04** | 0.0485 ± 0.0013 | 0.0404 ± 0.0015 | 0.1741 ± 0.0018 | 0.0371 ± 0.0021 | 0.7875 ± 0.0309 | **Fourier Flow** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 3.276 ± 0.0637 | **0.3534 ± 0.1253** | 1.367 ± 0.4499 | 1.154 ± 0.2019 | 0.9262 ± 0.1315 | 1.122 ± 0.0447 | 0.3781 ± 0.3311 | 1.065 ± 0.261 | **TimeGAN** |
| A31 Rolling Vol KS (w=5) ↓ | 0.3435 ± 6.43e-04 | 0.2540 ± 0.1093 | **0.0740 ± 0.0014** | 0.2558 ± 0.0078 | 0.2180 ± 0.0082 | 0.9871 ± 0.0045 | 0.1828 ± 0.0102 | 0.9373 ± 0.0076 | **Fourier Flow** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 6.00e-06 | 8.97e-04 ± 8.69e-04 | 6.88e-04 ± 9.20e-05 | 0.0016 ± 3.80e-05 | 0.0010 ± 2.10e-05 | 0.0046 ± 5.60e-05 | **6.75e-04 ± 5.80e-05** | 0.0181 ± 0.0011 | **TimeVQVAE** |
| **— Heston Spec —** | | | | | | |  | | |
| A33 Teacher-Sigma Corr ↑ | 0.0046 ± 0.0019 | 0.0021 ± 0.0090 | 7.85e-04 ± 0.0038 | −0.0036 ± 0.0032 | 0.0084 ± 0.0040 | **0.0273 ± 0.0050** | 0.0037 ± 0.0036 | −0.0043 ± 0.0131 | **TimeVAE** |
| A34 Teacher-Sigma RMSE ↓ | 0.0955 ± 9.10e-05 | 0.1183 ± 0.0184 | **0.0894 ± 0.0013** | 0.0960 ± 7.41e-04 | 0.0985 ± 6.61e-04 | 0.1793 ± 0.0016 | 0.1008 ± 0.0010 | 0.8095 ± 0.0288 | **Fourier Flow** |
| PS-MC CRPS H=32 ↓ | 2.761 ± 0.004 | 3.087 ± 0.340 | 2.742 ± 0.027 | 2.717 ± 0.003 | **2.713 ± 0.005** | 3.855 ± 0.070 | 2.771 ± 0.015 | 4.657 ± 0.775 | **CSDI** |
| PS-MC CRPS H=64 ↓ | 3.900 ± 0.008 | 4.372 ± 0.431 | 3.992 ± 0.106 | 3.845 ± 0.005 | **3.814 ± 0.007** | 5.634 ± 0.124 | 3.889 ± 0.017 | 5.834 ± 0.763 | **CSDI** |
| Training (8 192×128) | **— (no training)** | ~6.5 min / A100 | ~8.2 min / CPU | ~14.6 min / A100 | ~29.3 min / A100 | ~13 min / A100 | ~53 min / A100 | ~4.3 min / A100 | **SBTS** |
| Generation (8 192×128) | ~6.3 min / 64 CPUs | **<1 s / A100** | ~1.5 s / CPU | — (500-step DDPM) | ~10.2 s / A100 | <1 s / A100 | ~6 s / A100 | — (LSTM fwd, not sep. timed) | **TimeGAN** |

> **A33 Teacher-Sigma Corr**: floor = 0.614 (not 1.0) — 5-step rolling QV is a noisy estimator of
> instantaneous variance vₜ. TimeVAE (0.027) now has the highest correlation, then CSDI (0.008), SBTS
> (0.005), TimeVQVAE (0.004), TimeGAN (0.002), Fourier Flow (7.9e-04), Diffusion-TS (−0.004) and
> COSCI-GAN (−0.004) — the last two slightly negative. None of them meaningfully preserves stochastic
> volatility relative to the 0.614 floor — TimeVAE simply wins a race among near-zero correlations.
>
> **A28 Kurtosis Ratio**: target = 1.0. CSDI (0.854) is still closest, with TimeVQVAE (0.825) a close second:
> |CSDI−1| = 0.146 < |TimeVQVAE−1| = 0.175 < |TimeVAE−1| = 0.722 < |DTS−1| = 0.866 < |SBTS−1| = 0.989 < |FF−1| = 2.039 < |TimeGAN−1| = 2.095 < |COSCI-GAN−1| = 8.994.
> TimeVAE (0.278) is heavily under-dispersed (kurtosis far below real) but closer to 1 than the flow/GAN.
> COSCI-GAN's mean ratio is **negative** (−7.994, sign-flipping across seeds −22.8 … +6.2), the farthest
> from 1 — its generated tails are near-Gaussian-to-thin against Heston's mildly fat tails.

**Fourier Flow wins 15/38, CSDI wins 11/38, Diffusion-TS wins 3/38, TimeVAE wins 3/38, TimeGAN wins 2/38, TimeVQVAE wins 2/38, SBTS wins 1/38, COSCI-GAN wins 1/38** (on A1-A34 + PS-MC, excluding training/generation rows). COSCI-GAN's single win is **A5 Hill tail-index error (1.563)** — the lowest of any method, taking that metric from CSDI (1.992).

**Interpretation:**
- **CSDI dominates distribution, autocorrelation and forecasting** (A1, A6, A10, A11, A17, A18-MLP, A21, A22, A28, PS-MC H=32/H=64): the score-based diffusion pool gives the closest kurtosis moment (A1 kurtosis error 0.096), the tightest multi-sample distribution (Path MMD A6, Terminal/Path SWD A10/A11), the closest kurtosis ratio (A28 0.854 → |·−1| = 0.146), the strongest aggregate ACF fit (A21/A22 an order of magnitude below the baselines), the least MLP-separable samples (A18-MLP 0.005), and both Path-Shadowing horizons (CRPS 2.713/3.814 — lowest of all methods). It **cedes A5 Hill tail-index** to COSCI-GAN this round (1.992 vs 1.563).
- **TimeVAE narrowly wins the two mean-trajectory metrics and the teacher-sigma correlation** (A13, A25, A33): its posterior-mean reconstructions produce the closest average price path (A13 0.298 < CSDI 0.310) and average vol path (A25 0.340 < CSDI 0.373), and the highest — though still near-zero — teacher-sigma correlation (A33 0.027 vs CSDI 0.008; floor 0.614). These are averaging wins: a VAE decoder regresses to the conditional mean, so its *mean* trajectory is accurate even though its *marginal* fit is the worst of all six methods (A1 kurtosis error 2.26, A9 vol MMD 3.59, A31 rolling-vol KS 0.987, A28 kurtosis ratio 0.278 — heavily under-dispersed). It wins **0** of the six B curve-shape plots and its Path-Shadowing CRPS (3.855/5.634) does not beat the random-walk baseline.
- **Fourier Flow keeps marginal and spectral fidelity** (A2–A4, A8, A9, A12, A14, A16, A19, A26, A27, A29, A31, A34): the explicit-likelihood flow trained in the frequency domain fits log-return tails (A2/A3/A4), increment MMD (A8/A9), and vol moments (A26/A27/A29) tighter than any method — often by an order of magnitude (A27: 6.7e-05 vs 2.6e-03; A26: 0.006 vs 0.26). It narrowly cedes the vol-of-vol metric (A32) to TimeVQVAE (6.75e-04 vs 6.88e-04).
- **Diffusion-TS keeps the lag-1 autocorrelation and terminal MMD** (A7, A23, A24): its seasonal-trend decoder nails the single-lag ACF (A23 0.004, A24 0.004) and terminal MMD (A7 0.0021) tighter than CSDI, though CSDI wins the aggregate-lag ACF (A21/A22).
- **TimeGAN keeps the GRU discriminator and cross-sectional vol** (A18-GRU, A30 cross-sectional vol): the recurrent generator is hardest for the GRU probe (A18-GRU 0.008) and captures cross-sectional vol dispersion (A30 0.353) better than the others. It cedes the price-level covariance metric (A20) to TimeVQVAE this round (16.6 vs 17.8).
- **TimeVQVAE wins the two structural-error metrics** (A20 covariance, A32 vol-of-vol): its tokenised time-frequency prior produces the tightest multi-step price-level covariance (A20: 16.6 vs TimeGAN 17.8 vs CSDI 36 vs DTS 38 vs FF 64 vs SBTS 145) and the closest vol-of-vol (A32: 6.75e-04 vs FF 6.88e-04), both by a hair. Elsewhere it is a solid mid-pack generator — third-best on several fat-tail and MMD metrics (A2 0.004, A6 0.004, A9 0.20) — and its Path-Shadowing CRPS (2.771/3.889) clears the random-walk baseline at both horizons, second only to CSDI/Diffusion-TS.
- **SBTS keeps the skew moment** (A15): its kernel construction preserves the sign/magnitude of skewness (0.023) best.
- **COSCI-GAN wins the Hill tail-index only** (A5): its adversarial channel-GAN reproduces the raw tail decay slope (Hill 1.563, best of all methods) and has good *scalar* low-order moments for the VAE/GAN family (A1 kurtosis error 0.561, A15 skewness 0.044), but those scalars do **not** carry to the full-density curves — it ranks 6th–8th of 8 on every B plot (dead-last on QQ; §B below). It is the **only** method whose A18 discriminative score saturates at the maximum (0.500 both GRU and MLP — paths near-perfectly separable), its kurtosis ratio is negative (A28 −7.994, thin/near-Gaussian tails), and its Path-Shadowing CRPS (4.657/5.834) **fails to beat the random-walk baseline** at both horizons. See [`Heston/COSCI-GAN/README.md`](Heston/COSCI-GAN/README.md).
- **A18 discriminative**: CSDI is least separable by the MLP probe (0.005), Fourier Flow next (0.005); TimeGAN is hardest for the GRU probe (0.008). CSDI's GRU score has a large ±0.09 spread (seed-4 floors at 0.23, the other four near 0).
- **Path Shadowing MC**: CSDI now wins both horizons (H=32 CRPS 2.713 < DTS 2.717 < FF 2.742 < SBTS 2.761 < TimeVQVAE 2.771; H=64 3.814 < DTS 3.845 < TimeVQVAE 3.889 < SBTS 3.900 < FF 3.992) — its diffusion-generated pool provides the tightest, best-calibrated nearest-neighbour futures on Heston. TimeVQVAE clears the random-walk baseline (RW 3.732/5.301) at both horizons.
- **Cross-seed stability**: CSDI is very stable on moment/ACF metrics (A19-GRU std 3.2e-05, A34 std 6.6e-04, A28 std 0.030) but shows real spread on the GRU discriminator (A18-GRU std 0.09) and mean-path metrics (A25 std 0.41, A13 std 0.30 — one seed at 0.91/1.19); SBTS remains the most deterministic overall (A1 std 0.006).

---

## B — Curve-Shape Metrics Cross-Method Comparison — Heston (mean ± std, 5 seeds)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we
build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = **sum** of the three
  seed-means; combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (**quadrature**).
- **% err row**: the **function-level MAPE on the curve L itself**, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 — one division. The
  derivative / 2nd-derivative MAPE is **excluded** (near-zero true differences make it explode into meaningless 10⁴-% values); combined std = **sample std across the 5 seeds**. Bold marks the lower % error.

↓ lower is better for both rows. Histogram bins use real-data [0.5th, 99.5th]-percentile edges.
**Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Fourier Flow | Diffusion-TS | CSDI | TimeVAE | TimeVQVAE | COSCI-GAN | Perfect | Winner |
|------|---------|:----:|:-------:|:------------:|:------------:|:----:|:-------:| :------: |:--------:|:------:|:------:|
| **Log-return histogram** | MSE   | 12.14 ± 0.16 | 144.2 ± 120.6 | **2.847 ± 0.141** | 14.505 ± 1.469 | 13.854 ± 1.500 | 2887.3 ± 311.0 | 13.20 ± 2.42 | 128.6 ± 5.9 | 0 | **Fourier Flow** |
|                          | % err | 38.98% ± 0.132% | 33.42% ± 6.512% | **9.072% ± 0.571%** | 41.94% ± 0.996% | 35.03% ± 1.059% | 114.83% ± 0.588% | 30.69% ± 1.773% | 248.65% ± 7.948% | 0 | |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | 7.09e-6 ± 3.3e-6 | **4.43e-7 ± 6.6e-8** | 1.03e-5 ± 5.2e-7 | 6.94e-6 ± 4.6e-7 | 1.19e-4 ± 1.8e-6 | 5.34e-6 ± 6.5e-7 | 2.48e-3 ± 2.0e-4 | 0 | **Fourier Flow** |
|                          | % err | 21.27% ± 0.364% | 34.29% ± 11.19% | **9.363% ± 2.272%** | 25.39% ± 1.704% | 23.93% ± 1.070% | 90.29% ± 1.536% | 23.54% ± 2.378% | 437.16% ± 19.28% | 0 | |
| **ACF \|r\| lags 1–20**  | MSE   | 4.57e-3 ± 3.7e-5 | 1.05e-2 ± 8.5e-3 | 1.30e-3 ± 3.8e-5 | 5.76e-4 ± 1.26e-4 | **6.27e-5 ± 2.10e-5** | 1.00e-1 ± 4.53e-2 | 3.04e-4 ± 1.05e-4 | 2.56e-2 ± 6.9e-3 | 0 | **CSDI** |
|                          | % err | 143% ± 1.580% | 164% ± 101% | 115.19% ± 1.926% | 74.76% ± 12.02% | **15.15% ± 5.425%** | 891% ± 249% | 48.69% ± 12.79% | 211.80% ± 41.11% | 0 | |
| **ACF r² lags 1–20**     | MSE   | 5.17e-3 ± 5.7e-5 | 5.77e-3 ± 3.3e-3 | 9.43e-4 ± 3.5e-5 | 4.34e-4 ± 1.07e-4 | **5.59e-5 ± 1.61e-5** | 7.98e-2 ± 3.30e-2 | 2.65e-4 ± 7.78e-5 | 2.63e-2 ± 7.0e-3 | 0 | **CSDI** |
|                          | % err | 160% ± 1.615% | 110% ± 60.72% | 117.36% ± 2.638% | 73.90% ± 14.29% | **16.27% ± 4.883%** | 903% ± 234% | 50.69% ± 11.90% | 251.70% ± 48.25% | 0 | |
| **Rolling vol histogram**| MSE   | 1227.3 ± 5.1 | 439.3 ± 216.7 | **92.44 ± 8.16** | 652.35 ± 44.95 | 463.79 ± 36.61 | 47159 ± 4926 | 332.8 ± 40.7 | 4202.5 ± 102.3 | 0 | **Fourier Flow** |
|                          | % err | 84.04% ± 0.124% | 56.06% ± 20.98% | **25.29% ± 3.210%** | 68.61% ± 1.420% | 61.28% ± 2.323% | 334.81% ± 11.76% | 53.75% ± 2.409% | 802.75% ± 14.02% | 0 | |
| **Tail survival**        | MSE   | 5.74e-3 ± 6.6e-5 | 1.17e-2 ± 9.2e-3 | **5.30e-4 ± 4.6e-5** | 6.70e-3 ± 5.97e-4 | 5.817e-3 ± 5.5e-4 | 2.17e-1 ± 5.7e-3 | 5.07e-3 ± 8.3e-4 | 1.79e-1 ± 6.0e-3 | 0 | **Fourier Flow** |
|                          | % err | 26.48% ± 0.114% | 23.60% ± 6.040% | **5.759% ± 0.237%** | 28.25% ± 0.842% | 24.63% ± 0.880% | 90.08% ± 0.638% | 22.18% ± 1.379% | 342.77% ± 8.325% | 0 | |

> **Reading the two rows**: the **MSE** row is an absolute squared-error on the curve (+ its slope +
> its curvature); the **% err** row is the function-level MAPE of the curve L only. It stays in a sane
> range (≈ 15–164%): the ACF plots are largest because the true ACF ≈ 0.05 sits near zero, so any
> deviation is a big *relative* error — a property of the curve, not a bug. The ill-posed derivative
> MAPE is excluded, so the figures no longer explode to 10⁴-%.
>
> **Log-return histogram**: SBTS (MSE 12.1) much smaller than TimeGAN (144.2) — kernel smoothing closely
> preserves marginal returns. TimeGAN std=120.6 is driven by a seed-2 collapse (504.5 vs 11–170).
>
> **QQ plot**: TimeGAN wins on MSE (7.1e-6 vs 8.9e-6) among the non-flow methods; SBTS wins on % err.
>
> **Rolling vol histogram**: with real-data bins, SBTS's near-constant vol produces high MSE (1227 vs 439);
> Fourier Flow wins here.
>
> **Tail survival**: Fourier Flow wins decisively on both rows; SBTS is next on % err.

**Fourier Flow wins B: 4/6 plots on MSE; CSDI wins the two ACF plots.** Fourier Flow's spectral
objective fits the marginal-shape diagnostics (log-return histogram, QQ, rolling-vol, tail survival)
tighter than any method, while CSDI wins both autocorrelation curves (ACF \|r\| and ACF r²) on
**both** MSE and % err by an order of magnitude — its score-based diffusion reproduces Heston's weak
vol-clustering ACF that the spectral flow slightly over-smooths (ACF \|r\| MSE 6.27e-05 vs DTS 5.76e-04
vs FF 1.30e-03; % err 15.1% vs DTS 74.8% vs FF 115.2%).
**TimeVAE loses all six B plots** — its posterior-mean decoder collapses the marginal shape (log-return
histogram MSE 2887 vs FF 2.8, rolling-vol histogram MSE 47159 vs FF 92), consistent with its heavily
under-dispersed samples.
**TimeVQVAE also wins no B plot but is consistently mid-pack** — third-best log-return histogram (MSE 13.2,
just behind SBTS 12.1 and far ahead of TimeGAN 144), comparable QQ (5.3e-06) and tail-survival (5.1e-03)
curves to CSDI/SBTS, and a rolling-vol histogram MSE (333) between FF (92) and CSDI (464) — a full 140×
tighter marginal shape than TimeVAE.
**COSCI-GAN wins no B plot and ranks near the bottom of every one** — it is 6th–8th of 8 on all six plots
and **dead-last on QQ** (MSE 2.48e-03, worse than TimeVAE's 1.19e-04). Its log-return-histogram MSE (128.6)
beats only TimeGAN (144) and TimeVAE (2887), and is ~45× worse than Fourier Flow (2.85); its tail-survival
MSE (0.179) and rolling-vol MSE (4202) are the second-worst of the benchmark behind TimeVAE. This is the
direct curve-level counterpart to its scalar-moment table: the near-Gaussian generated marginal matches
low-order moments (A1, A15) but not the full density curve or tails. Each value is computed over the same
**5 seeds** per method.

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
| Fourier Flow | [`Heston/FourierFlow/`](Heston/FourierFlow/) | [`../methods/FourierFlow/`](../methods/FourierFlow/) |
| Diffusion-TS | [`Heston/DiffusionTS/`](Heston/DiffusionTS/) | [`../methods/DiffusionTS/`](../methods/DiffusionTS/) |
| CSDI | [`Heston/CSDI/`](Heston/CSDI/) | [`../methods/CSDI/`](../methods/CSDI/) |
| TimeVAE | [`Heston/TimeVAE/`](Heston/TimeVAE/) | [`../methods/TimeVAE/`](../methods/TimeVAE/) |
| TimeVQVAE | [`Heston/TimeVQVAE/`](Heston/TimeVQVAE/) | [`../methods/TimeVQVAE/`](../methods/TimeVQVAE/) |
| COSCI-GAN | [`Heston/COSCI-GAN/`](Heston/COSCI-GAN/) | [`../methods/COSCI-GAN/`](../methods/COSCI-GAN/) |
| Perfect recovery (floor) | — | [`../methods/perfect_recovery/`](../methods/perfect_recovery/) |

---

## Methods

### TimeGAN — Time-series Generative Adversarial Network
**Paper:** Yoon, Jarrett, van der Schaar — *Time-series GAN* — NeurIPS 2019, [arXiv:2010.00782](https://arxiv.org/abs/2010.00782)
**Code:** [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) — PyTorch reimplementation in this repo

TimeGAN is a **neural GAN** with five interacting GRU components:
- **Embedder + Recovery** (3-layer GRU, hidden=24): maps price paths ↔ latent embedding space
- **Generator** (3-layer GRU): generates latent sequences from Gaussian noise
- **Supervisor** (2-layer GRU): enforces step-by-step temporal consistency in latent space
- **Discriminator** (3-layer GRU): distinguishes real from generated latent sequences

**Training**: 3-phase adversarial, 20 000 steps (5 k embed → 5 k supervisor → 10 k joint).
**Hardware**: GPU required (A100 80 GB). ~6–8 min/seed.
**Generation**: Milliseconds (GRU forward pass). Sequences start near S₀=100 via internal min-max denorm.

### SBTS — Schrödinger Bridge Time Series
**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)
**Code:** [alexouadi/SBTS](https://github.com/alexouadi/SBTS) — Numba-accelerated reimplementation in this repo

SBTS is a **non-parametric kernel method** with no neural network and no training:
- Estimates the Schrödinger-bridge drift from training data using a compact quartic kernel K_h
- Simulates paths via Euler-Maruyama with the estimated drift (N_pi=200 substeps per interval)
- Markovian order K=1: weight of path m depends only on its most recent state X_i^m (valid for Heston)
- Internally operates on **scaled log-returns** R̃ = R × √Δt / σ(R) — not on prices or log-prices — then reconstructs prices: S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])

**Generation**: No training phase. ~6.3 min/seed with 64 CPU workers.
**Hardware**: CPU-only (Numba JIT). GPUs only used for A13/A14 metric evaluation.

### Fourier Flow — Generative Time-series Modeling with Fourier Flows
**Paper:** Alaa, Chan, van der Schaar — *Generative Time-series Modeling with Fourier Flows* — ICLR 2021, [OpenReview](https://openreview.net/forum?id=PpshD0AXfA)
**Code:** [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) — released-code-as-is reimplementation in this repo

Fourier Flow is an **explicit-likelihood normalizing flow that operates in the frequency domain**:
- Applies a **Discrete Fourier Transform** to each path, then a chain of invertible spectral filters (3 flows)
- Each **SpectralFilter** is an MLP (hidden=200) coupling layer acting on the real/imaginary spectral bins
- Trained by **direct negative-log-likelihood** minimisation (loss `(−log_pz − log_jacob).mean()`), full-batch Adam + ExponentialLR (γ=0.999), **1000 epochs**, **CPU-only** (numpy.fft)
- Inverts the flow and applies the **inverse DFT** to sample new price paths

**Two numerical guards** make training finite on Heston (paths start at a deterministic S₀=100, so the spectral covariance is near-singular at the DC bin): a **zero-std spectral-bin clamp** (necessary but not sufficient) and a **gradient clip = 1.0** (the actual fix that removes the NaN blow-up). See [`Heston/FourierFlow/README.md`](Heston/FourierFlow/README.md).

**Training**: ~8.2 min/seed (490 s, CPU). **Generation**: ~1.5 s/seed. **Hardware**: CPU-only; GPUs only used for A13/A14 metric evaluation.

### Diffusion-TS — Interpretable Diffusion for General Time Series Generation
**Paper:** Yuan, Qiao — *Diffusion-TS: Interpretable Diffusion for General Time Series Generation* — ICLR 2024, [arXiv:2403.01742](https://arxiv.org/abs/2403.01742)
**Code:** [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) — released-code-as-is reimplementation in this repo

Diffusion-TS is a **non-autoregressive denoising diffusion model (DDPM)** with an interpretable
encoder-decoder transformer:
- Generates a whole length-T series in one reverse-diffusion trajectory (no step-by-step roll-out)
- **Predicts the clean signal x̂₀ directly** at each diffusion step (not the added noise ε), making the target a reconstruction of the series
- The decoder reconstructs x̂₀ as an explicit sum of a polynomial **trend** block and Fourier-based **seasonal** blocks (disentangled seasonal-trend decomposition)
- Trained by a **reweighted L1 + Fourier-FFT reconstruction loss** with a **cosine β** schedule over **500** diffusion steps; EMA weights (decay 0.995) used for sampling
- Uses the `mujoco` preset (n_layer_enc = n_layer_dec = 3, d_model = 64, 544 147 params, 12 000 steps) — chosen by an identical 3 000-step smoke test that scored `mujoco` Context-FID 0.7367 vs `etth` 2.3192 vs `stocks` 36.05 (lower is better). See [`../methods/DiffusionTS/code/README.md`](../methods/DiffusionTS/code/README.md).

**Training**: ~14.6 min/seed (878 s, A100 GPU). **Generation**: 500-step DDPM sampling with EMA weights (not separately timed). **Hardware**: GPU required (A100 80 GB); GPUs also used for A13/A14 metric evaluation.

### CSDI — Conditional Score-based Diffusion Models for Imputation
**Paper:** Tashiro, Song, Song, Ermon — *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* — NeurIPS 2021, [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)
**Code:** [ermongroup/CSDI](https://github.com/ermongroup/CSDI) — released-code-as-is reimplementation in this repo

CSDI is a **score-based denoising diffusion model (DDPM)** whose denoiser is a **2-D
(time × feature) transformer** with residual layers:
- For unconditional Heston generation we set `is_unconditional = 1` and the conditioning mask ≡ 0, so the
  model reduces to a **standard DDPM** that denoises the whole length-128 series in one reverse trajectory
- **Predicts the added noise ε** at each diffusion step (ε-matching), target = the injected Gaussian noise
- The denoiser stacks 4 residual blocks (64 channels, 8 attention heads) with a temporal transformer and a
  feature transformer, plus a 128-d diffusion-step embedding and a 16-d feature embedding
- Trained on **z-scored prices** (mean 101.33, std 9.97) by **noise-prediction MSE**
  E_t‖ε − ε_θ(x_t, t)‖² with a **quadratic β** schedule over **50** diffusion steps (β 1e-4 → 0.5);
  Adam lr 1e-3, weight-decay 1e-6, batch 16, 200 epochs, MultiStepLR (×0.1 at 75%/90% of training)
- ~413 k parameters. See [`../methods/CSDI/code/README.md`](../methods/CSDI/code/README.md).

**Training**: ~29.3 min/seed (1 756 s, A100 GPU). **Generation**: ~10.2 s/seed (50-step DDPM). **Hardware**: GPU required (A100 80 GB); GPUs also used for A13/A14 metric evaluation.

### TimeVAE — Variational Auto-Encoder for Multivariate Time Series
**Paper:** Desai, Freeman, Beaver, Wang — *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation* — 2021, [arXiv:2111.08095](https://arxiv.org/abs/2111.08095)
**Code:** [abudesai/timeVAE](https://github.com/abudesai/timeVAE) — PyTorch reimplementation in this repo (the official code is TensorFlow/Keras, which has no working GPU build for this machine's CUDA driver)

TimeVAE is a **variational auto-encoder** with a convolutional encoder and a decoder that reconstructs the
whole length-T series in one forward pass:
- **Encoder**: stacked 1-D convolutions (hidden channels 50 → 100 → 200) → flatten → Linear to a **latent
  dimension of 8** (posterior mean + log-var), reparameterised sample z ~ N(μ, σ²)
- **Decoder** (TimeVAE-**Base**): Linear + transposed convolutions map z back to the length-128 series; the
  optional interpretable **trend** (`trend_poly=0`) and **seasonal** (`custom_seas=None`) blocks are disabled,
  so this is the pure convolutional base with a residual connection (`use_residual_conn=True`)
- Trained by the **ELBO**: `reconstruction_wt · (SSE + feature-mean SSE) + KL`, with `reconstruction_wt = 3.0`;
  Adam lr 1e-3, batch 16, EarlyStopping (5 seeds stop between 230–340 epochs)
- ~247 k parameters (feat_dim 1, seq_len 128). See [`../methods/TimeVAE/code/README.md`](../methods/TimeVAE/code/README.md).

Because the decoder regresses toward the **posterior mean**, TimeVAE produces accurate *average* trajectories
(it wins A13/A25 mean-path RMSE and A33 teacher-sigma correlation) but heavily **under-dispersed** marginals
(worst-of-six on fat-tail A1, vol MMD A9, rolling-vol KS A31; loses all six B curve-shape plots). Its
Path-Shadowing CRPS (3.855/5.634) does not beat the random-walk baseline.

**Training**: ~13 min/seed (A100 GPU). **Generation**: <1 s/seed (single decoder forward pass). **Hardware**: GPU used for training and A13/A14 metric evaluation.

### TimeVQVAE — Vector Quantized Time Series Generation
**Paper:** Lee, Malacarne, Aune — *Vector Quantized Time Series Generation with a Bidirectional Prior Model* — AISTATS 2023, [arXiv:2303.04743](https://arxiv.org/abs/2303.04743)
**Code:** [ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) — reference code (commit `b9650e9d`, PyTorch + PyTorch-Lightning) run as-is behind a thin data-plumbing wrapper in this repo

TimeVQVAE is a **two-stage vector-quantized generative model** that operates in the STFT
time-frequency domain:
- **Stage 1 — VQ tokenization**: an STFT (`n_fft=8`) splits each path into a low-frequency (LF, bin 0)
  and high-frequency (HF, bins 1:) branch; each branch has its own ResNet encoder/decoder (dim 64, 4
  blocks) and a **codebook of 32 codes** (dim 64, EMA decay 0.8) that discretises the latent into tokens
- **Stage 2 — MaskGIT bidirectional prior**: a masked bidirectional transformer (hidden 256, 4 layers,
  2 heads, RMSNorm, `p_uncond = 0.2`) learns the token prior; the HF token stream is **conditioned on
  the LF tokens**
- **Generation** via `unconditional_sample` — iterative MaskGIT decoding (T=10 steps, choice temperature
  4, guidance 1.0) fills the token grid, then the Stage-1 decoders + inverse STFT map tokens back to a
  price path
- Trained on **globally z-normalized prices** (paper `data_scaling=True`, mean 101.33, std 9.97),
  inverted to price scale before saving. Epoch budget stage1 = 250 / stage2 = 1000 (matched to the
  paper's gradient-step count on the 16×-larger Heston set). See
  [`../methods/TimeVQVAE/code/README.md`](../methods/TimeVQVAE/code/README.md).

TimeVQVAE **wins the two structural-error metrics** (A20 covariance 16.6, A32 vol-of-vol 6.75e-04),
is a solid mid-pack generator on the fat-tail/MMD/curve-shape diagnostics, and its Path-Shadowing CRPS
(2.771/3.889) clears the random-walk baseline at both horizons.

**Training**: ~53 min/seed (A100 GPU, two stages). **Generation**: ~6 s/seed (MaskGIT decode + iSTFT). **Hardware**: GPU required (A100 80 GB); GPUs also used for A13/A14 metric evaluation.

### COSCI-GAN — COmmon Source CoordInated GAN
**Paper:** Seyfi, Rajotte, Ng — *Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)* — NeurIPS 2022, [arXiv:2210.07248](https://arxiv.org/abs/2210.07248)
**Code:** [aliseyfi75/COSCI-GAN](https://github.com/aliseyfi75/COSCI-GAN) — PyTorch reimplementation in this repo

COSCI-GAN is a **channel-decomposed GAN** designed for *multivariate* series: one univariate
"Channel GAN" per feature, all sharing the **same noise vector z**, plus a **Central Discriminator (CD)**
that couples the channels to preserve cross-channel dependence:
- **Channel GAN** (×C): an LSTM generator (z → LSTM 32→256 → Linear→128) and an LSTM discriminator
  (hidden 256, 1 layer, sigmoid), one per channel
- **Central Discriminator**: an MLP (128→256→128→64→1, LeakyReLU 0.1 + Dropout 0.3) that sees all
  channels jointly; three-player minimax `loss_G_i = BCE(D_i, 1) − γ·loss_CD`, γ=5
- Adam betas (0.5, 0.9), BCE, 120 epochs, ~800 k parameters

**Heston is univariate (C = 1)**, so COSCI-GAN runs with a **single channel** and the CD becomes
degenerate: it receives the same 128-dim vector as the single channel discriminator, so `loss_CD ≈ ln2 ≈
0.693` at equilibrium and the paper's native cross-channel correlation-matrix metric is structurally
undefined (the correlation matrix is a scalar 1, MAE ≡ 0 for any generator). We reproduced the paper's
EEG eye-state Table-4 correlation-MAE separately for validation (ours 0.1085 ± 0.0066 vs paper COSCI-GAN
0.111 ± 0.005). On Heston, COSCI-GAN wins **A5 Hill tail-index only**, has good scalar low-order moments
(A1 0.561, A15 0.044) but weak full-density curves (6th–8th of 8 on every B plot), a **saturated A18
discriminative score** (0.500 both probes — near-perfectly separable), thin/near-Gaussian tails (A28
−7.994) and a Path-Shadowing CRPS (4.657/5.834) that does **not** beat the random-walk baseline. See
[`Heston/COSCI-GAN/README.md`](Heston/COSCI-GAN/README.md).

**Training**: ~4.3 min/seed (257 s, A100 GPU). **Generation**: LSTM forward pass over shared noise (not separately timed). **Hardware**: GPU used for training and A13/A14 metric evaluation.

### Perfect recovery — reproducible floor
A row-shuffled copy of the real dataset (`S_real[rng.permutation(N)]`, one permutation per seed). Because a
permutation preserves every column-wise marginal exactly, most A-metrics and all B-metrics hit **0**; the
residual non-zero floors (A1–A6 MMD/SWD, A13/A14 learned scores, A15/A21 = 0.614) are pure finite-sample
noise. This is the single source of truth for every "Perfect floor" column in the repo —
see [`../methods/perfect_recovery/`](../methods/perfect_recovery/).

---

## Key differences

| Aspect | TimeGAN | SBTS | Fourier Flow | Diffusion-TS | CSDI | TimeVAE | TimeVQVAE | COSCI-GAN |
|--------|:-------:|:----:|:------------:|:------------:|:----:|:-------:|:---------:|:---------:|
| **Type** | Neural GAN (5 GRU components) | Non-parametric kernel estimator | Explicit-likelihood normalizing flow (frequency domain) | Denoising diffusion (DDPM) + seasonal-trend transformer | Score-based diffusion (DDPM) + time×feature transformer | Variational auto-encoder (conv encoder + decoder, Base) | Two-stage vector-quantized (STFT VQ-VAE + MaskGIT prior) | Channel-decomposed GAN (per-channel LSTM GANs + MLP central discriminator) |
| **Learnable parameters** | ~120 k (GRU weights) | **0** (no parameters) | ~360 k (3 spectral-filter MLPs, hidden=200) | ~544 k (enc/dec transformer, mujoco) | ~413 k (2-D transformer, 4 residual layers) | ~247 k (conv encoder/decoder, latent 8) | LF+HF codebooks (32×64) + MaskGIT transformer (hidden 256, 4 layers) | ~800 k (LSTM channel gen/disc + MLP central disc) |
| **Training time / seed** | ~6–8 min (A100 GPU) | No training | ~8.2 min (CPU, 1000 epochs) | ~14.6 min (A100 GPU, 12 000 steps) | ~29.3 min (A100 GPU, 200 epochs) | ~13 min (A100 GPU, EarlyStop 230–340 epochs) | ~53 min (A100 GPU, stage1 250 + stage2 1000 epochs) | ~4.3 min (A100 GPU, 120 epochs) |
| **Generation time / seed** | <1 s (GPU inference) | ~6.3 min (64 CPU workers) | ~1.5 s (CPU inverse flow + iDFT) | 500-step DDPM sampling (GPU) | ~10.2 s (50-step DDPM, GPU) | <1 s (single decoder forward pass) | ~6 s (MaskGIT decode + iSTFT, GPU) | LSTM forward over shared noise (not sep. timed, GPU) |
| **Temporal memory** | Full (GRU sees all past steps) | **Markov-1 only** | Global (per-frequency spectral coupling) | Global (transformer self-attention over full window) | Global (2-D transformer over time × feature) | Global (conv receptive field over full window) | Global (bidirectional MaskGIT transformer over token grid) | Full (LSTM sees all past steps) |
| **Internal representation** | Latent embeddings (min-max) | Scaled log-returns R̃ | DFT spectral bins (real/imag) | x̂₀ = trend + seasonal (time + Fourier domain) | z-scored prices + diffusion noise | 8-d Gaussian latent z | STFT VQ tokens (LF + HF codebooks) | Per-channel LSTM hidden state (shared noise z) |
| **Final output** | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) | Price paths (S_t) |
| **Cross-seed stability** | Moderate (GAN variance) | **High** (deterministic kernel) | High on moments, moderate on covariance | High on moments/ACF, moderate on GRU disc | High on moments/ACF, moderate on GRU disc | High on moments, moderate on mean-path (A13/A25 std ~0.2–0.3) | High on PS-MC (std 0.015), moderate on covariance (A20 std 14.7) | Moderate (GAN variance); wide A5/A28 spread (sign-flipping kurtosis ratio) |
| **Scales to long T** | Well (RNN) | Degrades (K=1 insufficient) | Well (fixed spectral size) | Well (transformer handles any T) | Well (transformer handles any T) | Well (fixed conv/latent size) | Well (transformer + more STFT tokens) | Well (LSTM); central disc degenerate at C=1 (univariate) |
| **Hyperparameter sensitivity** | Many (arch, lr, steps) | One critical: h (bandwidth) | Few (n_flows, hidden, grad-clip guard) | Moderate (depth preset, timesteps, EMA) | Moderate (layers, channels, diffusion steps, β schedule) | Few (latent dim, reconstruction_wt, hidden sizes) | Moderate (n_fft, codebook size, MaskGIT steps/temperature) | Moderate (γ central-disc weight, lr, epochs, LSTM hidden) |
| **Training objective** | Adversarial + supervised | Schrödinger-bridge drift (closed-form) | **Exact negative log-likelihood** | Reweighted L1 + Fourier-FFT reconstruction | Noise-prediction MSE (ε-matching) | ELBO (weighted reconstruction + KL) | Stage-1 VQ reconstruction + Stage-2 masked-token cross-entropy | Three-player adversarial (channel BCE − γ·central-disc BCE) |
