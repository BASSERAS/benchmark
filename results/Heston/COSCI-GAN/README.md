# Metrics — COSCI-GAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** COSCI-GAN (COmmon Source CoordInated GAN, Seyfi, Rajotte, Ng, NeurIPS 2022).
Per channel: one LSTM Channel-GAN (LSTMGenerator z→LSTM(32→256)→Linear(256→128);
LSTMDiscriminator x→LSTM(128→256)→Linear→Sigmoid), plus **one** MLP Central Discriminator
(128→256→128→64→1, LeakyReLU 0.1 + Dropout 0.3) coordinating channels through a **shared noise
source** z. Three-player minimax: `loss_G_i = BCE(D_i(fake_i),1) − γ·loss_CD`, γ=5. PyTorch,
Adam betas (0.5, 0.9), 120 epochs, 799 618 params. See
[`../../../methods/COSCI-GAN/code/README.md`](../../../methods/COSCI-GAN/code/README.md).

> ⚠️ **C=1 degeneracy (documented honestly).** Heston is **univariate** (price only), so COSCI-GAN
> runs with a **single channel** (`n_groups=1`). The Central Discriminator then receives the *same*
> 128-dim vector as the lone Channel-Discriminator — it becomes a redundant second critic with no
> cross-channel correlation to coordinate. Its healthy equilibrium is therefore `loss_CD ≈ ln 2 ≈ 0.693`
> (CD at chance). COSCI-GAN's *design purpose* — coordinating correlations across channels — cannot be
> exercised at C=1; on Heston it reduces to a single LSTM-GAN with a decorative coordinator. The paper's
> own headline metric (cross-channel correlation MAE) is **structurally undefined** here — see
> [§ Comparison with the paper](#comparison-with-the-paper-cosci-gans-native-metric).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case a perfect generator reaches with finite
samples, from a row-shuffled copy of the real data (see
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/)). Most floors are 0 because a
permutation preserves every column-wise marginal; the residual non-zero floors are pure finite-sample
noise, and are **identical across methods** (same real data, same permutation).

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | |
| A1 | Kurtosis Error | 0.5612 ± 0.1128 | 0.5685 | 0.6518 | 0.4435 | 0.4273 | 0.7147 | 0 |
| A2 | \|r\| q95 Error | 0.0972 ± 0.0035 | 0.1027 | 0.0962 | 0.0934 | 0.0995 | 0.0942 | 0 |
| A3 | \|r\| q99 Error | 0.1240 ± 0.0060 | 0.1261 | 0.1214 | 0.1201 | 0.1346 | 0.1178 | 0 |
| A4 | Tail QQ Error | 0.0957 ± 0.0035 | 0.1012 | 0.0947 | 0.0919 | 0.0983 | 0.0925 | 0 |
| A5 | Hill Tail Index Error | 1.563 ± 1.206 | 1.622 | 2.912 | 2.879 | 0.2466 | 0.1548 | 0 |
| | **— Distribution —** | | | | | | | |
| A6 | Path MMD² | 0.0467 ± 0.0038 | 0.0511 | 0.0401 | 0.0495 | 0.0477 | 0.0451 | 0.0015 |
| A7 | Terminal MMD² | 0.0138 ± 0.0137 | 0.0034 | 0.0014 | 0.0056 | 0.0211 | 0.0374 | 0.0016 |
| A8 | Increment MMD² | 0.4784 ± 0.0108 | 0.4951 | 0.4794 | 0.4614 | 0.4763 | 0.4799 | 7.45e-04 |
| A9 | Volatility MMD | 3.960 ± 0.0432 | 3.953 | 4.015 | 3.888 | 3.992 | 3.954 | 0.0071 |
| A10 | Terminal SWD | 4.550 ± 3.115 | 2.723 | 1.086 | 2.336 | 8.052 | 8.552 | 0.6873 |
| A11 | Path SWD | 3.486 ± 0.1871 | 3.764 | 3.222 | 3.617 | 3.412 | 3.415 | 0.4381 |
| A12 | RV Law Loss | 118.77 ± 7.929 | 129.78 | 118.37 | 107.77 | 124.96 | 112.96 | 0 |
| A13 | Mean Path RMSE | 4.007 ± 0.1941 | 4.272 | 3.736 | 4.181 | 3.922 | 3.922 | 0 |
| A14 | KS Log-returns | 0.3208 ± 0.0073 | 0.3156 | 0.3236 | 0.3122 | 0.3332 | 0.3194 | 0 |
| A15 | Skewness Error | 0.0445 ± 0.0386 | 0.1128 | 0.0146 | 0.0615 | 0.0217 | 0.0118 | 0 |
| A16 | QQ RMSE (300-pt) | 0.0486 ± 0.0020 | 0.0514 | 0.0487 | 0.0458 | 0.0498 | 0.0472 | 0 |
| A17 | Terminal Price KS | 0.1481 ± 0.0985 | 0.0992 | 0.0253 | 0.0936 | 0.2261 | 0.2963 | 0 |
| | **— Adversarial —** | | | | | | | |
| A18 GRU | Discriminative Score GRU | 0.4998 ± 2.4e-04 | 0.4994 | 0.5000 | 0.4997 | 0.5000 | 0.5000 | 0.0042 |
| A18 MLP | Discriminative Score MLP | 0.5000 ± 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0067 |
| | **— Predictive —** | | | | | | | |
| A19 GRU | Predictive Score GRU | 0.1308 ± 0.0177 | 0.1476 | 0.1065 | 0.1362 | 0.1499 | 0.1138 | 0.0537 |
| A19 MLP | Predictive Score MLP | 0.1089 ± 0.0071 | 0.1182 | 0.1116 | 0.0967 | 0.1109 | 0.1071 | 0.0539 |
| | **— Temporal —** | | | | | | | |
| A20 | Covariance Error | 28.90 ± 25.36 | 10.84 | 4.775 | 67.76 | 10.66 | 50.45 | 0 |
| A21 | ACF \|r\| Error (lags) | 0.0806 ± 0.0206 | 0.0787 | 0.1093 | 0.0657 | 0.0970 | 0.0524 | 0 |
| A22 | ACF r² Error (lags) | 0.0902 ± 0.0214 | 0.0853 | 0.1206 | 0.0737 | 0.1082 | 0.0630 | 0 |
| A23 | ACF \|r\| Lag-1 Error | 0.1663 ± 0.0493 | 0.1758 | 0.1955 | 0.1414 | 0.2319 | 0.0870 | 0 |
| A24 | ACF r² Lag-1 Error | 0.1916 ± 0.0511 | 0.2109 | 0.2327 | 0.1673 | 0.2432 | 0.1038 | 0 |
| | **— Vol —** | | | | | | | |
| A25 | Mean RMSE | 4.500 ± 3.382 | 3.279 | 0.7856 | 1.410 | 8.232 | 8.791 | 0 |
| A26 | Return Std Error | 5.033 ± 0.2229 | 5.300 | 5.083 | 4.684 | 5.212 | 4.888 | 0 |
| A27 | Log-Return Std Error | 0.0498 ± 0.0020 | 0.0525 | 0.0497 | 0.0470 | 0.0513 | 0.0483 | 0 |
| A28 | Kurtosis Ratio | −7.994 ± 11.88 | −19.07 | −9.027 | 6.221 | 4.696 | −22.79 | 1.000 |
| A29 | Sigma Mean Error | 0.7875 ± 0.0309 | 0.8308 | 0.7881 | 0.7440 | 0.8097 | 0.7648 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | 1.065 ± 0.2608 | 1.023 | 0.8842 | 1.569 | 0.9986 | 0.8478 | 0 |
| A31 | Rolling Vol KS (w=5) | 0.9373 ± 0.0076 | 0.9437 | 0.9453 | 0.9237 | 0.9372 | 0.9364 | 0 |
| A32 | Vol-of-Vol Error | 0.0181 ± 0.0011 | 0.0191 | 0.0168 | 0.0181 | 0.0195 | 0.0167 | 0 |
| | **— Heston Spec —** | | | | | | | |
| A33 | Teacher-Sigma Corr ↑ | −0.0043 ± 0.0131 | −0.0291 | −0.0055 | 0.0026 | 0.0076 | 0.0028 | 0.6143 |
| A34 | Teacher-Sigma RMSE | 0.8095 ± 0.0288 | 0.8555 | 0.8105 | 0.7721 | 0.8219 | 0.7875 | 0.0654 |

**Reading the table — an honest mixed result.** COSCI-GAN captures the **centre** of the Heston
log-return distribution very well but **fails the adversarial and tail tests**.

The good — scalar low-order moments. **A5 Hill tail-index error 1.56 is the best of *any* method in
the benchmark** (beating CSDI 1.99 — a genuine win). **A1 kurtosis error 0.561** is the best of the
VAE/GAN family (TimeVAE 2.26, TimeGAN 2.96), though behind CSDI (0.096) / SBTS (0.119) overall, and
**A15 skewness error 0.044** is near-perfect (4th of 8). The **autocorrelation** structure is also
comparatively well matched — A21 ACF-|r| 0.081, A23 lag-1 0.166 — an order of magnitude better than
TimeVAE (0.39 / 0.46), because the shared-noise LSTM generator carries some ARCH-like memory. **Note
the tension**, resolved in the B section below: these good *scalar* moments do **not** carry over to
the full-density *curve* diagnostics, where COSCI-GAN ranks near the bottom.

The bad — and it is fundamental. **A18 discriminative score = 0.4998 (GRU) / 0.5000 (MLP)** — this is
the **maximum** value the metric can take (score = |acc − 0.5|, 0 = indistinguishable, 0.5 = perfectly
separable). The GRU and MLP judges classify real-vs-generated **near-perfectly**. This is a **real
quality signal, not a metric artefact**: the judge's training BCE **collapses to ≈1e-6 (MLP) / ≈1e-3
(GRU)** (see the Classifier-Loss section) — the paths are simply distinguishable — whereas a method the
judge cannot separate (e.g. TimeVAE seed 3) leaves the BCE parked at ln 2 ≈ 0.693. The judge hidden dim
is floored at `max(8, n_features·8)=8`, so this is **not** the `int(dim/2)=0` degeneracy that can
falsely inflate the score on 1-D data.

The tails are thin. **A28 kurtosis ratio −7.99** (± 11.9; seeds flip sign, −22.8 … +6.2) — COSCI-GAN's
log-returns are **near-Gaussian to slightly platykurtic**, so the ratio to Heston's mild positive excess
kurtosis straddles zero. Combined with **A9 volatility MMD 3.96** (high, comparable to TimeVAE) and
**A31 rolling-vol KS 0.937**, the picture is: a well-centred marginal with **no fat tails and no
stochastic-volatility signature**. As with every method here, **A33 teacher-sigma correlation ≈ 0** — no
generator recovers the latent Heston variance from prices alone (perfect floor 0.614, unreachable
without the hidden state). Net: a **strong marginal generator whose individual paths remain trivially
distinguishable** and whose volatility dynamics are absent.

---

## Stylised Facts Diagnostic (Heston vs COSCI-GAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means;
  combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one
  division: the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is
  **excluded** (near-zero true diffs make it explode). Combined mean/std = mean and sample std across the 5 seeds.

↓ lower is better for both rows. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals).
**Honest reversal of the A-table story.** Despite the excellent *scalar* moments (A1/A15) above,
COSCI-GAN's full-density *curves* are **among the weakest in the whole benchmark**: on the cross-method
B table ([`../../README.md`](../../README.md)) it ranks **6th–8th of 8 on every plot** — dead-last on QQ
(MSE 2.48e-03, worse than all incl. TimeVAE), and second-worst overall (only TimeVAE is worse). The
generated log-return distribution is **near-Gaussian**: it matches Heston's low-order moments but not
the true density *shape* (QQ, tails) or its volatility structure (rolling-vol MSE 4 202, ACF %err ~210%).
The log-return-histogram MSE (128.6) is better than only TimeGAN (144) and TimeVAE (2887), and ~45× worse
than Fourier Flow (2.85).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 128.59 ± 5.889 | 124.09 | 122.33 | 124.90 | 135.90 | 135.74 | 0 |
| | % err | 248.65% ± 7.948% | 241.57% | 242.41% | 242.55% | 259.15% | 257.56% | 0 |
| **QQ plot** | MSE | 2.477e-03 ± 1.97e-04 | 2.755e-03 | 2.473e-03 | 2.203e-03 | 2.625e-03 | 2.330e-03 | 0 |
| | % err | 437.16% ± 19.28% | 448.24% | 436.79% | 401.12% | 457.44% | 442.23% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 0.0256 ± 0.0069 | 0.0243 | 0.0458 | 0.0223 | 0.0213 | 0.0143 | 0 |
| | % err | 211.80% ± 41.11% | 190.40% | 289.82% | 189.33% | 214.92% | 174.52% | 0 |
| **ACF r² lags 1–20** | MSE | 0.0263 ± 0.0070 | 0.0207 | 0.0454 | 0.0290 | 0.0229 | 0.0136 | 0 |
| | % err | 251.70% ± 48.25% | 201.78% | 343.90% | 242.67% | 232.58% | 237.54% | 0 |
| **Rolling vol histogram** | MSE | 4202.5 ± 102.3 | 4179.7 | 4337.2 | 4087.9 | 4105.1 | 4302.5 | 0 |
| | % err | 802.75% ± 14.02% | 799.52% | 821.48% | 785.06% | 791.47% | 816.21% | 0 |
| **Tail survival** | MSE | 0.1794 ± 0.0060 | 0.1786 | 0.1791 | 0.1695 | 0.1880 | 0.1820 | 0 |
| | % err | 342.77% ± 8.325% | 343.62% | 344.95% | 327.90% | 353.68% | 343.69% | 0 |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

---

## Discriminative & Predictive Classifier Losses (A18 / A19)

BCE loss during GRU/MLP discriminator training (A18) and MAE loss during GRU/MLP predictor training on
*synthetic* data (A19, TSTR), 5 seeds. A discriminator BCE near ln(2) ≈ 0.693 means real and generated
are indistinguishable — **the opposite of what we see here.** COSCI-GAN's judge BCE **drives to ≈1e-6
(MLP) / ≈1e-3 (GRU)**, the direct evidence that A18 = 0.50 reflects genuinely separable paths (not a
degenerate judge).

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (COSCI-GAN's native metric)

> ⚠️ **The paper's headline metric is undefined on univariate Heston.** COSCI-GAN's NeurIPS 2022
> evaluation (Table 4) is a **cross-channel correlation-matrix MAE**: it compares the C×C Pearson
> correlation matrix of the generated channels against the real one. This metric **requires C ≥ 2
> channels** and measures exactly the quantity COSCI-GAN is designed to model — inter-channel
> coordination. Heston is **univariate (C = 1)**, so the correlation matrix is the scalar 1, the MAE is
> identically 0 for any generator, and the metric carries **no information**. There is therefore no
> meaningful "Ours — Heston" entry for the paper metric.

Because the native metric cannot be exercised on Heston, we validated our COSCI-GAN port **on the paper's
own multivariate dataset** instead — the **EEG eye-state** benchmark from Table 4 — before running the
univariate Heston experiment above.

| Dataset | Metric | Ours (PyTorch, 5 seeds) | Paper — COSCI-GAN | Paper — GroupGAN | Paper — TimeGAN | Paper — FourierFlow | Verdict |
|---------|--------|:-----------------------:|:-----------------:|:----------------:|:---------------:|:-------------------:|---------|
| EEG eye-state | Corr-matrix MAE ↓ | 0.1085 ± 0.0066 | 0.111 ± 0.005 | 0.111 | 0.257 | 0.146 | **matches** ✓ |

Our re-trained COSCI-GAN reaches **0.1085 ± 0.0066**, statistically indistinguishable from the paper's
**0.111 ± 0.005** and matching its ranking (COSCI-GAN ≈ GroupGAN ≪ FourierFlow < TimeGAN). This confirms
the port reproduces COSCI-GAN's **cross-channel coordination** faithfully on a truly multivariate task —
the capability that simply has nothing to coordinate on univariate Heston prices. Full write-up:
[`../../../methods/COSCI-GAN/paper_reimplementation/`](../../../methods/COSCI-GAN/paper_reimplementation/).
Source:
[`../../../methods/COSCI-GAN/paper_reimplementation/results/eeg_table4.json`](../../../methods/COSCI-GAN/paper_reimplementation/results/eeg_table4.json).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K=77 nearest COSCI-GAN paths by L2 in z-scored space, forecast with their price-anchored futures.

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 4.657 ± 0.775 | 4.656 ± 0.773 | 5.834 ± 0.763 | 5.834 ± 0.764 | 3.73 / 5.30 |
| MAE    | 6.030 ± 0.891 | 6.027 ± 0.888 | 7.674 ± 0.866 | 7.673 ± 0.866 | 3.73 / 5.30 |
| RMSE   | 7.613 ± 0.940 | 7.610 ± 0.938 | 9.782 ± 0.944 | 9.780 ± 0.944 | 5.07 / 7.18 |

PS-MC over the COSCI-GAN pool **does not beat the naive random walk**: CRPS 4.657 > 3.73 at H=32 and
5.834 > 5.30 at H=64. Only **1 of 5 seeds** (seed 2, CRPS 3.538) edges below the RW floor at H=32; the
per-seed CRPS spread is 4.188 / 4.673 / 3.538 / 5.063 / 5.825. Because the generated paths carry no
Heston volatility structure (A9/A31/A33 above), their nearest-neighbour futures are no more informative
than a driftless RW. This contrasts with Diffusion-TS, whose pool **beats** RW on CRPS (2.72 / 3.85).
Uniform ≈ Gaussian: Heston is time-homogeneous. Full breakdown:
[`path_shadowing/README.md`](path_shadowing/README.md).

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B two-subline aggregates (MSE + % err) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts (summary.json + per-seed + plots + README) |

→ Cross-method comparison with TimeGAN, SBTS, Fourier Flow, Diffusion-TS, CSDI, TimeVAE & TimeVQVAE: [`results/README.md`](../../README.md)
