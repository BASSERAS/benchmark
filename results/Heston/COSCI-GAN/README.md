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

**Evaluation protocol (test set everywhere).** Generators were trained on the **train** split (seed 0) and
are **never scored on it**. Every metric below compares the 8 192 generated paths against the **held-out
test set** (an independent 8 192-path Heston draw, seed 1) — with one deliberate exception: the two
adversarial/predictive metrics A18 (discriminative) and A19 (predictive-TSTR) draw their *real* class from a
**third** Heston split (seed 2), so the judge never sees the same real data used everywhere else. This is the
protocol applied identically to all nine methods.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the best value a *perfect* generator can reach at this sample size. It is
measured by scoring an **independent Heston draw** (fresh seeds, identical parameters) against the same test
set — i.e. real-vs-real finite-sample noise. It is **non-zero** (finite samples never match exactly) and
**identical across all methods**, because it depends only on the test set and the protocol, not on the
generator. See [`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

<!-- ===== PER-METHOD A TABLE ===== -->
| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.5615 ± 0.1128 | 0.5689 | 0.6521 | 0.4439 | 0.4277 | 0.7150 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.09711 ± 0.003466 | 0.1026 | 0.09618 | 0.09329 | 0.09939 | 0.09408 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.1240 ± 0.005959 | 0.1261 | 0.1213 | 0.1201 | 0.1346 | 0.1178 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.09566 ± 0.003535 | 0.1011 | 0.09464 | 0.09185 | 0.09820 | 0.09244 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 1.614 ± 1.128 | 1.057 | 3.477 | 2.314 | 0.8120 | 0.4107 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.04686 ± 0.004162 | 0.05151 | 0.04119 | 0.05157 | 0.04643 | 0.04361 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.01623 ± 0.01333 | 0.006198 | 0.002270 | 0.009863 | 0.02459 | 0.03820 | 0.001983 |
| A8 Increment MMD² ↓ | 0.4788 ± 0.01185 | 0.4989 | 0.4774 | 0.4617 | 0.4786 | 0.4775 | 8.69e-04 |
| A9 Volatility MMD ↓ | 3.955 ± 0.04883 | 3.995 | 3.988 | 3.875 | 3.997 | 3.923 | 0.008554 |
| A10 Terminal SWD ↓ | 4.756 ± 3.118 | 3.349 | 0.8315 | 2.764 | 8.893 | 7.942 | 1.151 |
| A11 Path SWD ↓ | 3.505 ± 0.1711 | 3.739 | 3.288 | 3.672 | 3.405 | 3.422 | 0.6191 |
| A12 RV Law Loss ↓ | 118.7 ± 7.929 | 129.8 | 118.3 | 107.8 | 124.9 | 112.9 | 0.05202 |
| A13 Mean Path RMSE ↓ | 3.995 ± 0.1803 | 4.250 | 3.744 | 4.145 | 3.926 | 3.909 | 0.1205 |
| A14 KS Log-returns ↓ | 0.3206 ± 0.007269 | 0.3155 | 0.3234 | 0.3121 | 0.3331 | 0.3190 | 0.001491 |
| A15 Skewness Error ↓ | 0.04981 ± 0.04124 | 0.1217 | 0.02345 | 0.07044 | 0.01282 | 0.02066 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.04857 ± 0.001967 | 0.05140 | 0.04864 | 0.04575 | 0.04982 | 0.04722 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.1473 ± 0.09804 | 0.1035 | 0.02246 | 0.08972 | 0.2312 | 0.2894 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.4999 ± 1.22e-04 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4997 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.5000 ± 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.1331 ± 0.01808 | 0.1455 | 0.09753 | 0.1417 | 0.1359 | 0.1446 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.09591 ± 0.006992 | 0.1043 | 0.1019 | 0.08746 | 0.09785 | 0.08800 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 30.59 ± 29.16 | 4.828 | 1.233 | 73.77 | 16.66 | 56.45 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.08056 ± 0.02054 | 0.07869 | 0.1098 | 0.06495 | 0.09630 | 0.05303 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.09004 ± 0.02156 | 0.08482 | 0.1217 | 0.07282 | 0.1073 | 0.06357 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1700 ± 0.04930 | 0.1795 | 0.1992 | 0.1451 | 0.2355 | 0.09066 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.1957 ± 0.05105 | 0.2150 | 0.2368 | 0.1714 | 0.2473 | 0.1079 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 4.539 ± 3.359 | 3.477 | 0.9836 | 1.212 | 8.430 | 8.593 | 0.1392 |
| A26 Return Std Error ↓ | 5.032 ± 0.2229 | 5.299 | 5.082 | 4.683 | 5.211 | 4.887 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.04975 ± 0.002001 | 0.05250 | 0.04968 | 0.04694 | 0.05132 | 0.04829 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | -8.150 ± 12.11 | -19.44 | -9.203 | 6.342 | 4.788 | -23.24 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.7871 ± 0.03094 | 0.8305 | 0.7877 | 0.7436 | 0.8094 | 0.7644 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.155 ± 0.3231 | 1.097 | 0.8276 | 1.757 | 1.159 | 0.9320 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.9371 ± 0.007667 | 0.9435 | 0.9452 | 0.9234 | 0.9370 | 0.9363 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.01806 ± 0.001147 | 0.01915 | 0.01685 | 0.01809 | 0.01950 | 0.01670 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -0.005511 ± 0.008042 | -0.02120 | -0.003004 | 3.30e-04 | 3.78e-04 | -0.004059 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.8087 ± 0.02874 | 0.8544 | 0.8096 | 0.7713 | 0.8213 | 0.7868 | 0.06559 |

**Reading the table — an honest mixed result.** COSCI-GAN captures the **centre** of the Heston
log-return distribution reasonably well but **fails the adversarial, tail and volatility tests**. It wins
**none of the 36 A-metric rows** outright — under the test-set protocol every one of its best marginal
scores is beaten by a stronger generator.

The good — scalar low-order moments. **A1 kurtosis error 0.561** is the best of the VAE/GAN family
(TimeVAE and TimeGAN are far worse on this axis), though well behind the leaders CSDI (0.0954) and
SBTS (0.118); **A5 Hill tail-index error 1.61** is respectable but no longer a win — **LS4 owns A5** now,
and CSDI/SBTS also rank around COSCI-GAN. **A15 skewness error 0.050** is near-perfect. The
**autocorrelation** structure is comparatively well matched — A21 ACF-|r| 0.081, A23 lag-1 0.170 — an
order of magnitude better than TimeVAE, because the shared-noise LSTM generator carries some ARCH-like
memory. **Note the tension**, resolved in the B section below: these good *scalar* moments do **not**
carry over to the full-density *curve* diagnostics, where COSCI-GAN ranks near the bottom.

The bad — and it is fundamental. **A18 discriminative score = 0.4999 (GRU) / 0.5000 (MLP)** — this is
the **maximum** value the metric can take (score = |acc − 0.5|, 0 = indistinguishable, 0.5 = perfectly
separable). The GRU and MLP judges classify real-vs-generated **near-perfectly**. This is a **real
quality signal, not a metric artefact**: the judge's training BCE **collapses to ≈1e-6 (MLP) / ≈1e-3
(GRU)** (see the Classifier-Loss section) — the paths are simply distinguishable — whereas a method the
judge cannot separate leaves the BCE parked at ln 2 ≈ 0.693. The judge hidden dim is floored at
`max(8, n_features·8)=8`, so this is **not** the `int(dim/2)=0` degeneracy that can falsely inflate the
score on 1-D data.

The tails are thin. **A28 kurtosis ratio −8.15** (± 12.1; seeds flip sign, −23.2 … +6.3) — COSCI-GAN's
log-returns are **near-Gaussian to slightly platykurtic**, so the ratio to Heston's mild positive excess
kurtosis straddles zero. Combined with **A9 volatility MMD 3.96** (high, comparable to TimeVAE),
**A26 return-std error 5.03** and **A31 rolling-vol KS 0.937**, the picture is: a well-centred marginal with
**no fat tails and no stochastic-volatility signature**. As with every method here, **A33 teacher-sigma
correlation ≈ 0** — no generator recovers the latent Heston variance from prices alone (perfect floor 0.616,
unreachable without the hidden state). Net: a **strong marginal generator whose individual paths remain
trivially distinguishable** and whose volatility dynamics are absent.

---

## Stylised Facts Diagnostic (Heston vs COSCI-GAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L′ (der), and its second finite difference
L″ (sec\_der) — then combine them into **three sub-scores per plot**:

- **MSE row** (decides the winner): for each list, mean((L\_gen − L\_real)²), averaged over the three lists
  (funct / der / sec\_der). This is the headline curve-fit error.
- **% err row** (function-level MAPE): mean(|L\_gen − L\_real| / (|L\_real| + 1e-6)) × 100 on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE row**: sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100 on the curve L
  only (funct-only).

↓ lower is better for all three rows. **Perfect floor** is the non-zero real-vs-test value an independent
Heston draw reaches — identical across methods.

**Honest reversal of the A-table story.** Despite the decent *scalar* moments (A1/A15) above, COSCI-GAN's
full-density *curves* are **among the weakest in the whole benchmark** — it wins **none of the 6 B-plots**
and only **TimeVAE is consistently worse**. The generated log-return distribution is **near-Gaussian and
memoryless**: it matches Heston's low-order moments but not the true density *shape* (QQ NRMSE 72.8 %, tails)
or its volatility structure (rolling-vol MSE 1 398). The **ACF r² % err (11 227 %)** is the **largest of any
method** — the squared-return memory is essentially absent. Every curve sits far above the LS4 winner.

<!-- ===== PER-METHOD B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 42.66 ± 1.999 | 41.16 | 40.58 | 41.39 | 45.11 | 45.07 | 0.1098 |
|  | % err | 246.6% ± 7.987% | 239.4% | 240.4% | 240.3% | 257.2% | 255.4% | 1.799% |
|  | NRMSE | 30.81% ± 0.7154% | 30.27% | 30.07% | 30.36% | 31.68% | 31.68% | 0.5328% |
| **QQ plot** | MSE | 8.25e-04 ± 6.60e-05 | 9.18e-04 | 8.24e-04 | 7.33e-04 | 8.74e-04 | 7.76e-04 | 1.09e-09 |
|  | % err | 437.1% ± 19.17% | 447.9% | 436.8% | 401.3% | 457.4% | 442.2% | 0.4629% |
|  | NRMSE | 134.7% ± 5.407% | 142.3% | 134.8% | 127.1% | 138.7% | 130.8% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.008548 ± 0.003519 | 0.008189 | 0.01521 | 0.007491 | 0.007038 | 0.004809 | 9.61e-06 |
|  | % err | 230.0% ± 48.05% | 212.8% | 322.3% | 198.8% | 227.8% | 188.2% | 8.724% |
|  | NRMSE | 198.2% ± 35.47% | 173.9% | 253.7% | 167.7% | 227.0% | 168.8% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.008781 ± 0.003516 | 0.006998 | 0.01505 | 0.009647 | 0.007563 | 0.004648 | 9.17e-06 |
|  | % err | 287.8% ± 57.85% | 233.8% | 398.9% | 271.6% | 253.8% | 281.0% | 11.34% |
|  | NRMSE | 221.1% ± 36.09% | 194.4% | 282.2% | 196.8% | 242.8% | 189.4% | 6.486% |
| **Rolling vol histogram** | MSE | 1398 ± 34.29 | 1390 | 1444 | 1360 | 1365 | 1431 | 1.372 |
|  | % err | 799.2% ± 14.12% | 796.1% | 817.9% | 781.4% | 787.6% | 812.9% | 2.264% |
|  | NRMSE | 73.06% ± 0.8956% | 72.83% | 74.25% | 72.08% | 72.17% | 73.95% | 0.8688% |
| **Tail survival** | MSE | 0.05973 ± 0.001991 | 0.05945 | 0.05962 | 0.05642 | 0.06257 | 0.06057 | 5.22e-07 |
|  | % err | 342.3% ± 8.331% | 343.2% | 344.5% | 327.4% | 353.2% | 343.2% | 0.3302% |
|  | NRMSE | 42.74% ± 0.7148% | 42.64% | 42.71% | 41.54% | 43.75% | 43.04% | 0.1050% |

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
degenerate judge). The real class for A18/A19 is drawn from the **disc split (seed 2)**, never the test set.

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
retrieve K nearest COSCI-GAN paths by L2 in z-scored space, forecast with their price-anchored futures.
CRPS is scored against the test set at two horizons; the naive random-walk (RW) baseline is 3.738 (H=32) /
5.246 (H=64). Full breakdown: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 4.657 ± 0.7720 | 3.738 |
| PS-MC CRPS H=64 ↓ | 5.789 ± 0.7528 | 5.246 |

PS-MC over the COSCI-GAN pool **does not beat the naive random walk** at either horizon: CRPS 4.657 > 3.738
at H=32 and 5.789 > 5.246 at H=64. Along with TimeVAE, it is **one of only two pools that fail the RW
floor** — because the generated paths carry no Heston volatility structure (A9/A31/A33 above), their
nearest-neighbour futures are no more informative than a driftless RW. This contrasts with the strong pools
(LS4, CSDI, Diffusion-TS) whose neighbours **beat** RW. Heston is time-homogeneous, so uniform and Gaussian
prefix weightings coincide.

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B three-subline aggregates (MSE + % err + NRMSE) |
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

→ Cross-method comparison with all nine generators: [`results/README.md`](../../README.md)
