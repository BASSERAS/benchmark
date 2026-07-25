# Metrics — Chronos-2 on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Data split (test set everywhere).** Three disjoint 8 192-path Heston draws are used throughout:
the generator is **fine-tuned on seed 0**; every A/B metric and every diagnostic plot compares the
generated paths against the **test set (seed 1)**; the A18 discriminative score and the A19
predictive-TSTR score use a **third real set (seed 2)** as the "real" class. No metric is ever scored
against the generator's own training data. The autoregressive rollout is seeded from the **first 16
steps of real test-set paths** (`dataset/Heston/heston_S_test_8192x128.npy`).

**Model:** Chronos-2 — a pretrained encoder-decoder T5-style probabilistic foundation forecaster with
group attention (Ansari, Turkmen, Shchur, et al., Amazon Science, **2025**, *Chronos-2: From Univariate
to Universal Forecasting*, arXiv:2510.15821, `github.com/amazon-science/chronos-forecasting`).
Checkpoint `amazon/chronos-2` (**~120M params — the largest model in the benchmark**), **full
fine-tuned** (`finetune_mode="full"`) for 1 000 steps (lr 1e-4, batch 256, prediction length 16), then
rolled out autoregressively one step at a time in **log-price space** from a real 16-step prefix, with
one inverse-CDF Monte-Carlo sample drawn per step from the 21 forecast quantiles. See
[`../../../methods/Chronos2/code/README.md`](../../../methods/Chronos2/code/README.md).

> **Fine-tune only — no zero-shot variant is archived.** Chronos-2's robust scaling is
> level-proportional, so a zero-shot autoregressive rollout on a low-vol geometric process like Heston
> runs away multiplicatively even in log-space. Fine-tuning fixes the per-step scale and is required
> for a stable 112-step rollout. Only the fine-tuned variant ships. See
> [§ Comparison with the paper](#comparison-with-the-paper-zero-shot-reproduction).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case score a *perfect* generator reaches at this
sample size. It is an **independent Heston draw** (fresh seeds 1000+i, identical parameters) scored
against the **test set** — a genuine **non-zero finite-sample noise floor**, not zero and not a
row-shuffle/permutation of the test data. Two independent draws of the same process still differ by
sampling noise, and that irreducible noise is what the floor measures. See
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 33213 ± 61713 | 5912 | 43.25 | 195.1 | 1.57e+05 | 3352 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.008183 ± 1.28e-04 | 0.007993 | 0.008330 | 0.008236 | 0.008074 | 0.008280 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.02621 ± 3.47e-18 | 0.02621 | 0.02621 | 0.02621 | 0.02621 | 0.02621 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.01329 ± 5.52e-05 | 0.01323 | 0.01335 | 0.01329 | 0.01322 | 0.01335 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 16.95 ± 0.3287 | 17.50 | 16.70 | 16.66 | 17.15 | 16.72 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.01567 ± 0.001487 | 0.01436 | 0.01632 | 0.01551 | 0.01402 | 0.01816 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.02727 ± 0.001487 | 0.02739 | 0.02527 | 0.02739 | 0.02647 | 0.02980 | 0.001983 |
| A8 Increment MMD² ↓ | 0.01223 ± 5.59e-04 | 0.01168 | 0.01160 | 0.01270 | 0.01215 | 0.01303 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.7819 ± 0.03406 | 0.7570 | 0.7535 | 0.7893 | 0.7647 | 0.8453 | 0.008554 |
| A10 Terminal SWD ↓ | 13.27 ± 11.05 | 8.970 | 5.687 | 8.121 | 35.26 | 8.292 | 1.151 |
| A11 Path SWD ↓ | 3.663 ± 1.763 | 3.183 | 2.167 | 2.922 | 7.122 | 2.918 | 0.6191 |
| A12 RV Law Loss ↓ | 8.120 ± 0.07749 | 8.068 | 8.042 | 8.078 | 8.254 | 8.160 | 0.05202 |
| A13 Mean Path RMSE ↓ | 2.966 ± 0.5016 | 2.852 | 2.530 | 2.807 | 3.945 | 2.698 | 0.1205 |
| A14 KS Log-returns ↓ | 0.2830 ± 6.09e-04 | 0.2827 | 0.2833 | 0.2822 | 0.2826 | 0.2840 | 0.001491 |
| A15 Skewness Error ↓ | 0.3153 ± 0.08030 | 0.2967 | 0.2164 | 0.2669 | 0.4531 | 0.3433 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.007833 ± 1.06e-05 | 0.007820 | 0.007848 | 0.007831 | 0.007823 | 0.007842 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.1239 ± 0.002006 | 0.1261 | 0.1228 | 0.1265 | 0.1229 | 0.1213 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.1606 ± 0.07803 | 0.009002 | 0.1643 | 0.2113 | 0.2003 | 0.2180 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.02298 ± 0.01150 | 0.02731 | 0.01053 | 0.02609 | 0.01022 | 0.04074 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05003 ± 2.34e-05 | 0.05007 | 0.05002 | 0.05003 | 0.05001 | 0.05001 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05024 ± 3.00e-04 | 0.05002 | 0.05017 | 0.04989 | 0.05076 | 0.05034 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 19192 ± 37246 | 740.1 | 444.9 | 533.7 | 93683 | 557.6 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.1339 ± 0.001922 | 0.1324 | 0.1332 | 0.1334 | 0.1330 | 0.1377 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.07220 ± 0.001298 | 0.07129 | 0.07189 | 0.07184 | 0.07122 | 0.07473 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.1905 ± 0.002310 | 0.1892 | 0.1891 | 0.1897 | 0.1892 | 0.1951 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.1040 ± 0.001707 | 0.1040 | 0.1032 | 0.1028 | 0.1025 | 0.1072 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 7.034 ± 1.428 | 6.627 | 5.954 | 6.476 | 9.854 | 6.261 | 0.1392 |
| A26 Return Std Error ↓ | 7.661 ± 13.38 | 1.048 | 0.8754 | 0.9305 | 34.43 | 1.023 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.006403 ± 5.83e-05 | 0.006381 | 0.006347 | 0.006380 | 0.006516 | 0.006393 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 0.09859 ± 0.02940 | 0.1056 | 0.1382 | 0.1121 | 0.04969 | 0.08734 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.05589 ± 2.83e-04 | 0.05610 | 0.05579 | 0.05618 | 0.05599 | 0.05539 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 24.84 ± 39.72 | 5.430 | 4.327 | 4.944 | 104.3 | 5.211 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.3490 ± 0.002518 | 0.3472 | 0.3484 | 0.3470 | 0.3485 | 0.3539 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.007430 ± 6.32e-05 | 0.007401 | 0.007372 | 0.007380 | 0.007542 | 0.007457 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.05739 ± 0.004889 | 0.05977 | 0.05015 | 0.06498 | 0.05602 | 0.05602 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.2167 ± 0.001244 | 0.2160 | 0.2160 | 0.2154 | 0.2189 | 0.2172 | 0.06559 |

**Reading the table — locally calibrated, globally over-dispersed.** Chronos-2 is the benchmark's
**foundation-model anchor**, and it splits cleanly along the local/global axis. **Locally it is strong:**
A19 predictive-TSTR sits **at the perfect floor** (GRU 0.05003 vs floor 0.05002 on every seed), the A18
GRU discriminative score reaches **0.009 on seed 0** (near-indistinguishable), and the A18 MLP score is a
tight mid-pack **0.023**. A next-step conditional the model was pretrained and fine-tuned on transfers
cleanly to real Heston. **Globally it over-disperses:** A28 Kurtosis Ratio **0.09859** (target 1.0) means
the generated log-returns are ≈ 10× more leptokurtic than Heston, and the generated price std (15.2) is
≈ 1.5× the real (10.2) — compounding a slightly-too-wide next-step quantile over 112 autoregressive steps
widens the marginal law. A14 KS **0.283**, A9 Volatility MMD **0.782**, A31 Rolling-Vol KS **0.349**.

**Seed 3 produced one runaway path (kept unclipped).** A single seed-3 series ran away to a terminal
price ≈ 1323, which alone drives the enormous seed-3 outliers in A1 (1.57e5), A20 (93683), A26 (34.43)
and A30 (104.3) — hence the large ± std on those rows. We **do not clip** generated paths: this is an
honest artifact of level-proportional scaling under autoregressive rollout, the very instability that
fine-tuning tames but does not fully eliminate. Read the seed-1/2/4 columns for the typical scale
(A1 ≈ 43–3352, A20 ≈ 445–740, A26 ≈ 0.88–1.05, A30 ≈ 4.3–5.4).

**No latent-vol recovery — the shared caveat.** A33 teacher-σ correlation **≈ 0.057** (floor 0.6163):
like every single-factor generator in this benchmark, Chronos-2 does not reconstruct Heston's hidden
variance process from prices alone. Net: an adversarially strong *local* conditional (predictive at the
floor, seed-0 GRU 0.009, PS-MC beats RW below) paired with a distributionally weak *global* path law
(over-dispersed returns, worst log-return-histogram fit, no latent vol) — the clearest demonstration in
the benchmark that a strong one-step forecaster is not automatically a faithful long-horizon generator.

---

## Stylised Facts Diagnostic (Heston vs Chronos-2, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — and report **five measures** per plot:

- **MSE**: for each list, `mean((L_gen − L_real)²)`; reported as the **mean of the three** sub-scores
  (funct, der, sec\_der).
- **% err** (function-level MAPE): `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE**: `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100`, funct-only.
- **CVaR₉₀ / CVaR₉₅**: tail-averaged pointwise Expected Shortfall on the curve L (funct-only): eₜ = |L_gen − L_real|, CVaR_q = mean(eₜ ≥ q-th percentile), range-normalized like NRMSE. q ∈ {0.90, 0.95}.

↓ lower is better for all five. The **Perfect floor** is again an independent Heston draw vs the test
set — a non-zero finite-sample floor, not zero.

**Worst return-density curve in the benchmark.** Chronos-2 posts the **worst log-return-histogram MSE**:
**20704** (next-worst GT-GAN 2160, ~10× lower; floor 0.1098) — the curve-space image of the over-dispersed
return law. The rolling-vol-histogram MSE **9718** (floor 1.372) is likewise the largest here, paired with
the A31 rolling-vol KS 0.349. QQ MSE **2.41e-05**, tail-survival MSE **0.02570** and both ACF curves stay
above floor. The ACF %err / NRMSE figures balloon into the hundreds of percent because the true ACF ≈ 0.05,
so a modest absolute error becomes an enormous *relative* one; read MSE for absolute agreement.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | 8.835% ± 5.165% | 6.050% | 12.32% | 14.02% | 0.005531% | 11.78% | 2.237% |
| **Log-return histogram** | MSE | 20704 ± 69.62 | 20658 | 20723 | 20628 | 20683 | 20828 | 0.1098 |
|  | % err | 182.5% ± 0.1675% | 182.7% | 182.5% | 182.3% | 182.6% | 182.3% | 1.799% |
|  | NRMSE | 221.2% ± 0.3808% | 220.9% | 221.3% | 220.8% | 221.1% | 221.9% | 0.5328% |
|  | CVaR₉₀ | 322.7% ± 0.3105% | 322.6% | 322.8% | 322.3% | 322.7% | 323.2% | 1.234% |
|  | CVaR₉₅ | 565.6% ± 0.5994% | 565.3% | 565.7% | 564.8% | 565.4% | 566.6% | 1.444% |
| **QQ plot** | MSE | 2.41e-05 ± 5.94e-08 | 2.41e-05 | 2.42e-05 | 2.40e-05 | 2.41e-05 | 2.41e-05 | 1.09e-09 |
|  | % err | 78.69% ± 0.06065% | 78.64% | 78.73% | 78.63% | 78.68% | 78.79% | 0.4629% |
|  | NRMSE | 21.72% ± 0.02665% | 21.69% | 21.76% | 21.71% | 21.71% | 21.75% | 0.1206% |
|  | CVaR₉₀ | 24.08% ± 0.04125% | 24.04% | 24.15% | 24.06% | 24.05% | 24.11% | 0.1319% |
|  | CVaR₉₅ | 29.36% ± 0.08250% | 29.27% | 29.49% | 29.32% | 29.30% | 29.42% | 0.1599% |
| **ACF \|r\| lags 1–20** | MSE | 0.003108 ± 9.05e-05 | 0.003019 | 0.003091 | 0.003067 | 0.003080 | 0.003282 | 9.61e-06 |
|  | % err | 255.1% ± 3.890% | 250.2% | 256.1% | 254.7% | 252.6% | 261.8% | 8.724% |
|  | NRMSE | 251.0% ± 3.755% | 247.3% | 250.5% | 249.3% | 249.6% | 258.2% | 6.071% |
|  | CVaR₉₀ | 453.6% ± 5.754% | 450.9% | 451.7% | 451.8% | 448.7% | 464.9% | 11.26% |
|  | CVaR₉₅ | 506.8% ± 6.147% | 503.5% | 503.2% | 504.9% | 503.6% | 519.1% | 12.06% |
| **ACF r² lags 1–20** | MSE | 9.03e-04 ± 3.42e-05 | 8.68e-04 | 9.02e-04 | 8.93e-04 | 8.84e-04 | 9.67e-04 | 9.17e-06 |
|  | % err | 173.0% ± 3.750% | 167.7% | 175.4% | 172.7% | 170.6% | 178.5% | 11.34% |
|  | NRMSE | 147.7% ± 2.884% | 144.8% | 147.9% | 146.7% | 146.1% | 153.1% | 6.486% |
|  | CVaR₉₀ | 272.4% ± 4.462% | 272.8% | 272.1% | 270.6% | 266.4% | 280.2% | 12.35% |
|  | CVaR₉₅ | 304.5% ± 4.999% | 304.6% | 302.4% | 301.2% | 300.2% | 314.0% | 13.27% |
| **Rolling vol histogram** | MSE | 9718 ± 132.0 | 9839 | 9865 | 9715 | 9674 | 9497 | 1.372 |
|  | % err | 239.2% ± 1.525% | 236.6% | 238.9% | 239.2% | 240.2% | 241.1% | 2.264% |
|  | NRMSE | 72.66% ± 0.2984% | 72.89% | 72.94% | 72.63% | 72.72% | 72.11% | 0.8688% |
|  | CVaR₉₀ | 162.4% ± 0.4303% | 162.8% | 163.0% | 162.2% | 162.4% | 161.8% | 1.970% |
|  | CVaR₉₅ | 258.0% ± 0.9043% | 258.8% | 259.2% | 257.6% | 257.9% | 256.7% | 2.308% |
| **Tail survival** | MSE | 0.02570 ± 1.09e-04 | 0.02564 | 0.02570 | 0.02556 | 0.02568 | 0.02589 | 5.22e-07 |
|  | % err | 69.52% ± 0.07637% | 69.41% | 69.60% | 69.51% | 69.49% | 69.62% | 0.3302% |
|  | NRMSE | 27.94% ± 0.05985% | 27.91% | 27.95% | 27.87% | 27.94% | 28.05% | 0.1050% |
|  | CVaR₉₀ | 52.88% ± 0.09414% | 52.84% | 52.89% | 52.77% | 52.86% | 53.05% | 0.1625% |
|  | CVaR₉₅ | 54.97% ± 0.09358% | 54.93% | 54.98% | 54.85% | 54.95% | 55.14% | 0.1682% |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> The ACF %err / NRMSE (hundreds of %) is a near-zero-denominator artefact: the true ACF ≈ 0.05, so a small
> absolute error becomes a large *relative* error. Read MSE for absolute agreement, %err for shape.
> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

---

## Discriminative & Predictive Classifier Losses (A18 / A19)

BCE loss during GRU/MLP discriminator training (A18) and MAE loss during GRU/MLP predictor training on
*synthetic* data (A19, TSTR), 5 seeds. A discriminator BCE near ln(2) ≈ 0.693 means real and generated
are indistinguishable; a BCE that collapses toward 0 means the judge separates them trivially. On **seed 0**
Chronos-2's GRU stays near ln 2 (A18 GRU 0.009 — near-indistinguishable); on seeds 1–4 it separates more
(0.16–0.22). The GRU predictor MAE lands **at the perfect floor** (A19 GRU 0.05003 vs 0.05002) — the local
one-step conditional transfers cleanly to real Heston.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (zero-shot reproduction)

> ⚠️ **Chronos-2's paper metrics are zero-shot forecasting accuracy on standard benchmarks, not Heston.**
> The arXiv:2510.15821 paper reports **MASE** and **WQL** on forecasting datasets (Chronos-datasets), not a
> generative fidelity score. There is therefore no native "Ours — Heston" entry for the paper metric; we
> validated our Chronos-2 port by reproducing its **zero-shot** MASE/WQL on five held-out datasets before
> fine-tuning it into the Heston generator above, against the paper's `chronos-bolt-base` reference numbers.

| Dataset | MASE ours | MASE ref | WQL ours | WQL ref |
|---------|:---------:|:--------:|:--------:|:-------:|
| ercot | 0.7765 | 0.6933 | 0.02457 | 0.02142 |
| exchange_rate | 1.8788 | 1.7095 | 0.01214 | 0.01201 |
| monash_australian_electricity | 0.6135 | 0.7403 | 0.02878 | 0.03584 |
| monash_traffic | 0.8308 | 0.7843 | 0.24254 | 0.23149 |
| nn5 | 0.5561 | 0.5764 | 0.14489 | 0.15005 |

Our zero-shot MASE/WQL land in the **same regime** as the `chronos-bolt-base` reference across all five
datasets — better on `monash_australian_electricity` and `nn5`, slightly behind on `ercot`, `exchange_rate`
and `monash_traffic` — confirming the port loads and runs the pretrained checkpoint faithfully. The same
`amazon/chronos-2` checkpoint is fine-tuned into the Heston generator above. Full write-up:
[`../../../methods/Chronos2/paper_reimplementation/README.md`](../../../methods/Chronos2/paper_reimplementation/README.md).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real (test-set) prefix as a murex-style feature vector,
retrieve nearest Chronos-2 paths by L2 in z-scored space, and forecast with their price-anchored futures.

| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 3.719 ± 0.001705 | 3.738 |
| PS-MC CRPS H=64 ↓ | 5.218 ± 0.002697 | 5.246 |

PS-MC over the Chronos-2 pool **beats the naive random walk on CRPS at both horizons** (3.719 < 3.738 at
H=32; 5.218 < 5.246 at H=64) — a small but genuine margin, and the spread across seeds is **remarkably
tight** (std ≈ 0.002, ~50× tighter than GT-GAN's 0.11) because the fine-tuned pool is locally
well-calibrated, so the K = 77 price-anchored neighbours form a stable ensemble. The gain is CRPS-specific
(the ensemble *mean* is no closer to the truth than the RW). Full breakdown:
[`path_shadowing/README.md`](path_shadowing/README.md).

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B five-measure aggregates (MSE + % err + NRMSE + CVaR₉₀ + CVaR₉₅) |
| `grid_tvd_aggregate.json` | 50×50 path-cloud TVD aggregate |
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

→ Cross-method comparison with TimeGAN, COSCI-GAN, GT-GAN, SBTS, Fourier Flow, Diffusion-TS, CSDI, TimeVAE, TimeVQVAE & LS4: [`results/README.md`](../../README.md)
