# Metrics — Diffusion-TS on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** Diffusion-TS (Yuan & Qiao, ICLR 2024), a non-autoregressive DDPM with an interpretable
seasonal-trend transformer decoder that predicts the clean signal $\hat{x}_0$ directly. Loss =
reweighted L1 + Fourier-FFT reconstruction, cosine β over 500 diffusion steps. The canonical run uses
the **`mujoco`** preset (encoder/decoder depth 3, 12 000 steps, 544 147 params), selected by a Context-FID
smoke test (0.7367, vs 2.3192 `etth` / 36.05 `stocks`). See
[`../../../methods/DiffusionTS/code/README.md`](../../../methods/DiffusionTS/code/README.md).

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
| A1 Kurtosis Error ↓ | 0.4242 ± 0.02303 | 0.4235 | 0.3935 | 0.4649 | 0.4177 | 0.4214 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.006902 ± 1.57e-04 | 0.006982 | 0.007108 | 0.006664 | 0.006971 | 0.006786 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.01032 ± 1.75e-04 | 0.01036 | 0.01042 | 0.01012 | 0.01058 | 0.01014 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.006781 ± 1.50e-04 | 0.006847 | 0.006982 | 0.006561 | 0.006852 | 0.006662 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 3.047 ± 0.2789 | 2.844 | 3.402 | 2.995 | 2.670 | 3.325 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.004476 ± 8.48e-04 | 0.004700 | 0.003512 | 0.003644 | 0.005848 | 0.004675 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.003676 ± 0.001070 | 0.003800 | 0.002500 | 0.002506 | 0.005278 | 0.004298 | 0.001983 |
| A8 Increment MMD² ↓ | 0.01109 ± 7.52e-04 | 0.01176 | 0.01203 | 0.009961 | 0.01062 | 0.01107 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.3846 ± 0.02464 | 0.4163 | 0.4075 | 0.3522 | 0.3632 | 0.3840 | 0.008554 |
| A10 Terminal SWD ↓ | 1.684 ± 0.3010 | 1.625 | 1.480 | 1.313 | 2.187 | 1.815 | 1.151 |
| A11 Path SWD ↓ | 1.212 ± 0.1556 | 1.281 | 1.037 | 1.034 | 1.437 | 1.270 | 0.6191 |
| A12 RV Law Loss ↓ | 2.274 ± 0.04910 | 2.292 | 2.343 | 2.204 | 2.297 | 2.234 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.4399 ± 0.2584 | 0.3142 | 0.6895 | 0.8016 | 0.2562 | 0.1379 | 0.1205 |
| A14 KS Log-returns ↓ | 0.06048 ± 0.001904 | 0.06076 | 0.06349 | 0.05761 | 0.06083 | 0.05971 | 0.001491 |
| A15 Skewness Error ↓ | 0.06445 ± 0.03230 | 0.07100 | 0.02133 | 0.03854 | 0.07706 | 0.1143 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.003073 ± 8.32e-05 | 0.003100 | 0.003196 | 0.002953 | 0.003101 | 0.003014 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.04436 ± 0.007030 | 0.03638 | 0.05078 | 0.05090 | 0.04846 | 0.03528 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.08987 ± 0.1524 | 0.03708 | 0.004425 | 0.008697 | 0.005340 | 0.3938 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.02426 ± 0.03140 | 0.005951 | 0.009002 | 0.003814 | 0.08651 | 0.01602 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05112 ± 1.22e-04 | 0.05123 | 0.05095 | 0.05115 | 0.05100 | 0.05126 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05112 ± 1.21e-04 | 0.05108 | 0.05122 | 0.05117 | 0.05091 | 0.05124 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 44.18 ± 10.64 | 33.69 | 41.70 | 36.37 | 63.86 | 45.29 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.01812 ± 0.002352 | 0.01788 | 0.01389 | 0.01844 | 0.02089 | 0.01951 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.01587 ± 0.002662 | 0.01479 | 0.01112 | 0.01724 | 0.01808 | 0.01809 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.002410 ± 0.001465 | 0.002879 | 0.004665 | 0.001428 | 0.002751 | 3.27e-04 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.007895 ± 0.002645 | 0.004063 | 0.01104 | 0.01044 | 0.005909 | 0.008032 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.7610 ± 0.4617 | 0.5277 | 1.226 | 1.375 | 0.5129 | 0.1636 | 0.1392 |
| A26 Return Std Error ↓ | 0.3107 ± 0.009292 | 0.3141 | 0.3245 | 0.2965 | 0.3128 | 0.3058 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.003240 ± 8.19e-05 | 0.003269 | 0.003357 | 0.003126 | 0.003278 | 0.003172 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 1.903 ± 0.2558 | 1.902 | 1.462 | 2.170 | 2.142 | 1.838 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.04883 ± 0.001266 | 0.04931 | 0.05074 | 0.04704 | 0.04913 | 0.04792 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.365 ± 0.2012 | 1.349 | 1.101 | 1.217 | 1.679 | 1.477 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.2576 ± 0.007919 | 0.2557 | 0.2716 | 0.2498 | 0.2602 | 0.2508 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 0.001587 ± 3.82e-05 | 0.001584 | 0.001564 | 0.001589 | 0.001656 | 0.001542 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.001823 ± 0.004419 | -0.001107 | -0.004280 | 0.008148 | 0.001159 | 0.005192 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.09645 ± 9.09e-04 | 0.09667 | 0.09806 | 0.09535 | 0.09613 | 0.09601 | 0.06559 |

**Reading the table.** Diffusion-TS wins **2 of the 36 A-metric rows**, both in short-range temporal
structure: **A23 ACF |r| lag-1 (0.002410)** and **A24 ACF r² lag-1 (0.007895)** — it matches the one-lag
ARCH autocorrelation of Heston returns better than any other generator, and A23 actually lands *below* its
own perfect floor (0.002652) on the seed mean, i.e. within finite-sample noise of a real independent draw.
Its broader marginal fidelity is also strong for a diffusion model: A2–A4 tail-quantile errors (~0.007–0.010),
A16 QQ RMSE (0.0031) and A15 skewness (0.064) are small and tight across seeds. Two caveats replace the
old claims: A1 kurtosis error is **0.424** — good, but **CSDI now wins A1 outright (0.0954)**, so Diffusion-TS
is no longer the marginal-tail leader; and A28 kurtosis ratio **1.90** is closer to the ideal 1.0 than the
flat generators but **CSDI wins A28 (0.871)**, so Diffusion-TS is second-best there, not best. Where it is
**noisy** is A18 GRU discriminative (0.090 ± 0.152) — a high-variance judge that on seed 4 cleanly separates
real from fake (0.394) and on others is fooled (≈0.005); the MLP judge is far tighter (0.024). As with every
method, A33 teacher-sigma correlation ≈ 0 — no generator recovers the latent Heston variance process from
prices alone (floor 0.616, unreachable without the hidden state).

---

## Stylised Facts Diagnostic (Heston vs Diffusion-TS, seed 0)

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
- **% err row** (scale-aware ε-floor MAPE): mean(|L\_gen − L\_real| / (|L\_real| + ε)) × 100 with
  ε = 1e-3·(max|L\_real| + 1e-12), averaged over the three lists — except **Tail survival**, whose % err and
  NRMSE use the **function curve only** (its near-zero survival tail makes the derivative MAPE explode).
- **NRMSE row**: sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100, averaged over
  the three lists (funct-only for Tail survival).

↓ lower is better for all three rows. **Perfect floor** is the non-zero real-vs-test value an independent
Heston draw reaches — identical across methods.

<!-- ===== PER-METHOD B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 4.883 ± 0.5079 | 5.006 | 5.702 | 4.145 | 4.921 | 4.641 | 0.1098 |
|  | % err | 313.8% ± 111.7% | 443.7% | 195.3% | 445.1% | 287.8% | 197.3% | 290.3% |
|  | NRMSE | 26.20% ± 2.004% | 28.85% | 27.77% | 23.08% | 25.28% | 26.04% | 17.81% |
| **QQ plot** | MSE | 3.48e-06 ± 1.75e-07 | 3.53e-06 | 3.73e-06 | 3.22e-06 | 3.56e-06 | 3.35e-06 | 1.09e-09 |
|  | % err | 31.90% ± 0.7667% | 31.69% | 32.95% | 30.87% | 32.59% | 31.39% | 16.51% |
|  | NRMSE | 6.488% ± 0.1372% | 6.486% | 6.369% | 6.437% | 6.751% | 6.397% | 0.3436% |
| **ACF \|r\| lags 1–20** | MSE | 1.72e-04 ± 4.79e-05 | 1.55e-04 | 9.01e-05 | 1.83e-04 | 2.33e-04 | 1.98e-04 | 9.61e-06 |
|  | % err | 215.5% ± 57.66% | 155.4% | 155.1% | 244.9% | 307.7% | 214.3% | 114.3% |
|  | NRMSE | 117.1% ± 20.99% | 92.80% | 91.21% | 132.8% | 141.4% | 127.2% | 43.89% |
| **ACF r² lags 1–20** | MSE | 1.32e-04 ± 4.43e-05 | 1.11e-04 | 5.68e-05 | 1.48e-04 | 1.85e-04 | 1.57e-04 | 9.17e-06 |
|  | % err | 365.5% ± 60.60% | 329.5% | 381.9% | 452.7% | 390.6% | 272.9% | 381.5% |
|  | NRMSE | 97.45% ± 20.23% | 77.18% | 69.08% | 114.1% | 118.0% | 108.9% | 34.19% |
| **Rolling vol histogram** | MSE | 220.2 ± 15.36 | 217.2 | 248.3 | 205.7 | 222.3 | 207.3 | 1.372 |
|  | % err | 167.1% ± 16.21% | 191.0% | 179.6% | 146.4% | 163.3% | 155.1% | 127.9% |
|  | NRMSE | 35.77% ± 1.403% | 36.11% | 38.37% | 34.83% | 34.60% | 34.95% | 16.66% |
| **Tail survival** | MSE | 0.002258 ± 2.00e-04 | 0.002299 | 0.002595 | 0.002002 | 0.002274 | 0.002121 | 5.22e-07 |
|  | % err | 27.97% ± 0.8384% | 28.21% | 29.27% | 26.82% | 28.19% | 27.35% | 0.3256% |
|  | NRMSE | 8.301% ± 0.3648% | 8.383% | 8.907% | 7.823% | 8.337% | 8.053% | 0.1050% |

Diffusion-TS wins **none of the 6 B-plots** — the MSE headline on every curve is well above the LS4 winner.
Its best relative showing is the **QQ curve** (MSE 3.48e-06, NRMSE 6.5 %): the quantile shape is decent, but
the log-return-histogram and rolling-vol-histogram MSEs (4.88, 220.2) are an order of magnitude worse than
the leaders, reflecting the same variance-inflation the A9/A26/A31 vol metrics show.

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
are indistinguishable. The real class for A18/A19 is drawn from the **disc split (seed 2)**, never the test set.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (paper's own metrics: Context-FID, Correlational, Discriminative, Predictive)

This section uses **the paper's own four metrics** (Yuan & Qiao, Table 1) — Context-FID
(`Utils/context_fid`), Correlational (`Utils/cross_correlation`), Discriminative
(`Utils/discriminative_metric`), Predictive (`Utils/predictive_metric`) — the released code, unchanged.
Three columns:

- **Paper (Table 1)** — the published Diffusion-TS numbers on the paper's Stocks dataset.
- **Ours — Stocks** — the *released* model + `Config/stocks.yaml` run verbatim on the paper's own dataset
  (6 features, seq_len 24, 3 662 windows); validates the port independently of Heston. Full write-up:
  [`../../../methods/DiffusionTS/paper_reimplementation/`](../../../methods/DiffusionTS/paper_reimplementation/).
- **Ours — Heston** — the **same four metric functions** applied to our 5-seed Diffusion-TS paths on the
  Heston benchmark dataset. Prices are placed on the paper's [0,1] scale by a single global MinMax fit on
  the real Heston prices (applied to both real and synthetic). Source:
  [`../../../methods/DiffusionTS/paper_reimplementation/results/heston_paper_metrics.json`](../../../methods/DiffusionTS/paper_reimplementation/results/heston_paper_metrics.json).

| Metric (paper's own) | Paper (Table 1, Stocks) | Ours — Stocks (paper dataset) | Ours — Heston |
|----------------------|:-----------------------:|:-----------------------------:|:-------------:|
| Context-FID ↓ | 0.147 ± 0.025 | **0.2024 ± 0.0245** | **0.0307 ± 0.0077** |
| Correlational ↓ | 0.004 ± 0.001 | **0.0106 ± 0.0000** [1] | **≈ 6×10⁻⁹** [2] (degenerate — real: A21 ACF-abs 0.0181 ± 0.0024) |
| Discriminative ↓ | 0.067 ± 0.015 | **0.0914 ± 0.0178** | **0.0000** [3] (degenerate — real: A18 GRU 0.0899 ± 0.152) |
| Predictive ↓ | 0.036 ± 0.000 | **0.0371 ± 0.00005** | **0.0653** [4] (degenerate — real: A19 GRU 0.0511 ± 0.0001) |

> The four "≈ 0 / ± 0.0000" cells above are **genuinely computed values**, not placeholders or a stalled
> metric. Their exact per-run/per-seed numbers are in
> [`heston_paper_metrics.json`](../../../methods/DiffusionTS/paper_reimplementation/results/heston_paper_metrics.json)
> and [`stocks_comparison.json`](../../../methods/DiffusionTS/paper_reimplementation/results/stocks_comparison.json).
> They are near-zero for the **structural** reasons below, verified in the released metric code:
>
> - **[1] Correlational std = 0 on Stocks is expected, not stalled.** `CrossCorrelLoss` is *deterministic*
>   given a fixed (real, generated) pair, and we score **one** saved generated sample
>   (`OUTPUT/stock/ddpm_fake_stock.npy`) 5×, so all 5 runs are byte-identical (`0.01058272086083889`,
>   `ci95 = 0.0`). The paper's ±0.001 comes from **re-sampling the model** each run; we sampled once.
> - **[2] Correlational ≈ 6×10⁻⁹ on Heston is the SAME class of univariate degeneracy as [3]/[4]** (different
>   code path, same "undefined on 1 feature" outcome). The metric scores cross-**feature** correlation error;
>   Heston here is a **single** feature `(N,T,1)`, so `tril_indices(1,1)` leaves only the feature-with-itself
>   term, and `cacf_torch` **standardizes** first → the lag-0 self-correlation is **identically 1.0** for any
>   real and any fake. Mathematically incapable of being nonzero on one feature — verified by reading + running
>   `code/reference/Utils/cross_correlation.py`. **Our comparable
>   number:** the benchmark's own **temporal** correlation metrics (defined on univariate — lag-autocorrelation,
>   not cross-feature) rate the same paths at **A21 ACF-abs 0.0181 ± 0.0024**, A22 ACF-sq 0.0159 ± 0.0027,
>   A23 lag-1 ACF-abs 0.0024 ± 0.0015 (tables above).
> - **[3] Discriminative = 0.0000 on Heston is a metric-code degeneracy on univariate data, NOT a quality
>   signal.** `Utils/discriminative_metric.py` line 74 sets `hidden_dim = int(dim/2)`; for univariate Heston
>   (`dim = 1`) that is `int(1/2) = 0` → `GRUCell(num_units=0)` → a zero-capacity judge with no hidden state →
>   constant output → exactly 0.5 accuracy on the balanced split → |0.5 − 0.5| = 0 on **all 5 seeds** (0-std is
>   impossible for a working judge). On Stocks (6 features) `int(6/2) = 3` works — the collapse is
>   univariate-specific, verified in `code/reference/Utils/discriminative_metric.py`. **Our comparable number:**
>   the benchmark's own judge (A18, hidden dim floored at `max(8, n_features·8)`) rates the same paths at
>   **GRU 0.0899 ± 0.152**, MLP 0.0243 ± 0.0314 — moderately distinguishable, which the paper metric's 0.0 hides.
> - **[4] Predictive = 0.0653 is the SAME `hidden_dim = int(dim/2) = 0` degeneracy**, in
>   `Utils/predictive_metric.py` line 52. For `dim = 1` the zero-capacity GRU predictor emits a constant, so
>   0.0653 is a mean-absolute-deviation artifact, not a synthetic-trained one-step forecast. **Our comparable
>   number:** the benchmark's own predictor (A19, floored hidden dim) gives a genuine TSTR MAE of
>   **GRU 0.0511 ± 0.0001**, MLP 0.0511 ± 0.0001 on the same paths.

**Stocks — same-code reproduction, honest gap.** The hyperparameters were run **verbatim** from the official
`Config/stocks.yaml` (n_layer_enc/dec = 2, d_model = 64, timesteps = 500, l1 loss, cosine β, base_lr 1e-5,
ema decay 0.995, 10000 epochs, batch 64 — see the paper_reimplementation README). Predictive 0.0371 vs 0.036
is a bulls-eye, proving the pipeline is correct. Context-FID 0.2024 vs 0.147 and Discriminative 0.0914 vs
0.067 land slightly above (worse) the paper — a **reproduction gap**, not a hyperparameter error: we score a
**single** milestone-10 EMA checkpoint and a **single** DDPM draw, whereas the paper reports its best across
re-sampling. We did **not** re-sample to chase the number (see [1]).

**Heston — read the caveat.** The Heston column (Context-FID 0.031, Correlational ≈ 0, Discriminative 0.000,
Predictive 0.065) is **not** a super-paper result. **Three of the four are degenerate on univariate data** (only Context-FID is genuine — TS2Vec embeds to 320-dim
regardless of feature count) for reasons [2]–[4] above. The **real** numbers come from the benchmark's
own metrics that stay defined on univariate data: correlation structure **A21 ACF-abs 0.0181 ± 0.0024**,
distinguishability **A18 disc GRU 0.0899 ± 0.152**, one-step forecast **A19 pred GRU 0.0511 ± 0.0001** (tables
above). Context-FID is lower than Stocks only because one smooth feature is easier to match in TS2Vec space
than 6-feature Stocks. Full discussion:
[`../../../methods/DiffusionTS/paper_reimplementation/README.md`](../../../methods/DiffusionTS/paper_reimplementation/README.md).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K nearest Diffusion-TS paths by L2 in z-scored space, forecast with their price-anchored
futures. CRPS is scored against the test set at two horizons; the naive random-walk (RW) baseline is
3.738 (H=32) / 5.246 (H=64). Full analysis: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 2.717 ± 0.002200 | 3.738 |
| PS-MC CRPS H=64 ↓ | 3.804 ± 0.007848 | 5.246 |

PS-MC **beats the naive RW on CRPS** at both horizons (2.717 < 3.738 at H=32; 3.804 < 5.246 at H=64), on all
5 seeds. At H=32 its CRPS (2.717) is effectively **tied with CSDI (2.718)** and sits just behind the best pool,
**LS4 (2.704)**, making Diffusion-TS one of the three strongest forecasting pools; at H=64 (3.804) it slips
slightly behind both LS4 (3.763) and CSDI (3.776). Heston is time-homogeneous, so the uniform and Gaussian
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
| `path_shadowing/` | Path-shadowing MC forecasts |

→ Cross-method comparison with all nine generators: [`results/README.md`](../../README.md)
