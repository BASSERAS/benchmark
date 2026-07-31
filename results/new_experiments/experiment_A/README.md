# Experiment A — Delayed Drawdown Memory: cross-method comparison

**2 methods · 5 seeds each · 8192 × 128 paths per seed · shared seeds 0–4 · A100-SXM4-80GB · 2026-07-31**

| | CSDI (Diffusion) | LS4 (VAE) |
|---|---|---|
| Source | `methods/CSDI` (released `base.yaml`) | `methods/LS4` (released `solar_weekly` preset) |
| Trainable parameters | 412,945 | 2,146,857 |
| Epochs | 200 | 100 |
| Training, mean per seed | 2387 s (2092–3290) | 1643 s (959–2349) |
| Generation, mean per seed | 15.6 s (10.3–18.6) | 9.1 s (8.5–10.6) |
| Preprocessing | affine standardization of raw prices | affine standardization of raw prices |
| Per-method README | [`CSDI/README.md`](CSDI/README.md) | [`LS4/README.md`](LS4/README.md) |

Neither method uses a log-return transform, a quantile map, or **path shadowing**. Both were
trained on `train.npy` only.

**Verdict, one line: LS4 wins Experiment A, and not narrowly.** Across the protocol evaluator
LS4 takes 17 metrics, CSDI 1, with 3 ties and 9 rows that are constants rather than contests.
On the standard battery the margin is wider still: A1–A34 goes 32–2 to LS4, and the curve-shape
suite goes 31–0. The single metric CSDI wins outright on the protocol suite is
`generated_memory.future_rv_no_hit_mean`.

**But neither method solves the experiment.** On the four PRIMARY metrics the protocol PDF
designates, the best method is still 2.6× to 49× worse than a fresh draw from the true DGP. The
comparison below is between two failures of different magnitude, not between a success and a
failure.

---

## 1. PDF metrics — protocol evaluator

Produced by `dataset/Heston/new_experiments/protocol/experiments/scripts/evaluate_drawdown_memory.py`,
**run unchanged** (PDF §7, checklist item 7). Aggregation: mean ± sample std (ddof = 1) over
5 seeds.

**The paired interval exists here, and this is the only place in the whole tree where it can.**
PDF §5 requires the five seedwise differences and a paired interval *"for comparisons between
models run with aligned seeds"*. A method-vs-floor comparison cannot satisfy that — the floor is
five true-DGP draws at seeds 1000–1004, which have no correspondence to model seeds 0–4, so
`tools/aggregate_pdf_metrics.py` refuses to pair them. CSDI and LS4 were both run on seeds 0–4,
so the difference **is** paired, and the `Mean diff` and `Paired 95% CI` columns below are the
PDF §5 quantity rather than a substitute for it.

**Read the CI before the means.** A paired CI that contains zero means the two methods were not
consistently separated seed by seed, however far apart their averages look. Those rows are
reported as `tie` in the `Winner` column even where one mean is visibly smaller —
`errors.terminal_log_price_ks` (0.0414 vs 0.0248, CI ±0.022 on a mean difference of 0.0167) is
the clearest example. Three PDF rows fall this way.

**How `Winner` is decided.** Not by "smaller number wins". Each metric is scored by distance to
its own reference: error/distance metrics reference 0; a `generated_memory.X` row references its
own `target_memory.X` row (the oracle's reading of `test.npy`); a `novelty.*` row references the
**perfect floor**, because being *further* from the training set than a real DGP draw is not a
virtue any more than being nearer is. `target_memory.*` rows are constants computed from
`test.npy` alone, identical in every column, and get no winner at all.

### 1.1 PRIMARY panel

The four metrics PDF §2 designates as primary, with each method's distance from the floor:

| Metric | CSDI | LS4 | Perfect floor | CSDI × floor | LS4 × floor | Winner |
|---|---|---|---|---|---|---|
| `early_hit_rate_error` | 0.161279 ± 0.0182 | **0.0081543 ± 0.00691** | 0.00310059 ± 0.00224 | 52× | **2.6×** | LS4 |
| `future_rv_hit_gap_error` | 0.0830031 ± 0.00635 | **0.0503776 ± 0.00241** | 0.00102294 ± 0.000516 | 81× | **49×** | LS4 |
| `early_history_incremental_r2_error` | 0.227474 ± 0.0151 | **0.118178 ± 0.0154** | 0.00641167 ± 0.00544 | 35× | **18×** | LS4 |
| `future_rv_wasserstein` | 0.0576958 ± 0.00278 | **0.0119597 ± 0.00142** | 0.00123331 ± 0.00021 | 47× | **9.7×** | LS4 |

LS4 sweeps the primary panel. The one place it comes close to the floor is
`early_hit_rate_error` at 2.6×; everywhere else both methods are an order of magnitude out or
worse. `future_rv_hit_gap_error` is the metric that resists both: 49× is LS4's *best* result on
it, and that metric is precisely the one that requires reading volatility 32 steps after its
cause.

### 1.2 What actually separates them

**CSDI does not encode the delayed memory at all.** `generated_memory.early_history_incremental_r2`
— the extra explanatory power the first 32 steps carry about future realized volatility, which
is the entire point of the DGP — is 0.0610 for CSDI against a target of 0.2885. LS4 reaches
0.1703. Both fall short; CSDI recovers roughly a fifth of the structure, LS4 roughly three
fifths. The paired CI (±0.0312 on a difference of 0.109) excludes zero comfortably, so this is a
per-seed-consistent gap, not sampling noise.

**CSDI's banks sit measurably closer to the training set than a real DGP draw does.** This is
the one finding that is not a simple "LS4 is better":

| Novelty metric | CSDI | LS4 | Perfect floor |
|---|---|---|---|
| `distinct_nearest_training_paths` | 1266 ± 19.8 | 1596.4 ± 33.1 | 1600.8 ± 11.5 |
| `mean_standardized_nearest_train_path_rmse` | 0.804888 ± 0.0052 | 0.949641 ± 0.00608 | 0.937306 ± 0.000826 |
| `median_standardized_nearest_train_path_rmse` | 0.840909 ± 0.0075 | 0.988202 ± 0.00971 | 0.989367 ± 0.00106 |

LS4 is statistically indistinguishable from the floor on all three. CSDI's 8192 generated paths
collapse onto 1266 distinct training neighbours where a genuine DGP draw finds 1601, and they
sit 14 % nearer in standardized RMSE. **This is a concentration signature, not evidence of
memorisation** — no generated path is a copy of a training path — but it means CSDI's sample
diversity is lower than the data it was fit on, and any downstream use that depends on tail
coverage should account for it.

**CSDI's one win is a real one.** `generated_memory.future_rv_no_hit_mean` (target 0.202984):
CSDI 0.220232, LS4 0.242114. CSDI is nearer, the paired CI (±0.00624 on 0.0219) excludes zero.
The quiet-regime volatility level is the one piece of the structure CSDI places better.

### 1.3 Full table

<!-- BEGIN GENERATED: experiment A table pdf -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Mean diff</th><th rowspan="2">Paired 95% CI</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>LS4</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>0.0189235 ± 0.000518</td><td><b>0.0144903 ± 0.0024</b></td><td>0.00156102 ± 0.000141</td><td>0.00443316 ± 0.00289</td><td>±0.00359</td><td>LS4</td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>0.227474 ± 0.0151</td><td><b>0.118178 ± 0.0154</b></td><td>0.00641167 ± 0.00544</td><td>0.109296 ± 0.0252</td><td>±0.0312</td><td>LS4</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>0.161279 ± 0.0182</td><td><b>0.0081543 ± 0.00691</b></td><td>0.00310059 ± 0.00224</td><td>0.153125 ± 0.0171</td><td>±0.0212</td><td>LS4</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>0.59727 ± 0.0408</td><td><b>0.302096 ± 0.158</b></td><td>0.023094 ± 0.0118</td><td>0.295175 ± 0.188</td><td>±0.233</td><td>LS4</td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>0.0830031 ± 0.00635</td><td><b>0.0503776 ± 0.00241</b></td><td>0.00102294 ± 0.000516</td><td>0.0326255 ± 0.0076</td><td>±0.00943</td><td>LS4</td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>0.0576958 ± 0.00278</td><td><b>0.0119597 ± 0.00142</b></td><td>0.00123331 ± 0.00021</td><td>0.0457361 ± 0.003</td><td>±0.00372</td><td>LS4</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>0.00290882 ± 0.000109</td><td><b>0.000202294 ± 0.00012</b></td><td>2.79631e-05 ± 1.44e-05</td><td>0.00270652 ± 0.00014</td><td>±0.000174</td><td>LS4</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>0.00867602 ± 0.00106</td><td>0.00776208 ± 0.00602</td><td>0.00161632 ± 0.000193</td><td>0.000913942 ± 0.00665</td><td>±0.00826</td><td><i>tie</i></td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>0.0414307 ± 0.00992</td><td>0.0247803 ± 0.00912</td><td>0.0120117 ± 0.00319</td><td>0.0166504 ± 0.0177</td><td>±0.022</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>0.453933 ± 0.0163</td><td><b>0.548288 ± 0.0087</b></td><td>0.673625 ± 0.00375</td><td>-0.094355 ± 0.0182</td><td>±0.0226</td><td>LS4</td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>0.392884 ± 0.00864</td><td>0.377943 ± 0.0139</td><td>0.380345 ± 0.00425</td><td>0.0149406 ± 0.0129</td><td>±0.0161</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>0.0610487 ± 0.0151</td><td><b>0.170344 ± 0.0154</b></td><td>0.29328 ± 0.00726</td><td>-0.109296 ± 0.0252</td><td>±0.0312</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>0.57039 ± 0.0256</td><td><b>0.695678 ± 0.0117</b></td><td>0.809204 ± 0.00274</td><td>-0.125287 ± 0.0308</td><td>±0.0383</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>0.468359 ± 0.0182</td><td><b>0.636279 ± 0.0087</b></td><td>0.626538 ± 0.00224</td><td>-0.16792 ± 0.0225</td><td>±0.0279</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>0.614215 ± 0.0757</td><td><b>1.08214 ± 0.048</b></td><td>1.51798 ± 0.0125</td><td>-0.46792 ± 0.104</td><td>±0.13</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>0.140868 ± 0.00635</td><td><b>0.173494 ± 0.00241</b></td><td>0.223988 ± 0.00125</td><td>-0.0326255 ± 0.0076</td><td>±0.00943</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>0.361101 ± 0.00224</td><td><b>0.415608 ± 0.00299</b></td><td>0.427455 ± 0.000421</td><td>-0.0545071 ± 0.00343</td><td>±0.00425</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td><b>0.220232 ± 0.00509</b></td><td>0.242114 ± 0.000655</td><td>0.203467 ± 0.000875</td><td>-0.0218817 ± 0.00503</td><td>±0.00624</td><td>CSDI</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>1266 ± 19.8</td><td><b>1596.4 ± 33.1</b></td><td>1600.8 ± 11.5</td><td>-330.4 ± 28.2</td><td>±35</td><td>LS4</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>0.804888 ± 0.0052</td><td><b>0.949641 ± 0.00608</b></td><td>0.937306 ± 0.000826</td><td>-0.144753 ± 0.00903</td><td>±0.0112</td><td>LS4</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>0.840909 ± 0.0075</td><td><b>0.988202 ± 0.00971</b></td><td>0.989367 ± 0.00106</td><td>-0.147293 ± 0.0144</td><td>±0.0179</td><td>LS4</td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
</table>
<!-- END GENERATED -->

*Reproduce:* `python tools/make_comparison_tables.py --experiment A --table pdf`
(mean diff and CI: `tools/aggregate_pdf_metrics.py --model-dir experiment_A/CSDI --paired-dir experiment_A/LS4`).

---

## 2. A1–A34, cross-method comparison (mean ± std, 5 seeds)

The repo's standard battery. This is **context, not the verdict** — Section 1 is what the
protocol asks. Per-seed values are in each method's own README; this table is the aggregate.

No paired CI is available here: `make_metrics_tables.py` emits aggregates only, so `Winner`
compares means without a per-seed consistency test. Treat narrow gaps with suspicion
accordingly.

> **A1 is not usable on this experiment.** The perfect floor's kurtosis error is
> 0.0601 ± 0.0382 — a std more than half the mean — so a method scoring "close to floor" on A1
> is inside the floor's own noise. A2–A4 (tail quantiles and tail QQ) are the stable tail
> statistics; read those instead.

<!-- BEGIN GENERATED: experiment A table A -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>LS4</th></tr>
<tr><td colspan="5"><b>Fat Tail</b></td></tr>
<tr><td>A1 Kurtosis Error ↓</td><td>0.674298 ± 0.078</td><td><b>0.202463 ± 0.11</b></td><td>0.0601011 ± 0.0382</td><td>LS4</td></tr>
<tr><td>A2 |r| q95 Error ↓</td><td>0.00619595 ± 0.000208</td><td><b>0.000373524 ± 0.000359</b></td><td>6.48561e-05 ± 4.46e-05</td><td>LS4</td></tr>
<tr><td>A3 |r| q99 Error ↓</td><td>0.00782819 ± 0.000344</td><td><b>0.000270752 ± 0.000205</b></td><td>8.275e-05 ± 3.63e-05</td><td>LS4</td></tr>
<tr><td>A4 Tail QQ Error ↓</td><td>0.00611364 ± 0.000219</td><td><b>0.000474312 ± 0.000292</b></td><td>9.25658e-05 ± 4.42e-05</td><td>LS4</td></tr>
<tr><td>A5 Hill Tail Index Error ↓</td><td>1.73483 ± 0.327</td><td><b>0.182615 ± 0.106</b></td><td>0.407724 ± 0.315</td><td>LS4</td></tr>
<tr><td colspan="5"><b>Distribution</b></td></tr>
<tr><td>A6 Path MMD² ↓</td><td>0.00263734 ± 0.000233</td><td><b>0.00199709 ± 0.000231</b></td><td>0.00177285 ± 5.64e-05</td><td>LS4</td></tr>
<tr><td>A7 Terminal MMD² ↓</td><td>0.002392 ± 0.000891</td><td><b>0.00232808 ± 0.00111</b></td><td>0.00170387 ± 0.000258</td><td>LS4</td></tr>
<tr><td>A8 Increment MMD² ↓</td><td>0.00411459 ± 0.000585</td><td><b>0.00131215 ± 0.000165</b></td><td>0.00107654 ± 6.13e-05</td><td>LS4</td></tr>
<tr><td>A9 Volatility MMD ↓</td><td>0.107136 ± 0.0104</td><td><b>0.0175338 ± 0.00182</b></td><td>0.0105444 ± 0.00044</td><td>LS4</td></tr>
<tr><td>A10 Terminal SWD ↓</td><td>1.84424 ± 0.393</td><td><b>1.5827 ± 0.657</b></td><td>1.02907 ± 0.185</td><td>LS4</td></tr>
<tr><td>A11 Path SWD ↓</td><td>0.878639 ± 0.0801</td><td><b>0.663082 ± 0.0887</b></td><td>0.542455 ± 0.0723</td><td>LS4</td></tr>
<tr><td>A12 RV Law Loss ↓</td><td>3.31151 ± 0.114</td><td><b>0.40919 ± 0.0954</b></td><td>0.0622546 ± 0.0132</td><td>LS4</td></tr>
<tr><td>A13 Mean Path RMSE ↓</td><td>0.552904 ± 0.408</td><td><b>0.272603 ± 0.139</b></td><td>0.119593 ± 0.0243</td><td>LS4</td></tr>
<tr><td>A14 KS Log-returns ↓</td><td>0.0431914 ± 0.00206</td><td><b>0.0165854 ± 0.00143</b></td><td>0.00113131 ± 9.72e-05</td><td>LS4</td></tr>
<tr><td>A15 Skewness Error ↓</td><td><b>0.0112248 ± 0.00329</b></td><td>0.0552681 ± 0.0347</td><td>0.00329532 ± 0.00104</td><td>CSDI</td></tr>
<tr><td>A16 QQ RMSE ↓</td><td>0.00288986 ± 0.000108</td><td><b>0.000604692 ± 5.22e-05</b></td><td>4.91034e-05 ± 8.54e-06</td><td>LS4</td></tr>
<tr><td>A17 Terminal KS ↓</td><td>0.0414307 ± 0.00992</td><td><b>0.0247803 ± 0.00912</b></td><td>0.0120117 ± 0.00319</td><td>LS4</td></tr>
<tr><td colspan="5"><b>Adversarial</b></td></tr>
<tr><td>A18 Discriminative (GRU) ↓</td><td>0.0141288 ± 0.022</td><td><b>0.0063776 ± 0.00256</b></td><td>0.0073542 ± 0.00565</td><td>LS4</td></tr>
<tr><td>A18 Discriminative (MLP) ↓</td><td>0.0097348 ± 0.00571</td><td><b>0.004486 ± 0.00519</b></td><td>0.0042418 ± 0.00229</td><td>LS4</td></tr>
<tr><td colspan="5"><b>Predictive</b></td></tr>
<tr><td>A19 Predictive (GRU) ↓</td><td>0.046486 ± 1.86e-05</td><td><b>0.046411 ± 5.24e-06</b></td><td>0.0464168 ± 1.53e-05</td><td>LS4</td></tr>
<tr><td>A19 Predictive (MLP) ↓</td><td><b>0.0480262 ± 2.66e-05</b></td><td>0.0480982 ± 9.26e-05</td><td>0.048241 ± 0.00042</td><td>CSDI</td></tr>
<tr><td colspan="5"><b>Temporal</b></td></tr>
<tr><td>A20 Covariance Error ↓</td><td>86.5044 ± 18.1</td><td><b>27.6665 ± 34.3</b></td><td>17.2826 ± 9.01</td><td>LS4</td></tr>
<tr><td>A21 ACF |r| ↓</td><td>0.0254815 ± 0.00207</td><td><b>0.00452511 ± 0.00146</b></td><td>0.00160018 ± 0.000752</td><td>LS4</td></tr>
<tr><td>A22 ACF r² ↓</td><td>0.0149441 ± 0.00108</td><td><b>0.00370279 ± 0.00165</b></td><td>0.00129902 ± 0.000643</td><td>LS4</td></tr>
<tr><td>A23 ACF lag-1 |r| Error ↓</td><td>0.0331209 ± 0.00344</td><td><b>0.00452018 ± 0.00243</b></td><td>0.00151053 ± 0.00114</td><td>LS4</td></tr>
<tr><td>A24 ACF lag-1 r² Error ↓</td><td>0.0236946 ± 0.00262</td><td><b>0.00675596 ± 0.00249</b></td><td>0.000984905 ± 0.000792</td><td>LS4</td></tr>
<tr><td colspan="5"><b>Volatility</b></td></tr>
<tr><td>A25 Mean RMSE ↓</td><td>0.788538 ± 0.708</td><td><b>0.57019 ± 0.2</b></td><td>0.203377 ± 0.121</td><td>LS4</td></tr>
<tr><td>A26 Std Error ↓</td><td>0.278546 ± 0.00803</td><td><b>0.0251112 ± 0.0159</b></td><td>0.00441299 ± 0.00239</td><td>LS4</td></tr>
<tr><td>A27 Log-return Std Error ↓</td><td>0.00290882 ± 0.000109</td><td><b>0.000202294 ± 0.00012</b></td><td>2.79631e-05 ± 1.44e-05</td><td>LS4</td></tr>
<tr><td>A28 Kurtosis Ratio → 1</td><td>0.819936 ± 0.0101</td><td><b>1.04109 ± 0.122</b></td><td>0.991589 ± 0.00428</td><td>LS4</td></tr>
<tr><td>A29 Sigma Mean Error ↓</td><td>0.0456726 ± 0.00186</td><td><b>0.00543401 ± 0.00202</b></td><td>0.000392072 ± 0.000269</td><td>LS4</td></tr>
<tr><td>A30 Vol Path RMSE ↓</td><td>1.30152 ± 0.21</td><td><b>0.327853 ± 0.251</b></td><td>0.190091 ± 0.0906</td><td>LS4</td></tr>
<tr><td>A31 Rolling Vol KS ↓</td><td>0.162503 ± 0.00249</td><td><b>0.0691744 ± 0.00281</b></td><td>0.0019172 ± 0.000476</td><td>LS4</td></tr>
<tr><td>A32 Vol-of-vol Error ↓</td><td>0.00118022 ± 7.13e-05</td><td><b>0.000305445 ± 3.72e-05</b></td><td>2.13612e-05 ± 1.1e-05</td><td>LS4</td></tr>
<tr><td colspan="5"><b>Heston-specific (dropped)</b></td></tr>
<tr><td>A33 Sigma Correlation (dropped)</td><td>n/a</td><td>n/a</td><td>n/a</td><td>—</td></tr>
<tr><td>A34 Sigma RMSE (dropped)</td><td>n/a</td><td>n/a</td><td>n/a</td><td>—</td></tr>
</table>
<!-- END GENERATED -->

*Reproduce:* `python tools/make_comparison_tables.py --experiment A --table A`

**Score: LS4 32, CSDI 2, 2 dropped.** CSDI's two wins are A15 (skewness error, 0.0112 vs
0.0553) and A19-MLP (predictive score, 0.0480262 vs 0.0480982 — a gap in the fifth significant
figure, which is exactly the kind of margin the missing paired test would likely dissolve).
A33/A34 are dropped protocol-wide: they require the latent σ path, which the DGP exposes only
through `*_sigma.npy`, a file the information firewall forbids the generator side from reading.

---

## 3. Curve-shape metrics, cross-method comparison (mean ± std, 5 seeds)

Shape-of-the-curve measures: for each diagnostic plot, the MSE of the function and its first
and second derivatives, plus percentage error, NRMSE and the two CVaR levels.

<!-- BEGIN GENERATED: experiment A table B -->
<table>
<tr><th rowspan="2">Plot</th><th rowspan="2">Measure</th><th>Diffusion</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>LS4</th></tr>
<tr><td><b>Grid TVD (%)</b></td><td>—</td><td>561.8 ± 80.4</td><td><b>217.113 ± 29</b></td><td>155.077 ± 11.9</td><td>LS4</td></tr>
<tr><td><b>Log-return histogram</b></td><td>MSE (funct/der/sec-der avg)</td><td>1.87864 ± 0.0788</td><td><b>0.360244 ± 0.0107</b></td><td>0.0545244 ± 0.0165</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>113.208 ± 8.08</td><td><b>107.839 ± 8.14</b></td><td>95.391 ± 10.9</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>16.6572 ± 0.266</td><td><b>11.0789 ± 1.09</b></td><td>8.62821 ± 1.4</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>17.4322 ± 0.463</td><td><b>6.67777 ± 0.369</b></td><td>0.823273 ± 0.12</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>22.8937 ± 0.417</td><td><b>8.26687 ± 0.407</b></td><td>1.03955 ± 0.195</td><td>LS4</td></tr>
<tr><td><b>QQ plot</b></td><td>MSE (funct/der/sec-der avg)</td><td>2.93394e-06 ± 2.23e-07</td><td><b>1.271e-07 ± 2.46e-08</b></td><td>1.70382e-09 ± 4.39e-10</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>31.2204 ± 2.57</td><td><b>24.0374 ± 1.83</b></td><td>19.5871 ± 3.8</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>2.62797 ± 0.146</td><td><b>0.828455 ± 0.0966</b></td><td>0.311846 ± 0.0568</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>5.0292 ± 0.19</td><td><b>0.802367 ± 0.0966</b></td><td>0.101555 ± 0.0197</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>5.57556 ± 0.227</td><td><b>0.840247 ± 0.11</b></td><td>0.12425 ± 0.0237</td><td>LS4</td></tr>
<tr><td><b>ACF of |log-returns|</b></td><td>MSE (funct/der/sec-der avg)</td><td>0.000179114 ± 2.76e-05</td><td><b>1.2571e-05 ± 3.1e-06</b></td><td>5.34639e-06 ± 2.22e-06</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>207.99 ± 66.8</td><td><b>136.929 ± 39.7</b></td><td>135.251 ± 51.9</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>61.9089 ± 13</td><td><b>37.3337 ± 5.55</b></td><td>37.6473 ± 9.58</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>23.9138 ± 2.03</td><td><b>7.52395 ± 0.871</b></td><td>2.56508 ± 0.508</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>26.3283 ± 2.74</td><td><b>8.06721 ± 0.891</b></td><td>2.70023 ± 0.475</td><td>LS4</td></tr>
<tr><td><b>ACF of squared log-returns</b></td><td>MSE (funct/der/sec-der avg)</td><td>5.76535e-05 ± 9.75e-06</td><td><b>1.50972e-05 ± 6.16e-06</b></td><td>6.27026e-06 ± 3.17e-06</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>98.8259 ± 21</td><td><b>79.9839 ± 10.3</b></td><td>70.776 ± 18.9</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>59.1195 ± 10.6</td><td><b>46.975 ± 3.89</b></td><td>37.3861 ± 9.97</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>23.757 ± 2.03</td><td><b>8.37476 ± 3</b></td><td>3.86506 ± 0.764</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>28.2766 ± 3.13</td><td><b>8.97683 ± 2.7</b></td><td>4.48217 ± 0.963</td><td>LS4</td></tr>
<tr><td><b>Rolling volatility histogram</b></td><td>MSE (funct/der/sec-der avg)</td><td>77.0595 ± 2.44</td><td><b>13.013 ± 1.01</b></td><td>0.363092 ± 0.0823</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>219.587 ± 21.8</td><td><b>188.179 ± 26.9</b></td><td>194.499 ± 9.6</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>28.2051 ± 1.13</td><td><b>12.8265 ± 1.1</b></td><td>9.07535 ± 1.12</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>42.7545 ± 0.654</td><td><b>16.0333 ± 0.555</b></td><td>1.05297 ± 0.186</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>52.471 ± 0.862</td><td><b>18.4897 ± 0.639</b></td><td>1.33425 ± 0.317</td><td>LS4</td></tr>
<tr><td><b>Tail survival</b></td><td>MSE (funct/der/sec-der avg)</td><td>0.00107587 ± 6.45e-05</td><td><b>0.000138794 ± 1.93e-05</b></td><td>1.41954e-07 ± 6.99e-08</td><td>LS4</td></tr>
<tr><td></td><td>% error</td><td>5348.12 ± 383</td><td><b>4795.13 ± 328</b></td><td>4094.07 ± 296</td><td>LS4</td></tr>
<tr><td></td><td>NRMSE</td><td>71991.6 ± 1.94e+03</td><td><b>34914.3 ± 1.15e+03</b></td><td>10677.2 ± 384</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₀</td><td>8.3101 ± 0.236</td><td><b>3.16732 ± 0.196</b></td><td>0.115426 ± 0.0234</td><td>LS4</td></tr>
<tr><td></td><td>CVaR₉₅</td><td>8.35368 ± 0.235</td><td><b>3.18372 ± 0.195</b></td><td>0.122115 ± 0.0214</td><td>LS4</td></tr>
</table>
<!-- END GENERATED -->

*Reproduce:* `python tools/make_comparison_tables.py --experiment A --table B`

**Score: LS4 31, CSDI 0.** This is the most one-sided of the three suites, and it is consistent
with Section 1.2: CSDI's marginals are wrong in a way that shows up on every curve
simultaneously. Note that both methods are far from the floor on the tail-survival rows
(NRMSE 71992 and 34914 against a floor of 10677) — the tail of this DGP is hard for both, the
floor itself scores 4094 % error there, and the percentage-error column on that row should not
be read as if 100 % were the reference.

---

## 4. Stylised curves

Each method's diagnostic panel, generated by `tools/plot_stylised_facts.py` from that method's
seed-0 bank against `test.npy`.

**CSDI (Diffusion)**

![CSDI stylised facts](CSDI/plots/heston_diagnostics.png)

**LS4 (VAE)**

![LS4 stylised facts](LS4/plots/heston_diagnostics.png)

The memory structure the experiment actually targets — hit rate, the future-RV gap between hit
and no-hit paths, and the incremental R² of the early history — is plotted separately for each
method:

| CSDI | LS4 |
|---|---|
| ![CSDI memory structure](CSDI/plots/memory_structure_seed0.png) | ![LS4 memory structure](LS4/plots/memory_structure_seed0.png) |

Loss convergence and the per-seed PCA/t-SNE projections live under each method's `plots/` and
`losses/` directories.

---

## 5. Dataset description

The data-generating process is **not Heston**, despite the directory name. It is a latching
drawdown-memory volatility process defined in `dataset/Heston/new_experiments/protocol/`
(PDF §2):

```
memory_t = max(decay · memory_{t−1}, 1[drawdown_t ≥ 0.04]),   decay = exp(−ln2 / 60)
σ_t      = σ_low + (σ_high − σ_low) · memory_{t−32}            σ ∈ [0.12, 0.45]
```

Three properties make it hard, and all three are deliberate:

* **latching** — the drawdown indicator ratchets the memory state back to 1, so the state is not
  a smooth AR process and cannot be approximated by one;
* **a 32-step response delay** — the volatility reaction is decoupled in time from its cause, so
  a model matching contemporaneous correlations learns nothing about it;
* **half-life 60 > recent_window 20** — a model that reads only the last 20 steps *cannot*
  recover the signal, no matter how well it fits the marginals. This is why a method can score
  well on the standard battery and still fail Section 1.

### Files

| File | Shape | Who may read it |
|---|---|---|
| `train.npy` | 8192 × 128 | generators, evaluators |
| `disc.npy` | 8192 × 128 | validation only — model selection, early stopping |
| `test.npy` | 8192 × 128 | **evaluators only** |
| `*_sigma.npy` | 8192 × 128 | **evaluators only** — latent volatility path |

All banks are prices, float32, beginning at S₀ = 100.

**Information firewall.** The generator side of both methods reads `train.npy`, and `disc.npy`
for validation, and nothing else. It never opens `test.npy`, `*_sigma.npy`, `*_labels.npy`,
`oracle_*`, `oracle.joblib`, `gate_report.json`, or anything under `perfect_floor/`. This is the
condition that makes the perfect floor meaningful: the floor is what a *generator that knew the
DGP exactly* would score, and if the generator could see the test bank the comparison would be
vacuous.

**Perfect floor.** Five independent re-simulations of the true DGP at seeds 1000–1004, scored by
the same frozen evaluator. It is the best score any generator can achieve, and it is *not* zero,
because the evaluator compares two finite 8192-path samples. A method at 50× floor is 50× worse
than knowing the answer.

---

## Layout

```
experiment_A/
├── README.md              ← this file (cross-method comparison)
├── CSDI/                  ← Diffusion — full method README, banks, weights, losses, plots
├── LS4/                   ← VAE       — full method README, banks, weights, losses, plots
└── perfect_floor/         ← 5 true-DGP re-simulations + their metrics
```

Every number in Sections 1–3 is generated from the artefacts by
`tools/make_comparison_tables.py`, which shells out to the same two producers the method
READMEs are checked against. Refresh the whole file after a re-run with:

```bash
python tools/make_comparison_tables.py --inject experiment_A/README.md
```

Protocol and conventions: [`../guideline_new_experiment.md`](../guideline_new_experiment.md).
Known ambiguities in the protocol PDF and the readings adopted:
[`../open_questions_to_protocol_author.md`](../open_questions_to_protocol_author.md).
