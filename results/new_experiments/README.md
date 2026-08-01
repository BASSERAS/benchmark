# New experiments — cross-method comparison

Two experiments, three generative methods each, five seeds each, one shared perfect-floor
reference. This page carries **only the protocol-evaluator (PDF) comparison table for each
experiment**. Everything else — the A1–A34 battery, the curve-shape suite, the stylised
plots, the dataset description — lives one level down:

| | Experiment | Comparison README | Method READMEs |
|---|---|---|---|
| **A** | Delayed drawdown memory | [`experiment_A/README.md`](experiment_A/README.md) | [CSDI](experiment_A/CSDI/README.md) · [TimeDiT](experiment_A/TimeDiT/README.md) · [LS4](experiment_A/LS4/README.md) |
| **B** | Balanced Heston parameter mixture | [`experiment_B/README.md`](experiment_B/README.md) | [CSDI](experiment_B/CSDI/README.md) · [TimeDiT](experiment_B/TimeDiT/README.md) · [LS4](experiment_B/LS4/README.md) |

**How to read the `Winner` column.** It is not "smaller number wins". Each metric has a
reference value it is supposed to reach — `0` for an error or distance, the matching `target_*`
row for a memory or mixture statistic — and the winner is the method whose mean sits closer to
that reference. On top of that, a winner is only declared when the **paired 95 % confidence
interval over the five aligned seeds excludes zero** (PDF §5). When it does not, the row reads
*tie* however far apart the means look: the gap is not reproducible seed by seed.

**With three methods the interval that gates a row is the one between the best and the
runner-up**, not between whichever pair a reader is interested in. A method that lands *between*
the previous two therefore converts wins into ties without any number changing, which is exactly
what happened when TimeDiT was added — §1 quantifies it. Every pairwise interval is printed in
full in the blocks that follow each table, so any pair can still be read directly.

Two classes of row carry **no winner at all**:

* `—` — a **constant**. `target_*` rows are one reading of `test.npy`, printed identically in
  all three columns. Not a contest.
* `diagnostic (§…)` — a **raw diagnostic**. PDF §5: *"All displayed fidelity metrics are errors
  or distances, so lower is better, except explicitly identified raw diagnostics such as
  standard-deviation ratio, posterior confidence, group means, and novelty."* §4 adds that
  novelty *"must not be combined with fidelity metrics into a score"*; §3.3 that posterior
  entropy, maximum probability and low-confidence fraction *"are diagnostics, not separate
  winner-selection criteria."* These rows print their value and the clause that exempts them,
  and are excluded from every tally on this page.

**Bold** marks the method closest to the reference on that row — which is not always the winner,
because the paired interval can still straddle zero. The `Perfect` column is the floor — a
generator that draws from the true data-generating process — and is excluded from the contest;
it is there to say how far *all three* methods still are from the achievable optimum.

**No aggregate score is constructed** (PDF §3.4). The tallies below count independently-decided
rows so a reader can see the split. They are not an index and they are not weighted; the
PRIMARY panels in the two experiment READMEs are what the protocol actually asks about. And a
tally is a property of the *field*, not of a method: the same artefacts scored against a
different set of competitors give a different tally, as §1 shows.

---

## 1. Experiment A — delayed drawdown memory

Of the 30 rows: **9 constants**, **5 raw diagnostics**, and **16 contested fidelity metrics**.
Of those 16 — **LS4 wins 6, TimeDiT wins 0, CSDI wins 0, 10 are ties.** Experiment A is still
not a close call in the direction of the mean — LS4 leads almost everywhere — but most of that
lead is no longer reproducible against the field.

**Adding a third method moved seven rows out of LS4's column and not one number changed.**
Against CSDI alone the same artefacts read **LS4 13 / CSDI 0 / 3 ties**; with TimeDiT present
they read **LS4 6 / TimeDiT 0 / CSDI 0 / 10 ties**. The cause is mechanical: the winner is
gated on the paired interval between the best and the *runner-up*, TimeDiT is now the runner-up
on those seven rows, and its interval straddles zero. The seven are

```
errors.abs_return_acf_rmse_lags_1_50          generated_memory.augmented_r2
errors.excess_kurtosis_error                  generated_memory.early_hit_future_rv_correlation
errors.future_rv_hit_gap_error                generated_memory.early_hit_rate
                                              generated_memory.future_rv_hit_gap
```

"LS4 beats the field" was always shorthand for "LS4 beats the one other method in the table".
**The tally is a property of the field, not of a method.**

And no method solves it. On the four PRIMARY metrics of PDF §2 the *best of the three* is still
2.6× to 39× worse than a fresh draw from the true DGP. This is a comparison between three
failures of different magnitude.

<!-- BEGIN GENERATED: experiment A table pdf -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>Diffusion Transformer</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>TimeDiT</th><th>LS4</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>0.0189235 ± 0.000518</td><td>0.0130564 ± 0.00532</td><td>0.0144903 ± 0.0024</td><td>0.00156102 ± 0.000141</td><td><i>tie</i></td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>0.227474 ± 0.0151</td><td>0.169834 ± 0.0233</td><td><b>0.118178 ± 0.0154</b></td><td>0.00641167 ± 0.00544</td><td>LS4</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>0.161279 ± 0.0182</td><td>0.0871826 ± 0.0493</td><td><b>0.0081543 ± 0.00691</b></td><td>0.00310059 ± 0.00224</td><td>LS4</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>0.59727 ± 0.0408</td><td>0.583825 ± 0.341</td><td>0.302096 ± 0.158</td><td>0.023094 ± 0.0118</td><td><i>tie</i></td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>0.0830031 ± 0.00635</td><td>0.0397687 ± 0.0143</td><td>0.0503776 ± 0.00241</td><td>0.00102294 ± 0.000516</td><td><i>tie</i></td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>0.0576958 ± 0.00278</td><td>0.0360582 ± 0.0141</td><td><b>0.0119597 ± 0.00142</b></td><td>0.00123331 ± 0.00021</td><td>LS4</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>0.00290882 ± 0.000109</td><td>0.00140951 ± 0.000614</td><td><b>0.000202294 ± 0.00012</b></td><td>2.79631e-05 ± 1.44e-05</td><td>LS4</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>0.00867602 ± 0.00106</td><td>0.0105506 ± 0.00193</td><td>0.00776208 ± 0.00602</td><td>0.00161632 ± 0.000193</td><td><i>tie</i></td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>0.0414307 ± 0.00992</td><td>0.0950684 ± 0.0271</td><td>0.0247803 ± 0.00912</td><td>0.0120117 ± 0.00319</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>0.453933 ± 0.0163</td><td>0.534224 ± 0.0468</td><td>0.548288 ± 0.0087</td><td>0.673625 ± 0.00375</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>0.392884 ± 0.00864</td><td>0.415535 ± 0.043</td><td>0.377943 ± 0.0139</td><td>0.380345 ± 0.00425</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>0.0610487 ± 0.0151</td><td>0.118689 ± 0.0233</td><td><b>0.170344 ± 0.0154</b></td><td>0.29328 ± 0.00726</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>0.57039 ± 0.0256</td><td>0.666913 ± 0.0331</td><td>0.695678 ± 0.0117</td><td>0.809204 ± 0.00274</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>0.468359 ± 0.0182</td><td>0.573413 ± 0.0893</td><td>0.636279 ± 0.0087</td><td>0.626538 ± 0.00224</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>0.614215 ± 0.0757</td><td>0.898544 ± 0.078</td><td><b>1.08214 ± 0.048</b></td><td>1.51798 ± 0.0125</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>0.140868 ± 0.00635</td><td>0.184103 ± 0.0143</td><td>0.173494 ± 0.00241</td><td>0.223988 ± 0.00125</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>0.361101 ± 0.00224</td><td>0.406354 ± 0.0191</td><td>0.415608 ± 0.00299</td><td>0.427455 ± 0.000421</td><td><i>diagnostic (§5 group mean)</i></td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td>0.220232 ± 0.00509</td><td>0.222251 ± 0.0283</td><td>0.242114 ± 0.000655</td><td>0.203467 ± 0.000875</td><td><i>diagnostic (§5 group mean)</i></td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>1266 ± 19.8</td><td>1528.6 ± 92.6</td><td>1596.4 ± 33.1</td><td>1600.8 ± 11.5</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>0.804888 ± 0.0052</td><td>0.898962 ± 0.0752</td><td>0.949641 ± 0.00608</td><td>0.937306 ± 0.000826</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>0.840909 ± 0.0075</td><td>0.948006 ± 0.0682</td><td>0.988202 ± 0.00971</td><td>0.989367 ± 0.00106</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>—</td></tr>
</table>

<p><b>Paired, aligned-seed: CSDI vs TimeDiT</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>+0.01082, -0.001345, +0.008927, +0.002862, +0.008073</td><td>0.0058671 ± 0.00499</td><td>±0.0062</td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>+0.09952, +0.05703, +0.05811, +0.02727, +0.04627</td><td>0.0576405 ± 0.0265</td><td>±0.0329</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>+0.04126, +0.02673, +0.1639, +0.07776, +0.06079</td><td>0.0740967 ± 0.0538</td><td>±0.0668</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>-0.209, -0.2273, +0.508, +0.3103, -0.3148</td><td>0.0134458 ± 0.37</td><td>±0.459</td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>+0.05732, +0.06461, +0.03071, +0.01975, +0.04378</td><td>0.0432344 ± 0.0185</td><td>±0.0229</td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>+0.01883, +0.00913, +0.04754, +0.02423, +0.008449</td><td>0.0216376 ± 0.0159</td><td>±0.0198</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>+0.001264, +0.001077, +0.002582, +0.001774, +0.0007987</td><td>0.0014993 ± 0.000702</td><td>±0.000872</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>-0.002882, -0.005378, +0.0016, -0.002974, +0.0002608</td><td>-0.00187463 ± 0.00279</td><td>±0.00346</td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>-0.0509, -0.09778, -0.006592, -0.052, -0.06091</td><td>-0.0536377 ± 0.0325</td><td>±0.0403</td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>-0.1303, -0.1401, -0.04014, -0.003769, -0.08718</td><td>-0.0802914 ± 0.0583</td><td>±0.0724</td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>-0.03078, -0.08303, +0.01796, +0.0235, -0.04091</td><td>-0.0226509 ± 0.0442</td><td>±0.0549</td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>-0.09952, -0.05703, -0.05811, -0.02727, -0.04627</td><td>-0.0576405 ± 0.0265</td><td>±0.0329</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>-0.1426, -0.1389, -0.07131, -0.0229, -0.1069</td><td>-0.0965224 ± 0.0502</td><td>±0.0623</td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>-0.04126, -0.02673, -0.1879, -0.2086, -0.06079</td><td>-0.105054 ± 0.0862</td><td>±0.107</td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>-0.4057, -0.3023, -0.2865, -0.1802, -0.247</td><td>-0.284329 ± 0.0826</td><td>±0.103</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>-0.05732, -0.06461, -0.03071, -0.01975, -0.04378</td><td>-0.0432344 ± 0.0185</td><td>±0.0229</td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>-0.04094, -0.03792, -0.05895, -0.06656, -0.0219</td><td>-0.0452531 ± 0.0177</td><td>±0.022</td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td>+0.01639, +0.02669, -0.02824, -0.04681, +0.02189</td><td>-0.00201862 ± 0.0333</td><td>±0.0413</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-259, -175, -319, -404, -156</td><td>-262.6 ± 103</td><td>±128</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.04827, -0.03665, -0.1662, -0.1922, -0.02703</td><td>-0.094074 ± 0.0786</td><td>±0.0976</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.06028, -0.06169, -0.1721, -0.1978, -0.04369</td><td>-0.107097 ± 0.072</td><td>±0.0893</td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
</table>

<p><b>Paired, aligned-seed: CSDI vs LS4</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>-8.005e-05, +0.006656, +0.007086, +0.003526, +0.004978</td><td>0.00443316 ± 0.00289</td><td>±0.00359</td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>+0.09389, +0.1202, +0.1052, +0.08103, +0.1462</td><td>0.109296 ± 0.0252</td><td>±0.0312</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>+0.1439, +0.1561, +0.1696, +0.1287, +0.1674</td><td>0.153125 ± 0.0171</td><td>±0.0212</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>-0.01828, +0.3813, +0.3317, +0.4804, +0.3007</td><td>0.295175 ± 0.188</td><td>±0.233</td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>+0.0337, +0.0359, +0.02921, +0.02199, +0.04233</td><td>0.0326255 ± 0.0076</td><td>±0.00943</td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>+0.04186, +0.0438, +0.04959, +0.04715, +0.04628</td><td>0.0457361 ± 0.003</td><td>±0.00372</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>+0.002797, +0.00257, +0.002847, +0.002778, +0.002542</td><td>0.00270652 ± 0.00014</td><td>±0.000174</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>-0.01034, +0.00432, +0.006376, +0.0003482, +0.003862</td><td>0.000913942 ± 0.00665</td><td>±0.00826</td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>-0.009521, +0.01501, +0.03601, +0.01196, +0.02979</td><td>0.0166504 ± 0.0177</td><td>±0.022</td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>-0.09516, -0.1102, -0.08462, -0.06908, -0.1127</td><td>-0.094355 ± 0.0182</td><td>±0.0226</td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>-0.001275, +0.01004, +0.02056, +0.01195, +0.03343</td><td>0.0149406 ± 0.0129</td><td>±0.0161</td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>-0.09389, -0.1202, -0.1052, -0.08103, -0.1462</td><td>-0.109296 ± 0.0252</td><td>±0.0312</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>-0.1213, -0.1443, -0.1154, -0.082, -0.1635</td><td>-0.125287 ± 0.0308</td><td>±0.0383</td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>-0.1439, -0.1561, -0.1823, -0.1577, -0.1996</td><td>-0.16792 ± 0.0225</td><td>±0.0279</td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>-0.4317, -0.5235, -0.4417, -0.3326, -0.6101</td><td>-0.46792 ± 0.104</td><td>±0.13</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>-0.0337, -0.0359, -0.02921, -0.02199, -0.04233</td><td>-0.0326255 ± 0.0076</td><td>±0.00943</td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>-0.05304, -0.05382, -0.05456, -0.05097, -0.06015</td><td>-0.0545071 ± 0.00343</td><td>±0.00425</td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td>-0.01934, -0.01792, -0.02535, -0.02898, -0.01782</td><td>-0.0218817 ± 0.00503</td><td>±0.00624</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-328, -306, -327, -313, -378</td><td>-330.4 ± 28.2</td><td>±35</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.1332, -0.1398, -0.1548, -0.1431, -0.1528</td><td>-0.144753 ± 0.00903</td><td>±0.0112</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.1245, -0.145, -0.1605, -0.1478, -0.1587</td><td>-0.147293 ± 0.0144</td><td>±0.0179</td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
</table>

<p><b>Paired, aligned-seed: TimeDiT vs LS4</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>-0.0109, +0.008001, -0.001842, +0.0006638, -0.003095</td><td>-0.00143395 ± 0.00682</td><td>±0.00846</td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>-0.005633, +0.06318, +0.04708, +0.05375, +0.0999</td><td>0.0516551 ± 0.038</td><td>±0.0471</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>+0.1027, +0.1294, +0.005615, +0.0509, +0.1066</td><td>0.0790283 ± 0.0501</td><td>±0.0622</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>+0.1907, +0.6086, -0.1762, +0.1701, +0.6155</td><td>0.281729 ± 0.335</td><td>±0.416</td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>-0.02362, -0.02871, -0.001496, +0.002239, -0.001455</td><td>-0.010609 ± 0.0144</td><td>±0.0179</td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>+0.02302, +0.03467, +0.002047, +0.02292, +0.03784</td><td>0.0240985 ± 0.014</td><td>±0.0174</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>+0.001532, +0.001492, +0.0002651, +0.001003, +0.001743</td><td>0.00120722 ± 0.000592</td><td>±0.000735</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>-0.007454, +0.009698, +0.004776, +0.003322, +0.003601</td><td>0.00278857 ± 0.00628</td><td>±0.00779</td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>+0.04138, +0.1128, +0.0426, +0.06396, +0.0907</td><td>0.0702881 ± 0.0311</td><td>±0.0386</td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>+0.03514, +0.02989, -0.04448, -0.06531, -0.02556</td><td>-0.0140636 ± 0.0448</td><td>±0.0556</td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>+0.02951, +0.09307, +0.002598, -0.01155, +0.07433</td><td>0.0375916 ± 0.0451</td><td>±0.056</td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>+0.005633, -0.06318, -0.04708, -0.05375, -0.0999</td><td>-0.0516551 ± 0.038</td><td>±0.0471</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>+0.02136, -0.005385, -0.04413, -0.0591, -0.05657</td><td>-0.028765 ± 0.0353</td><td>±0.0438</td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>-0.1027, -0.1294, +0.005615, +0.0509, -0.1388</td><td>-0.0628662 ± 0.0857</td><td>±0.106</td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>-0.02595, -0.2212, -0.1552, -0.1525, -0.3631</td><td>-0.183591 ± 0.123</td><td>±0.152</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>+0.02362, +0.02871, +0.001496, -0.002239, +0.001455</td><td>0.010609 ± 0.0144</td><td>±0.0179</td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>-0.01211, -0.0159, +0.004389, +0.01559, -0.03825</td><td>-0.00925408 ± 0.0206</td><td>±0.0256</td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td>-0.03573, -0.04461, +0.002892, +0.01783, -0.0397</td><td>-0.0198631 ± 0.0283</td><td>±0.0351</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-69, -131, -8, +91, -222</td><td>-67.8 ± 119</td><td>±148</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.08496, -0.1032, +0.01139, +0.04913, -0.1258</td><td>-0.0506789 ± 0.0765</td><td>±0.0949</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.0642, -0.08328, +0.0116, +0.04995, -0.1151</td><td>-0.0401958 ± 0.0686</td><td>±0.0852</td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
</table>
<!-- END GENERATED -->

Full detail — including the primary-metric panel, the ×floor ratios, the novelty analysis,
the A1–A34 battery and the curve-shape suite — is in
[`experiment_A/README.md`](experiment_A/README.md).

---

## 2. Experiment B — balanced Heston parameter mixture

Of the 51 rows: **11 constants**, **17 raw diagnostics**, and **23 contested fidelity metrics**.
Of those 23 — **TimeDiT wins 7, CSDI wins 2, LS4 wins 1, 13 are ties.** The sign is the opposite
of Experiment A's, and it is the reversal that matters: the method with **zero** contested wins
on A takes the most of them on B.

**Ten of the two incumbents' eighteen wins became ties and five went to TimeDiT.** The full
transition, computed by diffing the pre-TimeDiT table against this one:

| Was | Became | Rows |
|---|---|---|
| CSDI | *tie* | 6 |
| CSDI | TimeDiT | 2 |
| CSDI | CSDI | 2 |
| LS4 | *tie* | 4 |
| LS4 | TimeDiT | 3 |
| LS4 | LS4 | 1 |
| *tie* | TimeDiT | 2 |
| *tie* | *tie* | 3 |

**Nothing moved between CSDI and LS4 — not one row.** Every change is a row where TimeDiT
became the runner-up or the leader. That is the same mechanism as §1 acting in the other
direction, and it is why neither experiment's tally should be read as a ranking of methods.

The ties are still where the interest is: **two of the four PRIMARY metrics** — regime-proportion
TVD and the leverage-curve RMSE — have a paired CI containing zero. Half the panel the protocol
asks about is undecided at 5 seeds, whatever the means suggest. And as on A, no method is close:
the best of the three sits 1.5× to 30× above the floor across that panel.

<!-- BEGIN GENERATED: experiment B table pdf -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>Diffusion Transformer</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>TimeDiT</th><th>LS4</th></tr>
<tr><td><code>mixture_fidelity.generated_low_confidence_fraction</code></td><td>0.4104 ± 0.0102</td><td>0.371411 ± 0.0146</td><td>0.37334 ± 0.0141</td><td>0.174683 ± 0.00361</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_mean_max_probability</code></td><td>0.641929 ± 0.00439</td><td>0.666199 ± 0.00791</td><td>0.666752 ± 0.00569</td><td>0.792255 ± 0.00126</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_mean_posterior_entropy</code></td><td>0.911906 ± 0.0109</td><td>0.853934 ± 0.0191</td><td>0.856733 ± 0.0131</td><td>0.600259 ± 0.00188</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.0</code></td><td>0.296851 ± 0.01</td><td>0.188867 ± 0.0341</td><td>0.189526 ± 0.0394</td><td>0.123315 ± 0.00119</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.1</code></td><td>0.279663 ± 0.0114</td><td>0.175244 ± 0.0146</td><td>0.196851 ± 0.0411</td><td>0.125098 ± 0.000986</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.2</code></td><td>0.0312012 ± 0.00275</td><td>0.058667 ± 0.0333</td><td>0.00078125 ± 0.000557</td><td>0.118506 ± 0.00137</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.3</code></td><td>0.0351074 ± 0.00172</td><td>0.0440186 ± 0.0134</td><td>0.00168457 ± 0.000698</td><td>0.118774 ± 0.000707</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.4</code></td><td>0.0773926 ± 0.00431</td><td>0.170435 ± 0.0428</td><td>0.195288 ± 0.0291</td><td>0.135107 ± 0.00329</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.5</code></td><td>0.073999 ± 0.00335</td><td>0.12959 ± 0.0333</td><td>0.194507 ± 0.0136</td><td>0.13186 ± 0.00173</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.6</code></td><td>0.0964844 ± 0.0103</td><td>0.138696 ± 0.0251</td><td>0.110791 ± 0.0212</td><td>0.123413 ± 0.00377</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.7</code></td><td>0.109302 ± 0.0086</td><td>0.0944824 ± 0.027</td><td>0.110571 ± 0.0176</td><td>0.123926 ± 0.00261</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.mean_error</code></td><td>0.0226021 ± 0.0195</td><td>0.0916107 ± 0.052</td><td>0.0180758 ± 0.0118</td><td>0.00217876 ± 0.0024</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q05_error</code></td><td>0.0099726 ± 0.00234</td><td>0.006996 ± 0.00482</td><td>0.0980166 ± 0.0245</td><td>0.0007326 ± 0.000494</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q95_error</code></td><td>0.0144606 ± 0.00559</td><td>0.0101046 ± 0.00557</td><td>0.0939972 ± 0.00958</td><td>0.000924 ± 0.000753</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.std_ratio</code></td><td>0.85191 ± 0.00828</td><td>0.884255 ± 0.0192</td><td>0.752922 ± 0.0256</td><td>0.998133 ± 0.00167</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.support_normalized_wasserstein</code></td><td>0.0751098 ± 0.00394</td><td>0.0697717 ± 0.0124</td><td>0.112548 ± 0.011</td><td>0.00293792 ± 0.00066</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.wasserstein</code></td><td>0.148717 ± 0.0078</td><td>0.138148 ± 0.0245</td><td>0.222845 ± 0.0218</td><td>0.00581707 ± 0.00131</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.mean_error</code></td><td>0.0128947 ± 0.000916</td><td>0.00433576 ± 0.00149</td><td>0.00890553 ± 0.00366</td><td>0.000118867 ± 8.97e-05</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q05_error</code></td><td>3.46945e-19 ± 7.76e-19</td><td>0 ± 0</td><td>0.000138667 ± 3.48e-05</td><td>0 ± 0</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q95_error</code></td><td>0.000837333 ± 2.39e-05</td><td>6.4e-05 ± 3.04e-05</td><td>6.4e-05 ± 5.84e-05</td><td>2.77556e-18 ± 6.21e-18</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.std_ratio</code></td><td>0.8367 ± 0.00879</td><td>0.952414 ± 0.0246</td><td>0.919967 ± 0.017</td><td>0.999393 ± 0.00205</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.support_normalized_wasserstein</code></td><td>0.161236 ± 0.0114</td><td><b>0.0597516 ± 0.009</b></td><td>0.114659 ± 0.0387</td><td>0.00237222 ± 0.000584</td><td>TimeDiT</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.wasserstein</code></td><td>0.0128989 ± 0.000915</td><td><b>0.00478013 ± 0.00072</b></td><td>0.00917275 ± 0.0031</td><td>0.000189777 ± 4.68e-05</td><td>TimeDiT</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.mean_error</code></td><td>0.105508 ± 0.00579</td><td>0.0789575 ± 0.0244</td><td>0.179389 ± 0.0252</td><td>0.00209811 ± 0.00139</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q05_error</code></td><td>0.0056275 ± 0.00102</td><td>0.00675 ± 0.000685</td><td><b>0.00235 ± 0.000945</b></td><td>0.0001225 ± 0.000125</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q95_error</code></td><td>0.0556775 ± 0.00377</td><td><b>0.02025 ± 0.00375</b></td><td>0.0602775 ± 0.0388</td><td>0.00025 ± 0.000177</td><td>TimeDiT</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.std_ratio</code></td><td>0.677446 ± 0.00693</td><td>0.7727 ± 0.0198</td><td>0.724715 ± 0.0599</td><td>1.00183 ± 0.000666</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.support_normalized_wasserstein</code></td><td>0.179548 ± 0.00443</td><td><b>0.13798 ± 0.0161</b></td><td>0.239191 ± 0.0337</td><td>0.00360015 ± 0.00129</td><td>TimeDiT</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.wasserstein</code></td><td>0.134661 ± 0.00332</td><td><b>0.103485 ± 0.0121</b></td><td>0.179393 ± 0.0252</td><td>0.00270012 ± 0.000969</td><td>TimeDiT</td></tr>
<tr><td><code>mixture_fidelity.regime_proportion_tvd</code></td><td>0.33042 ± 0.0143</td><td>0.194141 ± 0.0338</td><td>0.26687 ± 0.039</td><td>0.00847168 ± 0.00129</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.target_low_confidence_fraction</code></td><td>0.171899 ± 5.46e-05</td><td>0.171875 ± 0</td><td>0.171875 ± 8.63e-05</td><td>0.171875 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_mean_max_probability</code></td><td>0.792163 ± 0</td><td>0.792163 ± 0</td><td>0.792163 ± 0</td><td>0.792163 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_mean_posterior_entropy</code></td><td>0.600794 ± 0</td><td>0.600794 ± 0</td><td>0.600794 ± 0</td><td>0.600794 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.0</code></td><td>0.123169 ± 0</td><td>0.123169 ± 0</td><td>0.123169 ± 0</td><td>0.123169 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.1</code></td><td>0.122925 ± 0</td><td>0.122925 ± 0</td><td>0.122925 ± 0</td><td>0.122925 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.2</code></td><td>0.119629 ± 0</td><td>0.119629 ± 0</td><td>0.119629 ± 0</td><td>0.119629 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.3</code></td><td>0.120361 ± 0</td><td>0.120361 ± 0</td><td>0.120361 ± 0</td><td>0.120361 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.4</code></td><td>0.133057 ± 0</td><td>0.133057 ± 0</td><td>0.133057 ± 0</td><td>0.133057 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.5</code></td><td>0.130249 ± 0</td><td>0.130249 ± 0</td><td>0.130249 ± 0</td><td>0.130249 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.6</code></td><td>0.124268 ± 0</td><td>0.124268 ± 0</td><td>0.124268 ± 0</td><td>0.124268 ± 0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.7</code></td><td>0.126343 ± 0</td><td>0.126343 ± 0</td><td>0.126343 ± 0</td><td>0.126343 ± 0</td><td>—</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>2323.6 ± 41.2</td><td>2795.2 ± 152</td><td>3055.2 ± 248</td><td>2941.8 ± 24.5</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>0.568307 ± 0.0122</td><td>0.764603 ± 0.0758</td><td>0.913301 ± 0.0866</td><td>0.808363 ± 0.00181</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>0.470814 ± 0.018</td><td>0.762712 ± 0.153</td><td>0.926082 ± 0.116</td><td>0.716746 ± 0.00338</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>observable_fidelity.abs_return_acf_rmse_lags_1_50</code></td><td><b>0.011025 ± 0.00362</b></td><td>0.0320231 ± 0.0152</td><td>0.104407 ± 0.00656</td><td>0.0034538 ± 0.00121</td><td>CSDI</td></tr>
<tr><td><code>observable_fidelity.excess_kurtosis_error</code></td><td>2.55432 ± 0.737</td><td><b>1.60599 ± 0.917</b></td><td>2.71879 ± 0.223</td><td>0.143903 ± 0.185</td><td>TimeDiT</td></tr>
<tr><td><code>observable_fidelity.leverage_curve_rmse_lags_0_20</code></td><td>0.0058035 ± 0.00238</td><td>0.0119523 ± 0.00353</td><td>0.00851555 ± 0.00264</td><td>0.00389969 ± 0.000993</td><td><i>tie</i></td></tr>
<tr><td><code>observable_fidelity.realized_volatility_wasserstein</code></td><td>0.0586454 ± 0.00303</td><td><b>0.0224224 ± 0.00418</b></td><td>0.0368266 ± 0.00187</td><td>0.00154572 ± 0.000216</td><td>TimeDiT</td></tr>
<tr><td><code>observable_fidelity.return_std_error</code></td><td>0.00403557 ± 0.000194</td><td>0.00125334 ± 0.000816</td><td>0.00149136 ± 0.000247</td><td>6.30604e-05 ± 3.92e-05</td><td><i>tie</i></td></tr>
<tr><td><code>observable_fidelity.squared_return_acf_rmse_lags_1_50</code></td><td><b>0.0169132 ± 0.00344</b></td><td>0.0417153 ± 0.00975</td><td>0.0369782 ± 0.00749</td><td>0.00729821 ± 0.00266</td><td>CSDI</td></tr>
<tr><td><code>observable_fidelity.terminal_log_price_ks</code></td><td>0.0449219 ± 0.0105</td><td>0.0856934 ± 0.015</td><td>0.0519287 ± 0.0171</td><td>0.0102539 ± 0.00242</td><td><i>tie</i></td></tr>
</table>

<p><b>Paired, aligned-seed: CSDI vs TimeDiT</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>mixture_fidelity.generated_low_confidence_fraction</code></td><td>+0.05249, +0.04675, +0.01294, +0.02576, +0.05701</td><td>0.0389893 ± 0.0188</td><td>±0.0234</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_max_probability</code></td><td>-0.03062, -0.03314, -0.01443, -0.01614, -0.02702</td><td>-0.0242696 ± 0.00851</td><td>±0.0106</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_posterior_entropy</code></td><td>+0.05965, +0.06982, +0.04452, +0.03492, +0.08096</td><td>0.0579718 ± 0.0186</td><td>±0.0231</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.0</code></td><td>+0.09253, +0.1493, +0.1519, +0.07288, +0.07336</td><td>0.107983 ± 0.0397</td><td>±0.0493</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.1</code></td><td>+0.1154, +0.1212, +0.1178, +0.1083, +0.05945</td><td>0.104419 ± 0.0256</td><td>±0.0318</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.2</code></td><td>-0.01379, +0.006836, -0.01465, -0.08594, -0.02979</td><td>-0.0274658 ± 0.0352</td><td>±0.0437</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.3</code></td><td>-0.004639, -0.0332, -0.01172, +0.004639, +0.0003662</td><td>-0.00891113 ± 0.0149</td><td>±0.0185</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.4</code></td><td>-0.1134, -0.1324, -0.1282, -0.02576, -0.06543</td><td>-0.093042 ± 0.0461</td><td>±0.0572</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.5</code></td><td>-0.04089, -0.09094, -0.0946, -0.0304, -0.02112</td><td>-0.0555908 ± 0.0347</td><td>±0.0431</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.6</code></td><td>-0.07373, -0.02026, -0.05713, -0.01965, -0.04028</td><td>-0.0422119 ± 0.0235</td><td>±0.0292</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.7</code></td><td>+0.03857, -0.0004883, +0.03662, -0.02405, +0.02344</td><td>0.0148193 ± 0.0267</td><td>±0.0332</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.mean_error</code></td><td>-0.1184, +0.002175, -0.06177, -0.0846, -0.08243</td><td>-0.0690086 ± 0.0447</td><td>±0.0555</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q05_error</code></td><td>+0.01122, -0.00066, -0.00132, -0.002277, +0.00792</td><td>0.0029766 ± 0.00616</td><td>±0.00764</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q95_error</code></td><td>-0.001683, -0.00594, +0.00396, +0.00594, +0.0195</td><td>0.004356 ± 0.00968</td><td>±0.012</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.std_ratio</code></td><td>-0.01525, -0.03444, -0.01209, -0.03361, -0.06633</td><td>-0.0323444 ± 0.0216</td><td>±0.0268</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.support_normalized_wasserstein</code></td><td>-0.007845, +0.01962, -0.001009, +0.007297, +0.008629</td><td>0.00533807 ± 0.0104</td><td>±0.0129</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.wasserstein</code></td><td>-0.01553, +0.03884, -0.001997, +0.01445, +0.01709</td><td>0.0105694 ± 0.0206</td><td>±0.0256</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.mean_error</code></td><td>+0.0108, +0.008419, +0.008394, +0.007097, +0.008087</td><td>0.00855895 ± 0.00136</td><td>±0.00169</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q05_error</code></td><td>+0, +1.735e-18, +0, +0, +0</td><td>3.46945e-19 ± 7.76e-19</td><td>±9.63e-19</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q95_error</code></td><td>+0.0008, +0.0008, +0.0008, +0.0006933, +0.0007733</td><td>0.000773333 ± 4.62e-05</td><td>±5.73e-05</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.std_ratio</code></td><td>-0.1002, -0.1446, -0.1528, -0.0739, -0.1071</td><td>-0.115714 ± 0.0327</td><td>±0.0406</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.support_normalized_wasserstein</code></td><td>+0.1101, +0.1053, +0.105, +0.08613, +0.101</td><td>0.101485 ± 0.00917</td><td>±0.0114</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.wasserstein</code></td><td>+0.008808, +0.00842, +0.008397, +0.006891, +0.008078</td><td>0.00811878 ± 0.000734</td><td>±0.000911</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.mean_error</code></td><td>+0.01518, +0.01067, +0.01469, +0.06435, +0.02787</td><td>0.0265506 ± 0.0221</td><td>±0.0274</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q05_error</code></td><td>-0.0008625, -0.00125, -0.00025, -0.00325, -5.551e-17</td><td>-0.0011225 ± 0.00129</td><td>±0.0016</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q95_error</code></td><td>+0.028, +0.03739, +0.038, +0.039, +0.03475</td><td>0.0354275 ± 0.00444</td><td>±0.00551</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.std_ratio</code></td><td>-0.0903, -0.1054, -0.09346, -0.1186, -0.0685</td><td>-0.0952544 ± 0.0187</td><td>±0.0232</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.support_normalized_wasserstein</code></td><td>+0.03778, +0.03836, +0.03652, +0.0676, +0.02757</td><td>0.0415679 ± 0.0152</td><td>±0.0189</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.wasserstein</code></td><td>+0.02834, +0.02877, +0.02739, +0.0507, +0.02068</td><td>0.0311759 ± 0.0114</td><td>±0.0142</td></tr>
<tr><td><code>mixture_fidelity.regime_proportion_tvd</code></td><td>+0.1128, +0.1631, +0.1353, +0.1771, +0.09314</td><td>0.136279 ± 0.0346</td><td>±0.043</td></tr>
<tr><td><code>mixture_fidelity.target_low_confidence_fraction</code></td><td>+0, +0, +0, +0, +0.0001221</td><td>2.44141e-05 ± 5.46e-05</td><td>±6.78e-05</td></tr>
<tr><td><code>mixture_fidelity.target_mean_max_probability</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_mean_posterior_entropy</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.0</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.1</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.3</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.4</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.5</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.6</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.7</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-370, -657, -686, -310, -335</td><td>-471.6 ± 184</td><td>±228</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.1664, -0.2673, -0.3072, -0.1303, -0.1104</td><td>-0.196296 ± 0.0866</td><td>±0.107</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.2747, -0.4532, -0.4782, -0.1388, -0.1145</td><td>-0.291898 ± 0.17</td><td>±0.211</td></tr>
<tr><td><code>observable_fidelity.abs_return_acf_rmse_lags_1_50</code></td><td>-0.03076, -0.0289, -0.02297, -0.0258, +0.003442</td><td>-0.0209981 ± 0.014</td><td>±0.0174</td></tr>
<tr><td><code>observable_fidelity.excess_kurtosis_error</code></td><td>+0.9803, +1.201, -0.1317, +1.154, +1.538</td><td>0.948327 ± 0.637</td><td>±0.79</td></tr>
<tr><td><code>observable_fidelity.leverage_curve_rmse_lags_0_20</code></td><td>-0.009247, -0.002536, -0.005657, -0.01266, -0.0006467</td><td>-0.00614879 ± 0.00489</td><td>±0.00607</td></tr>
<tr><td><code>observable_fidelity.realized_volatility_wasserstein</code></td><td>+0.03739, +0.04328, +0.0399, +0.03291, +0.02763</td><td>0.036223 ± 0.00611</td><td>±0.00759</td></tr>
<tr><td><code>observable_fidelity.return_std_error</code></td><td>+0.002447, +0.004033, +0.003615, +0.00198, +0.001837</td><td>0.00278223 ± 0.000988</td><td>±0.00123</td></tr>
<tr><td><code>observable_fidelity.squared_return_acf_rmse_lags_1_50</code></td><td>-0.02668, -0.03221, -0.03404, -0.02355, -0.007527</td><td>-0.0248021 ± 0.0105</td><td>±0.0131</td></tr>
<tr><td><code>observable_fidelity.terminal_log_price_ks</code></td><td>-0.07751, -0.0293, -0.02197, -0.05164, -0.02344</td><td>-0.0407715 ± 0.0237</td><td>±0.0295</td></tr>
</table>

<p><b>Paired, aligned-seed: CSDI vs LS4</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>mixture_fidelity.generated_low_confidence_fraction</code></td><td>+0.03967, +0.01172, +0.03857, +0.02832, +0.06702</td><td>0.0370605 ± 0.0202</td><td>±0.025</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_max_probability</code></td><td>-0.02461, -0.01574, -0.02304, -0.02235, -0.03837</td><td>-0.0248226 ± 0.0083</td><td>±0.0103</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_posterior_entropy</code></td><td>+0.05053, +0.0264, +0.05518, +0.05973, +0.08403</td><td>0.055173 ± 0.0206</td><td>±0.0256</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.0</code></td><td>+0.1195, +0.1326, +0.1276, +0.02698, +0.13</td><td>0.107324 ± 0.0452</td><td>±0.0561</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.1</code></td><td>+0.1036, +0.1283, +0.09753, +0.01013, +0.07446</td><td>0.0828125 ± 0.0449</td><td>±0.0558</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.2</code></td><td>+0.02869, +0.03345, +0.03015, +0.02734, +0.03247</td><td>0.0304199 ± 0.00255</td><td>±0.00316</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.3</code></td><td>+0.03394, +0.03235, +0.03162, +0.03601, +0.0332</td><td>0.0334229 ± 0.00169</td><td>±0.0021</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.4</code></td><td>-0.1414, -0.1434, -0.1254, -0.0647, -0.1146</td><td>-0.117896 ± 0.032</td><td>±0.0397</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.5</code></td><td>-0.1163, -0.1326, -0.1272, -0.09351, -0.1329</td><td>-0.120508 ± 0.0165</td><td>±0.0205</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.6</code></td><td>-0.03101, -0.04236, -0.02087, +0.03101, -0.008301</td><td>-0.0143066 ± 0.0283</td><td>±0.0351</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.7</code></td><td>+0.00293, -0.008301, -0.01343, +0.02673, -0.01428</td><td>-0.00126953 ± 0.0171</td><td>±0.0212</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.mean_error</code></td><td>+0.01707, -0.01054, +0.01646, -0.0342, +0.03385</td><td>0.00452632 ± 0.0269</td><td>±0.0333</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q05_error</code></td><td>-0.0792, -0.07722, -0.07326, -0.131, -0.07956</td><td>-0.088044 ± 0.0241</td><td>±0.03</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q95_error</code></td><td>-0.07626, -0.08484, -0.0792, -0.09768, -0.0597</td><td>-0.0795366 ± 0.0138</td><td>±0.0171</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.std_ratio</code></td><td>+0.08511, +0.1026, +0.1005, +0.1426, +0.06418</td><td>0.0989881 ± 0.0288</td><td>±0.0358</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.support_normalized_wasserstein</code></td><td>-0.03101, -0.03933, -0.0393, -0.05602, -0.02153</td><td>-0.0374381 ± 0.0127</td><td>±0.0158</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.wasserstein</code></td><td>-0.06141, -0.07787, -0.07782, -0.1109, -0.04263</td><td>-0.0741274 ± 0.0252</td><td>±0.0313</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.mean_error</code></td><td>+0.002126, +0.001534, +0.004594, +0.009121, +0.002571</td><td>0.00398917 ± 0.00309</td><td>±0.00384</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q05_error</code></td><td>-0.00016, -0.0001333, -0.00016, -8e-05, -0.00016</td><td>-0.000138667 ± 3.48e-05</td><td>±4.32e-05</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q95_error</code></td><td>+0.0008, +0.0008267, +0.0008, +0.00064, +0.0008</td><td>0.000773333 ± 7.54e-05</td><td>±9.36e-05</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.std_ratio</code></td><td>-0.0747, -0.06207, -0.1046, -0.08247, -0.09249</td><td>-0.0832668 ± 0.0163</td><td>±0.0203</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.support_normalized_wasserstein</code></td><td>+0.02663, +0.0192, +0.05747, +0.09738, +0.03221</td><td>0.046577 ± 0.0318</td><td>±0.0395</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.wasserstein</code></td><td>+0.002131, +0.001536, +0.004597, +0.00779, +0.002577</td><td>0.00372616 ± 0.00255</td><td>±0.00316</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.mean_error</code></td><td>-0.06628, -0.05164, -0.06251, -0.1194, -0.06958</td><td>-0.0738813 ± 0.0263</td><td>±0.0327</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q05_error</code></td><td>+0.003887, +0.0025, +0.00425, +0.0005, +0.00525</td><td>0.0032775 ± 0.00184</td><td>±0.00228</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q95_error</code></td><td>+0.0055, +0.01489, +0.01775, -0.07689, +0.01575</td><td>-0.0046 ± 0.0407</td><td>±0.0505</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.std_ratio</code></td><td>-0.0693, -0.07311, -0.08112, +0.06645, -0.07927</td><td>-0.0472692 ± 0.0637</td><td>±0.0791</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.support_normalized_wasserstein</code></td><td>-0.04541, -0.03369, -0.04727, -0.121, -0.05082</td><td>-0.059643 ± 0.0349</td><td>±0.0433</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.wasserstein</code></td><td>-0.03406, -0.02527, -0.03545, -0.09077, -0.03811</td><td>-0.0447323 ± 0.0262</td><td>±0.0325</td></tr>
<tr><td><code>mixture_fidelity.regime_proportion_tvd</code></td><td>+0.08105, +0.1003, +0.08838, -0.01758, +0.06555</td><td>0.0635498 ± 0.0471</td><td>±0.0584</td></tr>
<tr><td><code>mixture_fidelity.target_low_confidence_fraction</code></td><td>+0, +0, -0.0001221, +0, +0.0002441</td><td>2.44141e-05 ± 0.000134</td><td>±0.000166</td></tr>
<tr><td><code>mixture_fidelity.target_mean_max_probability</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_mean_posterior_entropy</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.0</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.1</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.3</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.4</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.5</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.6</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.7</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-857, -900, -832, -229, -840</td><td>-731.6 ± 282</td><td>±350</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.3879, -0.4028, -0.381, -0.1749, -0.3785</td><td>-0.344994 ± 0.0956</td><td>±0.119</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.4927, -0.5415, -0.5072, -0.2303, -0.5047</td><td>-0.455268 ± 0.127</td><td>±0.158</td></tr>
<tr><td><code>observable_fidelity.abs_return_acf_rmse_lags_1_50</code></td><td>-0.09792, -0.09759, -0.08375, -0.09939, -0.08826</td><td>-0.0933824 ± 0.00695</td><td>±0.00863</td></tr>
<tr><td><code>observable_fidelity.excess_kurtosis_error</code></td><td>+0.1462, +1.048, -0.2785, -1.303, -0.435</td><td>-0.164467 ± 0.858</td><td>±1.07</td></tr>
<tr><td><code>observable_fidelity.leverage_curve_rmse_lags_0_20</code></td><td>-0.002668, -0.008303, +0.001146, -0.004635, +0.0008995</td><td>-0.00271205 ± 0.00397</td><td>±0.00492</td></tr>
<tr><td><code>observable_fidelity.realized_volatility_wasserstein</code></td><td>+0.01961, +0.02329, +0.0269, +0.01912, +0.02017</td><td>0.0218187 ± 0.00328</td><td>±0.00407</td></tr>
<tr><td><code>observable_fidelity.return_std_error</code></td><td>+0.002508, +0.002832, +0.002988, +0.001878, +0.002515</td><td>0.00254421 ± 0.000426</td><td>±0.000529</td></tr>
<tr><td><code>observable_fidelity.squared_return_acf_rmse_lags_1_50</code></td><td>-0.01147, -0.03117, -0.01768, -0.02129, -0.01871</td><td>-0.020065 ± 0.00718</td><td>±0.00891</td></tr>
<tr><td><code>observable_fidelity.terminal_log_price_ks</code></td><td>-0.01465, -0.01831, +0.01819, -0.03662, +0.01636</td><td>-0.00700684 ± 0.0237</td><td>±0.0294</td></tr>
</table>

<p><b>Paired, aligned-seed: TimeDiT vs LS4</b> (PDF §5, seedwise differences and paired 95% CI)</p>
<table>
<tr><th>Metric</th><th>Seedwise differences</th><th>Mean diff</th><th>Paired 95% CI</th></tr>
<tr><td><code>mixture_fidelity.generated_low_confidence_fraction</code></td><td>-0.01282, -0.03503, +0.02563, +0.002563, +0.01001</td><td>-0.00192871 ± 0.0231</td><td>±0.0287</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_max_probability</code></td><td>+0.006005, +0.01741, -0.008616, -0.006206, -0.01135</td><td>-0.000553052 ± 0.012</td><td>±0.0149</td></tr>
<tr><td><code>mixture_fidelity.generated_mean_posterior_entropy</code></td><td>-0.009121, -0.04341, +0.01066, +0.02482, +0.003065</td><td>-0.00279885 ± 0.0258</td><td>±0.0321</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.0</code></td><td>+0.02698, -0.01672, -0.02429, -0.0459, +0.05664</td><td>-0.00065918 ± 0.0416</td><td>±0.0516</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.1</code></td><td>-0.01172, +0.00708, -0.02026, -0.09814, +0.01501</td><td>-0.0216064 ± 0.0451</td><td>±0.0559</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.2</code></td><td>+0.04248, +0.02661, +0.0448, +0.1133, +0.06226</td><td>0.0578857 ± 0.0334</td><td>±0.0415</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.3</code></td><td>+0.03857, +0.06555, +0.04333, +0.03137, +0.03284</td><td>0.042334 ± 0.0138</td><td>±0.0172</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.4</code></td><td>-0.02795, -0.01099, +0.002808, -0.03894, -0.04919</td><td>-0.0248535 ± 0.021</td><td>±0.026</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.5</code></td><td>-0.07544, -0.04163, -0.03259, -0.06311, -0.1118</td><td>-0.064917 ± 0.0312</td><td>±0.0388</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.6</code></td><td>+0.04272, -0.02209, +0.03625, +0.05066, +0.03198</td><td>0.0279053 ± 0.0288</td><td>±0.0358</td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.7</code></td><td>-0.03564, -0.007812, -0.05005, +0.05078, -0.03772</td><td>-0.0160889 ± 0.0404</td><td>±0.0502</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.mean_error</code></td><td>+0.1355, -0.01272, +0.07822, +0.0504, +0.1163</td><td>0.0735349 ± 0.0584</td><td>±0.0726</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q05_error</code></td><td>-0.09042, -0.07656, -0.07194, -0.1287, -0.08748</td><td>-0.0910206 ± 0.0224</td><td>±0.0278</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q95_error</code></td><td>-0.07458, -0.0789, -0.08316, -0.1036, -0.0792</td><td>-0.0838926 ± 0.0114</td><td>±0.0142</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.std_ratio</code></td><td>+0.1004, +0.137, +0.1126, +0.1762, +0.1305</td><td>0.131333 ± 0.029</td><td>±0.036</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.support_normalized_wasserstein</code></td><td>-0.02317, -0.05895, -0.03829, -0.06331, -0.03016</td><td>-0.0427761 ± 0.0177</td><td>±0.0219</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.wasserstein</code></td><td>-0.04588, -0.1167, -0.07582, -0.1254, -0.05971</td><td>-0.0846968 ± 0.035</td><td>±0.0434</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.mean_error</code></td><td>-0.008673, -0.006884, -0.0038, +0.002024, -0.005516</td><td>-0.00456978 ± 0.0041</td><td>±0.00509</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q05_error</code></td><td>-0.00016, -0.0001333, -0.00016, -8e-05, -0.00016</td><td>-0.000138667 ± 3.48e-05</td><td>±4.32e-05</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q95_error</code></td><td>+1.388e-17, +2.667e-05, +1.388e-17, -5.333e-05, +2.667e-05</td><td>1.11022e-17 ± 3.27e-05</td><td>±4.05e-05</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.std_ratio</code></td><td>+0.02547, +0.08251, +0.04822, -0.008563, +0.0146</td><td>0.0324474 ± 0.0347</td><td>±0.043</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.support_normalized_wasserstein</code></td><td>-0.08347, -0.08605, -0.0475, +0.01124, -0.06876</td><td>-0.0549078 ± 0.04</td><td>±0.0497</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.wasserstein</code></td><td>-0.006677, -0.006884, -0.0038, +0.0008993, -0.005501</td><td>-0.00439262 ± 0.0032</td><td>±0.00398</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.mean_error</code></td><td>-0.08146, -0.06231, -0.07719, -0.1837, -0.09745</td><td>-0.100432 ± 0.0482</td><td>±0.0599</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q05_error</code></td><td>+0.00475, +0.00375, +0.0045, +0.00375, +0.00525</td><td>0.0044 ± 0.000652</td><td>±0.000809</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q95_error</code></td><td>-0.0225, -0.0225, -0.02025, -0.1159, -0.019</td><td>-0.0400275 ± 0.0424</td><td>±0.0527</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.std_ratio</code></td><td>+0.021, +0.03228, +0.01234, +0.1851, -0.01077</td><td>0.0479852 ± 0.0782</td><td>±0.0971</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.support_normalized_wasserstein</code></td><td>-0.08319, -0.07205, -0.08379, -0.1886, -0.07839</td><td>-0.101211 ± 0.0491</td><td>±0.0609</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.wasserstein</code></td><td>-0.0624, -0.05404, -0.06285, -0.1415, -0.05879</td><td>-0.0759082 ± 0.0368</td><td>±0.0457</td></tr>
<tr><td><code>mixture_fidelity.regime_proportion_tvd</code></td><td>-0.03174, -0.06274, -0.04688, -0.1947, -0.02759</td><td>-0.0727295 ± 0.0696</td><td>±0.0864</td></tr>
<tr><td><code>mixture_fidelity.target_low_confidence_fraction</code></td><td>+0, +0, -0.0001221, +0, +0.0001221</td><td>0 ± 8.63e-05</td><td>±0.000107</td></tr>
<tr><td><code>mixture_fidelity.target_mean_max_probability</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_mean_posterior_entropy</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.0</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.1</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.2</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.3</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.4</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.5</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.6</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.7</code></td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>-487, -243, -146, +81, -505</td><td>-260 ± 246</td><td>±305</td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>-0.2215, -0.1355, -0.0738, -0.04453, -0.2681</td><td>-0.148698 ± 0.0951</td><td>±0.118</td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>-0.218, -0.08829, -0.029, -0.09147, -0.3901</td><td>-0.163369 ± 0.144</td><td>±0.179</td></tr>
<tr><td><code>observable_fidelity.abs_return_acf_rmse_lags_1_50</code></td><td>-0.06716, -0.06869, -0.06078, -0.07359, -0.0917</td><td>-0.0723843 ± 0.0117</td><td>±0.0146</td></tr>
<tr><td><code>observable_fidelity.excess_kurtosis_error</code></td><td>-0.8341, -0.1533, -0.1468, -2.457, -1.973</td><td>-1.11279 ± 1.06</td><td>±1.31</td></tr>
<tr><td><code>observable_fidelity.leverage_curve_rmse_lags_0_20</code></td><td>+0.006579, -0.005767, +0.006804, +0.008021, +0.001546</td><td>0.00343675 ± 0.00571</td><td>±0.00709</td></tr>
<tr><td><code>observable_fidelity.realized_volatility_wasserstein</code></td><td>-0.01778, -0.01998, -0.013, -0.01379, -0.007467</td><td>-0.0144042 ± 0.00482</td><td>±0.00598</td></tr>
<tr><td><code>observable_fidelity.return_std_error</code></td><td>+6.165e-05, -0.001201, -0.0006263, -0.0001022, +0.0006775</td><td>-0.000238024 ± 0.000711</td><td>±0.000883</td></tr>
<tr><td><code>observable_fidelity.squared_return_acf_rmse_lags_1_50</code></td><td>+0.0152, +0.001039, +0.01637, +0.002262, -0.01119</td><td>0.00473708 ± 0.0114</td><td>±0.0141</td></tr>
<tr><td><code>observable_fidelity.terminal_log_price_ks</code></td><td>+0.06287, +0.01099, +0.04016, +0.01501, +0.03979</td><td>0.0337646 ± 0.0212</td><td>±0.0263</td></tr>
</table>
<!-- END GENERATED -->

Full detail — including the per-parameter Wasserstein breakdown, the eight-regime coverage
table, the observable-fidelity comparison, the A1–A34 battery and the curve-shape suite — is
in [`experiment_B/README.md`](experiment_B/README.md).

---

## PDF §5 disclosure — what every comparison table must state

PDF §5 requires six items alongside any comparison table. They are identical for both
experiments except where marked:

| # | §5 requirement | Value |
|---|---|---|
| 1 | Exact training, validation and test files | Under `dataset/Heston/new_experiments/experiment_A/` and `…/experiment_B/`: training `train.npy`, validation `disc.npy`, test `test.npy`. Evaluator-only companions: `test_sigma.npy` (A), `test_labels.npy` + `oracle.joblib` (B). Floor: `perfect_floor/floor_seed1000.npy` … `floor_seed1004.npy` |
| 2 | Generated bank size and all model seeds | 8192 × 128 per seed; seeds **0, 1, 2, 3, 4** for all three methods in both experiments. The seeds are *aligned*, which is the only reason the `Seedwise differences` and `Paired 95% CI` columns in the pairwise blocks above are the §5 quantity rather than a substitute for it |
| 3 | Official code, and its revision | **Two yes, one no.** CSDI `methods/CSDI` @ `4189d37`; LS4 `methods/LS4/code` @ `27df71e` with experiment wrapper `train_ls4_experiment.py` @ `ba7c748`. **TimeDiT has no official release**: `methods/TimeDiT/code` @ `1312f00` is ours, built on the basis `arXiv:2409.02322 App. C -- no official release; DiT-S backbone reimplemented from facebookresearch/DiT`. Recorded per bank in `generation_manifest.json → model.official_implementation / model.source_revision / model.reimplementation_basis` |
| 4 | Hyperparameters: defaults or validation-selected | **None was validation-selected on these datasets.** CSDI and LS4: `hyperparameter_origin = "official-default"` (released `base.yaml`, released `solar_weekly` preset). TimeDiT has no released config, so its origin is `paper-reproduction-selected (Sine+Stocks HP search in methods/TimeDiT/paper_reimplementation; no tuning on Heston, train.npy or disc.npy)` — that search ran on the paper's own Sine and Stocks data, before this protocol. Experiment B reuses Experiment A's settings unchanged for all three. `disc.npy` was used for early stopping and model selection only, never for hyperparameter search |
| 5 | Trainable parameters, training time, generation time, hardware | CSDI 412,945 params — A: 2387 s train / 15.6 s generate, B: 3044 s / 13.1 s. TimeDiT 32,463,745 params — A: 5105 s / 3071.8 s, B: 5271 s / 3163.7 s. LS4 2,146,857 params — A: 1643 s / 9.1 s, B: 1424 s / 9.4 s. All per seed. Hardware: 1× A100-SXM4-80GB per run, 2× AMD EPYC 7763 host; 8 cores pinned per run for CSDI and LS4, 4 for TimeDiT, never more than 16 cores in total |
| 6 | Number and reason for failed runs | **Zero, across all 30 runs.** `failure_information.failed_or_unstable = false`, `first_nan_epoch = null`, `nan_in_bank = false` in every manifest. No seed was replaced, retried or dropped (§1.5) |

One deviation the PDF does not ask about but which is disclosed anyway: **Experiment B's scoring
oracle is a local refit** (`a36d64eb…` vs the PDF's `54c3f2c9…` in §6). It passes its accuracy
gate at 0.9094 ≥ 0.90 and it scored all three methods identically, so every paired CI in
Section 2 is instrument-consistent — but absolute distances from the floor carry the caveat.
Detail in [`experiment_B/README.md`](experiment_B/README.md) §1.

Item 5 deserves a second look. **TimeDiT's generation time is roughly 240× CSDI's and 340×
LS4's** — 51 to 53 minutes per bank against 13 and 9 *seconds* — because its sampler is
ancestral DDPM over the full 1000-step chain with no stride, and there are 8192 paths per bank.
Nothing in the protocol scores cost, so this appears nowhere in the tallies above; a reader
comparing methods for use rather than for fidelity should weigh it.

Items 3, 4 and 6 are re-derived from the thirty `generation_manifest.json` files, not typed:

```bash
python tools/check_pdf5_disclosure.py --experiment A --readme README.md
python tools/check_pdf5_disclosure.py --experiment B --readme README.md
```

## Regenerating this page

Every table above is emitted by a tool and pasted verbatim; no number here was typed by hand.
To refresh them after a re-run:

```bash
cd results/new_experiments
python tools/make_comparison_tables.py --inject README.md
python tools/make_comparison_tables.py --inject experiment_A/README.md
python tools/make_comparison_tables.py --inject experiment_B/README.md
```

The tool shells out to `tools/aggregate_pdf_metrics.py` and `tools/make_metrics_tables.py` —
the same producers whose output `tools/check_readme_values.py` verifies cell-for-cell against
the method-level READMEs — so a cell here cannot disagree with the checked cell it mirrors.

## Layout

```
results/new_experiments/
├── README.md                     <- this file: the two PDF comparison tables
├── guideline_new_experiment.md   <- the protocol these runs must satisfy
├── README_TEMPLATE.md            <- template for a method-level README
├── open_questions_to_protocol_author.md
├── tools/                        <- table producers and the value checker
├── experiment_A/
│   ├── README.md                 <- A: PDF + A1-A34 + curve shape + plots + dataset
│   ├── CSDI/README.md            <- Diffusion
│   ├── TimeDiT/README.md         <- Diffusion Transformer
│   ├── LS4/README.md             <- VAE
│   └── perfect_floor/
└── experiment_B/
    ├── README.md                 <- B: PDF + A1-A34 + curve shape + plots + dataset
    ├── CSDI/README.md            <- Diffusion
    ├── TimeDiT/README.md         <- Diffusion Transformer
    ├── LS4/README.md             <- VAE
    └── perfect_floor/
```
