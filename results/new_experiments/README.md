# New experiments — cross-method comparison

Two experiments, two generative methods each, five seeds each, one shared perfect-floor
reference. This page carries **only the protocol-evaluator (PDF) comparison table for each
experiment**. Everything else — the A1–A34 battery, the curve-shape suite, the stylised
plots, the dataset description — lives one level down:

| | Experiment | Comparison README | Method READMEs |
|---|---|---|---|
| **A** | Delayed drawdown memory | [`experiment_A/README.md`](experiment_A/README.md) | [CSDI](experiment_A/CSDI/README.md) · [LS4](experiment_A/LS4/README.md) |
| **B** | Balanced Heston parameter mixture | [`experiment_B/README.md`](experiment_B/README.md) | [CSDI](experiment_B/CSDI/README.md) · [LS4](experiment_B/LS4/README.md) |

**How to read the `Winner` column.** It is not "smaller number wins". Each metric has a
reference value it is supposed to reach — `0` for an error or distance, the matching `target_*`
row for a memory or mixture statistic — and the winner is the method whose mean sits closer to
that reference. On top of that, a winner is only declared when the **paired 95 % confidence
interval over the five aligned seeds excludes zero** (PDF §5). When it does not, the row reads
*tie* however far apart the two means look: the gap is not reproducible seed by seed.

Two classes of row carry **no winner at all**:

* `—` — a **constant**. `target_*` rows are one reading of `test.npy`, printed identically in
  both columns. Not a contest.
* `diagnostic (§…)` — a **raw diagnostic**. PDF §5: *"All displayed fidelity metrics are errors
  or distances, so lower is better, except explicitly identified raw diagnostics such as
  standard-deviation ratio, posterior confidence, group means, and novelty."* §4 adds that
  novelty *"must not be combined with fidelity metrics into a score"*; §3.3 that posterior
  entropy, maximum probability and low-confidence fraction *"are diagnostics, not separate
  winner-selection criteria."* These rows print their value and the clause that exempts them,
  and are excluded from every tally on this page.

**Bold** marks the better method on that row. The `Perfect` column is the floor — a generator
that draws from the true data-generating process — and is excluded from the contest; it is
there to say how far *both* methods still are from the achievable optimum.

**No aggregate score is constructed** (PDF §3.4). The tallies below count independently-decided
rows so a reader can see the split. They are not an index and they are not weighted; the
PRIMARY panels in the two experiment READMEs are what the protocol actually asks about.

---

## 1. Experiment A — delayed drawdown memory

Of the 30 rows: **9 constants**, **5 raw diagnostics**, and **16 contested fidelity metrics**.
Of those 16 — **LS4 wins 13, CSDI wins 0, 3 are ties.** Experiment A is not a close call: CSDI
does not take a single contested fidelity metric.

But neither method solves it. On the four PRIMARY metrics of PDF §2, LS4 is still 2.6× to 49×
worse than a fresh draw from the true DGP. This is a comparison between two failures of
different magnitude.

<!-- BEGIN GENERATED: experiment A table pdf -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Seedwise differences</th><th rowspan="2">Mean diff</th><th rowspan="2">Paired 95% CI</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>LS4</th></tr>
<tr><td><code>errors.abs_return_acf_rmse_lags_1_50</code></td><td>0.0189235 ± 0.000518</td><td><b>0.0144903 ± 0.0024</b></td><td>0.00156102 ± 0.000141</td><td>-8.005e-05, +0.006656, +0.007086, +0.003526, +0.004978</td><td>0.00443316 ± 0.00289</td><td>±0.00359</td><td>LS4</td></tr>
<tr><td><code>errors.early_history_incremental_r2_error</code></td><td>0.227474 ± 0.0151</td><td><b>0.118178 ± 0.0154</b></td><td>0.00641167 ± 0.00544</td><td>+0.09389, +0.1202, +0.1052, +0.08103, +0.1462</td><td>0.109296 ± 0.0252</td><td>±0.0312</td><td>LS4</td></tr>
<tr><td><code>errors.early_hit_rate_error</code></td><td>0.161279 ± 0.0182</td><td><b>0.0081543 ± 0.00691</b></td><td>0.00310059 ± 0.00224</td><td>+0.1439, +0.1561, +0.1696, +0.1287, +0.1674</td><td>0.153125 ± 0.0171</td><td>±0.0212</td><td>LS4</td></tr>
<tr><td><code>errors.excess_kurtosis_error</code></td><td>0.59727 ± 0.0408</td><td><b>0.302096 ± 0.158</b></td><td>0.023094 ± 0.0118</td><td>-0.01828, +0.3813, +0.3317, +0.4804, +0.3007</td><td>0.295175 ± 0.188</td><td>±0.233</td><td>LS4</td></tr>
<tr><td><code>errors.future_rv_hit_gap_error</code></td><td>0.0830031 ± 0.00635</td><td><b>0.0503776 ± 0.00241</b></td><td>0.00102294 ± 0.000516</td><td>+0.0337, +0.0359, +0.02921, +0.02199, +0.04233</td><td>0.0326255 ± 0.0076</td><td>±0.00943</td><td>LS4</td></tr>
<tr><td><code>errors.future_rv_wasserstein</code></td><td>0.0576958 ± 0.00278</td><td><b>0.0119597 ± 0.00142</b></td><td>0.00123331 ± 0.00021</td><td>+0.04186, +0.0438, +0.04959, +0.04715, +0.04628</td><td>0.0457361 ± 0.003</td><td>±0.00372</td><td>LS4</td></tr>
<tr><td><code>errors.return_std_error</code></td><td>0.00290882 ± 0.000109</td><td><b>0.000202294 ± 0.00012</b></td><td>2.79631e-05 ± 1.44e-05</td><td>+0.002797, +0.00257, +0.002847, +0.002778, +0.002542</td><td>0.00270652 ± 0.00014</td><td>±0.000174</td><td>LS4</td></tr>
<tr><td><code>errors.squared_return_acf_rmse_lags_1_50</code></td><td>0.00867602 ± 0.00106</td><td>0.00776208 ± 0.00602</td><td>0.00161632 ± 0.000193</td><td>-0.01034, +0.00432, +0.006376, +0.0003482, +0.003862</td><td>0.000913942 ± 0.00665</td><td>±0.00826</td><td><i>tie</i></td></tr>
<tr><td><code>errors.terminal_log_price_ks</code></td><td>0.0414307 ± 0.00992</td><td>0.0247803 ± 0.00912</td><td>0.0120117 ± 0.00319</td><td>-0.009521, +0.01501, +0.03601, +0.01196, +0.02979</td><td>0.0166504 ± 0.0177</td><td>±0.022</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.augmented_r2</code></td><td>0.453933 ± 0.0163</td><td><b>0.548288 ± 0.0087</b></td><td>0.673625 ± 0.00375</td><td>-0.09516, -0.1102, -0.08462, -0.06908, -0.1127</td><td>-0.094355 ± 0.0182</td><td>±0.0226</td><td>LS4</td></tr>
<tr><td><code>generated_memory.baseline_r2</code></td><td>0.392884 ± 0.00864</td><td>0.377943 ± 0.0139</td><td>0.380345 ± 0.00425</td><td>-0.001275, +0.01004, +0.02056, +0.01195, +0.03343</td><td>0.0149406 ± 0.0129</td><td>±0.0161</td><td><i>tie</i></td></tr>
<tr><td><code>generated_memory.early_history_incremental_r2</code></td><td>0.0610487 ± 0.0151</td><td><b>0.170344 ± 0.0154</b></td><td>0.29328 ± 0.00726</td><td>-0.09389, -0.1202, -0.1052, -0.08103, -0.1462</td><td>-0.109296 ± 0.0252</td><td>±0.0312</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_future_rv_correlation</code></td><td>0.57039 ± 0.0256</td><td><b>0.695678 ± 0.0117</b></td><td>0.809204 ± 0.00274</td><td>-0.1213, -0.1443, -0.1154, -0.082, -0.1635</td><td>-0.125287 ± 0.0308</td><td>±0.0383</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_rate</code></td><td>0.468359 ± 0.0182</td><td><b>0.636279 ± 0.0087</b></td><td>0.626538 ± 0.00224</td><td>-0.1439, -0.1561, -0.1823, -0.1577, -0.1996</td><td>-0.16792 ± 0.0225</td><td>±0.0279</td><td>LS4</td></tr>
<tr><td><code>generated_memory.early_hit_standardized_coefficient</code></td><td>0.614215 ± 0.0757</td><td><b>1.08214 ± 0.048</b></td><td>1.51798 ± 0.0125</td><td>-0.4317, -0.5235, -0.4417, -0.3326, -0.6101</td><td>-0.46792 ± 0.104</td><td>±0.13</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_hit_gap</code></td><td>0.140868 ± 0.00635</td><td><b>0.173494 ± 0.00241</b></td><td>0.223988 ± 0.00125</td><td>-0.0337, -0.0359, -0.02921, -0.02199, -0.04233</td><td>-0.0326255 ± 0.0076</td><td>±0.00943</td><td>LS4</td></tr>
<tr><td><code>generated_memory.future_rv_hit_mean</code></td><td>0.361101 ± 0.00224</td><td>0.415608 ± 0.00299</td><td>0.427455 ± 0.000421</td><td>-0.05304, -0.05382, -0.05456, -0.05097, -0.06015</td><td>-0.0545071 ± 0.00343</td><td>±0.00425</td><td><i>diagnostic (§5 group mean)</i></td></tr>
<tr><td><code>generated_memory.future_rv_no_hit_mean</code></td><td>0.220232 ± 0.00509</td><td>0.242114 ± 0.000655</td><td>0.203467 ± 0.000875</td><td>-0.01934, -0.01792, -0.02535, -0.02898, -0.01782</td><td>-0.0218817 ± 0.00503</td><td>±0.00624</td><td><i>diagnostic (§5 group mean)</i></td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>1266 ± 19.8</td><td>1596.4 ± 33.1</td><td>1600.8 ± 11.5</td><td>-328, -306, -327, -313, -378</td><td>-330.4 ± 28.2</td><td>±35</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>0.804888 ± 0.0052</td><td>0.949641 ± 0.00608</td><td>0.937306 ± 0.000826</td><td>-0.1332, -0.1398, -0.1548, -0.1431, -0.1528</td><td>-0.144753 ± 0.00903</td><td>±0.0112</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>0.840909 ± 0.0075</td><td>0.988202 ± 0.00971</td><td>0.989367 ± 0.00106</td><td>-0.1245, -0.145, -0.1605, -0.1478, -0.1587</td><td>-0.147293 ± 0.0144</td><td>±0.0179</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>target_memory.augmented_r2</code></td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>0.67399 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.baseline_r2</code></td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>0.385467 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_history_incremental_r2</code></td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>0.288523 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_future_rv_correlation</code></td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>0.80836 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_rate</code></td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>0.629639 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.early_hit_standardized_coefficient</code></td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>1.4987 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_gap</code></td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>0.223872 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_hit_mean</code></td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>0.426855 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>target_memory.future_rv_no_hit_mean</code></td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>0.202984 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
</table>
<!-- END GENERATED -->

Full detail — including the primary-metric panel, the ×floor ratios, the novelty analysis,
the A1–A34 battery and the curve-shape suite — is in
[`experiment_A/README.md`](experiment_A/README.md).

---

## 2. Experiment B — balanced Heston parameter mixture

Of the 51 rows: **11 constants**, **17 raw diagnostics**, and **23 contested fidelity metrics**.
Of those 23 — **CSDI wins 10, LS4 wins 8, 5 are ties.** Experiment B is a genuine split, not a
ranking, and the sign is the opposite of Experiment A's.

The ties are where the interest is: **two of the four PRIMARY metrics** — regime-proportion TVD
and the leverage-curve RMSE — have a paired CI containing zero. Half the panel the protocol
asks about is undecided at 5 seeds, whatever the means suggest.

<!-- BEGIN GENERATED: experiment B table pdf -->
<table>
<tr><th rowspan="2">Metric</th><th>Diffusion</th><th>VAE</th><th rowspan="2">Perfect</th><th rowspan="2">Seedwise differences</th><th rowspan="2">Mean diff</th><th rowspan="2">Paired 95% CI</th><th rowspan="2">Winner</th></tr>
<tr><th>CSDI</th><th>LS4</th></tr>
<tr><td><code>mixture_fidelity.generated_low_confidence_fraction</code></td><td>0.4104 ± 0.0102</td><td>0.37334 ± 0.0141</td><td>0.174683 ± 0.00361</td><td>+0.03967, +0.01172, +0.03857, +0.02832, +0.06702</td><td>0.0370605 ± 0.0202</td><td>±0.025</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_mean_max_probability</code></td><td>0.641929 ± 0.00439</td><td>0.666752 ± 0.00569</td><td>0.792255 ± 0.00126</td><td>-0.02461, -0.01574, -0.02304, -0.02235, -0.03837</td><td>-0.0248226 ± 0.0083</td><td>±0.0103</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_mean_posterior_entropy</code></td><td>0.911906 ± 0.0109</td><td>0.856733 ± 0.0131</td><td>0.600259 ± 0.00188</td><td>+0.05053, +0.0264, +0.05518, +0.05973, +0.08403</td><td>0.055173 ± 0.0206</td><td>±0.0256</td><td><i>diagnostic (§3.3 posterior confidence)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.0</code></td><td>0.296851 ± 0.01</td><td>0.189526 ± 0.0394</td><td>0.123315 ± 0.00119</td><td>+0.1195, +0.1326, +0.1276, +0.02698, +0.13</td><td>0.107324 ± 0.0452</td><td>±0.0561</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.1</code></td><td>0.279663 ± 0.0114</td><td>0.196851 ± 0.0411</td><td>0.125098 ± 0.000986</td><td>+0.1036, +0.1283, +0.09753, +0.01013, +0.07446</td><td>0.0828125 ± 0.0449</td><td>±0.0558</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.2</code></td><td>0.0312012 ± 0.00275</td><td>0.00078125 ± 0.000557</td><td>0.118506 ± 0.00137</td><td>+0.02869, +0.03345, +0.03015, +0.02734, +0.03247</td><td>0.0304199 ± 0.00255</td><td>±0.00316</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.3</code></td><td>0.0351074 ± 0.00172</td><td>0.00168457 ± 0.000698</td><td>0.118774 ± 0.000707</td><td>+0.03394, +0.03235, +0.03162, +0.03601, +0.0332</td><td>0.0334229 ± 0.00169</td><td>±0.0021</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.4</code></td><td>0.0773926 ± 0.00431</td><td>0.195288 ± 0.0291</td><td>0.135107 ± 0.00329</td><td>-0.1414, -0.1434, -0.1254, -0.0647, -0.1146</td><td>-0.117896 ± 0.032</td><td>±0.0397</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.5</code></td><td>0.073999 ± 0.00335</td><td>0.194507 ± 0.0136</td><td>0.13186 ± 0.00173</td><td>-0.1163, -0.1326, -0.1272, -0.09351, -0.1329</td><td>-0.120508 ± 0.0165</td><td>±0.0205</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.6</code></td><td>0.0964844 ± 0.0103</td><td>0.110791 ± 0.0212</td><td>0.123413 ± 0.00377</td><td>-0.03101, -0.04236, -0.02087, +0.03101, -0.008301</td><td>-0.0143066 ± 0.0283</td><td>±0.0351</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.generated_regime_proportions.7</code></td><td>0.109302 ± 0.0086</td><td>0.110571 ± 0.0176</td><td>0.123926 ± 0.00261</td><td>+0.00293, -0.008301, -0.01343, +0.02673, -0.01428</td><td>-0.00126953 ± 0.0171</td><td>±0.0212</td><td><i>diagnostic (§5 raw proportion)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.mean_error</code></td><td>0.0226021 ± 0.0195</td><td>0.0180758 ± 0.0118</td><td>0.00217876 ± 0.0024</td><td>+0.01707, -0.01054, +0.01646, -0.0342, +0.03385</td><td>0.00452632 ± 0.0269</td><td>±0.0333</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q05_error</code></td><td><b>0.0099726 ± 0.00234</b></td><td>0.0980166 ± 0.0245</td><td>0.0007326 ± 0.000494</td><td>-0.0792, -0.07722, -0.07326, -0.131, -0.07956</td><td>-0.088044 ± 0.0241</td><td>±0.03</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.q95_error</code></td><td><b>0.0144606 ± 0.00559</b></td><td>0.0939972 ± 0.00958</td><td>0.000924 ± 0.000753</td><td>-0.07626, -0.08484, -0.0792, -0.09768, -0.0597</td><td>-0.0795366 ± 0.0138</td><td>±0.0171</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.std_ratio</code></td><td>0.85191 ± 0.00828</td><td>0.752922 ± 0.0256</td><td>0.998133 ± 0.00167</td><td>+0.08511, +0.1026, +0.1005, +0.1426, +0.06418</td><td>0.0989881 ± 0.0288</td><td>±0.0358</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.support_normalized_wasserstein</code></td><td><b>0.0751098 ± 0.00394</b></td><td>0.112548 ± 0.011</td><td>0.00293792 ± 0.00066</td><td>-0.03101, -0.03933, -0.0393, -0.05602, -0.02153</td><td>-0.0374381 ± 0.0127</td><td>±0.0158</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.rho.wasserstein</code></td><td><b>0.148717 ± 0.0078</b></td><td>0.222845 ± 0.0218</td><td>0.00581707 ± 0.00131</td><td>-0.06141, -0.07787, -0.07782, -0.1109, -0.04263</td><td>-0.0741274 ± 0.0252</td><td>±0.0313</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.mean_error</code></td><td>0.0128947 ± 0.000916</td><td><b>0.00890553 ± 0.00366</b></td><td>0.000118867 ± 8.97e-05</td><td>+0.002126, +0.001534, +0.004594, +0.009121, +0.002571</td><td>0.00398917 ± 0.00309</td><td>±0.00384</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q05_error</code></td><td><b>3.46945e-19 ± 7.76e-19</b></td><td>0.000138667 ± 3.48e-05</td><td>0 ± 0</td><td>-0.00016, -0.0001333, -0.00016, -8e-05, -0.00016</td><td>-0.000138667 ± 3.48e-05</td><td>±4.32e-05</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.q95_error</code></td><td>0.000837333 ± 2.39e-05</td><td><b>6.4e-05 ± 5.84e-05</b></td><td>2.77556e-18 ± 6.21e-18</td><td>+0.0008, +0.0008267, +0.0008, +0.00064, +0.0008</td><td>0.000773333 ± 7.54e-05</td><td>±9.36e-05</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.std_ratio</code></td><td>0.8367 ± 0.00879</td><td>0.919967 ± 0.017</td><td>0.999393 ± 0.00205</td><td>-0.0747, -0.06207, -0.1046, -0.08247, -0.09249</td><td>-0.0832668 ± 0.0163</td><td>±0.0203</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.support_normalized_wasserstein</code></td><td>0.161236 ± 0.0114</td><td><b>0.114659 ± 0.0387</b></td><td>0.00237222 ± 0.000584</td><td>+0.02663, +0.0192, +0.05747, +0.09738, +0.03221</td><td>0.046577 ± 0.0318</td><td>±0.0395</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.theta.wasserstein</code></td><td>0.0128989 ± 0.000915</td><td><b>0.00917275 ± 0.0031</b></td><td>0.000189777 ± 4.68e-05</td><td>+0.002131, +0.001536, +0.004597, +0.00779, +0.002577</td><td>0.00372616 ± 0.00255</td><td>±0.00316</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.mean_error</code></td><td><b>0.105508 ± 0.00579</b></td><td>0.179389 ± 0.0252</td><td>0.00209811 ± 0.00139</td><td>-0.06628, -0.05164, -0.06251, -0.1194, -0.06958</td><td>-0.0738813 ± 0.0263</td><td>±0.0327</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q05_error</code></td><td>0.0056275 ± 0.00102</td><td><b>0.00235 ± 0.000945</b></td><td>0.0001225 ± 0.000125</td><td>+0.003887, +0.0025, +0.00425, +0.0005, +0.00525</td><td>0.0032775 ± 0.00184</td><td>±0.00228</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.q95_error</code></td><td>0.0556775 ± 0.00377</td><td>0.0602775 ± 0.0388</td><td>0.00025 ± 0.000177</td><td>+0.0055, +0.01489, +0.01775, -0.07689, +0.01575</td><td>-0.0046 ± 0.0407</td><td>±0.0505</td><td><i>tie</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.std_ratio</code></td><td>0.677446 ± 0.00693</td><td>0.724715 ± 0.0599</td><td>1.00183 ± 0.000666</td><td>-0.0693, -0.07311, -0.08112, +0.06645, -0.07927</td><td>-0.0472692 ± 0.0637</td><td>±0.0791</td><td><i>diagnostic (§5 std-deviation ratio)</i></td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.support_normalized_wasserstein</code></td><td><b>0.179548 ± 0.00443</b></td><td>0.239191 ± 0.0337</td><td>0.00360015 ± 0.00129</td><td>-0.04541, -0.03369, -0.04727, -0.121, -0.05082</td><td>-0.059643 ± 0.0349</td><td>±0.0433</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.parameters.xi.wasserstein</code></td><td><b>0.134661 ± 0.00332</b></td><td>0.179393 ± 0.0252</td><td>0.00270012 ± 0.000969</td><td>-0.03406, -0.02527, -0.03545, -0.09077, -0.03811</td><td>-0.0447323 ± 0.0262</td><td>±0.0325</td><td>CSDI</td></tr>
<tr><td><code>mixture_fidelity.regime_proportion_tvd</code></td><td>0.33042 ± 0.0143</td><td><b>0.26687 ± 0.039</b></td><td>0.00847168 ± 0.00129</td><td>+0.08105, +0.1003, +0.08838, -0.01758, +0.06555</td><td>0.0635498 ± 0.0471</td><td>±0.0584</td><td>LS4</td></tr>
<tr><td><code>mixture_fidelity.target_low_confidence_fraction</code></td><td>0.171899 ± 5.46e-05</td><td>0.171875 ± 8.63e-05</td><td>0.171875 ± 0</td><td>+0, +0, -0.0001221, +0, +0.0002441</td><td>2.44141e-05 ± 0.000134</td><td>±0.000166</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_mean_max_probability</code></td><td>0.792163 ± 0</td><td>0.792163 ± 0</td><td>0.792163 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_mean_posterior_entropy</code></td><td>0.600794 ± 0</td><td>0.600794 ± 0</td><td>0.600794 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.0</code></td><td>0.123169 ± 0</td><td>0.123169 ± 0</td><td>0.123169 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.1</code></td><td>0.122925 ± 0</td><td>0.122925 ± 0</td><td>0.122925 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.2</code></td><td>0.119629 ± 0</td><td>0.119629 ± 0</td><td>0.119629 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.3</code></td><td>0.120361 ± 0</td><td>0.120361 ± 0</td><td>0.120361 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.4</code></td><td>0.133057 ± 0</td><td>0.133057 ± 0</td><td>0.133057 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.5</code></td><td>0.130249 ± 0</td><td>0.130249 ± 0</td><td>0.130249 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.6</code></td><td>0.124268 ± 0</td><td>0.124268 ± 0</td><td>0.124268 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>mixture_fidelity.target_regime_proportions.7</code></td><td>0.126343 ± 0</td><td>0.126343 ± 0</td><td>0.126343 ± 0</td><td>+0, +0, +0, +0, +0</td><td>0 ± 0</td><td>±0</td><td>—</td></tr>
<tr><td><code>novelty.distinct_nearest_training_paths</code></td><td>2323.6 ± 41.2</td><td>3055.2 ± 248</td><td>2941.8 ± 24.5</td><td>-857, -900, -832, -229, -840</td><td>-731.6 ± 282</td><td>±350</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.mean_standardized_nearest_train_path_rmse</code></td><td>0.568307 ± 0.0122</td><td>0.913301 ± 0.0866</td><td>0.808363 ± 0.00181</td><td>-0.3879, -0.4028, -0.381, -0.1749, -0.3785</td><td>-0.344994 ± 0.0956</td><td>±0.119</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>novelty.median_standardized_nearest_train_path_rmse</code></td><td>0.470814 ± 0.018</td><td>0.926082 ± 0.116</td><td>0.716746 ± 0.00338</td><td>-0.4927, -0.5415, -0.5072, -0.2303, -0.5047</td><td>-0.455268 ± 0.127</td><td>±0.158</td><td><i>diagnostic (§4 novelty)</i></td></tr>
<tr><td><code>observable_fidelity.abs_return_acf_rmse_lags_1_50</code></td><td><b>0.011025 ± 0.00362</b></td><td>0.104407 ± 0.00656</td><td>0.0034538 ± 0.00121</td><td>-0.09792, -0.09759, -0.08375, -0.09939, -0.08826</td><td>-0.0933824 ± 0.00695</td><td>±0.00863</td><td>CSDI</td></tr>
<tr><td><code>observable_fidelity.excess_kurtosis_error</code></td><td>2.55432 ± 0.737</td><td>2.71879 ± 0.223</td><td>0.143903 ± 0.185</td><td>+0.1462, +1.048, -0.2785, -1.303, -0.435</td><td>-0.164467 ± 0.858</td><td>±1.07</td><td><i>tie</i></td></tr>
<tr><td><code>observable_fidelity.leverage_curve_rmse_lags_0_20</code></td><td>0.0058035 ± 0.00238</td><td>0.00851555 ± 0.00264</td><td>0.00389969 ± 0.000993</td><td>-0.002668, -0.008303, +0.001146, -0.004635, +0.0008995</td><td>-0.00271205 ± 0.00397</td><td>±0.00492</td><td><i>tie</i></td></tr>
<tr><td><code>observable_fidelity.realized_volatility_wasserstein</code></td><td>0.0586454 ± 0.00303</td><td><b>0.0368266 ± 0.00187</b></td><td>0.00154572 ± 0.000216</td><td>+0.01961, +0.02329, +0.0269, +0.01912, +0.02017</td><td>0.0218187 ± 0.00328</td><td>±0.00407</td><td>LS4</td></tr>
<tr><td><code>observable_fidelity.return_std_error</code></td><td>0.00403557 ± 0.000194</td><td><b>0.00149136 ± 0.000247</b></td><td>6.30604e-05 ± 3.92e-05</td><td>+0.002508, +0.002832, +0.002988, +0.001878, +0.002515</td><td>0.00254421 ± 0.000426</td><td>±0.000529</td><td>LS4</td></tr>
<tr><td><code>observable_fidelity.squared_return_acf_rmse_lags_1_50</code></td><td><b>0.0169132 ± 0.00344</b></td><td>0.0369782 ± 0.00749</td><td>0.00729821 ± 0.00266</td><td>-0.01147, -0.03117, -0.01768, -0.02129, -0.01871</td><td>-0.020065 ± 0.00718</td><td>±0.00891</td><td>CSDI</td></tr>
<tr><td><code>observable_fidelity.terminal_log_price_ks</code></td><td>0.0449219 ± 0.0105</td><td>0.0519287 ± 0.0171</td><td>0.0102539 ± 0.00242</td><td>-0.01465, -0.01831, +0.01819, -0.03662, +0.01636</td><td>-0.00700684 ± 0.0237</td><td>±0.0294</td><td><i>tie</i></td></tr>
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
| 2 | Generated bank size and all model seeds | 8192 × 128 per seed; seeds **0, 1, 2, 3, 4** for both methods in both experiments. The seeds are *aligned*, which is the only reason the `Seedwise differences` and `Paired 95% CI` columns above are the §5 quantity rather than a substitute for it |
| 3 | Official code, and its revision | Both **yes**. CSDI `methods/CSDI` @ `4189d37`; LS4 `methods/LS4/code` @ `27df71e` with experiment wrapper `train_ls4_experiment.py` @ `ba7c748`. Recorded per bank in `generation_manifest.json → model.official_implementation / model.source_revision` |
| 4 | Hyperparameters: defaults or validation-selected | **Defaults.** `hyperparameter_origin = "official-default"` in all 20 manifests. CSDI released `base.yaml`, LS4 released `solar_weekly` preset; Experiment B reuses Experiment A's settings unchanged. `disc.npy` was used for early stopping only, never for hyperparameter search |
| 5 | Trainable parameters, training time, generation time, hardware | CSDI 412,945 params — A: 2387 s train / 15.6 s generate, B: 3044 s / 13.1 s. LS4 2,146,857 params — A: 1643 s / 9.1 s, B: 1424 s / 9.4 s. All per seed. Hardware: 1× A100-SXM4-80GB per run, 8 cores pinned, 2× AMD EPYC 7763 host, at most two runs concurrent |
| 6 | Number and reason for failed runs | **Zero, across all 20 runs.** `failure_information.failed_or_unstable = false`, `first_nan_epoch = null`, `nan_in_bank = false` in every manifest. No seed was replaced, retried or dropped (§1.5) |

One deviation the PDF does not ask about but which is disclosed anyway: **Experiment B's scoring
oracle is a local refit** (`a36d64eb…` vs the PDF's `54c3f2c9…` in §6). It passes its accuracy
gate at 0.9094 ≥ 0.90 and it scored both methods identically, so every paired CI in Section 2
is instrument-consistent — but absolute distances from the floor carry the caveat. Detail in
[`experiment_B/README.md`](experiment_B/README.md) §1.

Items 3, 4 and 6 are re-derived from the twenty `generation_manifest.json` files, not typed:

```bash
python tools/check_pdf5_disclosure.py --experiment A
python tools/check_pdf5_disclosure.py --experiment B
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
│   ├── CSDI/README.md
│   ├── LS4/README.md
│   └── perfect_floor/
└── experiment_B/
    ├── README.md                 <- B: PDF + A1-A34 + curve shape + plots + dataset
    ├── CSDI/README.md
    ├── LS4/README.md
    └── perfect_floor/
```
