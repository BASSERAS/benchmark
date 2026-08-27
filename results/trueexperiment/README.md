# TrueDataset, d = 8 -- method comparison

Real Binance spot, 8 assets (BTC ETH BNB SOL XRP DOGE ADA LINK / USDT), 30-second
bars, `T = 128` (64 minutes), variant `om_2022-07_N6144`, holdout-era split with a
45.5-day embargo. Scored against `true_S_test_6144x128x8.npy`.

Three tables, nothing else. Per-seed numbers, provenance and diagnostics live in
each method's own README: [`SBTS/README.md`](SBTS/README.md), [`CSDI/README.md`](CSDI/README.md), [`Deep-MKV-TS/README.md`](Deep-MKV-TS/README.md), [`reference/README.md`](reference/README.md).

**Reading the floor.** `real_floor/` holds the *real* train, val and valdisc splits
scored against the *real* test split. On a "lower is better" metric its value is not
zero and not a target -- it is what two honest samples of this market cost each
other. The `xfloor` columns are multiples of it: **1x = indistinguishable from
another real slice on that statistic**, 5x = five times further apart than real data
gets from itself.

**Sub-floor values are flagged, not rewarded.** A value below the floor is printed in
bold with a `!`. The floor's splits are drawn from 2022-07 -> 2024-12 and the test
split from 2025-02 -> 2026-07, so the floor bundles sampling noise *with* a regime
change -- and both methods train only on the earlier era, so they inherit that same
gap. Scoring under the floor therefore means landing closer to a market the model
never saw than the data it was trained on does. That has two possible causes and
this table cannot tell them apart: (i) memorisation, guideline 7.1, where a bank
sits nearer the target than honest data can -- the worked example scores `vol_err`
6.06 % against a real-vs-real *minimum* of 8.72 % while hugging the training set
five times too closely; or (ii) a generator bias that happens to point the same way
as the era shift, e.g. a smoother thinning tails towards a calmer test era. The
NNratio separates them, and it is measured per method, not guessed here:

> NNratio (healthy band 0.932-1.000, `losses/memorisation.json`): SBTS **1.0815**; CSDI **0.5612**; Deep-MKV-TS **1.0230**; reference **1.5094**.
> Neither method is a memoriser. SBTS sits *above* the band -- further from train
> than real held-out data is. CSDI sits below it, and `CSDI/code/check_memorisation_scale.py`
> shows that rescaling its log-returns to the real training std moves the ratio to
> 1.0802, past the band; shrinking the real val split to CSDI's own volatility
> reproduces 0.475-0.516 from scale alone. That is shrunken support, not copying.

So read the `!` rows as cause (ii) throughout: sharper on tails and marginals than
the era gap, weaker where the gap is not the binding constraint.

"Closer to floor" ranks by |log ratio| -- distance from the floor in *either*
direction. It is deliberately not "lowest wins".

Seeds: SBTS 5 seeds, CSDI 5 seeds, Deep-MKV-TS 5 seeds, reference 5 seeds; floor 3. Sub-floor rows -- SBTS 15/34; CSDI 5/34; Deep-MKV-TS 15/34; reference 3/34.

---

## Table A -- distributional, temporal, adversarial (A1-A32)

| Metric | Real-vs-real floor | SBTS (mean +/- sd) | SBTS xfloor | CSDI (mean +/- sd) | CSDI xfloor | Deep-MKV-TS (mean +/- sd) | Deep-MKV-TS xfloor | reference (mean +/- sd) | reference xfloor | Closer to floor |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fat tail** | | | | | | | | | | |
| A1 Kurtosis error v | 259.6 | 150.5 +/- 62.69 | **0.58x** ! | 30.17 +/- 4.516 | **0.12x** ! | 37.8 +/- 0.441 | **0.15x** ! | 29.75 +/- 0.04771 | **0.11x** ! | SBTS |
| A2 \|r\| q95 error v | 1.85e-04 | 1.50e-04 +/- 1.66e-06 | **0.81x** ! | 5.29e-04 +/- 2.38e-05 | 2.86x | 1.68e-04 +/- 3.91e-05 | **0.91x** ! | 7.02e-04 +/- 1.23e-05 | 3.79x | Deep-MKV-TS |
| A3 \|r\| q99 error v | 4.53e-04 | 3.19e-04 +/- 4.38e-06 | **0.70x** ! | 8.65e-04 +/- 4.43e-05 | 1.91x | 4.02e-04 +/- 4.91e-05 | **0.89x** ! | 0.001639 +/- 2.98e-05 | 3.62x | Deep-MKV-TS |
| A4 Tail QQ error v | 1.94e-04 | 1.62e-04 +/- 1.23e-06 | **0.83x** ! | 5.30e-04 +/- 2.40e-05 | 2.73x | 1.79e-04 +/- 3.37e-05 | **0.92x** ! | 7.17e-04 +/- 1.22e-05 | 3.70x | Deep-MKV-TS |
| A5 Hill tail index error v | 39.99 | 52.63 +/- 11.05 | 1.32x | 44.52 +/- 11.7 | 1.11x | 39.87 +/- 6.07 | **1.00x** ! | 104.5 +/- 2.729 | 2.61x | Deep-MKV-TS |
| **Distribution** | | | | | | | | | | |
| A6 Path MMD2 v | 0.00108 | 0.003007 +/- 3.94e-04 | 2.78x | 0.002545 +/- 0.001358 | 2.36x | 0.00121 +/- 2.52e-04 | 1.12x | 0.007803 +/- 5.30e-04 | 7.22x | Deep-MKV-TS |
| A7 Terminal MMD2 v | 0.00199 | 0.00822 +/- 0.001185 | 4.13x | 0.003394 +/- 0.001483 | 1.71x | 0.00291 +/- 5.03e-04 | 1.46x | 0.01418 +/- 0.001105 | 7.12x | Deep-MKV-TS |
| A8 Increment MMD2 v | 1.71e-05 | 1.31e-05 +/- 8.10e-07 | **0.76x** ! | 1.55e-05 +/- 1.91e-06 | **0.91x** ! | 1.14e-05 +/- 1.39e-06 | **0.67x** ! | 5.06e-05 +/- 1.93e-06 | 2.95x | CSDI |
| A9 Volatility MMD v | 0.003092 | 0.04406 +/- 0.003151 | 14.25x | 0.01272 +/- 0.001864 | 4.11x | 0.01125 +/- 0.002016 | 3.64x | 0.04086 +/- 0.002798 | 13.22x | Deep-MKV-TS |
| A10 Terminal SWD v | 0.09816 | 0.1501 +/- 0.01114 | 1.53x | 0.1086 +/- 0.02187 | 1.11x | 0.1146 +/- 0.01366 | 1.17x | 0.2732 +/- 0.02104 | 2.78x | CSDI |
| A11 Path SWD v | 0.06425 | 0.09562 +/- 0.003978 | 1.49x | 0.07482 +/- 0.01516 | 1.16x | 0.06791 +/- 0.00414 | 1.06x | 0.1358 +/- 0.005041 | 2.11x | Deep-MKV-TS |
| A12 RV law loss v | 0.00733 | 0.006799 +/- 2.72e-04 | **0.93x** ! | 0.009663 +/- 3.56e-04 | 1.32x | 0.005646 +/- 7.58e-04 | **0.77x** ! | 0.01989 +/- 3.56e-04 | 2.71x | SBTS |
| A13 Mean path RMSE v | 0.01055 | 0.0128 +/- 0.001665 | 1.21x | 0.03573 +/- 0.02958 | 3.39x | 0.01069 +/- 0.004756 | 1.01x | 0.01213 +/- 0.004245 | 1.15x | Deep-MKV-TS |
| A14 KS log-returns v | 0.06536 | 0.07803 +/- 1.41e-04 | 1.19x | 0.112 +/- 0.005099 | 1.71x | 0.08593 +/- 0.003758 | 1.31x | 0.124 +/- 4.70e-04 | 1.90x | SBTS |
| A15 Skewness error v | 1.365 | 1.056 +/- 0.2374 | **0.77x** ! | 0.4047 +/- 0.05196 | **0.30x** ! | 0.2393 +/- 0.005003 | **0.18x** ! | 0.244 +/- 0.008608 | **0.18x** ! | SBTS |
| A16 QQ RMSE (300-pt) v | 1.08e-04 | 9.58e-05 +/- 9.30e-07 | **0.88x** ! | 2.43e-04 +/- 1.04e-05 | 2.24x | 1.20e-04 +/- 1.36e-05 | 1.11x | 3.36e-04 +/- 4.62e-06 | 3.10x | Deep-MKV-TS |
| A17 Terminal price KS v | 0.032 | 0.07703 +/- 0.001093 | 2.41x | 0.07554 +/- 0.02358 | 2.36x | 0.04337 +/- 0.002411 | 1.36x | 0.05557 +/- 0.002249 | 1.74x | Deep-MKV-TS |
| **Adversarial** | | | | | | | | | | |
| A18 Disc score GRU v | 0.01044 | 0.007323 +/- 0.00233 | **0.70x** ! | 0.006021 +/- 0.002776 | **0.58x** ! | 0.005614 +/- 0.002972 | **0.54x** ! | 0.05004 +/- 0.08692 | 4.79x | SBTS |
| A18 Disc score MLP v | 0.003797 | 0.004069 +/- 0.002831 | 1.07x | 0.00773 +/- 0.005403 | 2.04x | 0.007893 +/- 0.004483 | 2.08x | 0.008462 +/- 0.003578 | 2.23x | SBTS |
| **Predictive** | | | | | | | | | | |
| A19 Pred score GRU v | 0.003637 | 0.00357 +/- 1.42e-05 | **0.98x** ! | 0.003577 +/- 2.19e-05 | **0.98x** ! | 0.003551 +/- 1.06e-05 | **0.98x** ! | 0.003565 +/- 1.01e-05 | **0.98x** ! | CSDI |
| A19 Pred score MLP v | 0.003776 | 0.003876 +/- 5.01e-05 | 1.03x | 0.003799 +/- 7.15e-05 | 1.01x | 0.004045 +/- 1.26e-04 | 1.07x | 0.003918 +/- 8.36e-05 | 1.04x | CSDI |
| **Temporal** | | | | | | | | | | |
| A20 Covariance error v | 1.157 | 0.9965 +/- 0.07298 | **0.86x** ! | 1.881 +/- 0.1018 | 1.63x | 1.312 +/- 0.2892 | 1.13x | 2.69 +/- 0.1495 | 2.33x | Deep-MKV-TS |
| A21 ACF \|r\| error v | 0.009538 | 0.05778 +/- 5.46e-04 | 6.06x | 0.02648 +/- 0.003388 | 2.78x | 0.03224 +/- 0.00162 | 3.38x | 0.05079 +/- 0.001052 | 5.33x | CSDI |
| A22 ACF r2 error v | 0.006465 | 0.0432 +/- 3.94e-04 | 6.68x | 0.02253 +/- 0.002698 | 3.49x | 0.03071 +/- 0.00132 | 4.75x | 0.04415 +/- 7.90e-04 | 6.83x | CSDI |
| A23 ACF \|r\| lag-1 error v | 0.0123 | 0.1379 +/- 9.13e-04 | 11.21x | 0.05086 +/- 0.004664 | 4.14x | 0.03784 +/- 0.002085 | 3.08x | 0.05698 +/- 9.33e-04 | 4.63x | Deep-MKV-TS |
| A24 ACF r2 lag-1 error v | 0.008313 | 0.1035 +/- 7.34e-04 | 12.45x | 0.04405 +/- 0.003862 | 5.30x | 0.03254 +/- 0.001892 | 3.91x | 0.0548 +/- 7.90e-04 | 6.59x | Deep-MKV-TS |
| **Volatility** | | | | | | | | | | |
| A25 Mean RMSE v | 0.04728 | 0.06186 +/- 0.01264 | 1.31x | 0.1607 +/- 0.1209 | 3.40x | 0.05082 +/- 0.02574 | 1.07x | 0.0582 +/- 0.02048 | 1.23x | Deep-MKV-TS |
| A26 Return std error v | 0.01284 | 0.01094 +/- 4.26e-04 | **0.85x** ! | 0.02647 +/- 0.001236 | 2.06x | 0.008288 +/- 0.001563 | **0.65x** ! | 0.03731 +/- 5.90e-04 | 2.91x | SBTS |
| A27 Log-return std error v | 1.28e-04 | 1.09e-04 +/- 3.82e-06 | **0.85x** ! | 2.66e-04 +/- 1.26e-05 | 2.07x | 8.27e-05 +/- 1.55e-05 | **0.64x** ! | 3.72e-04 +/- 5.66e-06 | 2.90x | SBTS |
| A28 Kurtosis ratio (-> 1) | 0.4156 | 0.5529 +/- 0.2495 | 0.4471 | 2.141 +/- 0.3799 | 1.141 | 13.25 +/- 0.6865 | 12.25 | 3.323 +/- 0.0158 | 2.323 | SBTS |
| A29 Sigma mean error v | 0.00146 | 0.001683 +/- 1.74e-05 | 1.15x | 0.003909 +/- 1.71e-04 | 2.68x | 0.001393 +/- 2.13e-04 | **0.95x** ! | 0.005054 +/- 6.95e-05 | 3.46x | Deep-MKV-TS |
| A30 Cross-sect. vol path RMSE v | 0.09316 | 0.07302 +/- 0.007161 | **0.78x** ! | 0.1129 +/- 0.0105 | 1.21x | 0.06736 +/- 0.01593 | **0.72x** ! | 0.256 +/- 0.009015 | 2.75x | CSDI |
| A31 Rolling vol KS (w=5) v | 0.1196 | 0.1312 +/- 1.13e-04 | 1.10x | 0.3522 +/- 0.01587 | 2.94x | 0.1693 +/- 0.005388 | 1.42x | 0.2736 +/- 8.63e-04 | 2.29x | SBTS |
| A32 Vol-of-vol error v | 1.04e-04 | 6.66e-05 +/- 2.30e-06 | **0.64x** ! | 1.41e-04 +/- 7.60e-06 | 1.36x | 7.69e-05 +/- 9.38e-06 | **0.74x** ! | 2.26e-04 +/- 3.81e-06 | 2.18x | Deep-MKV-TS |

Rows won: **SBTS 10** | **CSDI 7** | **Deep-MKV-TS 17** | **reference 0** of
34 decided.

`+/-` is the sample spread **across seeds**, not a confidence interval. A33-A34 are
absent by construction: they score against the dataset's latent variance path and a
real market has none, so `compute_all_multiasset.py` passes `v_test=None` and skips
them rather than emitting a fake row. A28 targets 1.0 rather than the floor, so its
`xfloor` column carries the raw distance `|x - 1|` instead of a ratio.

---

## Table B -- curve shape and path cloud

| Panel | Measure | Real-vs-real floor | SBTS | SBTS xfloor | CSDI | CSDI xfloor | Deep-MKV-TS | Deep-MKV-TS xfloor | reference | reference xfloor | Closer to floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Path cloud** *(50x50 2-D histogram)* | grid_tvd (%) v | 3.683% | 9.534% +/- 0.4652 | 2.59x | 8.816% +/- 1.527 | 2.39x | 6.168% +/- 0.2454 | 1.67x | 8.036% +/- 0.232 | 2.18x | Deep-MKV-TS |
| **Log-return histogram** | MSE v | 3.806e+05 | 3.493e+05 +/- 532.4 | **0.92x** ! | 3.52e+05 +/- 681.7 | **0.92x** ! | 3.584e+05 +/- 728.1 | **0.94x** ! | 3.649e+05 +/- 392.2 | **0.96x** ! | reference |
|  | % err v | 1.765e+07 | 2.494e+08 +/- 8.655e+05 | 14.12x | 3.535e+08 +/- 9.101e+06 | 20.02x | 1.739e+08 +/- 6.811e+06 | 9.85x | 1.546e+08 +/- 4.377e+05 | 8.76x | reference |
|  | NRMSE v | 10.24 | 7.012 +/- 0.009238 | **0.68x** ! | 9.558 +/- 0.2897 | **0.93x** ! | 8.085 +/- 0.1469 | **0.79x** ! | 9.217 +/- 0.01451 | **0.90x** ! | CSDI |
|  | CVaR90 v | 24.39 | 15.35 +/- 0.03567 | **0.63x** ! | 23.27 +/- 0.8245 | **0.95x** ! | 15.72 +/- 0.2721 | **0.64x** ! | 19.28 +/- 0.04214 | **0.79x** ! | CSDI |
|  | CVaR95 v | 35.63 | 24.19 +/- 0.055 | **0.68x** ! | 35.35 +/- 1.189 | **0.99x** ! | 25.06 +/- 0.4663 | **0.70x** ! | 29.56 +/- 0.05054 | **0.83x** ! | CSDI |
| **QQ plot** | MSE v | 7.92e-09 | 5.26e-09 +/- 1.67e-10 | **0.66x** ! | 2.59e-08 +/- 2.03e-09 | 3.26x | 7.91e-09 +/- 2.55e-09 | **1.00x** ! | 5.75e-08 +/- 1.36e-09 | 7.26x | Deep-MKV-TS |
|  | % err v | 326.6 | 685.2 +/- 0.9942 | 2.10x | 499.6 +/- 12.71 | 1.53x | 1033 +/- 37.2 | 3.16x | 1247 +/- 3.872 | 3.82x | CSDI |
|  | NRMSE v | 5.085 | 4.245 +/- 0.05426 | **0.83x** ! | 10.59 +/- 0.4751 | 2.08x | 5.598 +/- 0.6252 | 1.10x | 17.95 +/- 0.223 | 3.53x | Deep-MKV-TS |
|  | CVaR90 v | 5.59 | 4.283 +/- 0.05987 | **0.77x** ! | 12.38 +/- 0.5865 | 2.21x | 5.467 +/- 0.7199 | **0.98x** ! | 21.33 +/- 0.2999 | 3.82x | Deep-MKV-TS |
|  | CVaR95 v | 7.278 | 5.445 +/- 0.1026 | **0.75x** ! | 15.11 +/- 0.7476 | 2.08x | 6.8 +/- 0.8852 | **0.93x** ! | 28.33 +/- 0.4515 | 3.89x | Deep-MKV-TS |
| **ACF \|r\|** | MSE v | 5.48e-05 | 7.65e-04 +/- 8.25e-06 | 13.97x | 3.30e-04 +/- 3.33e-05 | 6.02x | 5.81e-04 +/- 2.74e-05 | 10.61x | 8.69e-04 +/- 3.44e-05 | 15.87x | CSDI |
|  | % err v | 237.1 | 390 +/- 3.387 | 1.64x | 351.5 +/- 48.28 | 1.48x | 865.2 +/- 20.45 | 3.65x | 1081 +/- 26.6 | 4.56x | CSDI |
|  | NRMSE v | 11.29 | 51.5 +/- 0.5236 | 4.56x | 26.18 +/- 3.362 | 2.32x | 44.03 +/- 1.532 | 3.90x | 60.94 +/- 1.325 | 5.40x | CSDI |
|  | CVaR90 v | 17.29 | 121.5 +/- 1.185 | 7.03x | 47.43 +/- 4.881 | 2.74x | 58.04 +/- 2.007 | 3.36x | 81.93 +/- 1.533 | 4.74x | CSDI |
|  | CVaR95 v | 18.72 | 163.8 +/- 1.107 | 8.75x | 59.81 +/- 5.896 | 3.19x | 60.48 +/- 1.906 | 3.23x | 84.53 +/- 1.49 | 4.52x | CSDI |
| **ACF r²** | MSE v | 3.14e-05 | 4.25e-04 +/- 5.26e-06 | 13.52x | 2.03e-04 +/- 2.02e-05 | 6.46x | 4.12e-04 +/- 1.86e-05 | 13.11x | 5.33e-04 +/- 2.16e-05 | 16.95x | CSDI |
|  | % err v | 321.1 | 3021 +/- 155.1 | 9.41x | 528.1 +/- 265.6 | 1.64x | 2910 +/- 225.3 | 9.07x | 2641 +/- 171.9 | 8.23x | CSDI |
|  | NRMSE v | 9.25 | 46.16 +/- 0.4136 | 4.99x | 26.23 +/- 3.076 | 2.84x | 47.87 +/- 1.161 | 5.17x | 58.01 +/- 1.069 | 6.27x | CSDI |
|  | CVaR90 v | 14.23 | 107.9 +/- 0.9459 | 7.59x | 48.29 +/- 4.243 | 3.39x | 63.26 +/- 2.096 | 4.45x | 82.8 +/- 1.318 | 5.82x | CSDI |
|  | CVaR95 v | 15.52 | 145.9 +/- 1.064 | 9.40x | 62.01 +/- 5.062 | 4.00x | 65.28 +/- 2.383 | 4.21x | 85.07 +/- 1.299 | 5.48x | CSDI |
| **Rolling vol histogram** | MSE v | 3.897e+04 | 3.308e+04 +/- 136.7 | **0.85x** ! | 1.667e+05 +/- 1.877e+04 | 4.28x | 5.956e+04 +/- 1498 | 1.53x | 1.018e+05 +/- 407.8 | 2.61x | SBTS |
|  | % err v | 42.82 | 32.57 +/- 0.1618 | **0.76x** ! | 92.42 +/- 3.781 | 2.16x | 47.15 +/- 5.947 | 1.10x | 155.9 +/- 1.607 | 3.64x | Deep-MKV-TS |
|  | NRMSE v | 10 | 11.31 +/- 0.01637 | 1.13x | 31.34 +/- 1.67 | 3.13x | 15.09 +/- 0.2084 | 1.51x | 19.36 +/- 0.05128 | 1.94x | SBTS |
|  | CVaR90 v | 25.83 | 29.09 +/- 0.05076 | 1.13x | 81.99 +/- 4.108 | 3.17x | 38.62 +/- 0.7345 | 1.49x | 49.95 +/- 0.1224 | 1.93x | SBTS |
|  | CVaR95 v | 31.15 | 35.9 +/- 0.0435 | 1.15x | 114.2 +/- 6.769 | 3.67x | 45.31 +/- 0.8382 | 1.45x | 57.38 +/- 0.1217 | 1.84x | SBTS |
| **Tail survival** | MSE v | 0.002323 | 0.003772 +/- 2.72e-06 | 1.62x | 0.00984 +/- 6.89e-04 | 4.24x | 0.005306 +/- 3.64e-04 | 2.28x | 0.01113 +/- 6.10e-05 | 4.79x | SBTS |
|  | % err v | 17.91 | 17.53 +/- 0.06977 | **0.98x** ! | 40.31 +/- 1.658 | 2.25x | 24.96 +/- 2.093 | 1.39x | 58.06 +/- 0.4239 | 3.24x | SBTS |
|  | NRMSE v | 8.573 | 10.79 +/- 0.005985 | 1.26x | 18.84 +/- 0.7501 | 2.20x | 13.85 +/- 0.6401 | 1.62x | 20.04 +/- 0.06696 | 2.34x | SBTS |
|  | CVaR90 v | 14.21 | 18.72 +/- 0.01049 | 1.32x | 25.97 +/- 1.021 | 1.83x | 20.95 +/- 0.8593 | 1.47x | 29.15 +/- 0.08693 | 2.05x | SBTS |
|  | CVaR95 v | 14.69 | 19.19 +/- 0.01073 | 1.31x | 26.21 +/- 1.032 | 1.78x | 21.02 +/- 0.8527 | 1.43x | 29.24 +/- 0.08628 | 1.99x | SBTS |

MSE rows won: **SBTS 2** | **CSDI 2** | **Deep-MKV-TS 2** | **reference 1**.

One panel contributes five measures of the same curve; only the MSE row is counted
so a single panel cannot vote five times. `grid_tvd` is the total-variation
distance between 50x50 2-D histograms of the price clouds, averaged over assets --
the one row here that looks at joint path geometry rather than a marginal.

---

## Table C -- conditional generation, forecast CRPS by path shadowing

Section 8. Condition on bars `0..64`, forecast `65..96` (`H = 32`), retrieve
`K = 256` neighbours jointly across the 8 assets from an 8 192-path bank, score the
CRPS of the implied predictive law against the real continuation. `paper`
convention (`--weight-mode paper --standardize bank`), CRPS x 1000, lower is better.

| Bank | Seeds | Cumulative return v | Increment v | Realized vol v | xfloor (cum. return) |
|---|---|---|---|---|---|
| **Real train split as bank** *(floor)* | 1 | 1.257 +/- 0.026 | 0.335 +/- 0.005 | 0.970 +/- 0.027 | 1.00x |
| **SBTS** | 5 | **1.269 +/- 0.002** | **0.337 +/- 0.000** | **1.043 +/- 0.007** | 1.01x |
| **CSDI** | 5 | **1.248 +/- 0.002** | **0.342 +/- 0.001** | **1.230 +/- 0.024** | **0.99x** ! |
| **Deep-MKV-TS** | 5 | **1.248 +/- 0.002** | **0.333 +/- 0.000** | **0.887 +/- 0.020** | **0.99x** ! |
| **reference** | 5 | **1.258 +/- 0.001** | **0.338 +/- 0.000** | **0.983 +/- 0.004** | 1.00x |
| Block bootstrap *(baseline)* | 1 | 1.279 +/- 0.025 | 0.338 +/- 0.005 | 1.110 +/- 0.024 | 1.02x |
| Session bootstrap *(baseline)* | 1 | 1.262 +/- 0.026 | 0.337 +/- 0.005 | 0.975 +/- 0.027 | 1.00x |

The `xfloor` column carries the same sub-floor convention as tables A and B, and it
fires here: a bank that forecasts the test era *better* than a bank of real training
paths is flagged, not credited. Cause (ii) above is the live explanation -- a
narrower predictive law is rewarded by CRPS when the target era is calmer than the
training era, which on this split it is (annualised vol falls between the two eras
on 6 of the 8 assets). The three targets are scored on the same queries, so compare
across a row before reading down a column.

Each method's CRPS is a mean over 5 generator seeds, matching tables A
and B, so the method `+/-` are sd over the same *n* and are comparable to each
other. The floor row and the two bootstrap baselines are deterministic
given `--baseline-seed 1234`, hence one "seed" each.

**Two different `+/-` appear in this table and they are not the same quantity.** On
the floor and baseline rows it is the half-width of the 95 % bootstrap CI over the
6 144 test queries -- sampling noise. On the method rows it is the sample sd across
seeds -- generator noise. They are never merged.

---

*Generated by [`render_comparison.py`](render_comparison.py). Every number is read
from `real_floor/`, `SBTS/`, `CSDI/`, `Deep-MKV-TS/` and `reference/` at render time; none is typed by hand. Do
not edit this file -- re-run the script.*
