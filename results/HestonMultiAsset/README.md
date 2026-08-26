# Multi-Asset Heston (d = 8) — Method Comparison

Mean ± std across 5 seeds, SBTS vs LS4. Per-seed columns, figures,
hyperparameters and implementation notes are on each method's own page: [SBTS](SBTS/README.md) · [LS4](LS4/README.md).
Dataset construction, per-asset vs native metric scoping and the memorisation
diagnostic are in [`oldreadme.md`](oldreadme.md).

**Perfect floor** is the *independent-draw* floor: five fresh draws from the same SDE
with the same frozen per-asset parameters at seeds 1000–1004, scored with byte-identical
metric code. It is **non-zero everywhere** — two independent 8 192-path draws never
produce identical histograms, ACFs, quantiles or covariance matrices. A method scoring
*below* the floor is a red flag, not a win.

## A1–A34, mean ± std across 5 seeds

> ↓ lower is better; ↑ higher is better;, no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every
> other row is computed on each of the 8 univariate slices and reported as the **mean over
> assets**. All metrics on **log-returns** unless noted; A26 uses price increments.

| Metric | SBTS | LS4 | Perfect floor | Winner |
|---|---|---|---|---|
| **, Fat Tail, ** | | | | |
| A1 Kurtosis Error ↓ | 0.01811 ± 0.003116 | 0.7708 ± 0.005457 | 0.008385 | **SBTS** |
| A2 \|r\| q95 Error ↓ | 1.40e-04 ± 1.01e-05 | 0.007707 ± 7.45e-04 | 4.58e-05 | **SBTS** |
| A3 \|r\| q99 Error ↓ | 2.08e-04 ± 2.20e-05 | 0.0124 ± 0.001045 | 8.08e-05 | **SBTS** |
| A4 Tail QQ Error ↓ | 1.35e-04 ± 7.31e-06 | 0.007604 ± 7.35e-04 | 5.62e-05 | **SBTS** |
| A5 Hill Tail Index Error ↓ | 0.7294 ± 0.1953 | 2.793 ± 0.4285 | 0.5896 | **SBTS** |
| **, Distribution, ** | | | | |
| A6 Path MMD² ↓ *(native d=8)* | 0.002046 ± 9.17e-05 | 0.01225 ± 0.00266 | 0.001948 | **SBTS** |
| A7 Terminal MMD² ↓ *(native d=8)* | 0.002055 ± 4.96e-05 | 0.006468 ± 0.001546 | 0.001954 | **SBTS** |
| A8 Increment MMD² ↓ *(native d=8)* | 0.00101 ± 1.22e-05 | 0.01435 ± 0.004248 | 8.71e-04 | **SBTS** |
| A9 Volatility MMD ↓ *(native d=8)* | 0.009935 ± 3.62e-04 | 0.6814 ± 0.1357 | 0.008587 | **SBTS** |
| A10 Terminal SWD ↓ *(native d=8)* | 1.355 ± 0.3244 | 4.281 ± 0.6338 | 1.141 | **SBTS** |
| A11 Path SWD ↓ *(native d=8)* | 0.8259 ± 0.09446 | 2.933 ± 0.2927 | 0.7258 | **SBTS** |
| A12 RV Law Loss ↓ | 0.145 ± 0.007379 | 4.867 ± 0.4648 | 0.06398 | **SBTS** |
| A13 Mean Path RMSE ↓ | 0.2766 ± 0.03025 | 0.6314 ± 0.261 | 0.1834 | **SBTS** |
| A14 KS Log-returns ↓ | 0.002583 ± 2.36e-04 | 0.06227 ± 0.008789 | 9.64e-04 | **SBTS** |
| A15 Skewness Error ↓ | 0.007407 ± 6.69e-04 | 0.0209 ± 0.001063 | 0.003568 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 8.09e-05 ± 5.38e-06 | 0.00333 ± 3.52e-04 | 3.04e-05 | **SBTS** |
| A17 Terminal Price KS ↓ | 0.0217 ± 0.002147 | 0.09221 ± 0.01855 | 0.01466 | **SBTS** |
| **, Adversarial, ** | | | | |
| A18 Disc Score GRU ↓ *(native d=8)* | 0.02658 ± 0.00467 | 0.3987 ± 0.1967 | 0.005523 | **SBTS** |
| A18 Disc Score MLP ↓ *(native d=8)* | 0.007904 ± 0.007351 | 0.005279 ± 0.002527 | 0.006012 | — *(floor)* |
| **, Predictive, ** | | | | |
| A19 Pred Score GRU ↓ | 0.0492 ± 2.76e-06 | 0.0492 ± 3.69e-06 | 0.0492 | — *(floor)* |
| A19 Pred Score MLP ↓ | 0.04932 ± 6.13e-05 | 0.04941 ± 5.12e-05 | 0.04931 | — |
| **, Temporal, ** | | | | |
| A20 Covariance Error ↓ *(native d=8)* | 80.27 ± 12.98 | 1084 ± 90.03 | 55.2 | **SBTS** |
| A21 ACF \|r\| Error (lags) ↓ | 0.004719 ± 3.10e-04 | 0.06824 ± 0.00132 | 0.001066 | **SBTS** |
| A22 ACF r² Error (lags) ↓ | 0.003448 ± 3.00e-04 | 0.05623 ± 0.001435 | 0.001107 | **SBTS** |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.005199 ± 3.32e-04 | 0.07919 ± 0.001432 | 0.001038 | **SBTS** |
| A24 ACF r² Lag-1 Error ↓ | 0.003714 ± 3.33e-04 | 0.06584 ± 0.001548 | 0.001036 | **SBTS** |
| **, Vol, ** | | | | |
| A25 Mean RMSE ↓ *(native d=8)* | 1.649 ± 0.2965 | 4.066 ± 2.772 | 0.9234 | — |
| A26 Return Std Error ↓ | 0.006541 ± 4.14e-04 | 0.3366 ± 0.03949 | 0.001745 | **SBTS** |
| A27 Log-Return Std Error ↓ | 6.34e-05 ± 5.34e-06 | 0.003519 ± 3.71e-04 | 2.21e-05 | **SBTS** |
| A28 Kurtosis Ratio (→ 1) | 1.02 ± 0.004402 | 5.437 ± 0.5817 | 1 | **SBTS** |
| A29 Sigma Mean Error ↓ | 0.001005 ± 9.10e-05 | 0.05358 ± 0.00578 | 3.32e-04 | **SBTS** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.3011 ± 0.02041 | 3.859 ± 0.3462 | 0.1596 | **SBTS** |
| A31 Rolling Vol KS (w=5) ↓ | 0.007704 ± 3.42e-04 | 0.2578 ± 0.03221 | 0.00208 | **SBTS** |
| A32 Vol-of-Vol Error ↓ | 4.80e-05 ± 3.56e-06 | 0.001949 ± 1.39e-04 | 1.14e-05 | **SBTS** |
| **, Heston Spec, ** | | | | |
| A33 Teacher-Sigma Corr ↑ | -4.17e-04 ± 6.08e-04 | 1.46e-04 ± 4.00e-04 | -1.35e-04 | — *(floor)* |
| A34 Teacher-Sigma RMSE ↓ | 0.1006 ± 8.39e-05 | 0.09441 ± 0.002176 | 0.1013 | — *(floor)* |

**How to read the tally.** Every cell above is an average twice over: a mean ± std across
the 5 seeds, of a quantity that — on every row *not* marked *(native d=8)* — is itself
already the mean over the 8 assets. So the std carries both seed-to-seed and asset-to-asset
variation, and a row is only decided when the gap between two methods survives it. Across
the **36 A rows**: **SBTS** wins 30, **LS4** wins 0, 2 are ties inside one cross-seed std and 4 sit at the independent-draw floor, where the row separates nothing. A row is *not* awarded to anyone when
either method has reached the independent-draw floor — past the floor the metric stops
measuring fidelity — or when the gap is no wider than the summed stds.

## B, curve-shape metrics, mean ± std across 5 seeds

> All ↓ lower is better. Each stylised-fact plot yields a curve, not a scalar; the curve is
> computed per asset and averaged over the 8 assets before the combination. The **MSE row
> decides the winner** for a plot — the % err / NRMSE / CVaR rows are function-only
> diagnostics, reported because a method can win on MSE while failing in the tail.

| Plot | Measure | SBTS | LS4 | Perfect floor | Winner |
|---|---|---|---|---|---|
| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | 3.317% ± 0.05786% | 16.06% ± 1.739% | 2.189% | **SBTS** |
| **Log-return histogram** | MSE | 0.06192 ± 0.002754 | 5.82 ± 1.703 | 0.0554 | **SBTS** |
|  | % err | 1.672% ± 0.05289% | 46.17% ± 4.066% | 1.302% | **SBTS** |
|  | NRMSE | 0.533% ± 0.01813% | 10.31% ± 1.678% | 0.3601% | **SBTS** |
|  | CVaR₉₀ | 1.267% ± 0.04072% | 20.68% ± 4.051% | 0.8339% | **SBTS** |
|  | CVaR₉₅ | 1.547% ± 0.06047% | 21.36% ± 4.22% | 0.994% | **SBTS** |
| **QQ plot** | MSE | 2.94e-09 ± 3.50e-10 | 4.45e-06 ± 9.40e-07 | 5.66e-10 | **SBTS** |
|  | % err | 1.753% ± 0.1874% | 22.72% ± 3.296% | 0.5364% | **SBTS** |
|  | NRMSE | 0.2277% ± 0.01445% | 9.535% ± 0.9334% | 0.08644% | **SBTS** |
|  | CVaR₉₀ | 0.2406% ± 0.01323% | 11.61% ± 1.03% | 0.09922% | **SBTS** |
|  | CVaR₉₅ | 0.2901% ± 0.01661% | 14.08% ± 1.178% | 0.125% | **SBTS** |
| **ACF \|r\|** | MSE | 1.16e-05 ± 8.48e-07 | 0.001134 ± 4.05e-05 | 3.33e-06 | **SBTS** |
|  | % err | 6.188% ± 0.4192% | 89.21% ± 2.269% | 2.043% | **SBTS** |
|  | NRMSE | 7.721% ± 0.5052% | 105.8% ± 2.464% | 2.523% | **SBTS** |
|  | CVaR₉₀ | 12.91% ± 0.6494% | 146.6% ± 3.237% | 4.991% | **SBTS** |
|  | CVaR₉₅ | 13.69% ± 0.6415% | 149.4% ± 3.209% | 5.542% | **SBTS** |
| **ACF r²** | MSE | 1.06e-05 ± 6.70e-07 | 7.42e-04 ± 3.50e-05 | 3.98e-06 | **SBTS** |
|  | % err | 5.183% ± 0.3729% | 85.85% ± 2.655% | 2.37% | **SBTS** |
|  | NRMSE | 6.296% ± 0.4904% | 96.22% ± 2.769% | 2.772% | **SBTS** |
|  | CVaR₉₀ | 11.55% ± 0.838% | 135.6% ± 3.791% | 5.52% | **SBTS** |
|  | CVaR₉₅ | 12.61% ± 0.8087% | 138.4% ± 3.713% | 6.131% | **SBTS** |
| **Rolling vol histogram** | MSE | 0.9039 ± 0.04565 | 258.9 ± 61.08 | 0.6039 | **SBTS** |
|  | % err | 2.839% ± 0.083% | 69.89% ± 5.593% | 1.536% | **SBTS** |
|  | NRMSE | 1.117% ± 0.03384% | 28.72% ± 3.751% | 0.5137% | **SBTS** |
|  | CVaR₉₀ | 2.472% ± 0.06549% | 59.69% ± 8.219% | 1.172% | **SBTS** |
|  | CVaR₉₅ | 2.732% ± 0.07558% | 62.56% ± 8.626% | 1.362% | **SBTS** |
| **Tail survival** | MSE | 2.07e-06 ± 2.69e-07 | 0.002558 ± 7.29e-04 | 2.03e-07 | **SBTS** |
|  | % err | 0.7097% ± 0.04721% | 29.77% ± 3.145% | 0.2297% | **SBTS** |
|  | NRMSE | 0.2349% ± 0.01479% | 8.509% ± 1.216% | 0.06634% | **SBTS** |
|  | CVaR₉₀ | 0.3569% ± 0.02001% | 12.32% ± 1.665% | 0.1122% | **SBTS** |
|  | CVaR₉₅ | 0.3623% ± 0.02021% | 12.37% ± 1.669% | 0.1175% | **SBTS** |

**How to read the tally.** Each stylised-fact curve is computed per asset and averaged over
the 8 assets *before* the four summary measures are taken, then averaged again across the
5 seeds — so, as in the A table, every cell is an average of averages. The headline count
is the MSE row, which is the one that decides a plot: across the **6 plots**,
**SBTS** wins 6 and **LS4** wins 0. Counting every row in the table instead — all **31** of them,
including the % err / NRMSE / CVaR diagnostics and the path-cloud TVD —
**SBTS** wins 31 and **LS4** wins 0. The two counts are given separately, and the first is the one to
quote, because the four diagnostic rows describe the *same* curve as their MSE row — they
exist to show *where* a curve fails, not to be counted as four extra wins.
