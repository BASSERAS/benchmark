# Multi-Asset Heston (d = 8) — Method Comparison

Mean ± std across 5 seeds, SBTS vs LS4 vs CSDI vs Deep-MKV-TS vs Deep-MKV-TS-SBTSref vs reference. Per-seed columns, figures,
hyperparameters and implementation notes are on each method's own page: [SBTS](SBTS/README.md) · [LS4](LS4/README.md) · [CSDI](CSDI/README.md) · [Deep-MKV-TS](Deep-MKV-TS/README.md) · [Deep-MKV-TS-SBTSref](Deep-MKV-TS-SBTSref/README.md) · [reference](reference/README.md).
Dataset construction, per-asset vs native metric scoping and the memorisation
diagnostic are in [`oldreadme.md`](oldreadme.md).

**Perfect floor** is the *independent-draw* floor: five fresh draws from the same SDE
with the same frozen per-asset parameters at seeds 1000–1004, scored with byte-identical
metric code. It is **non-zero everywhere** — two independent 8 192-path draws never
produce identical histograms, ACFs, quantiles or covariance matrices. A method scoring
*below* the floor is a red flag, not a win.

**`reference` is not a trained generator.** It is σ<sup>ref</sup>, the frozen parametric SDE that Deep-MKV-TS Algorithm 1 starts from, sampled with the neural control switched off and verified to be exactly zero — 19 fitted coefficients, **zero gradient steps**. It is adjudicated on equal terms above rather than shown as a passive baseline, because a row where a 19-coefficient SDE beats a trained network is a real result about that network, not a footnote. Read its column two ways: against **Deep-MKV-TS** it isolates what the learned control bought — and any row where Deep-MKV-TS scores *worse* is a row where Algorithm 1 damaged a working reference; against the other methods it is the bar a neural generator has to clear to justify itself. Generation seeds are shared with Deep-MKV-TS, so those two columns are **paired** — same Brownian streams, difference free of simulation noise. Details: [`reference/README.md`](reference/README.md).

## A1–A34, mean ± std across 5 seeds

> ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every
> other row is computed on each of the 8 univariate slices and reported as the **mean over
> assets**. All metrics on **log-returns** unless noted; A26 uses price increments.

| Metric | SBTS | LS4 | CSDI | Deep-MKV-TS | Deep-MKV-TS-SBTSref | reference | Perfect floor | Winner |
|---|---|---|---|---|---|---|---|---|
| **Fat Tail** | | | | | | | | |
| A1 Kurtosis Error ↓ | 0.01811 ± 0.003116 | 0.7708 ± 0.005457 | 0.1464 ± 0.03503 | 1.039 ± 0.08859 | 0.2268 ± 0.006905 | 0.6296 ± 0.002013 | 0.008385 | **SBTS** |
| A2 \|r\| q95 Error ↓ | 1.40e-04 ± 1.01e-05 | 0.007707 ± 7.45e-04 | 0.005902 ± 2.80e-04 | 5.15e-04 ± 1.75e-04 | 5.05e-04 ± 9.60e-05 | 0.00355 ± 1.61e-05 | 4.58e-05 | **SBTS** |
| A3 \|r\| q99 Error ↓ | 2.08e-04 ± 2.20e-05 | 0.0124 ± 0.001045 | 0.008171 ± 3.66e-04 | 0.001332 ± 2.97e-04 | 0.001067 ± 5.02e-05 | 0.004809 ± 3.38e-05 | 8.08e-05 | **SBTS** |
| A4 Tail QQ Error ↓ | 1.35e-04 ± 7.31e-06 | 0.007604 ± 7.35e-04 | 0.005814 ± 2.75e-04 | 5.51e-04 ± 1.32e-04 | 6.92e-04 ± 7.03e-05 | 0.003534 ± 1.72e-05 | 5.62e-05 | **SBTS** |
| A5 Hill Tail Index Error ↓ | 0.7294 ± 0.1953 | 2.793 ± 0.4285 | 1.466 ± 0.3679 | 2.136 ± 0.2054 | 5.135 ± 0.1114 | 3.274 ± 0.1463 | 0.5896 | **SBTS** |
| **Distribution** | | | | | | | | |
| A6 Path MMD² ↓ *(native d=8)* | 0.002046 ± 9.17e-05 | 0.01225 ± 0.00266 | 0.003622 ± 3.20e-04 | 0.002128 ± 9.06e-05 | 0.00299 ± 3.54e-04 | 0.002745 ± 9.90e-05 | 0.001948 | — |
| A7 Terminal MMD² ↓ *(native d=8)* | 0.002055 ± 4.96e-05 | 0.006468 ± 0.001546 | 0.002716 ± 1.58e-04 | 0.002087 ± 6.46e-05 | 0.002685 ± 2.39e-04 | 0.002586 ± 6.34e-05 | 0.001954 | — |
| A8 Increment MMD² ↓ *(native d=8)* | 0.00101 ± 1.22e-05 | 0.01435 ± 0.004248 | 0.009969 ± 1.98e-04 | 8.87e-04 ± 1.62e-05 | 0.001676 ± 1.56e-04 | 9.00e-04 ± 4.65e-06 | 8.71e-04 | — |
| A9 Volatility MMD ↓ *(native d=8)* | 0.009935 ± 3.62e-04 | 0.6814 ± 0.1357 | 0.4098 ± 0.01038 | 0.033 ± 0.001978 | 0.06617 ± 0.006584 | 0.2756 ± 0.002493 | 0.008587 | **SBTS** |
| A10 Terminal SWD ↓ *(native d=8)* | 1.355 ± 0.3244 | 4.281 ± 0.6338 | 2.836 ± 0.4006 | 1.664 ± 0.1645 | 2.252 ± 0.309 | 2.217 ± 0.1102 | 1.141 | — |
| A11 Path SWD ↓ *(native d=8)* | 0.8259 ± 0.09446 | 2.933 ± 0.2927 | 1.699 ± 0.1648 | 1.012 ± 0.06409 | 1.501 ± 0.1677 | 1.434 ± 0.02652 | 0.7258 | **SBTS** |
| A12 RV Law Loss ↓ | 0.145 ± 0.007379 | 4.867 ± 0.4648 | 4.138 ± 0.1787 | 0.7273 ± 0.02372 | 1.151 ± 0.05499 | 3.063 ± 0.01584 | 0.06398 | **SBTS** |
| A13 Mean Path RMSE ↓ | 0.2766 ± 0.03025 | 0.6314 ± 0.261 | 1.391 ± 0.6026 | 0.4456 ± 0.02541 | 0.4067 ± 0.0314 | 0.4311 ± 0.1046 | 0.1834 | **SBTS** |
| A14 KS Log-returns ↓ | 0.002583 ± 2.36e-04 | 0.06227 ± 0.008789 | 0.06252 ± 0.002812 | 0.01869 ± 9.95e-04 | 0.02742 ± 8.01e-04 | 0.04336 ± 2.75e-04 | 9.64e-04 | **SBTS** |
| A15 Skewness Error ↓ | 0.007407 ± 6.69e-04 | 0.0209 ± 0.001063 | 0.06925 ± 0.005297 | 0.05419 ± 9.79e-04 | 0.04499 ± 0.001099 | 0.05418 ± 0.001213 | 0.003568 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 8.09e-05 ± 5.38e-06 | 0.00333 ± 3.52e-04 | 0.002819 ± 1.44e-04 | 5.11e-04 ± 1.61e-05 | 8.72e-04 ± 4.07e-05 | 0.001943 ± 7.27e-06 | 3.04e-05 | **SBTS** |
| A17 Terminal Price KS ↓ | 0.0217 ± 0.002147 | 0.09221 ± 0.01855 | 0.06658 ± 0.01179 | 0.03581 ± 0.003285 | 0.03958 ± 0.001928 | 0.04949 ± 0.002247 | 0.01466 | **SBTS** |
| **Adversarial** | | | | | | | | |
| A18 Disc Score GRU ↓ *(native d=8)* | 0.01999 ± 0.004875 | 0.3987 ± 0.1967 | 0.2844 ± 0.2272 | 0.1157 ± 0.1464 | 0.4734 ± 0.01189 | 0.4808 ± 0.01072 | 0.005523 | — |
| A18 Disc Score MLP ↓ *(native d=8)* | 0.00534 ± 0.004294 | 0.005279 ± 0.002527 | 0.006195 ± 0.004601 | 0.007598 ± 0.005273 | 0.006622 ± 0.003915 | 0.005279 ± 0.005004 | 0.006012 | — *(floor)* |
| **Predictive** | | | | | | | | |
| A19 Pred Score GRU ↓ | 0.0492 ± 2.72e-06 | 0.0492 ± 3.69e-06 | 0.04953 ± 1.96e-05 | 0.0492 ± 3.77e-06 | 0.0492 ± 5.28e-06 | 0.0492 ± 4.09e-06 | 0.0492 | — *(floor)* |
| A19 Pred Score MLP ↓ | 0.04944 ± 1.71e-04 | 0.04941 ± 5.12e-05 | 0.04968 ± 4.91e-05 | 0.04947 ± 1.04e-04 | 0.04945 ± 1.73e-04 | 0.0496 ± 8.11e-05 | 0.04931 | — |
| **Temporal** | | | | | | | | |
| A20 Covariance Error ↓ *(native d=8)* | 80.27 ± 12.98 | 1084 ± 90.03 | 527.4 ± 45.47 | 256.2 ± 32.76 | 412.5 ± 53.48 | 754.6 ± 6.179 | 55.2 | **SBTS** |
| A21 ACF \|r\| Error (lags) ↓ | 0.004719 ± 3.10e-04 | 0.06824 ± 0.00132 | 0.01092 ± 0.001795 | 0.04793 ± 7.88e-04 | 0.06168 ± 0.001082 | 0.04602 ± 2.82e-04 | 0.001066 | **SBTS** |
| A22 ACF r² Error (lags) ↓ | 0.003448 ± 3.00e-04 | 0.05623 ± 0.001435 | 0.00997 ± 0.001346 | 0.03615 ± 7.48e-04 | 0.04995 ± 0.001083 | 0.0333 ± 2.93e-04 | 0.001107 | **SBTS** |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.005199 ± 3.32e-04 | 0.07919 ± 0.001432 | 0.02199 ± 0.002742 | 0.05567 ± 8.25e-04 | 0.07262 ± 0.001157 | 0.05307 ± 3.44e-04 | 0.001038 | **SBTS** |
| A24 ACF r² Lag-1 Error ↓ | 0.003714 ± 3.33e-04 | 0.06584 ± 0.001548 | 0.02007 ± 0.00194 | 0.04232 ± 6.90e-04 | 0.05938 ± 0.001185 | 0.03839 ± 4.24e-04 | 0.001036 | **SBTS** |
| **Vol** | | | | | | | | |
| A25 Mean RMSE ↓ *(native d=8)* | 1.649 ± 0.2965 | 4.066 ± 2.772 | 6.091 ± 2.464 | 2.409 ± 0.1371 | 2.043 ± 0.1728 | 2.087 ± 0.4813 | 0.9234 | — |
| A26 Return Std Error ↓ | 0.006541 ± 4.14e-04 | 0.3366 ± 0.03949 | 0.288 ± 0.003593 | 0.01675 ± 8.19e-04 | 0.111 ± 0.005592 | 0.1868 ± 3.21e-04 | 0.001745 | **SBTS** |
| A27 Log-Return Std Error ↓ | 6.34e-05 ± 5.34e-06 | 0.003519 ± 3.71e-04 | 0.002921 ± 1.46e-04 | 1.78e-04 ± 4.39e-05 | 6.26e-04 ± 5.51e-05 | 0.001888 ± 9.11e-06 | 2.21e-05 | **SBTS** |
| A28 Kurtosis Ratio (→ 1) | 1.02 ± 0.004402 | 5.437 ± 0.5817 | 0.8747 ± 0.02461 | 1.451 ± 0.07378 | 1.779 ± 0.04949 | 1.302 ± 0.01066 | 1 | **SBTS** |
| A29 Sigma Mean Error ↓ | 0.001005 ± 9.10e-05 | 0.05358 ± 0.00578 | 0.04547 ± 0.002292 | 0.002899 ± 8.07e-04 | 0.01003 ± 8.44e-04 | 0.03008 ± 1.25e-04 | 3.32e-04 | **SBTS** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.3011 ± 0.02041 | 3.859 ± 0.3462 | 1.745 ± 0.1586 | 0.2275 ± 0.02226 | 0.9101 ± 0.0753 | 2.079 ± 0.03411 | 0.1596 | **Deep-MKV-TS** |
| A31 Rolling Vol KS (w=5) ↓ | 0.007704 ± 3.42e-04 | 0.2578 ± 0.03221 | 0.2451 ± 0.01367 | 0.06854 ± 0.00314 | 0.106 ± 0.003065 | 0.1648 ± 4.29e-04 | 0.00208 | **SBTS** |
| A32 Vol-of-Vol Error ↓ | 4.80e-05 ± 3.56e-06 | 0.001949 ± 1.39e-04 | 0.001181 ± 3.97e-05 | 4.01e-04 ± 4.78e-05 | 3.90e-04 ± 2.27e-05 | 6.34e-04 ± 4.43e-06 | 1.14e-05 | **SBTS** |
| **Heston Spec** | | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -4.17e-04 ± 6.08e-04 | 1.46e-04 ± 4.00e-04 | 0.002651 ± 0.00109 | -0.001293 ± 0.001289 | -0.00259 ± 0.002108 | -4.96e-04 ± 0.001518 | -1.35e-04 | — *(floor)* |
| A34 Teacher-Sigma RMSE ↓ | 0.1006 ± 8.39e-05 | 0.09441 ± 0.002176 | 0.1011 ± 7.47e-04 | 0.09602 ± 5.64e-04 | 0.09568 ± 4.47e-04 | 0.1018 ± 1.67e-04 | 0.1013 | — *(floor)* |

**How to read the tally.** Every cell above is an average twice over: a mean ± std across
the 5 seeds, of a quantity that — on every row *not* marked *(native d=8)* — is itself
already the mean over the 8 assets. So the std carries both seed-to-seed and asset-to-asset
variation, and a row is only decided when the gap between two methods survives it. Across
the **36 A rows**: **SBTS** wins 24, **LS4** wins 0, **CSDI** wins 0, **Deep-MKV-TS** wins 1, **Deep-MKV-TS-SBTSref** wins 0, **reference** wins 0, 7 are ties inside one cross-seed std and 4 sit at the independent-draw floor, where the row separates nothing. A row is *not* awarded to anyone when
either method has reached the independent-draw floor — past the floor the metric stops
measuring fidelity — or when the gap is no wider than the summed stds.

## B, curve-shape metrics, mean ± std across 5 seeds

> All ↓ lower is better. Each stylised-fact plot yields a curve, not a scalar; the curve is
> computed per asset and averaged over the 8 assets before the combination. The **MSE row
> decides the winner** for a plot — the % err / NRMSE / CVaR rows are function-only
> diagnostics, reported because a method can win on MSE while failing in the tail.

| Plot | Measure | SBTS | LS4 | CSDI | Deep-MKV-TS | Deep-MKV-TS-SBTSref | reference | Perfect floor | Winner |
|---|---|---|---|---|---|---|---|---|---|
| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | 3.317% ± 0.05786% | 16.06% ± 1.739% | 7.214% ± 0.3744% | 4.404% ± 0.2595% | 7.899% ± 0.1405% | 7.881% ± 0.1491% | 2.189% | **SBTS** |
| **Log-return histogram** | MSE | 0.06192 ± 0.002754 | 5.82 ± 1.703 | 7.216 ± 1.094 | 1.059 ± 0.0749 | 1.866 ± 0.09227 | 3.94 ± 0.02815 | 0.0554 | **SBTS** |
|  | % err | 1.672% ± 0.05289% | 46.17% ± 4.066% | 38.88% ± 1.862% | 8.323% ± 0.4029% | 12.12% ± 0.5918% | 29.47% ± 0.158% | 1.302% | **SBTS** |
|  | NRMSE | 0.533% ± 0.01813% | 10.31% ± 1.678% | 11.38% ± 0.8472% | 3.844% ± 0.142% | 5.592% ± 0.1306% | 7.793% ± 0.02095% | 0.3601% | **SBTS** |
|  | CVaR₉₀ | 1.267% ± 0.04072% | 20.68% ± 4.051% | 27.84% ± 2.331% | 9.4% ± 0.4271% | 13.68% ± 0.3047% | 17.78% ± 0.06125% | 0.8339% | **SBTS** |
|  | CVaR₉₅ | 1.547% ± 0.06047% | 21.36% ± 4.22% | 30.29% ± 2.67% | 10.78% ± 0.4642% | 15.26% ± 0.3334% | 19.27% ± 0.06476% | 0.994% | **SBTS** |
| **QQ plot** | MSE | 2.94e-09 ± 3.50e-10 | 4.45e-06 ± 9.40e-07 | 2.87e-06 ± 2.87e-07 | 1.09e-07 ± 5.50e-09 | 2.69e-07 ± 2.32e-08 | 1.60e-06 ± 1.11e-08 | 5.66e-10 | **SBTS** |
|  | % err | 1.753% ± 0.1874% | 22.72% ± 3.296% | 28.35% ± 3.839% | 10.76% ± 0.4023% | 14.77% ± 0.4263% | 22.41% ± 0.2656% | 0.5364% | **SBTS** |
|  | NRMSE | 0.2277% ± 0.01445% | 9.535% ± 0.9334% | 7.951% ± 0.4024% | 1.508% ± 0.03763% | 2.402% ± 0.1166% | 5.564% ± 0.02401% | 0.08644% | **SBTS** |
|  | CVaR₉₀ | 0.2406% ± 0.01323% | 11.61% ± 1.03% | 8.626% ± 0.3878% | 1.377% ± 0.118% | 1.784% ± 0.09059% | 5.414% ± 0.02613% | 0.09922% | **SBTS** |
|  | CVaR₉₅ | 0.2901% ± 0.01661% | 14.08% ± 1.178% | 9.909% ± 0.4259% | 1.718% ± 0.231% | 1.869% ± 0.09074% | 6.101% ± 0.03164% | 0.125% | **SBTS** |
| **ACF \|r\|** | MSE | 1.16e-05 ± 8.48e-07 | 0.001134 ± 4.05e-05 | 3.33e-05 ± 7.89e-06 | 6.12e-04 ± 1.74e-05 | 9.00e-04 ± 3.56e-05 | 5.67e-04 ± 5.62e-06 | 3.33e-06 | **SBTS** |
|  | % err | 6.188% ± 0.4192% | 89.21% ± 2.269% | 10.12% ± 1.319% | 57.85% ± 1.375% | 76.12% ± 2.154% | 56.01% ± 0.4432% | 2.043% | **SBTS** |
|  | NRMSE | 7.721% ± 0.5052% | 105.8% ± 2.464% | 15.57% ± 2.257% | 72.15% ± 1.728% | 93.4% ± 2.191% | 68.29% ± 0.524% | 2.523% | **SBTS** |
|  | CVaR₉₀ | 12.91% ± 0.6494% | 146.6% ± 3.237% | 34.58% ± 5.017% | 101.3% ± 2.076% | 134.3% ± 2.437% | 94.37% ± 0.6908% | 4.991% | **SBTS** |
|  | CVaR₉₅ | 13.69% ± 0.6415% | 149.4% ± 3.209% | 44.61% ± 5.326% | 103.1% ± 2.052% | 137.3% ± 2.446% | 96.17% ± 0.9253% | 5.542% | **SBTS** |
| **ACF r²** | MSE | 1.06e-05 ± 6.70e-07 | 7.42e-04 ± 3.50e-05 | 2.66e-05 ± 4.61e-06 | 3.38e-04 ± 1.28e-05 | 5.63e-04 ± 2.84e-05 | 2.96e-04 ± 3.78e-06 | 3.98e-06 | **SBTS** |
|  | % err | 5.183% ± 0.3729% | 85.85% ± 2.655% | 10.58% ± 1.065% | 52.01% ± 1.528% | 71.26% ± 2.467% | 47.82% ± 0.4534% | 2.37% | **SBTS** |
|  | NRMSE | 6.296% ± 0.4904% | 96.22% ± 2.769% | 15.5% ± 1.927% | 61.44% ± 1.727% | 83.32% ± 2.366% | 55.1% ± 0.5336% | 2.772% | **SBTS** |
|  | CVaR₉₀ | 11.55% ± 0.838% | 135.6% ± 3.791% | 35.13% ± 4.357% | 88.03% ± 1.957% | 122.5% ± 2.695% | 76.84% ± 0.7162% | 5.52% | **SBTS** |
|  | CVaR₉₅ | 12.61% ± 0.8087% | 138.4% ± 3.713% | 44.96% ± 4.326% | 90.04% ± 1.8% | 125.4% ± 2.657% | 78.58% ± 1.029% | 6.131% | **SBTS** |
| **Rolling vol histogram** | MSE | 0.9039 ± 0.04565 | 258.9 ± 61.08 | 226.6 ± 27.6 | 36.22 ± 0.6324 | 57.54 ± 2.041 | 116.5 ± 0.4119 | 0.6039 | **SBTS** |
|  | % err | 2.839% ± 0.083% | 69.89% ± 5.593% | 68.13% ± 3.046% | 19.72% ± 1.157% | 21.32% ± 0.4801% | 48.79% ± 0.2709% | 1.536% | **SBTS** |
|  | NRMSE | 1.117% ± 0.03384% | 28.72% ± 3.751% | 27.19% ± 1.68% | 9.885% ± 0.1103% | 13.28% ± 0.252% | 18.44% ± 0.03629% | 0.5137% | **SBTS** |
|  | CVaR₉₀ | 2.472% ± 0.06549% | 59.69% ± 8.219% | 59.14% ± 4.176% | 19.69% ± 0.3317% | 26.88% ± 0.5212% | 36.9% ± 0.06561% | 1.172% | **SBTS** |
|  | CVaR₉₅ | 2.732% ± 0.07558% | 62.56% ± 8.626% | 62.21% ± 4.552% | 20.87% ± 0.4474% | 28.38% ± 0.5038% | 38.48% ± 0.09122% | 1.362% | **SBTS** |
| **Tail survival** | MSE | 2.07e-06 ± 2.69e-07 | 0.002558 ± 7.29e-04 | 0.002536 ± 3.16e-04 | 1.91e-04 ± 2.61e-05 | 4.78e-04 ± 3.18e-05 | 0.001417 ± 9.41e-06 | 2.03e-07 | **SBTS** |
|  | % err | 0.7097% ± 0.04721% | 29.77% ± 3.145% | 27.35% ± 1.382% | 5.196% ± 0.2167% | 9.121% ± 0.4664% | 20.07% ± 0.09337% | 0.2297% | **SBTS** |
|  | NRMSE | 0.2349% ± 0.01479% | 8.509% ± 1.216% | 8.749% ± 0.547% | 2.217% ± 0.1638% | 3.747% ± 0.1305% | 5.956% ± 0.01871% | 0.06634% | **SBTS** |
|  | CVaR₉₀ | 0.3569% ± 0.02001% | 12.32% ± 1.665% | 12.07% ± 0.7518% | 3.267% ± 0.2222% | 5.343% ± 0.17% | 8.417% ± 0.02636% | 0.1122% | **SBTS** |
|  | CVaR₉₅ | 0.3623% ± 0.02021% | 12.37% ± 1.669% | 12.11% ± 0.7545% | 3.28% ± 0.2226% | 5.36% ± 0.1697% | 8.446% ± 0.02685% | 0.1175% | **SBTS** |

**How to read the tally.** Each stylised-fact curve is computed per asset and averaged over
the 8 assets *before* the four summary measures are taken, then averaged again across the
5 seeds — so, as in the A table, every cell is an average of averages. The headline count
is the MSE row, which is the one that decides a plot: across the **6 plots**,
**SBTS** wins 6, **LS4** wins 0, **CSDI** wins 0, **Deep-MKV-TS** wins 0, **Deep-MKV-TS-SBTSref** wins 0 and **reference** wins 0. Counting every row in the table instead — all **31** of them,
including the % err / NRMSE / CVaR diagnostics and the path-cloud TVD —
**SBTS** wins 31, **LS4** wins 0, **CSDI** wins 0, **Deep-MKV-TS** wins 0, **Deep-MKV-TS-SBTSref** wins 0 and **reference** wins 0. The two counts are given separately, and the first is the one to
quote, because the four diagnostic rows describe the *same* curve as their MSE row — they
exist to show *where* a curve fails, not to be counted as four extra wins.
