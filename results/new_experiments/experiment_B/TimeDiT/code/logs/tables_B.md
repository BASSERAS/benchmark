===== SECTION 1 : PDF metrics, test side =====
<!-- model seeds: 5 | floor seeds: 5 -->
| Metric | TimeDiT (mean ± std) | 95% CI half-width | Perfect floor (mean ± std) |
|---|---|---|---|
| `mixture_fidelity.generated_low_confidence_fraction` | 0.371411 ± 0.0146 | 0.0181 | 0.174683 ± 0.00361 |
| `mixture_fidelity.generated_mean_max_probability` | 0.666199 ± 0.00791 | 0.00982 | 0.792255 ± 0.00126 |
| `mixture_fidelity.generated_mean_posterior_entropy` | 0.853934 ± 0.0191 | 0.0237 | 0.600259 ± 0.00188 |
| `mixture_fidelity.generated_regime_proportions.0` | 0.188867 ± 0.0341 | 0.0424 | 0.123315 ± 0.00119 |
| `mixture_fidelity.generated_regime_proportions.1` | 0.175244 ± 0.0146 | 0.0181 | 0.125098 ± 0.000986 |
| `mixture_fidelity.generated_regime_proportions.2` | 0.058667 ± 0.0333 | 0.0414 | 0.118506 ± 0.00137 |
| `mixture_fidelity.generated_regime_proportions.3` | 0.0440186 ± 0.0134 | 0.0167 | 0.118774 ± 0.000707 |
| `mixture_fidelity.generated_regime_proportions.4` | 0.170435 ± 0.0428 | 0.0531 | 0.135107 ± 0.00329 |
| `mixture_fidelity.generated_regime_proportions.5` | 0.12959 ± 0.0333 | 0.0413 | 0.13186 ± 0.00173 |
| `mixture_fidelity.generated_regime_proportions.6` | 0.138696 ± 0.0251 | 0.0312 | 0.123413 ± 0.00377 |
| `mixture_fidelity.generated_regime_proportions.7` | 0.0944824 ± 0.027 | 0.0336 | 0.123926 ± 0.00261 |
| `mixture_fidelity.parameters.rho.mean_error` | 0.0916107 ± 0.052 | 0.0645 | 0.00217876 ± 0.0024 |
| `mixture_fidelity.parameters.rho.q05_error` | 0.006996 ± 0.00482 | 0.00598 | 0.0007326 ± 0.000494 |
| `mixture_fidelity.parameters.rho.q95_error` | 0.0101046 ± 0.00557 | 0.00691 | 0.000924 ± 0.000753 |
| `mixture_fidelity.parameters.rho.std_ratio` | 0.884255 ± 0.0192 | 0.0239 | 0.998133 ± 0.00167 |
| `mixture_fidelity.parameters.rho.support_normalized_wasserstein` | 0.0697717 ± 0.0124 | 0.0153 | 0.00293792 ± 0.00066 |
| `mixture_fidelity.parameters.rho.wasserstein` | 0.138148 ± 0.0245 | 0.0304 | 0.00581707 ± 0.00131 |
| `mixture_fidelity.parameters.theta.mean_error` | 0.00433576 ± 0.00149 | 0.00185 | 0.000118867 ± 8.97e-05 |
| `mixture_fidelity.parameters.theta.q05_error` | 0 ± 0 | 0 | 0 ± 0 |
| `mixture_fidelity.parameters.theta.q95_error` | 6.4e-05 ± 3.04e-05 | 3.77e-05 | 2.77556e-18 ± 6.21e-18 |
| `mixture_fidelity.parameters.theta.std_ratio` | 0.952414 ± 0.0246 | 0.0306 | 0.999393 ± 0.00205 |
| `mixture_fidelity.parameters.theta.support_normalized_wasserstein` | 0.0597516 ± 0.009 | 0.0112 | 0.00237222 ± 0.000584 |
| `mixture_fidelity.parameters.theta.wasserstein` | 0.00478013 ± 0.00072 | 0.000894 | 0.000189777 ± 4.68e-05 |
| `mixture_fidelity.parameters.xi.mean_error` | 0.0789575 ± 0.0244 | 0.0303 | 0.00209811 ± 0.00139 |
| `mixture_fidelity.parameters.xi.q05_error` | 0.00675 ± 0.000685 | 0.00085 | 0.0001225 ± 0.000125 |
| `mixture_fidelity.parameters.xi.q95_error` | 0.02025 ± 0.00375 | 0.00465 | 0.00025 ± 0.000177 |
| `mixture_fidelity.parameters.xi.std_ratio` | 0.7727 ± 0.0198 | 0.0246 | 1.00183 ± 0.000666 |
| `mixture_fidelity.parameters.xi.support_normalized_wasserstein` | 0.13798 ± 0.0161 | 0.02 | 0.00360015 ± 0.00129 |
| `mixture_fidelity.parameters.xi.wasserstein` | 0.103485 ± 0.0121 | 0.015 | 0.00270012 ± 0.000969 |
| `mixture_fidelity.regime_proportion_tvd` | 0.194141 ± 0.0338 | 0.0419 | 0.00847168 ± 0.00129 |
| `mixture_fidelity.target_low_confidence_fraction` | 0.171875 ± 0 | 0 | 0.171875 ± 0 |
| `mixture_fidelity.target_mean_max_probability` | 0.792163 ± 0 | 0 | 0.792163 ± 0 |
| `mixture_fidelity.target_mean_posterior_entropy` | 0.600794 ± 0 | 0 | 0.600794 ± 0 |
| `mixture_fidelity.target_regime_proportions.0` | 0.123169 ± 0 | 0 | 0.123169 ± 0 |
| `mixture_fidelity.target_regime_proportions.1` | 0.122925 ± 0 | 0 | 0.122925 ± 0 |
| `mixture_fidelity.target_regime_proportions.2` | 0.119629 ± 0 | 0 | 0.119629 ± 0 |
| `mixture_fidelity.target_regime_proportions.3` | 0.120361 ± 0 | 0 | 0.120361 ± 0 |
| `mixture_fidelity.target_regime_proportions.4` | 0.133057 ± 0 | 0 | 0.133057 ± 0 |
| `mixture_fidelity.target_regime_proportions.5` | 0.130249 ± 0 | 0 | 0.130249 ± 0 |
| `mixture_fidelity.target_regime_proportions.6` | 0.124268 ± 0 | 0 | 0.124268 ± 0 |
| `mixture_fidelity.target_regime_proportions.7` | 0.126343 ± 0 | 0 | 0.126343 ± 0 |
| `novelty.distinct_nearest_training_paths` | 2795.2 ± 152 | 189 | 2941.8 ± 24.5 |
| `novelty.mean_standardized_nearest_train_path_rmse` | 0.764603 ± 0.0758 | 0.0941 | 0.808363 ± 0.00181 |
| `novelty.median_standardized_nearest_train_path_rmse` | 0.762712 ± 0.153 | 0.19 | 0.716746 ± 0.00338 |
| `observable_fidelity.abs_return_acf_rmse_lags_1_50` | 0.0320231 ± 0.0152 | 0.0189 | 0.0034538 ± 0.00121 |
| `observable_fidelity.excess_kurtosis_error` | 1.60599 ± 0.917 | 1.14 | 0.143903 ± 0.185 |
| `observable_fidelity.leverage_curve_rmse_lags_0_20` | 0.0119523 ± 0.00353 | 0.00438 | 0.00389969 ± 0.000993 |
| `observable_fidelity.realized_volatility_wasserstein` | 0.0224224 ± 0.00418 | 0.00519 | 0.00154572 ± 0.000216 |
| `observable_fidelity.return_std_error` | 0.00125334 ± 0.000816 | 0.00101 | 6.30604e-05 ± 3.92e-05 |
| `observable_fidelity.squared_return_acf_rmse_lags_1_50` | 0.0417153 ± 0.00975 | 0.0121 | 0.00729821 ± 0.00266 |
| `observable_fidelity.terminal_log_price_ks` | 0.0856934 ± 0.015 | 0.0186 | 0.0102539 ± 0.00242 |

===== SECTION 1.3 : PDF metrics, validation (disc) side =====
<!-- model seeds: 5 | floor seeds: 5 -->
| Metric | TimeDiT (mean ± std) | 95% CI half-width | Perfect floor (mean ± std) |
|---|---|---|---|
| `mixture_fidelity.generated_low_confidence_fraction` | 0.371436 ± 0.0145 | 0.0181 | 0.174683 ± 0.00361 |
| `mixture_fidelity.generated_mean_max_probability` | 0.666199 ± 0.00791 | 0.00982 | 0.792255 ± 0.00126 |
| `mixture_fidelity.generated_mean_posterior_entropy` | 0.853934 ± 0.0191 | 0.0237 | 0.600259 ± 0.00188 |
| `mixture_fidelity.generated_regime_proportions.0` | 0.188867 ± 0.0341 | 0.0424 | 0.123315 ± 0.00119 |
| `mixture_fidelity.generated_regime_proportions.1` | 0.175244 ± 0.0146 | 0.0181 | 0.125098 ± 0.000986 |
| `mixture_fidelity.generated_regime_proportions.2` | 0.0586914 ± 0.0334 | 0.0414 | 0.118506 ± 0.00137 |
| `mixture_fidelity.generated_regime_proportions.3` | 0.0440186 ± 0.0134 | 0.0167 | 0.118774 ± 0.000707 |
| `mixture_fidelity.generated_regime_proportions.4` | 0.170435 ± 0.0428 | 0.0531 | 0.135107 ± 0.00329 |
| `mixture_fidelity.generated_regime_proportions.5` | 0.12959 ± 0.0333 | 0.0413 | 0.13186 ± 0.00173 |
| `mixture_fidelity.generated_regime_proportions.6` | 0.138696 ± 0.0251 | 0.0312 | 0.123413 ± 0.00377 |
| `mixture_fidelity.generated_regime_proportions.7` | 0.094458 ± 0.027 | 0.0335 | 0.123926 ± 0.00261 |
| `mixture_fidelity.parameters.rho.mean_error` | 0.0905464 ± 0.0506 | 0.0628 | 0.00256722 ± 0.00164 |
| `mixture_fidelity.parameters.rho.q05_error` | 0.0066 ± 0.00445 | 0.00553 | 0.0003366 ± 0.00033 |
| `mixture_fidelity.parameters.rho.q95_error` | 0.0094446 ± 0.00557 | 0.00691 | 0.000528 ± 0.000552 |
| `mixture_fidelity.parameters.rho.std_ratio` | 0.885023 ± 0.0192 | 0.0239 | 0.999 ± 0.00167 |
| `mixture_fidelity.parameters.rho.support_normalized_wasserstein` | 0.0698003 ± 0.0129 | 0.016 | 0.00300821 ± 0.000536 |
| `mixture_fidelity.parameters.rho.wasserstein` | 0.138205 ± 0.0256 | 0.0318 | 0.00595626 ± 0.00106 |
| `mixture_fidelity.parameters.theta.mean_error` | 0.00429775 ± 0.00153 | 0.0019 | 8.59121e-05 ± 7.17e-05 |
| `mixture_fidelity.parameters.theta.q05_error` | 0 ± 0 | 0 | 0 ± 0 |
| `mixture_fidelity.parameters.theta.q95_error` | 6.4e-05 ± 3.04e-05 | 3.77e-05 | 0 ± 0 |
| `mixture_fidelity.parameters.theta.std_ratio` | 0.953384 ± 0.0247 | 0.0306 | 1.00041 ± 0.00205 |
| `mixture_fidelity.parameters.theta.support_normalized_wasserstein` | 0.059365 ± 0.00827 | 0.0103 | 0.00216402 ± 0.000708 |
| `mixture_fidelity.parameters.theta.wasserstein` | 0.0047492 ± 0.000662 | 0.000822 | 0.000173122 ± 5.67e-05 |
| `mixture_fidelity.parameters.xi.mean_error` | 0.0778392 ± 0.0244 | 0.0303 | 0.00179244 ± 0.00112 |
| `mixture_fidelity.parameters.xi.q05_error` | 0.0075 ± 0.000685 | 0.00085 | 0.0006275 ± 0.000125 |
| `mixture_fidelity.parameters.xi.q95_error` | 0.02 ± 0.00375 | 0.00465 | 0.0005 ± 0.000177 |
| `mixture_fidelity.parameters.xi.std_ratio` | 0.770838 ± 0.0198 | 0.0246 | 0.999419 ± 0.000665 |
| `mixture_fidelity.parameters.xi.support_normalized_wasserstein` | 0.138239 ± 0.0162 | 0.0201 | 0.0032544 ± 0.000939 |
| `mixture_fidelity.parameters.xi.wasserstein` | 0.10368 ± 0.0121 | 0.0151 | 0.0024408 ± 0.000704 |
| `mixture_fidelity.regime_proportion_tvd` | 0.191943 ± 0.0337 | 0.0419 | 0.00898437 ± 0.00249 |
| `mixture_fidelity.target_low_confidence_fraction` | 0.179028 ± 6.69e-05 | 8.3e-05 | 0.179004 ± 6.69e-05 |
| `mixture_fidelity.target_mean_max_probability` | 0.792618 ± 5.55e-17 | 6.89e-17 | 0.792618 ± 5.55e-17 |
| `mixture_fidelity.target_mean_posterior_entropy` | 0.59815 ± 0 | 0 | 0.59815 ± 0 |
| `mixture_fidelity.target_regime_proportions.0` | 0.12561 ± 0 | 0 | 0.12561 ± 0 |
| `mixture_fidelity.target_regime_proportions.1` | 0.122681 ± 0 | 0 | 0.122681 ± 0 |
| `mixture_fidelity.target_regime_proportions.2` | 0.117554 ± 0 | 0 | 0.117554 ± 0 |
| `mixture_fidelity.target_regime_proportions.3` | 0.119507 ± 0 | 0 | 0.119507 ± 0 |
| `mixture_fidelity.target_regime_proportions.4` | 0.13147 ± 0 | 0 | 0.131494 ± 5.46e-05 |
| `mixture_fidelity.target_regime_proportions.5` | 0.133911 ± 0 | 0 | 0.133887 ± 5.46e-05 |
| `mixture_fidelity.target_regime_proportions.6` | 0.12439 ± 0 | 0 | 0.12439 ± 0 |
| `mixture_fidelity.target_regime_proportions.7` | 0.124878 ± 0 | 0 | 0.124878 ± 0 |
| `novelty.distinct_nearest_training_paths` | 2795.2 ± 152 | 189 | 2941.8 ± 24.5 |
| `novelty.mean_standardized_nearest_train_path_rmse` | 0.764603 ± 0.0758 | 0.0941 | 0.808363 ± 0.00181 |
| `novelty.median_standardized_nearest_train_path_rmse` | 0.762712 ± 0.153 | 0.19 | 0.716746 ± 0.00338 |
| `observable_fidelity.abs_return_acf_rmse_lags_1_50` | 0.0308706 ± 0.0147 | 0.0182 | 0.00364878 ± 0.00139 |
| `observable_fidelity.excess_kurtosis_error` | 1.83409 ± 0.917 | 1.14 | 0.181803 ± 0.0428 |
| `observable_fidelity.leverage_curve_rmse_lags_0_20` | 0.0123 ± 0.00264 | 0.00328 | 0.00333932 ± 0.000823 |
| `observable_fidelity.realized_volatility_wasserstein` | 0.0226645 ± 0.0047 | 0.00583 | 0.00167737 ± 0.000339 |
| `observable_fidelity.return_std_error` | 0.00129214 ± 0.000844 | 0.00105 | 2.4158e-05 ± 2.84e-05 |
| `observable_fidelity.squared_return_acf_rmse_lags_1_50` | 0.0370066 ± 0.00974 | 0.0121 | 0.00663483 ± 0.00224 |
| `observable_fidelity.terminal_log_price_ks` | 0.0820312 ± 0.0164 | 0.0203 | 0.0116455 ± 0.00206 |

===== SECTION 2.1 : A1-A34 =====
| Metric | TimeDiT (mean ± std) | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|---|---|---|---|---|---|---|---|
| **Fat Tail** | | | | | | | |
| A1 Kurtosis Error ↓ | 26.3564 ± 0.978 | 26.697 | 26.9423 | 27.4804 | 25.3381 | 25.3241 | 18.4078 ± 18.1 |
| A2 \|r\| q95 Error ↓ | 0.00250602 ± 0.00156 | 0.00335284 | 0.000471406 | 0.00123336 | 0.00336712 | 0.00410538 | 7.20744e-05 ± 3.83e-05 |
| A3 \|r\| q99 Error ↓ | 0.00470394 ± 0.00258 | 0.00692546 | 0.00353305 | 0.000775183 | 0.00550983 | 0.00677617 | 0.000217235 ± 0.000133 |
| A4 Tail QQ Error ↓ | 0.00256886 ± 0.00147 | 0.00332537 | 0.000763014 | 0.00125362 | 0.00338334 | 0.00411896 | 8.65607e-05 ± 2.12e-05 |
| A5 Hill Tail Index Error ↓ | 7.16987 ± 2.17 | 9.41719 | 7.41674 | 7.99008 | 3.55788 | 7.46744 | 0.175861 ± 0.126 |
| **Distribution** | | | | | | | |
| A6 Path MMD² ↓ | 0.00535477 ± 0.00127 | 0.00728107 | 0.00386754 | 0.0048102 | 0.00506579 | 0.00574923 | 0.00176849 ± 0.000309 |
| A7 Terminal MMD² ↓ | 0.00538637 ± 0.00259 | 0.00976324 | 0.00382233 | 0.00397 | 0.00573951 | 0.00363678 | 0.00139414 ± 0.000296 |
| A8 Increment MMD² ↓ | 0.00225244 ± 0.000813 | 0.00134884 | 0.00178311 | 0.00327247 | 0.00192733 | 0.00293049 | 0.000855771 ± 7.2e-05 |
| A9 Volatility MMD ↓ | 0.039183 ± 0.0136 | 0.0427622 | 0.0320112 | 0.0613582 | 0.0271836 | 0.0325998 | 0.0088589 ± 0.001 |
| A10 Terminal SWD ↓ | 3.08849 ± 0.588 | 3.56513 | 2.19392 | 2.99097 | 3.67286 | 3.01957 | 1.05719 ± 0.324 |
| A11 Path SWD ↓ | 2.37605 ± 0.389 | 2.46596 | 1.80212 | 2.4225 | 2.30452 | 2.88514 | 0.774511 ± 0.273 |
| A12 RV Law Loss ↓ | 1.50588 ± 0.19 | 1.50679 | 1.32113 | 1.36347 | 1.53418 | 1.80384 | 0.122529 ± 0.0173 |
| A13 Mean Path RMSE ↓ | 2.08259 ± 0.35 | 1.8982 | 1.64281 | 2.0006 | 2.37937 | 2.49198 | 0.204231 ± 0.0621 |
| A14 KS Log-returns ↓ | 0.0358875 ± 0.00551 | 0.0353696 | 0.032439 | 0.0325168 | 0.033603 | 0.0455092 | 0.00160787 ± 0.000425 |
| A15 Skewness Error ↓ | 0.0287002 ± 0.0262 | 0.0172226 | 0.0417063 | 0.0684274 | 0.00638334 | 0.00976139 | 0.0185046 ± 0.00954 |
| A16 QQ RMSE ↓ | 0.00134584 ± 0.00046 | 0.00134266 | 0.000725318 | 0.00111678 | 0.00162237 | 0.00192206 | 6.22975e-05 ± 1.34e-05 |
| A17 Terminal KS ↓ | 0.0856934 ± 0.015 | 0.108398 | 0.0710449 | 0.0771484 | 0.0928955 | 0.0789795 | 0.0102539 ± 0.00242 |
| **Adversarial** | | | | | | | |
| A18 Discriminative (GRU) ↓ | 0.0043026 ± 0.00411 | 0.007476 | 0.001373 | 0.009918 | 0.001068 | 0.001678 | 0.0057674 ± 0.00447 |
| A18 Discriminative (MLP) ↓ | 0.0063778 ± 0.00496 | 0.011749 | 0.007476 | 0.010223 | 0.000763 | 0.001678 | 0.0037534 ± 0.00204 |
| **Predictive** | | | | | | | |
| A19 Predictive (GRU) ↓ | 0.0224856 ± 2.77e-05 | 0.022495 | 0.022445 | 0.022482 | 0.022522 | 0.022484 | 0.0224832 ± 3.98e-05 |
| A19 Predictive (MLP) ↓ | 0.022774 ± 0.000367 | 0.02292 | 0.022375 | 0.022936 | 0.023225 | 0.022414 | 0.02249 ± 0.000189 |
| **Temporal** | | | | | | | |
| A20 Covariance Error ↓ | 111.38 ± 16.3 | 90.1325 | 97.5489 | 126.117 | 122.607 | 120.494 | 23.4477 ± 17.3 |
| A21 ACF \|r\| ↓ | 0.0164254 ± 0.00591 | 0.00773636 | 0.0165272 | 0.0156122 | 0.0242304 | 0.0180208 | 0.00101954 ± 0.000509 |
| A22 ACF r² ↓ | 0.00991677 ± 0.00214 | 0.00688835 | 0.00934333 | 0.00980897 | 0.012756 | 0.0107872 | 0.00120096 ± 0.000344 |
| A23 ACF lag-1 \|r\| Error ↓ | 0.0234891 ± 0.0207 | 0.0125557 | 0.00838539 | 0.0055199 | 0.0514881 | 0.0394962 | 0.00103214 ± 0.00101 |
| A24 ACF lag-1 r² Error ↓ | 0.0137397 ± 0.0122 | 0.012673 | 0.0031255 | 0.000893137 | 0.0284948 | 0.0235121 | 0.00137151 ± 0.000644 |
| **Volatility** | | | | | | | |
| A25 Mean RMSE ↓ | 2.15933 ± 0.711 | 2.64913 | 1.63994 | 1.70305 | 3.17195 | 1.63256 | 0.14112 ± 0.0836 |
| A26 Std Error ↓ | 0.164483 ± 0.127 | 0.216331 | 0.0500623 | 0.0113496 | 0.299856 | 0.244819 | 0.0176179 ± 0.00892 |
| A27 Log-return Std Error ↓ | 0.00125334 ± 0.000816 | 0.00155867 | 0.000212097 | 0.000586779 | 0.00178655 | 0.0021226 | 6.30604e-05 ± 3.92e-05 |
| A28 Kurtosis Ratio → 1 | 1.4313 ± 0.308 | 1.41783 | 1.76767 | 1.7157 | 1.12514 | 1.13014 | 0.976711 ± 0.029 |
| A29 Sigma Mean Error ↓ | 0.0172935 ± 0.00944 | 0.0164448 | 0.00402525 | 0.0145605 | 0.0218523 | 0.0295848 | 0.000902254 ± 0.000441 |
| A30 Vol Path RMSE ↓ | 2.05253 ± 0.42 | 1.46257 | 1.82161 | 2.51417 | 2.34431 | 2.12001 | 0.404985 ± 0.0995 |
| A31 Rolling Vol KS ↓ | 0.0626278 ± 0.015 | 0.0447264 | 0.0497918 | 0.0700297 | 0.0675605 | 0.0810309 | 0.00307538 ± 0.0011 |
| A32 Vol-of-vol Error ↓ | 0.000835684 ± 0.000468 | 0.00128084 | 0.000665451 | 0.000103247 | 0.00100507 | 0.00112382 | 4.27908e-05 ± 4.1e-05 |
| **Heston-specific (dropped)** | | | | | | | |
| A33 Sigma Correlation (dropped) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| A34 Sigma RMSE (dropped) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

===== SECTION 2.2 : B curve-shape =====
| Plot | Measure | TimeDiT (mean ± std) | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|---|---|---|---|---|---|---|---|---|
| **Grid TVD (%)** | — | 961.103 ± 138 | 961.571 | 839.109 | 913.649 | 896.416 | 1194.77 | 139.194 ± 33 |
| **Log-return histogram** | MSE (funct/der/sec-der avg) | 26.7487 ± 11.2 | 33.3637 | 36.405 | 31.9082 | 23.4843 | 8.58233 | 0.0555117 ± 0.0114 |
|  | % error | 106.005 ± 19.3 | 93.9254 | 125.191 | 90.5994 | 128.816 | 91.4929 | 89.34 ± 7.48 |
|  | NRMSE | 3.89368 ± 0.883 | 4.35156 | 4.63705 | 4.32565 | 3.72486 | 2.42926 | 0.183744 ± 0.0176 |
|  | CVaR₉₀ | 3.85474 ± 0.649 | 3.75324 | 3.07656 | 3.50925 | 4.15309 | 4.78157 | 0.266769 ± 0.0363 |
|  | CVaR₉₅ | 6.7665 ± 1.24 | 6.93784 | 5.45598 | 5.70586 | 7.21445 | 8.51836 | 0.344172 ± 0.0567 |
| **QQ plot** | MSE (funct/der/sec-der avg) | 7.82102e-07 ± 4.43e-07 | 8.10527e-07 | 2.76637e-07 | 4.42306e-07 | 9.88937e-07 | 1.3921e-06 | 2.41572e-09 ± 1e-09 |
|  | % error | 85.8822 ± 7.63 | 85.237 | 89.7649 | 90.7421 | 72.8566 | 90.8103 | 7.56456 ± 0.242 |
|  | NRMSE | 2.6698 ± 0.435 | 3.1878 | 2.54586 | 2.05047 | 2.60531 | 2.95955 | 0.293045 ± 0.082 |
|  | CVaR₉₀ | 3.1797 ± 1.25 | 3.90355 | 1.66935 | 2.04797 | 3.74612 | 4.53151 | 0.148592 ± 0.0357 |
|  | CVaR₉₅ | 3.85011 ± 1.61 | 5.14757 | 2.1826 | 2.10067 | 4.39209 | 5.42765 | 0.195192 ± 0.0546 |
| **ACF of \|log-returns\|** | MSE (funct/der/sec-der avg) | 8.3064e-05 ± 3.38e-05 | 2.78618e-05 | 0.00011101 | 0.00010224 | 9.99557e-05 | 7.4253e-05 | 5.46152e-06 ± 2.19e-06 |
|  | % error | 127.255 ± 30.2 | 97.0718 | 143.224 | 167.336 | 98.0384 | 130.605 | 85.0322 ± 20.7 |
|  | NRMSE | 36.8758 ± 4.81 | 29.9894 | 34.5985 | 41.5831 | 37.105 | 41.103 | 23.8351 ± 5.21 |
|  | CVaR₉₀ | 15.3983 ± 6.88 | 6.78044 | 13.9569 | 12.9233 | 25.4194 | 17.9111 | 2.15097 ± 0.331 |
|  | CVaR₉₅ | 17.8599 ± 9.29 | 7.52182 | 14.0651 | 13.2061 | 30.8454 | 23.6613 | 2.45396 ± 0.388 |
| **ACF of squared log-returns** | MSE (funct/der/sec-der avg) | 3.46726e-05 ± 1.07e-05 | 1.84155e-05 | 3.98257e-05 | 4.75429e-05 | 3.28427e-05 | 3.47363e-05 | 5.52755e-06 ± 2.53e-06 |
|  | % error | 184.083 ± 42.6 | 163.124 | 181.59 | 257.819 | 151.97 | 165.911 | 115.528 ± 42 |
|  | NRMSE | 38.1508 ± 4.28 | 34.5181 | 33.5166 | 42.7905 | 37.6765 | 42.252 | 28.6009 ± 6.81 |
|  | CVaR₉₀ | 14.404 ± 3.97 | 9.21133 | 12.9395 | 14.2059 | 20.0973 | 15.566 | 2.85902 ± 0.497 |
|  | CVaR₉₅ | 17.4499 ± 6.15 | 11.6046 | 12.9404 | 15.0819 | 26.0926 | 21.53 | 3.28585 ± 0.731 |
| **Rolling volatility histogram** | MSE (funct/der/sec-der avg) | 29.1676 ± 23.1 | 9.1708 | 8.67968 | 30.8183 | 31.9112 | 65.2582 | 0.373539 ± 0.113 |
|  | % error | 218.818 ± 53.4 | 194.686 | 211.234 | 300.109 | 155.78 | 232.281 | 194.618 ± 50.5 |
|  | NRMSE | 18.8267 ± 10.7 | 9.81411 | 9.2632 | 22.2741 | 17.5853 | 35.1967 | 3.10664 ± 0.504 |
|  | CVaR₉₀ | 12.0968 ± 3.5 | 8.35511 | 9.22843 | 17.0642 | 13.4931 | 12.3433 | 0.853517 ± 0.176 |
|  | CVaR₉₅ | 18.0461 ± 6.29 | 11.5792 | 10.8314 | 22.1547 | 23.7152 | 21.95 | 1.0985 ± 0.26 |
| **Tail survival** | MSE (funct/der/sec-der avg) | 0.000379022 ± 0.000237 | 0.000104365 | 0.000225343 | 0.000374063 | 0.00046849 | 0.000722848 | 7.92185e-07 ± 5.62e-07 |
|  | % error | 7907.09 ± 1.32e+03 | 8141.2 | 6303.5 | 7246.89 | 7953.34 | 9890.54 | 4143.5 ± 296 |
|  | NRMSE | 135452 ± 2.53e+04 | 125294 | 113033 | 117965 | 146570 | 174399 | 10943.5 ± 446 |
|  | CVaR₉₀ | 4.82038 ± 0.992 | 3.79433 | 4.21799 | 4.58626 | 5.14606 | 6.35726 | 0.233416 ± 0.0789 |
|  | CVaR₉₅ | 5.0816 ± 0.767 | 4.47669 | 4.66564 | 4.72485 | 5.16412 | 6.37671 | 0.240094 ± 0.0794 |
