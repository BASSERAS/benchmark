# Seed-0 GATE: LS4 log-return preproc vs original LS4

New diagnostics figure: `plots/heston_diagnostics.png`

Original figure: `../../LS4/plots/heston_diagnostics.png`


## A-metrics (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| A1_kurtosis_error | 0.390043 | 0.219152 | -43.8% |
| A2_abs_r_q95_error | 0.00051269 | 0.0020367 | +297.3% |
| A3_abs_r_q99_error | 0.00127766 | 0.00254538 | +99.2% |
| A4_tail_qq_error | 0.000485352 | 0.00196146 | +304.1% |
| A5_hill_tail_index_error | 1.85751 | 1.45889 | -21.5% |
| A6_path_mmd2 | 0.0018075 | 0.00186555 | +3.2% |
| A7_terminal_mmd2 | 0.00114361 | 0.00129402 | +13.2% |
| A8_increment_mmd2 | 0.000923958 | 0.00131016 | +41.8% |
| A9_volatility_mmd | 0.012398 | 0.0329394 | +165.7% |
| A10_terminal_swd | 0.450716 | 0.80591 | +78.8% |
| A11_path_swd | 0.418172 | 0.537705 | +28.6% |
| A12_rv_law_loss | 0.251759 | 0.727464 | +189.0% |
| A13_mean_path_rmse | 0.407092 | 0.239939 | -41.1% |
| A14_ks_logreturns | 0.01347 | 0.0158595 | +17.7% |
| A15_skewness_error | 0.0233848 | 0.140043 | +498.9% |
| A16_qq_rmse | 0.000339146 | 0.000895504 | +164.0% |
| A17_terminal_ks | 0.0253906 | 0.0251465 | -1.0% |
| A18_disc_score_gru | 0.004425 | 0.011897 | +168.9% |
| A18_disc_score_mlp | 0.009307 | 0.02288 | +145.8% |
| A19_pred_score_gru | 0.050013 | 0.056401 | +12.8% |
| A19_pred_score_mlp | 0.050115 | 0.056254 | +12.2% |
| A20_cov_error | 20.4973 | 25.8065 | +25.9% |
| A21_acf_abs | 0.0150427 | 0.00847435 | -43.7% |
| A22_acf_sq | 0.00873071 | 0.00797791 | -8.6% |
| A23_acf_lag1_abs_error | 0.026171 | 0.0179727 | -31.3% |
| A24_acf_lag1_sq_error | 0.0178567 | 0.015628 | -12.5% |
| A25_mean_rmse | 0.753823 | 0.448466 | -40.5% |
| A26_std_error | 0.00113942 | 0.0703641 | +6075.4% |
| A27_logreturn_std_error | 3.84465e-05 | 0.000882894 | +2196.4% |
| A28_kurtosis_ratio | 1.50838 | 0.992195 | -34.2% |
| A29_sigma_mean_error | 0.000750857 | 0.0115968 | +1444.5% |
| A30_vol_path_rmse | 0.431354 | 0.53043 | +23.0% |
| A31_rolling_vol_ks | 0.0390873 | 0.0557216 | +42.6% |
| A32_vol_of_vol_error | 0.0003483 | 0.000546728 | +57.0% |
| A33_sigma_corr | 0.00483266 | 0.0118425 | +145.1% |
| A34_sigma_rmse | 0.0944767 | 0.0954774 | +1.1% |

## B curve funct MSE + %err (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| B_acf_abs_r_funct | 0.000148881 | 4.72389e-05 | -68.3% |
| B_acf_sq_r_funct | 5.47136e-05 | 4.06397e-05 | -25.7% |
| B_log_ret_hist_funct | 0.989696 | 0.895318 | -9.5% |
| B_qq_plot_funct | 1.27294e-07 | 8.57041e-07 | +573.3% |
| B_roll_vol_hist_funct | 24.7358 | 29.4509 | +19.1% |
| B_tail_surv_funct | 0.000183221 | 0.000397426 | +116.9% |
| B_acf_abs_r_funct_pct | 38.5681 | 20.7767 | -46.1% |
| B_acf_sq_r_funct_pct | 24.8466 | 23.3526 | -6.0% |
| B_log_ret_hist_funct_pct | 5.49964 | 12.9689 | +135.8% |
| B_qq_plot_funct_pct | 6.81285 | 9.85544 | +44.7% |
| B_roll_vol_hist_funct_pct | 12.7187 | 22.0252 | +73.2% |
| B_tail_surv_funct_pct | 3.26471 | 7.88542 | +141.5% |
