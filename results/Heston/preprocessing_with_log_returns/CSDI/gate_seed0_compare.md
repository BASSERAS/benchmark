# Seed-0 GATE: CSDI log-return preproc vs original CSDI

New diagnostics figure: `plots/heston_diagnostics.png`

Original figure: `../../CSDI/plots/heston_diagnostics.png`

NOTE: original CSDI = 8192 paths, this run = 4096 paths (documented confound, GUIDELINE 7).


## A-metrics (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| A1_kurtosis_error | 0.117746 | 0.0914827 | -22.3% |
| A2_abs_r_q95_error | 0.00532261 | 0.00313604 | -41.1% |
| A3_abs_r_q99_error | 0.00712637 | 0.00453663 | -36.3% |
| A4_tail_qq_error | 0.00521107 | 0.00307066 | -41.1% |
| A5_hill_tail_index_error | 1.73662 | 2.1053 | +21.2% |
| A6_path_mmd2 | 0.00376652 | 0.00363378 | -3.5% |
| A7_terminal_mmd2 | 0.00298195 | 0.00517253 | +73.5% |
| A8_increment_mmd2 | 0.00741964 | 0.00238525 | -67.9% |
| A9_volatility_mmd | 0.227465 | 0.0628576 | -72.4% |
| A10_terminal_swd | 1.44573 | 3.10513 | +114.8% |
| A11_path_swd | 1.07895 | 1.46242 | +35.5% |
| A12_rv_law_loss | 1.89204 | 1.21732 | -35.7% |
| A13_mean_path_rmse | 0.197892 | 1.98768 | +904.4% |
| A14_ks_logreturns | 0.0528295 | 0.0377015 | -28.6% |
| A15_skewness_error | 0.0386523 | 0.0391988 | +1.4% |
| A16_qq_rmse | 0.00253833 | 0.00156237 | -38.4% |
| A17_terminal_ks | 0.0308838 | 0.104736 | +239.1% |
| A18_disc_score_gru | 0.002289 | 0.008847 | +286.5% |
| A18_disc_score_mlp | 0.010833 | 0.005186 | -52.1% |
| A19_pred_score_gru | 0.050241 | 0.056376 | +12.2% |
| A19_pred_score_mlp | 0.050155 | 0.056256 | +12.2% |
| A20_cov_error | 41.8088 | 49.4471 | +18.3% |
| A21_acf_abs | 0.00979924 | 0.0110421 | +12.7% |
| A22_acf_sq | 0.0101803 | 0.00692952 | -31.9% |
| A23_acf_lag1_abs_error | 0.0222728 | 0.00832932 | -62.6% |
| A24_acf_lag1_sq_error | 0.0218587 | 0.00467127 | -78.6% |
| A25_mean_rmse | 0.342393 | 3.3171 | +868.8% |
| A26_std_error | 0.255194 | 0.132394 | -48.1% |
| A27_logreturn_std_error | 0.00262113 | 0.0016152 | -38.4% |
| A28_kurtosis_ratio | 0.825705 | 1.01814 | +23.3% |
| A29_sigma_mean_error | 0.0401719 | 0.0245837 | -38.8% |
| A30_vol_path_rmse | 1.16022 | 1.20019 | +3.4% |
| A31_rolling_vol_ks | 0.216628 | 0.109907 | -49.3% |
| A32_vol_of_vol_error | 0.00101612 | 0.000573626 | -43.5% |
| A33_sigma_corr | 0.000919131 | 0.0105956 | +1052.8% |
| A34_sigma_rmse | 0.0993194 | 0.0983085 | -1.0% |

## B curve funct MSE + %err (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| B_acf_abs_r_funct | 3.74931e-05 | 0.000150276 | +300.8% |
| B_acf_sq_r_funct | 3.75673e-05 | 7.68045e-05 | +104.4% |
| B_log_ret_hist_funct | 13.0582 | 4.91518 | -62.4% |
| B_qq_plot_funct | 6.77569e-06 | 2.58633e-06 | -61.8% |
| B_roll_vol_hist_funct | 449.342 | 104.789 | -76.7% |
| B_tail_surv_funct | 0.00572188 | 0.00209786 | -63.3% |
| B_acf_abs_r_funct_pct | 11.944 | 48.3974 | +305.2% |
| B_acf_sq_r_funct_pct | 14.0485 | 44.5332 | +217.0% |
| B_log_ret_hist_funct_pct | 34.7877 | 21.2536 | -38.9% |
| B_qq_plot_funct_pct | 23.7302 | 21.0731 | -11.2% |
| B_roll_vol_hist_funct_pct | 60.9625 | 36.4043 | -40.3% |
| B_tail_surv_funct_pct | 24.4572 | 15.1667 | -38.0% |
