# Seed-0 GATE: TimeDiT log-return preproc vs original TimeDiT

New diagnostics figure: `plots/heston_diagnostics.png`

Original figure: `../../TimeDiT/plots/heston_diagnostics.png`


## A-metrics (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| A1_kurtosis_error | 0.061219 | 0.567194 | +826.5% |
| A2_abs_r_q95_error | 0.000509574 | 0.00201158 | +294.8% |
| A3_abs_r_q99_error | 0.000708298 | 0.0045515 | +542.6% |
| A4_tail_qq_error | 0.000489441 | 0.00203274 | +315.3% |
| A5_hill_tail_index_error | 0.216019 | 5.62757 | +2505.1% |
| A6_path_mmd2 | 0.00242749 | 0.00806238 | +232.1% |
| A7_terminal_mmd2 | 0.00219327 | 0.0113652 | +418.2% |
| A8_increment_mmd2 | 0.00144504 | 0.0014442 | -0.1% |
| A9_volatility_mmd | 0.0224849 | 0.0382201 | +70.0% |
| A10_terminal_swd | 0.715087 | 4.88408 | +583.0% |
| A11_path_swd | 1.0427 | 2.70866 | +159.8% |
| A12_rv_law_loss | 0.2323 | 0.740206 | +218.6% |
| A13_mean_path_rmse | 1.69683 | 2.46824 | +45.5% |
| A14_ks_logreturns | 0.00979158 | 0.0151867 | +55.1% |
| A15_skewness_error | 0.0199322 | 0.0633881 | +218.0% |
| A16_qq_rmse | 0.000325695 | 0.000916167 | +181.3% |
| A17_terminal_ks | 0.0583496 | 0.133057 | +128.0% |
| A18_disc_score_gru | 0.003814 | 0.002135 | -44.0% |
| A18_disc_score_mlp | 0.080714 | 0.112569 | +39.5% |
| A19_pred_score_gru | 0.050115 | 0.056464 | +12.7% |
| A19_pred_score_mlp | 0.050898 | 0.056311 | +10.6% |
| A20_cov_error | 15.2028 | 62.4059 | +310.5% |
| A21_acf_abs | 0.0102475 | 0.0101717 | -0.7% |
| A22_acf_sq | 0.00854029 | 0.0124052 | +45.3% |
| A23_acf_lag1_abs_error | 0.0181757 | 0.0196353 | +8.0% |
| A24_acf_lag1_sq_error | 0.0156611 | 0.0210888 | +34.7% |
| A25_mean_rmse | 1.57052 | 4.17138 | +165.6% |
| A26_std_error | 0.0438435 | 0.0592832 | +35.2% |
| A27_logreturn_std_error | 0.000291444 | 0.000959539 | +229.2% |
| A28_kurtosis_ratio | 1.02487 | 4.03666 | +293.9% |
| A29_sigma_mean_error | 0.00415323 | 0.0130825 | +215.0% |
| A30_vol_path_rmse | 0.222417 | 1.59718 | +618.1% |
| A31_rolling_vol_ks | 0.0352694 | 0.0442133 | +25.4% |
| A32_vol_of_vol_error | 0.000113944 | 0.000596607 | +423.6% |
| A33_sigma_corr | 0.00276815 | 0.0110322 | +298.5% |
| A34_sigma_rmse | 0.0998842 | 0.0938012 | -6.1% |

## B curve funct MSE + %err (seed 0)

| Metric | Original s0 | New s0 | delta% |
|--------|------------:|-------:|-------:|
| B_acf_abs_r_funct | 4.10145e-05 | 5.05407e-05 | +23.2% |
| B_acf_sq_r_funct | 2.90091e-05 | 7.87369e-05 | +171.4% |
| B_log_ret_hist_funct | 0.31798 | 0.277489 | -12.7% |
| B_qq_plot_funct | 1.10409e-07 | 1.02391e-06 | +827.4% |
| B_roll_vol_hist_funct | 11.516 | 12.9102 | +12.1% |
| B_tail_surv_funct | 8.83498e-05 | 0.000141482 | +60.1% |
| B_acf_abs_r_funct_pct | 15.8451 | 20.0102 | +26.3% |
| B_acf_sq_r_funct_pct | 16.7531 | 35.0401 | +109.2% |
| B_log_ret_hist_funct_pct | 4.33653 | 9.38958 | +116.5% |
| B_qq_plot_funct_pct | 3.14759 | 13.803 | +338.5% |
| B_roll_vol_hist_funct_pct | 11.4456 | 22.5005 | +96.6% |
| B_tail_surv_funct_pct | 2.91711 | 6.95071 | +138.3% |
