# Results — Methods Comparison on Heston

All methods are evaluated on the same dataset:
**8 192 Heston price paths, seq\_len = 128**
(μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250)

Every method is trained on the **train** split (seed 0) and scored against the **held-out test set** (an
independent 8 192-path Heston draw, seed 1). The **Perfect floor** is a *fresh* independent Heston
simulation (seeds 1000+) scored against the same test set exactly as every method is — so it is the
**non-zero** finite-sample noise a genuine Heston sample cannot avoid, not a degenerate zero. See
[`../methods/perfect_recovery/README.md`](../methods/perfect_recovery/README.md).

---

## A1–A34 — Cross-method comparison (mean ± std, 5 seeds)

Methods are grouped by model family. ↓ = lower is better, ↑ = higher is better, A28 target = 1.0.
**Bold** = best across methods.

<table>
<thead>
  <tr>
    <th rowspan="2">Metric</th>
    <th colspan="2">GAN</th>
    <th colspan="2">Diffusion</th>
    <th colspan="3">VAE</th>
    <th>Schrödinger Bridge</th>
    <th>Fourier Flow</th>
    <th rowspan="2">Perfect</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>TimeGAN</th>
    <th>COSCI-GAN</th>
    <th>Diffusion-TS</th>
    <th>CSDI</th>
    <th>TimeVAE</th>
    <th>TimeVQVAE</th>
    <th>LS4</th>
    <th>SBTS</th>
    <th>Fourier Flow</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="12"><b>— Fat Tail —</b></td></tr>
  <tr><td>A1 Kurtosis Error ↓</td><td>2.954 ± 2.098</td><td>0.5615 ± 0.1128</td><td>0.4242 ± 0.02303</td><td><b>0.09543 ± 0.02623</b></td><td>2.257 ± 0.5719</td><td>0.1363 ± 0.09243</td><td>0.3684 ± 0.01609</td><td>0.1183 ± 0.006001</td><td>0.5761 ± 0.008273</td><td>0.008092 ± 0.006811</td><td><b>CSDI</b></td></tr>
  <tr><td>A2 \|r\| q95 Error ↓</td><td>0.003196 ± 0.001907</td><td>0.09711 ± 0.003466</td><td>0.006902 ± 1.57e-04</td><td>0.005393 ± 1.50e-04</td><td>0.02227 ± 1.22e-04</td><td>0.004515 ± 2.54e-04</td><td><b>3.99e-04 ± 1.13e-04</b></td><td>0.006390 ± 2.97e-05</td><td>7.21e-04 ± 2.10e-04</td><td>6.57e-05 ± 5.96e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A3 \|r\| q99 Error ↓</td><td>0.004342 ± 0.002767</td><td>0.1240 ± 0.005959</td><td>0.01032 ± 1.75e-04</td><td>0.007327 ± 2.29e-04</td><td>0.03082 ± 1.05e-04</td><td>0.006058 ± 3.03e-04</td><td><b>0.001156 ± 1.66e-04</b></td><td>0.009803 ± 4.84e-05</td><td>0.002325 ± 5.06e-04</td><td>5.98e-05 ± 3.25e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A4 Tail QQ Error ↓</td><td>0.003401 ± 0.001522</td><td>0.09566 ± 0.003535</td><td>0.006781 ± 1.50e-04</td><td>0.005296 ± 1.50e-04</td><td>0.02191 ± 1.17e-04</td><td>0.004444 ± 2.48e-04</td><td><b>4.05e-04 ± 8.23e-05</b></td><td>0.006290 ± 2.63e-05</td><td>7.42e-04 ± 1.38e-04</td><td>6.75e-05 ± 3.70e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A5 Hill Tail Index Error ↓</td><td>36.32 ± 17.05</td><td>1.614 ± 1.128</td><td>3.047 ± 0.2789</td><td>1.426 ± 0.5856</td><td>1.831 ± 0.6794</td><td>3.777 ± 1.193</td><td><b>1.225 ± 0.4268</b></td><td>10.06 ± 0.3457</td><td>5.802 ± 2.000</td><td>0.5266 ± 0.5572</td><td><b>LS4</b></td></tr>
  <tr><td colspan="12"><b>— Distribution —</b></td></tr>
  <tr><td>A6 Path MMD² ↓</td><td>0.01866 ± 0.01472</td><td>0.04686 ± 0.004162</td><td>0.004476 ± 8.48e-04</td><td>0.003646 ± 4.16e-04</td><td>0.01914 ± 0.001334</td><td>0.003433 ± 7.97e-04</td><td><b>0.001926 ± 2.51e-04</b></td><td>0.01106 ± 8.13e-04</td><td>0.005527 ± 0.002289</td><td>0.001842 ± 2.55e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A7 Terminal MMD² ↓</td><td>0.03072 ± 0.02472</td><td>0.01623 ± 0.01333</td><td>0.003676 ± 0.001070</td><td>0.003605 ± 8.41e-04</td><td>0.004951 ± 0.001715</td><td>0.003838 ± 0.001368</td><td><b>0.001520 ± 3.61e-04</b></td><td>0.009545 ± 0.001668</td><td>0.01105 ± 0.006414</td><td>0.001983 ± 8.89e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A8 Increment MMD² ↓</td><td>0.008280 ± 0.004303</td><td>0.4788 ± 0.01185</td><td>0.01109 ± 7.52e-04</td><td>0.008062 ± 7.11e-04</td><td>0.2130 ± 0.001204</td><td>0.007018 ± 0.001054</td><td><b>9.63e-04 ± 3.76e-05</b></td><td>0.007378 ± 3.39e-04</td><td>0.001124 ± 6.46e-05</td><td>8.69e-04 ± 2.70e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A9 Volatility MMD ↓</td><td>0.3975 ± 0.2486</td><td>3.955 ± 0.04883</td><td>0.3846 ± 0.02464</td><td>0.2498 ± 0.01607</td><td>3.575 ± 0.4476</td><td>0.1932 ± 0.02799</td><td><b>0.01447 ± 0.001550</b></td><td>0.3139 ± 0.01207</td><td>0.05871 ± 0.007003</td><td>0.008554 ± 0.001549</td><td><b>LS4</b></td></tr>
  <tr><td>A10 Terminal SWD ↓</td><td>2.917 ± 1.131</td><td>4.756 ± 3.118</td><td>1.684 ± 0.3010</td><td>1.618 ± 0.2760</td><td>1.947 ± 0.3598</td><td>1.356 ± 0.2690</td><td><b>0.7480 ± 0.3255</b></td><td>3.710 ± 0.2944</td><td>2.710 ± 1.034</td><td>1.151 ± 0.4868</td><td><b>LS4</b></td></tr>
  <tr><td>A11 Path SWD ↓</td><td>1.678 ± 0.5770</td><td>3.505 ± 0.1711</td><td>1.212 ± 0.1556</td><td>1.069 ± 0.1305</td><td>1.167 ± 0.1135</td><td>0.8781 ± 0.2081</td><td><b>0.5744 ± 0.1246</b></td><td>2.498 ± 0.1451</td><td>1.334 ± 0.3806</td><td>0.6191 ± 0.1960</td><td><b>LS4</b></td></tr>
  <tr><td>A12 RV Law Loss ↓</td><td>1.558 ± 0.3879</td><td>118.7 ± 7.929</td><td>2.274 ± 0.04910</td><td>1.920 ± 0.05633</td><td>5.010 ± 0.008395</td><td>1.706 ± 0.08942</td><td><b>0.2415 ± 0.01757</b></td><td>2.175 ± 0.007357</td><td>0.5397 ± 0.1300</td><td>0.05202 ± 0.006560</td><td><b>LS4</b></td></tr>
  <tr><td>A13 Mean Path RMSE ↓</td><td>0.5356 ± 0.2514</td><td>3.995 ± 0.1803</td><td>0.4399 ± 0.2584</td><td>0.3654 ± 0.3226</td><td>0.3196 ± 0.2225</td><td>0.7593 ± 0.1340</td><td><b>0.1722 ± 0.1200</b></td><td>0.8477 ± 0.1819</td><td>0.4336 ± 0.3651</td><td>0.1205 ± 0.05175</td><td><b>LS4</b></td></tr>
  <tr><td>A14 KS Log-returns ↓</td><td>0.08474 ± 0.03769</td><td>0.3206 ± 0.007269</td><td>0.06048 ± 0.001904</td><td>0.05391 ± 0.001972</td><td>0.3670 ± 0.004602</td><td>0.05084 ± 0.003747</td><td><b>0.01258 ± 6.74e-04</b></td><td>0.05413 ± 3.75e-04</td><td>0.01895 ± 0.002028</td><td>0.001491 ± 5.79e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A15 Skewness Error ↓</td><td>0.3412 ± 0.3279</td><td>0.04981 ± 0.04124</td><td>0.06445 ± 0.03230</td><td>0.03681 ± 0.002124</td><td>0.5479 ± 0.09837</td><td>0.03079 ± 0.008248</td><td>0.02998 ± 0.01249</td><td>0.03158 ± 0.003742</td><td><b>0.02288 ± 0.01115</b></td><td>0.005274 ± 0.001459</td><td><b>Fourier Flow</b></td></tr>
  <tr><td>A16 QQ RMSE (300-pt) ↓</td><td>0.002506 ± 6.49e-04</td><td>0.04857 ± 0.001967</td><td>0.003073 ± 8.32e-05</td><td>0.002576 ± 8.57e-05</td><td>0.01057 ± 8.40e-05</td><td>0.002268 ± 1.38e-04</td><td><b>3.41e-04 ± 9.53e-06</b></td><td>0.002853 ± 1.15e-05</td><td>5.81e-04 ± 4.14e-05</td><td>4.19e-05 ± 1.89e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A17 Terminal Price KS ↓</td><td>0.1109 ± 0.05875</td><td>0.1473 ± 0.09804</td><td>0.04436 ± 0.007030</td><td>0.03667 ± 0.004476</td><td>0.05127 ± 0.007848</td><td>0.05522 ± 0.009093</td><td><b>0.01584 ± 0.005488</b></td><td>0.09102 ± 0.005462</td><td>0.08098 ± 0.01617</td><td>0.01099 ± 0.001563</td><td><b>LS4</b></td></tr>
  <tr><td colspan="12"><b>— Adversarial —</b></td></tr>
  <tr><td>A18 Disc Score GRU ↓</td><td>0.03305 ± 0.05328</td><td>0.4999 ± 1.22e-04</td><td>0.08987 ± 0.1524</td><td>0.06302 ± 0.1056</td><td>0.4272 ± 0.08815</td><td>0.07174 ± 0.06503</td><td><b>0.005890 ± 0.001676</b></td><td>0.1246 ± 0.1517</td><td>0.009185 ± 0.009209</td><td>0.006195 ± 0.007171</td><td><b>LS4</b></td></tr>
  <tr><td>A18 Disc Score MLP ↓</td><td>0.08792 ± 0.04703</td><td>0.5000 ± 0</td><td>0.02426 ± 0.03140</td><td>0.01138 ± 0.002541</td><td>0.1358 ± 0.1503</td><td>0.009002 ± 0.003393</td><td>0.006256 ± 0.002539</td><td>0.008331 ± 0.004230</td><td><b>0.005951 ± 0.002921</b></td><td>0.005951 ± 0.003469</td><td><b>Fourier Flow</b></td></tr>
  <tr><td colspan="12"><b>— Predictive —</b></td></tr>
  <tr><td>A19 Pred Score GRU ↓</td><td>0.05277 ± 0.001115</td><td>0.1331 ± 0.01808</td><td>0.05112 ± 1.22e-04</td><td>0.05024 ± 1.88e-05</td><td>0.05385 ± 7.71e-04</td><td>0.05014 ± 2.87e-05</td><td><b>0.05001 ± 3.66e-06</b></td><td>0.05453 ± 3.55e-05</td><td>0.05004 ± 2.00e-05</td><td>0.05002 ± 1.08e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A19 Pred Score MLP ↓</td><td>0.05322 ± 0.001031</td><td>0.09591 ± 0.006992</td><td>0.05112 ± 1.21e-04</td><td>0.05025 ± 1.43e-04</td><td>0.05243 ± 1.91e-04</td><td>0.05018 ± 6.79e-05</td><td><b>0.05006 ± 1.23e-04</b></td><td>0.05428 ± 3.54e-04</td><td>0.05032 ± 3.48e-04</td><td>0.05036 ± 6.63e-04</td><td><b>LS4</b></td></tr>
  <tr><td colspan="12"><b>— Temporal —</b></td></tr>
  <tr><td>A20 Covariance Error ↓</td><td>21.36 ± 9.068</td><td>30.59 ± 29.16</td><td>44.18 ± 10.64</td><td>41.55 ± 5.776</td><td>57.28 ± 1.758</td><td>22.61 ± 14.72</td><td><b>13.63 ± 6.662</b></td><td>139.3 ± 4.886</td><td>60.80 ± 36.58</td><td>4.923 ± 3.284</td><td><b>LS4</b></td></tr>
  <tr><td>A21 ACF \|r\| Error (lags) ↓</td><td>0.1278 ± 0.06738</td><td>0.08056 ± 0.02054</td><td>0.01812 ± 0.002352</td><td><b>0.01126 ± 0.003095</b></td><td>0.3890 ± 0.1057</td><td>0.01979 ± 0.004246</td><td>0.01294 ± 0.001791</td><td>0.05886 ± 4.70e-04</td><td>0.04095 ± 5.50e-04</td><td>0.002234 ± 6.62e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A22 ACF r² Error (lags) ↓</td><td>0.08676 ± 0.03470</td><td>0.09004 ± 0.02156</td><td>0.01587 ± 0.002662</td><td>0.01124 ± 0.002605</td><td>0.3609 ± 0.08849</td><td>0.01817 ± 0.003251</td><td><b>0.006752 ± 0.001737</b></td><td>0.06136 ± 5.71e-04</td><td>0.03498 ± 5.56e-04</td><td>0.002206 ± 6.32e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A23 ACF \|r\| Lag-1 Error ↓</td><td>0.2301 ± 0.1034</td><td>0.1700 ± 0.04930</td><td><b>0.002410 ± 0.001465</b></td><td>0.02252 ± 0.004755</td><td>0.4674 ± 0.1346</td><td>0.01523 ± 0.008014</td><td>0.01743 ± 0.005532</td><td>0.1474 ± 0.001169</td><td>0.04897 ± 7.04e-04</td><td>0.002652 ± 0.001035</td><td><b>Diffusion-TS</b></td></tr>
  <tr><td>A24 ACF r² Lag-1 Error ↓</td><td>0.1760 ± 0.06259</td><td>0.1957 ± 0.05105</td><td><b>0.007895 ± 0.002645</b></td><td>0.02168 ± 0.003561</td><td>0.4630 ± 0.1189</td><td>0.01323 ± 0.007254</td><td>0.009068 ± 0.005290</td><td>0.1706 ± 0.001690</td><td>0.04195 ± 7.01e-04</td><td>0.002790 ± 9.39e-04</td><td><b>Diffusion-TS</b></td></tr>
  <tr><td colspan="12"><b>— Vol —</b></td></tr>
  <tr><td>A25 Mean RMSE ↓</td><td>0.7781 ± 0.3669</td><td>4.539 ± 3.359</td><td>0.7610 ± 0.4617</td><td>0.5139 ± 0.4595</td><td>0.3883 ± 0.2340</td><td>1.033 ± 0.1905</td><td><b>0.3270 ± 0.2333</b></td><td>1.499 ± 0.2776</td><td>0.7990 ± 0.7970</td><td>0.1392 ± 0.06359</td><td><b>LS4</b></td></tr>
  <tr><td>A26 Return Std Error ↓</td><td>0.1525 ± 0.08911</td><td>5.032 ± 0.2229</td><td>0.3107 ± 0.009292</td><td>0.2580 ± 0.009849</td><td>1.074 ± 0.007809</td><td>0.2316 ± 0.01420</td><td>0.004853 ± 0.003540</td><td>0.2501 ± 0.001833</td><td><b>0.004832 ± 0.002757</b></td><td>0.002523 ± 0.001767</td><td><b>Fourier Flow</b></td></tr>
  <tr><td>A27 Log-Return Std Error ↓</td><td>0.001703 ± 7.89e-04</td><td>0.04975 ± 0.002001</td><td>0.003240 ± 8.19e-05</td><td>0.002667 ± 8.89e-05</td><td>0.01098 ± 7.75e-05</td><td>0.002336 ± 1.37e-04</td><td><b>4.63e-05 ± 2.22e-05</b></td><td>0.003028 ± 1.23e-05</td><td>7.64e-05 ± 5.51e-05</td><td>3.15e-05 ± 2.48e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A28 Kurtosis Ratio (→ 1)</td><td>-1.116 ± 3.593</td><td>-8.150 ± 12.11</td><td>1.903 ± 0.2558</td><td><b>0.8706 ± 0.03043</b></td><td>0.2834 ± 0.04765</td><td>0.8410 ± 0.06953</td><td>1.565 ± 0.07840</td><td>2.028 ± 0.01851</td><td>3.098 ± 0.7754</td><td>1.006 ± 0.009834</td><td><b>CSDI</b></td></tr>
  <tr><td>A29 Sigma Mean Error ↓</td><td>0.03089 ± 0.009106</td><td>0.7871 ± 0.03094</td><td>0.04883 ± 0.001266</td><td>0.04078 ± 0.001489</td><td>0.1745 ± 0.001776</td><td>0.03743 ± 0.002059</td><td><b>0.001445 ± 6.99e-04</b></td><td>0.04432 ± 1.84e-04</td><td>0.002245 ± 8.77e-04</td><td>4.96e-04 ± 4.24e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A30 Cross-Sect. Vol Path RMSE ↓</td><td>0.4742 ± 0.2079</td><td>1.155 ± 0.3231</td><td>1.365 ± 0.2012</td><td>1.134 ± 0.1303</td><td>1.325 ± 0.04564</td><td>0.5701 ± 0.3404</td><td><b>0.3372 ± 0.1171</b></td><td>3.066 ± 0.06387</td><td>1.381 ± 0.4336</td><td>0.1432 ± 0.03018</td><td><b>LS4</b></td></tr>
  <tr><td>A31 Rolling Vol KS (w=5) ↓</td><td>0.2552 ± 0.1101</td><td>0.9371 ± 0.007667</td><td>0.2576 ± 0.007919</td><td>0.2202 ± 0.008329</td><td>0.9869 ± 0.004527</td><td>0.1850 ± 0.01013</td><td><b>0.03798 ± 0.001391</b></td><td>0.3456 ± 6.49e-04</td><td>0.07213 ± 0.001372</td><td>0.003814 ± 0.001210</td><td><b>LS4</b></td></tr>
  <tr><td>A32 Vol-of-Vol Error ↓</td><td>8.96e-04 ± 8.69e-04</td><td>0.01806 ± 0.001147</td><td>0.001587 ± 3.82e-05</td><td>0.001048 ± 2.14e-05</td><td>0.004576 ± 5.62e-05</td><td>6.76e-04 ± 5.79e-05</td><td><b>3.21e-04 ± 4.23e-05</b></td><td>0.002109 ± 5.57e-06</td><td>6.89e-04 ± 9.20e-05</td><td>1.54e-05 ± 9.93e-06</td><td><b>LS4</b></td></tr>
  <tr><td colspan="12"><b>— Heston Spec —</b></td></tr>
  <tr><td>A33 Teacher-Sigma Corr ↑</td><td>0.002745 ± 0.01354</td><td>-0.005511 ± 0.008042</td><td>0.001823 ± 0.004419</td><td>0.003948 ± 0.003596</td><td><b>0.02254 ± 0.003796</b></td><td>7.04e-04 ± 0.005837</td><td>-3.94e-04 ± 0.006577</td><td>0.002758 ± 0.002975</td><td>-0.002564 ± 0.002730</td><td>0.6163 ± 0.002371</td><td><b>TimeVAE</b></td></tr>
  <tr><td>A34 Teacher-Sigma RMSE ↓</td><td>0.1186 ± 0.01863</td><td>0.8087 ± 0.02874</td><td>0.09645 ± 9.09e-04</td><td>0.09917 ± 6.44e-04</td><td>0.1803 ± 0.001643</td><td>0.1014 ± 9.08e-04</td><td>0.09513 ± 7.87e-04</td><td>0.09615 ± 1.38e-04</td><td><b>0.08963 ± 0.001225</b></td><td>0.06559 ± 1.37e-04</td><td><b>Fourier Flow</b></td></tr>
</tbody>
</table>

> **A33 Teacher-Sigma Corr**: floor = **0.6163** (not 1.0) — the 5-step rolling quadratic-variation is a
> noisy estimator of instantaneous variance vₜ. TimeVAE (0.02254) has the highest correlation, then CSDI
> (0.003948), SBTS (0.002758), TimeGAN (0.002745), Diffusion-TS (0.001823), TimeVQVAE (7.0e-04), LS4
> (−3.9e-04), Fourier Flow (−0.002564) and COSCI-GAN (−0.005511) — the last three slightly negative. **None**
> meaningfully preserves stochastic volatility relative to the 0.6163 floor: TimeVAE merely wins a race among
> near-zero correlations, and LS4's single-factor latent-S4 prior cannot recover the two-factor Heston vol.
>
> **A28 Kurtosis Ratio**: target = 1.0. CSDI (0.8706) is closest — |CSDI−1| = 0.129 < |TimeVQVAE−1| = 0.159
> < |LS4−1| = 0.565 < |TimeVAE−1| = 0.717 < |DTS−1| = 0.903 < |SBTS−1| = 1.028 < |TimeGAN−1| = 2.116 <
> |FF−1| = 2.098 < |COSCI-GAN−1| = 9.150. LS4 (1.565) is mildly **platykurtic** — its single-factor latent
> decoder generates slightly thinner-than-Heston tails, the standard limitation of a one-factor generator on
> a two-factor SDE. TimeVAE (0.2834) is heavily under-dispersed. TimeGAN (−1.116) and COSCI-GAN (−8.150)
> have **negative** mean ratios (sign-flipping across seeds), the farthest from 1.

**LS4 wins 26 of 36 A-metrics; Fourier Flow 4; CSDI 3; Diffusion-TS 2; TimeVAE 1.** SBTS, TimeGAN,
COSCI-GAN and TimeVQVAE win none outright. (36 = the 34 metrics with A18 and A19 each split into a GRU and
an MLP variant.) With every method scored against the held-out **test set**, **LS4**'s latent-S4 state-space
prior dominates the benchmark: it sweeps the tail quantiles (A2–A4) and Hill index (A5), the entire
distributional family (A6–A14, A16, A17), **both** adversarial-GRU and predictive scores (A18-GRU
**0.005890**, A19-GRU **0.05001** / A19-MLP **0.05006**, all at or under the finite-sample floor), and most
of the temporal/vol family (A20, A22, A25, A27, A29–A32). Two structural weaknesses remain real: LS4 carries
**no latent-volatility recovery** (A33 σ-corr ≈ −4e-4, at the zero all single-factor generators share) and
its return tails run slightly thin (A28 kurtosis ratio 1.565 vs the ideal 1.0, A1 kurtosis error 0.368
mid-pack). See [`Heston/LS4/README.md`](Heston/LS4/README.md).

The other families defend narrow, interpretable niches. **CSDI**'s score-based diffusion keeps the three
metrics that reward its faithful vol-clustering autocorrelation and heaviest tails: the **kurtosis error**
(A1 **0.09543**, best of any method), the **ACF |r| lag-average** (A21 **0.01126**) and the **kurtosis ratio**
(A28 **0.8706**, the only method within 0.13 of 1.0). **Fourier Flow** takes four moment/near-Gaussian
metrics — **skewness** (A15 **0.02288**), the **MLP discriminative score** (A18-MLP **0.005951**, edging LS4),
the **return-std error** (A26 **0.004832**) and the **teacher-sigma RMSE** (A34 **0.08963**). **Diffusion-TS**
owns the two **lag-1 ACF** metrics (A23 **0.002410**, A24 **0.007895**) where its interpretable seasonal-trend
decoder is sharpest. **TimeVAE** keeps the single metric its posterior-mean vol reconstruction is built for —
the **teacher-sigma correlation** (A33 **0.02254**), though even this is a near-zero recovery far below the
0.6163 floor. **SBTS, TimeGAN, COSCI-GAN and TimeVQVAE** win no A-metric outright: each is a competent second
on a handful of axes but is edged everywhere by the four family leaders above.

---

## B — Curve-shape metrics cross-method comparison (mean ± std, 5 seeds)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. **MSE** combines three
lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der); **% err**
and **NRMSE** are **funct-only** (curve L only):

- **MSE**: dᵢ = mean((L_gen − L_real)²), averaged over curve / der / sec\_der — the winner-deciding number. Combined std = quadrature of the three seed-std.
- **% err** (function-level MAPE, funct-only): dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 — one division, on the curve L only. The der / sec\_der MAPE is excluded as ill-posed (near-zero denominators explode).
- **NRMSE** (funct-only): sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100 on the curve L only — RMSE normalised by the reference curve's range.

**% err and NRMSE are funct-only for every plot**: the first and second finite differences of these curves are
near-zero, so their relative error is ill-posed and would explode; only **MSE** averages all three sub-curves.
↓ lower is better. Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, so the reference
curve is fixed. The **Perfect** column is an independent Heston draw (seeds 1000+) scored against the test set
the same way — a **non-zero** finite-sample floor, not a degenerate zero. Winner is by MSE.

<table>
<thead>
  <tr>
    <th rowspan="2">Plot</th>
    <th rowspan="2">Measure</th>
    <th colspan="2">GAN</th>
    <th colspan="2">Diffusion</th>
    <th colspan="3">VAE</th>
    <th>Schrödinger Bridge</th>
    <th>Fourier Flow</th>
    <th rowspan="2">Perfect</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>TimeGAN</th>
    <th>COSCI-GAN</th>
    <th>Diffusion-TS</th>
    <th>CSDI</th>
    <th>TimeVAE</th>
    <th>TimeVQVAE</th>
    <th>LS4</th>
    <th>SBTS</th>
    <th>Fourier Flow</th>
  </tr>
</thead>
<tbody>
  <tr><td rowspan="3"><b>Log-return histogram</b></td><td>MSE</td><td>45.40 ± 57.91</td><td>42.66 ± 1.999</td><td>4.883 ± 0.5079</td><td>4.644 ± 0.4940</td><td>968.0 ± 183.1</td><td>4.386 ± 0.8335</td><td><b>0.4517 ± 0.02799</b></td><td>4.082 ± 0.04782</td><td>0.9211 ± 0.02370</td><td>0.1098 ± 0.02492</td><td rowspan="3"><b>LS4</b></td></tr>
  <tr><td>% err</td><td>33.41% ± 6.533%</td><td>246.6% ± 7.987%</td><td>42.14% ± 1.003%</td><td>35.27% ± 1.063%</td><td>114.9% ± 0.6458%</td><td>30.95% ± 1.747%</td><td>5.429% ± 0.1852%</td><td>39.17% ± 0.1361%</td><td>9.167% ± 0.5606%</td><td>1.799% ± 0.04483%</td></tr>
  <tr><td>NRMSE</td><td>21.38% ± 14.34%</td><td>30.81% ± 0.7154%</td><td>10.28% ± 0.5317%</td><td>9.998% ± 0.5467%</td><td>123.7% ± 6.783%</td><td>9.691% ± 0.9011%</td><td>2.779% ± 0.08180%</td><td>9.368% ± 0.06168%</td><td>4.186% ± 0.1102%</td><td>0.5328% ± 0.02035%</td></tr>
  <tr><td rowspan="3"><b>QQ plot</b></td><td>MSE</td><td>2.38e-06 ± 1.14e-06</td><td>8.25e-04 ± 6.60e-05</td><td>3.48e-06 ± 1.75e-07</td><td>2.36e-06 ± 1.57e-07</td><td>3.99e-05 ± 5.99e-07</td><td>1.82e-06 ± 2.20e-07</td><td><b>4.59e-08 ± 2.12e-09</b></td><td>3.01e-06 ± 2.28e-08</td><td>1.45e-07 ± 2.63e-08</td><td>1.09e-09 ± 6.13e-10</td><td rowspan="3"><b>LS4</b></td></tr>
  <tr><td>% err</td><td>34.50% ± 11.22%</td><td>437.1% ± 19.17%</td><td>25.71% ± 1.743%</td><td>24.22% ± 1.083%</td><td>90.53% ± 1.555%</td><td>23.84% ± 2.434%</td><td>6.022% ± 0.6435%</td><td>21.47% ± 0.3841%</td><td>9.342% ± 2.293%</td><td>0.4629% ± 0.1067%</td></tr>
  <tr><td>NRMSE</td><td>6.960% ± 1.738%</td><td>134.7% ± 5.407%</td><td>8.689% ± 0.2248%</td><td>7.188% ± 0.2370%</td><td>29.57% ± 0.2260%</td><td>6.308% ± 0.3785%</td><td>0.9701% ± 0.02323%</td><td>8.083% ± 0.03106%</td><td>1.687% ± 0.1351%</td><td>0.1206% ± 0.04670%</td></tr>
  <tr><td rowspan="3"><b>ACF \|r\| lags 1–20</b></td><td>MSE</td><td>0.003597 ± 0.003199</td><td>0.008548 ± 0.003519</td><td>1.72e-04 ± 4.79e-05</td><td><b>3.02e-05 ± 1.61e-05</b></td><td>0.03390 ± 0.01422</td><td>1.22e-04 ± 3.84e-05</td><td>5.14e-05 ± 1.08e-05</td><td>0.001512 ± 1.42e-05</td><td>3.83e-04 ± 1.20e-05</td><td>9.61e-06 ± 3.40e-06</td><td rowspan="3"><b>CSDI</b></td></tr>
  <tr><td>% err</td><td>186.2% ± 107.8%</td><td>230.0% ± 48.05%</td><td>73.33% ± 13.17%</td><td>19.26% ± 8.314%</td><td>983.6% ± 273.1%</td><td>63.03% ± 14.21%</td><td>37.09% ± 3.059%</td><td>149.0% ± 1.780%</td><td>117.2% ± 2.149%</td><td>8.724% ± 1.843%</td></tr>
  <tr><td>NRMSE</td><td>224.6% ± 123.4%</td><td>198.2% ± 35.47%</td><td>51.98% ± 7.840%</td><td>19.33% ± 5.196%</td><td>795.3% ± 212.4%</td><td>45.54% ± 9.362%</td><td>29.46% ± 2.604%</td><td>127.9% ± 0.8849%</td><td>88.45% ± 1.425%</td><td>6.071% ± 1.301%</td></tr>
  <tr><td rowspan="3"><b>ACF r² lags 1–20</b></td><td>MSE</td><td>0.001982 ± 0.001602</td><td>0.008781 ± 0.003516</td><td>1.32e-04 ± 4.43e-05</td><td>2.71e-05 ± 1.16e-05</td><td>0.02694 ± 0.01034</td><td>1.05e-04 ± 3.00e-05</td><td><b>2.48e-05 ± 6.52e-06</b></td><td>0.001723 ± 2.85e-05</td><td>2.80e-04 ± 1.13e-05</td><td>9.17e-06 ± 3.08e-06</td><td rowspan="3"><b>LS4</b></td></tr>
  <tr><td>% err</td><td>130.0% ± 65.84%</td><td>287.8% ± 57.85%</td><td>73.19% ± 16.72%</td><td>21.75% ± 10.67%</td><td>1026% ± 265.1%</td><td>70.37% ± 13.75%</td><td>24.39% ± 3.127%</td><td>171.3% ± 1.908%</td><td>120.8% ± 3.065%</td><td>11.34% ± 2.219%</td></tr>
  <tr><td>NRMSE</td><td>168.2% ± 70.21%</td><td>221.1% ± 36.09%</td><td>46.32% ± 8.702%</td><td>20.43% ± 5.060%</td><td>782.1% ± 188.7%</td><td>45.61% ± 7.936%</td><td>19.10% ± 2.524%</td><td>145.2% ± 1.200%</td><td>82.92% ± 1.680%</td><td>6.486% ± 1.351%</td></tr>
  <tr><td rowspan="3"><b>Rolling vol histogram</b></td><td>MSE</td><td>150.2 ± 75.22</td><td>1398 ± 34.29</td><td>220.2 ± 15.36</td><td>157.5 ± 12.45</td><td>16019 ± 2352</td><td>113.9 ± 13.91</td><td><b>8.514 ± 0.7580</b></td><td>412.9 ± 1.772</td><td>29.88 ± 2.639</td><td>1.372 ± 0.07269</td><td rowspan="3"><b>LS4</b></td></tr>
  <tr><td>% err</td><td>56.76% ± 21.18%</td><td>799.2% ± 14.12%</td><td>69.05% ± 1.441%</td><td>61.91% ± 2.364%</td><td>340.0% ± 11.74%</td><td>54.51% ± 2.433%</td><td>11.70% ± 1.165%</td><td>84.56% ± 0.1274%</td><td>25.42% ± 3.199%</td><td>2.264% ± 0.07625%</td></tr>
  <tr><td>NRMSE</td><td>22.64% ± 7.203%</td><td>73.06% ± 0.8956%</td><td>28.87% ± 0.9919%</td><td>24.39% ± 0.9523%</td><td>221.5% ± 13.05%</td><td>20.68% ± 1.268%</td><td>5.275% ± 0.3034%</td><td>39.59% ± 0.08241%</td><td>10.43% ± 0.4823%</td><td>0.8688% ± 0.05532%</td></tr>
  <tr><td rowspan="3"><b>Tail survival</b></td><td>MSE</td><td>0.003912 ± 0.003064</td><td>0.05973 ± 0.001991</td><td>0.002258 ± 2.00e-04</td><td>0.001960 ± 1.85e-04</td><td>0.07224 ± 0.001903</td><td>0.001709 ± 2.78e-04</td><td><b>6.90e-05 ± 8.10e-06</b></td><td>0.001937 ± 2.20e-05</td><td>1.71e-04 ± 1.49e-05</td><td>5.22e-07 ± 5.50e-07</td><td rowspan="3"><b>LS4</b></td></tr>
  <tr><td>% err</td><td>23.64% ± 6.097%</td><td>342.3% ± 8.331%</td><td>28.39% ± 0.8411%</td><td>24.78% ± 0.8772%</td><td>90.06% ± 0.6385%</td><td>22.34% ± 1.374%</td><td>3.345% ± 0.1144%</td><td>26.62% ± 0.1128%</td><td>5.711% ± 0.2437%</td><td>0.3302% ± 0.2167%</td></tr>
  <tr><td>NRMSE</td><td>10.02% ± 4.365%</td><td>42.74% ± 0.7148%</td><td>8.301% ± 0.3648%</td><td>7.733% ± 0.3598%</td><td>46.97% ± 0.6196%</td><td>7.206% ± 0.5711%</td><td>1.449% ± 0.08321%</td><td>7.694% ± 0.04378%</td><td>2.287% ± 0.09795%</td><td>0.1050% ± 0.06651%</td></tr>
</tbody>
</table>

**LS4 wins B: 5/6 plots on MSE (log-return histogram, QQ, ACF r², rolling-vol, tail survival); CSDI keeps
ACF \|r\|.** LS4's latent-S4 decoder fits the marginal-shape diagnostics tighter than any method — its
log-return histogram MSE (0.4517) is 2× tighter than Fourier Flow (0.9211, second best) and its QQ MSE
(4.59e-08) is 3× tighter (vs FF 1.45e-07), leading on all three measures. **CSDI** wins the **ACF \|r\|**
curve (MSE 3.02e-05 vs LS4's 5.14e-05) — its score-based diffusion reproduces Heston's weak vol-clustering
autocorrelation that LS4's smooth prior slightly over-damps, the same axis on which CSDI keeps A21 in the
A-table; on **ACF r²** LS4 edges back ahead (2.48e-05 vs CSDI 2.71e-05). **Fourier Flow** is the clear
second on the four marginal-shape curves. **TimeVAE loses all six B plots** by one-to-three orders of
magnitude — its posterior-mean decoder collapses the marginal shape (log-return histogram MSE 968 vs LS4
0.45, rolling-vol MSE 16019 vs 8.51), consistent with its heavily under-dispersed samples. **TimeVQVAE** and
**COSCI-GAN** win no B plot; COSCI-GAN ranks near the bottom of every one (worst QQ MSE at 8.25e-04), its
near-Gaussian marginal matching the low-order *scalar* moments (A5) but not the full-density *curves*. No
method reaches the non-zero Perfect floor on any curve, but LS4 is within ~4× of it on the log-return
histogram (0.4517 vs 0.1098). Each value is computed over the same **5 seeds** per method.

---

## PS-MC — Path-Shadowing Monte-Carlo forecast (CRPS)

Path Shadowing Monte-Carlo (Morel–Bouchaud 2023) forecasts the future of a partial path by finding its
nearest neighbours ("shadows") in the generated set and averaging their continuations. We score the forecast
with the **CRPS** of the predicted terminal-price distribution at horizons **H=32** and **H=64** days,
averaged over held-out **test**-set query paths (↓ lower is better). The **RW baseline** is a Gaussian random
walk calibrated to the test set's log-return volatility — a method whose CRPS beats it carries genuine
forecast information beyond the marginal variance.

<table>
<thead>
  <tr>
    <th rowspan="2">Metric</th>
    <th colspan="2">GAN</th>
    <th colspan="2">Diffusion</th>
    <th colspan="3">VAE</th>
    <th>Schrödinger Bridge</th>
    <th>Fourier Flow</th>
    <th rowspan="2">RW baseline</th>
    <th rowspan="2">Perfect</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>TimeGAN</th>
    <th>COSCI-GAN</th>
    <th>Diffusion-TS</th>
    <th>CSDI</th>
    <th>TimeVAE</th>
    <th>TimeVQVAE</th>
    <th>LS4</th>
    <th>SBTS</th>
    <th>Fourier Flow</th>
  </tr>
</thead>
<tbody>
  <tr><td>PS-MC CRPS H=32 ↓</td><td>3.085 ± 0.3332</td><td>4.657 ± 0.7720</td><td>2.717 ± 0.002200</td><td>2.718 ± 0.003646</td><td>3.912 ± 0.07154</td><td>2.779 ± 0.01655</td><td><b>2.704 ± 0.002510</b></td><td>2.759 ± 0.006411</td><td>2.744 ± 0.03009</td><td>3.738</td><td>2.721 ± 0.004183</td><td><b>LS4</b></td></tr>
  <tr><td>PS-MC CRPS H=64 ↓</td><td>4.337 ± 0.4329</td><td>5.789 ± 0.7528</td><td>3.804 ± 0.007848</td><td>3.776 ± 0.005153</td><td>5.670 ± 0.1222</td><td>3.851 ± 0.02210</td><td><b>3.763 ± 0.005851</b></td><td>3.859 ± 0.01236</td><td>3.961 ± 0.1098</td><td>5.246</td><td>3.788 ± 0.006463</td><td><b>LS4</b></td></tr>
</tbody>
</table>

**LS4 wins both horizons** (CRPS 2.704 at H=32, 3.763 at H=64) — its shadows carry the sharpest forecast.
**CSDI** and **Diffusion-TS** follow within ~0.5% at H=32 (2.718 / 2.717), and CSDI is second at H=64
(3.776). Every method except **COSCI-GAN** (4.657 / 5.789) and **TimeVAE** (3.912 / 5.670) beats the RW
baseline (3.738 / 5.246) at both horizons, so the generated paths carry real predictive structure beyond
the marginal variance; the two exceptions overshoot the random walk because their samples are over-dispersed
(COSCI-GAN) or collapsed (TimeVAE).

---

## Stylised curves

The 8-panel diagnostic below overlays each method's generated paths (blue) against the held-out **test set**
(orange) on the eight stylised facts the B-metrics quantify: price fan, log-return histogram, QQ plot, ACF
of |r| and r², rolling-volatility histogram, tail-survival and mean-path. One panel figure per method,
ordered by family.

### GAN

#### TimeGAN
![TimeGAN diagnostics](Heston/TimeGAN/plots/heston_diagnostics.png)

#### COSCI-GAN
![COSCI-GAN diagnostics](Heston/COSCI-GAN/plots/heston_diagnostics.png)

---

### Diffusion

#### Diffusion-TS
![Diffusion-TS diagnostics](Heston/DiffusionTS/plots/heston_diagnostics.png)

#### CSDI
![CSDI diagnostics](Heston/CSDI/plots/heston_diagnostics.png)

---

### VAE

#### TimeVAE
![TimeVAE diagnostics](Heston/TimeVAE/plots/heston_diagnostics.png)

#### TimeVQVAE
![TimeVQVAE diagnostics](Heston/TimeVQVAE/plots/heston_diagnostics.png)

#### LS4
![LS4 diagnostics](Heston/LS4/plots/heston_diagnostics.png)

---

### Schrödinger Bridge

#### SBTS
![SBTS diagnostics](Heston/SBTS/plots/heston_diagnostics.png)

---

### Fourier Flow

#### Fourier Flow
![Fourier Flow diagnostics](Heston/FourierFlow/plots/heston_diagnostics.png)

---

## Detailed per-method results

| Method | Results folder | Method folder |
|--------|---------------|---------------|
| TimeGAN | [`Heston/TimeGAN/`](Heston/TimeGAN/) | [`../methods/TimeGAN/`](../methods/TimeGAN/) |
| SBTS | [`Heston/SBTS/`](Heston/SBTS/) | [`../methods/SBTS/`](../methods/SBTS/) |
| Fourier Flow | [`Heston/FourierFlow/`](Heston/FourierFlow/) | [`../methods/FourierFlow/`](../methods/FourierFlow/) |
| Diffusion-TS | [`Heston/DiffusionTS/`](Heston/DiffusionTS/) | [`../methods/DiffusionTS/`](../methods/DiffusionTS/) |
| CSDI | [`Heston/CSDI/`](Heston/CSDI/) | [`../methods/CSDI/`](../methods/CSDI/) |
| TimeVAE | [`Heston/TimeVAE/`](Heston/TimeVAE/) | [`../methods/TimeVAE/`](../methods/TimeVAE/) |
| TimeVQVAE | [`Heston/TimeVQVAE/`](Heston/TimeVQVAE/) | [`../methods/TimeVQVAE/`](../methods/TimeVQVAE/) |
| COSCI-GAN | [`Heston/COSCI-GAN/`](Heston/COSCI-GAN/) | [`../methods/COSCI-GAN/`](../methods/COSCI-GAN/) |
| LS4 | [`Heston/LS4/`](Heston/LS4/) | [`../methods/LS4/`](../methods/LS4/) |
| Perfect recovery (floor) | — | [`../methods/perfect_recovery/`](../methods/perfect_recovery/) |

---

## Methods

### TimeGAN — Time-series Generative Adversarial Network
**Paper:** Yoon, Jarrett, van der Schaar — *Time-series GAN* — NeurIPS 2019, [arXiv:2010.00782](https://arxiv.org/abs/2010.00782)
**Code:** [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) — PyTorch reimplementation in this repo

TimeGAN is a **neural GAN** with five interacting GRU components:
- **Embedder + Recovery** (3-layer GRU, hidden=24): maps price paths ↔ latent embedding space
- **Generator** (3-layer GRU): generates latent sequences from Gaussian noise
- **Supervisor** (2-layer GRU): enforces step-by-step temporal consistency in latent space
- **Discriminator** (3-layer GRU): distinguishes real from generated latent sequences

**Training**: 3-phase adversarial, 20 000 steps (5 k embed → 5 k supervisor → 10 k joint).
**Hardware**: GPU required (A100 80 GB). ~6–8 min/seed.
**Generation**: Milliseconds (GRU forward pass). Sequences start near S₀=100 via internal min-max denorm.
On Heston, TimeGAN wins **no** A-metric or B-plot outright — it is a competent mid-pack generator whose
Path-Shadowing CRPS (3.085/4.337) clears the random-walk baseline at both horizons.

### SBTS — Schrödinger Bridge Time Series
**Paper:** Alouadi, Barreau, Carlier, Pham — *Robust Time Series Generation via Schrödinger Bridge* — ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)
**Code:** [alexouadi/SBTS](https://github.com/alexouadi/SBTS) — Numba-accelerated reimplementation in this repo

SBTS is a **non-parametric kernel method** with no neural network and no training:
- Estimates the Schrödinger-bridge drift from training data using a compact quartic kernel K_h
- Simulates paths via Euler-Maruyama with the estimated drift (N_pi=200 substeps per interval)
- Markovian order K=1: weight of path m depends only on its most recent state X_i^m (valid for Heston)
- Internally operates on **scaled log-returns** R̃ = R × √Δt / σ(R) — not on prices or log-prices — then reconstructs prices: S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t])

**Generation**: No training phase. ~6.3 min/seed with 64 CPU workers.
**Hardware**: CPU-only (Numba JIT). GPUs only used for A18/A19 metric evaluation.
On Heston, SBTS wins **no** A-metric outright; it is the most deterministic method (tightest cross-seed std)
and its Path-Shadowing CRPS (2.759/3.859) clears the random-walk baseline at both horizons.

### Fourier Flow — Generative Time-series Modeling with Fourier Flows
**Paper:** Alaa, Chan, van der Schaar — *Generative Time-series Modeling with Fourier Flows* — ICLR 2021, [OpenReview](https://openreview.net/forum?id=PpshD0AXfA)
**Code:** [ahmedmalaa/Fourier-flows](https://github.com/ahmedmalaa/Fourier-flows) — released-code-as-is reimplementation in this repo

Fourier Flow is an **explicit-likelihood normalizing flow that operates in the frequency domain**:
- Applies a **Discrete Fourier Transform** to each path, then a chain of invertible spectral filters (3 flows)
- Each **SpectralFilter** is an MLP (hidden=200) coupling layer acting on the real/imaginary spectral bins
- Trained by **direct negative-log-likelihood** minimisation (loss `(−log_pz − log_jacob).mean()`), full-batch Adam + ExponentialLR (γ=0.999), **1000 epochs**, **CPU-only** (numpy.fft)
- Inverts the flow and applies the **inverse DFT** to sample new price paths

**Two numerical guards** make training finite on Heston (paths start at a deterministic S₀=100, so the spectral covariance is near-singular at the DC bin): a **zero-std spectral-bin clamp** (necessary but not sufficient) and a **gradient clip = 1.0** (the actual fix that removes the NaN blow-up). See [`Heston/FourierFlow/README.md`](Heston/FourierFlow/README.md).

On Heston, Fourier Flow wins **four** A-metrics — skewness (A15 0.02288), the MLP discriminative score
(A18-MLP 0.005951), the return-std error (A26 0.004832) and the teacher-sigma RMSE (A34 0.08963) — and is
the clear second to LS4 on every marginal-shape B-plot.

**Training**: ~8.2 min/seed (490 s, CPU). **Generation**: ~1.5 s/seed. **Hardware**: CPU-only; GPUs only used for A18/A19 metric evaluation.

### Diffusion-TS — Interpretable Diffusion for General Time Series Generation
**Paper:** Yuan, Qiao — *Diffusion-TS: Interpretable Diffusion for General Time Series Generation* — ICLR 2024, [arXiv:2403.01742](https://arxiv.org/abs/2403.01742)
**Code:** [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) — released-code-as-is reimplementation in this repo

Diffusion-TS is a **non-autoregressive denoising diffusion model (DDPM)** with an interpretable
encoder-decoder transformer:
- Generates a whole length-T series in one reverse-diffusion trajectory (no step-by-step roll-out)
- **Predicts the clean signal x̂₀ directly** at each diffusion step (not the added noise ε), making the target a reconstruction of the series
- The decoder reconstructs x̂₀ as an explicit sum of a polynomial **trend** block and Fourier-based **seasonal** blocks (disentangled seasonal-trend decomposition)
- Trained by a **reweighted L1 + Fourier-FFT reconstruction loss** with a **cosine β** schedule over **500** diffusion steps; EMA weights (decay 0.995) used for sampling
- Uses the `mujoco` preset (n_layer_enc = n_layer_dec = 3, d_model = 64, 544 147 params, 12 000 steps) — chosen by an identical 3 000-step smoke test that scored `mujoco` Context-FID 0.7367 vs `etth` 2.3192 vs `stocks` 36.05 (lower is better). See [`../methods/DiffusionTS/code/README.md`](../methods/DiffusionTS/code/README.md).

On Heston, Diffusion-TS wins the two **lag-1 ACF** metrics (A23 0.002410, A24 0.007895) where its
seasonal-trend decoder is sharpest, and its Path-Shadowing CRPS (2.717/3.804) is among the tightest, second
only to LS4/CSDI at H=32.

**Training**: ~14.6 min/seed (878 s, A100 GPU). **Generation**: 500-step DDPM sampling with EMA weights (not separately timed). **Hardware**: GPU required (A100 80 GB); GPUs also used for A18/A19 metric evaluation.

### CSDI — Conditional Score-based Diffusion Models for Imputation
**Paper:** Tashiro, Song, Song, Ermon — *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* — NeurIPS 2021, [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)
**Code:** [ermongroup/CSDI](https://github.com/ermongroup/CSDI) — released-code-as-is reimplementation in this repo

CSDI is a **score-based denoising diffusion model (DDPM)** whose denoiser is a **2-D
(time × feature) transformer** with residual layers:
- For unconditional Heston generation we set `is_unconditional = 1` and the conditioning mask ≡ 0, so the
  model reduces to a **standard DDPM** that denoises the whole length-128 series in one reverse trajectory
- **Predicts the added noise ε** at each diffusion step (ε-matching), target = the injected Gaussian noise
- The denoiser stacks 4 residual blocks (64 channels, 8 attention heads) with a temporal transformer and a
  feature transformer, plus a 128-d diffusion-step embedding and a 16-d feature embedding
- Trained on **z-scored prices** (mean 101.33, std 9.97) by **noise-prediction MSE**
  E_t‖ε − ε_θ(x_t, t)‖² with a **quadratic β** schedule over **50** diffusion steps (β 1e-4 → 0.5);
  Adam lr 1e-3, weight-decay 1e-6, batch 16, 200 epochs, MultiStepLR (×0.1 at 75%/90% of training)
- ~413 k parameters. See [`../methods/CSDI/code/README.md`](../methods/CSDI/code/README.md).

On Heston, CSDI wins the three metrics that reward its faithful vol-clustering autocorrelation and heaviest
tails: the **kurtosis error** (A1 0.09543, best of any method), the **ACF |r| lag-average** (A21 0.01126)
and the **kurtosis ratio** (A28 0.8706, the only method within 0.13 of 1.0). It also keeps the **ACF |r|**
B-plot (MSE 3.02e-05) and its Path-Shadowing CRPS (2.718/3.776) is second at H=64.

**Training**: ~29.3 min/seed (1 756 s, A100 GPU). **Generation**: ~10.2 s/seed (50-step DDPM). **Hardware**: GPU required (A100 80 GB); GPUs also used for A18/A19 metric evaluation.

### TimeVAE — Variational Auto-Encoder for Multivariate Time Series
**Paper:** Desai, Freeman, Beaver, Wang — *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation* — 2021, [arXiv:2111.08095](https://arxiv.org/abs/2111.08095)
**Code:** [abudesai/timeVAE](https://github.com/abudesai/timeVAE) — PyTorch reimplementation in this repo (the official code is TensorFlow/Keras, which has no working GPU build for this machine's CUDA driver)

TimeVAE is a **variational auto-encoder** with a convolutional encoder and a decoder that reconstructs the
whole length-T series in one forward pass:
- **Encoder**: stacked 1-D convolutions (hidden channels 50 → 100 → 200) → flatten → Linear to a **latent
  dimension of 8** (posterior mean + log-var), reparameterised sample z ~ N(μ, σ²)
- **Decoder** (TimeVAE-**Base**): Linear + transposed convolutions map z back to the length-128 series; the
  optional interpretable **trend** (`trend_poly=0`) and **seasonal** (`custom_seas=None`) blocks are disabled,
  so this is the pure convolutional base with a residual connection (`use_residual_conn=True`)
- Trained by the **ELBO**: `reconstruction_wt · (SSE + feature-mean SSE) + KL`, with `reconstruction_wt = 3.0`;
  Adam lr 1e-3, batch 16, EarlyStopping (5 seeds stop between 230–340 epochs)
- ~247 k parameters (feat_dim 1, seq_len 128). See [`../methods/TimeVAE/code/README.md`](../methods/TimeVAE/code/README.md).

Because the decoder regresses toward the **posterior mean**, TimeVAE keeps the single metric its
posterior-mean vol reconstruction is built for — the **teacher-sigma correlation** (A33 0.02254, highest of
the pool, though still a near-zero recovery far below the 0.6163 floor) — but produces heavily
**under-dispersed** marginals (worst-of-pool on fat-tail A1, vol MMD A9, rolling-vol KS A31; loses **all six**
B curve-shape plots). Its Path-Shadowing CRPS (3.912/5.670) does **not** beat the random-walk baseline.

**Training**: ~13 min/seed (A100 GPU). **Generation**: <1 s/seed (single decoder forward pass). **Hardware**: GPU used for training and A18/A19 metric evaluation.

### TimeVQVAE — Vector Quantized Time Series Generation
**Paper:** Lee, Malacarne, Aune — *Vector Quantized Time Series Generation with a Bidirectional Prior Model* — AISTATS 2023, [arXiv:2303.04743](https://arxiv.org/abs/2303.04743)
**Code:** [ML4ITS/TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) — reference code (commit `b9650e9d`, PyTorch + PyTorch-Lightning) run as-is behind a thin data-plumbing wrapper in this repo

TimeVQVAE is a **two-stage vector-quantized generative model** that operates in the STFT
time-frequency domain:
- **Stage 1 — VQ tokenization**: an STFT (`n_fft=8`) splits each path into a low-frequency (LF, bin 0)
  and high-frequency (HF, bins 1:) branch; each branch has its own ResNet encoder/decoder (dim 64, 4
  blocks) and a **codebook of 32 codes** (dim 64, EMA decay 0.8) that discretises the latent into tokens
- **Stage 2 — MaskGIT bidirectional prior**: a masked bidirectional transformer (hidden 256, 4 layers,
  2 heads, RMSNorm, `p_uncond = 0.2`) learns the token prior; the HF token stream is **conditioned on
  the LF tokens**
- **Generation** via `unconditional_sample` — iterative MaskGIT decoding (T=10 steps, choice temperature
  4, guidance 1.0) fills the token grid, then the Stage-1 decoders + inverse STFT map tokens back to a
  price path
- Trained on **globally z-normalized prices** (paper `data_scaling=True`, mean 101.33, std 9.97),
  inverted to price scale before saving. Epoch budget stage1 = 250 / stage2 = 1000 (matched to the
  paper's gradient-step count on the 16×-larger Heston set). See
  [`../methods/TimeVQVAE/code/README.md`](../methods/TimeVQVAE/code/README.md).

On the test set, TimeVQVAE wins **no** A-metric or B-plot outright — its previous structural-error wins
(A20 covariance, A32 vol-of-vol) both fall to LS4 (A20 13.63, A32 3.21e-04). It remains a solid mid-pack
generator — third-best on several fat-tail/MMD/curve-shape diagnostics — and its Path-Shadowing CRPS
(2.779/3.851) clears the random-walk baseline at both horizons.

**Training**: ~53 min/seed (A100 GPU, two stages). **Generation**: ~6 s/seed (MaskGIT decode + iSTFT). **Hardware**: GPU required (A100 80 GB); GPUs also used for A18/A19 metric evaluation.

### COSCI-GAN — COmmon Source CoordInated GAN
**Paper:** Seyfi, Rajotte, Ng — *Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)* — NeurIPS 2022, [arXiv:2210.07248](https://arxiv.org/abs/2210.07248)
**Code:** [aliseyfi75/COSCI-GAN](https://github.com/aliseyfi75/COSCI-GAN) — PyTorch reimplementation in this repo

COSCI-GAN is a **channel-decomposed GAN** designed for *multivariate* series: one univariate
"Channel GAN" per feature, all sharing the **same noise vector z**, plus a **Central Discriminator (CD)**
that couples the channels to preserve cross-channel dependence:
- **Channel GAN** (×C): an LSTM generator (z → LSTM 32→256 → Linear→128) and an LSTM discriminator
  (hidden 256, 1 layer, sigmoid), one per channel
- **Central Discriminator**: an MLP (128→256→128→64→1, LeakyReLU 0.1 + Dropout 0.3) that sees all
  channels jointly; three-player minimax `loss_G_i = BCE(D_i, 1) − γ·loss_CD`, γ=5
- Adam betas (0.5, 0.9), BCE, 120 epochs, ~800 k parameters

**Heston is univariate (C = 1)**, so COSCI-GAN runs with a **single channel** and the CD becomes
degenerate: it receives the same 128-dim vector as the single channel discriminator, so `loss_CD ≈ ln2 ≈
0.693` at equilibrium and the paper's native cross-channel correlation-matrix metric is structurally
undefined (the correlation matrix is a scalar 1, MAE ≡ 0 for any generator). We reproduced the paper's
EEG eye-state Table-4 correlation-MAE separately for validation (ours 0.1085 ± 0.0066 vs paper COSCI-GAN
0.111 ± 0.005). On the test set, COSCI-GAN wins **no** A-metric or B-plot: it has good scalar low-order
moments (A1 0.561, A15 0.050) but weak full-density curves (near the bottom of every B plot, **dead-last on
QQ**), a **saturated A18 discriminative score** (0.500 / 0.4999 — near-perfectly separable), thin/near-Gaussian
tails with a negative sign-flipping kurtosis ratio (A28 −8.150) and a Path-Shadowing CRPS (4.657/5.789) that
does **not** beat the random-walk baseline. See [`Heston/COSCI-GAN/README.md`](Heston/COSCI-GAN/README.md).

**Training**: ~4.3 min/seed (257 s, A100 GPU). **Generation**: LSTM forward pass over shared noise (not separately timed). **Hardware**: GPU used for training and A18/A19 metric evaluation.

### LS4 — Deep Latent State-Space Model
**Paper:** Zhou, Poli, Xu, Massaroli, Ermon — *Deep Latent State Space Models for Time-Series Generation* — ICML 2023, [arXiv:2212.12749](https://arxiv.org/abs/2212.12749)
**Code:** [alexzhou907/ls4](https://github.com/alexzhou907/ls4) — official code, run verbatim (one required fix, below)

LS4 is a **VAE-style latent state-space model**: a continuous latent `z` evolves under a **structured
S4 (Latent-S4) prior**, with an S4 posterior and an S4 decoder. It is trained on the **ELBO**
(`total = kld_loss + nll_loss`, `mse_loss` is a diagnostic); the `autoreg` backbone rolls the prior
forward. On Heston it uses `z_dim = 8`, `d_model = 128`, `d_state = 64`, 4 S4 blocks per module
(≈ **2.15 M parameters**), with **global** standardisation `(X − μ) / σ` (μ ≈ 101.325, σ ≈ 9.972).
Generation uses the STEP-mode `latent.step` recurrence (one timestep at a time).

**Required fix — the Cauchy sum.** LS4's generation rolls the S4 prior with `latent.step`
(**STEP-mode recurrence**). On a CUDA-13 A100 the fast Cauchy kernels (`pykeops` / the bundled CUDA
extension) are unavailable, so S4 falls back to the **naive Python Cauchy kernel**, which as-shipped
sums over the *full* pole set instead of over **conjugate pole pairs** — correct for the keops/CUDA path
but wrong for the naive path used at generation time. Without the fix the generator degenerates
(the paper's Solar-Weekly marginal score plateaus at 0.197 vs 0.046). The one-line patch
(`code/reference/models/s4.py:795`, conjugate-pair sum) restores the paper regime and is carried into
the Heston generator. No other reference code was modified.

On Heston, LS4 **dominates the benchmark — 26 of 36 A-metrics** — sweeping the tail quantiles (A2–A4) and
Hill index (A5), the entire distributional family (A6–A14, A16, A17), **both** adversarial-GRU and predictive
scores (A18-GRU 0.005890, A19-GRU 0.05001 / A19-MLP 0.05006), most of the temporal/vol family (A20, A22, A25,
A27, A29–A32), **five of six** B curve-shape plots and **both** Path-Shadowing horizons (CRPS 2.704/3.763).
Its two structural gaps are the **thin tails** (A28 kurtosis ratio 1.565 — mildly platykurtic; A1 mid-pack)
and the **teacher-sigma correlation** (A33 −3.9e-04 — a single-factor latent cannot recover Heston's
two-factor stochastic vol). See [`Heston/LS4/README.md`](Heston/LS4/README.md).

**Training**: ~16 min/seed (973 s, A100 GPU, 100 epochs). **Generation**: ~9 s/seed (STEP-mode `latent.step`, A100 GPU). **Hardware**: GPU used for training, generation and A18/A19 metric evaluation.

### Perfect recovery — reproducible floor
An **independent Heston simulation** (a fresh 8 192-path draw with seeds 1000+i, one per benchmark seed)
scored against the **held-out test set** exactly as every method is scored. Because it is a genuine — but
*independent* — Heston sample, it does **not** hit 0 on any metric: the residual is pure finite-sample noise
(e.g. A33 σ-corr floor 0.6163, log-return histogram MSE 0.1098, QQ MSE 1.09e-09). This is the single source
of truth for every "Perfect floor" column in the repo — see
[`../methods/perfect_recovery/`](../methods/perfect_recovery/).

---

## Key differences

<table>
<thead>
  <tr>
    <th rowspan="2">Aspect</th>
    <th colspan="2">GAN</th>
    <th colspan="2">Diffusion</th>
    <th colspan="3">VAE</th>
    <th>Schrödinger Bridge</th>
    <th>Fourier Flow</th>
  </tr>
  <tr>
    <th>TimeGAN</th>
    <th>COSCI-GAN</th>
    <th>Diffusion-TS</th>
    <th>CSDI</th>
    <th>TimeVAE</th>
    <th>TimeVQVAE</th>
    <th>LS4</th>
    <th>SBTS</th>
    <th>Fourier Flow</th>
  </tr>
</thead>
<tbody>
  <tr><td>**Type**</td><td>Neural GAN (5 GRU components)</td><td>Channel-decomposed GAN (per-channel LSTM GANs + MLP central discriminator)</td><td>Denoising diffusion (DDPM) + seasonal-trend transformer</td><td>Score-based diffusion (DDPM) + time×feature transformer</td><td>Variational auto-encoder (conv encoder + decoder, Base)</td><td>Two-stage vector-quantized (STFT VQ-VAE + MaskGIT prior)</td><td>VAE-style latent state-space model (S4 prior + S4 posterior + S4 decoder)</td><td>Non-parametric kernel estimator</td><td>Explicit-likelihood normalizing flow (frequency domain)</td></tr>
  <tr><td>**Learnable parameters**</td><td>~120 k (GRU weights)</td><td>~800 k (LSTM channel gen/disc + MLP central disc)</td><td>~544 k (enc/dec transformer, mujoco)</td><td>~413 k (2-D transformer, 4 residual layers)</td><td>~247 k (conv encoder/decoder, latent 8)</td><td>LF+HF codebooks (32×64) + MaskGIT transformer (hidden 256, 4 layers)</td><td>~2.15 M (Latent-S4 prior/posterior/decoder, d_model 128, d_state 64)</td><td>**0** (no parameters)</td><td>~360 k (3 spectral-filter MLPs, hidden=200)</td></tr>
  <tr><td>**Training time / seed**</td><td>~6–8 min (A100 GPU)</td><td>~4.3 min (A100 GPU, 120 epochs)</td><td>~14.6 min (A100 GPU, 12 000 steps)</td><td>~29.3 min (A100 GPU, 200 epochs)</td><td>~13 min (A100 GPU, EarlyStop 230–340 epochs)</td><td>~53 min (A100 GPU, stage1 250 + stage2 1000 epochs)</td><td>~16 min (A100 GPU, 100 epochs)</td><td>No training</td><td>~8.2 min (CPU, 1000 epochs)</td></tr>
  <tr><td>**Generation time / seed**</td><td><1 s (GPU inference)</td><td>LSTM forward over shared noise (not sep. timed, GPU)</td><td>500-step DDPM sampling (GPU)</td><td>~10.2 s (50-step DDPM, GPU)</td><td><1 s (single decoder forward pass)</td><td>~6 s (MaskGIT decode + iSTFT, GPU)</td><td>~9 s (STEP-mode `latent.step`, GPU)</td><td>~6.3 min (64 CPU workers)</td><td>~1.5 s (CPU inverse flow + iDFT)</td></tr>
  <tr><td>**Temporal memory**</td><td>Full (GRU sees all past steps)</td><td>Full (LSTM sees all past steps)</td><td>Global (transformer self-attention over full window)</td><td>Global (2-D transformer over time × feature)</td><td>Global (conv receptive field over full window)</td><td>Global (bidirectional MaskGIT transformer over token grid)</td><td>Global (S4 structured state-space over full window)</td><td>**Markov-1 only**</td><td>Global (per-frequency spectral coupling)</td></tr>
  <tr><td>**Internal representation**</td><td>Latent embeddings (min-max)</td><td>Per-channel LSTM hidden state (shared noise z)</td><td>x̂₀ = trend + seasonal (time + Fourier domain)</td><td>z-scored prices + diffusion noise</td><td>8-d Gaussian latent z</td><td>STFT VQ tokens (LF + HF codebooks)</td><td>Global-standardized prices + latent S4 state z</td><td>Scaled log-returns R̃</td><td>DFT spectral bins (real/imag)</td></tr>
  <tr><td>**Final output**</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td><td>Price paths (S_t)</td></tr>
  <tr><td>**Cross-seed stability**</td><td>Moderate (GAN variance)</td><td>Moderate (GAN variance); wide A5/A28 spread (sign-flipping kurtosis ratio)</td><td>High on moments/ACF, moderate on GRU disc</td><td>High on moments/ACF, moderate on GRU disc</td><td>High on moments, moderate on mean-path (A13/A25 std ~0.2–0.3)</td><td>High on PS-MC (std 0.017), moderate on covariance (A20 std 6.66)</td><td>High on distribution/tail metrics (A2 std 1.1e-04, PS-MC std 0.003–0.006), moderate on mean-path (A13/A25)</td><td>**High** (deterministic kernel)</td><td>High on moments, moderate on covariance</td></tr>
  <tr><td>**Scales to long T**</td><td>Well (RNN)</td><td>Well (LSTM); central disc degenerate at C=1 (univariate)</td><td>Well (transformer handles any T)</td><td>Well (transformer handles any T)</td><td>Well (fixed conv/latent size)</td><td>Well (transformer + more STFT tokens)</td><td>Well (S4 SSM designed for long sequences)</td><td>Degrades (K=1 insufficient)</td><td>Well (fixed spectral size)</td></tr>
  <tr><td>**Hyperparameter sensitivity**</td><td>Many (arch, lr, steps)</td><td>Moderate (γ central-disc weight, lr, epochs, LSTM hidden)</td><td>Moderate (depth preset, timesteps, EMA)</td><td>Moderate (layers, channels, diffusion steps, β schedule)</td><td>Few (latent dim, reconstruction_wt, hidden sizes)</td><td>Moderate (n_fft, codebook size, MaskGIT steps/temperature)</td><td>Moderate (z_dim, d_model, d_state, S4 blocks; Cauchy-sum fix required)</td><td>One critical: h (bandwidth)</td><td>Few (n_flows, hidden, grad-clip guard)</td></tr>
  <tr><td>**Training objective**</td><td>Adversarial + supervised</td><td>Three-player adversarial (channel BCE − γ·central-disc BCE)</td><td>Reweighted L1 + Fourier-FFT reconstruction</td><td>Noise-prediction MSE (ε-matching)</td><td>ELBO (weighted reconstruction + KL)</td><td>Stage-1 VQ reconstruction + Stage-2 masked-token cross-entropy</td><td>ELBO (KL + reconstruction NLL)</td><td>Schrödinger-bridge drift (closed-form)</td><td>**Exact negative log-likelihood**</td></tr>
</tbody>
</table>
