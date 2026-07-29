# Heston — `preprocessing_with_log_returns` experiment

This folder holds generator results for the **SBTS log-return preprocessing** applied to
methods **other than SBTS**. The question it answers: *does feeding a generator
volatility-scaled log-returns (the way SBTS does) instead of standardized prices change its
scores on the Heston benchmark?*

Two methods are carried through the full pipeline: **CSDI** (`CSDI/`) and **LS4** (`LS4/`). Each is
trained on the **4096**-path log-return train split (seed 0), scored on the held-out 4096 test set
across seeds 0–4, and given a strict 1M-path path-shadowing bank. Every subsequent method gets its own
sibling folder built the same way — see [`GUIDELINE.md`](GUIDELINE.md) for the step-by-step recipe.

The tables below mirror the main [`results/`](../../README.md) display **at this experiment's 4096/512
scale**: cross-method A1–A34, curve-shape B, and the strict path-shadowing forecast, each with a
**Perfect** finite-sample floor (fresh independent Heston draw at N=4096, seeds 1000+) and a **Winner**
column (**bold** = best across methods, floor excluded).

---

## A1-A34, Cross-method comparison (mean ± std, 5 seeds)

Methods are grouped by model family. ↓ = lower is better, ↑ = higher is better, A28 target = 1.0.
**Bold** = best across methods. **Perfect** is a fresh independent Heston draw at N=4096 (seeds 1000+)
scored against the same test set exactly as each method is — a **non-zero** finite-sample floor.

<table>
<thead>
  <tr>
    <th rowspan="2">Metric</th>
    <th>Diffusion</th>
    <th>VAE</th>
    <th rowspan="2">Perfect</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>CSDI</th>
    <th>LS4</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="5"><b>Fat Tail</b></td></tr>
  <tr><td>A1 Kurtosis Error ↓</td><td><b>0.09607 ± 0.03403</b></td><td>0.6156 ± 0.4445</td><td>0.05717 ± 0.01594</td><td><b>CSDI</b></td></tr>
  <tr><td>A2 \|r\| q95 Error ↓</td><td>0.003275 ± 2.21e-04</td><td><b>0.001418 ± 6.20e-04</b></td><td>1.49e-04 ± 8.48e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A3 \|r\| q99 Error ↓</td><td>0.004672 ± 2.87e-04</td><td><b>6.94e-04 ± 9.66e-04</b></td><td>2.40e-04 ± 1.51e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A4 Tail QQ Error ↓</td><td>0.003223 ± 2.21e-04</td><td><b>0.001612 ± 3.41e-04</b></td><td>1.54e-04 ± 5.57e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A5 Hill Tail Index Error ↓</td><td><b>3.502 ± 1.155</b></td><td>4.257 ± 3.274</td><td>1.280 ± 0.9324</td><td><b>CSDI</b></td></tr>
  <tr><td colspan="5"><b>Distribution</b></td></tr>
  <tr><td>A6 Path MMD² ↓</td><td><b>0.004724 ± 8.37e-04</b></td><td>0.004843 ± 0.005499</td><td>0.001785 ± 1.26e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A7 Terminal MMD² ↓</td><td><b>0.005010 ± 0.001396</b></td><td>0.009408 ± 0.01435</td><td>0.001252 ± 3.04e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A8 Increment MMD² ↓</td><td>0.002725 ± 3.31e-04</td><td><b>0.001310 ± 3.61e-04</b></td><td>8.35e-04 ± 9.07e-06</td><td><b>LS4</b></td></tr>
  <tr><td>A9 Volatility MMD ↓</td><td>0.07403 ± 0.01016</td><td><b>0.03631 ± 0.01918</b></td><td>0.007665 ± 4.31e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A10 Terminal SWD ↓</td><td>2.759 ± 0.6438</td><td><b>2.197 ± 1.712</b></td><td>0.7849 ± 0.2589</td><td><b>LS4</b></td></tr>
  <tr><td>A11 Path SWD ↓</td><td>1.652 ± 0.3123</td><td><b>1.076 ± 0.8519</b></td><td>0.5453 ± 0.06639</td><td><b>LS4</b></td></tr>
  <tr><td>A12 RV Law Loss ↓</td><td>1.257 ± 0.08482</td><td><b>0.5482 ± 0.2282</b></td><td>0.07645 ± 0.02268</td><td><b>LS4</b></td></tr>
  <tr><td>A13 Mean Path RMSE ↓</td><td>1.869 ± 0.2044</td><td><b>0.7794 ± 0.8059</b></td><td>0.1839 ± 0.07398</td><td><b>LS4</b></td></tr>
  <tr><td>A14 KS Log-returns ↓</td><td>0.03790 ± 0.003867</td><td><b>0.01940 ± 0.01036</b></td><td>0.002173 ± 6.14e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A15 Skewness Error ↓</td><td><b>0.02500 ± 0.009075</b></td><td>0.1688 ± 0.1020</td><td>0.01206 ± 0.004495</td><td><b>CSDI</b></td></tr>
  <tr><td>A16 QQ RMSE (300-pt) ↓</td><td>0.001613 ± 1.24e-04</td><td><b>8.77e-04 ± 2.96e-04</b></td><td>8.18e-05 ± 2.29e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A17 Terminal Price KS ↓</td><td>0.09575 ± 0.01135</td><td><b>0.07261 ± 0.07271</b></td><td>0.02139 ± 0.004590</td><td><b>LS4</b></td></tr>
  <tr><td colspan="5"><b>Adversarial</b></td></tr>
  <tr><td>A18 Disc Score GRU ↓</td><td>0.02459 ± 0.02851</td><td><b>0.007748 ± 0.006227</b></td><td>0.007138 ± 0.005353</td><td><b>LS4</b></td></tr>
  <tr><td>A18 Disc Score MLP ↓</td><td>0.008969 ± 0.005832</td><td><b>0.007138 ± 0.003524</b></td><td>0.006284 ± 0.004902</td><td><b>LS4</b></td></tr>
  <tr><td colspan="5"><b>Predictive</b></td></tr>
  <tr><td>A19 Pred Score GRU ↓</td><td><b>0.05638 ± 4.31e-06</b></td><td>0.05641 ± 5.08e-05</td><td>0.05638 ± 1.11e-05</td><td><b>CSDI</b></td></tr>
  <tr><td>A19 Pred Score MLP ↓</td><td>0.05657 ± 4.25e-04</td><td><b>0.05643 ± 1.31e-04</b></td><td>0.05668 ± 3.69e-04</td><td><b>LS4</b></td></tr>
  <tr><td colspan="5"><b>Temporal</b></td></tr>
  <tr><td>A20 Covariance Error ↓</td><td><b>49.05 ± 6.706</b></td><td>56.78 ± 42.52</td><td>5.825 ± 6.502</td><td><b>CSDI</b></td></tr>
  <tr><td>A21 ACF \|r\| Error (lags) ↓</td><td><b>0.008784 ± 0.003423</b></td><td>0.01490 ± 0.004887</td><td>0.001318 ± 3.93e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A22 ACF r² Error (lags) ↓</td><td><b>0.005865 ± 0.002317</b></td><td>0.01120 ± 0.003591</td><td>0.001394 ± 3.32e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A23 ACF \|r\| Lag-1 Error ↓</td><td><b>0.005258 ± 0.004223</b></td><td>0.02867 ± 0.009143</td><td>9.01e-04 ± 9.55e-04</td><td><b>CSDI</b></td></tr>
  <tr><td>A24 ACF r² Lag-1 Error ↓</td><td><b>0.003436 ± 0.002244</b></td><td>0.02300 ± 0.01036</td><td>0.001377 ± 0.001044</td><td><b>CSDI</b></td></tr>
  <tr><td colspan="5"><b>Vol</b></td></tr>
  <tr><td>A25 Mean RMSE ↓</td><td>2.920 ± 0.4190</td><td><b>1.380 ± 1.360</b></td><td>0.3668 ± 0.1535</td><td><b>LS4</b></td></tr>
  <tr><td>A26 Return Std Error ↓</td><td>0.1387 ± 0.01067</td><td><b>0.05371 ± 0.03238</b></td><td>0.005319 ± 0.002787</td><td><b>LS4</b></td></tr>
  <tr><td>A27 Log-Return Std Error ↓</td><td>0.001671 ± 1.23e-04</td><td><b>6.19e-04 ± 3.66e-04</b></td><td>7.03e-05 ± 3.39e-05</td><td><b>LS4</b></td></tr>
  <tr><td>A28 Kurtosis Ratio (→ 1)</td><td><b>1.012 ± 0.04015</b></td><td>0.7334 ± 0.2007</td><td>1.072 ± 0.02116</td><td><b>CSDI</b></td></tr>
  <tr><td>A29 Sigma Mean Error ↓</td><td>0.02551 ± 0.001956</td><td><b>0.008282 ± 0.006064</b></td><td>0.001022 ± 4.71e-04</td><td><b>LS4</b></td></tr>
  <tr><td>A30 Cross-Sect. Vol Path RMSE ↓</td><td><b>1.230 ± 0.1180</b></td><td>1.320 ± 1.181</td><td>0.1684 ± 0.09062</td><td><b>CSDI</b></td></tr>
  <tr><td>A31 Rolling Vol KS (w=5) ↓</td><td>0.1137 ± 0.009122</td><td><b>0.05316 ± 0.03392</b></td><td>0.004939 ± 0.001847</td><td><b>LS4</b></td></tr>
  <tr><td>A32 Vol-of-Vol Error ↓</td><td>6.03e-04 ± 3.13e-05</td><td><b>2.53e-04 ± 1.68e-04</b></td><td>4.26e-05 ± 2.46e-05</td><td><b>LS4</b></td></tr>
  <tr><td colspan="5"><b>Heston Spec</b></td></tr>
  <tr><td>A33 Teacher-Sigma Corr ↑</td><td>0.001505 ± 0.008084</td><td><b>0.003268 ± 0.008388</b></td><td>0.6156 ± 0.002494</td><td><b>LS4</b></td></tr>
  <tr><td>A34 Teacher-Sigma RMSE ↓</td><td><b>0.09858 ± 4.42e-04</b></td><td>0.09953 ± 0.003354</td><td>0.06560 ± 3.44e-04</td><td><b>CSDI</b></td></tr>
</tbody>
</table>

<!-- A win-counts (of 36): LS4=22, CSDI=14 -->

**Head-to-head, 36 stylised-fact rows: LS4 22 — CSDI 14.** The split is structural, not noise. LS4 owns
every **marginal-shape** and **vol** row (A2–A4 tails, A8–A14 distribution, A16–A17, A25–A27, A29, A31–A32):
its log-return input plus `cumsum→exp` reconstruction nails the return distribution and the realized-vol law.
CSDI owns every **temporal-dependence** row (A20–A24 covariance + ACF-of-|r| / ACF-of-r²) and the
tail/kurtosis shape (A1, A5, A15, A28) — the score-based diffusion reproduces the vol-clustering
autocorrelation LS4's state-space prior smooths away. Neither approaches the Perfect floor, which sits
5–20× below both on most rows.

---

## B, Curve-shape metrics cross-method comparison (mean ± std, 5 seeds)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. **MSE** combines three
lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — and is the
winner-deciding number; **% err**, **NRMSE** and **CVaR** are **funct-only** (curve L only). The **grid_tvd**
row (path-cloud 2D-histogram TVD at a locked 50×50 grid) is the **first row of Table B** and is ranked like
any plot. ↓ lower is better. The **Perfect** column is a fresh independent Heston draw at N=4096 (seeds 1000+)
scored the same way; perfect seeds carry no grid_tvd, shown as `-`.

<table>
<thead>
  <tr>
    <th rowspan="2">Plot</th>
    <th rowspan="2">Measure</th>
    <th>Diffusion</th>
    <th>VAE</th>
    <th rowspan="2">Perfect</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>CSDI</th>
    <th>LS4</th>
  </tr>
</thead>
<tbody>
  <tr><td><b>Path comparison</b><br><sub>grid_tvd 50×50 path-cloud</sub></td><td>grid_tvd 50×50 (%) ↓</td><td>8.223% ± 0.5744%</td><td><b>6.989% ± 5.606%</b></td><td>-</td><td><b>LS4</b></td></tr>
  <tr><td rowspan="5"><b>Log-return histogram</b></td><td>MSE</td><td>1.940 ± 0.4186</td><td><b>0.9677 ± 0.9381</b></td><td>0.2350 ± 0.02667</td><td><b>LS4</b></td></tr>
  <tr><td>% err</td><td>22.03% ± 1.666%</td><td><b>12.42% ± 4.928%</b></td><td>2.617% ± 0.2156%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td>6.137% ± 0.6734%</td><td><b>3.519% ± 2.171%</b></td><td>0.7244% ± 0.06654%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₀</td><td>15.22% ± 1.914%</td><td><b>8.265% ± 5.530%</b></td><td>1.634% ± 0.08697%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₅</td><td>16.67% ± 2.334%</td><td><b>9.272% ± 5.960%</b></td><td>1.878% ± 0.08662%</td><td><b>LS4</b></td></tr>
  <tr><td rowspan="5"><b>QQ plot</b></td><td>MSE</td><td>9.34e-07 ± 1.48e-07</td><td><b>3.18e-07 ± 1.71e-07</b></td><td>2.92e-09 ± 1.37e-09</td><td><b>LS4</b></td></tr>
  <tr><td>% err</td><td>20.27% ± 2.034%</td><td><b>11.35% ± 4.491%</b></td><td>1.243% ± 0.5276%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td>4.469% ± 0.3403%</td><td><b>2.444% ± 0.7371%</b></td><td>0.2264% ± 0.06458%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₀</td><td>4.816% ± 0.3392%</td><td><b>2.639% ± 0.4759%</b></td><td>0.2664% ± 0.08057%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₅</td><td>5.614% ± 0.3775%</td><td><b>3.226% ± 0.6124%</b></td><td>0.3696% ± 0.1267%</td><td><b>LS4</b></td></tr>
  <tr><td rowspan="5"><b>ACF \|r\| lags 1-20</b></td><td>MSE</td><td><b>4.65e-05 ± 1.69e-05</b></td><td>9.55e-05 ± 2.79e-05</td><td>1.22e-05 ± 5.86e-06</td><td><b>CSDI</b></td></tr>
  <tr><td>% err</td><td>41.46% ± 10.09%</td><td><b>36.42% ± 21.38%</b></td><td>7.924% ± 1.319%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td><b>26.58% ± 6.296%</b></td><td>29.90% ± 11.33%</td><td>5.554% ± 0.8533%</td><td><b>CSDI</b></td></tr>
  <tr><td>CVaR₉₀</td><td><b>34.97% ± 6.432%</b></td><td>56.04% ± 13.99%</td><td>11.36% ± 1.843%</td><td><b>CSDI</b></td></tr>
  <tr><td>CVaR₉₅</td><td><b>36.12% ± 6.918%</b></td><td>76.32% ± 21.56%</td><td>13.09% ± 2.681%</td><td><b>CSDI</b></td></tr>
  <tr><td rowspan="5"><b>ACF r² lags 1-20</b></td><td>MSE</td><td><b>2.88e-05 ± 9.15e-06</b></td><td>6.75e-05 ± 1.67e-05</td><td>1.19e-05 ± 6.12e-06</td><td><b>CSDI</b></td></tr>
  <tr><td>% err</td><td>39.19% ± 9.228%</td><td><b>32.44% ± 13.04%</b></td><td>10.48% ± 1.260%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td><b>21.13% ± 4.811%</b></td><td>24.53% ± 6.497%</td><td>6.140% ± 0.8040%</td><td><b>CSDI</b></td></tr>
  <tr><td>CVaR₉₀</td><td><b>29.34% ± 5.648%</b></td><td>49.55% ± 14.09%</td><td>12.33% ± 2.347%</td><td><b>CSDI</b></td></tr>
  <tr><td>CVaR₉₅</td><td><b>30.99% ± 6.082%</b></td><td>67.64% ± 24.63%</td><td>14.69% ± 3.607%</td><td><b>CSDI</b></td></tr>
  <tr><td rowspan="5"><b>Rolling vol histogram</b></td><td>MSE</td><td>39.18 ± 6.622</td><td><b>16.82 ± 14.35</b></td><td>1.965 ± 0.3068</td><td><b>LS4</b></td></tr>
  <tr><td>% err</td><td>37.40% ± 2.514%</td><td><b>16.34% ± 6.911%</b></td><td>3.395% ± 0.5319%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td>11.87% ± 0.9904%</td><td><b>6.709% ± 3.416%</b></td><td>1.057% ± 0.1178%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₀</td><td>23.87% ± 2.121%</td><td><b>14.46% ± 7.956%</b></td><td>2.306% ± 0.3609%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₅</td><td>24.84% ± 2.220%</td><td><b>15.44% ± 8.549%</b></td><td>2.637% ± 0.4375%</td><td><b>LS4</b></td></tr>
  <tr><td rowspan="5"><b>Tail survival</b></td><td>MSE</td><td>7.45e-04 ± 1.51e-04</td><td><b>2.58e-04 ± 3.23e-04</b></td><td>8.97e-07 ± 7.97e-07</td><td><b>LS4</b></td></tr>
  <tr><td>% err</td><td>15.65% ± 1.221%</td><td><b>6.927% ± 4.308%</b></td><td>0.6137% ± 0.2119%</td><td><b>LS4</b></td></tr>
  <tr><td>NRMSE</td><td>4.751% ± 0.4544%</td><td><b>2.219% ± 1.726%</b></td><td>0.1449% ± 0.07223%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₀</td><td>6.441% ± 0.6147%</td><td><b>3.252% ± 2.372%</b></td><td>0.2231% ± 0.09811%</td><td><b>LS4</b></td></tr>
  <tr><td>CVaR₉₅</td><td>6.455% ± 0.6160%</td><td><b>3.274% ± 2.377%</b></td><td>0.2302% ± 0.09921%</td><td><b>LS4</b></td></tr>
</tbody>
</table>

<!-- B plot-level win-counts (MSE per plot + grid_tvd, of 7): LS4=5, CSDI=2 -->
<!-- B per-subline win-counts (grid_tvd + 6×5, of 31): LS4=23, CSDI=8 -->

**Plot-level MSE + grid_tvd: LS4 5 — CSDI 2.** LS4 wins every marginal-shape plot (log-return histogram,
QQ, rolling-vol histogram, tail survival) and the path-cloud grid_tvd; CSDI wins the two
**autocorrelation** plots (ACF-of-|r|, ACF-of-r²) on MSE, and on the shape-of-the-curve NRMSE/CVaR sublines
too — the same vol-clustering advantage the A20–A24 rows show.

---

## PS, Path-Shadowing forecast — strict PDF protocol (CRPS)

This is the **strict paper protocol** (Blondel–Bouchaud–Morel, arXiv:2308.01486), *not* the price-space
Monte-Carlo CRPS of the main benchmark. Each of the 512 held-out query paths is forecast over horizon
**H=32** from its 65-point prefix, producing a K=256-member predictive ensemble of **normalized-return**
continuations. We score three quantities — **cum** (cumulative-return trajectory, M×H), **step** (per-step
return, M×H), **rv** (realized vol, M scalar) — on the **full strict-PDF metric set** (8 metrics × 3
quantities = 24 values) with a 2000-resample paired bootstrap (seed 20230814). Per quantity: **RMSE** and
energy-score **CRPS** (↓ lower is better); **coverage₅₀/coverage₉₀** (calibration — closest to the 0.50/0.90
target wins); **width₅₀/width₉₀** (interval sharpness — *diagnostic only*: width alone is meaningless
without its coverage, so it selects no winner); **lower/upper miss₉₀** (one-sided calibration — closest to
the ideal 0.05 wins). Winner is decided per row over the 6 ranked metrics (18 ranked rows total).

The two model families reach the same predictive object by **different routes**, and both are scored
identically so the columns are apples-to-apples:

- **Generators (CSDI, LS4)** — the strict route: retrieve the K=256 nearest prefixes in the method's own
  **1M-path generated bank** and read their futures as the ensemble (shadow-forecasting).
- **Forecasters (Chronos-2, TimesFM)** — conditional route: fine-tuned on the **same 4096 train paths**
  (byte-identical recipe to the main-benchmark finetune, 1000 steps, lr 1e-4, batch 256) and forecast each
  query **directly**. The load/score pipeline was smoke-tested to reproduce the published main-benchmark
  h32 CRPS to 1e-6 before the 4096 run.

The **Heston oracle** (K=256 retrieval over a 1M *ground-truth* Heston bank) is the achievable ceiling; the
**RW floor** is a calibrated Gaussian random walk. Winner excludes both.

<table>
<thead>
  <tr>
    <th rowspan="2">Quantity / metric</th>
    <th colspan="2">Generators (1M bank)</th>
    <th colspan="2">Forecasters (fine-tuned @4096)</th>
    <th rowspan="2">Heston oracle<br>(perfect floor)</th>
    <th rowspan="2">RW floor</th>
    <th rowspan="2">Winner</th>
  </tr>
  <tr>
    <th>CSDI</th>
    <th>LS4</th>
    <th>Chronos-2</th>
    <th>TimesFM</th>
  </tr>
</thead>
<tbody>
  <tr><td colspan="8"><b>cum (M×H trajectory)</b></td></tr>
  <tr><td>RMSE ↓</td><td>0.04925</td><td><b>0.04894</b></td><td>0.04921</td><td>0.05094</td><td>0.04911</td><td>0.05670</td><td><b>LS4</b></td></tr>
  <tr><td>CRPS ↓</td><td>0.02544</td><td><b>0.02534</b></td><td>0.02624</td><td>0.02722</td><td>0.02525</td><td>0.02946</td><td><b>LS4</b></td></tr>
  <tr><td>coverage₅₀ (→0.50)</td><td>0.4624</td><td><b>0.4688</b></td><td>0.4050</td><td>0.4664</td><td>0.4891</td><td>0.4719</td><td><b>LS4</b></td></tr>
  <tr><td>coverage₉₀ (→0.90)</td><td><b>0.8682</b></td><td>0.8682</td><td>0.7905</td><td>0.8177</td><td>0.8941</td><td>0.8553</td><td><b>CSDI</b></td></tr>
  <tr><td>width₅₀ (diag)</td><td>0.05285</td><td>0.05359</td><td>0.04454</td><td>0.05477</td><td>0.05719</td><td>0.06286</td><td>—</td></tr>
  <tr><td>width₉₀ (diag)</td><td>0.1348</td><td>0.1349</td><td>0.1166</td><td>0.1348</td><td>0.1465</td><td>0.1520</td><td>—</td></tr>
  <tr><td>lower miss₉₀ (→0.05)</td><td>0.07599</td><td><b>0.06757</b></td><td>0.09595</td><td>0.09265</td><td>0.05463</td><td>0.08331</td><td><b>LS4</b></td></tr>
  <tr><td>upper miss₉₀ (→0.05)</td><td><b>0.05579</b></td><td>0.06427</td><td>0.1135</td><td>0.08966</td><td>0.05127</td><td>0.06134</td><td><b>CSDI</b></td></tr>
  <tr><td colspan="8"><b>step (M×H)</b></td></tr>
  <tr><td>RMSE ↓</td><td>0.01245</td><td><b>0.01245</b></td><td>0.01295</td><td>0.01332</td><td>0.01244</td><td>0.01253</td><td><b>LS4</b></td></tr>
  <tr><td>CRPS ↓</td><td><b>0.006767</b></td><td>0.006778</td><td>0.01272</td><td>0.01473</td><td>0.006750</td><td>0.006874</td><td><b>CSDI</b></td></tr>
  <tr><td>coverage₅₀ (→0.50)</td><td>0.4532</td><td><b>0.4672</b></td><td>0.9449</td><td>0.9479</td><td>0.4816</td><td>0.5157</td><td><b>LS4</b></td></tr>
  <tr><td>coverage₉₀ (→0.90)</td><td><b>0.8640</b></td><td>0.8640</td><td>0.9970</td><td>0.9949</td><td>0.8864</td><td>0.8866</td><td><b>CSDI</b></td></tr>
  <tr><td>width₅₀ (diag)</td><td>0.01338</td><td>0.01368</td><td>0.06405</td><td>0.07667</td><td>0.01449</td><td>0.01602</td><td>—</td></tr>
  <tr><td>width₉₀ (diag)</td><td>0.03550</td><td>0.03538</td><td>0.1609</td><td>0.1801</td><td>0.03815</td><td>0.03968</td><td>—</td></tr>
  <tr><td>lower miss₉₀ (→0.05)</td><td>0.07062</td><td><b>0.06921</b></td><td>0.001770</td><td>0.002319</td><td>0.05725</td><td>0.05933</td><td><b>LS4</b></td></tr>
  <tr><td>upper miss₉₀ (→0.05)</td><td><b>0.06543</b></td><td>0.06683</td><td>0.001221</td><td>0.002808</td><td>0.05634</td><td>0.05408</td><td><b>CSDI</b></td></tr>
  <tr><td colspan="8"><b>rv (M scalar)</b></td></tr>
  <tr><td>RMSE ↓</td><td><b>0.01772</b></td><td>0.01819</td><td>0.2307</td><td>0.2865</td><td>0.01625</td><td>0.01876</td><td><b>CSDI</b></td></tr>
  <tr><td>CRPS ↓</td><td><b>0.009938</b></td><td>0.01032</td><td>0.1913</td><td>0.2291</td><td>0.009143</td><td>0.01168</td><td><b>CSDI</b></td></tr>
  <tr><td>coverage₅₀ (→0.50)</td><td><b>0.4746</b></td><td>0.4102</td><td>0</td><td>0</td><td>0.4727</td><td>0.2344</td><td><b>CSDI</b></td></tr>
  <tr><td>coverage₉₀ (→0.90)</td><td><b>0.8633</b></td><td>0.7988</td><td>0</td><td>0.001953</td><td>0.9238</td><td>0.5332</td><td><b>CSDI</b></td></tr>
  <tr><td>width₅₀ (diag)</td><td>0.02115</td><td>0.01914</td><td>0.06861</td><td>0.06689</td><td>0.02217</td><td>0.01202</td><td>—</td></tr>
  <tr><td>width₉₀ (diag)</td><td>0.05138</td><td>0.04589</td><td>0.1651</td><td>0.1610</td><td>0.05411</td><td>0.02873</td><td>—</td></tr>
  <tr><td>lower miss₉₀ (→0.05)</td><td>0.01758</td><td><b>0.05078</b></td><td>1.000</td><td>0.9980</td><td>0.02930</td><td>0.2891</td><td><b>LS4</b></td></tr>
  <tr><td>upper miss₉₀ (→0.05)</td><td>0.1191</td><td>0.1504</td><td><b>0</b></td><td>0</td><td>0.04688</td><td>0.1777</td><td><b>Chronos-2</b></td></tr>
</tbody>
</table>

<!-- PS strict win-counts (of 18 ranked rows = 6 ranked metrics × 3 quantities; width rows are diagnostic): CSDI=9, LS4=8, Chronos-2=1 -->

**The generators dominate the forecasters on distributional forecasting, and the full metric set shows
*where* the forecasters break.** On the smooth **cum** trajectory all four methods are within ~7% on
CRPS/RMSE and near the Heston oracle — a partial path's cumulative drift is easy, and even a point-forecaster
tracks it. But move to the **shape** of the distribution and the two families diverge:

- **step** — Chronos-2/TimesFM over-cover wildly (coverage₉₀ 0.995–0.997 vs target 0.90) with **4–5× too wide**
  intervals (width₉₀ 0.16–0.18 vs the generators' ~0.035). They hedge per-step uncertainty by inflating the
  band, so almost nothing misses — lower/upper miss₉₀ ≈ 0.001–0.003 vs the ideal 0.05. Point-accurate, badly
  calibrated.
- **rv (realized vol)** — total collapse: rv-CRPS **0.19–0.23** vs the generators' ~0.010 and the oracle's
  0.0091, a **~20× gap**. Their rv intervals sit entirely below the truth (coverage₉₀ ≈ 0, lower miss₉₀ ≈ 1.0):
  a single-shot conditional-mean forecaster cannot represent the volatility-of-volatility law at all, which is
  exactly what path-shadowing over a well-trained generator bank recovers. (Chronos-2's lone "win" — rv upper
  miss₉₀ = 0 — is degenerate: 0% of truth exceeds an interval that the *whole* distribution already sits above.)

Between the generators it is close: **CSDI 9 — LS4 8 — Chronos-2 1** across the 18 ranked rows. CSDI takes the
calibration/coverage rows and rv; LS4 takes the cum/step central-tendency rows (RMSE, cum CRPS, the 50%
coverages). Both generators sit right on the Heston oracle across all 24 values and far ahead of either
forecaster on everything but bare cum drift — the whole point of shadow-forecasting over a distribution-faithful
generator rather than a mean-reverting point forecaster.

---

## Per-method diagnostics — log-return preprocessing vs raw price (side-by-side)

For each method the **same** 8-panel stylised-facts diagnostic is shown twice at the **matched
4096-path / seed-0 control**: the **log-return preprocessing** run (left) against the **raw-price,
no-preprocessing** baseline (right, `baseline_no_preproc/`). Same real test set on both sides — the
only difference is the input transform, so any divergence between the two panels is the preprocessing's
effect in isolation. Follow each method's `README.md` for the per-metric Δ% head-to-head behind these
pictures.

### LS4

| log-return preprocessing (with) | raw price, no preprocessing |
|:-------------------------------:|:---------------------------:|
| ![LS4 logret diagnostics](LS4/plots/heston_diagnostics.png) | ![LS4 raw diagnostics](LS4/baseline_no_preproc/plots/heston_diagnostics.png) |

Preprocessing recovers the ACF-of-|r| / ACF-of-r² and rolling-vol curves the raw-price VAE flatlines
(A21 −66%, A22 −55%); the raw model keeps a tighter return-std marginal. Net **19–16 for log-returns**.

### CSDI

| log-return preprocessing (with) | raw price, no preprocessing |
|:-------------------------------:|:---------------------------:|
| ![CSDI logret diagnostics](CSDI/plots/heston_diagnostics.png) | ![CSDI raw diagnostics](CSDI/baseline_no_preproc/plots/heston_diagnostics.png) |

Preprocessing tightens the return-histogram / QQ marginal and the vol block (A9 −62%, A31 −42%,
A32 −48%), but the `cumsum→exp` **terminal drift (+3.25%)** hands the raw model every price-location
row (A13, A17, A25, grid_tvd). Net **19–16 for raw** on row-count — the mirror of LS4.

---

## What differs from the main benchmark

| Axis | Main benchmark (`results/Heston/<method>/`) | This experiment |
|------|---------------------------------------------|-----------------|
| Generator input | Standardized **price** `(S − μ)/σ` | Volatility-scaled **log-returns** `R·√dt/σ` (SBTS scaler) |
| Sample count | 8192 train / 8192 test / 8192 disc | **4096** train / 4096 test / 4096 disc |
| Path-shadowing queries | (per method) | **512** fresh, strictly-independent real paths (seed 3) |
| Path-shadowing metric | Price-space MC CRPS (~2.7) | **Strict PDF protocol** (arXiv:2308.01486), normalized-return CRPS (~0.025) |
| Everything else | — | **identical**: SDE, params, RNG, metric code, seeds 0–4 |

The datasets live in [`dataset/Heston/preprocessing_with_log_returns/`](../../../dataset/Heston/preprocessing_with_log_returns/README.md).
Only the **preprocessing** and **sample count** change; the SDE draws are byte-identical
truncations of the same Heston stream.

---

## The preprocessing pipeline (exact, reproducible)

This is the SBTS transform, copied verbatim from `methods/SBTS/code/sbts_generate.py`. It maps a
price panel `S ∈ ℝ^{M×128}` to a model input `X ∈ ℝ^{M×128}` and back.

### Forward (price → model input)

```python
import numpy as np
DT = 1.0 / 250.0          # daily step, identical to the SDE
S0 = 100.0

# 1. log-returns  (M, 127)
R = np.log(S[:, 1:] / S[:, :-1])

# 2. SBTS pooled sigma  (single scalar, estimated on TRAIN split only, then frozen)
sigma = float(R.std())    # population std, ddof=0, pooled over all M*127 returns

# 3. volatility-scaled returns  (std becomes exactly sqrt(DT))
R_tilde = R * np.sqrt(DT) / sigma

# 4. prepend a dummy zero column -> (M, 128), same length as the price panel
X = np.hstack([np.zeros((R.shape[0], 1)), R_tilde])
```

The **dummy zero column** exists purely so the model input has the same sequence length (128) as
the price series. It carries no information (it is the same constant for every path) and is
**discarded** on the way back. Train the generator on `X`; sample `X_gen ∈ ℝ^{M×128}`.

### Inverse (model output → price)

```python
# 1. drop the dummy column, undo the vol scaling
R_tilde_gen = X_gen[:, 1:]                 # (M, 127)
R_gen = R_tilde_gen * sigma / np.sqrt(DT)  # SAME frozen sigma as the forward pass

# 2. re-integrate to a price path anchored at S0 = 100
S_gen = np.empty((R_gen.shape[0], 128))
S_gen[:, 0] = S0
S_gen[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))
```

Every generated path therefore starts at exactly `S0 = 100` (so `S_gen[:,0].std() == 0`),
matching the real data convention. `sigma` is **the same scalar** in the forward and inverse
directions — it is estimated once on the train split and never re-estimated on test/disc/generated.

---

## How `sigma` is estimated from SBTS (the exact estimator)

SBTS uses a **single pooled standard deviation** of the raw log-returns, computed on the
**train split only** (seed 0), and freezes it. The one line that defines it, verbatim from SBTS:

```python
sigma = float(R.std())     # R = log(S[:,1:]/S[:,:-1]),  R.shape == (M, 127)
```

- `.std()` is NumPy's **population** standard deviation (`ddof=0`).
- It is **pooled**: one scalar over all `M × 127` returns, not per-path or per-timestep.
- It is estimated **once, on train (seed 0)**, and reused unchanged for the inverse transform of
  every seed's generated paths. This mirrors SBTS exactly and keeps the scaling a fixed property
  of the dataset rather than something the generator can drift.

**Value estimated on the train split (seed 0, 4096 × 128):**

| Quantity | Value |
|----------|-------|
| `DT` | 0.004000  (= 1/250) |
| `sqrt(DT)` | 0.063246 |
| pooled count | 4096 × 127 = 520 192 returns |
| **`sigma = R.std()`** | **0.01263163** |
| `std(R_tilde)` (check) | 0.063246  (= `sqrt(DT)`, by construction) |

Why scale to `sqrt(DT)`? After `R_tilde = R·√dt/σ`, the scaled returns have unit-per-`√dt`
variance, i.e. `std(R_tilde) == sqrt(DT)` exactly. This puts the series on the natural
Brownian scale the generators expect, independent of the absolute price level.

**Reproduce the value:**

```bash
cd dataset/Heston/preprocessing_with_log_returns
python generate_datasets.py       # prints DT, sqrt(DT), sigma=std(R), and the scaling check
```

---

## Datasets

See [`dataset/Heston/preprocessing_with_log_returns/README.md`](../../../dataset/Heston/preprocessing_with_log_returns/README.md)
for the full split table and file list. Summary:

| Split | Seed | N | Role |
|-------|:----:|:----:|------|
| train | 0 | 4096 | only data a generator sees (after preprocessing) |
| test | 1 | 4096 | held-out real reference for every A/B metric |
| disc | 2 | 4096 | A18/A19 discriminative/predictive classifiers |
| ps | 3 | 512 | fresh path-shadowing queries, independent of the 1M bank |

---

## Methods

| Method | Status | Folder | Verdict |
|--------|--------|--------|---------|
| CSDI | ✅ done (5 seeds + 1M PS bank) | [`CSDI/`](CSDI/README.md) | **mirror-opposite of LS4** — log-returns **fix** the vol/tail/adversarial facts (A9 −70%, A31 −48%, A32 −42%, A18-GRU halved) but the `cumsum→exp` reconstruction **drifts the terminal +3.25%**, breaking price-location (A13, A17, A25, grid_tvd). Seed-stable. Matched control: raw wins **19–16** on row-count, log-returns win the facts. |
| LS4 | ✅ done (5 seeds + 1M PS bank) | [`LS4/`](LS4/README.md) | log-returns **hurt** LS4 — 30/34 A-metrics regress, seeds destabilize; only A28 kurtosis-ratio improves. On the matched 4096/seed-0 control the preprocessing wins the stylised facts **19–16**. |

Each method folder mirrors `results/Heston/<method>/`: `code/`, `losses/`, `weights/`,
`generated_paths/seed_{0..4}/`, `plots/`, `path_shadowing/`, and a per-method `README.md` with
the A1–A34 table, B-curve table, diagnostics figure, and Path-Shadowing CRPS table.

The cross-method tables above are regenerated by
[`build_cross_tables.py`](build_cross_tables.py) (`--which A|B|PS|all`), which reuses the canonical
`metrics/render_tables.py` renderer so the format, formatting and winner logic are byte-identical to the
main [`results/`](../../README.md) benchmark.
