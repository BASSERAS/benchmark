# Deep-MKV-TS, Deep McKean–Vlasov Time Series generation

**Paper:** Basseras et al., *Deep McKean–Vlasov Time Series Generation* (paper reimplementation vendored under [`methods/Deep-MKV-TS/paper_reimplementation/`](../../../methods/Deep-MKV-TS/paper_reimplementation/))

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** path-dependent McKean–Vlasov stochastic control. A **frozen** Guyon–Lekeufack
reference SDE supplies the drift and a baseline diffusion; a 1-layer GRU
(`hidden_dim = 96`, **47 330 parameters**) learns a **volatility correction only**.
Objective = six-component MMD discrepancy + specific-entropy running cost (η = 1).
Trained 3 000 Adam steps at `lr = 2e-3`; the **step-2500** checkpoint is reported (paper K = 2500).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

**Evaluation protocol (test set everywhere).** Generated paths were built from the **train**
split (seed 0) and are **never scored on it**. Every metric below compares the 8 192 generated
paths against the held-out **test** split `heston_S_test_8192x128.npy` (an independent 8 192-path
Heston draw), with one deliberate exception: A18 (discriminative) and A19 (predictive-TSTR) draw
their *real* class from the **disc** split `heston_S_disc_8192x128.npy`, so the judge never sees
the same real data used everywhere else. This is the protocol applied identically to every method.

> **On `test_split_access_authorized: false`.**
> [`paper_reimplementation/metric/PROTOCOL.json`](../../../methods/Deep-MKV-TS/paper_reimplementation/metric/PROTOCOL.json)
> carries that flag, and it is honoured: **training and hyperparameter selection never read `test`.**
> Selection ran on a separate `val` / `valdisc` pair and the winner was scored on `disc` exactly once.
> The benchmark's *post-hoc* scoring on `test`, which produces every number in this file, is mandated
> by GUIDELINE §5.0 and applies to all methods equally. The two statements describe different stages
> of the pipeline and are not in conflict.

---

## What we generate, price paths from the Heston SDE

The **target process** is the Heston stochastic volatility model:

$$dS_t = \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \quad \text{Corr}(dW^S, dW^v) = \rho$$

Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Deep-MKV-TS generates price paths $S_t$ directly, in the original price scale.** There is no
MinMax rescaling and no inverse transform: exported paths run from ≈ 42 to ≈ 141 with S₀ = 100.
The pipeline is:

```
Step 0: fit the Guyon-Lekeufack reference SDE by Gaussian quasi-likelihood on 80% of the
        8192 training paths, select on the remaining 20%.  Reference is then FROZEN.
          - short/long trend signals + short/long activity signals
          - activity recursion (paper eq. 2):
                V_{i+1} = e^{-lambda_{V,a} dt_i} V_i + lambda_{V,a} (v_i^ref)^{o2} dt_i
          - sigma clipped to [1e-3, 0.6];  d = 1
          - seed 0 fit: calibration NLL = -1.1466, validation NLL = -1.1542

Step 1: roll out the frozen reference to get the SOURCE law mu^ref.

Step 2: train the correction network alpha_theta (1-layer GRU, hidden 96, 47 330 params) to minimise
             J(alpha) = D(mu^alpha, mu^data) + eta * E[ int 1/2 ||alpha_t||^2 sigma_t^{-2} dt ],  eta = 1
        where D is the weighted sum of six MMD^2 terms:
             observed path . increments . terminal . global RV
           . |r| ACF (weight 0.25) . r^2 ACF (weight 0.125)
        The DRIFT IS NEVER TOUCHED - only `expected_adjoint_noise_next_head` receives gradients
        (`--adjoint-weight 0 --adjoint-noise-weight 1`).

Step 3: at step 2500, roll out the CONTROLLED SDE with bank_seed = 70000 + seed to draw
        8192 fresh paths of length 128.
Output: S_gen (8192, 128), float64, price scale, S0 = 100.
```

Because the drift is a single frozen object shared by all five seeds, seed-to-seed variation
enters only through the 47 330 correction parameters and the sampling noise. This is why the
cross-seed standard deviations below are 1–2 orders of magnitude tighter than for methods that
learn a full generator from scratch.

---

## Results (mean ± std across 5 seeds)

### A1-A34, Metrics by category

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.
> The last column is the **Perfect-Recovery floor** (§5.4): an independent Heston draw scored against
> the same test set, regenerated via `render_tables.py`, identical for every method. It is **non-zero**.

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **Fat Tail** | | | | | | | |
| A1 Kurtosis Error ↓ | 0.3334 ± 0.02706 | 0.3756 | 0.3383 | 0.3020 | 0.3063 | 0.3450 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 2.10e-04 ± 1.13e-04 | 2.90e-04 | 1.76e-04 | 9.81e-06 | 3.33e-04 | 2.41e-04 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 3.75e-04 ± 2.54e-04 | 8.39e-04 | 3.34e-04 | 8.90e-05 | 2.17e-04 | 3.94e-04 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 2.70e-04 ± 2.95e-05 | 2.92e-04 | 2.65e-04 | 2.70e-04 | 3.03e-04 | 2.17e-04 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 5.110 ± 0.6663 | 5.895 | 5.850 | 4.713 | 4.908 | 4.184 | 0.5266 |
| **Distribution** | | | | | | | |
| A6 Path MMD² ↓ | 0.002031 ± 1.48e-04 | 0.001779 | 0.002152 | 0.002088 | 0.002180 | 0.001955 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.002535 ± 0.001066 | 0.001171 | 0.003995 | 0.003484 | 0.001663 | 0.002362 | 0.001983 |
| A8 Increment MMD² ↓ | 9.42e-04 ± 3.37e-05 | 9.99e-04 | 9.21e-04 | 9.17e-04 | 9.62e-04 | 9.12e-04 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.01743 ± 9.18e-04 | 0.01887 | 0.01792 | 0.01713 | 0.01712 | 0.01612 | 0.008554 |
| A10 Terminal SWD ↓ | 0.9334 ± 0.1952 | 0.7016 | 0.9100 | 1.244 | 1.044 | 0.7676 | 1.151 |
| A11 Path SWD ↓ | 0.5509 ± 0.05390 | 0.4997 | 0.5020 | 0.5661 | 0.6466 | 0.5404 | 0.6191 |
| A12 RV Law Loss ↓ | 0.1215 ± 0.04230 | 0.1952 | 0.08286 | 0.09060 | 0.1423 | 0.09636 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.1500 ± 0.05181 | 0.1100 | 0.2300 | 0.1840 | 0.1405 | 0.08543 | 0.1205 |
| A14 KS Log-returns ↓ | 0.008172 ± 0.001151 | 0.009454 | 0.007865 | 0.008078 | 0.006233 | 0.009230 | 0.001491 |
| A15 Skewness Error ↓ | 0.05862 ± 0.003704 | 0.06011 | 0.06239 | 0.05798 | 0.05177 | 0.06082 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 2.13e-04 ± 2.73e-05 | 2.60e-04 | 2.10e-04 | 2.01e-04 | 1.76e-04 | 2.16e-04 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.03333 ± 0.006508 | 0.02722 | 0.03430 | 0.03662 | 0.04321 | 0.02527 | 0.01099 |
| **Adversarial** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.007355 ± 0.003169 | 0.006256 | 0.002289 | 0.007782 | 0.008392 | 0.01205 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.003448 ± 0.001642 | 0.004425 | 0.003509 | 4.58e-04 | 0.003509 | 0.005340 | 0.005951 |
| **Predictive** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05003 ± 2.27e-05 | 0.05001 | 0.05006 | 0.05002 | 0.05001 | 0.05007 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05010 ± 2.26e-04 | 0.04992 | 0.04996 | 0.05004 | 0.05003 | 0.05054 | 0.05036 |
| **Temporal** | | | | | | | |
| A20 Covariance Error ↓ | 12.22 ± 3.900 | 18.34 | 12.30 | 12.49 | 11.95 | 6.017 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.01413 ± 0.001621 | 0.01626 | 0.01337 | 0.01152 | 0.01427 | 0.01522 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.008819 ± 0.001354 | 0.01076 | 0.008410 | 0.006776 | 0.008394 | 0.009757 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.01762 ± 0.001716 | 0.01985 | 0.01661 | 0.01491 | 0.01872 | 0.01803 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.01144 ± 0.001556 | 0.01317 | 0.01141 | 0.008572 | 0.01175 | 0.01232 | 0.002790 |
| **Vol** | | | | | | | |
| A25 Mean RMSE ↓ | 0.2653 ± 0.07286 | 0.2234 | 0.3360 | 0.3273 | 0.2967 | 0.1430 | 0.1392 |
| A26 Return Std Error ↓ | 0.01122 ± 0.005529 | 0.008511 | 0.008698 | 0.004810 | 0.02094 | 0.01313 | 0.002523 |
| A27 Log-Return Std Error ↓ | 3.90e-05 ± 3.32e-05 | 4.41e-06 | 2.77e-05 | 1.01e-04 | 4.09e-05 | 2.07e-05 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | 1.182 ± 0.1352 | 1.389 | 1.261 | 1.082 | 1.001 | 1.177 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.001118 ± 4.45e-04 | 0.001145 | 0.001052 | 0.001918 | 5.71e-04 | 9.03e-04 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.2623 ± 0.05918 | 0.3785 | 0.2541 | 0.2202 | 0.2315 | 0.2273 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.02488 ± 0.003857 | 0.03146 | 0.02323 | 0.02342 | 0.01997 | 0.02633 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 1.72e-04 ± 6.04e-05 | 2.75e-04 | 1.71e-04 | 9.29e-05 | 1.37e-04 | 1.83e-04 | 1.54e-05 |
| **Heston Spec** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -0.001032 ± 0.006621 | 0.007350 | -0.007386 | 0.001398 | 0.003547 | -0.01007 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.09718 ± 0.001019 | 0.09522 | 0.09741 | 0.09815 | 0.09775 | 0.09739 | 0.06559 |

> **Headline:** Deep-MKV-TS **wins 9 of the 36 A-metric rows** — second overall, behind SBTS (12) and
> ahead of LS4 (8). It also **wins both PS-MC horizons (2/2)** and **0 of the 7 ranked B contests**.

**Footnotes on the non-obvious IDs.**

- **A5 Hill Tail Index Error = 5.110** (floor 0.527) is the single worst row in the table. The entropy
  running cost charges the control for every unit of deviation from the frozen reference, and the
  cheapest way to shrink the six MMD terms is to tighten the bulk rather than to reproduce the extreme
  quantiles. The tails come out too thin.
- **A18** discriminative classifier, score = |accuracy − 0.5|. GRU 0.00736 (just above the 0.00620 floor),
  MLP 0.00345 (**below** the 0.00595 floor). The MLP judge sees only the marginal, which Deep-MKV-TS
  matches almost exactly; the GRU judge sees the temporal structure and still finds a small edge.
- **A19** TSTR MAE, at the irreducible ≈ 0.050 floor for both judges. One-step prediction on a
  well-matched 1-D return marginal is easy for every method in the benchmark; this row separates nothing.
- **A28 Kurtosis Ratio 1.182 > 1** (perfect 1.0): generated returns are *less* leptokurtic than real
  ones. Together with A5 this is the clearest quantitative statement of the method's one systematic bias.
- **A33 Teacher-Sigma Corr ≈ 0** (floor 0.616): the generated price paths do not carry the latent Heston
  variance path that produced them. This is not specific to Deep-MKV-TS — **every generator in this
  benchmark scores ≈ 0 here**. The objective $\mathcal{J}(\alpha)$ is purely distributional; nothing in
  it asks the model to recover the particular $v_t$ realisation behind a given price path.
- **A34 Teacher-Sigma RMSE 0.0972** (floor 0.0656): the *scale* of the reproduced volatility is roughly
  right even though its *timing* (A33) is uncorrelated.

**Four cells sit below the Perfect floor** — A10 (0.9334 vs 1.151), A11 (0.5509 vs 0.6191),
A18-MLP (0.00345 vs 0.00595), and both PS-MC CRPS horizons. **This is a variance deficit, not
superiority.** The Perfect floor is an independent *finite-sample* draw and therefore carries the
full sampling variance of a real 8 192-path cloud; a slightly under-dispersed generated cloud sits
closer to the test set's empirical measure than a second honest draw does, and beats the floor on
any distance that rewards concentration. A5 and A28 independently confirm the under-dispersion.

---

## Stylised Facts Diagnostic

Eight-panel comparison (seed 0): sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

### A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near $\ln 2 \approx 0.693$ means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

### A19, Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Curve-shape metrics (B), mean ± std across 5 seeds

> Each stylised-fact plot yields a **curve** $L$, not a scalar. From $L$, its 1st difference (der) and
> its 2nd difference (sec\_der) we compute the measures below, combined into one number per plot
> (combined std = sample std across the 5 seeds):
> - **MSE**: `mean((L_gen − L_real)²)` per sub-metric; combined = **mean-of-3** (funct + der + sec\_der)/3. **Decides the winner.**
> - **% err**: `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` — MAPE with a fixed 1e-6 floor; **funct-only**.
> - **NRMSE**: `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100`; **funct-only**.
> - **CVaR₉₀ / CVaR₉₅**: mean of the worst 10% / 5% pointwise relative errors along the curve; **funct-only**.
>
> % err, NRMSE and the two CVaRs are funct-only for every plot: the der / sec\_der true values sit near
> zero and blow the ratios into meaningless 10⁴-% figures. MSE keeps mean-of-3. All ↓ lower is better.
> The Perfect floor is **non-zero for every plot** (independent draw vs test set, §5.4).
>
> `grid_tvd` (50×50 total-variation distance on the joint grid) is the **first ranked row of Table B**;
> there are **7 ranked B contests** in total (`grid_tvd` + 6 curves).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **grid_tvd 50×50 (%)** | TVD | 3.575% ± 0.4069% | 2.986% | 3.724% | 3.777% | 4.135% | 3.252% | 2.237% |
| **Log-return histogram** | MSE | 0.2575 ± 0.05127 | 0.3450 | 0.2220 | 0.1991 | 0.2407 | 0.2805 | 0.1098 |
|  | % err | 3.424% ± 0.3123% | 3.965% | 3.341% | 2.991% | 3.424% | 3.400% | 1.799% |
|  | NRMSE | 1.898% ± 0.1993% | 2.212% | 1.837% | 1.776% | 1.641% | 2.024% | 0.5328% |
|  | CVaR₉₀ | 4.831% ± 0.4587% | 5.537% | 4.685% | 4.541% | 4.242% | 5.150% | 1.234% |
|  | CVaR₉₅ | 5.967% ± 0.6218% | 6.779% | 5.846% | 5.808% | 4.962% | 6.440% | 1.444% |
| **QQ plot** | MSE | 1.94e-08 ± 5.35e-09 | 2.93e-08 | 1.83e-08 | 1.68e-08 | 1.33e-08 | 1.90e-08 | 1.09e-09 |
|  | % err | 4.987% ± 0.6969% | 5.415% | 4.990% | 4.640% | 3.915% | 5.974% | 0.4629% |
|  | NRMSE | 0.6142% ± 0.08099% | 0.7571% | 0.6064% | 0.5794% | 0.5089% | 0.6194% | 0.1206% |
|  | CVaR₉₀ | 0.5913% ± 0.06009% | 0.6994% | 0.5612% | 0.5175% | 0.5873% | 0.5913% | 0.1319% |
|  | CVaR₉₅ | 0.8014% ± 0.09523% | 0.9737% | 0.7753% | 0.6883% | 0.8135% | 0.7562% | 0.1599% |
| **ACF \|r\| lags 1-20** | MSE | 5.06e-05 ± 1.08e-05 | 6.88e-05 | 4.03e-05 | 4.04e-05 | 4.74e-05 | 5.62e-05 | 9.61e-06 |
|  | % err | 36.73% ± 6.242% | 46.34% | 32.94% | 29.61% | 33.10% | 41.64% | 8.724% |
|  | NRMSE | 30.12% ± 3.761% | 35.87% | 27.81% | 25.45% | 28.53% | 32.95% | 6.071% |
|  | CVaR₉₀ | 46.82% ± 5.219% | 54.35% | 43.30% | 39.32% | 47.06% | 50.10% | 11.26% |
|  | CVaR₉₅ | 47.96% ± 5.550% | 55.87% | 44.20% | 39.67% | 49.81% | 50.25% | 12.06% |
| **ACF r² lags 1-20** | MSE | 2.53e-05 ± 5.29e-06 | 3.48e-05 | 2.10e-05 | 2.13e-05 | 2.20e-05 | 2.72e-05 | 9.17e-06 |
|  | % err | 26.21% ± 6.110% | 36.78% | 22.85% | 20.96% | 21.09% | 29.37% | 11.34% |
|  | NRMSE | 21.19% ± 3.446% | 26.83% | 19.66% | 17.33% | 18.81% | 23.36% | 6.486% |
|  | CVaR₉₀ | 35.77% ± 4.879% | 44.43% | 33.55% | 30.01% | 33.75% | 37.12% | 12.35% |
|  | CVaR₉₅ | 36.91% ± 5.102% | 46.56% | 33.68% | 32.44% | 34.41% | 37.47% | 13.27% |
| **Rolling vol histogram** | MSE | 4.491 ± 0.9730 | 6.288 | 3.731 | 3.595 | 4.182 | 4.660 | 1.372 |
|  | % err | 6.961% ± 1.556% | 9.839% | 5.757% | 5.459% | 6.649% | 7.099% | 2.264% |
|  | NRMSE | 3.558% ± 0.4497% | 4.328% | 3.144% | 3.113% | 3.453% | 3.752% | 0.8688% |
|  | CVaR₉₀ | 7.883% ± 0.7741% | 9.269% | 7.274% | 7.321% | 7.347% | 8.203% | 1.970% |
|  | CVaR₉₅ | 8.409% ± 0.8117% | 9.844% | 7.782% | 7.828% | 7.806% | 8.784% | 2.308% |
| **Tail survival** | MSE | 2.60e-05 ± 8.36e-06 | 3.84e-05 | 2.55e-05 | 2.51e-05 | 1.23e-05 | 2.87e-05 | 5.22e-07 |
|  | % err | 1.782% ± 0.3354% | 2.375% | 1.760% | 1.573% | 1.377% | 1.828% | 0.3302% |
|  | NRMSE | 0.8779% ± 0.1525% | 1.082% | 0.8825% | 0.8762% | 0.6117% | 0.9367% | 0.1050% |
|  | CVaR₉₀ | 1.335% ± 0.2110% | 1.636% | 1.329% | 1.311% | 0.9822% | 1.419% | 0.1625% |
|  | CVaR₉₅ | 1.346% ± 0.2069% | 1.645% | 1.335% | 1.321% | 1.003% | 1.425% | 0.1682% |

**Deep-MKV-TS wins 0 of the 7 ranked B contests** (SBTS 6, LS4 1). Read in multiples of the Perfect
floor rather than in absolute terms:

| Plot | MSE / floor | Comment |
|------|------------:|---------|
| grid_tvd | 1.6× | best B row |
| Log-return histogram | 2.3× | |
| ACF r² | 2.8× | |
| Rolling vol histogram | 3.3× | |
| ACF \|r\| | 5.3× | volatility clustering *shape* is reproduced |
| QQ plot | 17.8× | bulk quantiles good, extremes pull it out |
| Tail survival | **49.8×** | worst B row — the tail deficit again |

> **The ACF % errors look catastrophic and are not.** The true Heston ACF of $|r|$ sits near **0.05**,
> so a 0.015 absolute miss is a 30% relative miss by construction. Against the floor it is 5.3× — for
> reference, TimeGAN sits at roughly 370× on the same row.
>
> **Where the losses actually come from.** The two rows furthest from the floor — tail survival (49.8×)
> and the QQ plot (17.8×) — are precisely the two that read off the tails. Same under-dispersion the A
> table shows through A5 and A28, seen through a curve-shape lens. Deep-MKV-TS loses every B row not
> because its curves are misshapen but because SBTS sits at or near the floor on six of the seven.

---

## Comparison with the paper (Deep-MKV-TS, Table 1, Heston)

> ⚠️ **Unlike most methods in this benchmark, the paper's own dataset *is* Heston**, so this comparison
> is genuinely like-for-like on the process. Two caveats remain and matter:
>
> 1. **Different metric suite.** This section uses the **paper's own five Table-1 columns** (SWD, RV W₁,
>    |r| ACF, Early-future, MDD W₁). It does **not** use A1–A34; that comparison is the Results section above.
> 2. **Different scoring split.** Both "Ours" columns are scored on **`disc`**
>    (`heston_S_disc_8192x128.npy`), the split frozen into
>    [`PROTOCOL.json`](../../../methods/Deep-MKV-TS/paper_reimplementation/metric/PROTOCOL.json) as the
>    paper-reproduction evaluation split. They are **not** re-scored on `test`, because re-pointing the
>    frozen paper protocol at `test` would destroy the reproduction record it exists to preserve. The
>    test-split numbers for this method are the A/B/PS tables above.
>
> The paper reports **medians over four seeds (0, 1, 3, 4)**. Column 2 reproduces that exactly. Column 3
> adds seed 2 — which the benchmark requires and the paper does not use — and reports the median over all
> five, so it matches the benchmark's 5-seed A/B tables in construction.

### A. Hyperparameter verification

Every setting below was pinned from the paper text and cross-checked against the upstream checkpoint;
the "Ours" column is read verbatim from
[`methods/Deep-MKV-TS/weights/seed_0_config.json`](../../../methods/Deep-MKV-TS/weights/seed_0_config.json).

| Setting | Our reimplementation | Paper (source) |
|---------|:--------------------:|:--------------:|
| Task | Heston | §4, Table 1 |
| Reference model | Guyon–Lekeufack structural likelihood | §2.1, ref `[11]` |
| Reference activity update | structural variance, $(v^{\text{ref}})^{\odot 2}$-driven | eq. (2) |
| Reference calibration split | 80% fit / 20% select | §2.1 |
| Reference σ clip | `[1e-3, 0.6]` | §2.1 |
| Physical drift | frozen at the fitted reference drift | §2 |
| Drift-adjoint weight | **0** | §2 (volatility-only correction) |
| Noise-adjoint weight | **1** | §2 (volatility-only correction) |
| Running cost | specific entropy | §3 |
| Entropy weight η | **1** | §3 |
| λ scale (path) | **50** | Table 6 / App. B |
| κ scale (vol) | **100** | Table 6 / App. B |
| Learning rate | **2 × 10⁻³** | Table 6 / App. B |
| Gradient clip (norm) | **5** | Table 6 / App. B |
| Bank size | 8 192 | matches dataset size |
| Sample batch size | **2048** | App. B |
| Joint-volatility weight | **0** | Table 6 |
| \|r\| ACF weight | **0.25** | Table 6 |
| r² ACF weight | **0.125** | Table 6 |
| Steps trained | **3000** | App. B |
| **Step reported (K)** | **2500** | App. B (K = 2500) |
| Solver | online | App. B |
| Network | 1-layer GRU hidden 96 + two `Linear(96,96)→Linear(96,1)` heads, **47 330** params | read off the checkpoint |
| Weight decay | 1 × 10⁻⁵ | checkpoint `training` dict |

> **K = 2500 is exact, not approximate.** Every run trains 3 000 steps and reports the step-2500
> checkpoint. There is **no learning-rate scheduler anywhere in the codebase**, so the step-2500 weights
> are bitwise identical to what stopping at 2 500 would have produced.

### B. Score comparison vs the paper

| Metric (paper's own) | Paper (Table 1, Heston) | Ours — Heston (paper reimplementation, median 4 seeds) | Ours — Heston (benchmark, median 5 seeds) |
|----------------------|:-----------------------:|:------------------------------------------------------:|:------------------------------------------:|
| SWD ↓ | 0.062 | **0.0737** | **0.0720** |
| RV W₁ ↓ | 0.014 | **0.0128** | **0.0129** |
| \|r\| ACF ↓ | 0.016 | **0.0132** | **0.0128** |
| Early-future ↓ | 0.018 | **0.0178** | **0.0177** |
| MDD W₁ ↓ | 0.022 | **0.0232** | **0.0246** |
| **mean ratio (ours / paper)** | 1.000 | **0.994** | **1.006** |

Tolerance: within 25% of the paper value **or** 0.005 absolute, whichever is wider.
**5/5 metrics within tolerance on both columns.** Adding seed 2 moves the mean ratio from 0.994 to
1.006 — i.e. the reproduction is indistinguishable from the published row either way, and the four-seed
result was not a lucky seed selection.

Per-seed values behind column 3:

| Seed | SWD | RV W₁ | \|r\| ACF | Early-future | MDD W₁ | in paper? |
|------|------:|------:|----------:|-------------:|-------:|:---------:|
| 0 | 0.0528 | 0.0210 | 0.0230 | 0.0180 | 0.0103 | yes |
| 1 | 0.0655 | 0.0127 | 0.0128 | 0.0108 | 0.0218 | yes |
| 2 | 0.0720 | 0.0153 | 0.0094 | 0.0079 | 0.0296 | **no** (benchmark only) |
| 3 | 0.0818 | 0.0129 | 0.0064 | 0.0413 | 0.0423 | yes |
| 4 | 0.0839 | 0.0112 | 0.0135 | 0.0177 | 0.0246 | yes |
| **median (5)** | **0.0720** | **0.0129** | **0.0128** | **0.0177** | **0.0246** | |

### C. The Reference row, and why it is not scored

The paper's published Table 1 Reference row was generated with the `local_gaussian` reference (a
per-step ridge on the raw complete prefix). Paper §2.1 instead **describes** the price-only
Guyon–Lekeufack model fitted by Gaussian quasi-likelihood on 80% of the training paths, and the
author confirmed Guyon–Lekeufack is the correct reference. This reimplementation therefore runs
Guyon–Lekeufack everywhere, which is a materially stronger starting point:

| Row | Metric | Ours (GL, median 4 seeds) | Paper (published, `local_gaussian`) | Δ |
|-----|--------|--------------------------:|------------------------------------:|------:|
| Reference | SWD | 0.0381 | 0.060 | −0.0219 |
| Reference | RV W₁ | 0.0566 | 0.089 | −0.0324 |
| Reference | \|r\| ACF | 0.0484 | 0.068 | −0.0196 |
| Reference | Early-future | 0.1019 | 0.214 | −0.1121 |
| Reference | MDD W₁ | 0.0287 | 0.059 | −0.0303 |

Because this gap is a **deliberate, documented substitution**, the Reference row is reported for
context and **excluded from the pass count**. Only the Deep-MKV-TS row is scored against the paper.

### D. Hyperparameter search — a config that beats the paper exists

A 14-trial search was run on a **separate `val` / `valdisc` split pair** that neither `test` nor `disc`
overlaps. Score = mean over the five paper columns of ours/paper; below 1.000 beats the published row.

| Rank | Trial | SWD | RV W₁ | \|r\| ACF | Early-future | MDD W₁ | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `lambda100_kappa200_absacf0.50` | 0.0457 | 0.0139 | 0.0142 | 0.0052 | 0.0153 | **0.722** |
| 2 | `kappa200_absacf0.50` | 0.0433 | 0.0169 | 0.0173 | 0.0120 | 0.0128 | 0.848 |
| 3 | `jointvol1.0` | 0.0456 | 0.0228 | 0.0147 | 0.0126 | 0.0147 | 0.930 |
| … | | | | | | | |
| 11 | `baseline` (paper config) | 0.0420 | 0.0196 | 0.0217 | 0.0254 | 0.0115 | 1.074 |

Full ranking: [`runs/hpsearch/RANKING.md`](../../../methods/Deep-MKV-TS/paper_reimplementation/runs/hpsearch/RANKING.md).

> **The benchmark numbers in this file use the *paper* configuration, not the search winner.** The
> benchmark's job is to place the *published* method against the other generators, so the reported
> 5-seed run is the paper config (λ=50, κ=100, |r| ACF weight 0.25 — rank 11 on `valdisc`). The search
> result is recorded because it answers "does a config beating the paper exist?" — it does, by a wide
> margin on the validation split — but swapping it in would mean benchmarking a tuned variant against
> untuned baselines, which is not a comparison worth publishing.

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector
(63 log-returns + terminal cumulative return + realized volatility, z-scored on the generated pool),
retrieve the K = 77 nearest Deep-MKV-TS paths by L2 in z-scored space, forecast with their
**price-anchored** futures (steps 64–127). CRPS is scored against the test set at two horizons; the
naive random-walk (RW) baseline is 3.738 (H=32) / 5.246 (H=64).
Full analysis: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline | Perfect floor |
|--------|--------------------|-------------|---------------|
| PS-MC CRPS H=32 ↓ | **2.696 ± 0.004060** | 3.738 | 2.721 ± 0.004183 |
| PS-MC CRPS H=64 ↓ | **3.758 ± 0.004762** | 5.246 | 3.788 ± 0.006463 |

PS-MC over the Deep-MKV-TS pool **beats the naive RW on CRPS** at both horizons on all 5 seeds, and
**wins both PS-MC contests** across the benchmark (next best LS4: 2.704 / 3.763).

> **Stated plainly:** Deep-MKV-TS is the only method whose PS-MC CRPS lands *at or below* the
> Perfect-Recovery floor at both horizons. Path shadowing is a **retrieval** task and rewards a pool
> whose members hug the conditional mean; a mildly under-dispersed pool — which A5 and A28 independently
> establish this one to be — retrieves tighter, lower-CRPS ensembles than an honest independent draw.
> The 2/2 win is real under the benchmark's rules, but it is at least partly the same variance deficit
> that costs the method every B row.

Heston is time-homogeneous, so the uniform and Gaussian prefix weightings coincide to four decimals.

---

## Files

| Artifact | Path |
|----------|------|
| All A + B metrics (mean/std + per-seed) | `metrics_summary.csv` |
| Per-seed raw metric dumps | `seed_{0..4}_metrics.json` |
| B five-subline aggregates (MSE + % err + NRMSE + CVaR₉₀ + CVaR₉₅) | `curve_b_aggregate.json` |
| grid_tvd 50×50 aggregate | `grid_tvd_aggregate.json` |
| Classifier / predictor loss curves | `seed_{i}_{disc,pred}_{gru,mlp}_loss.csv` |
| Stylised-facts 8-panel diagnostic | `plots/heston_diagnostics.png` |
| A18 / A19 judge training curves | `plots/{disc_classifier_loss,pred_score_loss}.png` |
| PCA / t-SNE embeddings per seed | `plots/seed_{i}_{pca,tsne}.png` |
| Path-shadowing MC forecasts | `path_shadowing/` |
| Generated price paths (8192×128) | `../../../methods/Deep-MKV-TS/generated_paths/seed_{i}/generated_paths_8192x128.npy` |
| Model checkpoints + hyperparameter records | `../../../methods/Deep-MKV-TS/weights/seed_{i}_{model.pt,config.json}` |
| Training-loss CSVs + convergence plot | `../../../methods/Deep-MKV-TS/losses/` |
| Paper reproduction table | `../../../methods/Deep-MKV-TS/paper_reimplementation/results/PAPER_VS_OURS.md` |
| Frozen paper protocol | `../../../methods/Deep-MKV-TS/paper_reimplementation/metric/PROTOCOL.json` |

→ Method write-up: [`methods/Deep-MKV-TS/README.md`](../../../methods/Deep-MKV-TS/README.md)
→ Cross-method comparison with every generator: [`results/README.md`](../../README.md)
