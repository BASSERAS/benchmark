# Perfect Recovery — Independent-Draw Floor

The **perfect-recovery floor** is the score a *perfect* generator would obtain: one whose output
is distributionally identical to the real Heston process. We realise it honestly by drawing a
**fresh, independent set of 8192 Heston paths** — same parameters, brand-new random seeds
(`seed = 1000 + i` for `i = 0…4`) — and scoring that draw against the **test set** with the exact
same metric code every method uses (`metrics/compute_perfect_recovery.py`).

Because the two datasets come from the *same law* but *different randomness*, every metric lands on
its **irreducible finite-sample noise floor** — the value that survives even when the generated
distribution is genuinely perfect. This floor is **non-zero everywhere**: two independent 8192-path
Heston samples never produce identical histograms, ACFs, quantiles or moments. Any real method's
score should be read relative to this floor, not relative to 0.

> **This floor is not row-shuffling and not a permutation of the real data.** It is an independent
> re-simulation of the Heston SDE. A permutation would preserve every column-wise statistic exactly
> and collapse most metrics to 0 — which would be a misleading target. The independent draw is the
> honest lower bound a real generator can hope to reach.

> **It is not method-specific.** The floor depends only on the test set + the Heston law + the
> protocol, so it is identical whether you compare it to TimeGAN, SBTS, LS4 or any future method.
> That is why the same `Perfect floor` column appears in every per-method result README.

---

## Split protocol (test-set everywhere)

| Role | Data | Used for |
|------|------|----------|
| **real reference** | test set — 8192 paths, `seed = 1` | A1–A17, A20–A34, all B curves, PS-MC |
| **judge-real class** | disc set — 8192 paths, `seed = 2` | A18 discriminative "real", A19 predictive-TSTR "real" |
| **generated** | independent Heston draw, `seed = 1000 + i` | the "fake" side scored against the two sets above |

Generators are trained on `seed = 0` and never scored; the test set (`seed = 1`) is the sole real
reference; the disc set (`seed = 2`) supplies the adversary's "real" class so A18/A19 never touch the
test set. The perfect-recovery floor follows the identical protocol, substituting an independent
Heston draw for the generated paths.

---

## Why the floor is non-zero everywhere

`compute_one_seed` scores an independent draw against the test set, so **no** metric can hit 0:

| Metric group | Why the floor is non-zero |
|--------------|---------------------------|
| **A1–A17** (moments, MMD, SWD, KS, tails) | two independent 8192-path samples differ by pure Monte-Carlo sampling noise |
| **A18 / A19** (learned disc / predictor) | an independent draw is statistically indistinguishable from the disc set, so the classifier/predictor sit at their intrinsic noise floor (≈ 0.006 / ≈ 0.050) |
| **A20–A32** (covariance, ACF, vol) | finite-sample ACF and covariance estimates never coincide across two draws |
| **A28** (kurtosis ratio → 1) | lands at **1.006**, not exactly 1 — finite-sample kurtosis noise |
| **A33 / A34** (Heston teacher-σ) | **0.616 / 0.0656** — this is the *ceiling* of σ-recovery on this dataset: even the true variance path only correlates 0.62 with the σ estimated from prices, so no method can exceed it |
| **B** (curve-shape) | two independent draws never produce identical stylised-fact curves |

So every number below is the sampling/estimator noise you cannot train away — the honest target a
perfect generator would asymptote to.

---

## Metrics A1–A34 — mean ± std across 5 independent-draw seeds

<!-- ===== PERFECT-RECOVERY A TABLE ===== -->
| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|-----------|--------|--------|--------|--------|--------|
| **— Fat Tail —** | | | | | | |
| A1 Kurtosis Error ↓ | 0.008092 ± 0.006811 | 0.01176 | 0.004981 | 5.99e-04 | 0.003541 | 0.01958 |
| A2 \|r\| q95 Error ↓ | 6.57e-05 ± 5.96e-05 | 1.43e-04 | 9.59e-06 | 1.33e-04 | 2.13e-05 | 2.08e-05 |
| A3 \|r\| q99 Error ↓ | 5.98e-05 ± 3.25e-05 | 8.50e-05 | 1.36e-05 | 4.87e-05 | 4.55e-05 | 1.06e-04 |
| A4 Tail QQ Error ↓ | 6.75e-05 ± 3.70e-05 | 1.25e-04 | 2.62e-05 | 9.39e-05 | 5.57e-05 | 3.65e-05 |
| A5 Hill Tail Index Error ↓ | 0.5266 ± 0.5572 | 0.001467 | 1.605 | 0.3887 | 0.2574 | 0.3800 |
| **— Distribution —** | | | | | | |
| A6 Path MMD² ↓ | 0.001842 ± 2.55e-04 | 0.002264 | 0.001629 | 0.001832 | 0.001948 | 0.001540 |
| A7 Terminal MMD² ↓ | 0.001983 ± 8.89e-04 | 0.003007 | 0.001493 | 0.001179 | 0.003112 | 0.001122 |
| A8 Increment MMD² ↓ | 8.69e-04 ± 2.70e-05 | 9.10e-04 | 8.59e-04 | 8.37e-04 | 8.90e-04 | 8.48e-04 |
| A9 Volatility MMD ↓ | 0.008554 ± 0.001549 | 0.01015 | 0.007398 | 0.007333 | 0.01072 | 0.007167 |
| A10 Terminal SWD ↓ | 1.151 ± 0.4868 | 2.007 | 1.380 | 0.7720 | 0.8616 | 0.7348 |
| A11 Path SWD ↓ | 0.6191 ± 0.1960 | 0.9973 | 0.6002 | 0.5601 | 0.4630 | 0.4748 |
| A12 RV Law Loss ↓ | 0.05202 ± 0.006560 | 0.06191 | 0.04840 | 0.05066 | 0.05629 | 0.04284 |
| A13 Mean Path RMSE ↓ | 0.1205 ± 0.05175 | 0.07927 | 0.08428 | 0.2200 | 0.1206 | 0.09857 |
| A14 KS Log-returns ↓ | 0.001491 ± 5.79e-04 | 0.002405 | 9.62e-04 | 0.001936 | 9.83e-04 | 0.001168 |
| A15 Skewness Error ↓ | 0.005274 ± 0.001459 | 0.003912 | 0.006696 | 0.004703 | 0.003752 | 0.007307 |
| A16 QQ RMSE (300-pt) ↓ | 4.19e-05 ± 1.89e-05 | 7.25e-05 | 2.50e-05 | 5.51e-05 | 3.22e-05 | 2.48e-05 |
| A17 Terminal Price KS ↓ | 0.01099 ± 0.001563 | 0.01208 | 0.01196 | 0.01184 | 0.007935 | 0.01111 |
| **— Adversarial —** | | | | | | |
| A18 Disc Score GRU ↓ | 0.006195 ± 0.007171 | 0.01968 | 0.007476 | 0.001984 | 7.63e-04 | 0.001068 |
| A18 Disc Score MLP ↓ | 0.005951 ± 0.003469 | 0.006866 | 0.004730 | 0.005340 | 0.001068 | 0.01175 |
| **— Predictive —** | | | | | | |
| A19 Pred Score GRU ↓ | 0.05002 ± 1.08e-05 | 0.05002 | 0.05001 | 0.05001 | 0.05004 | 0.05001 |
| A19 Pred Score MLP ↓ | 0.05036 ± 6.63e-04 | 0.04990 | 0.05042 | 0.04990 | 0.04997 | 0.05163 |
| **— Temporal —** | | | | | | |
| A20 Covariance Error ↓ | 4.923 ± 3.284 | 7.509 | 4.925 | 1.776 | 0.8926 | 9.514 |
| A21 ACF \|r\| Error (lags) ↓ | 0.002234 ± 6.62e-04 | 0.001668 | 0.002936 | 0.003093 | 0.002010 | 0.001464 |
| A22 ACF r² Error (lags) ↓ | 0.002206 ± 6.32e-04 | 0.001293 | 0.002934 | 0.002854 | 0.002207 | 0.001745 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.002652 ± 0.001035 | 0.002203 | 0.002454 | 0.004686 | 0.002048 | 0.001870 |
| A24 ACF r² Lag-1 Error ↓ | 0.002790 ± 9.39e-04 | 0.001913 | 0.002253 | 0.004582 | 0.002786 | 0.002417 |
| **— Vol —** | | | | | | |
| A25 Mean RMSE ↓ | 0.1392 ± 0.06359 | 0.1726 | 0.01826 | 0.1998 | 0.1379 | 0.1677 |
| A26 Return Std Error ↓ | 0.002523 ± 0.001767 | 0.005739 | 0.001455 | 0.002377 | 4.96e-04 | 0.002548 |
| A27 Log-Return Std Error ↓ | 3.15e-05 ± 2.48e-05 | 6.84e-05 | 1.83e-05 | 5.34e-05 | 1.13e-05 | 6.04e-06 |
| A28 Kurtosis Ratio (→ 1) | 1.006 ± 0.009834 | 0.9990 | 1.022 | 1.006 | 1.009 | 0.9933 |
| A29 Sigma Mean Error ↓ | 4.96e-04 ± 4.24e-04 | 0.001112 | 4.06e-04 | 8.47e-04 | 3.63e-06 | 1.13e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.1432 ± 0.03018 | 0.1593 | 0.1481 | 0.1350 | 0.09138 | 0.1822 |
| A31 Rolling Vol KS (w=5) ↓ | 0.003814 ± 0.001210 | 0.005462 | 0.003457 | 0.004893 | 0.003135 | 0.002124 |
| A32 Vol-of-Vol Error ↓ | 1.54e-05 ± 9.93e-06 | 1.10e-05 | 1.46e-05 | 2.20e-06 | 3.27e-05 | 1.65e-05 |
| **— Heston Spec —** | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.6163 ± 0.002371 | 0.6139 | 0.6197 | 0.6157 | 0.6183 | 0.6138 |
| A34 Teacher-Sigma RMSE ↓ | 0.06559 ± 1.37e-04 | 0.06551 | 0.06542 | 0.06554 | 0.06571 | 0.06579 |

> **Convention:** ↓ lower is better; ↑ higher is better; A28 targets 1.
> **A5 Hill index (0.53 ± 0.56)** is intrinsically high-variance: the Hill estimator on a tail of
> only a few hundred exceedances swings widely between draws (seed 1 alone hits 1.6). Read A5 as
> "order 0.5", not a tight target.
> **A18 / A19** are non-zero because an independent draw is indistinguishable from the disc set, so
> the learned classifier/predictor operate at their intrinsic noise floor (≈ 0.006 / ≈ 0.050).
> **A33 = 0.616** is the *ceiling* of Heston teacher-σ recovery on this dataset: even the true
> variance path only correlates 0.62 with the σ estimated from prices, so no method can exceed it.

---

## B — Curve-Shape Metrics — mean ± std across 5 independent-draw seeds

Each stylised-fact curve is summarised by three sublines, all mean-of-three over
{function, 1st derivative, 2nd derivative} unless noted:

- **MSE** — `mean((L_gen − L_real)²)`, the winner-deciding number.
- **% err** — scale-aware ε-floor MAPE, `mean(|L_gen − L_real| / (|L_real| + ε)) × 100`, `ε = 1e-3·(max|L_real| + 1e-12)`.
- **NRMSE** — `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100`.

> **Tail survival** uses **function-only** for % err and NRMSE (its derivatives are near-zero, so the
> ε-floor MAPE explodes); MSE stays mean-of-three.

<!-- ===== PERFECT-RECOVERY B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|------|---------|-----------|--------|--------|--------|--------|--------|
| **Log-return histogram** | MSE | 0.1098 ± 0.02492 | 0.08507 | 0.1442 | 0.08047 | 0.1084 | 0.1310 |
|  | % err | 290.3% ± 140.6% | 258.8% | 463.5% | 129.7% | 444.5% | 155.0% |
|  | NRMSE | 17.81% ± 2.281% | 15.38% | 20.82% | 15.13% | 17.93% | 19.78% |
| **QQ plot** | MSE | 1.09e-09 ± 6.13e-10 | 2.21e-09 | 5.44e-10 | 1.27e-09 | 8.19e-10 | 6.31e-10 |
|  | % err | 16.51% ± 0.9368% | 15.49% | 17.48% | 17.79% | 15.87% | 15.92% |
|  | NRMSE | 0.3436% ± 0.03688% | 0.4054% | 0.2970% | 0.3291% | 0.3606% | 0.3260% |
| **ACF \|r\| lags 1–20** | MSE | 9.61e-06 ± 3.40e-06 | 1.16e-05 | 1.43e-05 | 1.07e-05 | 5.22e-06 | 6.23e-06 |
|  | % err | 114.3% ± 22.37% | 109.5% | 139.5% | 139.8% | 82.82% | 100.1% |
|  | NRMSE | 43.89% ± 8.375% | 50.95% | 54.14% | 45.71% | 32.10% | 36.57% |
| **ACF r² lags 1–20** | MSE | 9.17e-06 ± 3.08e-06 | 9.60e-06 | 1.45e-05 | 9.54e-06 | 5.74e-06 | 6.49e-06 |
|  | % err | 381.5% ± 102.0% | 483.2% | 482.8% | 422.4% | 261.8% | 257.4% |
|  | NRMSE | 34.19% ± 5.580% | 36.30% | 43.16% | 34.71% | 27.15% | 29.61% |
| **Rolling vol histogram** | MSE | 1.372 ± 0.07269 | 1.247 | 1.364 | 1.470 | 1.402 | 1.377 |
|  | % err | 127.9% ± 9.845% | 124.4% | 141.4% | 137.3% | 120.6% | 115.8% |
|  | NRMSE | 16.66% ± 0.4699% | 15.79% | 16.75% | 17.22% | 16.73% | 16.82% |
| **Tail survival** | MSE | 5.22e-07 ± 5.50e-07 | 1.50e-06 | 1.78e-07 | 7.66e-07 | 9.10e-08 | 7.67e-08 |
|  | % err | 0.3256% ± 0.2149% | 0.6631% | 0.1770% | 0.4966% | 0.1677% | 0.1237% |
|  | NRMSE | 0.1050% ± 0.06651% | 0.2129% | 0.07004% | 0.1513% | 0.04806% | 0.04275% |

> **Why % err stays large even for a perfect draw.** The ε-floor MAPE divides by curve values that
> are near zero over most of the support (histogram tails, high-lag ACF), so even sub-percent
> absolute error inflates to hundreds of percent. This is a property of the metric, not the draw —
> which is exactly why % err is a *context* subline and **MSE decides the winner**. Read a method's
> % err against *this* floor (e.g. ACF r² floor ≈ 382%), never against 0%.

---

## File layout

```
methods/perfect_recovery/
├── README.md                    ← this file
├── perfect_recovery.json        legacy aggregate summary
├── dataset/
│   └── seeds/                    materialised independent Heston draws (seed 1000+i)
└── results/
    ├── seed_{0..4}_metrics.json  per-seed A1–A34 + B (independent draw vs test set)
    ├── metrics_summary.csv       mean ± std + per-seed for every metric
    └── curve_b_aggregate.json    B three-subline aggregate (MSE + % err + NRMSE, per-seed)
```

## Reproduce

```bash
cd /home/tbasseras/benchmark
# Full run (A18/A19 learned metrics need a GPU)
CUDA_VISIBLE_DEVICES=0 \
    /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --seeds 5

# Fast run without the learned metrics (no GPU)
/home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --no-pytorch
```

Every number in the two tables above is emitted from the on-disk seed JSONs by
`metrics/render_tables.py`'s formatter, so the `Perfect floor` column in each per-method result
README matches these means exactly.
