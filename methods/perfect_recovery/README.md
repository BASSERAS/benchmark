# Perfect Recovery — Row-Shuffle Baseline

The **perfect-recovery floor** is the score an *ideal* generator would obtain: one whose output
is distributionally identical to the real Heston dataset. We realise it by **row-shuffling** the
real dataset — for each seed `i`, `S_gen = S_real[rng.permutation(N)]` with
`rng = np.random.default_rng(i)` — and scoring that shuffle against the real data with the exact
same metric code every method uses (`metrics/compute_perfect_recovery.py`).

Because a row-shuffle leaves every column-wise statistic untouched (each Heston path is i.i.d.),
this baseline tells us the **irreducible floor** of each metric: the value that survives even when
the generated distribution is *perfect*. Any real method's score should be read relative to this
floor, not relative to 0.

> **This is not method-specific.** The floor depends only on the real Heston dataset and the metric
> definitions, so it is identical whether you compare it to TimeGAN, SBTS, or any future method.

---

## Why most A-metrics are exactly 0

`compute_one_seed` evaluates two regimes:

| Metric group | Compared on | Floor behaviour |
|--------------|-------------|-----------------|
| **A1–A6** (MMD / SWD) | two **disjoint** 1024-path subsamples (real vs shuffled) | **non-zero** — pure Monte-Carlo sampling noise between two finite samples |
| **A7–A12, A16–A34** (moments, ACF, KS, tails) | **full** dataset, real vs shuffled | **exactly 0** — a permutation preserves every per-timestep moment, ACF and quantile |
| **A13 / A14** (learned classifier / predictor) | full log-returns | **small non-zero** — irreducible classifier/predictor noise floor |
| **A15 / A21** (Heston teacher-σ) | shuffled S with its matching variance rows | **0.614** — oracle σ-recovery ceiling on this dataset |
| **A19** (kurtosis ratio, → 1) | full dataset | **1.0** exactly |
| **B** (curve-shape) | full dataset | **0** for all six plots |

So a `0.0000` below means "provably at the theoretical floor", while the non-zero A1–A6 / A13 / A14
values are the sampling/estimator noise you cannot train away.

---

## Metrics A1–A34 + B — mean ± std across 5 shuffle seeds

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|
| | **— Fat Tail —** | | | | | | | | |
| A1 | Kurtosis Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A4 | Tail QQ Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Distribution —** | | | | | | | | |
| A6  | Path MMD²      | Distribution | ↓ | 0.001455 ± 0.000118 | 0.001322 | 0.001356 | 0.001648 | 0.001429 | 0.001521 |
| A7  | Terminal MMD²  | Distribution | ↓ | 0.001619 ± 0.000407 | 0.001181 | 0.001584 | 0.001899 | 0.002234 | 0.001196 |
| A8  | Increment MMD² | Distribution | ↓ | 0.000745 ± 0.000034 | 0.000794 | 0.000727 | 0.000708 | 0.000777 | 0.000719 |
| A9  | Volatility MMD | Distribution | ↓ | 0.007074 ± 0.000512 | 0.006556 | 0.006688 | 0.007775 | 0.007612 | 0.006739 |
| A10 | Terminal SWD   | Distribution | ↓ | 0.687260 ± 0.149519 | 0.473316 | 0.658255 | 0.942002 | 0.691434 | 0.671294 |
| A11 | Path SWD       | Distribution | ↓ | 0.438076 ± 0.053291 | 0.364185 | 0.450953 | 0.513768 | 0.393745 | 0.467732 |
| A12 | RV Law Loss    | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A13 | Mean Path RMSE | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A14 | KS on Log-returns | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A15 | Skewness Error | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A17 | Terminal Price KS | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Adversarial —** | | | | | | | | |
| A18 GRU | Disc Score GRU (log-ret.) | Adversarial | ↓ | 0.007110 ± 0.005782 | 0.001373 | 0.016631 | 0.008392 | 0.000763 | 0.008392 |
| A18 MLP | Disc Score MLP (log-ret.) | Adversarial | ↓ | 0.003326 ± 0.002430 | 0.000458 | 0.003509 | 0.003814 | 0.001373 | 0.007476 |
| | **— Predictive —** | | | | | | | | |
| A19 GRU | Pred Score GRU — TSTR | Predictive | ↓ | 0.053733 ± 0.000026 | 0.053706 | 0.053760 | 0.053703 | 0.053733 | 0.053765 |
| A19 MLP | Pred Score MLP — TSTR | Predictive | ↓ | 0.053707 ± 0.000121 | 0.053621 | 0.053918 | 0.053769 | 0.053613 | 0.053613 |
| | **— Temporal —** | | | | | | | | |
| A20 | Covariance Error | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A21 | ACF Error (abs returns) | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A22 | ACF Error (sq returns) | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Vol —** | | | | | | | | |
| A25 | Mean RMSE      | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A26 | Return Std Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A27 | Log-Return Std Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A28 | Kurtosis Ratio (target/model) | Vol | — | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| A29 | Sigma Mean Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A31 | Rolling Vol KS (window=5) | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Heston Spec —** | | | | | | | | |
| A33 | Teacher-Sigma Correlation | Heston Spec | ↑ | 0.614269 ± 0.0000 | 0.614269 | 0.614269 | 0.614269 | 0.614269 | 0.614269 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | 0.065445 ± 0.0000 | 0.065445 | 0.065445 | 0.065445 | 0.065445 | 0.065445 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction.
> **A6–A11** are the only distribution metrics with a non-zero floor: they are estimated on two
> disjoint 1024-path subsamples, so the value (~0.001–0.7) is Monte-Carlo sampling noise, not model error.
> **A33 = 0.614** is the *ceiling* of Heston teacher-σ recovery on this dataset: even the true
> variance path only correlates 0.61 with the σ estimated from prices — so no method can exceed this.
> **A18/A19** are non-zero because a shuffled pool is statistically indistinguishable from the real
> data, so the classifier/predictor operate at their intrinsic noise floor (score ≈ 0.005 / 0.054).

---

## B — Curve-Shape Metrics — mean ± std across 5 shuffle seeds

All six stylised-fact curves are identical under a row-shuffle, so **every B score is exactly 0**
(MSE and % error). This is the B floor referenced by every method README.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|------|---------|-----------|--------|--------|--------|--------|--------|
| **Log-return histogram** | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |
| **QQ plot**              | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |
| **ACF \|r\|**            | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |
| **ACF r²**               | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |
| **Rolling vol hist.**    | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |
| **Tail survival**        | MSE   | 0.0 ± 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|                          | % err | 0% ± 0%   | 0% | 0% | 0% | 0% | 0% |

> The ACF |r| and ACF r² MSE are literally ~1e-34 (floating-point round-off), rounded to 0 above.

---

## File layout

```
methods/perfect_recovery/
├── README.md                    ← this file
├── perfect_recovery.json        aggregate summary (all metrics, all seeds)
└── results/
    ├── seed_{0..4}_metrics.json  per-seed A1–A34 + B (36 keys)
    ├── metrics_summary.csv       mean ± std + per-seed for every metric
    └── curve_b_aggregate.json    B two-subline aggregate (MSE + % err)
```

## Reproduce

```bash
cd /home/tbasseras/benchmark
# Full run (A13/A14 need a GPU)
CUDA_VISIBLE_DEVICES=0 \
    /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --seeds 5

# Fast run without the learned metrics (no GPU)
/home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --no-pytorch
```
