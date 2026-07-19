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
| A10 | Kurtosis Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A17 | \|r\| q95 Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A18 | \|r\| q99 Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A30 | Tail QQ Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A34 | Hill Tail Index Error | Fat Tail | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Distribution —** | | | | | | | | |
| A1  | Path MMD²      | Distribution | ↓ | 0.001455 ± 0.000118 | 0.001322 | 0.001356 | 0.001648 | 0.001429 | 0.001521 |
| A2  | Terminal MMD²  | Distribution | ↓ | 0.001619 ± 0.000407 | 0.001181 | 0.001584 | 0.001899 | 0.002234 | 0.001196 |
| A3  | Increment MMD² | Distribution | ↓ | 0.000745 ± 0.000034 | 0.000794 | 0.000727 | 0.000708 | 0.000777 | 0.000719 |
| A4  | Volatility MMD | Distribution | ↓ | 0.007074 ± 0.000512 | 0.006556 | 0.006688 | 0.007775 | 0.007612 | 0.006739 |
| A5  | Terminal SWD   | Distribution | ↓ | 0.6873 ± 0.1495 | 0.4733 | 0.6583 | 0.9420 | 0.6914 | 0.6713 |
| A6  | Path SWD       | Distribution | ↓ | 0.4381 ± 0.0533 | 0.3642 | 0.4510 | 0.5138 | 0.3937 | 0.4677 |
| A24 | RV Law Loss    | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A25 | Mean Path RMSE | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A27 | KS on Log-returns | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A28 | Skewness Error | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A29 | QQ RMSE (300-pt) | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A33 | Terminal Price KS | Distribution | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Adversarial —** | | | | | | | | |
| A13 | Disc Score GRU (log-ret.) | Adversarial | ↓ | 0.00797 ± 0.00564 | 0.00076 | 0.00656 | 0.00351 | 0.01419 | 0.01480 |
| A13 | Disc Score MLP (log-ret.) | Adversarial | ↓ | 0.01004 ± 0.00634 | 0.02029 | 0.00381 | 0.00412 | 0.01419 | 0.00778 |
| | **— Predictive —** | | | | | | | | |
| A14 | Pred Score GRU — TSTR | Predictive | ↓ | 0.05372 ± 0.00003 | 0.05371 | 0.05370 | 0.05377 | 0.05370 | 0.05371 |
| A14 | Pred Score MLP — TSTR | Predictive | ↓ | 0.05374 ± 0.00010 | 0.05362 | 0.05382 | 0.05365 | 0.05371 | 0.05389 |
| | **— Temporal —** | | | | | | | | |
| A7  | Cov Error (%)  | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A11 | ACF Error \|log-returns\| | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A12 | ACF Error log-returns² | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A22 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A23 | ACF r² Lag-1 Error | Temporal | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Vol —** | | | | | | | | |
| A8  | Mean RMSE      | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A9  | Return Std Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A16 | Log-Return Std Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A19 | Kurtosis Ratio (target/model) | Vol | — | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| A20 | Sigma Mean Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A26 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A31 | Rolling Vol KS (window=5) | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **— Heston Spec —** | | | | | | | | |
| A15 | Sigma Corr (vol recovery) | Heston Spec | ↑ | 0.6143 ± 0.0000 | 0.6143 | 0.6143 | 0.6143 | 0.6143 | 0.6143 |
| A15 | Sigma RMSE     | Heston Spec | ↓ | 0.0654 ± 0.0000 | 0.0654 | 0.0654 | 0.0654 | 0.0654 | 0.0654 |
| A21 | Learned/Oracle Sigma Corr | Heston Spec | ↑ | 0.6143 ± 0.0000 | 0.6143 | 0.6143 | 0.6143 | 0.6143 | 0.6143 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction.
> **A1–A6** are the only distribution metrics with a non-zero floor: they are estimated on two
> disjoint 1024-path subsamples, so the value (~0.001–0.7) is Monte-Carlo sampling noise, not model error.
> **A15/A21 = 0.614** is the *ceiling* of Heston teacher-σ recovery on this dataset: even the true
> variance path only correlates 0.61 with the σ estimated from prices — so no method can exceed this.
> **A13/A14** are non-zero because a shuffled pool is statistically indistinguishable from the real
> data, so the classifier/predictor operate at their intrinsic noise floor (score ≈ 0.008 / 0.054).

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
