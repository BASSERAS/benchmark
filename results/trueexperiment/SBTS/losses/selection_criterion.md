# SBTS selection criterion — dataset/TrueDataset

**Pre-registered 2026-08-26, before the (h, K) grid was run.** At the time of writing the only
rows in hand were h ∈ {0.08, 0.10} from the K = 20 sweep plus one earlier h = 0.31 probe. Every
threshold below is derived from the dataset's own split structure, not from any generator output.

The Heston criterion (`results/HestonMultiAsset/SBTS/losses/selection_criterion.md`) remains
authoritative **for the Heston panel**. It is replaced here, and the replacement is not a
loosening — it is a correction of a criterion that is *structurally unsatisfiable* on real data.

---

## Why the Heston criterion cannot be reused

The Heston constraints are absolute tolerances against a reference split:

> vol error < 5 %, excess-kurtosis error < 20 %

They are well posed on `dataset/HestonMultiAsset` because its five splits are **five independent
draws from one known SDE**. Train and val agree to within sampling noise, so the tolerance measures
generator error and nothing else.

There is only **one** history of the real market. The five TrueDataset splits are carved out of it
by interleaved blocks, so they differ by genuine regime variation, not just sampling noise.
Measured over the committed build:

| pair (scored against `val`) | vol err | excess-kurtosis err |
|---|---:|---:|
| train vs val | **7.05 %** | **33.10 %** |
| test vs val | 9.84 % | 348.26 % |
| disc vs val | 8.47 % | 1012.81 % |
| valdisc vs val | 11.73 % | 53.84 % |

**A perfect generator — one that emits the training distribution exactly — scores 7.05 % / 33.10 %
and is rejected by both Heston constraints.** The criterion does not merely fail to be met; it
cannot be met by anything, including the data itself. Applying it unchanged and reporting
"SBTS fails on real data" would be an artefact of the criterion, not a result about SBTS.

---

## The replacement: a real-vs-real envelope, zero free parameters

Every threshold is the extreme value attained by **genuine market data scored against genuine
market data**, over all 20 ordered pairs of the five splits. The question each constraint asks is:

> is the generated sample, on this statistic, distinguishable from another honest slice of
> the same market?

| statistic | min | median | **MAX = the ceiling** |
|---|---:|---:|---:|
| vol err (%) | 7.05 | 11.31 | **18.06** |
| excess-kurtosis err (%) | 29.70 | 70.46 | 1012.81 |
| cross-asset corr err | 0.0241 | 0.0598 | **0.1173** |

### Hard constraints

1. **No collapse.** `min(S_gen) > 10`, all prices finite and strictly positive.
   *(Carried over unchanged. This is a sanity check, not a tolerance.)*
2. **Volatility.** `vol_err ≤ 18.06 %` — the worst real-vs-real pair.
3. **Cross-asset correlation.** `corr_err ≤ 0.1173` — the worst real-vs-real pair.
   *(Promoted from tie-break to constraint: this is the multi-asset benchmark, and the panel's
   0.573 mean pairwise correlation is the structure it exists to test.)*

### Excess kurtosis is NOT a constraint here

It is recorded and reported, but it gates nothing. Its real-vs-real spread is **29.7 % to
1012.8 %** — a factor of 34. A statistic whose value across two honest samples of the same market
varies by 34x carries no information about generator quality at this sample size. Excess kurtosis
is a fourth-moment functional dominated by a handful of extreme returns, and TrueDataset has a
12.9 % point mass at exactly zero return with pooled excess kurtosis 78.7. Using it as a gate
would be selecting on noise.

This is stated **before** the grid is run precisely so it cannot be mistaken for discarding an
inconvenient row after the fact.

---

## Objective among admissible rows

**Minimise `|log NNratio|`** — drive the nearest-neighbour ratio to 1.

```
NNratio = median NN(gen -> train) / median NN(val -> train)
```

in flattened log-return space. This is the memorisation diagnostic. Unlike vol and kurtosis, it
comes out **exceptionally well calibrated** on this dataset — across the four genuinely held-out
real splits it spans a +/-5 % band:

| | NNratio |
|---|---:|
| val -> train (denominator, by construction) | 1.0000 |
| valdisc -> train | 0.9801 |
| test -> train | 1.0310 |
| disc -> train | 1.0498 |

**Admissible band: 0.980 - 1.050.** A generated sample landing inside it is, on this diagnostic,
indistinguishable from held-out reality. `NNratio << 1` is memorisation — generated paths hug the
training set more tightly than real held-out data does. `NNratio >> 1` is the opposite failure:
paths that have left the data manifold entirely.

The Heston file says *maximise* NNratio. That was a proxy for "push toward 1" valid because every
Heston row sat below 1 and the vol/kurtosis constraints bounded it from above. With kurtosis
demoted, an unbounded maximise would reward off-manifold paths, so the objective is stated in the
form it always meant. **The two rules coincide whenever `NNratio < 1`.**

Tie-break (`|log NNratio|` within 0.02): lowest `corr_err`.

---

## Both knobs are swept

The Heston runs fixed `K = 20`, `N_pi = 50` at the SBTS author's values. On real data `K = 20` is
measurably pathological. Post-overflow-fix effective sample size `ESS = (sum w)^2 / sum w^2` out of
8192, zero-weight fraction 0.0000 everywhere:

| h \ K | 1 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| 0.16 | 3884 | 1433 | 557 | 245 | **2.91** |
| 0.22 | 5456 | 3084 | 1952 | 967 | 332 |
| 0.31 | 6663 | 4817 | 3785 | 2402 | 1231 |
| 0.45 | 7449 | 6330 | 5546 | 4233 | 2771 |

At `h = 0.16, K = 20` the drift is effectively conditioned on **three** training paths out of 8192.
That is the memorisation regime by construction. Holding `K` at the author's value would confound
"SBTS is unsuited to real data" with "this one hyperparameter was never re-tuned off Heston", so
`K` is swept alongside `h`. `N_pi = 50` stays fixed.

---

## Relation to the paper

`methods/Deep-MKV-TS/paper_reimplementation/results/paper_real_data_table4.json` records the
paper's own real-data results (arXiv:2608.19394v1, Table 4). Their absolute CRPS levels are **not**
transferable — they carry the units of 1-second ES/NQ/YM equity-index futures, against 30-second
crypto here. What transfers is the dimensionless comparison against the paper's two purely
historical bootstrap banks:

| | ES | NQ | YM |
|---|---:|---:|---:|
| SBTS RV / block-bootstrap RV | 0.808 | 0.916 | 0.946 |
| SBTS cum-return / block-bootstrap cum-return | 0.998 | 1.088 | 1.087 |

**SBTS does not dominate on real data in the paper either.** It beats the block bootstrap on
realised volatility for all three indices but loses to it on cumulative return for NQ and YM. So a
TrueDataset variant on which SBTS is merely *competitive with, not better than*, a bootstrap is
reproducing the paper's own real-data behaviour, and is evidence the dataset is sound rather than
evidence it is broken.

This is the "between the paper and the best absolute score" gate: the **absolute** score selects
`(h, K)` and the dataset variant via the envelope criterion above; the **paper** supplies the
sanity band that says how good SBTS is entitled to look on real data at all.

---

## Addendum 2026-08-26: measurement protocol for `vol_err`

**Added while the grid was still running and NO row had yet been admissible.** This changes how
`vol_err` is *measured*, not the threshold it is measured against. The ceiling stays 18.06 %.

Two rows at identical `(h, K) = (0.10, 20)` returned 17.97 % (`m_probe = 1024`) and 22.35 %
(`m_probe = 512`) — one side of the ceiling each. SBTS paths are simulated independently, so
`M_simu` is a Monte Carlo sample size, not a model parameter, and the two rows estimate the same
quantity. The gap is estimator noise.

Measured directly by resampling the real train split against `val`, 40 replications per size:

| m | mean vol_err | sd | observed range |
|---:|---:|---:|---|
| 512 | 7.71 % | **1.67 pp** | 4.33 – 13.36 |
| 1024 | 8.02 % | 1.25 pp | 5.22 – 11.20 |
| 4096 | 7.04 % | 0.65 pp | 5.48 – 8.75 |
| 8192 | 7.18 % | 0.42 pp | 6.26 – 8.43 |

Per-asset excess kurtosis is 32–172, which inflates `Var(s²)` by roughly `(κ+2)/n`. At `m = 512`
one distribution spans a 9 pp range. Deciding a 18.06 % cut on a single `m = 512` draw is selecting
on noise — precisely the reason excess kurtosis was demoted above.

**Protocol.** The grid at `m_probe = 512` is a **coarse screen only**; its `vol_err` column is
indicative and no admissibility verdict from it is final. Every candidate within **±3.3 pp**
(2 sd at m=512) of the ceiling is re-measured at `m_probe = 8192` over **3 seeds**, and
admissibility is decided on that re-measurement. `corr_err` and `NNratio` are re-measured in the
same runs.

The envelope ceilings are unaffected: they were computed from the full 8192-path splits, where the
estimator's sd is 0.42 pp.

---

## Addendum 2026-08-26 (b): split mode is decided per metric family, not globally

**Decision rule stated before the numbers were computed**, in the same spirit as the kurtosis
demotion above: build a chronological held-out-future variant, compute its real-vs-real envelope,
and adopt it for *everything* if its `vol_err` ceiling came in at or below 25 %; otherwise confine
it to the conditional metrics. The rule fired against adoption.

### The variant

`build_true_dataset.py --split-mode holdout-era`. Two eras separated by a 45.5-day embargo
(4 blocks x 256 windows x 128 bars x 30 s):

* **past era**, days 0-1263 (3.40 y): `train`, `val`, `valdisc` -- everything that fits a model
  or selects a hyperparameter;
* **future era**, days 1308-2037 (1.96 y): `disc`, `test` -- everything scored against.

Blocks are spread within each era, so every split covers its whole era rather than one contiguous
stretch of it. This is *not* the pre-existing `--split-mode chronological`, which deals the five
splits back-to-back and leaves `train` spanning only 1.00 y.

For reference, under `interleaved` the earliest `test` window precedes the latest `train` window by
1980 days. Interleaving is genuinely optimistic about leakage; that is the cost being weighed here.

### Measured envelopes

| | vol_err MAX | train vs test | corr MAX | NNratio band (real held-out) |
|---|---:|---:|---:|---|
| interleaved | **18.06 %** | 10.24 % | 0.1173 | 0.980 - 1.050 (+/-5 %) |
| holdout-era | **60.82 %** | 41.46 % | 0.1531 | 0.749 - 1.000 (+/-25 %) |

### Why holdout-era is rejected as the selection split

1. **A perfect generator scores 41.46 %.** Both sides of that comparison are real Binance data --
   no generator is involved. It is the 2021-2024 vs 2024-2026 volatility regime shift
   (BTC annualised vol 0.666 -> 0.456, SOL 1.415 -> 0.779). The ceiling would have to rise to
   60.82 %, at which point a generator emitting near-flat paths is admissible. The criterion stops
   discriminating.
2. **It destroys the objective function.** `NNratio` is the memorisation diagnostic and the thing
   being minimised. Its real-vs-real band widens 5x, and it moves the wrong way: `NN << 1` is the
   signature of memorisation, yet honest future data now reads 0.75 because the lower-volatility
   future paths sit nearer the dense centre of the training cloud. Memorisation and regime drift
   become unidentifiable.

### Where holdout-era IS used

**The conditional-forecast CRPS (paper Table 4) is run on `holdout-era`, not `interleaved`.**
Path shadowing conditions on a 65-price prefix; the prefix carries the current volatility level, so
retrieval matches regime and the drift is absorbed by the conditioning. This is why forecasting
evaluation tolerates a chronological split while unconditional distribution matching does not, and
it is the split the paper itself uses for Table 4.

It is additionally reported as a robustness result: `(h, K)` selected on the interleaved build,
re-scored on the holdout-era build, i.e. "calibrated 2021-2024, scored 2024-2026".

| metric family | split | reason |
|---|---|---|
| `(h, K)` selection, A1-A36 unconditional | `interleaved` | needs a tight envelope; no conditioning to absorb drift |
| conditional CRPS (Table 4) | `holdout-era` | forecasting; no look-ahead; conditioning absorbs drift |
| robustness / regime transfer | `holdout-era` | quantifies the cost of the 2024 regime change |

Envelopes for both are stored in `envelope_by_split_mode.json`.

---

## Addendum 2026-08-26 (c): the dataset period is cut to one volatility regime, and addendum (b) is OVERTURNED

Addendum (b) concluded that a chronological held-out-future split could not be used to select
`(h, K)`. **That conclusion was confounded and is withdrawn.** The damage attributed to
chronological ordering was caused by the 2021 H1 mania sitting inside the training era.

### The measurement

Panel-average annualised volatility, monthly, over the whole 2021-01..2026-07 history:

| period | panel ann. vol |
|---|---:|
| 2021-01 | 12.30 |
| 2021-02 | 11.20 |
| 2021-04 | 9.50 |
| 2021-05 | 12.35 |
| 2021-07 onward | 2.19 - 7.10 |

Whole history spans **5.65x** min-to-max. Dropping 2021 H1 brings it to 3.25x. The regime break is
at the *start* of the record, not at 2024 as addendum (b) assumed.

### Period / N trade curve (chronological `holdout-era`, all real-vs-real)

| start | N | years | vol_err MAX | train-vs-test | corr MAX | NN width |
|---|---:|---:|---:|---:|---:|---:|
| 2021-07 | 8192 | 5.00 | 25.26 | 25.26 | 0.0889 | 0.1351 |
| **2022-07** | **6144** | **4.00** | **24.59** | **21.10** | 0.1843 | **0.0684** |
| 2023-01 | 5120 | 3.50 | 35.40 | 11.02 | 0.1471 | 0.1797 |
| 2023-07 | 4096 | 3.00 | 32.78 | 29.13 | 0.1133 | 0.1310 |
| 2024-01 | 3072 | 2.50 | 33.77 | 24.87 | 0.1337 | 0.1275 |

Shortening the period is **not** monotonically better. Below 4 years the loss of paths (the vol
estimator's sd rises 0.50 -> 0.78 pp) and the loss of month-to-month variety outrun the reduction in
secular drift. The optimum is interior.

### Matched control: split mode with period and N held fixed

Period 2022-07..2026-07, N = 6144, identical windows, only the split rule differs:

| | vol_err MAX | corr MAX | NN width | train-vs-test |
|---|---:|---:|---:|---:|
| **holdout-era (chronological)** | **24.59** | 0.1843 | **0.0684** | 21.10 |
| interleaved (control) | 26.45 | 0.0379 | 0.0976 | 6.81 |

Chronological is *better* on the vol ceiling and *tighter* on the NN band. Addendum (b) compared
chronological-on-full-history against interleaved-on-full-history and attributed the gap to the
split rule; the matched control shows the gap was the mania period.

### Locked build

```
dataset/TrueDataset/variants/om_2022-07_N6144
  period      2022-07 .. 2026-07  (4.00 y, one volatility regime)
  N_SAMPLES   6144      seq_tag 6144x128x8
  split-mode  holdout-era: past era = train/val/valdisc, 34-day embargo,
                           future era = disc/test
```

**One split for every metric** -- `(h, K)` selection, the A1-A36 unconditional panel, and the
conditional-forecast CRPS. No metric is scored against data that any other metric trained on, and
no split contains a window later than a window it is scored against. The per-metric split table in
addendum (b) is void.

### Envelope for the locked build (supersedes the header table)

| statistic | min | median | CEILING |
|---|---:|---:|---:|
| vol err (%) | 8.72 | 14.96 | **24.59** |
| cross-asset corr err | 0.0208 | 0.0925 | **0.1843** |
| excess-kurtosis err (%) | 52.63 | 119.50 | 3044.68 (still NOT gated) |

NNratio admissible band **0.932 - 1.000** (test 0.932, disc 0.964, valdisc 0.968).
A perfect generator -- one emitting the training distribution exactly -- scores vol 9.03 %.
