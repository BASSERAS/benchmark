# TrueDataset — real multi-asset intraday panel, d = 8, seq_len = 128

## Quick description

A **real-data** counterpart to `dataset/HestonMultiAsset`, built to the same benchmark shape so the
metric drivers need no special casing. Eight liquid Binance USDT spot pairs, 1-second public klines
resampled to **30-second bars**, sliced into **non-overlapping** windows of 128 bars (64 minutes),
dealt into the five standard splits.

Nothing here is simulated. There is no latent variance, so the teacher-sigma metrics **A33/A34 are
skipped** for this dataset (see `metrics/compute_all_multiasset.py`, `DATASET_FILES`).

| | |
|---|---|
| Source | `https://data.binance.vision` — public monthly 1s spot klines, no API key |
| Price kind | **last trade price.** The spot archive publishes no quotes, so there is no mid and no bid/ask. |
| Period | 2021-01-01 .. 2026-07-31 UTC (67 months, 2 038 days) |
| Bar | 30 s on the UTC second grid (`86 400 / 30 = 2 880` bars per day, exactly tiling a day) |
| Raw volume | 536 files, 27.7 GiB, every one SHA256-verified against the published `.CHECKSUM` |

## The panel, and why these eight

```
BTCUSDT  ETHUSDT  BNBUSDT  SOLUSDT  XRPUSDT  DOGEUSDT  ADAUSDT  LINKUSDT
```

These are the eight most liquid USDT pairs **today**, not the eight with the longest history. That
choice was made by measurement, not preference. The long-history panel is forced to carry TRX and
ETC, and both decay badly:

| Probe month | max-history panel (…LTC TRX ETC) | max-liquidity panel (this one) |
|---|---:|---:|
| 2018-09 | 0.290 | n/a (SOL unlisted) |
| 2020-09 | 0.527 | 0.401 |
| 2023-06 | 0.497 | 0.563 |
| 2026-06 | 0.522 | **0.786** |

*(mean pairwise realised correlation of 30-second returns)*

By 2026-06 ETC reaches **72.1 %** zero 30s-returns and 27.1 % no-trade bars, TRX 53.0 %. A panel
whose assets stop trading cannot exhibit the cross-asset structure the multi-asset benchmark exists
to test. The liquid panel runs 0.56 → 0.79 with no asset above 35 % stale.

**Why the period starts at 2021-01.** SOL is the binding listing (first 1s month 2020-09), but SOL
and DOGE are still 25 % and 40 % stale at 30 s during autumn 2020, so the start is pushed to
2021-01. First available 1s month per symbol: BTC/ETH 2017-08, BNB 2017-11, ADA 2018-04,
XRP 2018-05, LINK 2019-01, DOGE 2019-07, SOL 2020-09.

## Why 30-second bars

Binance emits a complete 86 400-row 1-second grid per symbol per day, but a **complete grid is not
complete information**: no-trade seconds are filled with the carried-forward close. At `dt = 1 s`
the resulting staleness (45–87 % zero returns) collapses the realised cross-asset correlation to
**0.41** against a **0.71** plateau — the Epps effect, measured directly rather than assumed.

30 s is the coarsest bar that still admits enough non-overlapping windows to fill the benchmark
shape, and it recovers **0.685** of the 0.708 plateau.

## Why seq_len = 128

Over 2021-01 .. 2026-07 the 30 s panel yields **45 855** non-overlapping windows of length 128,
against the `5 × 8192 = 40 960` the benchmark contract requires — 12 % slack. Length 252 (the
Heston panel's length) would yield only **23 306**, forcing overlapping windows and destroying the
independence the splits depend on.

## The five benchmark splits

Split names and roles match `dataset/HestonMultiAsset` exactly. What differs is how independence is
obtained: the Heston panel draws five independent samples from a known SDE, but there is only **one**
history of the real market, so the splits must be carved out of it.

**Interleaved blocks.** `BLOCK_WINDOWS = 256` contiguous windows form a block; `5 × 32 = 160` blocks
are taken at uniform spacing across the whole history and dealt **round-robin** to the five splits.
Two consequences, both deliberate:

* every split spans the full 2021 → 2026 range, so each one sees every market regime (see the
  `first_utc` / `last_utc` per split in `dataset_stats.json`);
* the gaps between consecutive blocks act as an **embargo** against boundary autocorrelation, which
  a naive chronological cut would leave intact at the seam.

| Split | Role |
|-------|------|
| **train** | The only data a generator ever sees. Every method is fit on this split. |
| **test** | Held-out **real reference**. Every A/B metric scores generated paths against this split, never against train. |
| **disc** | A third disjoint split used **only** by the discriminative / predictive classifiers (A18 / A19) as their real-vs-fake training data, so those classifiers never touch the test split they are evaluated on. |
| **val** | Validation counterpart of `test`, for model selection and hyperparameter search. Never used for a reported number. |
| **valdisc** | Validation counterpart of `disc`, for the classifiers during model selection. |

The builder asserts that the five index sets are pairwise disjoint and that each has exactly 8 192
windows; a violation aborts the build.

## Storage contract

Inherited unchanged from `dataset/Heston` and `dataset/HestonMultiAsset`: arrays store **PRICE**
paths, rebased so that `S[:, 0, :] = 100.0` exactly. **NOT** log returns. Any log-return transform
belongs to the model-input layer, never to the dataset.

Rebasing is what makes windows from 2021 and 2026 comparable despite BTC moving through two orders
of magnitude; it discards the level, which no metric in this benchmark uses.

`dt` is `1 / (365 × 2880) = 9.5129e-07` — crypto trades 24/7, so a year is 365 calendar days, not
252 trading days.

## Handling of gaps and stale prices

* A symbol-month whose 1-second grid covers **< 95 %** of the month, or which starts after the first
  second of the month, is **rejected and the build fails**. A hole in one symbol would silently
  shift that asset's block layout relative to the others. (This matters: TRX 2018-06 covers only
  63.4 %, starting 2018-06-11 12:50:05 UTC. Nothing in the selected 2021+ window is affected.)
* Within an accepted month, missing seconds are forward-filled from the last trade — which is what
  a last-trade series *means*, not an imputation choice.
* Two staleness diagnostics are recorded per asset and must be read together: `zero_return_fraction`
  (bar closed at the previous close) and `no_trade_bar_fraction` (no trade at all in the 30 s).

Measured on the committed build:

| Asset | zero 30s-returns | no-trade bars | ann. vol (train) |
|---|---:|---:|---:|
| BTCUSDT | 4.8 % | 0.04 % | 0.622 |
| ETHUSDT | 3.4 % | 0.04 % | 0.818 |
| BNBUSDT | 18.4 % | 0.04 % | 0.723 |
| SOLUSDT | 12.0 % | 0.24 % | 1.255 |
| XRPUSDT | 13.9 % | 0.05 % | 0.966 |
| DOGEUSDT | 12.6 % | 0.21 % | 1.356 |
| ADAUSDT | 20.7 % | 0.64 % | 1.112 |
| LINKUSDT | 22.4 % | 1.92 % | 1.076 |

Pairwise realised correlation: **mean 0.573**, min 0.418, max 0.819.

> ⚠️ **The 12.9 % pooled zero-return rate is a real feature of this data, not noise to be cleaned.**
> It puts a point mass at the origin of the return distribution and drives the excess kurtosis to
> **78.7** (against 1.12 for the Heston panel). Methods calibrated on Heston should be expected to
> need recalibration, and some break outright — see
> `results/TrueDataset/SBTS/code/sbts_generate_true.py` for a worked example where the fat tail
> overflows a `float64` exponential that Heston never reaches.

## Files

All ten split arrays are **float64, shape (8192, 128, 8)** with axes `(path, time, asset)`; asset
order is the `ASSETS` list above.

| File | Split | Field | Description |
|------|-------|-------|-------------|
| `true_S_8192x128x8.npy` | train | $S_t$ | Price paths |
| `true_rv_8192x128x8.npy` | train | $RV_t$ | Per-bar realised variance from the 1s returns |
| `true_S_test_8192x128x8.npy` | test | $S_t$ | Price paths, scoring reference |
| `true_rv_test_8192x128x8.npy` | test | $RV_t$ | Realised variance |
| `true_S_disc_8192x128x8.npy` | disc | $S_t$ | Price paths, A18/A19 classifier real data |
| `true_rv_disc_8192x128x8.npy` | disc | $RV_t$ | Realised variance |
| `true_S_val_8192x128x8.npy` | val | $S_t$ | Price paths, model selection |
| `true_rv_val_8192x128x8.npy` | val | $RV_t$ | Realised variance |
| `true_S_valdisc_8192x128x8.npy` | valdisc | $S_t$ | Price paths, classifier model selection |
| `true_rv_valdisc_8192x128x8.npy` | valdisc | $RV_t$ | Realised variance |
| `true_t0_*_8192.npy` | each | — | Epoch-second start time of each window. **The audit trail**: it is what lets anyone re-derive a split from the raw archive. |
| `dataset_stats.json` | — | — | Panel, period, split map, per-split date bounds and annualised vols, staleness and correlation diagnostics. **Tracked.** |
| `true_config.py`, `download_binance.py`, `build_true_dataset.py` | — | — | Config, downloader, builder |

`true_rv` is **not** a latent variance and must not be used as one — it is an *observable* estimator,
the sum of squared 1-second log-returns inside each 30 s bar. It is provided as a volatility proxy;
the A33/A34 teacher-sigma metrics, which require the true latent state, are skipped for this dataset.

> ⚠️ **The `.npy` files are NOT committed** (64 MiB each, 640 MiB total), and neither are `raw/`
> (29 GiB) or `cache/` (403 MiB). All are git-ignored. `dataset_stats.json` **is** committed, so the
> exact build is pinned in the repo even though the samples are not.

## Reproduce

```bash
cd dataset/TrueDataset

# 1. fetch + SHA256-verify the raw archives (~28 GiB, resumable, safe to re-run)
python download_binance.py
python download_binance.py --check          # verify only, download nothing

# 2. build all five splits (~9 min on 12 cores; the parse cache makes re-runs ~1 min)
OMP_NUM_THREADS=1 taskset -c 0-11 python build_true_dataset.py --workers 12

# diagnostics only, no arrays written
python build_true_dataset.py --diagnose-only
```

The build is **deterministic** — the split layout is computed from `np.linspace` over the window
count, with no RNG anywhere — so a rebuild from the same raw archives is bitwise identical.

Useful flags: `--start/--end` (shorter history), `--split-mode {interleaved,chronological}`,
`--out-dir` (write a variant elsewhere while sharing the one parse cache), `--min-coverage`.

## Sanity checks

Verified on the committed build:

| Check | Result |
|-------|--------|
| `S[:, 0, :] == 100.0` exactly, all splits | True |
| `S > 0` and finite everywhere | True |
| Split index sets pairwise disjoint | True (asserted in the builder) |
| Windows per split | 8 192 each, 40 960 of 45 855 used |
| Every symbol-month ≥ 95 % second-coverage | True, 536 / 536 |
| Raw archives match published SHA256 | 536 / 536 |
| Every split spans 2021-01 → 2026-06/07 | True (`dataset_stats.json → splits`) |
| Bars tile the UTC day exactly | `86 400 / 30 = 2 880`, asserted in `true_config.py` |

## References

- Binance public data archive, `https://data.binance.vision` (spot monthly klines, 1s interval).
- Epps, T. W. (1979). Comovements in stock prices in the very short run. *Journal of the American
  Statistical Association* 74(366), 291–298.
- Hayashi, T. & Yoshida, N. (2005). On covariance estimation of non-synchronously observed diffusion
  processes. *Bernoulli* 11(2), 359–379.
- Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. (2003). Modeling and forecasting
  realized volatility. *Econometrica* 71(2), 579–625.
