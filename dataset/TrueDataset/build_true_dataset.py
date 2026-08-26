"""
Resample the raw Binance 1-second klines to 30-second bars and cut them into
the five benchmark splits.

Pipeline
--------
1. PARSE   one (symbol, month) zip at a time, in a worker process.  Only three
   of the twelve kline columns are read: open_time, close, count.

2. GRID    the month is laid out on its exact UTC 1-second grid
   (days_in_month * 86400 slots).  Binance is missing ~1% of seconds in the
   early years (exchange downtime) and the very first month of a listing can
   start mid-month, so a month whose coverage falls below --min-coverage is
   rejected rather than silently forward-filled across a multi-day hole.
   The remaining gaps are forward-filled; a month may not begin with a gap.

3. BARS    the 1s grid is reshaped to (-1, BAR_SECONDS) and reduced to
        close  = last second of the bucket
        rv     = sum of squared 1s log returns inside the bucket
        traded = whether ANY trade printed inside the bucket
   `rv` is a realised-variance PROXY.  It is not the Heston latent variance v_t
   and A33/A34 (which test the latent-variance marginal) do not apply to it.
   It is stored because the conditional-generation metrics need a volatility
   state, and because it is the only honest volatility observable on real data.

4. WINDOW  the per-symbol bar series are concatenated over months, stacked into
   one (n_bars, d) panel, and cut into NON-OVERLAPPING windows of SEQ_LEN bars.
   Each window is rebased to S0 = 100 at its own first bar, so a window carries
   shape, not level -- the same contract as dataset/Heston*.

5. SPLIT   interleaved blocks (see true_config.BLOCK_WINDOWS).  BLOCK_WINDOWS
   contiguous windows form a block; 5 * BLOCKS_PER_SPLIT blocks are taken at
   uniform spacing across the whole history and dealt round-robin to the five
   splits.  Every split therefore sees every market regime, the splits are
   disjoint, and the unused windows between selected blocks act as an embargo
   against boundary autocorrelation.  --split-mode chronological instead cuts
   one contiguous stretch per split, in the order train/val/valdisc/disc/test,
   which reproduces the held-out-future protocol of the paper.

Storage contract (inherited from dataset/HestonMultiAsset)
----------------------------------------------------------
arrays store PRICE paths anchored at S0 = 100, shape (n_samples, seq_len, d),
float64, axis order (path, time, asset).  Log returns belong to the model-input
layer, not to the dataset.

Outputs
-------
    true_S_8192x128x8.npy            train
    true_S_test_8192x128x8.npy       test      -- scoring reference
    true_S_disc_8192x128x8.npy       disc      -- real data for the A18/A19 classifier
    true_S_val_8192x128x8.npy        val       -- model selection
    true_S_valdisc_8192x128x8.npy    valdisc
    true_rv{suffix}_8192x128x8.npy   matching realised-variance proxy
    true_t0{suffix}_8192.npy         UTC epoch second of each window's first bar
    dataset_stats.json               panel, period, and the staleness diagnostics

Usage
-----
    python build_true_dataset.py                      # full build
    python build_true_dataset.py --diagnose-only      # measure, write nothing
    python build_true_dataset.py --split-mode chronological
"""
import argparse
import calendar
import datetime
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import true_config as cfg

CACHE_DIR = os.path.join(cfg.HERE, "cache")


def month_start_epoch(month):
    y, m = (int(x) for x in month.split("-"))
    return int(datetime.datetime(y, m, 1, tzinfo=datetime.timezone.utc).timestamp())


def month_seconds(month):
    y, m = (int(x) for x in month.split("-"))
    return calendar.monthrange(y, m)[1] * cfg.SECONDS_PER_DAY


def _read_klines(path):
    """(epoch_second, close, count) for one monthly zip."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            head = fh.readline()
        skip = 0 if head[:1].isdigit() else 1          # 2025+ files carry a header row
        with z.open(name) as fh:
            df = pd.read_csv(fh, header=None, skiprows=skip, usecols=[0, 4, 8],
                             names=["t", "close", "count"],
                             dtype={"t": np.int64, "close": np.float64, "count": np.int64})

    t = df["t"].to_numpy()
    # open_time is milliseconds before 2025-01-01 and microseconds from 2025-01-01.
    # 1.5e12 vs 1.5e15 -- three orders of magnitude apart, so magnitude is decisive.
    unit = 1_000 if t[0] < 2e13 else 1_000_000
    return t // unit, df["close"].to_numpy(), df["count"].to_numpy()


def build_month(symbol, month, min_coverage):
    """30s bars for one (symbol, month). Cached as .npz. Returns (close, rv, traded)."""
    cache = os.path.join(CACHE_DIR, symbol, f"{month}.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["close"], z["rv"], z["traded"]

    sec, close, count = _read_klines(cfg.zip_path(symbol, month))
    t0, n = month_start_epoch(month), month_seconds(month)
    idx = sec - t0
    keep = (idx >= 0) & (idx < n)
    idx, close, count = idx[keep], close[keep], count[keep]

    coverage = idx.size / n
    if coverage < min_coverage:
        raise ValueError(f"{symbol} {month}: only {coverage:.1%} of the 1s grid present "
                         f"(need {min_coverage:.0%}) -- move START_MONTH past this month")
    if idx[0] != 0:
        raise ValueError(f"{symbol} {month}: first tick is {idx[0]}s into the month; "
                         f"a leading gap cannot be forward-filled without inventing prices")

    px = np.full(n, np.nan)
    px[idx] = close
    trades = np.zeros(n, dtype=np.int64)
    trades[idx] = count

    # Forward-fill the missing seconds: carry the index of the last observed
    # second forward, then gather. Vectorised equivalent of pandas ffill.
    src = np.maximum.accumulate(np.where(np.isnan(px), 0, np.arange(n)))
    px = px[src]

    r = np.zeros(n)
    r[1:] = np.diff(np.log(px))

    b = cfg.BAR_SECONDS
    close_bar = px.reshape(-1, b)[:, -1].copy()
    rv_bar = (r.reshape(-1, b) ** 2).sum(axis=1)
    traded_bar = trades.reshape(-1, b).sum(axis=1) > 0

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, close=close_bar, rv=rv_bar, traded=traded_bar)
    return close_bar, rv_bar, traded_bar


def _job(args):
    symbol, month, min_coverage = args
    try:
        build_month(symbol, month, min_coverage)
        return symbol, month, None
    except Exception as exc:                          # noqa: BLE001 - collect, report at the end
        return symbol, month, f"{type(exc).__name__}: {exc}"


def build_panel(symbols, months, min_coverage, workers):
    """(n_bars, d) close / rv / traded panels plus the UTC epoch second of each bar."""
    jobs = [(s, m, min_coverage) for s in symbols for m in months]
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(CACHE_DIR, j[0], f"{j[1]}.npz"))]
    if todo:
        print(f"resampling {len(todo)} symbol-months on {workers} workers", flush=True)
        errors, done = [], 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for symbol, month, err in pool.map(_job, todo, chunksize=1):
                done += 1
                if err:
                    errors.append(f"{symbol} {month}  {err}")
                if done % 40 == 0 or done == len(todo):
                    print(f"  [{done}/{len(todo)}] errors={len(errors)}", flush=True)
        if errors:
            print("\n".join(errors))
            raise SystemExit(f"{len(errors)} symbol-month(s) failed; fix the panel or the period")
    else:
        print("all symbol-months already cached", flush=True)

    close, rv, traded = [], [], []
    for symbol in symbols:
        c, v, tr = zip(*(build_month(symbol, m, min_coverage) for m in months))
        close.append(np.concatenate(c))
        rv.append(np.concatenate(v))
        traded.append(np.concatenate(tr))

    t_bar = np.concatenate([
        month_start_epoch(m) + np.arange(month_seconds(m) // cfg.BAR_SECONDS) * cfg.BAR_SECONDS
        for m in months
    ])
    return (np.stack(close, axis=1), np.stack(rv, axis=1),
            np.stack(traded, axis=1), t_bar)


def _utc(ts):
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc)


def diagnose(close, traded, t_bar, symbols):
    """Staleness and realised-correlation diagnostics for the assembled panel."""
    r = np.diff(np.log(close), axis=0)
    zero = (r == 0.0).mean(axis=0)
    corr = np.corrcoef(r, rowvar=False)
    off = corr[~np.eye(len(symbols), dtype=bool)]

    print(f"\npanel {close.shape[0]:,} bars x {close.shape[1]} assets, "
          f"{_utc(t_bar[0]):%Y-%m-%d} .. {_utc(t_bar[-1]):%Y-%m-%d}")
    print(f"{'asset':10s} {'zero-return':>12s} {'no-trade bars':>14s}")
    for i, s in enumerate(symbols):
        print(f"{s:10s} {zero[i]:11.1%} {1 - traded[:, i].mean():13.1%}")
    print(f"pairwise realised corr: mean {off.mean():.3f}  "
          f"min {off.min():.3f}  max {off.max():.3f}")
    return {
        "zero_return_fraction": dict(zip(symbols, zero.round(5).tolist())),
        "no_trade_bar_fraction": dict(zip(symbols, (1 - traded.mean(axis=0)).round(5).tolist())),
        "realised_corr_mean": float(off.mean()),
        "realised_corr_min": float(off.min()),
        "realised_corr_max": float(off.max()),
    }


def split_blocks(n_windows, mode):
    """{suffix: (N_SAMPLES,) window indices}. Splits are disjoint by construction."""
    bw, bps = cfg.BLOCK_WINDOWS, cfg.BLOCKS_PER_SPLIT
    n_splits = len(cfg.SPLITS)
    need = n_splits * bps
    n_blocks = n_windows // bw
    if n_blocks < need:
        raise SystemExit(f"history yields {n_blocks} blocks of {bw} windows, need {need}. "
                         f"Extend the period or shorten SEQ_LEN.")

    if mode == "holdout-era":
        # Two eras. Everything that fits or selects a model (train, val, valdisc)
        # lives in the past era; everything scored against (disc, test) lives in
        # the future era. Blocks are spread within each era so a split covers its
        # whole era rather than one contiguous stretch of it.
        # One block spans 256*128*30 s = 11.4 days. Any embargo of a few blocks
        # is already ~1e3 x the ~10 min horizon at which ACF|r| decays into
        # noise on this panel, so spend at most 4 blocks on it and never buy
        # more by widening the period back into a different volatility regime.
        n_future = 2 * bps
        embargo = max(1, min(4, n_blocks - n_future - 3 * bps))
        n_past = n_blocks - n_future - embargo
        if n_past < 3 * bps:
            raise SystemExit(f"holdout-era needs {3 * bps + n_future + 1} blocks "
                             f"of {bw} windows, have {n_blocks}.")
        past = np.unique(np.linspace(0, n_past - 1, 3 * bps).round().astype(int))
        if past.size != 3 * bps:
            past = np.arange(3 * bps)
        future = np.arange(n_blocks - n_future, n_blocks)
        chosen = {s: past[k::3] for k, s in enumerate(["", "_val", "_valdisc"])}
        chosen.update({s: future[k::2] for k, s in enumerate(["_disc", "_test"])})
    elif mode == "chronological":
        # train first, test last: the held-out-future protocol of the paper.
        order = ["", "_val", "_valdisc", "_disc", "_test"]
        chosen = {s: np.arange(k * bps, (k + 1) * bps) for k, s in enumerate(order)}
    else:
        picked = np.unique(np.linspace(0, n_blocks - 1, need).round().astype(int))
        if picked.size != need:              # n_blocks barely >= need: no room to spread
            picked = np.arange(need)
        chosen = {s: picked[k::n_splits] for k, s in enumerate(cfg.SPLITS)}

    out = {}
    for suffix, blocks in chosen.items():
        idx = (blocks[:, None] * bw + np.arange(bw)[None, :]).ravel()
        assert idx.size == cfg.N_SAMPLES
        out[suffix] = idx
    seen = np.concatenate(list(out.values()))
    assert seen.size == np.unique(seen).size, "splits overlap"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=cfg.START_MONTH)
    ap.add_argument("--end", default=cfg.END_MONTH)
    ap.add_argument("--symbols", nargs="+", default=cfg.ASSETS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="reject a symbol-month whose 1s grid is less complete than this")
    ap.add_argument("--split-mode",
                    choices=["interleaved", "chronological", "holdout-era"],
                    default="interleaved")
    ap.add_argument("--out-dir", default=cfg.HERE,
                    help="where to write the .npy splits. Variants for the SBTS "
                         "sweeps go in subdirectories and share the parse cache.")
    ap.add_argument("--n-samples", type=int, default=cfg.N_SAMPLES,
                    help="paths per split. Lowering it buys a shorter, more "
                         "volatility-homogeneous period for the same 5 splits.")
    ap.add_argument("--diagnose-only", action="store_true")
    args = ap.parse_args()
    if args.n_samples != cfg.N_SAMPLES:
        if args.n_samples % cfg.BLOCK_WINDOWS:
            raise SystemExit(f"--n-samples must be a multiple of {cfg.BLOCK_WINDOWS}")
        cfg.N_SAMPLES = args.n_samples
        cfg.BLOCKS_PER_SPLIT = args.n_samples // cfg.BLOCK_WINDOWS
        cfg.TAG = f"{args.n_samples}x{cfg.SEQ_LEN}x{cfg.D}"
    os.makedirs(args.out_dir, exist_ok=True)

    months = cfg.months(args.start, args.end)
    close, rv, traded, t_bar = build_panel(args.symbols, months,
                                           args.min_coverage, args.workers)
    stats = diagnose(close, traded, t_bar, args.symbols)

    n_windows = close.shape[0] // cfg.SEQ_LEN
    print(f"\n{n_windows:,} non-overlapping windows of {cfg.SEQ_LEN} bars "
          f"({cfg.BAR_SECONDS}s); benchmark needs {len(cfg.SPLITS) * cfg.N_SAMPLES:,}")
    if args.diagnose_only:
        return 0

    usable = n_windows * cfg.SEQ_LEN
    S_all = close[:usable].reshape(n_windows, cfg.SEQ_LEN, -1)
    rv_all = rv[:usable].reshape(n_windows, cfg.SEQ_LEN, -1)
    t0_all = t_bar[:usable].reshape(n_windows, cfg.SEQ_LEN)[:, 0]

    splits = split_blocks(n_windows, args.split_mode)
    per_split = {}
    for suffix, idx in splits.items():
        S = S_all[idx]
        S = cfg.S0 * S / S[:, :1, :]                  # rebase every window to S0
        v = rv_all[idx]
        np.save(os.path.join(args.out_dir, f"true_S{suffix}_{cfg.TAG}.npy"), S)
        np.save(os.path.join(args.out_dir, f"true_rv{suffix}_{cfg.TAG}.npy"), v)
        np.save(os.path.join(args.out_dir, f"true_t0{suffix}_{cfg.N_SAMPLES}.npy"), t0_all[idx])

        r = np.diff(np.log(S), axis=1)
        ann = r.std(axis=(0, 1)) * np.sqrt(cfg.BARS_PER_YEAR)
        per_split[suffix or "train"] = {
            "n_samples": int(S.shape[0]),
            "first_utc": _utc(t0_all[idx].min()).isoformat(),
            "last_utc": _utc(t0_all[idx].max()).isoformat(),
            "ann_vol": ann.round(4).tolist(),
        }
        print(f"{(suffix or '(train)'):9s} S{S.shape}  "
              f"ann.vol per asset {np.array2string(ann, precision=3)}")

    meta = {
        "source": "Binance public monthly 1s spot klines (data.binance.vision)",
        "price_kind": "last trade price; the spot archive publishes no quotes",
        "assets": args.symbols,
        "period": {"start": args.start, "end": args.end},
        "bar_seconds": cfg.BAR_SECONDS, "seq_len": cfg.SEQ_LEN,
        "n_samples": cfg.N_SAMPLES, "d": len(args.symbols),
        "dt": cfg.DT, "bars_per_year": cfg.BARS_PER_YEAR, "S0": cfg.S0,
        "split_mode": args.split_mode,
        "block_windows": cfg.BLOCK_WINDOWS, "blocks_per_split": cfg.BLOCKS_PER_SPLIT,
        "n_windows_available": int(n_windows),
        "diagnostics": stats,
        "splits": per_split,
    }
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
