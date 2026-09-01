#!/usr/bin/env python
"""Rebuild results/sweep/corr_spread_cache.json from the arrays actually on disk.

Why this exists
---------------
``sweep_paper.py:corr_ratio`` caches the leverage-spread ratio by trial tag, on
the stated assumption that "generated arrays are immutable once written". That
assumption is false: a re-run of the same trial reuses the same tag and
overwrites ``X_sbbts_1000x252x2_<tag>.npy``, while the cache keeps serving the
value computed from the array that used to be there.

On 2026-09-01 four of eight beta-sweep tags had drifted this way
(t00s2, t00s3, t11s1, t11s2), which silently corrupted the beta=100 vs beta=300
comparison in paper_reimplementation/README.md. This script recomputes every
tag from the current array so the cache and the disk agree again.

Note on t00s2 / t11s2: for those two the array is NEWER than the matching
sbbts_heston_scores_<tag>.json, i.e. a re-run regenerated the paths and was
killed before its MLE finished. The array is still a valid draw from the same
configuration (the tag pins the config), so its leverage spread is a legitimate
measurement -- it simply comes from a different draw than the std_ratios in the
JSON. That is fine for a per-config mean; it would not be fine for a per-path
join, which nothing here does.

Usage:  python rebuild_corr_cache.py
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "sweep"
DS = ROOT / "dataset" / "X_heston_paper_5000x253x2.npy"
CACHE = OUT / "corr_spread_cache.json"


def corr_spread(levels):
    """Std across paths of Corr(dlog S, dlog v) -- identical to sweep_paper.py."""
    R = np.diff(np.log(levels.astype(np.float64)), axis=1)
    c = np.array([np.corrcoef(p, rowvar=False)[0, 1] for p in R])
    return float(np.nanstd(c))


def main():
    old = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    cache = {"__data__": corr_spread(np.load(DS))}
    if "__data__" in old:
        # sanity: the data spread is a fixed property of the dataset
        assert abs(cache["__data__"] - old["__data__"]) < 1e-12, "dataset changed"

    changed = []
    for arr in sorted(OUT.glob("X_sbbts_*.npy")):
        tag = arr.stem.split("_")[-1]
        gen = np.load(arr)
        keep = np.isfinite(gen).all(axis=(1, 2)) & (gen > 0).all(axis=(1, 2))
        if keep.sum() < 50:
            print(f"{tag:10s} skipped ({int(keep.sum())} usable paths)")
            continue
        cache[tag] = corr_spread(gen[keep]) / cache["__data__"]
        was = old.get(tag)
        flag = ""
        if was is not None and abs(was - cache[tag]) > 1e-9:
            flag = f"  <- was {was:.4f}"
            changed.append(tag)
        print(f"{tag:10s} {cache[tag]:.4f}{flag}")

    CACHE.write_text(json.dumps(cache, indent=1))
    print(f"\nwrote {CACHE} ({len(cache) - 1} tags, {len(changed)} corrected: "
          f"{', '.join(changed) or 'none'})")

    for name, tags in [("beta=100", ["t00", "t00s1", "t00s2", "t00s3"]),
                       ("beta=300", ["t11", "t11s1", "t11s2", "t11s3"])]:
        v = [cache[t] for t in tags if t in cache]
        print(f"{name}: {np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}  "
              f"{[round(x, 4) for x in v]}")


if __name__ == "__main__":
    main()
