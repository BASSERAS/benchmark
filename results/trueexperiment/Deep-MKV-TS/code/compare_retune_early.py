"""Single-seed verdict: does the lr=6.4e-06 retune beat the untrained reference?

Why a per-seed comparator exists
--------------------------------
The published tables are 5-seed aggregates, but the operator asked for a read on
the FIRST seed that finishes rather than a wait for all five. metrics_summary.csv
carries per-seed columns (``seed_0`` .. ``seed_4``) beside ``mean``/``std``, so a
single-seed comparison is available without re-running anything: retune seed S is
compared against reference seed S, the same generation seed index, both measured
against the same real-vs-real floor.

How "better" is decided
-----------------------
NOT "lower wins". Guideline section 7.1: the floor is what genuine market data
scores against genuine market data, so it is a TARGET, not a minimum. Scoring far
below it is as much a failure as scoring far above -- it means the generator is
reproducing the training split rather than the law. So the distance used here is
|log(value / floor)|, and a row is won by whichever of the two is CLOSER to 1.0
in either direction.

A28 is excluded: its target is 1.0 rather than the floor's own value, so a
floor-relative ratio would be meaningless for it (see render_comparison.py:206).

The verdict this prints is deliberately not a single number. "Clearly better"
has to survive the split that has dominated this investigation: the alpha=1 run
HELPED on marginal statistics while DESTROYING temporal structure, so an
aggregate tally that mixes the two hides exactly the failure mode we are hunting.
Rows are therefore also broken out by family.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRUE_ROOT = HERE.parent.parent

# Metrics whose floor-relative ratio is not interpretable; see module docstring.
EXCLUDE = {"A28_disc_auc"}

# The split that matters. Substrings, matched against the metric key.
TEMPORAL = ("acf", "rolling", "vol_clust", "lag", "hurst", "autocorr")


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    """metric -> {column: value}. Non-numeric cells are dropped."""
    if not path.is_file():
        raise SystemExit(f"ABORT: {path} missing")
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["metric"]
            vals: dict[str, float] = {}
            for col, cell in row.items():
                if col in ("metric", "scope"):
                    continue
                try:
                    vals[col] = float(cell)
                except (TypeError, ValueError):
                    continue
            out[key] = vals
    return out


def log_dist(value: float | None, floor: float | None) -> float | None:
    """|log(value/floor)|. None when either side is unusable."""
    if value is None or floor is None:
        return None
    if not (math.isfinite(value) and math.isfinite(floor)):
        return None
    if value <= 0.0 or floor <= 0.0:
        return None
    return abs(math.log(value / floor))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--retune-root", type=Path,
                    default=TRUE_ROOT / "Deep-MKV-TS" / "retune_lr")
    ap.add_argument("--reference-root", type=Path, default=TRUE_ROOT / "reference")
    ap.add_argument("--floor-root", type=Path, default=TRUE_ROOT / "real_floor")
    args = ap.parse_args()

    seed = int(args.seed)
    col = f"seed_{seed}"

    floor = load_summary(args.floor_root / "metrics_summary.csv")
    ref = load_summary(args.reference_root / "metrics_summary.csv")
    new = load_summary(args.retune_root / "metrics_summary.csv")

    rows = []
    for key in sorted(new):
        if key in EXCLUDE:
            continue
        # The floor is a 5-seed mean of real-vs-real draws; there is no
        # per-seed floor to pair with, and there should not be one -- the
        # ruler must not move between the two things being compared.
        f = floor.get(key, {}).get("mean")
        v_new = new.get(key, {}).get(col)
        v_ref = ref.get(key, {}).get(col)
        d_new = log_dist(v_new, f)
        d_ref = log_dist(v_ref, f)
        if d_new is None or d_ref is None:
            continue
        rows.append({
            "metric": key,
            "floor": f,
            "retune": v_new,
            "reference": v_ref,
            "retune_xfloor": v_new / f,
            "reference_xfloor": v_ref / f,
            "winner": "retune" if d_new < d_ref else "reference",
            "temporal": any(t in key.lower() for t in TEMPORAL),
        })

    if not rows:
        raise SystemExit("ABORT: no comparable rows -- check that both summaries "
                         f"contain column {col!r}")

    wins = sum(r["winner"] == "retune" for r in rows)
    t_rows = [r for r in rows if r["temporal"]]
    m_rows = [r for r in rows if not r["temporal"]]
    t_wins = sum(r["winner"] == "retune" for r in t_rows)
    m_wins = sum(r["winner"] == "retune" for r in m_rows)

    print(f"\n=== seed {seed}: lr=6.4e-06 retune vs untrained reference ===")
    print(f"{'metric':<34s} {'floor':>11s} {'ref xfloor':>11s} "
          f"{'retune xfloor':>14s}  winner")
    print("-" * 88)
    for r in rows:
        print(f"{r['metric']:<34s} {r['floor']:11.5g} "
              f"{r['reference_xfloor']:11.2f} {r['retune_xfloor']:14.2f}  "
              f"{r['winner']}{'   [temporal]' if r['temporal'] else ''}")

    print("-" * 88)
    print(f"rows compared          : {len(rows)}")
    print(f"retune closer to floor : {wins}/{len(rows)}")
    print(f"  marginal / other     : {m_wins}/{len(m_rows)}")
    print(f"  TEMPORAL (acf, vol)  : {t_wins}/{len(t_rows)}   "
          "<-- the family alpha=1 destroyed")

    out = args.retune_root / f"early_verdict_seed_{seed}.json"
    out.write_text(json.dumps({
        "seed": seed,
        "n_compared": len(rows),
        "retune_wins": wins,
        "reference_wins": len(rows) - wins,
        "retune_temporal_wins": t_wins,
        "n_temporal": len(t_rows),
        "retune_marginal_wins": m_wins,
        "n_marginal": len(m_rows),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
