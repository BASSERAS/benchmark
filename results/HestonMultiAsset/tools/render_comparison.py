#!/usr/bin/env python3
"""
render_comparison.py
--------------------
Generate results/HestonMultiAsset/README.md: the cross-method comparison page.

Scope is deliberately narrow -- **two tables and nothing else**:

  1. A1-A34, mean +/- std, one column per method.
  2. B curve-shape metrics, mean +/- std, one column per method.

Everything discursive (the dataset law, the per-asset vs native metric scoping,
the memorisation diagnostic, the leaderboard) lives in `oldreadme.md`; everything
method-specific lives in `<Method>/README.md`. This page exists to answer one
question -- which method is closer to the truth on each row -- and adding prose to
it would blur that.

Why the numbers are not retyped from the per-method pages
--------------------------------------------------------
They are read from the same `metrics_summary.csv` / `curve_b_aggregate.json` /
`grid_tvd_aggregate.json` artefacts that the per-method renderers read, and the
row list, labels, arrows and formatters are **imported from
`SBTS/code/render_readme.py`** rather than duplicated. A metric added or relabelled
there propagates here automatically; a hand-maintained second copy would drift and
the drift would be invisible.

The `Perfect floor` column is the independent-draw floor (GUIDELINE section 5.4):
five fresh draws from the same SDE with the same frozen per-asset parameters at
seeds 1000-1004, scored with byte-identical metric code. It is kept because
without it "LS4 beats SBTS on this row" is unreadable -- both could be far from
the truth, or both could already sit at the noise floor where the comparison is
meaningless. A method scoring *below* the floor is a red flag, not a win.

Reads, per method in --methods:
  results/HestonMultiAsset/<Method>/metrics_summary.csv
  results/HestonMultiAsset/<Method>/curve_b_aggregate.json
  results/HestonMultiAsset/<Method>/grid_tvd_aggregate.json
  results/HestonMultiAsset/perfect_recovery/<same three>

Writes
  results/HestonMultiAsset/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/tools/render_comparison.py
    ... --methods SBTS,LS4
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.dirname(HERE)                       # .../HestonMultiAsset
FLOOR = os.path.join(MA, "perfect_recovery")


def _load_shared():
    """Import SBTS/code/render_readme.py for the shared row list and formatters.

    Imported by path rather than copied so the two pages cannot disagree about
    what A17 is called or which direction A28 improves in. The module guards
    main() behind __name__ == "__main__", so importing it renders nothing.
    """
    p = os.path.join(MA, "SBTS", "code", "render_readme.py")
    if not os.path.exists(p):
        raise SystemExit(f"ABORT: {p} not found -- it is the source of the shared "
                         f"metric row list, this page cannot be built without it.")
    spec = importlib.util.spec_from_file_location("_sbts_render", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sbts_render"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_shared()


# --------------------------------------------------------------------------- A
def render_a(methods, data, floor):
    """A1-A34, one mean +/- std column per method, floor last."""
    head = ["Metric"] + list(methods) + ["Perfect floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for cat, rows in R.CATEGORIES:
        lines.append(f"| **, {cat}, ** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            cells, scope = [], ""
            for m in methods:
                s = data[m].get(key)
                if s is None:
                    cells.append("-")
                    continue
                cells.append(f"{R.fmt(s['mean'])} ± {R.fmt(s['std'])}")
                if s["scope"] == "native":
                    scope = " *(native d=8)*"
            f = floor.get(key)
            cells.append(R.fmt(f["mean"]) if f else "-")
            lines.append(f"| {label}{R.ARROW[direction]}{scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- B
def render_b(methods, aggs, tvds, agg_f, tvd_f):
    """B curve-shape metrics, one mean +/- std column per method.

    Plot order is the union over methods in first-seen order, so a method missing
    a plot leaves a dash rather than shifting every row that follows it.
    """
    head = ["Plot", "Measure"] + list(methods) + ["Perfect floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]

    if any(tvds.get(m) for m in methods):
        cells = []
        for m in methods:
            t = tvds.get(m)
            cells.append(f"{R.pct(t['mean'])} ± {R.pct(t['std'])}" if t else "-")
        cells.append(R.pct(tvd_f["mean"]) if tvd_f else "-")
        lines.append("| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | "
                     + " | ".join(cells) + " |")

    prefixes = []
    for m in methods:
        for p in aggs.get(m, {}):
            if p not in prefixes:
                prefixes.append(p)

    for prefix in prefixes:
        name = next((aggs[m][prefix]["name"] for m in methods
                     if prefix in aggs.get(m, {})), prefix)
        blk_f = agg_f.get(prefix, {})
        first = True
        for mk, mlabel, f_ in R.MEASURES:
            if not any(mk in aggs.get(m, {}).get(prefix, {}) for m in methods):
                continue
            cells = []
            for m in methods:
                blk = aggs.get(m, {}).get(prefix, {})
                cells.append(f"{f_(blk[mk]['mean'])} ± {f_(blk[mk]['std'])}"
                             if mk in blk else "-")
            cells.append(f_(blk_f[mk]["mean"]) if mk in blk_f else "-")
            plot = f"| **{R.esc(name)}** " if first else "|  "
            first = False
            lines.append(plot + f"| {mlabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="SBTS,LS4",
                    help="comma-separated, in display order")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    data, aggs, tvds = {}, {}, {}
    missing = []
    for m in methods:
        root = os.path.join(MA, m)
        data[m] = R.read_summary(os.path.join(root, "metrics_summary.csv"))
        aggs[m] = R.read_json(os.path.join(root, "curve_b_aggregate.json"))
        tvds[m] = R.read_json(os.path.join(root, "grid_tvd_aggregate.json"))
        if not data[m]:
            missing.append(f"{m}/metrics_summary.csv")
        if not aggs[m]:
            missing.append(f"{m}/curve_b_aggregate.json")

    floor = R.read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg_f = R.read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd_f = R.read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    if not floor:
        missing.append("perfect_recovery/metrics_summary.csv")
    if missing:
        raise SystemExit("ABORT: missing artefacts: " + ", ".join(missing) + "\n"
                         "Run metrics/compute_all_multiasset.py for each method first.")

    # Seed count is read off an artefact, never assumed, so a 3-seed re-run
    # cannot be described as a 5-seed one in the header.
    n_seeds = len(next(iter(data[methods[0]].values()))["seeds"])
    links = " · ".join(f"[{m}]({m}/README.md)" for m in methods)

    md = f"""# Multi-Asset Heston (d = 8) — Method Comparison

Mean ± std across {n_seeds} seeds, {" vs ".join(methods)}. Per-seed columns, figures,
hyperparameters and implementation notes are on each method's own page: {links}.
Dataset construction, per-asset vs native metric scoping and the memorisation
diagnostic are in [`oldreadme.md`](oldreadme.md).

**Perfect floor** is the *independent-draw* floor: five fresh draws from the same SDE
with the same frozen per-asset parameters at seeds 1000–1004, scored with byte-identical
metric code. It is **non-zero everywhere** — two independent 8 192-path draws never
produce identical histograms, ACFs, quantiles or covariance matrices. A method scoring
*below* the floor is a red flag, not a win.

## A1–A34, mean ± std across {n_seeds} seeds

> ↓ lower is better; ↑ higher is better;, no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every
> other row is computed on each of the 8 univariate slices and reported as the **mean over
> assets**. All metrics on **log-returns** unless noted; A26 uses price increments.

{render_a(methods, data, floor)}

## B, curve-shape metrics, mean ± std across {n_seeds} seeds

> All ↓ lower is better. Each stylised-fact plot yields a curve, not a scalar; the curve is
> computed per asset and averaged over the 8 assets before the combination. The **MSE row
> decides the winner** for a plot — the % err / NRMSE / CVaR rows are function-only
> diagnostics, reported because a method can win on MSE while failing in the tail.

{render_b(methods, aggs, tvds, agg_f, tvd_f)}
"""
    out = os.path.join(MA, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    n_a = md.count("\n| A")
    print(f"Wrote {out}  ({len(md.splitlines())} lines, "
          f"{len(methods)} methods, {n_seeds} seeds, {n_a} A-rows)")


if __name__ == "__main__":
    main()
