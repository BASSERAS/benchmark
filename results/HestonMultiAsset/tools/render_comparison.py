#!/usr/bin/env python3
"""
render_comparison.py
--------------------
Generate results/HestonMultiAsset/README.md: the cross-method comparison page.

Scope is deliberately narrow -- **two tables, each with a one-paragraph tally,
and nothing else**:

  1. A1-A34, mean +/- std, one column per method, plus a Winner column.
  2. B curve-shape metrics, mean +/- std, one column per method, plus a Winner column.

Everything discursive (the dataset law, the per-asset vs native metric scoping,
the memorisation diagnostic, the leaderboard) lives in `oldreadme.md`; everything
method-specific lives in `<Method>/README.md`. This page exists to answer one
question -- which method is closer to the truth on each row -- and adding prose to
it would blur that.

How the Winner column is decided
--------------------------------
Not by "smallest number wins". Two things have to be ruled out first, or the
column manufactures results out of noise (see `verdict()`):

  * If **any** method has reached the independent-draw floor on a row, the row is
    marked `-- (floor)` and nobody wins it. At the floor the metric can no longer
    tell a generator from a fresh draw of the true SDE, and *past* the floor is a
    red flag, not a victory.
  * If the gap between the best two methods is no larger than the sum of their
    cross-seed stds, the row is a `--` tie. Five seeds does not resolve that.

The tally paragraph under each table reports wins, ties and floored rows
separately for exactly this reason -- collapsing them into a single win count
would overstate the separation between methods.

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

`reference` in --methods
------------------------
`reference/` is not a trained generator: it is sigma^ref, the frozen parametric
SDE Deep-MKV-TS Algorithm 1 starts from, sampled with the neural control verified
to be exactly zero -- 19 fitted coefficients, zero gradient steps. It is
adjudicated **on equal terms** in the Winner column rather than shown as a passive
baseline like `Perfect floor`, because a row where a 19-coefficient SDE beats a
trained network is a real result about that network rather than a footnote. The
consequence is that a reader who does not know what the column is will misread
"reference wins N" as a method claim, so `main()` emits a labelled paragraph
whenever the column is present -- conditionally, so the paragraph can never
describe a column that is not there.

Because `reference` and `Deep-MKV-TS` share generation seeds, those two columns
are **paired** (same Brownian streams), which makes their difference free of
simulation noise and is the whole reason the reference row is worth rendering.

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/tools/render_comparison.py
    ... --methods SBTS,LS4,reference
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

TIE = "—"
FLOORED = "— *(floor)*"


def _f(x):
    """CSV cells arrive as strings; comparing them unconverted sorts lexically."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def verdict(names, means, stds, floor, direction, target=1.0):
    """Decide who wins a row -- or rule that nobody does. -> (cell, tally_key).

    Three outcomes, checked in this order, because the later test is only
    meaningful once the earlier one has passed:

      "floor"  At least one method has reached the independent-draw floor.
               The row is dead as a comparison: at the floor the metric cannot
               separate a generator from a fresh draw of the true SDE, and a
               score *past* the floor is a red flag rather than a win. Awarding
               it to whoever is furthest past the floor would invert the meaning
               of the number.
      "tie"    The gap between the best two methods is no larger than the sum of
               their cross-seed stds. Five seeds does not resolve a gap that
               small, so we decline to call it.
      <name>   A win that survives both tests.

    `direction` is the shared convention from render_readme.CATEGORIES:
    "down" lower is better, "up" higher is better, "none" closest to `target`
    wins. Only A28_kurtosis_ratio is "none", and its label already reads "→ 1".
    """
    usable = [(n, mu, sd or 0.0) for n, mu, sd in zip(names, means, stds)
              if mu is not None]
    if len(usable) < 2:
        return "-", "n/a"

    # Fold "closest to target" into "lower is better" by comparing distances,
    # so the floor and tie tests below need only one code path.
    if direction == "none":
        usable = [(n, abs(mu - target), sd) for n, mu, sd in usable]
        floor = None if floor is None else abs(floor - target)
        direction = "down"

    sign = 1.0 if direction == "down" else -1.0   # after this, smaller is better
    if floor is not None and any(sign * mu <= sign * floor for _, mu, _ in usable):
        return FLOORED, "floor"

    ranked = sorted(usable, key=lambda t: sign * t[1])
    (best, mu0, sd0), (_, mu1, sd1) = ranked[0], ranked[1]
    if abs(mu1 - mu0) <= sd0 + sd1:
        return TIE, "tie"
    return f"**{best}**", best


def tally_sentence(methods, tally):
    """"**SBTS** wins 28, **LS4** wins 0, 2 are ties ... and 4 sit at the floor"."""
    parts = [f"**{m}** wins {tally.get(m, 0)}" for m in methods]
    ties, floored = tally.get("tie", 0), tally.get("floor", 0)
    if ties:
        parts.append(f"{ties} {'is a tie' if ties == 1 else 'are ties'} "
                     f"inside one cross-seed std")
    if floored:
        parts.append(f"{floored} {'sits' if floored == 1 else 'sit'} at the "
                     f"independent-draw floor, where the row separates nothing")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# --------------------------------------------------------------------------- A
def render_a(methods, data, floor):
    """A1-A34, one mean +/- std column per method, floor, then Winner.

    Returns (table, tally, n_rows). The tally is accumulated here rather than by
    re-parsing the rendered markdown, so the paragraph under the table and the
    Winner column can never disagree.
    """
    head = ["Metric"] + list(methods) + ["Perfect floor", "Winner"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    tally, n_rows = {}, 0
    for cat, rows in R.CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            cells, scope, means, stds = [], "", [], []
            for m in methods:
                s = data[m].get(key)
                if s is None:
                    cells.append("-")
                    means.append(None)
                    stds.append(None)
                    continue
                cells.append(f"{R.fmt(s['mean'])} ± {R.fmt(s['std'])}")
                means.append(_f(s["mean"]))
                stds.append(_f(s["std"]))
                if s["scope"] == "native":
                    scope = " *(native d=8)*"
            f = floor.get(key)
            cells.append(R.fmt(f["mean"]) if f else "-")
            cell, tkey = verdict(methods, means, stds,
                                 _f(f["mean"]) if f else None, direction)
            cells.append(cell)
            tally[tkey] = tally.get(tkey, 0) + 1
            n_rows += 1
            lines.append(f"| {label}{R.ARROW[direction]}{scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines), tally, n_rows


# --------------------------------------------------------------------------- B
def render_b(methods, aggs, tvds, agg_f, tvd_f):
    """B curve-shape metrics, one mean +/- std column per method, then Winner.

    Plot order is the union over methods in first-seen order, so a method missing
    a plot leaves a dash rather than shifting every row that follows it.

    Returns (table, tally, n_rows, mse_tally). `mse_tally` is kept apart because
    the MSE row is what decides a plot -- the % err / NRMSE / CVaR rows are
    function-only diagnostics of the same curve, so counting all five as
    independent wins would quintuple-count one result.
    """
    head = ["Plot", "Measure"] + list(methods) + ["Perfect floor", "Winner"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    tally, mse_tally, n_rows = {}, {}, 0

    if any(tvds.get(m) for m in methods):
        cells, means, stds = [], [], []
        for m in methods:
            t = tvds.get(m)
            cells.append(f"{R.pct(t['mean'])} ± {R.pct(t['std'])}" if t else "-")
            means.append(_f(t["mean"]) if t else None)
            stds.append(_f(t["std"]) if t else None)
        cells.append(R.pct(tvd_f["mean"]) if tvd_f else "-")
        cell, tkey = verdict(methods, means, stds,
                             _f(tvd_f["mean"]) if tvd_f else None, "down")
        cells.append(cell)
        tally[tkey] = tally.get(tkey, 0) + 1
        n_rows += 1
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
            cells, means, stds = [], [], []
            for m in methods:
                blk = aggs.get(m, {}).get(prefix, {})
                cells.append(f"{f_(blk[mk]['mean'])} ± {f_(blk[mk]['std'])}"
                             if mk in blk else "-")
                means.append(_f(blk[mk]["mean"]) if mk in blk else None)
                stds.append(_f(blk[mk]["std"]) if mk in blk else None)
            cells.append(f_(blk_f[mk]["mean"]) if mk in blk_f else "-")
            cell, tkey = verdict(methods, means, stds,
                                 _f(blk_f[mk]["mean"]) if mk in blk_f else None,
                                 "down")
            cells.append(cell)
            tally[tkey] = tally.get(tkey, 0) + 1
            n_rows += 1
            if mk == "mse":
                mse_tally[tkey] = mse_tally.get(tkey, 0) + 1
            plot = f"| **{R.esc(name)}** " if first else "|  "
            first = False
            lines.append(plot + f"| {mlabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines), tally, n_rows, mse_tally


def main():
    ap = argparse.ArgumentParser()
    # The default MUST list every method currently on the committed page. This is not
    # cosmetic: the renderer is not auto-discovering, so a stale default silently drops a
    # column and rewrites README.md with one method missing -- it happened on 2026-08-26,
    # when the default still read "SBTS,LS4,reference" and re-rendering deleted CSDI from a
    # page that had shipped with it. Append your method here as well as to the command line.
    ap.add_argument("--methods",
                    default="SBTS,LS4,CSDI,Deep-MKV-TS,Deep-MKV-TS-SBTSref,reference",
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

    # `reference` competes on equal terms in the Winner column, but a reader who
    # does not know what it is will misread the tally as "an untrained model beat
    # LS4". It did -- and that is the point -- but only if the column is labelled.
    # Emitted conditionally so the paragraph can never describe an absent column.
    ref_note = ""
    if "reference" in methods:
        ref_note = (
            "\n**`reference` is not a trained generator.** It is σ<sup>ref</sup>, the frozen "
            "parametric SDE that Deep-MKV-TS Algorithm 1 starts from, sampled with the neural "
            "control switched off and verified to be exactly zero — 19 fitted coefficients, "
            "**zero gradient steps**. It is adjudicated on equal terms above rather than shown "
            "as a passive baseline, because a row where a 19-coefficient SDE beats a trained "
            "network is a real result about that network, not a footnote. Read its column two "
            "ways: against **Deep-MKV-TS** it isolates what the learned control bought — and any "
            "row where Deep-MKV-TS scores *worse* is a row where Algorithm 1 damaged a working "
            "reference; against the other methods it is the bar a neural generator has to clear "
            "to justify itself. Generation seeds are shared with Deep-MKV-TS, so those two "
            "columns are **paired** — same Brownian streams, difference free of simulation "
            "noise. Details: [`reference/README.md`](reference/README.md).\n"
        )

    a_tbl, a_tally, a_rows = render_a(methods, data, floor)
    b_tbl, b_tally, b_rows, mse_tally = render_b(methods, aggs, tvds, agg_f, tvd_f)
    n_plots = sum(mse_tally.values())

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
{ref_note}
## A1–A34, mean ± std across {n_seeds} seeds

> ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every
> other row is computed on each of the 8 univariate slices and reported as the **mean over
> assets**. All metrics on **log-returns** unless noted; A26 uses price increments.

{a_tbl}

**How to read the tally.** Every cell above is an average twice over: a mean ± std across
the {n_seeds} seeds, of a quantity that — on every row *not* marked *(native d=8)* — is itself
already the mean over the 8 assets. So the std carries both seed-to-seed and asset-to-asset
variation, and a row is only decided when the gap between two methods survives it. Across
the **{a_rows} A rows**: {tally_sentence(methods, a_tally)}. A row is *not* awarded to anyone when
either method has reached the independent-draw floor — past the floor the metric stops
measuring fidelity — or when the gap is no wider than the summed stds.

## B, curve-shape metrics, mean ± std across {n_seeds} seeds

> All ↓ lower is better. Each stylised-fact plot yields a curve, not a scalar; the curve is
> computed per asset and averaged over the 8 assets before the combination. The **MSE row
> decides the winner** for a plot — the % err / NRMSE / CVaR rows are function-only
> diagnostics, reported because a method can win on MSE while failing in the tail.

{b_tbl}

**How to read the tally.** Each stylised-fact curve is computed per asset and averaged over
the 8 assets *before* the four summary measures are taken, then averaged again across the
{n_seeds} seeds — so, as in the A table, every cell is an average of averages. The headline count
is the MSE row, which is the one that decides a plot: across the **{n_plots} plots**,
{tally_sentence(methods, mse_tally)}. Counting every row in the table instead — all **{b_rows}** of them,
including the % err / NRMSE / CVaR diagnostics and the path-cloud TVD —
{tally_sentence(methods, b_tally)}. The two counts are given separately, and the first is the one to
quote, because the four diagnostic rows describe the *same* curve as their MSE row — they
exist to show *where* a curve fails, not to be counted as four extra wins.
"""
    out = os.path.join(MA, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    n_a = md.count("\n| A")
    print(f"Wrote {out}  ({len(md.splitlines())} lines, "
          f"{len(methods)} methods, {n_seeds} seeds, {n_a} A-rows)")


if __name__ == "__main__":
    main()
