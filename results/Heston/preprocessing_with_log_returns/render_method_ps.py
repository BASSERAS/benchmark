#!/usr/bin/env python
"""Render (and inject) the three per-method path-shadowing tables at the 1M bank.

Each method README carries a *method vs Heston-oracle vs RW-floor* block for the
three forecast quantities. Those tables were hand-typed, and they drifted: when
cum/step moved to the textbook RMSE (GUIDELINE E16b) the cross-method tables in
the root README were regenerated but the per-method ones still showed the old
root-inside numbers. This script makes them generated, from the same
`build_cross_tables` row definitions and the same `_ps_key` routing, so drift is
no longer expressible.

    /home/tbasseras/gpu-venv/bin/python render_method_ps.py            # inject
    /home/tbasseras/gpu-venv/bin/python render_method_ps.py --print CSDI

Injection replaces the span from the "**Cumulative return" heading through the
blank line before the "**Reading it" paragraph -- prose is never touched.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_cross_tables as B  # noqa: E402

RT = B.RT
BANK = 1000000

# README stem -> (PS_MODELS name, column header used in that README)
METHOD_README = {
    "CSDI": ("CSDI", "CSDI+logret"),
    "LS4": ("LS4", "LS4+logret"),
    "SBTS": ("SBTS", "SBTS+logret"),
    "TimeDiT": ("TimeDiT (raw)", "TimeDiT raw"),
}

QUANT_HEADING = {
    "cum": "**Cumulative return (trajectory, u = 1…H)** — {m} vs Heston-oracle ceiling vs RW floor",
    "step": "**One-step return (trajectory, u = 1…H)**",
    "rv": "**Horizon realized vol (scalar) — the diagnostic quantity**",
}

# Compact row labels for the narrow 3-column per-method tables. The cross-method
# tables use arrows/subscripts; here the header already says the direction, so
# they would only add noise.
ROW_LABEL = {
    ("rv", "rmse"): "MAE",
    ("rv", "rmse_textbook"): "RMSE (true, root-last)",
    ("cum", "rmse"): "RMSE (textbook)",
    ("step", "rmse"): "RMSE (textbook)",
    "crps": "CRPS",
    "coverage50": "coverage 50",
    "coverage90": "coverage 90",
    "width50": "width 50",
    "width90": "width 90",
    "lower_miss90": "lower-miss 90",
    "upper_miss90": "upper-miss 90",
}


def _cell(leaf, bold=False):
    v = RT.fmt(leaf["value"])
    if bold:
        v = f"**{v}**"
    ci = leaf.get("ci")
    return f"{v} [{RT.fmt(ci[0])}, {RT.fmt(ci[1])}]" if ci else v


def method_ps_tables(readme_stem):
    name, col = METHOD_README[readme_stem]
    path = os.path.join(HERE, dict((n, p) for n, p, _ in B.PS_MODELS)[name])
    full = json.load(open(path))
    gen = full["by_bank_size"][str(BANK)]["quantities"]
    orc = full["heston_oracle"]["by_bank_size"][str(BANK)]["quantities"]
    rw = full["rw_baseline"]

    out = []
    for qn, _ in B.PS_QUANT:
        out += [QUANT_HEADING[qn].format(m=col), "",
                f"| metric | {col} | Heston oracle | RW floor |",
                "|--------|:----------:|:-------------:|:--------:|"]
        for metric, _mlabel, rule in B.ps_rows(qn):
            key = B._ps_key(qn, metric)
            leaves = [gen[qn][key], orc[qn][key], rw[qn][key]]
            wi = B._ps_winner_idx([l["value"] for l in leaves], rule)
            label = ROW_LABEL.get((qn, metric), ROW_LABEL.get(metric, metric))
            if rule == "min-uncounted":
                label += "†"
            cells = " | ".join(_cell(l, bold=(wi is not None and i == wi))
                               for i, l in enumerate(leaves))
            out.append(f"| {label} | {cells} |")
        out.append("")
    return "\n".join(out).rstrip("\n")


def _span(lines, start):
    """End of the three-table block: walk forward and stop the moment the THIRD
    markdown table has ended. Do NOT scan to the next prose heading -- TimeDiT
    carries a bank-size sweep and a diagnostics table between the rv table and
    its "Reading it" paragraph, and an end-marker based on that heading silently
    swallowed both the first time this ran."""
    seen, in_tbl, i = 0, False, start
    while i < len(lines) and seen < len(B.PS_QUANT):
        row = lines[i].startswith("|")
        if row and not in_tbl:
            in_tbl = True
        elif in_tbl and not row:
            in_tbl, seen = False, seen + 1
        i += 1
    if in_tbl:                       # file ended mid-table
        seen += 1
    assert seen == len(B.PS_QUANT), f"found {seen} tables, expected {len(B.PS_QUANT)}"
    return i - 1                     # index of the blank line closing table 3


def inject(readme_stem):
    p = os.path.join(HERE, readme_stem, "README.md")
    lines = open(p, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("**Cumulative return"))
    end = _span(lines, start)
    new = method_ps_tables(readme_stem).split("\n")
    open(p, "w", encoding="utf-8").write(
        "\n".join(lines[:start] + new + lines[end:]))
    return start + 1, end, len(new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", metavar="METHOD",
                    choices=sorted(METHOD_README), help="print instead of injecting")
    a = ap.parse_args()
    if a.show:
        print(method_ps_tables(a.show))
    else:
        for stem in sorted(METHOD_README):
            s, e, n = inject(stem)
            print(f"{stem}/README.md: replaced lines {s}-{e} with {n}")
