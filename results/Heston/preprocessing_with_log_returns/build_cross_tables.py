#!/usr/bin/env python3
"""
build_cross_tables.py — cross-method comparison tables for the
`preprocessing_with_log_returns` experiment, in the EXACT display of the root
`results/README.md` (family super-columns, Perfect + Winner columns, category
sublabels, **bold** = best, win-count comments).

It REUSES the canonical renderer `metrics/render_tables.py` (fmt / ms / A_ROWS /
CURVE_PLOTS / winner_idx / header_html / render_A / render_B) unchanged, only
repointing its data loaders at THIS experiment's folders and shrinking the
method set to the methods carried through the pipeline:

    Diffusion          : CSDI    (results/.../preprocessing_with_log_returns/CSDI)
    Diffusion          : TimeDiT (results/.../preprocessing_with_log_returns/TimeDiT)
    VAE                : LS4     (results/.../preprocessing_with_log_returns/LS4)
    Schrödinger Bridge : SBTS    (results/.../preprocessing_with_log_returns/SBTS)

NOTE (TimeDiT / PS): TimeDiT is present in the A and B tables but **deliberately
not** in PS_MODELS below, and this is NOT a TODO. Log-return preprocessing
degrades TimeDiT (raw price wins the matched seed-0 control 31-3), so its single
1M bank was built from the *no-preproc* checkpoint and the strict protocol was
run on that -> results live in
`TimeDiT/baseline_no_preproc/path_shadowing/pdf_summary.json`, NOT in
`TimeDiT/path_shadowing/` (which holds only the shared tooling). Adding that
file as a PS column would place a raw-conditioned model beside three
preprocessed ones and silently break the apples-to-apples property this table
depends on. Do not "fix" this by adding the entry, and do not build a
preprocessed bank to make it symmetric — that experiment was explicitly
descoped (Theo, 2026-07-31).

Perfect floor = the independent-seed (1000+) 4096-path draw in
`perfect_recovery/results/` (compute_perfect_4096.py), scored against this
experiment's 4096 TEST set exactly as a generator is — the method-agnostic,
non-zero finite-sample floor at N=4096.

The B-curve aggregate (`curve_b_aggregate.json`) is not stored per-method here,
so it is DERIVED from the seed JSONs with the identical formula the root renderer
verified against `results/Heston/LS4/curve_b_aggregate.json`:
  mse   = per-seed mean of (funct + der + sec_der)/3  -> mean ± std across seeds
  pct   = funct_pct   (funct-only)                    -> mean ± std
  nrmse = funct_nrmse (funct-only)
  cvar90= funct_cvar90 ; cvar95 = funct_cvar95
grid_tvd is read from each seed JSON's scalar `grid_tvd` key.

The third table (PS) is NOT the murex PS-MC of the root renderer but the STRICT
paper protocol (arXiv:2308.01486): cum / step / rv CRPS (+ cum coverage90) read
verbatim from each generator's 1M-bank `path_shadowing/pdf_summary.json`, the
Heston-oracle and RW-floor rows inside those same files, and the two conditional
forecasters (`forecaster/{chronos2,timesfm}_pdf.json`, fine-tuned on the 4096
TRAIN set, forecasting the 512 ps queries, scored with the SAME metrics + boot
index). Winner excludes the oracle/RW floors, like Perfect excludes the floor.

Usage:
  /home/tbasseras/.cc-venv/bin/python build_cross_tables.py --which A
  /home/tbasseras/.cc-venv/bin/python build_cross_tables.py --which B
  /home/tbasseras/.cc-venv/bin/python build_cross_tables.py --which PS
  /home/tbasseras/.cc-venv/bin/python build_cross_tables.py            # all three
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../preprocessing_with_log_returns
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))    # repo root
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
sys.path.insert(0, METRICS_DIR)
import render_tables as RT                                              # noqa: E402  (canonical renderer)

PERFECT_DIR = os.path.join(HERE, "perfect_recovery", "results")

# ── shrink the method set to this experiment's methods ──────────────────────
# family order follows the root table (Diffusion before VAE). TimeDiT is a
# denoising diffusion transformer (DiT), so it shares CSDI's Diffusion family.
RT.FAMILIES = [
    ("Diffusion",          [("CSDI", "CSDI"), ("TimeDiT", "TimeDiT")]),
    ("VAE",                [("LS4", "LS4")]),
    ("Schrödinger Bridge", [("SBTS", "SBTS")]),
]
RT.METHOD_DIRS = [d for _, ms in RT.FAMILIES for d, _ in ms]
RT.METHOD_NAMES = [n for _, ms in RT.FAMILIES for _, n in ms]


# ── repoint every loader at THIS experiment ─────────────────────────────────
def _results_dir(method):
    return os.path.join(HERE, method)


RT.results_dir = _results_dir


def _seed_docs(rdir):
    docs = []
    for s in range(5):
        with open(os.path.join(rdir, f"seed_{s}_metrics.json")) as f:
            docs.append(json.load(f))
    return docs


def _curve_from_seeds(rdir):
    """Derive the curve_b_aggregate structure the root renderer expects, from the
    raw B_ keys in the seed JSONs (formula verified against the root aggregate)."""
    docs = _seed_docs(rdir)
    out = {}
    for prefix, _ in RT.CURVE_PLOTS:
        mse_ps = [(d[f"{prefix}_funct"] + d[f"{prefix}_der"] + d[f"{prefix}_sec_der"]) / 3.0
                  for d in docs]
        cols = {
            "mse":    mse_ps,
            "pct":    [d[f"{prefix}_funct_pct"] for d in docs],
            "nrmse":  [d[f"{prefix}_funct_nrmse"] for d in docs],
            "cvar90": [d[f"{prefix}_funct_cvar90"] for d in docs],
            "cvar95": [d[f"{prefix}_funct_cvar95"] for d in docs],
        }
        out[prefix] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                           "per_seed": [float(x) for x in v]}
                       for k, v in cols.items()}
    return out


def _load_curve(method):
    return _curve_from_seeds(_results_dir(method))


def _load_grid_tvd(method):
    docs = _seed_docs(_results_dir(method))
    v = [float(d["grid_tvd"]) for d in docs]
    return {"name": "grid_tvd", "n_bins": [50, 50],
            "mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}


RT.load_curve = _load_curve
RT.load_grid_tvd = _load_grid_tvd


def _perfect_floor_a_stats():
    docs = _seed_docs(PERFECT_DIR)
    out = {}
    for _, key, _ in RT.A_ROWS:
        if key is None:
            continue
        vals = [d[key] for d in docs if key in d and d[key] is not None]
        out[key] = (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
    return out


def _perfect_floor_a():
    return {k: (v[0] if v else None) for k, v in _perfect_floor_a_stats().items()}


def _perfect_floor_curve():
    return _curve_from_seeds(PERFECT_DIR)


def _perfect_floor_grid_tvd():
    docs = _seed_docs(PERFECT_DIR)
    v = [float(d["grid_tvd"]) for d in docs]
    return {"name": "grid_tvd", "n_bins": [50, 50],
            "mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}


RT.perfect_floor_a_stats = _perfect_floor_a_stats
RT.perfect_floor_a = _perfect_floor_a
RT.perfect_floor_curve = _perfect_floor_curve
RT.perfect_floor_grid_tvd = _perfect_floor_grid_tvd


# ── B table (experiment-local: RT.render_B hardcodes the main-benchmark perfect
#    curve path, so we mirror its structure here with THIS experiment's loaders).
def render_B_exp():
    curves = {m: _load_curve(m) for m in RT.METHOD_DIRS}
    gtvd = {m: _load_grid_tvd(m) for m in RT.METHOD_DIRS}
    perfect = _perfect_floor_curve()             # perfect seed JSONs (no grid_tvd)
    plot_wins = {m: 0 for m in RT.METHOD_NAMES}
    subline_wins = {m: 0 for m in RT.METHOD_NAMES}

    sup = ['  <tr>', '    <th rowspan="2">Plot</th>', '    <th rowspan="2">Measure</th>']
    for fam, methods in RT.FAMILIES:
        n = len(methods)
        sup.append(f'    <th colspan="{n}">{fam}</th>' if n > 1 else f'    <th>{fam}</th>')
    sup += ['    <th rowspan="2">Perfect</th>', '    <th rowspan="2">Winner</th>', '  </tr>']
    sub = ['  <tr>'] + [f'    <th>{n}</th>' for n in RT.METHOD_NAMES] + ['  </tr>']
    out = ["<table>", "<thead>"] + sup + sub + ["</thead>", "<tbody>"]

    # grid_tvd 'Path comparison' ranked first row (no perfect floor here)
    gmeans = [gtvd[m]["mean"] for m in RT.METHOD_DIRS]
    gstds = [gtvd[m]["std"] for m in RT.METHOD_DIRS]
    gwi = RT.winner_idx(gmeans, "lo")
    gwin = RT.METHOD_NAMES[gwi] if gwi is not None else "-"
    if gwi is not None:
        plot_wins[gwin] += 1
        subline_wins[gwin] += 1
    grow = ['  <tr>',
            '<td><b>Path comparison</b><br><sub>grid_tvd 50×50 path-cloud</sub></td>',
            '<td>grid_tvd 50×50 (%) ↓</td>']
    for i in range(len(RT.METHOD_DIRS)):
        grow.append(RT.cell(f"{RT.fmt(gmeans[i])}% ± {RT.fmt(gstds[i])}%", bold=(i == gwi)))
    grow.append('<td>-</td>')
    grow.append(f'<td><b>{gwin}</b></td></tr>')
    out.append("".join(grow))

    for prefix, name in RT.CURVE_PLOTS:
        pf = perfect[prefix] if perfect else None
        measures = ("mse", "pct", "nrmse", "cvar90", "cvar95")
        pct_measures = ("pct", "nrmse", "cvar90", "cvar95")
        nmeas = len(measures)
        for mi, measure in enumerate(measures):
            means = [curves[m][prefix][measure]["mean"] for m in RT.METHOD_DIRS]
            wi = RT.winner_idx(means, "lo")
            win_name = RT.METHOD_NAMES[wi] if wi is not None else "-"
            if wi is not None:
                subline_wins[win_name] += 1
                if measure == "mse":
                    plot_wins[win_name] += 1
            row = ['  <tr>']
            if mi == 0:
                row.append(f'<td rowspan="{nmeas}"><b>{name}</b></td>')
            meas_label = {"mse": "MSE", "pct": "% err", "nrmse": "NRMSE",
                          "cvar90": "CVaR₉₀", "cvar95": "CVaR₉₅"}[measure]
            row.append(f'<td>{meas_label}</td>')
            for i, m in enumerate(RT.METHOD_DIRS):
                d = curves[m][prefix][measure]
                txt = RT.ms(d["mean"], d["std"])
                if measure in pct_measures:
                    txt = f"{RT.fmt(d['mean'])}% ± {RT.fmt(d['std'])}%"
                row.append(RT.cell(txt, bold=(i == wi)))
            if pf is not None:
                pm, psd = pf[measure]["mean"], pf[measure]["std"]
                if measure in pct_measures:
                    row.append(f'<td>{RT.fmt(pm)}% ± {RT.fmt(psd)}%</td>')
                else:
                    row.append(f'<td>{RT.ms(pm, psd)}</td>')
            else:
                row.append('<td>-</td>')
            row.append(f'<td><b>{win_name}</b></td>')
            row.append('</tr>')
            out.append("".join(row))
    out += ["</tbody>", "</table>"]
    return "\n".join(out), plot_wins, subline_wins


# ── strict-PDF PS table (this experiment's own — not the murex PS-MC) ────────
PS_MODELS = [
    ("CSDI",       "CSDI/path_shadowing/pdf_summary.json",   "gen"),
    ("LS4",        "LS4/path_shadowing/pdf_summary.json",    "gen"),
    ("SBTS",       "SBTS/path_shadowing/pdf_summary.json",   "gen"),
    ("Chronos-2",  "forecaster/chronos2_pdf.json",           "fc"),
    ("TimesFM",    "forecaster/timesfm_pdf.json",            "fc"),
]
PS_QUANT = [("cum", "cum (M×H trajectory)"), ("step", "step (M×H)"), ("rv", "rv (M scalar)")]

# (json_key, display label, ranking rule). rule: "min" = lower wins;
# float = calibration target, closest wins; "diag" = pure diagnostic, no winner
# (width alone is meaningless without its coverage — a degenerate 0-width interval
# would "win" by min while being completely uncalibrated).
PS_METRICS = [
    ("rmse",         "RMSE ↓",               "min"),
    ("crps",         "CRPS ↓",               "min"),
    ("coverage50",   "coverage₅₀ (→0.50)",   0.50),
    ("coverage90",   "coverage₉₀ (→0.90)",   0.90),
    ("width50",      "width₅₀ (diag)",       "diag"),
    ("width90",      "width₉₀ (diag)",       "diag"),
    ("lower_miss90", "lower miss₉₀ (→0.05)", 0.05),
    ("upper_miss90", "upper miss₉₀ (→0.05)", 0.05),
]


def ps_bank_sizes():
    """The GUIDELINE §9.1 nested-prefix sweep, read from the data rather than
    hardcoded so a protocol change propagates automatically."""
    d = json.load(open(os.path.join(HERE, PS_MODELS[0][1])))
    return sorted(d["by_bank_size"], key=int)


def _gen_quantities(path, bank_size):
    """Quantities dict {cum,step,rv:{metric:{value,ci}}} at one nested-prefix bank
    size, from a generator pdf_summary.json."""
    d = json.load(open(path))
    return d["by_bank_size"][str(bank_size)]["quantities"]


def _fc_quantities(path):
    """Forecasters retrieve nothing, so they have no bank-size axis: the same
    values are repeated in every bank-size table (flagged by a README footnote)."""
    return json.load(open(path))["quantities"]


def _ps_value(path, kind, qn, metric, bank_size):
    q = _gen_quantities(path, bank_size) if kind == "gen" else _fc_quantities(path)
    return q[qn][metric]["value"]


def _ps_winner_idx(vals, rule):
    """rule: 'min' -> argmin; float target -> closest; 'diag' -> None (no winner)."""
    if rule == "min":
        return int(np.argmin(vals))
    if rule == "diag":
        return None
    return int(np.argmin([abs(v - rule) for v in vals]))


def render_PS_strict(bank_size=1000000):
    """Full strict-PDF PS comparison at ONE nested-prefix bank size: all 8 metrics
    × 3 quantities (24 values) for every entry of PS_MODELS, with the Heston-oracle
    (= perfect floor, K=256 draws from the true SDE) and RW-floor columns and a
    per-row Winner (best model, floors excluded). Width rows are diagnostics (no
    winner: width alone is meaningless without its coverage). Mirrors the root PS
    block's HTML shape.

    The generator and oracle columns both move with `bank_size` (nested prefixes of
    the SAME bank, GUIDELINE §9.1); the forecaster columns and the RW floor have no
    bank-size axis and are therefore identical in every table."""
    # every generator carries identical oracle / rw rows -> read from the first gen.
    gen_path = os.path.join(HERE, PS_MODELS[0][1])
    full = json.load(open(gen_path))
    oracle_q = full["heston_oracle"]["by_bank_size"][str(bank_size)]["quantities"]
    rw_q = full["rw_baseline"]

    model_names = [n for n, _, _ in PS_MODELS]
    paths_kinds = [(os.path.join(HERE, p), k) for _, p, k in PS_MODELS]
    ncol = 1 + len(PS_MODELS) + 3  # Metric + models + Oracle + RW + Winner
    # colspans are COMPUTED from PS_MODELS, not hardcoded, so adding a generator
    # later keeps the header aligned. (Not applicable to TimeDiT — see the
    # module docstring for why it is intentionally excluded.)
    n_gen = sum(1 for _, _, k in PS_MODELS if k == "gen")
    n_fc = sum(1 for _, _, k in PS_MODELS if k == "fc")
    bs_label = f"{int(bank_size):,}".replace(",", " ")
    out = ["<table>", "<thead>",
           "  <tr>",
           '    <th rowspan="2">Quantity / metric</th>',
           f'    <th colspan="{n_gen}">Generators ({bs_label} bank)</th>',
           f'    <th colspan="{n_fc}">Forecasters (fine-tuned @4096)<br><sup>bank-independent</sup></th>',
           f'    <th rowspan="2">Heston oracle<br>(perfect floor, {bs_label})</th>',
           '    <th rowspan="2">RW floor<br><sup>bank-independent</sup></th>',
           '    <th rowspan="2">Winner</th>',
           "  </tr>",
           "  <tr>"]
    out += [f"    <th>{n}</th>" for n in model_names]
    out += ["  </tr>", "</thead>", "<tbody>"]

    wins = {n: 0 for n in model_names}

    for qn, qlabel in PS_QUANT:
        out.append('  <tr><td colspan="%d"><b>%s</b></td></tr>' % (ncol, qlabel))
        for metric, mlabel, rule in PS_METRICS:
            vals = [_ps_value(p, k, qn, metric, bank_size) for p, k in paths_kinds]
            wi = _ps_winner_idx(vals, rule)
            if wi is not None:
                wins[model_names[wi]] += 1
            cells = "".join(RT.cell(RT.fmt(v), bold=(wi is not None and i == wi))
                            for i, v in enumerate(vals))
            ov = RT.fmt(oracle_q[qn][metric]["value"])
            rv = RT.fmt(rw_q[qn][metric]["value"])
            win_cell = f"<b>{model_names[wi]}</b>" if wi is not None else "—"
            out.append(f'  <tr><td>{mlabel}</td>{cells}'
                       f'<td>{ov}</td><td>{rv}</td><td>{win_cell}</td></tr>')

    out += ["</tbody>", "</table>"]
    return "\n".join(out), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B", "PS", "all"], default="all")
    ap.add_argument("--all-bank-sizes", action="store_true",
                    help="emit one PS table per nested-prefix bank size (GUIDELINE §9.1) "
                         "instead of the 1M table alone")
    args = ap.parse_args()

    if args.which in ("A", "all"):
        html, wins, total = RT.render_A()
        print("<!-- ===== A1-A34 CROSS-METHOD TABLE ===== -->")
        print(html)
        print(f"\n<!-- A win-counts (of {total}): " +
              ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1])) + " -->\n")
    if args.which in ("B", "all"):
        html, plot_wins, subline_wins = render_B_exp()
        print("<!-- ===== B CURVE-SHAPE CROSS-METHOD TABLE ===== -->")
        print(html)
        print("\n<!-- B plot-level win-counts (MSE per plot + grid_tvd, of 7): " +
              ", ".join(f"{k}={v}" for k, v in sorted(plot_wins.items(), key=lambda x: -x[1]) if v) + " -->")
        print("<!-- B per-subline win-counts (grid_tvd + 6×5, of 31): " +
              ", ".join(f"{k}={v}" for k, v in sorted(subline_wins.items(), key=lambda x: -x[1]) if v) + " -->\n")
    if args.which in ("PS", "all"):
        sizes = ps_bank_sizes() if args.all_bank_sizes else ["1000000"]
        for bs in sizes:
            bs_label = f"{int(bs):,}".replace(",", " ")
            html, wins = render_PS_strict(bs)
            print(f"<!-- ===== PS STRICT-PDF CROSS-METHOD TABLE — bank {bs_label} ===== -->")
            print(html)
            print(f"\n<!-- PS strict win-counts @{bs_label} (of 18 ranked rows = 6 ranked metrics × 3 quantities; width rows are diagnostic): " +
                  ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1]) if v) + " -->\n")


if __name__ == "__main__":
    main()
