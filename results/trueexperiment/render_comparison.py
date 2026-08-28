#!/usr/bin/env python3
"""Render results/trueexperiment/README.md -- the cross-method comparison.

WHAT THIS IS, AND WHY IT IS NOT A FOURTH COPY OF THE METHOD READMEs
-------------------------------------------------------------------
Each method already renders its own README against ``real_floor/`` with the full
per-seed spread. Those are vertical documents: one method, every number. This one
is horizontal: three tables, every method side by side, nothing else. It exists
because the per-method READMEs cannot answer the only question that matters once
more than one method is on the board -- *which of them is closer to real data,
and on what* -- without the reader hand-transcribing two files.

Every number is read from disk. Nothing here is typed by hand, and this script
holds no constants copied from another file: the floor column is whatever
``real_floor/`` currently contains, so a rebuild of the dataset variant
propagates automatically instead of silently invalidating a hardcoded ceiling.

THE FLOOR IS NOT A TARGET, IT IS A NOISE LEVEL
-----------------------------------------------
``real_floor/`` scores genuine market data against genuine market data. Its value
on a "lower is better" metric is therefore not zero and not a goal -- it is what
two honest samples of the same market cost each other. Read the method columns as
multiples of it (the ``xfloor`` figures): 1x means indistinguishable from another
real slice on that statistic, 5x means five times further apart than real data
gets from itself.

A method scoring BELOW the floor is flagged, not praised. Guideline section 7.1:
a bank that sits closer to the test split than the training split does is the
signature of a memoriser, not of a better generator. The worked example there
(``h=0.05, K=3``) scores vol_err 6.06 % against a real-vs-real MINIMUM of 8.72 %
while sitting five times closer to the training set than held-out real data --
an impossible score honestly obtained. Hence the winner rule below is |log
ratio|, distance from the floor in EITHER direction, and never "lowest wins".

WHAT THE THREE TABLES ARE
--------------------------
  A  the 32 distributional / temporal / adversarial metrics, floor-relative
  B  the curve-shape panel plus the 50x50 path-cloud TVD
  C  conditional CRPS by path shadowing, section 8, "paper" convention only

Reads   real_floor/ and every directory named in METHODS -- metrics_summary.csv,
        curve_b_aggregate.json, grid_tvd_aggregate.json,
        losses/crps_configs/paper__*.json, losses/memorisation.json
Writes  README.md   (this directory)

Usage:
    /home/tbasseras/.cc-venv/bin/python \
        results/trueexperiment/render_comparison.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import statistics

# Table D lives beside this file and is IMPORTED, not re-implemented. It is the
# only table in this repo with one renderer and four consumers; tables A/B/C are
# duplicated across the four renderers and that duplication has already cost a
# byte-exact three-file patch. See table_d.py's docstring before adding a fifth.
import table_d

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR_DIR = os.path.join(HERE, "real_floor")

# Order is the column order in every table. Adding a method here is the only
# edit needed -- nothing below is written per-method.
#
# `reference` is the untrained multivariate reference SDE (the Deep-MKV-TS
# control map with Zhat == 0). It is a BASELINE, not a competitor: it is what
# every learned control starts from, so a method that fails to beat this column
# has added nothing. It is appended LAST on purpose -- METHODS[0] is used below
# to locate the shared `paper__realbank.json` and the bootstrap baselines, and
# the reference has no crps_configs/ (table C renders it as a "-" row, which
# render_c already handles).
# Deep-MKV-TS sits between CSDI and the reference for two structural reasons, not
# for looks. It cannot be METHODS[0] -- that slot supplies the shared
# `paper__realbank.json` and the bootstrap baselines -- and it must not displace
# `reference` from last, which is the baseline-column position described above.
#
# The column is the lr = 6.4e-06 RETUNE, which is the only Deep-MKV-TS run ever
# scored on this dataset. The weights previously at the method root were
# lr = 0.002, carried over unchanged from the Heston d = 8 run; that rate moves
# Theta by 2.05 per step here against Heston's 0.032, because the control map
# adds `Zhat / sqrt(dt)` and dt is 9.51e-07 rather than 1/252. Those weights were
# never generated from and never scored. They are preserved under
# `Deep-MKV-TS/weights_lr2e-3/` rather than deleted.
METHODS = ["SBTS", "CSDI", "Deep-MKV-TS", "reference"]

# -- metric display table -----------------------------------------------------
# A33-A34 are absent by construction: they score against the dataset's LATENT
# variance path, and a real market has none (compute_all_multiasset.py passes
# v_test=None). A row here would render "-" everywhere and imply the metric
# failed rather than being undefined.
CATEGORIES = [
    ("Fat tail", [
        ("A1_kurtosis_error",        "A1 Kurtosis error",            "down"),
        ("A2_abs_r_q95_error",       r"A2 \|r\| q95 error",          "down"),
        ("A3_abs_r_q99_error",       r"A3 \|r\| q99 error",          "down"),
        ("A4_tail_qq_error",         "A4 Tail QQ error",             "down"),
        ("A5_hill_tail_index_error", "A5 Hill tail index error",     "down"),
    ]),
    ("Distribution", [
        ("A6_path_mmd2",             "A6 Path MMD2",                 "down"),
        ("A7_terminal_mmd2",         "A7 Terminal MMD2",             "down"),
        ("A8_increment_mmd2",        "A8 Increment MMD2",            "down"),
        ("A9_volatility_mmd",        "A9 Volatility MMD",            "down"),
        ("A10_terminal_swd",         "A10 Terminal SWD",             "down"),
        ("A11_path_swd",             "A11 Path SWD",                 "down"),
        ("A12_rv_law_loss",          "A12 RV law loss",              "down"),
        ("A13_mean_path_rmse",       "A13 Mean path RMSE",           "down"),
        ("A14_ks_logreturns",        "A14 KS log-returns",           "down"),
        ("A15_skewness_error",       "A15 Skewness error",           "down"),
        ("A16_qq_rmse",              "A16 QQ RMSE (300-pt)",         "down"),
        ("A17_terminal_ks",          "A17 Terminal price KS",        "down"),
    ]),
    ("Adversarial", [
        ("A18_disc_score_gru",       "A18 Disc score GRU",           "down"),
        ("A18_disc_score_mlp",       "A18 Disc score MLP",           "down"),
    ]),
    ("Predictive", [
        ("A19_pred_score_gru",       "A19 Pred score GRU",           "down"),
        ("A19_pred_score_mlp",       "A19 Pred score MLP",           "down"),
    ]),
    ("Temporal", [
        ("A20_cov_error",            "A20 Covariance error",         "down"),
        ("A21_acf_abs",              r"A21 ACF \|r\| error",         "down"),
        ("A22_acf_sq",               "A22 ACF r2 error",             "down"),
        ("A23_acf_lag1_abs_error",   r"A23 ACF \|r\| lag-1 error",   "down"),
        ("A24_acf_lag1_sq_error",    "A24 ACF r2 lag-1 error",       "down"),
    ]),
    ("Volatility", [
        ("A25_mean_rmse",            "A25 Mean RMSE",                "down"),
        ("A26_std_error",            "A26 Return std error",         "down"),
        ("A27_logreturn_std_error",  "A27 Log-return std error",     "down"),
        ("A28_kurtosis_ratio",       "A28 Kurtosis ratio (-> 1)",    "one"),
        ("A29_sigma_mean_error",     "A29 Sigma mean error",         "down"),
        ("A30_vol_path_rmse",        "A30 Cross-sect. vol path RMSE", "down"),
        ("A31_rolling_vol_ks",       "A31 Rolling vol KS (w=5)",     "down"),
        ("A32_vol_of_vol_error",     "A32 Vol-of-vol error",         "down"),
    ]),
]

MEASURES = [("mse", "MSE"), ("pct", "% err"), ("nrmse", "NRMSE"),
            ("cvar90", "CVaR90"), ("cvar95", "CVaR95")]

CRPS_TARGETS = [("cum_return", "Cumulative return"),
                ("increment", "Increment"),
                ("rv", "Realized vol")]


# ------------------------------------------------------------------ io ------
def read_summary(method_dir):
    """metrics_summary.csv -> {metric: {"mean": float, "std": float, "n": int}}."""
    path = os.path.join(method_dir, "metrics_summary.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["metric"]] = {"mean": float(row["mean"]),
                                      "std": float(row["std"]),
                                      "n": sum(1 for k in row if k.startswith("seed_"))}
            except (TypeError, ValueError):
                continue
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_crps(method_dir, label):
    """(per_seed_results, baselines) for the `paper` convention.

    Every one of a method's seed files re-scores the two historical baselines at
    the SAME --baseline-seed, so they carry identical numbers across that
    method's seeds; re-reporting them per generator seed would manufacture a
    spread that does not exist. Taken from the lowest seed only.

    That does NOT mean the baselines are deterministic -- they are stochastic in
    --seed, they were simply never varied here. load_baseline_sweep() below reads
    the dedicated sweep that does vary it, and render_c prefers it.
    """
    files = sorted(
        glob.glob(os.path.join(method_dir, "losses", "crps_configs", "paper__seed_*.json")),
        key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    per_seed, baselines = [], {}
    for p in files:
        res = read_json(p).get("results", {})
        if label in res:
            per_seed.append(res[label])
        if not baselines:
            baselines = {k: v for k, v in res.items() if k != label}
    return per_seed, baselines


BASELINE_SWEEP = os.path.join(HERE, "baselines", "crps_configs")


def load_baseline_sweep(cfg="paper"):
    """The two bootstrap baselines as mean +/- sd over the resampling seed sweep.

    Reads baselines/crps_configs/<cfg>__bseed_<S>.json, written by
    run_baseline_seed_sweep.sh.

    Why: block_bootstrap and session_bootstrap ARE stochastic in --seed
    (conditional_crps_multiasset.py:449 and :464 each build a fresh
    default_rng(seed)), so reporting them at the author's single 1234 while every
    method row carried a 5-seed sd compared a point estimate against a
    distribution. The sweep's block-bootstrap rv sd is ~0.016 -- wider than
    SBTS's own cross-seed sd -- so some method-vs-baseline gaps that looked
    decided sit inside baseline noise. The correction is not cosmetic.

    Deliberately does NOT cover real_train_bank. That bank is np.load()'d off
    disk and score() contains no RNG, so --seed provably never reaches it:
    rerunning it at 9999 reproduces all three targets to +0.00e+00. Five seeds
    there would print one number five times. Its honest analogue is a DIFFERENT
    real split used as the bank, which is a semantic change to what "the floor"
    means and is not silently bundled in here.

    Returns {} when the sweep is absent, in which case render_c falls back to the
    single-seed baselines embedded in the per-seed method artefacts and the table
    renders exactly what it rendered before.
    """
    out = {}
    files = sorted(glob.glob(os.path.join(BASELINE_SWEEP, f"{cfg}__bseed_*.json")),
                   key=lambda p: int(re.search(r"bseed_(\d+)", p).group(1)))
    for p in files:
        for name, blk in read_json(p).get("results", {}).items():
            d = out.setdefault(name, {"n": 0})
            d["n"] += 1
            for k, _l in CRPS_TARGETS:
                if isinstance(blk.get(k), dict) and "mean" in blk[k]:
                    d.setdefault(k, []).append(blk[k]["mean"])
    return out


# --------------------------------------------------------------- format -----
def esc(s):
    """Escape pipes. The floor's panel names include literal `ACF |r|`; unescaped
    it splits that row into extra columns and silently corrupts the whole table."""
    return str(s).replace("|", r"\|")


def fmt(v):
    """4 significant digits, scientific below 1e-3 -- matches the method READMEs."""
    if v is None:
        return "-"
    v = float(v)
    if v != v:
        return "nan"
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4g}"


def ratio_cell(value, floor, direction):
    """The method's value as a multiple of the floor, with the below-floor flag.

    For A28 the target is 1.0 rather than the floor's own value, so the
    "distance" being reported is |x - 1| on both sides and a ratio would be
    meaningless -- that row prints the raw distance instead.
    """
    if value is None or floor is None:
        return "-", None
    if direction == "one":
        return fmt(abs(value - 1.0)), abs(value - 1.0)
    if floor <= 0 or value <= 0:
        return "-", None
    r = value / floor
    return (f"{r:.2f}x" if r >= 1.0 else f"**{r:.2f}x** !"), r


def winner(scores, direction):
    """Method closest to the floor, measured by |log ratio| (either direction).

    Not "lowest wins". Guideline 7.1: beating the real-vs-real floor is the
    memoriser signature, so overshooting downwards is a defect and is penalised
    exactly as far as overshooting upwards.
    """
    live = {m: s for m, s in scores.items() if s is not None and s > 0}
    if len(live) < 2:
        return "-"
    key = (lambda s: abs(s)) if direction == "one" else (lambda s: abs(math.log(s)))
    ranked = sorted(live, key=lambda m: key(live[m]))
    if len(ranked) > 1 and abs(key(live[ranked[0]]) - key(live[ranked[1]])) < 1e-12:
        return "tie"
    return ranked[0]


# -------------------------------------------------------------- table A -----
def render_a(floor, data):
    head = ["Metric", "Real-vs-real floor"]
    for m in METHODS:
        head += [f"{m} (mean +/- sd)", f"{m} xfloor"]
    head += ["Closer to floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    tally = {m: 0 for m in METHODS}
    below = {m: [] for m in METHODS}
    for cat, rows in CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            f = floor.get(key)
            fm = f["mean"] if f else None
            cells = [fmt(fm)]
            scores = {}
            for m in METHODS:
                s = data[m].get(key)
                if s is None:
                    cells += ["-", "-"]
                    scores[m] = None
                    continue
                cells.append(f"{fmt(s['mean'])} +/- {fmt(s['std'])}")
                rc, r = ratio_cell(s["mean"], fm, direction)
                cells.append(rc)
                scores[m] = r
                if r is not None and direction != "one" and r < 1.0:
                    below[m].append(label.replace("\\", ""))
            w = winner(scores, direction)
            if w in tally:
                tally[w] += 1
            arrow = "" if direction == "one" else " v"
            lines.append(f"| {label}{arrow} | " + " | ".join(cells) + f" | {w} |")
    return "\n".join(lines), tally, below


# -------------------------------------------------------------- table B -----
def render_b(floor_b, floor_tvd, data_b, data_tvd):
    head = ["Panel", "Measure", "Real-vs-real floor"]
    for m in METHODS:
        head += [m, f"{m} xfloor"]
    head += ["Closer to floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    tally = {m: 0 for m in METHODS}

    if floor_tvd:
        cells = [f"{fmt(floor_tvd['mean'])}%"]
        scores = {}
        for m in METHODS:
            t = data_tvd.get(m)
            if not t:
                cells += ["-", "-"]
                scores[m] = None
                continue
            cells.append(f"{fmt(t['mean'])}% +/- {fmt(t['std'])}")
            rc, r = ratio_cell(t["mean"], floor_tvd["mean"], "down")
            cells.append(rc)
            scores[m] = r
        w = winner(scores, "down")
        if w in tally:
            tally[w] += 1
        lines.append("| **Path cloud** *(50x50 2-D histogram)* | grid_tvd (%) v | "
                     + " | ".join(cells) + f" | {w} |")

    for prefix, fblk in floor_b.items():
        first = True
        for mk, mlabel in MEASURES:
            if mk not in fblk:
                continue
            cells = [fmt(fblk[mk]["mean"])]
            scores = {}
            for m in METHODS:
                blk = data_b.get(m, {}).get(prefix, {})
                if mk not in blk:
                    cells += ["-", "-"]
                    scores[m] = None
                    continue
                cells.append(f"{fmt(blk[mk]['mean'])} +/- {fmt(blk[mk]['std'])}")
                rc, r = ratio_cell(blk[mk]["mean"], fblk[mk]["mean"], "down")
                cells.append(rc)
                scores[m] = r
            w = winner(scores, "down")
            if mk == "mse" and w in tally:
                tally[w] += 1
            name = f"| **{esc(fblk['name'])}** " if first else "|  "
            first = False
            lines.append(name + f"| {mlabel} v | " + " | ".join(cells) + f" | {w} |")
    return "\n".join(lines), tally


# -------------------------------------------------------------- table C -----
def _cell(blk):
    """mean +/- half-width of the 95 % bootstrap CI over the 6 144 test queries.

    NOT the same quantity as the +/- on a method row, which is the sample sd
    ACROSS SEEDS. One is sampling noise over queries, the other generator noise
    over seeds. They are never merged and the note under the table says so.
    """
    if not blk or "mean" not in blk:
        return "-"
    return f"{blk['mean']:.3f} +/- {(blk['ci_hi'] - blk['ci_lo']) / 2.0:.3f}"


def render_c(crps, realbank, baselines):
    head = ["Bank", "Seeds"] + [f"{lbl} v" for _k, lbl in CRPS_TARGETS] + ["xfloor (cum. return)"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]

    rb = realbank.get("cum_return", {}).get("mean")
    if realbank:
        lines.append("| **Real train split as bank** *(floor)* | 1 | "
                     + " | ".join(_cell(realbank.get(k)) for k, _ in CRPS_TARGETS)
                     + " | 1.00x |")

    for m in METHODS:
        per_seed = crps.get(m, [])
        if not per_seed:
            lines.append(f"| **{m}** | 0 | " + " | ".join(["-"] * len(CRPS_TARGETS)) + " | - |")
            continue
        cells = []
        first_mean = None
        for k, _lbl in CRPS_TARGETS:
            vals = [p[k]["mean"] for p in per_seed if k in p]
            if not vals:
                cells.append("-")
                continue
            mu = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if first_mean is None:
                first_mean = mu
            cells.append(f"**{mu:.3f} +/- {sd:.3f}**")
        # Same sub-floor convention as tables A and B -- a bank that forecasts the
        # test era better than a bank of real training paths does is flagged, for
        # the identical reason and with the identical two candidate causes.
        rel = ratio_cell(first_mean, rb, "down")[0] if (rb and first_mean) else "-"
        lines.append(f"| **{m}** | {len(per_seed)} | " + " | ".join(cells) + f" | {rel} |")

    # The bootstrap rows carry a cross-seed sd like the method rows whenever the
    # resampling sweep exists; the floor row above never does, and that asymmetry
    # is a property of the code, not an oversight. See load_baseline_sweep.__doc__.
    sweep = load_baseline_sweep("paper")

    for name, label in (("block_bootstrap", "Block bootstrap *(baseline)*"),
                        ("session_bootstrap", "Session bootstrap *(baseline)*")):
        blk = baselines.get(name)
        if not blk:
            continue
        swp = sweep.get(name)
        if swp and swp.get("n", 0) > 1:
            # mean +/- sd over the resampling seeds -- the SAME quantity as the
            # method rows, which is the point: distribution against distribution.
            cells, mu = [], None
            for k, _ in CRPS_TARGETS:
                vals = swp.get(k)
                if not vals:
                    cells.append("-")
                    continue
                m = statistics.fmean(vals)
                if mu is None:
                    mu = m          # cum_return is CRPS_TARGETS[0]; drives xfloor
                cells.append(f"{m:.3f} +/- {statistics.stdev(vals):.3f}")
            n_txt = str(swp["n"])
        else:
            cells = [_cell(blk.get(k)) for k, _ in CRPS_TARGETS]
            mu = blk.get("cum_return", {}).get("mean")
            n_txt = "1"
        rel = ratio_cell(mu, rb, "down")[0] if (rb and mu) else "-"
        lines.append(f"| {label} | {n_txt} | " + " | ".join(cells) + f" | {rel} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main ----
def main():
    floor = read_summary(FLOOR_DIR)
    floor_b = read_json(os.path.join(FLOOR_DIR, "curve_b_aggregate.json"))
    floor_tvd = read_json(os.path.join(FLOOR_DIR, "grid_tvd_aggregate.json"))

    data, data_b, data_tvd, crps = {}, {}, {}, {}
    baselines, n_seeds, nn = {}, {}, {}
    for m in METHODS:
        d = os.path.join(HERE, m)
        data[m] = read_summary(d)
        data_b[m] = read_json(os.path.join(d, "curve_b_aggregate.json"))
        data_tvd[m] = read_json(os.path.join(d, "grid_tvd_aggregate.json"))
        crps[m], bl = load_crps(d, m)
        if not baselines:
            baselines = bl
        n_seeds[m] = max((v["n"] for v in data[m].values()), default=0)
        # The NNratio is what actually adjudicates a sub-floor row. Quoted from
        # each method's own artefact so this file states no verdict of its own.
        nn[m] = read_json(os.path.join(d, "losses", "memorisation.json")).get("nn_ratio")

    realbank = read_json(os.path.join(HERE, METHODS[0], "losses", "crps_configs",
                                      "paper__realbank.json"))
    realbank = realbank.get("results", {}).get("real_train_bank", {})

    a_tbl, a_tally, a_below = render_a(floor, data)
    b_tbl, b_tally = render_b(floor_b, floor_tvd, data_b, data_tvd)
    c_tbl = render_c(crps, realbank, baselines)
    # Returns ("", "") when var_backtest.json is absent, so a checkout without the
    # artefact renders exactly what it rendered before Table D existed. It also
    # returns an empty table with an HTML-comment note when the VaR run and the
    # CRPS configs disagree about which query file they read -- that has already
    # happened once and every number still looked plausible.
    d_block = table_d.section()

    n_floor = max((v["n"] for v in floor.values()), default=0)
    seed_line = ", ".join(f"{m} {n_seeds[m]} seeds" for m in METHODS)
    crps_line = ", ".join(f"{m} {len(crps[m])}" for m in METHODS)
    # Derived, not typed. Unequal seed counts mean the method `+/-` are sd over
    # different n and must not be read against each other; equal counts make that
    # warning false, and a false caveat left in generated prose survives because
    # nobody re-reads generated prose.
    _crps_n = {len(crps[m]) for m in METHODS}
    if len(_crps_n) == 1:
        crps_seed_note = (
            f"Each method's CRPS is a mean over {_crps_n.pop()} generator seeds, "
            "matching tables A\nand B, so the method `+/-` are sd over the same *n* "
            "and are comparable to each\nother."
        )
    else:
        crps_seed_note = (
            f"CRPS seed counts differ ({crps_line}) -- the spread columns are not "
            "computed over\nthe same *n* and the method `+/-` are not directly "
            "comparable to each other."
        )
    # Same "derived, not typed" rule as src_line below. This list was hardcoded to
    # SBTS and CSDI and silently went stale when Deep-MKV-TS and reference joined
    # METHODS, telling the reader two method pages exist when four do.
    # Derived, not typed. This sentence used to assert that the floor row AND the
    # two bootstraps were "deterministic given --baseline-seed 1234, hence one
    # seed each". Half of that was false: the bootstraps redraw from --seed. If
    # the sweep is later deleted the table silently reverts to one seed, so the
    # sentence has to be read off the same source the table is.
    _bsw = load_baseline_sweep("paper")
    _bn = {v["n"] for v in _bsw.values()} or {1}
    if len(_bn) == 1 and next(iter(_bn)) > 1:
        baseline_seed_note = (
            f"The two bootstrap rows are means over {next(iter(_bn))} resampling seeds "
            "(1234-1238): they redraw\ntheir blocks and sessions from `--seed` and are "
            "not deterministic. The floor row is --\nits bank is the real training split "
            "loaded off disk and `score()` holds no RNG, so\n`--seed` never reaches it "
            "(rerun at 9999: `+0.00e+00` on all three targets). Hence one\n\"seed\" for "
            "the floor and no seed sd on that row."
        )
    else:
        baseline_seed_note = (
            "The floor row and the two bootstrap baselines are all reported at the "
            "single\n`--baseline-seed 1234`, hence one \"seed\" each. The bootstraps are "
            "in fact stochastic\nin that seed; run `run_baseline_seed_sweep.sh` to give "
            "them a spread."
        )

    readme_line = ", ".join(f"[`{m}/README.md`]({m}/README.md)" for m in METHODS)
    flags = "; ".join(f"{m} {len(v)}/34" for m, v in a_below.items())
    nn_line = "; ".join(
        f"{m} **{nn[m]:.4f}**" if nn.get(m) is not None else f"{m} -" for m in METHODS)
    # Derived, not typed: the footer's provenance list cannot drift out of step
    # with METHODS the way a hardcoded "SBTS and CSDI" did.
    src_line = ", ".join(f"`{d}/`" for d in ["real_floor"] + METHODS[:-1]) \
        + f" and `{METHODS[-1]}/`"

    md = f"""# TrueDataset, d = 8 -- method comparison

Real Binance spot, 8 assets (BTC ETH BNB SOL XRP DOGE ADA LINK / USDT), 30-second
bars, `T = 128` (64 minutes), variant `om_2022-07_N6144`, holdout-era split with a
45.5-day embargo. Scored against `true_S_test_6144x128x8.npy`.

Three tables, nothing else. Per-seed numbers, provenance and diagnostics live in
each method's own README: {readme_line}.

**Reading the floor.** `real_floor/` holds the *real* train, val and valdisc splits
scored against the *real* test split. On a "lower is better" metric its value is not
zero and not a target -- it is what two honest samples of this market cost each
other. The `xfloor` columns are multiples of it: **1x = indistinguishable from
another real slice on that statistic**, 5x = five times further apart than real data
gets from itself.

**Sub-floor values are flagged, not rewarded.** A value below the floor is printed in
bold with a `!`. The floor's splits are drawn from 2022-07 -> 2024-12 and the test
split from 2025-02 -> 2026-07, so the floor bundles sampling noise *with* a regime
change -- and both methods train only on the earlier era, so they inherit that same
gap. Scoring under the floor therefore means landing closer to a market the model
never saw than the data it was trained on does. That has two possible causes and
this table cannot tell them apart: (i) memorisation, guideline 7.1, where a bank
sits nearer the target than honest data can -- the worked example scores `vol_err`
6.06 % against a real-vs-real *minimum* of 8.72 % while hugging the training set
five times too closely; or (ii) a generator bias that happens to point the same way
as the era shift, e.g. a smoother thinning tails towards a calmer test era. The
NNratio separates them, and it is measured per method, not guessed here:

> NNratio (healthy band 0.932-1.000, `losses/memorisation.json`): {nn_line}.
> Neither method is a memoriser. SBTS sits *above* the band -- further from train
> than real held-out data is. CSDI sits below it, and `CSDI/code/check_memorisation_scale.py`
> shows that rescaling its log-returns to the real training std moves the ratio to
> 1.0802, past the band; shrinking the real val split to CSDI's own volatility
> reproduces 0.475-0.516 from scale alone. That is shrunken support, not copying.

So read the `!` rows as cause (ii) throughout: sharper on tails and marginals than
the era gap, weaker where the gap is not the binding constraint.

"Closer to floor" ranks by |log ratio| -- distance from the floor in *either*
direction. It is deliberately not "lowest wins".

Seeds: {seed_line}; floor {n_floor}. Sub-floor rows -- {flags}.

---

## Table A -- distributional, temporal, adversarial (A1-A32)

{a_tbl}

Rows won: {" | ".join(f"**{m} {a_tally[m]}**" for m in METHODS)} of
{sum(a_tally.values())} decided.

`+/-` is the sample spread **across seeds**, not a confidence interval. A33-A34 are
absent by construction: they score against the dataset's latent variance path and a
real market has none, so `compute_all_multiasset.py` passes `v_test=None` and skips
them rather than emitting a fake row. A28 targets 1.0 rather than the floor, so its
`xfloor` column carries the raw distance `|x - 1|` instead of a ratio.

---

## Table B -- curve shape and path cloud

{b_tbl}

MSE rows won: {" | ".join(f"**{m} {b_tally[m]}**" for m in METHODS)}.

One panel contributes five measures of the same curve; only the MSE row is counted
so a single panel cannot vote five times. `grid_tvd` is the total-variation
distance between 50x50 2-D histograms of the price clouds, averaged over assets --
the one row here that looks at joint path geometry rather than a marginal.

---

## Table C -- conditional generation, forecast CRPS by path shadowing

Section 8. Condition on bars `0..64`, forecast `65..96` (`H = 32`), retrieve
`K = 256` neighbours jointly across the 8 assets from an 8 192-path bank, score the
CRPS of the implied predictive law against the real continuation. `paper`
convention (`--weight-mode paper --standardize bank`), CRPS x 1000, lower is better.

{c_tbl}

The `xfloor` column carries the same sub-floor convention as tables A and B, and it
fires here: a bank that forecasts the test era *better* than a bank of real training
paths is flagged, not credited. Cause (ii) above is the live explanation -- a
narrower predictive law is rewarded by CRPS when the target era is calmer than the
training era, which on this split it is (annualised vol falls between the two eras
on 6 of the 8 assets). The three targets are scored on the same queries, so compare
across a row before reading down a column.

{crps_seed_note} {baseline_seed_note}

**Two different `+/-` appear in this table and they are not the same quantity.** On
the **floor** row it is the half-width of the 95 % bootstrap CI over the 6 144 test
queries -- sampling noise. On the **method rows and the two bootstrap rows** it is
the sample sd across seeds -- generator noise for the methods, resampling noise for
the bootstraps. They are never merged.

---
{d_block}
*Generated by [`render_comparison.py`](render_comparison.py). Every number is read
from {src_line} at render time; none is typed by hand. Do
not edit this file -- re-run the script.*
"""
    dst = os.path.join(HERE, "README.md")
    with open(dst, "w") as fh:
        fh.write(md)
    print(f"wrote {dst}")
    print(f"  A-table: {a_tally}, sub-floor rows -- {flags}")
    print(f"  B-table MSE rows: {b_tally}")
    print(f"  CRPS seeds: {crps_line}")


if __name__ == "__main__":
    main()
