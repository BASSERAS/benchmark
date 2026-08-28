#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/trueexperiment/CSDI/README.md from the metric artefacts.

Section order and table shape mirror the SBTS entry in this same tree, so two
methods can be read side by side without re-learning the layout. The rendering
functions below are a verbatim copy of
`results/trueexperiment/SBTS/code/render_readme.py`; what differs is `main()`
(the prose is about a diffusion model, not a kernel), `render_timing` (CSDI logs
train/generate wall-clock and parameter counts, SBTS logs worker counts) and the
method-specific paragraphs of `render_memorisation`. **Keep the table-building
functions in sync with the SBTS copy** -- if `render_a_table` drifts between two
method trees, the two READMEs stop being comparable and the drift is invisible
because both still render.

Every number is READ FROM DISK, never typed by hand: the tables *and* the
headline counts are computed from the artefacts, so the prose cannot drift away
from the numbers it describes.

Reads
  results/trueexperiment/CSDI/metrics_summary.csv
  results/trueexperiment/CSDI/curve_b_aggregate.json
  results/trueexperiment/CSDI/grid_tvd_aggregate.json
  results/trueexperiment/CSDI/losses/generation_time.csv
  results/trueexperiment/CSDI/losses/memorisation.json
  results/trueexperiment/CSDI/losses/conditional_crps_seed_*.json
  results/trueexperiment/CSDI/losses/conditional_crps_realbank.json   (optional)
  results/trueexperiment/CSDI/weights/seed_0_config.json              (optional)
  results/trueexperiment/real_floor/metrics_summary.csv               (floor)
  results/trueexperiment/real_floor/curve_b_aggregate.json
  results/trueexperiment/real_floor/grid_tvd_aggregate.json

Writes
  results/trueexperiment/CSDI/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/CSDI/code/render_readme.py
"""

import csv
import glob
import json
import os
import re
import statistics
import sys

# Table D is a SHARED module (results/trueexperiment/table_d.py) imported by all
# four renderers, not a fourth copy. Tables A/B/C are duplicated across these
# files and that duplication has already cost a byte-exact three-file patch --
# read table_d.py's docstring before adding a fifth table anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import table_d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                            # .../CSDI
TE = os.path.dirname(METHOD_DIR)                              # .../trueexperiment
METHOD = os.path.basename(METHOD_DIR)                         # "CSDI"
FLOOR = os.path.join(TE, "real_floor")

FLOOR_COL = "Real-vs-real floor"

# -- metric display table -----------------------------------------------------
# A33-A34 are absent by construction: they score the generated path against the
# dataset's LATENT VARIANCE path, and a real market has none. compute_all_
# multiasset.py skips them (`v_test: None`), so a row here would render "-" in
# every column and imply the metric merely failed rather than being undefined.
CATEGORIES = [
    ("Fat Tail", [
        ("A1_kurtosis_error",        "A1 Kurtosis Error",           "down"),
        ("A2_abs_r_q95_error",       r"A2 \|r\| q95 Error",         "down"),
        ("A3_abs_r_q99_error",       r"A3 \|r\| q99 Error",         "down"),
        ("A4_tail_qq_error",         "A4 Tail QQ Error",            "down"),
        ("A5_hill_tail_index_error", "A5 Hill Tail Index Error",    "down"),
    ]),
    ("Distribution", [
        ("A6_path_mmd2",             "A6 Path MMD²",                "down"),
        ("A7_terminal_mmd2",         "A7 Terminal MMD²",            "down"),
        ("A8_increment_mmd2",        "A8 Increment MMD²",           "down"),
        ("A9_volatility_mmd",        "A9 Volatility MMD",           "down"),
        ("A10_terminal_swd",         "A10 Terminal SWD",            "down"),
        ("A11_path_swd",             "A11 Path SWD",                "down"),
        ("A12_rv_law_loss",          "A12 RV Law Loss",             "down"),
        ("A13_mean_path_rmse",       "A13 Mean Path RMSE",          "down"),
        ("A14_ks_logreturns",        "A14 KS Log-returns",          "down"),
        ("A15_skewness_error",       "A15 Skewness Error",          "down"),
        ("A16_qq_rmse",              "A16 QQ RMSE (300-pt)",        "down"),
        ("A17_terminal_ks",          "A17 Terminal Price KS",       "down"),
    ]),
    ("Adversarial", [
        ("A18_disc_score_gru",       "A18 Disc Score GRU",          "down"),
        ("A18_disc_score_mlp",       "A18 Disc Score MLP",          "down"),
    ]),
    ("Predictive", [
        ("A19_pred_score_gru",       "A19 Pred Score GRU",          "down"),
        ("A19_pred_score_mlp",       "A19 Pred Score MLP",          "down"),
    ]),
    ("Temporal", [
        ("A20_cov_error",            "A20 Covariance Error",        "down"),
        ("A21_acf_abs",              r"A21 ACF \|r\| Error (lags)", "down"),
        ("A22_acf_sq",               "A22 ACF r² Error (lags)",     "down"),
        ("A23_acf_lag1_abs_error",   r"A23 ACF \|r\| Lag-1 Error",  "down"),
        ("A24_acf_lag1_sq_error",    "A24 ACF r² Lag-1 Error",      "down"),
    ]),
    ("Vol", [
        ("A25_mean_rmse",            "A25 Mean RMSE",               "down"),
        ("A26_std_error",            "A26 Return Std Error",        "down"),
        ("A27_logreturn_std_error",  "A27 Log-Return Std Error",    "down"),
        ("A28_kurtosis_ratio",       "A28 Kurtosis Ratio (→ 1)",    "none"),
        ("A29_sigma_mean_error",     "A29 Sigma Mean Error",        "down"),
        ("A30_vol_path_rmse",        "A30 Cross-Sect. Vol Path RMSE", "down"),
        ("A31_rolling_vol_ks",       "A31 Rolling Vol KS (w=5)",    "down"),
        ("A32_vol_of_vol_error",     "A32 Vol-of-Vol Error",        "down"),
    ]),
]

ARROW = {"down": " ↓", "up": " ↑", "none": ""}

CRPS_TARGETS = [("cum_return", "Cumulative return"),
                ("increment", "Increment"),
                ("rv", "Realized vol")]


def fmt(x):
    """4 significant digits, scientific below 1e-3 -- matches the other READMEs."""
    if x is None or x == "":
        return "-"
    v = float(x)
    if v != v:
        return "nan"
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4g}"


def pct(x):
    return "-" if x is None or x == "" else f"{float(x):.4g}%"


def esc(s):
    """Escape pipes so a curve name containing '|' cannot break the table."""
    return str(s).replace("|", r"\|")


def read_summary(path):
    """metrics_summary.csv -> {metric: {"mean","std","seeds":[...],"scope"}}."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seeds = [row[k] for k in sorted(row) if k.startswith("seed_")]
            out[row["metric"]] = {"mean": row["mean"], "std": row["std"],
                                  "scope": row.get("scope", ""), "seeds": seeds}
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- A table ----
def render_a_table(sbts, floor):
    n_seeds = len(next(iter(sbts.values()))["seeds"]) if sbts else 5
    head = ["Metric", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + [FLOOR_COL]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for cat, rows in CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            s, f = sbts.get(key), floor.get(key)
            if s is None:
                cells, scope = ["-"] * (n_seeds + 2), ""
            else:
                cells = [f"{fmt(s['mean'])} ± {fmt(s['std'])}"]
                cells += [fmt(v) for v in s["seeds"]]
                cells += [fmt(f["mean"]) if f else "-"]
                scope = " *(native d=8)*" if s["scope"] == "native" else ""
            lines.append(f"| {label}{ARROW[direction]}{scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _better(sm, fm, direction):
    if direction == "up":
        return sm >= fm
    if direction == "none":              # A28: perfect = 1.0
        return abs(sm - 1.0) <= abs(fm - 1.0)
    return sm <= fm


def a_headline(sbts, floor):
    at_floor, total, gaps = [], 0, []
    for _cat, rows in CATEGORIES:
        for key, label, direction in rows:
            s, f = sbts.get(key), floor.get(key)
            if not s or not f:
                continue
            try:
                sm, fm = float(s["mean"]), float(f["mean"])
            except (TypeError, ValueError):
                continue
            total += 1
            clean = label.replace("\\", "")
            if _better(sm, fm, direction):
                at_floor.append(clean)
            elif direction == "down" and fm > 0:
                gaps.append((sm / fm, clean, sm, fm))
    gaps.sort(reverse=True)
    return at_floor, total, gaps


def _fmt_gap(g):
    ratio, label, sm, fm = g
    return f"**{label}** ({fmt(sm)} vs floor {fmt(fm)}, {ratio:.1f}×)"


# ---------------------------------------------------------------- B table ----
MEASURES = [("mse", "MSE", fmt), ("pct", "% err", pct), ("nrmse", "NRMSE", pct),
            ("cvar90", "CVaR₉₀", pct), ("cvar95", "CVaR₉₅", pct)]


def render_b_table(agg, agg_f, tvd, tvd_f):
    n_seeds = len(tvd.get("per_seed", [0] * 5)) if tvd else 5
    head = ["Plot", "Measure", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + [FLOOR_COL]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    if tvd:
        cells = [f"{pct(tvd['mean'])} ± {pct(tvd['std'])}"]
        cells += [pct(v) for v in tvd["per_seed"]]
        cells += [pct(tvd_f["mean"]) if tvd_f else "-"]
        lines.append("| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | "
                     + " | ".join(cells) + " |")
    for prefix, blk in agg.items():
        blk_f = agg_f.get(prefix, {})
        first = True
        for mk, mlabel, f_ in MEASURES:
            if mk not in blk:
                continue
            m = blk[mk]
            cells = [f"{f_(m['mean'])} ± {f_(m['std'])}"]
            cells += [f_(v) for v in m["per_seed"]]
            cells += [f_(blk_f[mk]["mean"]) if mk in blk_f else "-"]
            plot = f"| **{esc(blk['name'])}** " if first else "|  "
            first = False
            lines.append(plot + f"| {mlabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def b_headline(agg, agg_f):
    at, total = [], 0
    for prefix, blk in agg.items():
        blk_f = agg_f.get(prefix, {})
        if "mse" not in blk or "mse" not in blk_f:
            continue
        total += 1
        if float(blk["mse"]["mean"]) <= float(blk_f["mse"]["mean"]):
            at.append(blk["name"])
    return at, total


# ------------------------------------------------------------- CRPS table ----
def load_crps():
    """Collect every conditional-CRPS artefact into one dict.

    Returns (per_seed, baselines, realbank) where per_seed is
    {seed: {target: {mean, ci_lo, ci_hi}}} for the SBTS bank, baselines is
    {bank_name: {target: {...}}} taken from the LOWEST seed's file (the two
    historical banks are deterministic given --seed, so every file holds the
    same numbers and re-reporting them per seed would fake a spread), and
    realbank is the optional real-train-split-as-bank reference.
    """
    per_seed, baselines = {}, {}
    # Preferred source: losses/crps_configs/<config>__seed_*.json, written by
    # metrics/conditional_crps_multiasset.py once it grew --weight-mode/--standardize.
    # config="paper" is the author's Table 4 convention (sqrt(w) undivided, mu/sigma
    # per bank) and is the ONLY one allowed to populate Table C. The legacy flat
    # losses/conditional_crps_seed_*.json files predate the rv-axis fix of 2026-08-26
    # and are kept only so this renderer still works on an un-rerun checkout.
    files = sorted(glob.glob(os.path.join(METHOD_DIR, "losses", "crps_configs", "paper__seed_*.json")),
                   key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    if not files:
        files = sorted(glob.glob(os.path.join(METHOD_DIR, "losses", "conditional_crps_seed_*.json")),
                       key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    for p in files:
        seed = int(re.search(r"seed_(\d+)", p).group(1))
        res = read_json(p).get("results", {})
        # METHOD is the --label passed to conditional_crps_multiasset.py. Keying
        # on a hardcoded string here is how a renderer copied between method
        # trees silently produces an empty Table C: the file loads, the key
        # misses, and the table renders with baselines only.
        if METHOD in res:
            per_seed[seed] = res[METHOD]
        if not baselines:
            baselines = {k: v for k, v in res.items() if k != METHOD}
    rb = os.path.join(METHOD_DIR, "losses", "crps_configs", "paper__realbank.json")
    if not os.path.exists(rb):
        rb = os.path.join(METHOD_DIR, "losses", "conditional_crps_realbank.json")
    realbank = read_json(rb)
    return per_seed, baselines, realbank.get("results", {}).get("real_train_bank", realbank)


BASELINE_SWEEP = os.path.join(TE, "baselines", "crps_configs")


def load_baseline_sweep(cfg="paper"):
    """The two bootstrap baselines as mean +- sd over the resampling seed sweep.

    Reads results/trueexperiment/baselines/crps_configs/<cfg>__bseed_<S>.json,
    written by results/trueexperiment/run_baseline_seed_sweep.sh.

    Why this exists: block_bootstrap and session_bootstrap ARE stochastic in
    --seed (conditional_crps_multiasset.py:449 and :464 each build a fresh
    default_rng(seed)), so reporting them at the author's single 1234 while every
    method row carried a 5-seed sd compared a point estimate against a
    distribution. The sweep's block-bootstrap rv sd is ~0.016, wider than this
    method's own cross-seed sd, so the correction is not cosmetic.

    Deliberately does NOT cover real_train_bank: that bank is np.load()'d off
    disk and score() contains no RNG, so --seed provably never reaches it
    (rerunning it at 9999 reproduces all three targets to +0.00e+00). Five seeds
    there would print one number five times.

    Returns {} when the sweep directory is absent, in which case every caller
    falls back to the single-seed baselines embedded in the per-seed method
    artefacts and the table renders exactly what it rendered before.
    """
    out = {}
    files = sorted(glob.glob(os.path.join(BASELINE_SWEEP, f"{cfg}__bseed_*.json")),
                   key=lambda p: int(re.search(r"bseed_(\d+)", p).group(1)))
    for p in files:
        for name, blk in read_json(p).get("results", {}).items():
            d = out.setdefault(name, {"n": 0, "bank_shape": blk.get("bank_shape")})
            d["n"] += 1
            for k, _l in CRPS_TARGETS:
                if isinstance(blk.get(k), dict) and "mean" in blk[k]:
                    d.setdefault(k, []).append(blk[k]["mean"])
    return out


def _cell(blk):
    """mean +- half-width of the 95 % bootstrap CI over the 6 144 queries.

    The stored artefact holds ci_lo/ci_hi, and a percentile bootstrap CI is not
    exactly symmetric about the mean. Collapsing it to a single +- therefore
    discards a little information -- typically ~1e-3 of asymmetry on these rows,
    which is below the printed precision, so nothing visible is lost. If a row
    ever shows large skew this is the function that hides it; the raw ci_lo/ci_hi
    remain in losses/crps_configs/*.json.

    WARNING: this +- is NOT the same quantity as the +- on the bold aggregate
    method row, which is the sample sd ACROSS SEEDS. One is sampling noise over
    test queries, the other is generator noise over seeds. The note printed under
    the table says which is which; do not merge the two.

    Since the baseline seed sweep this function NO LONGER renders the two
    bootstrap rows -- those carry a cross-seed sd like the method row. It still
    renders the per-seed rows and the real-training-split floor, whose bank is
    np.load()'d off disk and carries no seed at all.
    """
    if not blk or "mean" not in blk:
        return "-"
    half = (blk["ci_hi"] - blk["ci_lo"]) / 2.0
    return f"{blk['mean']:.3f} ± {half:.3f}"


def render_crps_table(per_seed, baselines, realbank):
    if not per_seed and not baselines:
        return "_(no conditional-CRPS artefacts found)_", {}, {}, {}, 0

    head = ["Bank", "Bank size"] + [f"{lbl} CRPS ×1000 ↓" for _k, lbl in CRPS_TARGETS]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]

    sbts_mean = {}
    if per_seed:
        n = len(per_seed)
        cells = []
        for k, _lbl in CRPS_TARGETS:
            vals = [per_seed[s][k]["mean"] for s in sorted(per_seed)]
            m = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            sbts_mean[k] = m
            cells.append(f"**{m:.3f} ± {sd:.3f}**")
        b0 = (next(iter(per_seed.values())).get("bank_shape") or [None])[0]
        bsz = f"{b0:,}".replace(",", " ") if b0 else "-"
        lines.append(f"| **{METHOD}** (mean ± sd over {n} seeds) | {bsz} | "
                     + " | ".join(cells) + " |")
        for sd_ in sorted(per_seed):
            cells = [_cell(per_seed[sd_][k]) for k, _l in CRPS_TARGETS]
            lines.append(f"|  seed {sd_} | {bsz} | " + " | ".join(cells) + " |")

    # The bootstrap rows are seed-swept; the floor row below is not, and the
    # difference is a property of the code, not an oversight. See
    # load_baseline_sweep.__doc__.
    sweep = load_baseline_sweep("paper")

    NICE = {"block_bootstrap": "Moving-block bootstrap *(paper baseline)*",
            "session_bootstrap": "Session bootstrap *(paper baseline)*"}
    for name in ("block_bootstrap", "session_bootstrap"):
        blk = baselines.get(name)
        if not blk:
            continue
        size = (blk.get("bank_shape") or [None])[0]
        size_txt = f"{size:,}".replace(",", " ") if size else "-"
        swp = sweep.get(name)
        if swp and swp.get("n", 0) > 1:
            # mean +- sd over the resampling seeds -- the SAME quantity as the
            # bold method row, which is the point: the comparison is now
            # distribution against distribution.
            cells = [f"{statistics.fmean(swp[k]):.3f} ± {statistics.stdev(swp[k]):.3f}"
                     for k, _l in CRPS_TARGETS]
            label = f"{NICE[name]} (mean ± sd over {swp['n']} seeds)"
        else:
            cells = [_cell(blk[k]) for k, _l in CRPS_TARGETS]
            label = NICE[name]
        lines.append(f"| {label} | {size_txt} | " + " | ".join(cells) + " |")

    if realbank and "cum_return" in realbank:
        cells = [_cell(realbank[k]) for k, _l in CRPS_TARGETS]
        lines.append("| Real training split as bank *(reference, not in the paper)* | 6 144 | "
                     + " | ".join(cells) + " |")

    # The three ratio rows -- SBTS/block, SBTS/session, and the paper's own
    # ES/NQ/YM reference -- were REMOVED FROM THE TABLE on 2026-08-26 at the
    # author's request: the table shows the paper's protocol and nothing else.
    #
    # They are still COMPUTED, because they are the only dimensionless numbers
    # here. Absolute CRPS is set by the units and the intrinsic unpredictability
    # of this crypto panel and cannot be compared with the paper's futures data;
    # the ratio against a historical baseline can. So they are returned and the
    # headline prose below the table quotes them. Do not delete the computation
    # to "simplify" -- the headline would then silently report nothing.
    def _base_mean(name, blk):
        """Baseline mean for the ratios: the sweep mean when the sweep exists.

        The published single-seed block-bootstrap cum_return (1.2789) sits ~1.1 sd
        above the 5-seed mean (1.2749), so the old ratios were taken against a
        mildly unlucky draw. Falls back to the single-seed value so a checkout
        without baselines/crps_configs/ renders the old numbers rather than
        crashing.
        """
        swp = sweep.get(name)
        if swp and swp.get("n", 0) > 1:
            return {k: statistics.fmean(swp[k]) for k, _l in CRPS_TARGETS}
        return {k: blk[k]["mean"] for k, _l in CRPS_TARGETS} if blk else None

    bb = baselines.get("block_bootstrap")
    bb_m = _base_mean("block_bootstrap", bb)
    ratios = {k: sbts_mean[k] / bb_m[k] for k, _l in CRPS_TARGETS} \
        if bb_m and sbts_mean else {}
    sb = baselines.get("session_bootstrap")
    sb_m = _base_mean("session_bootstrap", sb)
    ratios_sess = {k: sbts_mean[k] / sb_m[k] for k, _l in CRPS_TARGETS} \
        if sb_m and sbts_mean else {}

    # Which differences the data can actually RESOLVE.
    #
    # A ratio of 0.993 is a "win" only if the 0.009 gap behind it is larger than
    # the noise on the means, and it is not: the query bootstrap half-width is
    # 0.025, roughly 3x the gap. Counting ratios by sign therefore reports three
    # wins where the evidence supports one, and the headline is the first line a
    # reader sees. So the headline counts only differences whose CIs do not
    # overlap: |gap| > half_SBTS + half_baseline.
    #
    # This is CONSERVATIVE and deliberately so. Both CIs are computed on the same
    # 6 144 queries, so the comparison is paired; a paired bootstrap on the
    # per-query differences would be tighter and could resolve more. It has not
    # been run. Under-claiming here is the safe direction.
    def _half(blk):
        return (blk["ci_hi"] - blk["ci_lo"]) / 2.0

    sbts_half = {k: statistics.fmean([_half(per_seed[s][k]) for s in sorted(per_seed)])
                 for k, _l in CRPS_TARGETS} if per_seed else {}
    resolved = {}
    for tag, name, base in (("block", "block_bootstrap", bb),
                            ("session", "session_bootstrap", sb)):
        if not (base and sbts_mean and sbts_half):
            continue
        bm = _base_mean(name, base)
        swp = sweep.get(name)
        # The baseline mean now carries TWO noise sources -- test-query sampling
        # AND the resampling seed. Ignoring the second would over-resolve. It is
        # added rather than combined in quadrature, matching the deliberately
        # conservative sum-of-half-widths already used on the line below.
        seed_sd = {k: (statistics.stdev(swp[k])
                       if swp and swp.get("n", 0) > 1 else 0.0)
                   for k, _l in CRPS_TARGETS}
        resolved[tag] = {
            k: abs(sbts_mean[k] - bm[k]) > (sbts_half[k] + _half(base[k]) + seed_sd[k])
            for k, _l in CRPS_TARGETS
        }
    return "\n".join(lines), ratios, ratios_sess, resolved, len(per_seed)


# ------------------------------------------------------------------ timing ---
def render_memorisation():
    """The memorisation section, entirely read from losses/memorisation.json.

    Deliberately placed after all three tables. On Heston a metric sinking below
    the floor is itself the memorisation alarm; here it is not (see the floor
    caption), so this number is the only guard there is, and nothing above it is
    safe to read until it has been read.
    """
    m = read_json(os.path.join(METHOD_DIR, "losses", "memorisation.json"))
    if not m:
        return ("## Memorisation check\n\n"
                "**MISSING** -- `losses/memorisation.json` not found. Run "
                "`code/measure_memorisation.py`; this section is required.\n")

    r = m["nn_ratio"]
    sd = m["nn_ratio_std"]
    r_test = m["nn_ratio_vs_test_era"]
    dup = m["n_exact_duplicates"]
    d_val = m["median_nn_heldout"]
    d_test = m["median_nn_test_era"]
    # The band's endpoints are the real splits scored against each other -- not
    # a tolerance anyone picked. Recomputed here so it tracks the data.
    lo, hi = d_test / d_val, 1.0

    if lo <= r <= hi:
        verdict = "sits **inside** the real-vs-real band"
    elif r > hi:
        verdict = ("sits **above** the band -- generated paths are *further* from the "
                   "training set than real data is, which is the opposite failure "
                   "from copying")
    else:
        verdict = ("sits **below** the band -- generated paths hug the training set. "
                   "**This is memorisation.**")

    # Capacity and epoch count are read from the checkpoint sidecar, not written
    # as literals: the whole point of the paragraph below is the ratio of
    # parameters to training paths, and a stale hardcoded 413 057 would turn the
    # one quantitative claim in this section into a lie the moment anyone
    # changed a layer count.
    _c = read_json(os.path.join(METHOD_DIR, "weights", "seed_0_config.json")) or {}
    _p = f"{_c.get('params', 413057):,}".replace(",", " ")
    _n = f"{_c.get('n_train', 6144):,}".replace(",", " ")
    _e = _c.get("epochs", 200)

    L = []
    L.append("## Memorisation check")
    L.append("")
    L.append("CSDI has no bandwidth knob, which makes it tempting to treat this section as an")
    L.append(f"SBTS formality. It is not. **{_p} parameters are fitted to {_n} training paths")
    L.append(f"for {_e} epochs**, so every training path is visited {_e} times, and a DDPM with that")
    L.append("much capacity relative to its sample count can put mass on near-copies of individual")
    L.append("training examples -- while winning every table above, because reproducing the")
    L.append("training set is how you win those. On the Heston tree a score below the floor would")
    L.append("expose it; **here it would not** (see the floor caption), so this number is the only")
    L.append("guard there is, and it is reported whatever it says.")
    L.append("")
    L.append("Unlike the SBTS entry, **nothing here was tuned against this diagnostic.** The")
    L.append("hyperparameters are the authors' published config and the ratio was never an")
    L.append("objective, so this is an out-of-sample readout rather than a constraint that was")
    L.append("optimised to -- which makes it more informative, not less.")
    L.append("")
    L.append("```")
    L.append("NNratio = median NN(generated -> train) / median NN(val -> train)")
    L.append("          " + m["space"])
    L.append("```")
    L.append("")
    L.append("| Seed | median NN(gen → train) | NNratio (vs `val`) | vs `test` era | Exact duplicates |")
    L.append("|------|-----------------------:|-------------------:|--------------:|-----------------:|")
    for d in m["per_seed"]:
        L.append(f"| {d['seed']} | {d['median_nn_generated']:.6f} | {d['nn_ratio']:.4f} | "
                 f"{d['nn_ratio_vs_test_era']:.4f} | {d['n_exact_duplicates']} |")
    L.append(f"| **mean** | **{m['median_nn_generated']:.6f}** | **{r:.4f} ± {sd:.4f}** | "
             f"**{r_test:.4f}** | **{dup}** |")
    L.append("")
    L.append(f"**NNratio = {r:.3f} ± {sd:.3f}**, against a real-vs-real band of "
             f"**{lo:.3f}–{hi:.3f}**. It {verdict}.")
    L.append("")
    L.append("The band is not a chosen tolerance. Its endpoints are what the real splits score")
    L.append("against each other: `median NN(test → train) / median NN(val → train)` = "
             f"{lo:.3f}, and")
    L.append("`val` against itself = 1.000. Zero free parameters, the same construction as the")
    L.append("floor column.")
    L.append("")
    L.append("> **The denominator is `val`, not `test`, and the direction is counter-intuitive.**")
    L.append("> `val` is the same era as train; `test` is the later era. Measured, `test` sits")
    L.append(f"> *closer* to train than `val` does ({d_test:.6f} vs {d_val:.6f}) — not because it")
    L.append("> resembles the training data more, but because annualised vol **falls** between the")
    L.append("> eras on 6 of 8 assets, so returns shrink and every distance contracts with them.")
    L.append(f"> A smaller denominator inflates the ratio: these same banks score {r:.3f} against")
    L.append(f"> `val` and {r_test:.3f} against `test`. Since memorisation is the *low*-ratio")
    L.append("> failure, a `test` denominator would push a copying generator toward the")
    L.append("> healthy-looking end. Both are in the table; only `val` decides the verdict, and it")
    L.append("> is the denominator the (h, K) sweep was scored against.")
    L.append("")
    L.append(f"**{dup} exact duplicates clears nothing, and is expected.** Ancestral sampling from")
    L.append("a continuous density lands on a bit-identical float64 path with probability zero, so")
    L.append("this counter can only ever fire on a plumbing bug -- a training array accidentally")
    L.append("saved as the bank. **The ratio is the number that matters.** For calibration of how")
    L.append("little the duplicate count tells you: the SBTS candidate `K=3, h=0.05`, rejected as a")
    L.append("memorisation trap, also had 0 duplicates while scoring NNratio 0.197.")
    L.append("")
    return "\n".join(L)


def render_timing(path):
    """Wall-clock table from losses/generation_time.csv.

    The CSDI schema is NOT the SBTS one. SBTS is a CPU kernel method, so its
    columns are `n_workers, elapsed_min`; CSDI trains on a GPU and writes
    `seed, role, n_paths, seq_len, d, train_min, gen_min, epochs, params,
    sec_per_epoch`, with two rows per seed (the 6 144-path A/B bank and the
    8 192-path CRPS pool). Training time is charged ONCE per seed even though it
    appears on both rows -- both banks come from the same checkpoint, which is
    the entire point of `generate_bank_true.py`. Summing the column blindly would
    double-count it.
    """
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, []

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=lambda r: (r.get("role", ""), int(r["seed"])))
    train_by_seed = {}
    for r in rows:
        t = _f(r.get("train_min"))
        if t:
            train_by_seed[r["seed"]] = max(train_by_seed.get(r["seed"], 0.0), t)
    total = sum(train_by_seed.values()) + sum(_f(r.get("gen_min")) for r in rows)

    ROLE = {"ab_bank": "A/B bank", "crps_pool": "CRPS pool"}
    out = ["| Seed | Role | Bank | Epochs | Train | Generate |",
           "|------|------|------|--------|-------|----------|"]
    for r in rows:
        n = r.get("n_paths", "")
        bank = (f"{int(n):,}".replace(",", " ") + f" × {r.get('seq_len','')} × {r.get('d','')}"
                if str(n).isdigit() else "-")
        tr = _f(r.get("train_min"))
        out.append(f"| {r['seed']} | {ROLE.get(r.get('role',''), r.get('role',''))} | {bank} | "
                   f"{r.get('epochs') or '-'} | {tr:.1f} min | {_f(r.get('gen_min')):.1f} min |")
    return "\n".join(out), total, rows


def main():
    meth = read_summary(os.path.join(METHOD_DIR, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(METHOD_DIR, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(METHOD_DIR, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    stats = read_json(os.path.join(METHOD_DIR, "losses", "dataset_stats.json"))

    missing = [n for n, o in ((f"{METHOD}/metrics_summary.csv", meth),
                              (f"{METHOD}/curve_b_aggregate.json", agg),
                              ("real_floor/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --dataset TrueDataset for "
            f"--method {METHOD} and --method real_floor first."
        )

    memo = render_memorisation()
    timing, total_min, trows = render_timing(
        os.path.join(METHOD_DIR, "losses", "generation_time.csv"))

    # Hyperparameters come from the checkpoint sidecar, never from a literal in
    # this file. train_true.py writes weights/seed_{i}_config.json next to the
    # .pt, including `paper_hyperparams` (bool) and `retuned_for_truedata`
    # (list of knob names). If a future run deviates from base.yaml, the
    # blockquote below says so automatically instead of the README quietly
    # claiming a verbatim replication that no longer holds.
    cfg0 = read_json(os.path.join(METHOD_DIR, "weights", "seed_0_config.json")) or {}
    params = cfg0.get("params", 413057)
    epochs = cfg0.get("epochs", 200)
    bs = cfg0.get("batch_size", 16)
    lr = cfg0.get("lr", 1e-3)
    nsteps = cfg0.get("num_steps", 50)
    spe = cfg0.get("steps_per_epoch", 384)
    tot_steps = cfg0.get("total_steps", spe * epochs)
    retuned = cfg0.get("retuned_for_truedata", [])
    if cfg0.get("paper_hyperparams", True) and not retuned:
        tuning_line = ("**Every value above is the authors' `config/base.yaml` verbatim.** "
                       "Nothing was re-tuned for this dataset.")
    else:
        tuning_line = ("**Deviations from the authors' `config/base.yaml`: "
                       + ", ".join(f"`{k}`" for k in retuned) + ".** "
                       "Every other value is verbatim.")

    at_floor, n_a, gaps = a_headline(meth, floor)
    b_at, n_b = b_headline(agg, agg_f)
    per_seed, baselines, realbank = load_crps()
    # section() returns an empty string when var_backtest.json is absent, so
    # this file renders exactly what it rendered before Table D existed. When
    # the VaR run and the CRPS configs disagree about which query file they
    # read it returns an HTML-comment note instead of a table -- that has
    # already happened once, and every number in the result looked plausible.
    d_block = table_d.section()
    crps_tbl, ratios, ratios_sess, resolved, n_crps = render_crps_table(
        per_seed, baselines, realbank)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    n_seeds = len(next(iter(meth.values()))["seeds"]) if meth else 4
    n_floor = len(next(iter(floor.values()))["seeds"]) if floor else 3
    n_crps_seeds = n_crps if n_crps else 4

    # The headline counts only differences the data can RESOLVE (non-overlapping
    # CIs), not every ratio that happens to fall on the right side of 1.000.
    # Counting by sign turned a 0.009 gap sitting inside a +-0.025 error bar into
    # a reported "win" -- see the resolved= comment in render_crps_table.
    rb = resolved.get("block", {})
    rs = resolved.get("session", {})
    if ratios:
        won = [lbl for k, lbl in CRPS_TARGETS if ratios.get(k, 9) < 1.0 and rb.get(k)]
        tied = [lbl for k, lbl in CRPS_TARGETS if k in ratios and not rb.get(k)]
        crps_head = (f"Against the moving-block bootstrap, **{len(won)} of the 3 targets "
                     f"is a difference this test can resolve**"
                     + (f" ({', '.join(won)})" if won else "")
                     + (f"; on {' and '.join(', '.join(tied).rsplit(', ', 1))} "
                        "the gap sits inside the error bars and "
                        f"{METHOD} and the bootstrap are indistinguishable" if tied else "")
                     + ". Ratios: " + ", ".join(f"{lbl} **{ratios[k]:.3f}**"
                                                for k, lbl in CRPS_TARGETS if k in ratios) + ".")
        # Never let the block-bootstrap ratio stand alone. If the session
        # bootstrap -- a baseline that resamples whole real training days and
        # learns nothing -- is ahead, that is the headline, not a footnote.
        if ratios_sess:
            lost = [lbl for k, lbl in CRPS_TARGETS if ratios_sess.get(k, 0) > 1.0 and rs.get(k)]
            sess_txt = ", ".join(f"{lbl} **{ratios_sess[k]:.3f}**"
                                 for k, lbl in CRPS_TARGETS if k in ratios_sess)
            if lost:
                crps_head += (f" **The session bootstrap is the harder baseline and {METHOD} loses "
                              f"to it, resolvably, on {', '.join(lost)}**: "
                              + sess_txt + ". Resampling whole real training days, with no model "
                              "at all, forecasts this panel better than the generator does.")
            else:
                crps_head += (" Against the session bootstrap, the harder of the two historical "
                              "baselines, no difference is resolvable: " + sess_txt + ".")
        crps_head += (" *Resolvable* = the two 95 % CIs do not overlap. That is a conservative "
                      "test — the CIs share the same 6 144 queries, so a paired bootstrap on "
                      "per-query differences would be tighter; it has not been run.")
    else:
        crps_head = "_(ratios unavailable: the block-bootstrap baseline is missing)_"

    n_paths = "6 144"
    for r in trows:
        if r.get("role") == "ab_bank" and str(r.get("n_paths", "")).isdigit():
            n_paths = f"{int(r['n_paths']):,}".replace(",", " ")
            break

    md = f"""# CSDI on TrueDataset — real crypto spot, d = 8

**Conditional Score-based Diffusion model for Imputation** (Tashiro, Song, Song & Ermon,
NeurIPS 2021, [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)) applied to **real market
data**: {n_paths} paths of 8 correlated crypto spot series
(BTC, ETH, BNB, SOL, XRP, DOGE, ADA, LINK), 30-second bars, seq\\_len = 128,
built from Binance public 1-second klines over 2022-07 → 2026-07.

This is the real-data counterpart of [`../HestonMultiAsset/CSDI/`](../HestonMultiAsset/CSDI/README.md).
Same generator, same metric code, same table layout — a different **question**. On Heston the
data-generating law is known, so "how close is the generator to the truth" has an exact answer.
Here it does not, so every reference in this README is a **real-vs-real** measurement: real
market data scored against other real market data through byte-identical code. Read
[`../truedatasetguideline.md`](../truedatasetguideline.md) before adding a method — it defines
the build, the five-split contract, the envelope, and what each reference column does and does
not mean.

**One joint model, not eight univariate ones.** `target_dim = 8` flows into CSDI's feature
embedding and into `forward_feature`, so the denoiser attends **across assets** at each of its
4 residual blocks. The 28 cross-asset correlations are therefore inside the model's hypothesis
class, not imposed afterwards — which is what makes A20 (terminal covariance error) and A6-A11
(the native-d=8 path kernels) load-bearing rows here rather than artefacts of post-processing.

**Unconditional use of a conditional model.** CSDI ships as an *imputer*. Setting
`is_unconditional = 1` and `cond_mask ≡ 0` makes `set_input_to_diffmodel` feed only the noisy
sequence, so `target_mask = observed_mask − cond_mask ≡ 1` and the training objective collapses
to the plain DDPM ε-MSE; at sampling time `impute` collapses to pure ancestral sampling. No
architecture surgery — the conditioning path is disabled by the authors' own flag.

> **Hyperparameters:** `epochs={epochs}`, `batch={bs}`, `lr={lr:g}`, `num_steps={nsteps}` (quad schedule,
> `beta_start=1e-4`, `beta_end=0.5`), `layers=4`, `channels=64`, `nheads=8`,
> `diffusion_embedding_dim=128`, `timeemb=128`, `featureemb=16` → **{params:,} parameters**.
> {tuning_line}
> The optimiser is theirs too: Adam with `weight_decay=1e-6` and
> `MultiStepLR(milestones=[int(0.75·epochs), int(0.9·epochs)], gamma=0.1)`.
> **Input space:** prices, **per-channel z-score fitted on the `train` split only** — identical
> to the Heston sibling. The fitted mean/std are stored in the checkpoint sidecar and read back
> from it at generation time, so a later bank can never silently re-fit them on a different split.
> **The step budget is not equal to the sibling's, and that is disclosed rather than hidden.**
> `epochs` is the authors' published number kept verbatim, but TrueDataset gives {spe} steps/epoch
> against the d = 8 Heston build's 512, so this run sees **{tot_steps:,} optimiser steps vs
> 102 400 there — 25 % fewer at the same nominal setting.** Nobody re-tuned it. The validation
> panel of [`plots/loss_convergence.png`](plots/loss_convergence.png) is the only evidence that
> {epochs} epochs was enough here; treat it as a precondition for the A-table below, not as an
> appendix figure.

---

## Metrics A1-A32 + B, mean ± std across {n_seeds} seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).
> **A33-A34 are absent, not failed.** They score the generated path against the dataset's
> *latent variance* path. A real market has none, so `compute_all_multiasset.py` skips them
> rather than fabricating a proxy.

{render_a_table(meth, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the real-vs-real floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}.
> **What the "{FLOOR_COL}" column is — read this before quoting it.** On Heston the floor is an *independent draw from the true SDE*: a genuine second sample of the data-generating law. **A real market has no law to re-draw from.** The column here is instead three **held-out real splits** — `train`, `val`, `valdisc` — pushed through the metric pipeline *as if they were generated banks* and scored against `test`, with byte-identical code. Three consequences you must not forget:
> 1. It is **not** a floor a generator ought to reach. It is the score a **perfect memoriser of the training era** achieves on the test era.
> 2. The build is **holdout-era** (train ends 2024-11-23, test starts 2025-02-10, 45.5-day embargo), so part of the distance is a genuine **regime change**, not finite-sample noise. Annualised vol falls from train to test on every asset (BTC 0.498 → 0.429, SOL 1.039 → 0.733). A generator that matched the *training* law perfectly would still score badly here — correctly so.
> 3. The `disc` split is **excluded on purpose**: it is the discriminator's real side, so using it as a fake bank would drive A18 to 0 by construction and manufacture an unreachable floor.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|, trained real=`disc` vs fake=generated. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix — the row that actually tests whether the 28 realised cross-asset correlations (mean 0.609, range 0.515-0.801) survived generation. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly. **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0.
> **Read A14 and A16 per asset, not as an 8-asset average.** 30-second bars on the thin end of
> this basket carry a large **point mass at exactly zero return** (LINK 24.4 %, ADA 22.0 %,
> BNB 20.9 % of all increments): the price simply did not move within the bar. CSDI is a
> continuous-density diffusion — its marginal has no atom, so it **cannot** reproduce that mass,
> and the KS / Wasserstein-family rows will be penalised on precisely those assets. That is a
> structural property of the model class meeting a microstructure property of the data, not a
> training failure, and averaging over 8 assets hides which assets are responsible. The
> breakdown is in [`metrics_per_asset.csv`](metrics_per_asset.csv).

---

## B, Curve-Shape Metrics, mean ± std across {n_seeds} seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. The curve is
computed **per asset and averaged over the 8 assets** *before* the combination below, so the
combination rules stay byte-identical to the Heston tree. For the real data (L_r) and generated
data (L_g) we build three lists, the curve L, its first finite difference L' (der), and its
second finite difference L'' (sec\\_der), then combine the three sub-scores into **one number
per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\\_der)/3; std = the sample std of that per-seed combined score across the {n_seeds} seeds. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE. Reported value = the **function-level MAPE on the curve L itself**; the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)**.
- **CVaR₉₀ / CVaR₉₅ rows**: tail-averaged pointwise curve error (Expected Shortfall) on the curve L **only (funct-only)**. Pointwise error eₜ = |L_g(t) − L_r(t)|; for q ∈ {{0.90, 0.95}}, CVaR_q = mean(eₜ for eₜ ≥ the q-th percentile of eₜ), then range-normalized like NRMSE.

All ↓ lower is better. The reference column is the same **real-vs-real** construction as above, with the same three caveats.

{render_b_table(agg, agg_f, tvd, tvd_f)}

> **Headline:** **{len(b_at)} of the {n_b} B plots** sit at or below the real-vs-real reference on the deciding MSE row: {b_at_txt}.
> **Cross-seed stability**: unlike the kernel methods in this tree, CSDI *is* trained by gradient
> descent, so the seed controls the weight initialisation, the minibatch order, the diffusion-step
> draw **and** the ancestral sampling noise. The ± here is therefore a genuine training-variance
> measurement, not merely simulation noise, and it is the column to check before believing any
> single-seed claim.

---

## C, Conditional generation — forecast CRPS by path shadowing

Every number above scores the **unconditional** law: does the generated cloud look like the
real cloud? That is not the question a trading desk asks. This table scores **conditional
skill**: given a real history, does the generator's conditional distribution of the *next*
32 returns beat a purely historical resampler?

Protocol is the paper's §3.3.1 (Deep-MKV-TS, Table 4), reproduced with **one documented
deviation** — retrieval is joint across the 8 assets, because the author's code refuses to
run at d ≠ 1. The deviation is spelled out under the table; every other constant is the
author's:

- Take the first **65 log-prices** of each of the 6 144 test paths as the query history.
- Retrieve the **256 nearest** histories from a pool of **8 192** paths, by Euclidean distance
  over four standardized feature blocks computed on the bank itself: the last 32 returns
  (w=1.0), a 24-point subsampled path shape (w=0.5), rolling volatility over 5-, 10- and
  20-step windows (w=2.0), and absolute/squared-return autocorrelations at lags 1, 2, 5, 10
  (w=1.0). Per-asset feature dim 73, × 8 assets = **584**.
- Use those neighbours' **next 32 returns** as the predictive ensemble and score it with CRPS.
- Compare against the paper's **two historical baselines**, both built from the training split
  only and both sized 8 192: a **moving-block bootstrap** (blocks of 8 consecutive returns,
  each drawn from a random training day, kept at its original intraday position) and a
  **session bootstrap** (an entire training day resampled whole).

Three targets: cumulative return over the horizon, the 32 individual increments, and
**realized volatility, `sqrt(sum_t r²_{{t,a}})` — summed over TIME, one scalar per asset**.
All ×1000, lower is better.
**Two different ± appear in this table and they mean different things.**
On the bold **{METHOD}** row *and on the two bootstrap baselines* it is a sample **sd across
seeds** — {n_crps_seeds} training seeds for {METHOD} (training + sampling noise), 5 resampling
seeds (1234–1238) for the bootstraps, whose blocks and sessions are redrawn from `--seed`.
On the **per-seed rows** and on the **real-training-split floor** it is the **half-width of the
95 % bootstrap CI over the 6 144 test queries** (2 000 replicates — sampling noise, i.e. how
much the average would move on a different draw of test histories); that query CI is roughly
4× the seed sd, so the two must never be read as the same quantity.
The floor carries no seed sd because it has none to carry: its bank is the real training split
loaded off disk and `conditional_crps_multiasset.py`'s `score()` contains no RNG, so `--seed`
provably never reaches it — rerunning it at seed 9999 reproduces all three targets to
`+0.00e+00`. Five seeds there would print one number five times.

> **The CRPS pool is a separate 8 192-path draw, not a resample of the A/B bank.**
> `generate_bank_true.py` reloads the seed's checkpoint and samples afresh, reading the z-score
> statistics **out of the checkpoint** rather than recomputing them, so the pool and the A/B
> bank are two independent draws from one fitted model rather than two views of one draw.

{crps_tbl}

> **Headline:** {crps_head}
> **The ratio row is the number that transfers**, not the absolute CRPS. Absolute CRPS is set by
> the intrinsic unpredictability of the market and the units of the data; the ratio against the
> block bootstrap is dimensionless, which is why the paper reports its generator and the
> bootstraps side by side for all three indices. The paper's own generator loses to the block
> bootstrap on cumulative return for NQ (1.088) and YM (1.087) and wins on realized vol
> everywhere (0.808 / 0.916 / 0.946).
> **The window is narrow, and that is a finding, not a caveat.** The *real training split used
> directly as the bank* — a bank that cannot be beaten by any generator fit on that same split —
> scores within ~2 % of the block bootstrap on cumulative return and ~1 % on increments. Only
> realized vol leaves meaningful room. A generator that "wins" by 0.5 % on cumulative return has
> not demonstrated conditional skill; it has demonstrated that the metric is saturated there.
> **Bank size is not a lever.** Measured on this build with K = 256 fixed, CRPS moves by 0.4 %
> across an 8× range of bank size (1 536 → 12 288) — averaging 256 neighbours makes the ensemble
> spread depend on intrinsic unpredictability, not on retrieval sharpness. The value is pinned to
> the paper's 8 192 anyway, so the comparison is like-for-like.
> **One deliberate deviation from the author's code:** retrieval is **joint across assets** (one
> 584-dim feature vector per path). The author's `real_data_protocol.py:513` hard-raises for
> d ≠ 1, so there is no reference behaviour to copy; per-asset retrieval would answer a different
> question (8 independent univariate forecasts) and would discard exactly the cross-asset
> structure this dataset exists to test.

> **Retrieval convention.** The paper weights each feature block by `sqrt(w)` on every
> coordinate, so a block's real influence is its weight x its length. The alternative that
> divides by the block length was run on the same banks and is stored alongside
> (`losses/crps_configs/perdim__*.json`); it changes the ratios by under one percent, so the
> table above stands. The paper's convention is what is reported.

---
{d_block}
{memo}
---

## Stylised Facts Diagnostic (TrueDataset vs {METHOD}, seed 0, asset 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

The third (black dashed) curve is the real **`val`** split, not a theory curve — there is no
closed form for BTC. It is drawn from `real_floor/generated_paths/seed_1`, chosen explicitly:
the floor's seed 0 is the *training* split, so plotting it here and calling it held-out would
be false. The legend is read from that directory's `metadata.json`, not hardcoded. **One asset is shown, not eight**: the *metrics* are averaged over all 8
assets, the *figure* is asset 0 (BTCUSDT). Pooling eight assets whose annualised vol spans
0.43-0.92 into one histogram would show the mixing, not the model.

![TrueDataset Diagnostics](plots/truedata_diagnostics.png)

---

## Training loss convergence

CSDI optimises a **single** term — the DDPM denoising objective
`E_t ‖ε − ε_θ(x_t, t)‖²` restricted to `target_mask`, which is 1 everywhere in the
unconditional regime. There is no KLD / NLL / reconstruction decomposition to break out, so the
figure is two panels rather than the five an ELBO model would need: train and validation, every
seed overlaid.

**The two panels are not the same quantity plotted twice.** `calc_loss` draws **one** random
diffusion step `t` per sample, so the train curve is a one-sample estimate and is dominated by
that sampling noise. `calc_loss_valid` averages over **all {nsteps}** diffusion steps, so the
validation curve is the low-variance one and is the series to read convergence from. Their
*levels* are not comparable, for the same reason — they are two different estimators of two
different averages, not train/test versions of one number. **A validation curve sitting below
the training curve here is expected and is evidence of nothing. Do not report the gap between
them as an overfitting measure.**

The validation series carries **20 points, not {epochs}**: `train_true.py` sets
`val_every = max(1, epochs // 20)`, i.e. one validation pass every {max(1, epochs // 20)} epochs at this
budget, because a full pass costs ~{nsteps}× a training epoch. Skipped epochs are written as `nan`
and dropped, which is why the val panel is marker-dotted and the train panel is not.

![Loss convergence](plots/loss_convergence.png)

Wall-clock ({len(trows)} rows, {epochs} epochs/seed, **{total_min:.0f} min total** — training is charged
once per seed, generation once per bank, so the two roles below do not double-count it):

{timing}

---

## A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.
The classifier is **native**: it sees all 8 channels at once, and its real side is the
`disc` split, never `test`.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

---

## A19, Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).
The predictor is run **once per asset**; the curves archived here are **asset 0's**, since the
driver stores the loss history only for `j == 0`. The A19 score in the table above is the mean
over all 8 assets.

![Predictive Score Loss](plots/pred_score_loss.png)

---

## File layout

```
results/trueexperiment/
├── truedatasetguideline.md               ← read this before adding a method
├── real_floor/                           real-vs-real reference ({n_floor} held-out real splits)
│   ├── generated_paths/seed_{{0..{n_floor - 1}}}/     symlinks to train / val / valdisc
│   ├── metrics_summary.csv
│   ├── curve_b_aggregate.json
│   └── grid_tvd_aggregate.json
└── {METHOD}/
    ├── README.md                         ← this file (generated by code/render_readme.py)
    ├── code/
    │   ├── README.md                     source notes, hyperparameters, deviations, md5 digests
    │   ├── reference/                    byte-identical copy of the authors' release (16 files)
    │   ├── csdi_true.py                  library half: BASE_CONFIG, CSDI_TrueData, z-score, rescale
    │   ├── train_true.py                 trains one seed, writes weights + losses + the A/B bank
    │   ├── generate_bank_true.py         reloads a checkpoint for the 8 192-path CRPS pool
    │   ├── collect_artifacts.py          §4 contract gate — recomputes from the .npy, exits 1 on breach
    │   ├── plot_diagnostics_true.py      the 8-panel stylised-facts figure
    │   ├── plot_losses.py                train/val convergence, all seeds overlaid
    │   ├── measure_memorisation.py       NN ratio vs the val split (same era as train)
    │   └── render_readme.py              regenerates this README from the artefacts
    ├── weights/
    │   ├── seed_{{i}}_model.pt              state_dict — gitignored
    │   └── seed_{{i}}_config.json           every hyperparameter + the fitted z-score (tracked)
    ├── generated_paths/seed_{{0..{n_seeds - 1}}}/
    │   ├── generated_paths_6144x128x8.npy   (6144, 128, 8) float64 — gitignored, 50 MB
    │   └── metadata.json                    seed, shape, min/max, timings, params (tracked)
    ├── crps_banks/generated_paths/seed_{{0..{n_crps_seeds - 1}}}/
    │   └── generated_paths_8192x128x8.npy   the paper's 8 192-path pool — gitignored, 67 MB
    ├── logs/train_seed{{i}}.log             raw stdout of each training run (tracked)
    ├── losses/
    │   ├── seed_{{i}}_losses.csv             step,phase,loss_total — train every epoch, val every 20
    │   ├── crps_configs/paper__seed_{{i}}.json     table C as reported (--weight-mode paper)
    │   ├── crps_configs/perdim__seed_{{i}}.json    the alternative convention, for the caveat above
    │   ├── crps_configs/paper__realbank.json      real training split used as the bank
    │   ├── memorisation.json                NN ratio vs val — the memorisation guard
    │   ├── dataset_stats.json               copy of the locked build's stats (provenance)
    │   └── generation_time.csv              seed,role,n_paths,...,train_min,gen_min,params
    ├── plots/
    │   ├── truedata_diagnostics.png      8-panel stylised facts (seed 0, asset 0 = BTCUSDT)
    │   ├── loss_convergence.png          train + val, all seeds overlaid, log y
    │   ├── disc_classifier_loss.png      A18 BCE curves
    │   └── pred_score_loss.png           A19 MAE curves (asset 0)
    ├── metrics_summary.csv               A1-A32, mean ± std, per seed
    ├── metrics_per_asset.csv             per-metric × per-asset breakdown (8 rows per metric)
    ├── curve_b_aggregate.json            B curve-shape aggregate
    ├── grid_tvd_aggregate.json           path-cloud TVD
    └── seed_{{i}}_metrics.json             full per-seed dump incl. the per_asset block
```

The `.npy` arrays and the `.pt` checkpoints are **gitignored**: `(6144, 128, 8)` float64 = 50 MB
each, and LFS was ruled out on 2026-07-30. They are fully reproducible from the tracked code and
the tracked `weights/seed_{{i}}_config.json`; the `metadata.json` beside each array **is** tracked,
so shapes, price ranges and generation times stay auditable without the payload. The raw Binance
archive is not committed either — `dataset/TrueDataset/` rebuilds it from `data.binance.vision`.

## Reproduce

```bash
cd /home/tbasseras/benchmark
V=dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
R=results/trueexperiment
P=/home/tbasseras/gpu-venv/bin/python     # anything importing torch

# 1. dataset — downloads Binance 1s klines and builds the 5 splits.
#    Only if $V/*.npy are absent. See dataset/TrueDataset/README.md.
/home/tbasseras/.cc-venv/bin/python dataset/TrueDataset/build_true_dataset.py \\
    --start 2022-07 --end 2026-07 --n-samples 6144 \\
    --split-mode holdout-era --out-dir $V

# 2. smoke-test before spending the GPU: 2 epochs, 64 paths. --tag probe writes
#    probe_* artefacts and NO canonical weights, so it cannot pollute step 3.
#    (train_true.py's bank-size flag is --gen-num; generate_bank_true.py's is
#    --m-simu. They are different scripts and the names do not interchange.)
CUDA_VISIBLE_DEVICES=0 $P $R/{METHOD}/code/train_true.py --seed 0 --data-dir $V \\
    --seq-tag $TAG --epochs 2 --gen-num 64 --val-n 64 --tag probe
rm -f $R/{METHOD}/generated_paths/seed_0/probe_* $R/{METHOD}/losses/probe_*

# 3. train the {n_seeds} seeds — 2 GPUs at a time, the hard limit on this box.
#    ~11 s/epoch => ~37 min/seed. Detached: these outlive the shell.
for S in {" ".join(str(i) for i in range(n_seeds))}; do
  G=$(( S % 2 )); C0=$(( G * 8 ))
  setsid nohup env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 \\
    taskset -c $C0-$(( C0 + 7 )) $P $R/{METHOD}/code/train_true.py --seed $S \\
    --data-dir $V --seq-tag $TAG --val-n 256 \\
    > $R/{METHOD}/logs/train_seed$S.log 2>&1 < /dev/null & disown
done
wait

# 4. the paper's 8 192-path CRPS pool, resampled from each checkpoint.
for S in {" ".join(str(i) for i in range(n_crps_seeds))}; do
  CUDA_VISIBLE_DEVICES=$(( S % 2 )) $P $R/{METHOD}/code/generate_bank_true.py \\
    --seed $S --data-dir $V --seq-tag $TAG --m-simu 8192 \\
    --out-root $R/{METHOD}/crps_banks
done

# 5. §4 contract gate. Recomputes shape / dtype / S0 / finiteness FROM THE .npy —
#    metadata is the author's claim, the array is the evidence. Exits 1 on breach.
/home/tbasseras/.cc-venv/bin/python $R/{METHOD}/code/collect_artifacts.py

# 6. real-vs-real reference: three held-out REAL splits as stand-in banks.
for i in 0 1 2; do mkdir -p $R/real_floor/generated_paths/seed_$i; done
ln -sf $PWD/$V/true_S_$TAG.npy          $R/real_floor/generated_paths/seed_0/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_val_$TAG.npy      $R/real_floor/generated_paths/seed_1/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_valdisc_$TAG.npy  $R/real_floor/generated_paths/seed_2/generated_paths_$TAG.npy

# 7. metrics — both sides go through byte-identical code.
for M in {METHOD} real_floor; do
  N=$([ $M = {METHOD} ] && echo {n_seeds} || echo {n_floor})
  CUDA_VISIBLE_DEVICES=0 $P metrics/compute_all_multiasset.py \\
    --method $M --dataset TrueDataset --seeds $N \\
    --data-dir $V --seq-tag $TAG --dt 9.51293759512e-07 --results-dir $R/$M
done

# 8. conditional CRPS (table C). BOTH conventions, and the realbank reference.
#    --label {METHOD} is load-bearing: render_readme.py keys table C on it, and a
#    mismatched label renders the table with baselines only, silently.
mkdir -p $R/{METHOD}/losses/crps_configs
for S in {" ".join(str(i) for i in range(n_crps_seeds))}; do
  for CFG in "paper paper bank" "perdim perdim realtrain"; do
    set -- $CFG
    $P metrics/conditional_crps_multiasset.py \\
      --data-dir $V --seq-tag $TAG --bank-size 8192 --label {METHOD} \\
      --weight-mode $2 --standardize $3 \\
      --bank $R/{METHOD}/crps_banks/generated_paths/seed_$S/generated_paths_8192x128x8.npy \\
      --out $R/{METHOD}/losses/crps_configs/$1__seed_$S.json
  done
done
# ... and the reference every row in table C is read against: the real TRAIN
# split used as the bank. Native size 6144 -- that is all the data there is --
# and --bank-size matches so the two bootstrap baselines are measured at the
# same bank size rather than a larger one.
for CFG in "paper paper bank" "perdim perdim realtrain"; do
  set -- $CFG
  $P metrics/conditional_crps_multiasset.py \\
    --data-dir $V --seq-tag $TAG --bank-size 6144 --label real_train_bank \\
    --weight-mode $2 --standardize $3 \\
    --bank $V/true_S_$TAG.npy \\
    --out $R/{METHOD}/losses/crps_configs/$1__realbank.json
done

# 9. memorisation -- the guard that stops an over-fitted checkpoint from winning
#    table A by copying. Denominator is the val split, NOT test. Measured, test
#    sits CLOSER to train than val does (0.017647 vs 0.018943) because annualised
#    vol falls across the era break, so a test denominator is SMALLER and inflates
#    the ratio -- which flatters a memoriser, since memorisation is the low-ratio
#    failure. See the Memorisation check section.
$P $R/{METHOD}/code/measure_memorisation.py \\
    --data-dir $V --seq-tag $TAG --seeds {",".join(str(i) for i in range(n_seeds))}

# 10. figures
$P $R/{METHOD}/code/plot_diagnostics_true.py --data-dir $V --seq-tag $TAG
$P $R/{METHOD}/code/plot_losses.py --seeds {",".join(str(i) for i in range(n_seeds))}
$P metrics/plot_score_losses.py \\
    --method {METHOD} --dataset TrueDataset --results-dir $R/{METHOD}

# 11. regenerate this README from the artefacts
$P $R/{METHOD}/code/render_readme.py
```
"""
    out = os.path.join(METHOD_DIR, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
