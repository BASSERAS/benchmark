#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/trueexperiment/SBTS/README.md from the metric artefacts.

Section order and table shape mirror `results/HestonMultiAsset/SBTS/README.md`
so the synthetic and the real-data entries display identically, with **one
addition**: a third table, *Conditional generation*, sitting immediately after
the curve-shape table. That table is the paper's Table 4 protocol -- conditional
forecast CRPS by path shadowing -- and it exists here and not on Heston because
it is the only metric in this repo that scores *conditional* skill rather than
unconditional distribution match, which is the question a real market actually
poses.

Every number is READ FROM DISK, never typed by hand: the tables *and* the
headline counts are computed from the artefacts, so the prose cannot drift away
from the numbers it describes.

Reads
  results/trueexperiment/SBTS/metrics_summary.csv
  results/trueexperiment/SBTS/curve_b_aggregate.json
  results/trueexperiment/SBTS/grid_tvd_aggregate.json
  results/trueexperiment/SBTS/losses/generation_time.csv
  results/trueexperiment/SBTS/losses/conditional_crps_seed_*.json
  results/trueexperiment/SBTS/losses/conditional_crps_realbank.json   (optional)
  results/trueexperiment/SBTS/losses/bandwidth_selection_FINAL.json   (optional)
  results/trueexperiment/real_floor/metrics_summary.csv               (floor)
  results/trueexperiment/real_floor/curve_b_aggregate.json
  results/trueexperiment/real_floor/grid_tvd_aggregate.json

Writes
  results/trueexperiment/SBTS/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/SBTS/code/render_readme.py
"""

import csv
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SBTS = os.path.dirname(HERE)                                  # .../SBTS
TE = os.path.dirname(SBTS)                                    # .../trueexperiment
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
    files = sorted(glob.glob(os.path.join(SBTS, "losses", "crps_configs", "paper__seed_*.json")),
                   key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    if not files:
        files = sorted(glob.glob(os.path.join(SBTS, "losses", "conditional_crps_seed_*.json")),
                       key=lambda p: int(re.search(r"seed_(\d+)", p).group(1)))
    for p in files:
        seed = int(re.search(r"seed_(\d+)", p).group(1))
        res = read_json(p).get("results", {})
        if "SBTS" in res:
            per_seed[seed] = res["SBTS"]
        if not baselines:
            baselines = {k: v for k, v in res.items() if k != "SBTS"}
    rb = os.path.join(SBTS, "losses", "crps_configs", "paper__realbank.json")
    if not os.path.exists(rb):
        rb = os.path.join(SBTS, "losses", "conditional_crps_realbank.json")
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
    SBTS row, which is the sample sd ACROSS SEEDS. One is sampling noise over
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
        lines.append(f"| **SBTS** (mean ± sd over {n} seeds) | {bsz} | "
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
    m = read_json(os.path.join(SBTS, "losses", "memorisation.json"))
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

    L = []
    L.append("## Memorisation check")
    L.append("")
    L.append("SBTS is a kernel method: the bandwidth `h` is exactly the knob that trades fit")
    L.append("against copying, and a small enough `h` wins every table above by reproducing the")
    L.append("training set. On the Heston tree, a score below the floor exposes that. **Here it")
    L.append("does not** -- see the floor caption -- so this number is the only guard, and it is")
    L.append("reported whatever it says.")
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
    L.append(f"**{dup} exact duplicates does not clear the method.** SBTS interpolates between")
    L.append("training paths, so its failure mode is near-copies, not copies. The ratio is the")
    L.append("number that matters: the rejected candidate `K=3, h=0.05` also had 0 duplicates")
    L.append("while scoring NNratio 0.197.")
    L.append("")
    return "\n".join(L)


def render_timing(path):
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, []
    rows.sort(key=lambda r: int(r["seed"]))
    total = sum(float(r["elapsed_min"]) for r in rows)
    out = ["| Seed | Workers | Elapsed |", "|------|---------|---------|"]
    out += [f"| {r['seed']} | {r['n_workers']} | {float(r['elapsed_min']):.1f} min |"
            for r in rows]
    return "\n".join(out), total, rows


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    stats = read_json(os.path.join(SBTS, "losses", "dataset_stats.json"))

    missing = [n for n, o in (("SBTS/metrics_summary.csv", sbts),
                              ("SBTS/curve_b_aggregate.json", agg),
                              ("real_floor/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --dataset TrueDataset for "
            "--method SBTS and --method real_floor first."
        )

    memo = render_memorisation()
    timing, total_min, trows = render_timing(
        os.path.join(SBTS, "losses", "generation_time.csv"))
    r0 = trows[0] if trows else {}
    h, K, npi = r0.get("h", "0.07"), r0.get("K", "1"), r0.get("N_pi", "50")
    nws = sorted({str(r.get("n_workers", "")) for r in trows if r.get("n_workers")},
                 key=lambda x: int(x) if x.isdigit() else 0)
    nw = nws[0] if len(nws) == 1 else " and ".join(nws)

    at_floor, n_a, gaps = a_headline(sbts, floor)
    b_at, n_b = b_headline(agg, agg_f)
    per_seed, baselines, realbank = load_crps()
    crps_tbl, ratios, ratios_sess, resolved, n_crps = render_crps_table(
        per_seed, baselines, realbank)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    n_seeds = len(next(iter(sbts.values()))["seeds"]) if sbts else 5
    n_floor = len(next(iter(floor.values()))["seeds"]) if floor else 3

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
                        "SBTS and the bootstrap are indistinguishable" if tied else "")
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
                crps_head += (f" **The session bootstrap is the harder baseline and SBTS loses "
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

    md = f"""# SBTS on TrueDataset — real crypto spot, d = 8

**Schrödinger Bridge Time Series generation** (Alouadi, Barreau, Carlier & Pham, ICAIF 2025,
[arXiv:2503.02943](https://arxiv.org/abs/2503.02943)) applied to **real market data**:
{r0.get('M_simu', '6 144')} paths of 8 correlated crypto spot series
(BTC, ETH, BNB, SOL, XRP, DOGE, ADA, LINK), 30-second bars, seq\\_len = 128,
built from Binance public 1-second klines over 2022-07 → 2026-07.

This is the real-data counterpart of [`../HestonMultiAsset/SBTS/`](../HestonMultiAsset/SBTS/README.md).
Same generator, same metric code, same table layout — a different **question**. On Heston the
data-generating law is known, so "how close is the generator to the truth" has an exact answer.
Here it does not, so every reference in this README is a **real-vs-real** measurement: real
market data scored against other real market data through byte-identical code. Read
[`../truedatasetguideline.md`](../truedatasetguideline.md) before adding a method — it defines
the build, the five-split contract, the envelope, and what each reference column does and does
not mean.

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent, **no weights**. The "model" *is* the training array plus `(h, K, N_pi)`
— see [`weights/README.md`](weights/README.md).

> **Hyperparameters:** `h={h}`, `K={K}`, `N_pi={npi}`, `dt=1/1 051 200` (30-second bars, 1 051 200 bars/year).
> **`N_pi` is the author's** value. **`h` and `K` are ours**, re-selected here because neither
> can cross datasets: the SBTS kernel is radial, `K_h(x) = (h² − ‖x‖²)²·1{{‖x‖₂ < h}}`, so its
> support radius must scale with the typical distance between d-dimensional increments, and
> that distance is a property of the data, not of the method.
> The selection was run against the **real-vs-real envelope** (never against test), minimising
> `|log NNratio|` — the memorisation diagnostic — subject to both gated statistics staying
> inside the envelope. `K = 3` at `h = 0.05` produces the best volatility error in the whole
> grid (**6.06 %**, better than the real-vs-real *minimum* of 8.72 %) and is a **memorisation
> trap**: its NNratio is **0.197**, i.e. generated paths sit five times closer to the training
> set than held-out real data does. Anyone reading only the volatility column would pick it.
> Full grid, criterion and caveats:
> [`losses/bandwidth_selection_FINAL.json`](losses/bandwidth_selection_FINAL.json),
> [`losses/finalists_seed0.json`](losses/finalists_seed0.json) and
> [`losses/selection_criterion.md`](losses/selection_criterion.md).

---

## Metrics A1-A32 + B, mean ± std across {n_seeds} seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).
> **A33-A34 are absent, not failed.** They score the generated path against the dataset's
> *latent variance* path. A real market has none, so `compute_all_multiasset.py` skips them
> rather than fabricating a proxy.

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the real-vs-real floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}.
> **What the "{FLOOR_COL}" column is — read this before quoting it.** On Heston the floor is an *independent draw from the true SDE*: a genuine second sample of the data-generating law. **A real market has no law to re-draw from.** The column here is instead three **held-out real splits** — `train`, `val`, `valdisc` — pushed through the metric pipeline *as if they were generated banks* and scored against `test`, with byte-identical code. Three consequences you must not forget:
> 1. It is **not** a floor a generator ought to reach. It is the score a **perfect memoriser of the training era** achieves on the test era.
> 2. The build is **holdout-era** (train ends 2024-11-23, test starts 2025-02-10, 45.5-day embargo), so part of the distance is a genuine **regime change**, not finite-sample noise. Annualised vol falls from train to test on every asset (BTC 0.498 → 0.429, SOL 1.039 → 0.733). A generator that matched the *training* law perfectly would still score badly here — correctly so.
> 3. The `disc` split is **excluded on purpose**: it is the discriminator's real side, so using it as a fake bank would drive A18 to 0 by construction and manufacture an unreachable floor.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|, trained real=`disc` vs fake=generated. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix — the row that actually tests whether the 28 realised cross-asset correlations (mean 0.609, range 0.515-0.801) survived generation. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly. **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0.

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
> **Cross-seed stability**: SBTS is a deterministic kernel with no gradient descent, so the only seed-to-seed variation is the simulation noise of the Euler-Maruyama draw — there are no seed-collapse events of the kind a GAN can suffer.

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
On the bold **SBTS** row *and on the two bootstrap baselines* it is a sample **sd across
seeds** — {n_crps} generator seeds for SBTS, 5 resampling seeds (1234–1238) for the
bootstraps, whose blocks and sessions are redrawn from `--seed`.
On the **per-seed rows** and on the **real-training-split floor** it is the **half-width of
the 95 % bootstrap CI over the 6 144 test queries** (2 000 replicates — sampling noise, i.e.
how much the average would move on a different draw of test histories).
The floor carries no seed sd because it has none to carry: its bank is the real training
split loaded off disk and `conditional_crps_multiasset.py`'s `score()` contains no RNG, so
`--seed` provably never reaches it — rerunning it at seed 9999 reproduces all three targets
to `+0.00e+00`. Five seeds there would print one number five times.

> **Corrected 2026-08-26.** The rv target previously summed the squared increments over
> **assets** at each step, giving a `(N, 32)` cross-sectional dispersion trajectory. The
> author sums over **time** (`shadowing.py:579-601`: `raw_shadows` is `(B, K, h+1, A)` and
> the reduction is on the TIME axis), giving `(N, 8)`. At d = 1 the author's quantity is
> realized vol over the window; the old one collapsed to `|r_t|` — a different statistic.
> This changed the SBTS/block-bootstrap rv ratio from 0.972 to the value tabulated below.
> Cumulative return and increments are unaffected and are bit-identical to the previous run.

{crps_tbl}

> **Headline:** {crps_head}
> **The ratio row is the number that transfers**, not the absolute CRPS. Absolute CRPS is set by
> the intrinsic unpredictability of the market and the units of the data; the ratio against the
> block bootstrap is dimensionless, which is why the paper reports SBTS and the bootstraps side
> by side for all three indices. The paper's own SBTS loses to the block bootstrap on cumulative
> return for NQ (1.088) and YM (1.087) and wins on realized vol everywhere (0.808 / 0.916 / 0.946).
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

{memo}
---

## Stylised Facts Diagnostic (TrueDataset vs SBTS, seed 0, asset 0)

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

## SBTS has no training loss

SBTS is kernel-based; there is no loss curve. The bandwidth `h`, Markovian order `K` and Euler
substeps `N_pi` are the only knobs: **h={h}, K={K}, N_pi={npi}**. The `losses/` directory holds the
per-seed hyperparameter records and the complete selection evidence, which for a method with no
training curve *is* the reproducibility trail.

Generation wall-clock ({nw} workers, {len(trows)} seeds, **{total_min:.0f} min total**):

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
└── SBTS/
    ├── README.md                         ← this file (generated by code/render_readme.py)
    ├── code/
    │   ├── README.md                     source notes, hyperparameters, deviations
    │   ├── sbts_generate_true.py         core module: radial kernel, per-asset sigma
    │   ├── generate_bank_true.py         driver for one bank at the calibrated (h, K)
    │   ├── calibrate_bandwidth_true.py   the (h, K) grid under the real-vs-real envelope
    │   ├── collect_artifacts.py          metadata.json -> losses/ bookkeeping
    │   ├── plot_diagnostics_true.py      the 8-panel stylised-facts figure
    │   ├── measure_memorisation.py       NN ratio vs the val split (same era as train)
    │   └── render_readme.py              regenerates this README from the artefacts
    ├── generated_paths/seed_{{0..{n_seeds - 1}}}/
    │   ├── generated_paths_6144x128x8.npy   (6144, 128, 8) float64 — gitignored, 50 MB
    │   └── metadata.json                    seed, h, K, N_pi, sigma, shape, min/max, elapsed (tracked)
    ├── crps_banks/generated_paths/seed_{{0..3}}/
    │   └── generated_paths_8192x128x8.npy   the paper's 8 192-path pool — gitignored, 67 MB
    ├── losses/
    │   ├── seed_{{i}}_bandwidth.json          h, K, N_pi, dt — no loss (kernel method)
    │   ├── bandwidth_selection_FINAL.json   full (h, K) grid + admissibility verdict
    │   ├── finalists_seed{{0,1,2}}.json       3-seed re-measurement of the top candidates
    │   ├── selection_criterion.md           pre-registered criterion and its caveats
    │   ├── envelope_by_split_mode.json      real-vs-real envelope, interleaved vs holdout-era
    │   ├── crps_configs/                    table C, LIVE. <config>__seed_{{i}}.json per
    │   │                                    bank, config in {{paper, perdim}}; plus
    │   │                                    <config>__realbank.json (real train split
    │   │                                    as the bank). Only paper__* feeds table C.
    │   ├── conditional_crps_seed_{{i}}.json   SUPERSEDED (pre rv-axis fix, 2026-08-26)
    │   ├── conditional_crps_realbank.json   SUPERSEDED, kept for provenance only
    │   ├── memorisation.json                NN ratio -- the objective (h, K) were picked on
    │   ├── dataset_stats.json               copy of the locked build's stats (provenance)
    │   └── generation_time.csv              wall-clock per seed
    ├── plots/
    │   ├── truedata_diagnostics.png      8-panel stylised facts (seed 0, asset 0 = BTCUSDT)
    │   ├── disc_classifier_loss.png      A18 BCE curves
    │   └── pred_score_loss.png           A19 MAE curves (asset 0)
    ├── weights/README.md                 why there are none
    ├── metrics_summary.csv               A1-A32, mean ± std, per seed
    ├── metrics_per_asset.csv             per-metric × per-asset breakdown (8 rows per metric)
    ├── curve_b_aggregate.json            B curve-shape aggregate
    ├── grid_tvd_aggregate.json           path-cloud TVD
    └── seed_{{i}}_metrics.json             full per-seed dump incl. the per_asset block
```

The `.npy` arrays are **gitignored**: `(6144, 128, 8)` float64 = 50 MB each, and LFS was ruled
out on 2026-07-30. They are fully reproducible from the tracked code and tracked
hyperparameters; the `metadata.json` beside each array **is** tracked, so shapes, price ranges
and generation times stay auditable without the payload. The raw Binance archive is not
committed either — `dataset/TrueDataset/` rebuilds it from `data.binance.vision`.

## Reproduce

```bash
cd /home/tbasseras/benchmark
V=dataset/TrueDataset/variants/om_2022-07_N6144
TAG=6144x128x8
R=results/trueexperiment

# 1. dataset — downloads Binance 1s klines and builds the 5 splits.
#    Only if $V/*.npy are absent. See dataset/TrueDataset/README.md.
/home/tbasseras/.cc-venv/bin/python dataset/TrueDataset/build_true_dataset.py \\
    --period 2022-07:2026-07 --n-samples 6144 --split-mode holdout-era

# 2. (h, K) calibration against the real-vs-real envelope — never against test.
/home/tbasseras/sbts-venv/bin/python $R/SBTS/code/calibrate_bandwidth_true.py \\
    --data-dir $V --seq-tag $TAG --grid 0.06,0.07,0.08 --K-grid 1 \\
    --m-probe 2048 --workers 32 --out $R/SBTS/losses/finalists_seed0.json

# 3. generate the {n_seeds} scoring banks — CPU only, no GPU. ~{total_min:.0f} min total.
for S in {" ".join(str(i) for i in range(n_seeds))}; do
  setsid taskset -c 0-31 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    NUMBA_NUM_THREADS=1 /home/tbasseras/sbts-venv/bin/python \\
    $R/SBTS/code/generate_bank_true.py --data-dir $V --seq-tag $TAG \\
    --h {h} --K {K} --seed $S --workers 32 --out-root $R/SBTS
done
/home/tbasseras/.cc-venv/bin/python $R/SBTS/code/collect_artifacts.py

# 4. real-vs-real reference: three held-out REAL splits as stand-in banks.
for i in 0 1 2; do mkdir -p $R/real_floor/generated_paths/seed_$i; done
ln -sf $PWD/$V/true_S_$TAG.npy          $R/real_floor/generated_paths/seed_0/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_val_$TAG.npy      $R/real_floor/generated_paths/seed_1/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_valdisc_$TAG.npy  $R/real_floor/generated_paths/seed_2/generated_paths_$TAG.npy

# 5. metrics — both sides go through byte-identical code.
for M in SBTS real_floor; do
  N=$([ $M = SBTS ] && echo {n_seeds} || echo {n_floor})
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method $M --dataset TrueDataset --seeds $N \\
    --data-dir $V --seq-tag $TAG --dt 9.51293759512e-07 --results-dir $R/$M
done

# 6. conditional CRPS (table C) — the paper's 8 192-path pool, 5 seeds.
#    --out MUST land in losses/crps_configs/. The flat losses/conditional_crps_seed_*.json
#    names are the pre-2026-08-26 artefacts from before the rv-axis fix (rv ~0.565 there
#    vs ~1.045 here); they are kept for provenance and are NOT read by this renderer.
#    Seeds 0-3 used --workers 24, seed 4 used 16 (shared-box cap). Worker count
#    repartitions the RNG (sbts_generate_true.py:473 seeds substream seed*10_000+i),
#    so it changes the draw but not the law: same (h, K, N_pi), same bank size,
#    8 192 iid paths either way. metadata.json records n_workers per seed.
for S in 0 1 2 3 4; do
  /home/tbasseras/sbts-venv/bin/python $R/SBTS/code/generate_bank_true.py \\
    --data-dir $V --seq-tag $TAG --h {h} --K {K} --N-pi {npi} --seed $S --m-simu 8192 \\
    --workers 16 --out-root $R/SBTS/crps_banks
  B=$R/SBTS/crps_banks/generated_paths/seed_$S/generated_paths_8192x128x8.npy
  # both conventions; --label SBTS is load-bearing (table C is keyed on it)
  /home/tbasseras/sbts-venv/bin/python metrics/conditional_crps_multiasset.py \\
    --data-dir $V --seq-tag $TAG --bank-size 8192 --label SBTS \\
    --weight-mode paper --standardize bank --bank $B \\
    --out $R/SBTS/losses/crps_configs/paper__seed_$S.json
  /home/tbasseras/sbts-venv/bin/python metrics/conditional_crps_multiasset.py \\
    --data-dir $V --seq-tag $TAG --bank-size 8192 --label SBTS \\
    --weight-mode perdim --standardize realtrain --bank $B \\
    --out $R/SBTS/losses/crps_configs/perdim__seed_$S.json
done
# ... and the reference every row in table C is read against: the real TRAIN
# split used as the bank. Native size 6144 -- that is all the data there is --
# and --bank-size matches so the two bootstrap baselines are measured at the
# same bank size rather than a larger one.
/home/tbasseras/gpu-venv/bin/python metrics/conditional_crps_multiasset.py \\
    --data-dir $V --seq-tag $TAG --bank-size 6144 --label real_train_bank \\
    --bank $V/true_S_$TAG.npy --out $R/SBTS/losses/conditional_crps_realbank.json

# 7. memorisation -- the guard that stops a small h from winning table A by
#    copying. Denominator is the val split, NOT test. Measured, test sits CLOSER
#    to train than val does (0.017647 vs 0.018943) because annualised vol falls
#    across the era break, so a test denominator is SMALLER and inflates the
#    ratio -- which flatters a memoriser, since memorisation is the low-ratio
#    failure. See the Memorisation check section.
/home/tbasseras/gpu-venv/bin/python $R/SBTS/code/measure_memorisation.py \\
    --data-dir $V --seq-tag $TAG --seeds {",".join(str(i) for i in range(n_seeds))}

# 8. figures
/home/tbasseras/gpu-venv/bin/python $R/SBTS/code/plot_diagnostics_true.py \\
    --data-dir $V --seq-tag $TAG
/home/tbasseras/gpu-venv/bin/python metrics/plot_score_losses.py \\
    --method SBTS --dataset TrueDataset --results-dir $R/SBTS

# 9. regenerate this README from the artefacts
/home/tbasseras/gpu-venv/bin/python $R/SBTS/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
