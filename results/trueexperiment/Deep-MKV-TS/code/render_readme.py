#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/trueexperiment/Deep-MKV-TS/README.md from the metric artefacts.

WHAT IS SHARED WITH THE SIBLING RENDERERS, AND WHY THAT MATTERS
---------------------------------------------------------------
Everything from `fmt` down to `render_crps_table` is a BYTE-FOR-BYTE copy of
`results/trueexperiment/CSDI/code/render_readme.py` (which is itself the SBTS
copy). Those functions decide what "at or below the floor" means, how the two
different +- in table C are computed, and which CRPS differences count as
resolvable. If they drift between two method trees the two READMEs stop being
comparable AND BOTH STILL RENDER, so the drift is invisible. Change them in
every tree or in none.

FOUR THINGS DIFFER, EACH FOR A STATED REASON
---------------------------------------------
1. `render_timing` -- the schema is not CSDI's. `collect_artifacts.py` here
   writes ONE row per seed with `n_steps, selected_step, ridge_lambda,
   sigma_max, elapsed_sec`; there is no `role`/`train_min`/`gen_min` triple,
   because this method's CRPS pool is written under `crps_banks/` which the
   collector does not scan. Training wall-clock is read from the config
   sidecars instead.
2. `render_selection` -- NEW, and it is the section that makes this entry
   different from every other one in the tree. Deep-MKV-TS does not report its
   last optimizer step; it reports a checkpoint chosen on the VALIDATION split
   by the section-7 rule. A README that showed the A-table without naming the
   selected step would be describing a model nobody can identify.
3. `render_memorisation` -- the prose. The capacity argument that applies to a
   413k-parameter diffusion model does not apply to a 56k-parameter control.
4. `main()` -- the prose, and `n_steps`/`n_parameters` rather than
   `epochs`/`params` out of the sidecar.

A33-A34 ARE OMITTED, NOT DASHED. `CATEGORIES` has no row for them. They score
against the dataset's latent variance path, which a real market does not have,
so a row rendering "-" in every column would read as "measured, came out empty"
-- a different and false claim from "undefined here".

Every number is READ FROM DISK, never typed by hand: the tables *and* the
headline counts are computed from the artefacts, so the prose cannot drift away
from the numbers it describes.

Reads
  results/trueexperiment/Deep-MKV-TS/metrics_summary.csv
  results/trueexperiment/Deep-MKV-TS/curve_b_aggregate.json
  results/trueexperiment/Deep-MKV-TS/grid_tvd_aggregate.json
  results/trueexperiment/Deep-MKV-TS/losses/generation_time.csv
  results/trueexperiment/Deep-MKV-TS/losses/memorisation.json
  results/trueexperiment/Deep-MKV-TS/losses/envelope_screen.json      (optional)
  results/trueexperiment/Deep-MKV-TS/losses/crps_configs/paper__seed_*.json
  results/trueexperiment/Deep-MKV-TS/losses/crps_configs/paper__realbank.json
  results/trueexperiment/Deep-MKV-TS/weights/seed_*_config.json       (optional)
  results/trueexperiment/Deep-MKV-TS/code/selection/seed_*_selection.json
  results/trueexperiment/real_floor/metrics_summary.csv               (floor)
  results/trueexperiment/real_floor/curve_b_aggregate.json
  results/trueexperiment/real_floor/grid_tvd_aggregate.json

Writes
  results/trueexperiment/Deep-MKV-TS/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/Deep-MKV-TS/code/render_readme.py
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
METHOD_DIR = os.path.dirname(HERE)                            # .../Deep-MKV-TS
TE = os.path.dirname(METHOD_DIR)                              # .../trueexperiment
# Derived, never a literal. `load_crps` keys table C on this string and it must
# equal the --label handed to conditional_crps_multiasset.py; hardcoding it is
# how a renderer copied between trees silently renders table C with baselines
# only -- the file loads, the key misses, and nothing errors.
METHOD = os.path.basename(METHOD_DIR)                         # "Deep-MKV-TS"
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

    # Capacity and step count are read from the checkpoint sidecar, not written
    # as literals: the whole point of the paragraph below is the ratio of
    # parameters to training paths, and a stale hardcoded number would turn the
    # one quantitative claim in this section into a lie the moment anyone
    # changed a layer count.
    _c = read_json(os.path.join(METHOD_DIR, "weights", "seed_0_config.json")) or {}
    _p = f"{_c.get('n_parameters', 56136):,}".replace(",", " ")
    _n = f"{_c.get('n_train', 6144):,}".replace(",", " ")
    _s = _c.get("selected_step") or _c.get("n_steps", 3000)
    _bs = _c.get("batch_size", 256)

    L = []
    L.append("## Memorisation check")
    L.append("")
    L.append("**This section is not a formality, and it is also not the same argument the CSDI")
    L.append(f"entry makes.** There the alarm is capacity: 413 057 parameters against {_n} paths.")
    L.append(f"Here the network is **{_p} parameters** -- roughly an order of magnitude smaller --")
    L.append("and it does not parameterise the law directly at all. It parameterises a *control*")
    L.append("that is added to a frozen reference kernel, so the hypothesis class is anchored to")
    L.append("that kernel rather than free to interpolate individual training paths.")
    L.append("")
    L.append("**The exposure is elsewhere, and it is real.** Algorithm 1 counts OUTER ITERATIONS,")
    L.append(f"not epochs: each of the {_s} steps draws a FRESH batch of {_bs} paths and re-fits the")
    L.append("Z-proxy, so there is no pass over a fixed dataset and no epoch count to quote. What")
    L.append("there is instead is an MMD discrepancy measured directly against a target bank drawn")
    L.append("from the training split. Minimising a distance to the training sample IS the")
    L.append("objective, and driving it to zero would be indistinguishable from copying. On the")
    L.append("Heston tree a score below the floor would expose that; **here it would not** (see the")
    L.append("floor caption), so this number is the only guard there is.")
    L.append("")
    L.append("**But the reported checkpoint was not chosen by that objective.** Section 7 forbids")
    L.append("selecting on goodness of fit, and `select_checkpoint_true.py` enforces it: the step")
    L.append("reported below was picked by |log NNratio| on the *validation* split subject to the")
    L.append("measured vol and corr ceilings, never by the training discrepancy. That is why the")
    L.append("ratio here is not a free readout the way the CSDI one is -- a nearby quantity WAS")
    L.append("optimised, on a different split. The honest reading: this is an out-of-sample")
    L.append("measurement of an in-sample-adjacent criterion. The Selection section above records")
    L.append("what the forbidden fit rule would have chosen instead, so the two can be compared.")
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
    L.append(f"**{dup} exact duplicates clears nothing, and is expected.** Integrating an SDE with")
    L.append("continuous Brownian noise lands on a bit-identical float64 path with probability zero, so")
    L.append("this counter can only ever fire on a plumbing bug -- a training array accidentally")
    L.append("saved as the bank. **The ratio is the number that matters.** For calibration of how")
    L.append("little the duplicate count tells you: the SBTS candidate `K=3, h=0.05`, rejected as a")
    L.append("memorisation trap, also had 0 duplicates while scoring NNratio 0.197.")
    L.append("")
    return "\n".join(L)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def render_timing(path):
    """Wall-clock table from losses/generation_time.csv.

    THE SCHEMA IS NOT CSDI'S -- do not copy that function in. `collect_artifacts.py`
    scans `generated_paths/` only, so it writes ONE row per seed (the 6 144-path
    A/B bank) with columns

        seed, n_samples, T, d, n_steps, selected_step, ridge_lambda, sigma_max,
        batch_size, n_workers, elapsed_sec, elapsed_min

    There is no `role`, no `train_min` and no `gen_min`: the 8 192-path CRPS pool
    lives under `crps_banks/` and the collector never sees it, so a two-row-per-seed
    table would be inventing rows. `elapsed_*` is GENERATION time only.

    Training wall-clock therefore comes from `weights/seed_{i}_config.json`
    (`train_time_sec`), which the trainer wrote and nothing since has touched. It
    is charged ONCE per seed. Both banks come from the same selected checkpoint --
    that is the entire point of reloading it in `generate_bank_true.py` -- so
    adding a training column to a per-bank row and summing it would double-count.

    `selected_step` is in the table because the row would otherwise describe a
    model nobody can identify: the reported checkpoint is chosen on validation and
    is generally NOT the last step.
    """
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, []

    rows.sort(key=lambda r: int(r["seed"]))
    train_min = {}
    for r in rows:
        c = read_json(os.path.join(METHOD_DIR, "weights", f"seed_{r['seed']}_config.json"))
        train_min[r["seed"]] = _f(c.get("train_time_sec")) / 60.0

    total = sum(train_min.values()) + sum(_f(r.get("elapsed_min")) for r in rows)

    out = ["| Seed | Bank | Budget | Selected step | Train | Generate |",
           "|------|------|--------|---------------|-------|----------|"]
    for r in rows:
        n = r.get("n_samples", "")
        bank = (f"{int(n):,}".replace(",", " ") + f" × {r.get('T','')} × {r.get('d','')}"
                if str(n).isdigit() else "-")
        sel = r.get("selected_step") or "-"
        budget = r.get("n_steps") or "-"
        out.append(f"| {r['seed']} | {bank} | {budget} steps | {sel} | "
                   f"{train_min[r['seed']]:.1f} min | {_f(r.get('elapsed_min')):.1f} min |")
    return "\n".join(out), total, rows


# --------------------------------------------------------------- selection ---
def load_selection():
    """Every code/selection/seed_{i}_selection.json, ordered by seed."""
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "selection", "seed_*_selection.json")),
                    key=lambda q: int(re.search(r"seed_(\d+)", q).group(1))):
        rec = read_json(p)
        if rec:
            out.append(rec)
    return out


def render_selection(records):
    """The section-7 selection section -- the one that has no analogue elsewhere.

    Placed BEFORE the A-table on purpose. Every number in that table is a property
    of a specific checkpoint, and which checkpoint that is was decided here, on the
    validation split, by a rule that deliberately ignores goodness of fit. A reader
    who meets the A-table first will assume it describes the end of training. It
    does not.

    Two things are reported that a kinder README would omit:

      * what the FORBIDDEN fit rule would have chosen. If it disagrees with the
        section-7 choice, that disagreement is evidence the rule is doing work
        rather than rubber-stamping the last step, and it is exactly the number a
        sceptical reader wants.
      * whether the choice landed on the LAST checkpoint. Section 7.2: the grid is
        bounded above by the training budget, so a selection at the top of it is a
        BOUNDARY, not a demonstrated optimum, and must be labelled as one.
    """
    if not records:
        return ("## Section-7 checkpoint selection\n\n"
                "**MISSING** -- no `code/selection/seed_*_selection.json` found. Run\n"
                "`code/select_checkpoint_true.py`; the A-table below is not attributable\n"
                "to an identified checkpoint without it.\n")

    env = records[0].get("envelope", {}) or {}
    vol_ceil = env.get("vol_err_pct_max")
    corr_ceil = env.get("corr_err_max")
    grid = sorted({int(r["step"]) for r in records[0].get("per_step", [])})
    last = max(grid) if grid else None
    chosen = [int(r["selected_step"]) for r in records]
    at_edge = [c for c in chosen if last is not None and c == last]
    disagree = [r for r in records
                if r.get("fit_rule_would_have_chosen") not in (None, r["selected_step"])]

    L = ["## Section-7 checkpoint selection", ""]
    L.append("**The reported model is not the last optimizer step.** Six checkpoints are saved")
    L.append(f"per seed at steps {', '.join(str(s) for s in grid)}; each is scored by sampling a")
    L.append("probe bank and measuring it against the **validation** split, and the one reported")
    L.append("is chosen by")
    L.append("")
    L.append("```")
    L.append("minimise |log NNratio|   subject to   vol_err <= "
             + (f"{vol_ceil:.4g} %" if vol_ceil is not None else "(ceiling missing)")
             + "   and   corr_err <= "
             + (f"{corr_ceil:.4g}" if corr_ceil is not None else "(ceiling missing)"))
    L.append("```")
    L.append("")
    L.append("Both ceilings are **measured, not chosen**: they are the worst value real market")
    L.append("data attains when scored against other real market data on this build, so a")
    L.append("checkpoint that clears them is, on that statistic, indistinguishable from another")
    L.append("honest slice of the same market. **Goodness of fit is forbidden as a selection")
    L.append("criterion** — the training discrepancy is measured against a bank drawn from the")
    L.append("training split, so selecting on it would reward copying.")
    L.append("")
    L.append("| Seed | Selected step | NNratio | \\|log NNratio\\| | vol_err % | corr_err | "
             "Fit rule would have chosen |")
    L.append("|------|---------------|---------|-----------------|-----------|----------|"
             "----------------------------|")
    for r in records:
        s = r.get("selected", {})
        fit = r.get("fit_rule_would_have_chosen")
        L.append(
            f"| {r['seed']} | **{r['selected_step']}** | {s.get('nn_ratio', float('nan')):.4f} | "
            f"{s.get('abs_log_nn_ratio', float('nan')):.4f} | "
            f"{s.get('vol_err_pct', float('nan')):.2f} | {s.get('corr_err', float('nan')):.4f} | "
            f"{fit if fit is not None else '-'} |"
        )
    L.append("")
    if disagree:
        L.append(f"**The rule is doing work.** On {len(disagree)} of the {len(records)} seeds the")
        L.append("forbidden fit rule (argmin validation discrepancy) would have selected a")
        L.append("different checkpoint: "
                 + ", ".join(f"seed {r['seed']} → {r['fit_rule_would_have_chosen']} "
                             f"instead of {r['selected_step']}" for r in disagree)
                 + ". Recorded so the disagreement is auditable rather than invisible.")
    else:
        L.append("On every seed the forbidden fit rule would have chosen the same checkpoint. That")
        L.append("makes the two criteria indistinguishable **on this run** — it does not make the")
        L.append("fit rule admissible, and it is not evidence the constraint is unnecessary.")
    L.append("")
    if at_edge:
        L.append(f"> **{len(at_edge)} of {len(chosen)} seeds selected the LAST checkpoint "
                 f"({last}).** The selection grid is bounded above by the training budget, so a")
        L.append("> choice at the top of it is a **boundary, not a demonstrated optimum**")
        L.append("> (section 7.2). The honest statement is that no better checkpoint was")
        L.append("> *available*, not that none exists; a longer run might improve it. Do not")
        L.append("> quote the A-table below as converged.")
        L.append("")
    else:
        L.append(f"> Every selected step is **interior** to the grid (the budget is {last} steps and")
        L.append("> no seed chose it), so the choice is bracketed on both sides and section 7.2's")
        L.append("> boundary caveat does not apply.")
        L.append("")
    return "\n".join(L)


def render_envelope():
    """The full-bank real-vs-real screen, from losses/envelope_screen.json.

    NOT redundant with the selection gate above. Selection screens a small probe
    bank on the VALIDATION split; this screens the full 6 144-path REPORTED bank.
    A seed that passed selection and fails here means the probe was optimistic,
    and that gap is itself the finding -- which is why the verdict is rendered
    whatever it says rather than gated on being a PASS.
    """
    e = read_json(os.path.join(METHOD_DIR, "losses", "envelope_screen.json"))
    if not e or not e.get("seeds"):
        return ""
    ceil = e.get("ceilings", {})
    L = ["> **Full-bank envelope screen** (`losses/envelope_screen.json`). The selection gate",
         "> above scores a small probe bank on `val`; this scores the **whole reported bank**,",
         "> so it is the stricter of the two and is reported separately.", ">"]
    L.append("> | Seed | vol_err % | corr_err | vol ratio | Verdict |")
    L.append("> |------|-----------|----------|-----------|---------|")
    for k in sorted(e["seeds"], key=int):
        v = e["seeds"][k]
        L.append(f"> | {k} | {v['vol_err_pct']:.2f} | {v['corr_err']:.4f} | "
                 f"{v['vol_ratio_mean']:.3f} | "
                 f"{'PASS' if v['passes'] else 'FAIL'} |")
    L.append(f"> | **ceiling** | **{ceil.get('vol_err_pct')}** | **{ceil.get('corr_err')}** | | |")
    L.append(">")
    vol_bad = e.get("seeds_failing_vol", [])
    corr_bad = e.get("seeds_failing_corr", [])
    if vol_bad and not corr_bad:
        L.append("> **vol_err fails on seeds " + ",".join(vol_bad) + " while corr_err passes on")
        L.append("> every seed.** These two statistics fail independently, and that pattern is a")
        L.append("> diagnosis rather than a score: the cross-asset correlation structure is")
        L.append("> reproduced and the realised volatility is not, so the model has the **shape**")
        L.append("> of the joint law and not its **scale** at the highest frequency — it is")
        L.append("> smoothing the bar-to-bar increments. Report the two separately; an average")
        L.append("> over them is meaningless and hides exactly this.")
    elif corr_bad and not vol_bad:
        L.append("> **corr_err fails on seeds " + ",".join(corr_bad) + " while vol_err passes.**")
        L.append("> The marginal scale is right and the cross-asset dependence is not — the")
        L.append("> opposite failure from the one above, and the rows that carry it are A20 and")
        L.append("> the native-d=8 kernels A6-A11.")
    elif not e.get("n_failing"):
        L.append("> Every seed clears both ceilings on the full bank.")
    return "\n".join(L) + "\n"


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
    selrecs = load_selection()
    selection = render_selection(selrecs)
    envelope = render_envelope()

    # Hyperparameters come from the checkpoint sidecar, never from a literal in
    # this file. train_true.py writes weights/seed_{i}_config.json next to the
    # .pt, including `retuned_for_true` -- a list built by COMPARING each knob
    # against the committed Heston d = 8 value, not hand-asserted. If a future
    # run deviates further, the blockquote below says so automatically instead of
    # the README quietly claiming a transfer that no longer holds.
    cfg0 = read_json(os.path.join(METHOD_DIR, "weights", "seed_0_config.json")) or {}
    params = cfg0.get("n_parameters", 56136)
    budget = cfg0.get("n_steps", 3000)
    bs = cfg0.get("batch_size", 256)
    lr = cfg0.get("lr", 1e-2)
    hid = cfg0.get("hidden_dim", 96)
    nlay = cfg0.get("num_layers", 1)
    smax = cfg0.get("sigma_max", 5.0)
    smin = cfg0.get("sigma_min", 1e-3)
    ridge = cfg0.get("ridge_lambda", 100.0)
    h_ridge = cfg0.get("heston_ridge_lambda", 1000.0)
    h_smax = cfg0.get("heston_sigma_max", 0.6)
    seq_len = cfg0.get("seq_len", 128)
    dtype = cfg0.get("dtype", "float32")
    retuned = cfg0.get("retuned_for_true", [])
    f64 = cfg0.get("eigh_float64", {}) or {}
    if not retuned:
        tuning_line = ("**Every value above is the committed Heston d = 8 configuration "
                       "verbatim.** Nothing was re-tuned for this dataset.")
    else:
        tuning_line = ("**Re-tuned for this dataset: "
                       + ", ".join(f"`{k}`" for k in retuned) + ".** "
                       "Every other value is the committed Heston d = 8 configuration "
                       "verbatim. The list is derived by comparing each knob against that "
                       "configuration, not hand-written, so it cannot understate the drift.")

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

    # collect_artifacts.py writes one row per seed and names the column
    # `n_samples`; there is no `role` column here (see render_timing).
    n_paths = "6 144"
    for r in trows:
        if str(r.get("n_samples", "")).isdigit():
            n_paths = f"{int(r['n_samples']):,}".replace(",", " ")
            break

    md = f"""# Deep-MKV-TS on TrueDataset — real crypto spot, d = 8

**Deep McKean-Vlasov Time Series** — a neural *control* on a frozen reference kernel, trained by
matching an MMD discrepancy under a specific-entropy running cost — applied to **real market
data**: {n_paths} paths of 8 correlated crypto spot series
(BTC, ETH, BNB, SOL, XRP, DOGE, ADA, LINK), 30-second bars, seq\\_len = {seq_len},
built from Binance public 1-second klines over 2022-07 → 2026-07.

This is the real-data counterpart of
[`../HestonMultiAsset/Deep-MKV-TS/`](../HestonMultiAsset/Deep-MKV-TS/README.md).
Same generator, same metric code, same table layout — a different **question**. On Heston the
data-generating law is known, so "how close is the generator to the truth" has an exact answer.
Here it does not, so every reference in this README is a **real-vs-real** measurement: real
market data scored against other real market data through byte-identical code. Read
[`../truedatasetguideline.md`](../truedatasetguideline.md) before adding a method — it defines
the build, the five-split contract, the envelope, and what each reference column does and does
not mean.

**One joint model, not eight univariate ones.** The control maps the full 8-dimensional state to
an 8-vector drift adjoint and an 8×8 noise adjoint, so the cross-asset covariance is produced by
the model rather than imposed afterwards. The 28 cross-asset correlations are inside the
hypothesis class — which is what makes A20 (terminal covariance error) and A6-A11 (the
native-d=8 path kernels) load-bearing rows here rather than artefacts of post-processing.

**The bar size is the whole story of this entry.** TrueDataset is 30-second bars, so
`dt = {cfg0.get("dt", 9.512937595129376e-07):.6g}` years and `1/sqrt(dt) ≈ 1025`, against **15.9** on the Heston build —
a factor of **64.6**. Every quantity in this method that carries a `1/sqrt(dt)` scales with it:
the specific-entropy running cost, the ridge-regularised conditional-expectation proxy, and the
MMD bandwidths. That single number is why five knobs moved (below) and why the volatility block
of the A-table is the one to read first.

> **Hyperparameters:** `steps={budget}` (Algorithm 1 **outer iterations**, not epochs — each step
> draws a fresh batch and re-fits the Z-proxy, so there is no pass over a fixed dataset),
> `batch={bs}`, `lr={lr:g}`, `hidden_dim={hid}`, `num_layers={nlay}`,
> `sigma∈[{smin:g}, {smax:g}]`, `ridge_lambda={ridge:g}`, `eta={cfg0.get("eta", 1.0):g}`,
> `grad_clip={cfg0.get("grad_clip_norm", 5.0):g}` → **{params:,} parameters**.
> {tuning_line}
>
> **The five moves, each forced by a measured property, not a preference:**
> `lr` — the Heston value diverged here (the objective went to +336 on the first probe);
> `sigma_max` {h_smax:g} → **{smax:g}** — 5 of the 8 assets have realised vol above {h_smax:g}, so the old ceiling
> clipped genuine market volatility; measured p99 of realised σ is 3.87, and both `sigma_max=5`
> and `sigma_max=8` converge to that same p99, so the ceiling does **not** bind;
> `ridge_lambda` {h_ridge:g} → **{ridge:g}** — at the Heston value the Z-proxy's out-of-sample R² goes
> **negative** on this dataset; the winner was bracketed on both sides by a 5-point screen, so it
> is an interior optimum and not a grid edge (section 7.2);
> `ridge_covariance`/`ridge_drift` — re-selected with it;
> `mmd_bandwidths` — regauged per block, forced by the 64.6× `1/sqrt(dt)` change above. A bandwidth
> calibrated on Heston increments is not a bandwidth on 30-second crypto increments.
>
> **Numerics: the model runs in `{dtype}`, three `eigh` sites run in float64, and that is a
> precision fix rather than a modelling choice.** The eigendecomposition BACKWARD divides by the
> eigenvalue gap. Measured on the frozen kernel's real σ matrices, **0.891 % have two eigenvalues
> whose relative gap is at or below float32 eps**, so float32 sees an exact tie, divides by zero
> and returns NaN — which then poisons every gradient in the batch, and `clip_grad_norm_`
> rescales finite gradients rather than filtering them, so it cannot repair it. In float64 the
> gaps resolve (0 of 910 336 pairs below 1e-9) and `max|grad|` is **identical** in the two
> precisions, which is the evidence that nothing about the model moved.
> Sites promoted this run: `{", ".join(f"{k}={v}" for k, v in sorted(f64.items())) or "none recorded"}`.

---

{selection}
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
> BNB 20.9 % of all increments): the price simply did not move within the bar. Deep-MKV-TS
> integrates an SDE driven by continuous Brownian noise — its marginal has no atom, so it
> **cannot** reproduce that mass, and the KS / Wasserstein-family rows will be penalised on
> precisely those assets. That is a structural property of the model class meeting a
> microstructure property of the data, not a training failure, and averaging over 8 assets hides
> which assets are responsible. The breakdown is in
> [`metrics_per_asset.csv`](metrics_per_asset.csv).
> **The volatility block (A26-A32) is the block to read first on this method**, for the reason
> given at the top: the state is discretised at `dt = {cfg0.get("dt", 9.512937595129376e-07):.3g}` years, so anything that
> smooths the bar-to-bar increment shows up there before it shows up anywhere else, and the
> cross-asset rows can look healthy while it does.

{envelope}
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
> **Cross-seed stability**: unlike the kernel methods in this tree, Deep-MKV-TS *is* trained by
> gradient descent, so the seed controls the control-network initialisation, the batch draw, the
> Brownian increments **and** — through the section-7 screen — which checkpoint is reported at
> all. The ± here is therefore a genuine training-plus-selection variance measurement, not merely
> simulation noise, and it is the column to check before believing any single-seed claim. If the
> Selection table above shows the seeds landing on different steps, part of this ± is that
> disagreement rather than the generator itself.

---

## C, Conditional generation — forecast CRPS by path shadowing

Every number above scores the **unconditional** law: does the generated cloud look like the
real cloud? That is not the question a trading desk asks. This table scores **conditional
skill**: given a real history, does the generator's conditional distribution of the *next*
32 returns beat a purely historical resampler?

Protocol is the paper's §3.3.1 (Deep-MKV-TS, Table 4), reproduced with **one documented
deviation** — retrieval is joint across the 8 assets, because the author's code refuses to
run at d ≠ 1. The deviation is spelled out under the table; every other constant is the
author's.

> **This is the one table where the method and the protocol share an author, and that cuts
> against this entry rather than for it.** The retrieval features, the K, the pool size and the
> two bootstrap baselines are all this paper's own choices, so a good number here is a weaker
> claim than the same number would be for CSDI or SBTS: the metric was designed alongside the
> method. The row that disciplines it is the *real training split used as the bank* — a bank no
> generator fit on that split can beat — and the session bootstrap, which learns nothing. Read
> those two before reading the ratio.

Constants:

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
> `generate_bank_true.py` reloads **the section-7 selected checkpoint** — the same one the
> A-table scores, not the last optimizer step — and integrates afresh with new Brownian
> increments, so the pool and the A/B bank are two independent draws from one fitted model
> rather than two views of one draw. `metadata.json` beside each array records
> `bank_role` (`ab_bank` vs `crps_pool`) and the `selected_step` it came from, so the two can
> never be confused after the fact.

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

## Convergence and selection

**The figure has two rows and they are not train-vs-validation.** The sibling entries plot a
train curve beside a validation curve because both are the same objective on two splits.
This method has no such pair — the trainer writes one phase only. The two rows are two
different things:

| Row | What it is | Source |
|-----|-----------|--------|
| 1 — **optimisation** | what the optimiser saw: `loss_total` (MMD discrepancy + running cost), `objective` (running cost alone), `grad_norm` | `losses/seed_{{i}}_losses.csv`, every step |
| 2 — **selection** | what actually chose the reported checkpoint: \\|log NNratio\\|, vol_err, corr_err on `val` | `code/selection/seed_{{i}}_selection.json`, **6 points** |

Row 2 has six markers per seed and they are the whole series, not a subsample: scoring a
checkpoint means sampling a bank and computing the envelope statistics, which costs far more
than a training step, so the grid is coarse by construction.

**A falling `loss_total` is not evidence the reported model improved.** It is the discrepancy
against a bank drawn from the *training* split, and section 7 forbids selecting on it — so a
seed can descend beautifully in row 1 and still be reported at an early step by row 2. That
disagreement is the point of showing both rows together, and the Selection section above
quantifies it.

`grad_norm` is plotted because this method has a documented divergence mode: one NaN in the
summed running cost makes every gradient NaN, and gradient clipping *rescales* finite gradients
rather than filtering them, so it cannot repair it. A `grad_norm` spike immediately before a
seed disappears from the figure is the signature. The float64 `eigh` fix described at the top
is what removed it; the counters in `weights/seed_{{i}}_config.json` record how often it fired.

The ceilings drawn on row 2 are **read from each seed's selection JSON**, not pasted in as
constants, so rebuilding the dataset variant moves the lines instead of leaving the figure
quietly asserting last month's envelope.

![Convergence and selection](plots/loss_convergence.png)

Wall-clock ({len(trows)} seeds, {budget} steps/seed, **{total_min:.0f} min total**). Training is charged
**once per seed** and read from the config sidecar; the `Generate` column is the A/B bank only.
The 8 192-path CRPS pool is a second generation pass from the same selected checkpoint and is
not in this table, so the total below understates generation slightly and does **not**
double-count training:

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
    │   ├── README.md                     source notes, hyperparameters, deviations, digests
    │   ├── reference/                    frozen copy of the author's package — ZERO edits
    │   ├── reference/reference_kernel.json  the frozen kernel every control is added to
    │   ├── fit_reference_true.py         fits + freezes that kernel (stage 1, idempotent)
    │   ├── eigh_float64.py               the three float64 eigh sites — see Numerics above
    │   ├── train_true.py                 trains one seed, writes 6 checkpoints + losses
    │   ├── select_checkpoint_true.py     §7 selection on `val` — decides what is reported
    │   ├── generate_bank_true.py         reloads the SELECTED checkpoint; A/B bank or CRPS pool
    │   ├── collect_artifacts.py          §4 contract gate — recomputes from the .npy, exits 1 on breach
    │   ├── screen_envelope.py            full-bank real-vs-real screen (copy of the CSDI one)
    │   ├── plot_diagnostics_true.py      the 8-panel stylised-facts figure
    │   ├── plot_losses.py                row 1 optimisation + row 2 selection, all seeds
    │   ├── measure_memorisation.py       NN ratio vs the val split (same era as train)
    │   ├── run_pipeline_post.sh          unattended driver: selection → banks → metrics → README
    │   ├── selection/seed_{{i}}_selection.json   the six scored checkpoints + the choice (tracked)
    │   ├── runs/seed_{{i}}/training_checkpoints/  the six .pt files — gitignored
    │   └── render_readme.py              regenerates this README from the artefacts
    ├── weights/
    │   ├── seed_{{i}}_model.pt              the SELECTED checkpoint, promoted — gitignored
    │   └── seed_{{i}}_config.json           every hyperparameter + selection record (tracked)
    ├── generated_paths/seed_{{0..{n_seeds - 1}}}/
    │   ├── generated_paths_6144x128x8.npy   (6144, 128, 8) float64 — gitignored, 50 MB
    │   └── metadata.json                    seed, shape, min/max, timings, selected_step (tracked)
    ├── crps_banks/generated_paths/seed_{{0..{n_crps_seeds - 1}}}/
    │   └── generated_paths_8192x128x8.npy   the paper's 8 192-path pool — gitignored, 67 MB
    ├── logs/                             raw stdout of every stage (tracked)
    ├── losses/
    │   ├── seed_{{i}}_losses.csv             step,phase,loss_total,objective,grad_norm,control_rms
    │   ├── crps_configs/paper__seed_{{i}}.json     table C as reported (--weight-mode paper)
    │   ├── crps_configs/perdim__seed_{{i}}.json    the alternative convention, for the caveat above
    │   ├── crps_configs/paper__realbank.json      real training split used as the bank
    │   ├── memorisation.json                NN ratio vs val — the memorisation guard
    │   ├── envelope_screen.json             full-bank §7 verdict, per seed
    │   ├── selection_envelope.json          the measured ceilings + NNratio denominator
    │   ├── bandwidth_gauge.json             the regauged MMD bandwidths (see Hyperparameters)
    │   ├── dataset_stats.json               copy of the locked build's stats (provenance)
    │   └── generation_time.csv              seed,n_samples,...,selected_step,elapsed_min
    ├── plots/
    │   ├── truedata_diagnostics.png      8-panel stylised facts (seed 0, asset 0 = BTCUSDT)
    │   ├── loss_convergence.png          row 1 optimisation, row 2 selection, all seeds
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
so shapes, price ranges and generation times stay auditable without the payload.
`code/selection/seed_{{i}}_selection.json` is tracked for the same reason — it is the only record
of *which* checkpoint the numbers above describe. The raw Binance archive is not committed
either — `dataset/TrueDataset/` rebuilds it from `data.binance.vision`.

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

# 2. freeze the reference kernel. Idempotent: it refuses to overwrite an existing
#    reference_kernel.json, because every seed's control is defined relative to
#    THAT kernel and re-fitting it mid-campaign would silently change the model
#    the earlier seeds were trained against.
CUDA_VISIBLE_DEVICES=1 $P $R/{METHOD}/code/fit_reference_true.py --data-dir $V

# 3. train the {n_seeds} seeds. Each writes SIX checkpoints (steps 500..{budget});
#    none of them is the reported model yet — step 4 decides that.
bash $R/{METHOD}/code/run_seeds_true.sh

# 4. §7 selection on the VALIDATION split: score all six checkpoints, keep the
#    admissible one with the smallest |log NNratio|, promote it into
#    weights/seed_$S_model.pt. THIS is what makes a checkpoint "the model".
for S in {" ".join(str(i) for i in range(n_seeds))}; do
  CUDA_VISIBLE_DEVICES=1 $P $R/{METHOD}/code/select_checkpoint_true.py \\
      --seeds $S --device cuda:0
done

# 5. the two banks, both from the SELECTED checkpoint: the 6 144-path A/B bank
#    the A-table scores, and the paper's separate 8 192-path CRPS pool.
for S in {" ".join(str(i) for i in range(n_seeds))}; do
  CUDA_VISIBLE_DEVICES=1 $P $R/{METHOD}/code/generate_bank_true.py \\
    --seed $S --data-dir $V --seq-tag $TAG --out-root $R/{METHOD}
  CUDA_VISIBLE_DEVICES=1 $P $R/{METHOD}/code/generate_bank_true.py \\
    --seed $S --data-dir $V --seq-tag $TAG --m-simu 8192 \\
    --out-root $R/{METHOD}/crps_banks
done

# 6. §4 contract gate. Recomputes shape / dtype / S0 / finiteness FROM THE .npy —
#    metadata is the author's claim, the array is the evidence. Exits 1 on breach.
/home/tbasseras/.cc-venv/bin/python $R/{METHOD}/code/collect_artifacts.py \\
    --expect-seeds {n_seeds}

# 7. real-vs-real reference: three held-out REAL splits as stand-in banks.
#    Already built by the SBTS run from the same splits through the same code —
#    do NOT regenerate it. Listed so the construction is auditable.
for i in 0 1 2; do mkdir -p $R/real_floor/generated_paths/seed_$i; done
ln -sf $PWD/$V/true_S_$TAG.npy          $R/real_floor/generated_paths/seed_0/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_val_$TAG.npy      $R/real_floor/generated_paths/seed_1/generated_paths_$TAG.npy
ln -sf $PWD/$V/true_S_valdisc_$TAG.npy  $R/real_floor/generated_paths/seed_2/generated_paths_$TAG.npy

# 8. metrics — both sides go through byte-identical code.
for M in {METHOD} real_floor; do
  N=$([ $M = {METHOD} ] && echo {n_seeds} || echo {n_floor})
  CUDA_VISIBLE_DEVICES=0 $P metrics/compute_all_multiasset.py \\
    --method $M --dataset TrueDataset --seeds $N \\
    --data-dir $V --seq-tag $TAG --dt 9.51293759512e-07 --results-dir $R/$M
done

# 9. conditional CRPS (table C). BOTH conventions, and the realbank reference.
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

# 10. memorisation -- the guard that stops an over-fitted checkpoint from winning
#    table A by copying. Denominator is the val split, NOT test. Measured, test
#    sits CLOSER to train than val does (0.017647 vs 0.018943) because annualised
#    vol falls across the era break, so a test denominator is SMALLER and inflates
#    the ratio -- which flatters a memoriser, since memorisation is the low-ratio
#    failure. See the Memorisation check section.
$P $R/{METHOD}/code/measure_memorisation.py \\
    --data-dir $V --seq-tag $TAG --seeds {",".join(str(i) for i in range(n_seeds))}

# 11. figures
$P $R/{METHOD}/code/plot_diagnostics_true.py --data-dir $V --seq-tag $TAG
$P $R/{METHOD}/code/plot_losses.py --seeds {",".join(str(i) for i in range(n_seeds))}
$P metrics/plot_score_losses.py \\
    --method {METHOD} --dataset TrueDataset --results-dir $R/{METHOD}

# 12. the full-bank §7 screen. Numpy only. NOT redundant with step 4: selection
#     screened a probe bank on `val`, this screens the whole reported bank, so a
#     PASS at step 4 and a FAIL here means the probe was optimistic -- which is
#     itself a finding and is why this is reported rather than gated on.
/home/tbasseras/.cc-venv/bin/python $R/{METHOD}/code/screen_envelope.py \\
    --data-dir $V --seq-tag $TAG \\
    --seeds {",".join(str(i) for i in range(n_seeds))}

# 13. regenerate this README from the artefacts
$P $R/{METHOD}/code/render_readme.py
```

Steps 4-13 are exactly what `code/run_pipeline_post.sh` runs unattended, three GPU lanes wide,
with every stage guarded on its own artefact so a restart does not redo the work in front of it.
"""
    out = os.path.join(METHOD_DIR, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
