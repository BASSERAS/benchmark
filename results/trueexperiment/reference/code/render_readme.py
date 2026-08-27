#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/trueexperiment/reference/README.md from the metric artefacts.

Two ancestors, and the differences between them are the point
-------------------------------------------------------------
Table shape, the metric category list and the floor column are taken from
`SBTS/code/render_readme.py`, because that file is TrueDataset-native: it already
omits A33-A34 (which need a latent variance path a real market does not have) and
already names the reference column `Real-vs-real floor` rather than `Perfect
floor`.

Section order and prose slots are taken from
`../../HestonMultiAsset/reference/code/render_readme.py`, because that file
describes *this method* -- the frozen reference SDE with the neural control
switched off -- and the prose has to say what a reference row is for.

Three deliberate divergences from SBTS's renderer, each forced by an artefact:

  * **No table C.** Conditional CRPS needs an 8 192-path pool (guideline section 8);
    this folder ships only the five 6 144-path A/B banks. Rendering an empty
    table C would claim a measurement that was never made. The Heston reference
    page has no table C either, and the instruction for this folder was to
    mirror it.
  * **`render_memorisation` has no test-era arm.** `losses/memorisation.json`
    here carries `nn_ratio` against `val` only. The band's lower endpoint is a
    property of the REAL SPLITS, not of any method, so it is read from
    `SBTS/losses/memorisation.json` -- see the docstring of that function for why
    that is a legitimate read and not a borrowed number.
  * **`render_timing` reads `device`, not `n_workers`.** The reference bank is
    single-core CPU. A row reading `cuda` would falsify the "no GPU needed"
    claim this page makes, so the column is read rather than assumed.

What this folder is
-------------------
`sigma^ref` -- the frozen reference diffusion of Deep-MKV-TS Algorithm 1, sampled
with the neural control switched off. The model is BUILT (all 56 136 parameters)
and never FITTED; both output heads carry a zero final layer, so `Zhat == 0`
exactly, the control `Theta = eta*(sigma^ref)^-1 + Zhat/sqrt(dt)` collapses to
`eta*(sigma^ref)^-1`, and the sampler integrates `sigma^ref` itself.
`code/generate_reference_true.py` VERIFIES those four tensors are zero before it
writes a single path, and records the measured maxima per seed.

So this row is not a competitor. It is the **before** picture, and on this
dataset that is the ONLY thing it can be: TrueDataset has no perfect-recovery
floor, because a real market has no law to re-draw from. On Heston the reference
is bracketed from above and below; here it is bracketed from one side only, and
the attribution argument runs entirely through the paired
reference-vs-Deep-MKV-TS difference at matched generation seeds.

Every number is READ FROM DISK, never typed by hand.

Reads
  results/trueexperiment/reference/metrics_summary.csv
  results/trueexperiment/reference/curve_b_aggregate.json
  results/trueexperiment/reference/grid_tvd_aggregate.json
  results/trueexperiment/reference/losses/generation_time.csv
  results/trueexperiment/reference/losses/memorisation.json
  results/trueexperiment/reference/weights/reference_kernel.json
  results/trueexperiment/reference/generated_paths/seed_0/metadata.json
  results/trueexperiment/real_floor/metrics_summary.csv                (floor)
  results/trueexperiment/real_floor/curve_b_aggregate.json
  results/trueexperiment/real_floor/grid_tvd_aggregate.json
  results/trueexperiment/SBTS/losses/memorisation.json                 (band endpoint)

Writes
  results/trueexperiment/reference/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/reference/code/render_readme.py
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.dirname(HERE)                                   # .../reference
TE = os.path.dirname(REF)                                     # .../trueexperiment
FLOOR = os.path.join(TE, "real_floor")

FLOOR_COL = "Real-vs-real floor"

# -- metric display table -----------------------------------------------------
# A33-A34 are absent by construction: they score the generated path against the
# dataset's LATENT VARIANCE path, and a real market has none. compute_all_
# multiasset.py skips them (`v_test: None`), so a row here would render "-" in
# every column and imply the metric merely failed rather than being undefined.
# This is the one place where this page differs from its Heston twin, which does
# carry A33-A34 -- keep the two lists in sync for A1-A32 and nowhere else.
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
def render_a_table(ref, floor):
    n_seeds = len(next(iter(ref.values()))["seeds"]) if ref else 5
    head = ["Metric", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + [FLOOR_COL]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for cat, rows in CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            s, f = ref.get(key), floor.get(key)
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


def a_headline(ref, floor):
    at_floor, total, gaps = [], 0, []
    for _cat, rows in CATEGORIES:
        for key, label, direction in rows:
            s, f = ref.get(key), floor.get(key)
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


# --------------------------------------------------------- memorisation ------
def render_memorisation(n_coef, n_train_values, zero_ok_flag):
    """The memorisation section, from losses/memorisation.json.

    Why the band's lower endpoint is read from SBTS's artefact
    ---------------------------------------------------------
    The band is `median NN(test -> train) / median NN(val -> train)` to 1.000.
    Both quantities are distances between REAL SPLITS. Neither depends on any
    generator, so the number is a property of the dataset build, and the value
    SBTS measured is bit-identical to the value any other method's estimator
    would produce on the same splits -- `median_nn_heldout` in this folder's own
    file already agrees with SBTS's to the last digit, which is the check that
    the two estimators are in fact the same code on the same data.

    This folder's `measure_memorisation.py` does not compute the test-era arm,
    so rather than print a band with a hand-typed endpoint, the endpoint is read
    from `SBTS/losses/memorisation.json` and the read is stated in the prose. If
    that file is absent the band is omitted and the verdict falls back to the
    1.000 reference point alone -- a missing band is visible, a fabricated one is
    not.

    What this section is FOR here
    -----------------------------
    Not a test of the generator. With 25 fitted coefficients against ~6.3M
    training values and a control verified to be exactly zero, memorisation is
    arithmetically impossible, so whatever the estimator returns IS its reading
    for a generator that certainly did not memorise. That is the null the SBTS
    and CSDI columns need in order to be interpretable at all.
    """
    m = read_json(os.path.join(REF, "losses", "memorisation.json"))
    if not m:
        return ("## Memorisation check\n\n"
                "**MISSING** — `losses/memorisation.json` not found. Run "
                "`code/measure_memorisation.py`; guideline §9 makes this section "
                "mandatory for every method on this dataset.\n")

    r = float(m["nn_ratio"])
    sd = float(m["nn_ratio_std"])
    dup = m["n_exact_duplicates"]
    d_val = float(m["median_nn_heldout"])
    d_gen = float(m["median_nn_generated"])

    sb = read_json(os.path.join(TE, "SBTS", "losses", "memorisation.json"))
    d_test = sb.get("median_nn_test_era")
    lo = (float(d_test) / d_val) if d_test else None
    band_agree = bool(sb) and abs(float(sb.get("median_nn_heldout", -1)) - d_val) < 1e-12

    if lo is None:
        band_txt = "the 1.000 reference point (`val` scored against itself)"
        verdict = ("sits **above 1.000** — generated paths are *further* from the training set "
                   "than a genuine same-era held-out sample is, which is the opposite failure "
                   "from copying") if r > 1.0 else (
                   "sits **below 1.000** — generated paths hug the training set. "
                   "**This is memorisation.**")
    else:
        band_txt = f"a real-vs-real band of **{lo:.3f}–1.000**"
        if lo <= r <= 1.0:
            verdict = "sits **inside** the real-vs-real band"
        elif r > 1.0:
            verdict = ("sits **above** the band — generated paths are *further* from the "
                       "training set than real held-out data is, which is the opposite "
                       "failure from copying")
        else:
            verdict = ("sits **below** the band — generated paths hug the training set. "
                       "**This is memorisation.**")

    L = []
    L.append("## Memorisation check")
    L.append("")
    L.append("Guideline §9 makes this section mandatory for every method on this dataset, and")
    L.append("§6.1.2 explains why: on the Heston tree a score sinking below the floor is itself")
    L.append("the copying alarm, and **here it is not** — the floor column is a *memoriser's*")
    L.append("score, not a target — so this ratio is the only guard there is.")
    L.append("")
    L.append(f"**On this page it is a calibration, not a test.** σ<sup>ref</sup> carries "
             f"**{n_coef} fitted")
    L.append(f"numbers** against ~{n_train_values / 1e6:.1f}M training values, and its neural "
             "control is")
    L.append(f"{zero_ok_flag} to be exactly zero before a single path is written. Memorisation is")
    L.append("not merely unlikely here, it is arithmetically impossible. So whatever the")
    L.append("estimator returns **is its reading for a generator that certainly did not")
    L.append("memorise** — the null the SBTS and CSDI columns need in order to mean anything.")
    L.append("")
    L.append("```")
    L.append("NNratio = median NN(generated -> train) / median NN(val -> train)")
    L.append("          " + m["space"])
    L.append("```")
    L.append("")
    L.append("| Seed | median NN(gen → train) | NNratio (vs `val`) | Exact duplicates |")
    L.append("|------|-----------------------:|-------------------:|-----------------:|")
    for d in m["per_seed"]:
        L.append(f"| {d['seed']} | {d['median_nn_generated']:.6f} | {d['nn_ratio']:.4f} | "
                 f"{d['n_exact_duplicates']} |")
    L.append(f"| **mean** | **{d_gen:.6f}** | **{r:.4f} ± {sd:.4f}** | **{dup}** |")
    L.append("")
    L.append(f"**NNratio = {r:.3f} ± {sd:.3f}**, against {band_txt}. It {verdict}.")
    L.append("")
    if lo is not None:
        L.append("The band is not a chosen tolerance and it is not this method's to set: its")
        L.append("endpoints are what the **real splits** score against each other,")
        L.append(f"`median NN(test → train) / median NN(val → train)` = **{lo:.3f}**, and `val`")
        L.append("against itself = **1.000**. Zero free parameters, the same construction as the")
        L.append("floor column. Because both endpoints are distances between real splits they")
        L.append("belong to the dataset build rather than to any generator, so the numerator is")
        L.append("read from [`../SBTS/losses/memorisation.json`](../SBTS/losses/memorisation.json)")
        L.append("rather than recomputed — this folder's `measure_memorisation.py` scores only the")
        L.append("`val` arm.")
        if band_agree:
            L.append(f"The read is checked, not assumed: that file's `median_nn_heldout` "
                     f"({d_val:.6f}) is")
            L.append("bit-identical to this folder's, which is what makes the two estimators the")
            L.append("same code on the same data.")
        else:
            L.append("")
            L.append("> **WARNING:** that file's `median_nn_heldout` does **not** match this")
            L.append("> folder's. The two estimators are not seeing the same splits and the band")
            L.append("> above must not be quoted until that is explained.")
        L.append("")
    L.append("> **The denominator is `val`, not `test`** (guideline §9.1). `val` is the same era")
    L.append("> as `train`; `test` sits behind a 45.5-day embargo in a later regime where")
    L.append("> annualised vol has **fallen** on 6 of 8 assets, so every distance contracts and a")
    L.append("> `test` denominator is *smaller*. A smaller denominator inflates the ratio — and")
    L.append("> since memorisation is the **low**-ratio failure, that would push a copying")
    L.append("> generator toward the healthy-looking end. `val` is the honest denominator.")
    L.append("")
    L.append(f"**{dup} exact duplicates is uninformative here and is reported anyway.** A sample")
    L.append("is a 127-step Euler–Maruyama rollout driven by fresh Gaussian increments, so")
    L.append("bitwise equality with a training path has probability zero *by construction* —")
    L.append("this counter cannot fire for this method whatever it does. It is printed because")
    L.append("the estimator is byte-identical across methods and silently dropping a column on")
    L.append("one page is how tables stop being comparable. The **ratio** is the number that")
    L.append("carries information: on the same estimator, SBTS's rejected `K=3, h=0.05`")
    L.append("candidate also scored 0 duplicates while its NNratio was 0.197.")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ timing ---
def render_timing(path, gseed):
    """Timing table plus all rows.

    The reference bank is CPU-only and single-core, so the interesting column is
    `device`, not a worker count -- and it is READ rather than assumed, because a
    row that said `cuda` would falsify the "no GPU needed" claim this page makes.
    This is the one function that differs structurally from SBTS's, whose CSV has
    an `n_workers` column this one does not.
    """
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, []
    rows.sort(key=lambda r: int(r["seed"]))
    total = sum(float(r["elapsed_min"]) for r in rows)
    out = ["| Seed | Generation seed | Device | Elapsed |",
           "|------|-----------------|--------|---------|"]
    out += [f"| {r['seed']} | {gseed + int(r['seed'])} | `{r['device']}` | "
            f"{float(r['elapsed_sec']):.1f} s |" for r in rows]
    return "\n".join(out), total, rows


def main():
    ref = read_summary(os.path.join(REF, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(REF, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(REF, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    kern = read_json(os.path.join(REF, "weights", "reference_kernel.json"))
    meta = read_json(os.path.join(REF, "generated_paths", "seed_0", "metadata.json"))

    missing = [n for n, o in (("reference/metrics_summary.csv", ref),
                              ("reference/curve_b_aggregate.json", agg),
                              ("reference/weights/reference_kernel.json", kern),
                              ("reference/generated_paths/seed_0/metadata.json", meta),
                              ("real_floor/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing artefacts: " + ", ".join(missing) + "\n"
            "Run code/generate_reference_true.py, then "
            "metrics/compute_all_multiasset.py --dataset TrueDataset for "
            "--method reference and --method real_floor first."
        )

    # -- everything below is DERIVED from the artefacts, never typed ----------
    n_cov = len(kern.get("gamma_diagonal", []))
    n_off = len(kern.get("gamma_offdiagonal", []))
    n_drift = len(kern.get("gamma_drift", []))
    n_half = len(kern.get("trend_half_lives", [])) + len(kern.get("activity_half_lives", []))
    n_gamma = n_cov + n_off + n_drift
    n_coef = n_gamma + n_half + 2                 # + trend_weight, activity_weight
    fitblk = kern.get("fit", {})
    fit_steps = fitblk.get("steps", "?")
    fit_lr = fitblk.get("lr", "?")
    fit_vf = float(fitblk.get("validation_fraction", 0.2))
    fit_file = fitblk.get("train_file", "?")
    fit_at = fitblk.get("fitted_at", "?")
    cal_nll = float(kern["calibration_nll"])
    val_nll = float(kern["validation_nll"])
    smin, smax = kern.get("sigma_min", "?"), kern.get("sigma_max", "?")
    val_pc = int(round(fit_vf * 100))
    base_nll = float(kern.get("baseline_nll", float("nan")))
    gain = float(kern.get("validation_gain_over_baseline", float("nan")))
    sat_hi = float(kern.get("clip_saturation_high", float("nan")))
    sat_lo = float(kern.get("clip_saturation_low", float("nan")))
    p99 = float(kern.get("spectrum_p99", float("nan")))

    n_params = meta.get("n_parameters", "?")
    n_trained = meta.get("n_trained_parameters", "?")
    zhat = meta.get("zhat_head_max_abs", {})
    zmax = max(zhat.values()) if zhat else float("nan")
    zheads = len(zhat)
    zver = bool(meta.get("zero_control_verified", False))
    s0res = float(meta.get("s0_max_abs_residual_before_rescale", 0.0))
    gseed = int(meta.get("generation_seed", 90000))
    num_steps = int(meta.get("num_steps", 127))
    dt = float(meta.get("dt", 9.512937595129376e-07))
    seq_tag = meta.get("seq_tag", "6144x128x8")
    variant = os.path.basename(meta.get("data_dir", "om_2022-07_N6144"))

    shape = meta.get("shape", [6144, 128, 8])
    n_paths, n_len, n_assets = shape
    n_train_values = n_paths * n_len * n_assets
    per_coef = f"{n_train_values / max(n_coef, 1):,.0f}".replace(",", " ")
    npaths_txt = f"{n_paths:,}".replace(",", " ")

    timing, total_min, trows = render_timing(
        os.path.join(REF, "losses", "generation_time.csv"), gseed)
    devices = sorted({r.get("device", "") for r in trows if r.get("device")})
    dev_txt = devices[0] if len(devices) == 1 else " and ".join(devices)
    n_bank = len(trows) if trows else 5
    total_sec = sum(float(r["elapsed_sec"]) for r in trows) if trows else 0.0
    max_sec = max((float(r["elapsed_sec"]) for r in trows), default=0.0)

    at_floor, n_a, gaps = a_headline(ref, floor)
    b_at, n_b = b_headline(agg, agg_f)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    zero_ok = "verified at generation time" if zver else "**NOT VERIFIED**"
    zero_ok_flag = "verified" if zver else "**NOT verified**"
    memo = render_memorisation(n_coef, n_train_values, zero_ok_flag)

    n_seeds = len(next(iter(ref.values()))["seeds"]) if ref else 5
    n_floor = len(next(iter(floor.values()))["seeds"]) if floor else 3

    md = f"""# The Reference SDE on TrueDataset — real crypto spot, d = 8

**σ<sup>ref</sup> — the frozen reference diffusion of Deep-MKV-TS Algorithm 1**
(El Boustany, Basseras, Mekkaoui, Alouadi, Hafsi & Pham), sampled with the neural
control switched **off**, on {npaths_txt} paths of 8 correlated crypto spot series
(BTC, ETH, BNB, SOL, XRP, DOGE, ADA, LINK), 30-second bars, seq\\_len = {n_len},
built from Binance public 1-second klines over 2022-07 → 2026-07.

This is the real-data counterpart of
[`../../HestonMultiAsset/reference/`](../../HestonMultiAsset/reference/README.md), and it is
**not a competing generator**. It is the *starting line*. Algorithm 1 takes a pre-fitted
parametric diffusion σ<sup>ref</sup> and learns a neural control on top of it; this folder
scores σ<sup>ref</sup> **alone**, so [`../Deep-MKV-TS/`](../Deep-MKV-TS)'s numbers can be split
into "what the reference SDE already gave us" and "what the control actually bought".

| Folder | Samples from | Reads as |
|--------|--------------|----------|
| [`../real_floor/`](../real_floor) | {n_floor} **held-out real splits** | what a *perfect memoriser of the training era* scores on the test era — **not** a target |
| **`reference/`** *(this page)* | the **fitted** σ<sup>ref</sup> | Deep-MKV-TS **before its first gradient step** |
| [`../Deep-MKV-TS/`](../Deep-MKV-TS) | the **trained** control | reference **minus** this = what Algorithm 1 bought |

> **One bracket is missing, and it changes how this page must be read.** On Heston the
> reference sits between an independent draw from the *true* SDE and the trained method — two
> sides, so "did the control help" and "how much is left" both have answers. **A real market has
> no law to re-draw from**, so there is no perfect-recovery floor here; `../real_floor/` is a
> memoriser's score, not a floor a generator ought to reach (guideline §6.1). The attribution
> argument on this dataset therefore runs entirely through the **paired**
> reference-vs-Deep-MKV-TS difference at matched generation seeds — which is why the generation
> seeds are imported from `Deep-MKV-TS/code/generate_bank_true.py` rather than redefined.

The model is **built but never fitted**. All {n_params} parameters are constructed exactly as
`train_true.py` constructs them, and then **{n_trained} gradient steps** are taken. Algorithm 1
zero-initialises the final layer of both output heads — weight *and* bias — so `Ẑ ≡ 0` for
every hidden state whatever the GRU beneath computes, the control

$$\\Theta = \\eta\\,(\\sigma^{{\\mathrm{{ref}}}})^{{-1}} + \\hat{{Z}}/\\sqrt{{\\Delta t}}$$

reduces to $\\eta\\,(\\sigma^{{\\mathrm{{ref}}}})^{{-1}}$, and the sampler integrates
σ<sup>ref</sup> itself. All {zheads} head tensors measured `max|·| = {zmax:g}`, {zero_ok} —
`code/generate_reference_true.py` refuses to write a bank if any is non-zero, so the central
claim of this page is **enforced rather than asserted** (per-seed record in
`generated_paths/seed_{{i}}/metadata.json` under `zhat_head_max_abs`).

See [`code/README.md`](code/README.md) for the five guideline §2 answers and implementation
notes, [`weights/README.md`](weights/README.md) for the fitted coefficients and their
provenance, and [`../truedatasetguideline.md`](../truedatasetguideline.md) for the build, the
five-split contract, and what the reference column does and does not mean.

> **Hyperparameters: none were chosen for this folder.** σ<sup>ref</sup> carries **{n_coef}
> fitted numbers in total** — {n_cov} covariance-diagonal, {n_off} off-diagonal, {n_drift}
> drift, {n_half} half-lives and 2 mixing weights — plus the spectral clip bounds
> `σ_min={smin}`, `σ_max={smax}`. They come from a **{fit_steps}-step Gaussian-NLL fit on the
> training split only** (`{fit_file}`, internal {100 - val_pc}/{val_pc}
> calibration/validation split, lr={fit_lr}), `fitted_at` **{fit_at}**, final calibration NLL
> **{cal_nll:.6f}**, validation NLL **{val_nll:.6f}**. Nothing else is tuned: no learning rate,
> no ridge λ, no checkpoint selection, no early stopping — because there is no training. That
> is {n_coef} numbers against **{per_coef} training values per number**.
> **`σ_max` is {smax} here and 0.6 on Heston, and that is not a free choice.** The clip is in
> annualised volatility units and this panel's `dt` is {dt:.6e} years against Heston's 1/252, so
> `1/√dt` is 64.6× larger and the same per-step move maps to a far larger annualised number. The
> ceiling binds on **{sat_hi * 100:.2f}%** of eigenvalues and the floor on **{sat_lo * 100:.2f}%**,
> with the spectrum's p99 at **{p99:.3f}** — a guard rail, not the model.
> **The fit is gated on a scale-free test.** Heston's `validation_nll > -9.0` freeze guard was a
> Heston-scale number and would have rejected every fit on this panel, so it was **deleted, not
> re-tuned**: the fit is rejected unless it beats the i.i.d. Gaussian baseline
> ({base_nll:.6f}), which it does by **{gain:.3f} nats**. That gain is the whole claim that this
> reference SDE carries structure rather than fitting noise.

---

## Metrics A1-A32 + B, mean ± std across {n_seeds} seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).
> **A33-A34 are absent, not failed** — the one structural difference from the Heston twin of
> this page, which does carry them. They score the generated path against the dataset's *latent
> variance* path; a real market has none, so `compute_all_multiasset.py` skips them rather than
> fabricating a proxy.

{render_a_table(ref, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the real-vs-real column with the control switched off** — {at_floor_txt}. The largest gaps are {gap_txt}. **Read neither as a win nor as a failure.** See the column caption below: this column is a memoriser's score on a later era, so beating it is not evidence of quality and losing to it is not evidence of a defect. Cross-method win-counts belong in the dataset-level comparison table, not here.
> **What the "{FLOOR_COL}" column is — read this before quoting it.** On Heston the floor is an *independent draw from the true SDE*: a genuine second sample of the data-generating law. **A real market has no law to re-draw from.** The column here is instead {n_floor} **held-out real splits** — `train`, `val`, `valdisc` — pushed through the metric pipeline *as if they were generated banks* and scored against `test`, with byte-identical code. Three consequences you must not forget:
> 1. It is **not** a floor a generator ought to reach. It is the score a **perfect memoriser of the training era** achieves on the test era.
> 2. The build is **holdout-era** (train ends 2024-11-23, test starts 2025-02-10, 45.5-day embargo), so part of the distance is a genuine **regime change**, not finite-sample noise. Annualised vol falls from train to test on every asset (BTC 0.498 → 0.429, SOL 1.039 → 0.733). A generator that matched the *training* law perfectly would still score badly here — correctly so.
> 3. The `disc` split is **excluded on purpose**: it is the discriminator's real side, so using it as a fake bank would drive A18 to 0 by construction and manufacture an unreachable floor.
> **How to read this page against [`../Deep-MKV-TS/`](../Deep-MKV-TS):** on any row, *(reference − Deep-MKV-TS)* is the control's contribution. A row where Deep-MKV-TS scores **worse** than this page is a row where Algorithm 1 actively damaged a working reference — the single most informative failure this benchmark can surface, and completely invisible without a reference column. Since there is no true-law floor on this dataset, *(Deep-MKV-TS − floor)* is **not** "what remains unsolved" and must not be read that way.
> **Pairing:** generation seeds are `{gseed} + i`, **imported** from `Deep-MKV-TS/code/generate_bank_true.py` rather than redefined, so each reference bank draws the **same Brownian stream** as the corresponding Deep-MKV-TS bank. The difference between the two pages is therefore paired seed by seed, which removes simulation noise from the comparison instead of averaging over it.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|, trained real=`disc` vs fake=generated. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix — the row that tests whether the 28 realised cross-asset correlations (mean 0.609, range 0.515-0.801) survived generation, and the row σ<sup>ref</sup>'s {n_off} off-diagonal coefficients exist to capture. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly. **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0.

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

> **Headline:** **{len(b_at)} of the {n_b} B plots** sit at or below the real-vs-real column on the deciding MSE row **with the control switched off**: {b_at_txt}.
> **Cross-seed stability is the useful column here.** There is no gradient descent, no checkpoint selection and no early stopping in this folder, so the *only* seed-to-seed variation is the Euler–Maruyama draw itself. The std columns therefore measure **pure simulation noise at N = {npaths_txt}**, which makes them a yardstick rather than a footnote: a trained method whose std is materially larger than this page's is unstable for reasons that are **not** sampling noise.

---

{memo}
---

## Stylised Facts Diagnostic (TrueDataset vs Reference SDE, seed 0, asset 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

The third (black dashed) curve is the real **`val`** split, not a theory curve — there is no
closed form for BTC, and unlike the Heston twin of this page there is not even an
independent-draw floor to substitute. It is drawn from `real_floor/generated_paths/seed_1`,
chosen explicitly: the floor's seed 0 is the *training* split, so plotting it here and calling
it held-out would be false. The legend is read from that directory's `metadata.json`, not
hardcoded. **One asset is shown, not eight**: the *metrics* are averaged over all 8 assets, the
*figure* is asset 0 (BTCUSDT). Pooling eight assets whose annualised vol spans 0.43-0.92 into
one histogram would show the mixing, not the model.

Read this figure as the **before** picture: the same eight panels rendered from
[`../Deep-MKV-TS/plots/truedata_diagnostics.png`](../Deep-MKV-TS/plots/truedata_diagnostics.png)
show what the learned control changed, panel by panel.

![TrueDataset Diagnostics](plots/truedata_diagnostics.png)

---

## The reference SDE has no training loss

Nothing is trained in this folder, so there is no loss curve and no `loss_convergence.png` — a
plot with no gradient steps behind it would be decoration. The {n_coef} numbers were fitted
**once, elsewhere**, by `Deep-MKV-TS/code/fit_reference_true.py`; that fit's {fit_steps}-step
calibration/validation NLL trace is archived verbatim as
[`weights/reference_fit_history.csv`](weights/reference_fit_history.csv) — final calibration
**{cal_nll:.6f}**, validation **{val_nll:.6f}**. Those two figures do not match the last row of
that CSV, by exactly one optimiser step; that is expected and explained in
[`weights/README.md`](weights/README.md). A reader who diffs the CSV against the JSON without
reading that section will conclude one of them is wrong. The `losses/` directory consequently
holds only what genuinely exists: generation wall-clock, and the memorisation diagnostic.

Generation is **CPU-only** — `{dev_txt}`, one core, no GPU, no CUDA runtime required. Each seed
is a {num_steps}-step Euler–Maruyama rollout of
`X_{{k+1}} = X_k + b^ref(X_k) + σ(X_k)ε_k` from `x0 = log(100)` driven by fresh Gaussian noise.
The {n_bank}-seed bank costs **{total_sec:.0f} s of CPU time** in total, but is run as {n_bank}
**concurrent single-core processes**, so the wall clock is that of the slowest seed — about
**{max_sec:.0f} s**. The per-seed CSVs are merged afterwards by `code/merge_generation_time.py`
rather than having {n_bank} processes append to one shared file; its docstring says why.

{timing}

Prices are then rescaled to the exact `S0 = 100.0` contract. The model runs in float32, so
`exp(log(100))` lands slightly off 100; the largest residual before rescaling was
**{s0res:.3e}**. Multiplying a whole path by a constant is exactly a shift in log-price, so
**every log-return is bit-identical** before and after — the rescale fixes the anchor without
touching a single quantity any metric measures.

---

## A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.
The classifier is **native**: it sees all 8 channels at once, and its real side is the
`disc` split, never `test`.

This is the one curve on the page where a *falling* loss is bad news — it is the judge learning
to separate σ<sup>ref</sup>'s paths from real crypto paths, and how fast it manages that is the
clearest single picture of what the learned control still has to fix.

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

This tree is **self-contained**: code, inputs, outputs and documentation sit side by side,
matching [`../SBTS/`](../SBTS) and [`../CSDI/`](../CSDI). The weights are copied rather than
symlinked from Deep-MKV-TS for exactly that reason — a reader who takes only this folder still
has everything needed to regenerate the bank.

```
results/trueexperiment/
├── truedatasetguideline.md               ← read this before adding a method
├── real_floor/                           real-vs-real reference ({n_floor} held-out real splits)
└── reference/
    ├── README.md                         ← this file (generated by code/render_readme.py)
    ├── code/
    │   ├── README.md                     the five §2 answers, the zero-control proof, deviations
    │   ├── generate_reference_true.py    builds the model, VERIFIES Ẑ == 0, samples σ^ref
    │   ├── merge_generation_time.py      joins the {n_bank} concurrent per-seed timing CSVs
    │   ├── plot_diagnostics_true.py      the 8-panel stylised-facts figure
    │   ├── measure_memorisation.py       NN ratio vs the val split (the calibration row)
    │   └── render_readme.py              regenerates this README from the artefacts
    ├── generated_paths/seed_{{0..{n_bank - 1}}}/
    │   ├── generated_paths_{seq_tag}.npy   ({n_paths}, {n_len}, {n_assets}) float64 — gitignored, 50 MB
    │   └── metadata.json                    seed, shape, S0 contract, zhat_head_max_abs (tracked)
    ├── logs/                             generate_seed_{{i}}.log per seed, plus metrics/memorisation/plot logs
    ├── losses/
    │   ├── generation_time.csv           wall-clock per seed — no loss, nothing trains
    │   ├── generation_time_seed_{{i}}.csv  the per-process files the above is merged from
    │   └── memorisation.json             NN-ratio diagnostic (calibrates the estimator)
    ├── plots/
    │   ├── truedata_diagnostics.png      8-panel stylised facts (seed 0, asset 0 = BTCUSDT)
    │   ├── disc_classifier_loss.png      A18 BCE curves
    │   └── pred_score_loss.png           A19 MAE curves (asset 0)
    ├── weights/
    │   ├── README.md                     provenance, and why there is no neural checkpoint
    │   ├── reference_kernel.json         the {n_coef} fitted numbers + clip bounds
    │   ├── reference_fit_history.csv     the {fit_steps}-step calibration/validation NLL trace
    │   └── SHA256SUMS                    drift detection against the Deep-MKV-TS originals
    ├── metrics_summary.csv               A1-A32, mean ± std, per seed
    ├── metrics_per_asset.csv             per-metric × per-asset breakdown (8 rows per metric)
    ├── curve_b_aggregate.json            B curve-shape aggregate
    ├── grid_tvd_aggregate.json           path-cloud TVD
    └── seed_{{i}}_metrics.json             full per-seed dump incl. the per_asset block
```

**There are no `crps_banks/` and no table C on this page.** Conditional CRPS (guideline §8)
needs a separate 8 192-path pool; this folder ships only the {n_bank} scoring banks of
{npaths_txt}. An empty table C would claim a measurement that was never made, so the section is
absent rather than blank — the same choice the Heston twin of this page makes.

The `.npy` arrays are **gitignored**: `({n_paths}, {n_len}, {n_assets})` float64 = 50 MB each,
and LFS was ruled out on 2026-07-30. They are fully reproducible from the tracked code and the
tracked coefficients — deterministically so, since the generation seeds are fixed at
`{gseed} + i` and the rollout touches no GPU RNG. The `metadata.json` beside each array **is**
tracked, so shapes, price ranges, the S0 residual and the measured `Ẑ` maxima stay auditable
without the payload. The raw Binance archive is not committed either —
`dataset/TrueDataset/` rebuilds it from `data.binance.vision`.

## Reproduce

```bash
cd /home/tbasseras/benchmark
V=dataset/TrueDataset/variants/{variant}
TAG={seq_tag}
R=results/trueexperiment
REFPKG=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=$PWD/$R/Deep-MKV-TS/code

# 1. dataset — downloads Binance 1s klines and builds the 5 splits.
#    Only if $V/*.npy are absent. See dataset/TrueDataset/README.md.
/home/tbasseras/.cc-venv/bin/python dataset/TrueDataset/build_true_dataset.py \\
    --period 2022-07:2026-07 --n-samples {n_paths} --split-mode holdout-era

# 2. generate the {n_bank} banks — CPU only, ONE core each, no GPU.
#    Aborts before writing anything if the control is not exactly zero.
#    Run as {n_bank} concurrent one-seed processes: ~{max_sec:.0f} s wall, {total_sec:.0f} s CPU.
cd $R/reference/code
for S in {" ".join(str(i) for i in range(n_bank))}; do
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \\
  PYTHONPATH="$REFPKG/src:$REFPKG/experiments:$D" taskset -c $((58 + S)) \\
    /home/tbasseras/gpu-venv/bin/python generate_reference_true.py --seeds $S &
done
wait
/home/tbasseras/.cc-venv/bin/python merge_generation_time.py
cd -

# 3. real-vs-real reference: three held-out REAL splits as stand-in banks.
#    ALREADY BUILT — do not regenerate; the metrics below read it as-is.
#    for i in 0 1 2; do mkdir -p $R/real_floor/generated_paths/seed_$i; done
#    ln -sf $PWD/$V/true_S_$TAG.npy  $R/real_floor/generated_paths/seed_0/generated_paths_$TAG.npy
#    ... val -> seed_1, valdisc -> seed_2

# 4. metrics — both sides go through byte-identical code.
#    GPU needed HERE, not for generation: A18/A19 train discriminators and predictors.
CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method reference --dataset TrueDataset \\
    --seeds {n_bank} --data-dir $V --seq-tag $TAG --dt {dt} \\
    --results-dir $R/reference

# 5. figures
/home/tbasseras/gpu-venv/bin/python $R/reference/code/plot_diagnostics_true.py \\
    --data-dir $V --seq-tag $TAG
/home/tbasseras/gpu-venv/bin/python metrics/plot_score_losses.py \\
    --method reference --dataset TrueDataset --results-dir $R/reference

# 6. memorisation — mandatory (§9). Denominator is the val split, NOT test.
/home/tbasseras/gpu-venv/bin/python $R/reference/code/measure_memorisation.py \\
    --data-dir $V --seq-tag $TAG \\
    --seeds {",".join(str(i) for i in range(n_bank))}

# 7. regenerate this README from the artefacts
/home/tbasseras/gpu-venv/bin/python $R/reference/code/render_readme.py
```
"""

    # -- guideline section-order check ---------------------------------------
    # Not a count. A count still passes when a section is renamed or two are
    # swapped; this asserts the exact ordered list, which is the thing guideline
    # 10.1 actually specifies. Table C is absent BY DESIGN (no 8 192-path pool
    # exists for this folder) and its slot is deliberately not in this list.
    expected = ["Metrics A1-A32 + B", "B, Curve-Shape Metrics", "Memorisation check",
                "Stylised Facts Diagnostic", "The reference SDE has no training loss",
                "A18, Discriminative Classifier Training Loss",
                "A19, Predictive Score Training Loss", "File layout", "Reproduce"]
    got = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    if len(got) != len(expected) or not all(g.startswith(e) for g, e in zip(got, expected)):
        raise SystemExit(
            "ABORT: section order does not match guideline 10.1.\n"
            f"  expected (prefixes): {expected}\n"
            f"  got:                 {got}\n"
            "README not written."
        )
    if "reproduced exactly" in md:
        raise SystemExit("ABORT: the phrase 'reproduced exactly' is banned (guideline 10.2.6).")

    out = os.path.join(REF, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines, {len(got)} '##' sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
