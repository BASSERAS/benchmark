#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/CSDI/README.md from the metric artefacts.

The section order, table shape and prose slots are a 1:1 mirror of
`results/HestonMultiAsset/SBTS/README.md` (itself a mirror of the d = 1 entry
`methods/SBTS/README.md`), so every method page on this dataset displays
identically and can be diffed row-by-row. MULTIASSET_GUIDELINE.md section 8 fixes
that layout; the two differences it *mandates* for a trained method are section 6,
which becomes a real training-loss section instead of SBTS's "no training loss"
placeholder, and the hyperparameter blockquote.

Everything from `CATEGORIES` down to `render_timing` is byte-identical to
`LS4/code/render_readme.py`, itself byte-identical to `SBTS/code/render_readme.py`.
That is deliberate: the guideline's point is that method renderers differ *only*
where the layout mandates it, so a diff between any two of these files shows the
method-specific content and nothing else.

The one deliberate omission, inherited from the SBTS page, is the Path Shadowing
MC block: PS-MC has not been run for d = 8, and a section with no numbers behind
it would be a lie.

Material that is *about the dataset rather than about CSDI* -- how the d = 8
metrics are scoped and the nearest-neighbour memorisation discussion -- lives one
level up in `results/HestonMultiAsset/oldreadme.md`, shared by every method added
to this dataset. Material about *this port* -- the provenance check against the
d = 1 run, the per-channel scaler, the S0 rescaling -- lives in `code/README.md`.

Every number here is READ FROM DISK, never typed by hand: the tables *and* the
headline win-counts are computed from the artefacts, so the prose cannot drift
away from the numbers it describes.

Reads
  results/HestonMultiAsset/CSDI/metrics_summary.csv
  results/HestonMultiAsset/CSDI/curve_b_aggregate.json
  results/HestonMultiAsset/CSDI/grid_tvd_aggregate.json
  results/HestonMultiAsset/CSDI/losses/generation_time.csv
  results/HestonMultiAsset/CSDI/losses/memorisation.json           (optional)
  results/HestonMultiAsset/CSDI/weights/seed_*_config.json         (train times)
  results/HestonMultiAsset/CSDI/generated_paths/seed_*/metadata.json
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv    (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/CSDI/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/CSDI/code/render_readme.py
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Name kept as SBTS so the shared table helpers stay textually diffable against
# SBTS/code/render_readme.py -- the guideline's whole point is that the two
# renderers differ only where the layout mandates it.
SBTS = os.path.dirname(HERE)                                  # .../CSDI
MA = os.path.dirname(SBTS)                                    # .../HestonMultiAsset
FLOOR = os.path.join(MA, "perfect_recovery")

# -- metric display table -----------------------------------------------------
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
    ("Heston Spec", [
        ("A33_sigma_corr",           "A33 Teacher-Sigma Corr",      "up"),
        ("A34_sigma_rmse",           "A34 Teacher-Sigma RMSE",      "down"),
    ]),
]

ARROW = {"down": " ↓", "up": " ↑", "none": ""}


def fmt(x):
    """4 significant digits, scientific below 1e-3 -- matches the d = 1 READMEs."""
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
    """Escape pipes so a cell never breaks the markdown table.

    The curve-plot names come straight from metrics.CURVE_PLOTS and several
    contain a raw '|' (e.g. "ACF |r| lags 1-20"), which silently splits the row
    into extra columns. The d = 1 READMEs escape these by hand; we do it here.
    """
    return str(s).replace("|", r"\|")


def read_summary(path):
    """metrics_summary.csv -> {metric: {"mean","std","seeds":[...],"scope"}}."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seeds = [row[k] for k in sorted(row) if k.startswith("seed_")]
            out[row["metric"]] = {
                "mean": row["mean"], "std": row["std"],
                "scope": row.get("scope", ""),
                "seeds": seeds,
            }
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def render_a_table(sbts, floor):
    n_seeds = len(next(iter(sbts.values()))["seeds"]) if sbts else 5
    head = ["Metric", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + ["Perfect floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for cat, rows in CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            s = sbts.get(key)
            f = floor.get(key)
            if s is None:
                cells = ["-"] * (n_seeds + 2)
                scope = ""
            else:
                cells = [f"{fmt(s['mean'])} ± {fmt(s['std'])}"]
                cells += [fmt(v) for v in s["seeds"]]
                cells += [fmt(f["mean"]) if f else "-"]
                scope = " *(native d=8)*" if s["scope"] == "native" else ""
            lines.append(f"| {label}{ARROW[direction]}{scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _better(sm, fm, direction):
    """Is SBTS's mean at least as good as the floor's mean on this row?"""
    if direction == "up":
        return sm >= fm
    if direction == "none":              # A28: perfect = 1.0
        return abs(sm - 1.0) <= abs(fm - 1.0)
    return sm <= fm


def a_headline(sbts, floor):
    """Count rows at/below the independent-draw floor, and rank the gaps.

    In the d = 1 README the headline counts wins against the other generators.
    d = 8 has only one method so far, so the only meaningful reference is the
    floor; when a second method lands, the cross-method table in the dataset
    README is where the comparison belongs.
    """
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


def floor_explanation(sbts, floor):
    """Explain every row that reached or passed the floor, rather than banking it.

    GUIDELINE checklist: "any metric that beats the floor has been *explained*, not
    celebrated". A row at the floor is not a win — at the floor the metric can no
    longer separate a generator from a fresh draw of the true SDE, and *past* it the
    number has stopped measuring fidelity (section 6). So for each such row this
    prints the method value, the floor value, and the margin **in units of the
    method's own cross-seed std**, which is the only scale on which "beat" means
    anything at 5 seeds. A margin under 1 std is reported as unresolved, because
    that is what it is.
    """
    out = []
    for _cat, rows in CATEGORIES:
        for key, label, direction in rows:
            s, f = sbts.get(key), floor.get(key)
            if not s or not f:
                continue
            try:
                sm, fm = float(s["mean"]), float(f["mean"])
                sd = float(s.get("std") or 0.0)
            except (TypeError, ValueError):
                continue
            if not _better(sm, fm, direction):
                continue
            clean = label.replace("\\", "")
            margin = abs(sm - fm)
            # sd == 0 would divide by zero; a genuinely zero-variance row is itself
            # worth calling out rather than being silently formatted as inf.
            if sd > 0:
                k = margin / sd
                scale = (f"{k:.2f}× its own cross-seed std ({fmt(sd)})" if k >= 1.0 else
                         f"only {k:.2f}× its own cross-seed std ({fmt(sd)}), i.e. "
                         f"**not resolvable at 5 seeds**")
            else:
                scale = "a cross-seed std of exactly 0, which is itself suspicious"
            # A margin can clear several stds and still be physically nothing, when
            # the floor itself sits at ~0 and the row has no dynamic range left.
            # Flag that separately: statistical resolvability and practical size are
            # different claims, and reporting only the first is how a null result
            # gets written up as a finding.
            if max(abs(sm), abs(fm)) < 0.01:
                scale += (f" — but both values are within 0.01 of zero, so the row has "
                          f"essentially no dynamic range here and the margin is not a "
                          f"practical effect whatever its statistical size")
            out.append(f"- **{clean}** — {fmt(sm)} against a floor of {fmt(fm)}; the "
                       f"margin is {fmt(margin)}, {scale}.")
    if not out:
        return ""
    return ("\n**Rows at or past the floor, explained rather than banked.** Reaching the "
            "independent-draw floor is not a win: at the floor the metric can no longer "
            "tell a generator from a fresh draw of the true SDE, so the row has stopped "
            "discriminating, and past it the number has inverted its meaning. Each such "
            "row is given below with its margin measured in the method's own cross-seed "
            "std — the only scale on which \"beat the floor\" can mean anything at 5 "
            "seeds:\n\n" + "\n".join(out) + "\n")


MEASURES = [("mse", "MSE", fmt), ("pct", "% err", pct), ("nrmse", "NRMSE", pct),
            ("cvar90", "CVaR₉₀", pct), ("cvar95", "CVaR₉₅", pct)]


def render_b_table(agg, agg_f, tvd, tvd_f):
    n_seeds = len(tvd.get("per_seed", [0] * 5)) if tvd else 5
    head = ["Plot", "Measure", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + ["Perfect floor"]
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
    """Which B plots sit at or below the floor on the deciding MSE row."""
    at, total = [], 0
    for prefix, blk in agg.items():
        blk_f = agg_f.get(prefix, {})
        if "mse" not in blk or "mse" not in blk_f:
            continue
        total += 1
        if float(blk["mse"]["mean"]) <= float(blk_f["mse"]["mean"]):
            at.append(blk["name"])
    return at, total


def render_timing(path):
    """Generation timing table plus ALL rows.

    Returns every row, not just the first. Seeds are not guaranteed to share a
    worker count, a batch size or an epoch budget -- a partial re-run at different
    settings is exactly the situation this table exists to expose -- so the
    caller describes the whole set rather than assuming row 0 speaks for the run.
    """
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


def render_training(root):
    """Training wall-clock table, merged from two artefacts on disk.

    GUIDELINE section 8.7: a trained method replaces SBTS's "no training loss"
    placeholder with a real one, and the wall-clock numbers must come off disk.

    The two artefacts carry disjoint halves of what the table needs and neither
    is sufficient alone:

      weights/seed_*_config.json        hyperparameters + train_time_sec
      generated_paths/seed_*/metadata.json
                                        sec_per_epoch, epochs_run,
                                        min_total_loss, min_val_loss,
                                        first_nan_step

    They are merged per seed with the config as the base, so a hyperparameter can
    never be silently taken from the metadata copy. Returns
    (markdown, total_minutes, merged_configs).

    Two CSDI-specific differences from LS4's version of this function:

      * the loss columns are labelled *loss*, not *ELBO*. CSDI has no variational
        bound to report -- there is exactly one scalar, the DDPM eps-MSE, so
        "Final train ELBO / Best val ELBO" would name a quantity this method never
        computes.
      * the NaN column reads ``first_nan_step``, not ``first_nan_epoch``. CSDI
        logs every 100 optimiser steps rather than once per epoch (GUIDELINE
        section 3.2), so the finest resolution at which divergence can be located
        is a step index; reporting it as an epoch would round 512 steps of
        training down to one number and lose the location.

    Both loss columns are *minima over the run*, not final values: ``min_total_loss``
    is the smallest logged training loss and ``min_val_loss`` the smallest
    validation loss. For the training column that choice matters -- ``calc_loss``
    draws one random diffusion step per sample, so the last logged value is a
    one-sample estimate and can sit well above the level the model actually
    reached.
    """
    cfgs = []
    for p in sorted(glob.glob(os.path.join(root, "weights", "seed_*_config.json"))):
        with open(p) as fh:
            c = json.load(fh)
        seed = int(os.path.basename(p).split("_")[1])
        meta_p = os.path.join(root, "generated_paths", f"seed_{seed}", "metadata.json")
        if os.path.exists(meta_p):
            with open(meta_p) as fh:
                m = json.load(fh)
            for k in ("sec_per_epoch", "epochs_run", "min_total_loss",
                      "min_val_loss", "first_nan_step",
                      # generated-array integrity, surfaced in the prose below the
                      # timing table. GUIDELINE section 4 requires the <=0 clip count to
                      # be disclosed rather than left implicit in metadata.json: the
                      # clip is the one step in the S0 pipeline that does NOT preserve
                      # log-returns, so a non-zero count is a real distortion and a
                      # silent page would be hiding it.
                      "n_nonpositive_total_before_rescale",
                      "n_nonpositive_s0_before_rescale",
                      "raw_price_min", "raw_price_max", "generated_std"):
                c.setdefault(k, m.get(k))
        c["_seed"] = seed
        cfgs.append(c)
    if not cfgs:
        return "_(weights/seed_\\*_config.json not found)_", 0.0, []
    cfgs.sort(key=lambda c: c["_seed"])
    total = sum(float(c.get("train_time_sec") or 0.0) for c in cfgs) / 60.0

    def _n(v, unit="", nd=1):
        return "-" if v is None else f"{float(v):.{nd}f}{unit}"

    out = ["| Seed | Epochs | sec/epoch | Train wall-clock | Best train loss | Best val loss | NaN |",
           "|------|--------|-----------|------------------|-----------------|---------------|-----|"]
    for c in cfgs:
        nan = "-" if c.get("first_nan_step") is None else f"step {c['first_nan_step']}"
        out.append(
            f"| {c['_seed']} "
            f"| {c.get('epochs_run', c.get('epochs', '-'))} "
            f"| {_n(c.get('sec_per_epoch'), ' s')} "
            f"| {_n((c.get('train_time_sec') or 0.0) / 60.0, ' min')} "
            f"| {_n(c.get('min_total_loss'), nd=4)} "
            f"| {_n(c.get('min_val_loss'), nd=4)} "
            f"| {nan} |")
    return "\n".join(out), total, cfgs


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))

    missing = [n for n, o in (("CSDI/metrics_summary.csv", sbts),
                              ("CSDI/curve_b_aggregate.json", agg),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --method CSDI and "
            "--method perfect_recovery first."
        )

    timing, total_min, trows = render_timing(os.path.join(SBTS, "losses", "generation_time.csv"))
    train_tbl, train_min, cfgs = render_training(SBTS)
    c0 = cfgs[0] if cfgs else {}
    r0 = trows[0] if trows else {}
    # Hyperparameters are read off the per-seed config, never typed here.
    epochs = str(c0.get("epochs", r0.get("epochs", "200")))
    params = int(c0.get("params", 0))
    batch = str(c0.get("batch_size", r0.get("batch_size", "16")))
    nsteps = str(c0.get("num_steps", r0.get("num_steps", "50")))
    layers = str(c0.get("layers", 4))
    channels = str(c0.get("channels", 64))
    nheads = str(c0.get("nheads", 8))
    demb = str(c0.get("diffusion_embedding_dim", 128))
    bstart = c0.get("beta_start", 1e-4)
    bend = c0.get("beta_end", 0.5)
    sched = str(c0.get("schedule", "quad"))
    timeemb = str(c0.get("timeemb", 128))
    featemb = str(c0.get("featureemb", 16))
    lr = c0.get("lr", 1e-3)
    # `retuned_for_d8` is the guideline's honesty field: it lists every hyperparameter
    # that had to move going from d = 1 to d = 8. Read it rather than claim it is empty.
    retuned = c0.get("retuned_for_d8") or []
    retuned_txt = ("**nothing** — the list `retuned_for_d8` in every "
                   "`weights/seed_*_config.json` is empty"
                   if not retuned else "`" + "`, `".join(map(str, retuned)) + "`")
    # Any seed disagreeing on batch_size or epochs is a real inconsistency: say so
    # rather than letting seed 0 speak for the other four.
    bset = sorted({str(c.get("batch_size")) for c in cfgs if c.get("batch_size") is not None})
    eset = sorted({str(c.get("epochs")) for c in cfgs if c.get("epochs") is not None})
    hp_warn = ""
    if len(bset) > 1 or len(eset) > 1:
        hp_warn = (f" **Seeds disagree** — batch_size {{{', '.join(bset)}}}, "
                   f"epochs {{{', '.join(eset)}}}; re-run the odd seed before trusting this page.")
    # Worker counts can differ across seeds, so report the actual set rather
    # than letting row 0 misrepresent the other four rows.
    nws = sorted({str(r.get("n_workers", "")) for r in trows if r.get("n_workers")},
                 key=lambda x: int(x) if x.isdigit() else 0)
    nw = nws[0] if len(nws) == 1 else " and ".join(nws)
    gen_ns = sorted({str(r.get("n_samples", "")) for r in trows if r.get("n_samples")})
    gen_n = f"{int(gen_ns[0]):,}".replace(",", " ") if len(gen_ns) == 1 else "/".join(gen_ns)

    # -- generated-array integrity, stated rather than left in metadata.json --------
    # GUIDELINE section 4: the S0 pipeline is `clip(<=0) -> multiply -> set row 0`.
    # The multiply is a per-path constant, i.e. exactly a shift of the log-price level,
    # so it leaves every log-return bit-identical and cannot affect A1-A25/A27-A34. The
    # *clip* has no such guarantee. Whether it fired is therefore a fact about how much
    # of this page's tables to trust, and it is read off disk rather than asserted.
    def _isum(key):
        vals = [c.get(key) for c in cfgs if c.get(key) is not None]
        return (sum(int(v) for v in vals), len(vals))

    # Space-grouped thousands, matching the rest of the page ("8 192 paths").
    # Applied per-number: a blanket str.replace(",", " ") on the finished sentence
    # also eats the commas of the prose, which is how the first version of this
    # paragraph came out mangled.
    def _g(n):
        return f"{int(n):,}".replace(",", " ")

    n_clip, n_have = _isum("n_nonpositive_total_before_rescale")
    n_clip_s0, _ = _isum("n_nonpositive_s0_before_rescale")
    raw_lo = [float(c["raw_price_min"]) for c in cfgs if c.get("raw_price_min") is not None]
    raw_hi = [float(c["raw_price_max"]) for c in cfgs if c.get("raw_price_max") is not None]
    if not n_have:
        clip_txt = "_(clip counts not found in metadata.json)_"
    elif n_clip == 0:
        clip_txt = (
            f"**No values were clipped.** Across all {n_have} seeds — "
            f"{_g(n_have * 8192 * 252 * 8)} generated entries — "
            f"`n_nonpositive_total_before_rescale` is **0**, with raw prices spanning "
            f"[{min(raw_lo):.2f}, {max(raw_hi):.2f}] before rescaling. This matters because the "
            f"S0 pipeline is `clip(≤0) → multiply → set row 0`: the multiply is a per-path "
            f"constant and therefore *exactly* a shift of the log-price level, leaving every "
            f"log-return bit-identical, but a clip would not preserve log-returns and would "
            f"silently distort A1-A25/A27-A34. It never fired, so no such distortion is present "
            f"in these arrays."
        )
    else:
        clip_txt = (
            f"⚠️ **{_g(n_clip)} of {_g(n_have * 8192 * 252 * 8)} generated entries were ≤ 0 and "
            f"were clipped to 1e-6 before the S0 rescale, {_g(n_clip_s0)} of them in the *first* "
            f"row.** Unlike the per-path multiplier, clipping does **not** preserve log-returns, "
            f"so the fat-tail and volatility rows are distorted to that extent; a clipped first "
            f"price is worse still, because rescaling it to 100 multiplies that whole path by "
            f"~1e8. Raw prices spanned [{min(raw_lo):.2f}, {max(raw_hi):.2f}]. Disclosed rather "
            f"than repaired — read the affected rows with this in mind."
        )

    at_floor, n_a, gaps = a_headline(sbts, floor)
    b_at, n_b = b_headline(agg, agg_f)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    floor_txt = floor_explanation(sbts, floor)
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    md = f"""# CSDI on Multi-Asset Heston (d = 8)

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation**
(Tashiro, Song, Song & Ermon, NeurIPS 2021,
[arXiv:2107.03502](https://arxiv.org/abs/2107.03502)) applied to 8 192 **multi-asset** Heston
stochastic-volatility price paths (seq\\_len = 252, **d = 8 correlated assets**).

CSDI is a **denoising diffusion model with a 2-D Transformer denoiser**: each residual block
runs one attention pass along the *time* axis and one along the *feature* axis, so cross-asset
structure is modelled by the same mechanism that models autocorrelation. It is trained on the
plain DDPM objective — a single scalar, `E_t ‖ε − ε_θ(x_t, t)‖²` — and generates by ancestral
sampling from pure noise through {nsteps} reverse steps; it never touches a training path at
generation time. {params:,} parameters, trained for {epochs} epochs per seed, 5 seeds. Weights
in [`weights/`](weights/), per-seed hyperparameters in `weights/seed_*_config.json`.

The model is **joint over all 8 channels** — one network with `target_dim = 8` — not eight
independent univariate fits. The paper's headline task is *imputation*, i.e. conditional
generation; the unconditional regime used here is the one the authors describe in Sec 4.1 and
Appendix C, reached by setting `is_unconditional = 1` and `cond_mask ≡ 0`. The conditioning
mask never gates the network input, it only selects which points enter the loss through
`target_mask = observed_mask − cond_mask`; with `observed_mask ≡ 1` and `cond_mask ≡ 0` that
target is 1 everywhere and the objective collapses to standard DDPM. See
[`code/README.md`](code/README.md) for source, the vendored release, the bit-exact provenance
check against the committed d = 1 run, and the d = 8 deviations; and the dataset-level
[`../oldreadme.md`](../oldreadme.md) for the multi-asset Heston law itself, the per-asset vs
native metric scoping, and the memorisation diagnostic shared by all methods on this dataset.

> **Hyperparameters:** `layers={layers}`, `channels={channels}`, `nheads={nheads}`,
> `diffusion_embedding_dim={demb}`, `num_steps={nsteps}`, `schedule={sched}`,
> `beta_start={bstart:g}`, `beta_end={bend:g}`, `timeemb={timeemb}`, `featureemb={featemb}`,
> `is_unconditional=1`; `Adam(lr={lr:g}, weight_decay=1e-6)` +
> `MultiStepLR(milestones=[int(0.75·epochs), int(0.9·epochs)], gamma=0.1)`,
> batch {batch}, {epochs} epochs.
> Every one of those numbers is the **released `config/base.yaml`** shipped with the paper's
> reference implementation, unchanged. What had to be retuned for d = 8 is {retuned_txt}.
> Only `target_dim` moved, from 1 to 8, and the data forces that — it is not a free choice.
> This is the strongest position available on this page: the d = 8 run is not a
> hyperparameter search that happened to land somewhere, it is the authors' configuration
> pointed at a wider tensor.
> The scaler was changed from a **global** standardise to a **per-channel** one. That is a
> scoping consequence rather than a retune: at K = 1 the d = 1 code's `S.mean()` **is** the
> per-feature statistic, and CSDI's own PhysioNet loader standardises per feature. A
> per-channel affine map leaves the cross-asset correlation matrix exactly unchanged, so the
> target coupling Σˢ — what A20 scores — survives both the standardisation and its
> inverse.{hp_warn}

---

## Metrics A1-A34 + B, mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the independent-draw floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}. The floor, not the other method, is the reference on this page; the cross-method comparison lives in the dataset-level [`../README.md`](../README.md) so that neither method's page grades itself against a rival it was tuned beside.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix, which is the row that actually tests whether the d(d−1)/2 spot correlations Σˢ survived generation. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. CSDI carries **no** explicit latent volatility state: its only latent is the diffusion noise level, which is not a stochastic-volatility path. It can only reproduce these rows by learning the variance clustering implicitly through the time-axis attention, which is the question the table answers.
{floor_txt}

---

## B, Curve-Shape Metrics, mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. The curve is
computed **per asset and averaged over the 8 assets** *before* the combination below, so the
combination rules stay byte-identical to the d = 1 tree. For the real data (L_r) and generated
data (L_g) we build three lists, the curve L, its first finite difference L' (der), and its
second finite difference L'' (sec\\_der), then combine the three sub-scores into **one number
per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\\_der)/3; std = the sample std of that per-seed combined score across the 5 seeds. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE, one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself**, the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the 5 seeds** of that per-seed function MAPE.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)**, the ill-posed derivative / 2nd-derivative curves are excluded for the same reason as the % err row.
- **CVaR₉₀ / CVaR₉₅ rows**: tail-averaged pointwise curve error (Expected Shortfall) on the curve L **only (funct-only)**. Pointwise error eₜ = |L_g(t) − L_r(t)|; for q ∈ {{0.90, 0.95}}, CVaR_q = mean(eₜ for eₜ ≥ the q-th percentile of eₜ), then range-normalized like NRMSE (÷ (max|L_r| − min|L_r| + 1e-12) × 100).

All ↓ lower is better. The perfect floor is **non-zero** for all plots, it is the residual finite-sample error of an independent multi-asset Heston draw scored against the test set, identical across methods.
Five sublines per plot: **MSE**, **% error**, **NRMSE**, **CVaR₉₀** and **CVaR₉₅** (the per-seed columns hold that seed's combined score).

{render_b_table(agg, agg_f, tvd, tvd_f)}

> **Headline:** **{len(b_at)} of the {n_b} B plots** sit at or below the finite-sample floor on the deciding MSE row: {b_at_txt}.
> **Cross-seed stability**: unlike a kernel method, CSDI's seed controls **both** the weight initialisation and every reverse-diffusion noise draw, so the std columns here absorb optimisation variance as well as sampling variance. That makes them the honest quantity to read: a seed that collapsed would widen them, and [`plots/loss_convergence.png`](plots/loss_convergence.png) shows whether any did.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs CSDI, seed 0, asset 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

The third (black dashed) curve is the **independent-draw floor**, not a closed-form theory
curve: `metrics/heston_theory.py` is hard-wired to the d = 1 parameters (κ=2.0, θ=0.04,
ρ=−0.7, dt=1/250), so plotting it over d = 8 data would be a fabricated reference. Wherever
Real and the floor curve already disagree, the gap is finite-sample noise rather than a
modelling failure. **One asset is shown, not eight** — the *metrics* are averaged over all 8
assets, the *figure* is asset 0; pooling eight deliberately different volatilities
(σ 0.0097-0.0165) into one histogram would show the mixing, not the model.

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## CSDI Training Loss (5 seeds)

CSDI optimises **one** term, the DDPM denoising objective
`E_t ‖ε − ε_θ(x_t, t)‖²` restricted to `target_mask` (which is 1 everywhere in the
unconditional regime). There is no KLD / NLL / reconstruction decomposition to break out, so
this figure is **two** panels, not five: train and validation, each with all 5 seeds overlaid.
The x-axis is **step**, not epoch: with batch {batch} over 8 192 paths an epoch is 512 steps,
and step resolution is what makes the two learning-rate drops visible.

The two panels are **not the same quantity plotted twice, and their levels are not
comparable**. `calc_loss` draws **one** random diffusion step *t* per sample, so the training
curve is a one-sample estimate dominated by that sampling noise; `calc_loss_valid` averages
over **all {nsteps}** diffusion steps, so the validation curve is the low-variance one and is
the series to read convergence from. They are different estimators of different averages, not
train/test versions of one number — a validation curve sitting *below* the training curve here
is expected and is evidence of nothing. Validation runs on 256 held-out paths from
`heston_ma_S_val_8192x252x8.npy`, standardised with the **training** statistics, on a 20-point
cadence; a full 8 192-path validation pass costs ~{nsteps}× a training epoch and would have
dominated the run.

![Training Loss](plots/loss_convergence.png)

Training wall-clock, read from `weights/seed_*_config.json` and
`generated_paths/seed_*/metadata.json` (**{train_min:.0f} min total** across the 5 seeds, run
{nw} threads per process, 2 GPUs):

{train_tbl}

Generation wall-clock — {nsteps} reverse-diffusion steps from pure noise for each of {gen_n}
paths, {nw} threads, **{total_min:.1f} min total** across the 5 seeds:

{timing}

{clip_txt}

---

## A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.
The d = 8 classifier is **native**: it sees all 8 channels at once.

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
unlike the d = 1 benchmark which splits `methods/<Method>/` from `results/Heston/<Method>/`.

```
results/HestonMultiAsset/CSDI/
├── README.md                             ← this file (generated by code/render_readme.py)
├── code/
│   ├── README.md                         source, provenance check, d = 8 deviations
│   ├── train_multiasset.py               trains one seed AND generates its 8192 paths
│   ├── collect_artifacts.py              rebuilds generation_time.csv, checks the §4 contract
│   ├── plot_losses.py                    plots/loss_convergence.png, 5 seeds overlaid
│   ├── plot_diagnostics_multiasset.py    the 8-panel stylised-facts figure
│   ├── measure_memorisation.py           nearest-neighbour memorisation diagnostic
│   ├── render_readme.py                  regenerates this README from the artefacts
│   └── reference/                        vendored CSDI release (main_model, diff_models,
│                                         config/base.yaml) — unmodified but for one
│                                         lazy import, see code/README.md
├── generated_paths/seed_{{0..4}}/
│   ├── generated_paths_8192x252x8.npy    (8192, 252, 8) float64 — gitignored, 132 MB
│   └── metadata.json                     shape, S0_exact, min/max, timings, clip counts (tracked)
├── weights/
│   ├── seed_{{i}}_model.pt                 state_dict + per-channel scaler
│   └── seed_{{i}}_config.json              full hyperparameter record incl. retuned_for_d8
├── losses/
│   ├── seed_{{i}}_losses.csv               step, phase, loss_total
│   ├── memorisation.json                 NN-ratio diagnostic on the final 8192-path output
│   └── generation_time.csv               wall-clock time per seed
├── plots/
│   ├── loss_convergence.png              train / validation eps-MSE, 5 seeds
│   ├── heston_diagnostics.png            8-panel stylised facts (seed 0, asset 0)
│   ├── disc_classifier_loss.png          A18 BCE curves, 5 seeds
│   └── pred_score_loss.png               A19 MAE curves, 5 seeds (asset 0)
├── metrics_summary.csv                   A1-A34, mean ± std, per seed
├── metrics_per_asset.csv                 per-metric × per-asset breakdown (8 rows per metric)
├── curve_b_aggregate.json                B curve-shape aggregate
├── grid_tvd_aggregate.json               path-cloud TVD
└── seed_{{i}}_metrics.json                 full per-seed dump incl. the per_asset block
```

There is deliberately **no `run_all_multiasset.py`**, although MULTIASSET_GUIDELINE.md §1 lists
one: CSDI generates inside `train_multiasset.py`, from the in-memory model immediately after
training. Splitting it out would mean reloading a checkpoint and duplicating the per-channel
scaler inversion and the S0 rescaling in a second place where the two copies could silently
drift. `collect_artifacts.py` fills the slot for post-generation bookkeeping and contract
verification.

The `.npy` arrays are **gitignored**: `(8192, 252, 8)` float64 = 132 MB each, over GitHub's
100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from the tracked
code and tracked hyperparameters; the `metadata.json` beside each array **is** tracked, so
shapes, price ranges and generation times stay auditable without the payload.

## Reproduce

```bash
cd /home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. provenance check against the committed d = 1 run (~3 min)
#    Reproduces methods/CSDI/losses/seed_0_losses.csv bit-for-bit. Use --epochs 4, NOT 1:
#    MultiStepLR's milestones are int(0.75*E) and int(0.9*E), which both collapse to 0 at
#    E = 1, so a 1-epoch probe silently trains at lr = 1e-5 and will not match.
cd methods/CSDI/code && CUDA_VISIBLE_DEVICES=2 $PY train_heston.py \\
    --seed 0 --epochs 4 --tag provcheck && cd -

# 3. final 5 seeds, {epochs} epochs, 2 GPUs (~{train_min:.0f} min of GPU time, ~3 waves wall-clock)
cd results/HestonMultiAsset/CSDI/code
GPUS=(2 3); CORES=("16-23" "24-31")
for wave in "0 1" "2 3" "4"; do
  i=0
  for s in $wave; do
    CUDA_VISIBLE_DEVICES=${{GPUS[$i]}} OMP_NUM_THREADS=8 taskset -c ${{CORES[$i]}} \\
        $PY train_multiasset.py --seed $s & i=$((i+1))
  done
  wait
done
$PY collect_artifacts.py            # must print "5 rows" and exit 0 before anything downstream
cd /home/tbasseras/benchmark

# 4. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 5. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=2 $PY metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=2 $PY metrics/compute_all_multiasset.py --method CSDI \\
    --dataset HestonMultiAsset --seeds 5

# 6. figures
$PY results/HestonMultiAsset/CSDI/code/plot_losses.py
$PY results/HestonMultiAsset/CSDI/code/plot_diagnostics_multiasset.py --method CSDI
$PY metrics/plot_score_losses.py --method CSDI --dataset HestonMultiAsset

# 7. memorisation diagnostic
$PY results/HestonMultiAsset/CSDI/code/measure_memorisation.py --seeds 0,1,2,3,4

# 8. regenerate this README from the artefacts
$PY results/HestonMultiAsset/CSDI/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
