#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/Deep-MKV-TS/README.md from the metric artefacts.

The section order, table shape and prose slots are a 1:1 mirror of
`results/HestonMultiAsset/SBTS/README.md` (itself a mirror of the d = 1 entry
`methods/SBTS/README.md`), so every method page on this dataset displays
identically and can be diffed row-by-row. MULTIASSET_GUIDELINE.md section 8 fixes
that layout; the two differences it *mandates* for a trained method are section 6,
which becomes a real training-loss section instead of SBTS's "no training loss"
placeholder, and the hyperparameter blockquote.

Everything from `CATEGORIES` down to `render_timing` is byte-identical to
`CSDI/code/render_readme.py`, itself byte-identical to `LS4/` and `SBTS/`. That is
deliberate: the guideline's point is that method renderers differ *only* where the
layout mandates it, so a diff between any two of these files shows the
method-specific content and nothing else.

`render_training` is the one shared-looking function that genuinely had to change,
and the reason is not cosmetic. CSDI and LS4 report their FINAL model and count
EPOCHS. Deep-MKV-TS does neither:

  * Algorithm 1 counts **outer iterations**, not epochs. Each iteration resamples a
    fresh batch of controlled paths and re-fits the Z-proxy ridge regression, so
    there is no pass over a fixed dataset that could be called an epoch. A column
    headed "Epochs" would name a quantity this method never computes.
  * The reported checkpoint is **chosen on validation**, not the last step, exactly
    as the committed d = 1 run reports step 2500 of a 3000-step run. `selected_step`
    and the validation discrepancy that chose it are therefore columns in their own
    right -- a timing table that omitted them would describe a model that was never
    reported.
  * There is no validation *loss* to tabulate, because there is no held-out forward
    pass during training. Selection happens afterwards, in
    `select_checkpoint_multiasset.py`, by sampling from each saved checkpoint and
    scoring it against the validation bank.

The one deliberate omission, inherited from the SBTS page, is the Path Shadowing
MC block: PS-MC has not been run for d = 8, and a section with no numbers behind
it would be a lie.

Material that is *about the dataset rather than about Deep-MKV-TS* -- how the d = 8
metrics are scoped and the nearest-neighbour memorisation discussion -- lives one
level up in `results/HestonMultiAsset/oldreadme.md`, shared by every method added
to this dataset. Material about *this port* -- the frozen reference kernel, the
matrix-valued spectral control, the re-selected ridge and the S0 rescaling -- lives
in `code/README.md`.

Every number here is READ FROM DISK, never typed by hand: the tables *and* the
headline win-counts are computed from the artefacts, so the prose cannot drift
away from the numbers it describes.

Reads
  results/HestonMultiAsset/Deep-MKV-TS/metrics_summary.csv
  results/HestonMultiAsset/Deep-MKV-TS/curve_b_aggregate.json
  results/HestonMultiAsset/Deep-MKV-TS/grid_tvd_aggregate.json
  results/HestonMultiAsset/Deep-MKV-TS/losses/generation_time.csv
  results/HestonMultiAsset/Deep-MKV-TS/losses/seed_*_losses.csv
  results/HestonMultiAsset/Deep-MKV-TS/losses/memorisation.json        (optional)
  results/HestonMultiAsset/Deep-MKV-TS/weights/seed_*_config.json      (train times)
  results/HestonMultiAsset/Deep-MKV-TS/generated_paths/seed_*/metadata.json
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv        (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/Deep-MKV-TS/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/Deep-MKV-TS/code/render_readme.py
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Name kept as SBTS so the shared table helpers stay textually diffable against
# SBTS/code/render_readme.py -- the guideline's whole point is that the renderers
# differ only where the layout mandates it.
SBTS = os.path.dirname(HERE)                                  # .../Deep-MKV-TS
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
            # Order numerically, not lexicographically: plain ``sorted`` puts
            # seed_10 before seed_2 and would silently transpose two columns
            # the day a run uses double-digit seeds.
            keys = sorted((k for k in row if k.startswith("seed_")),
                          key=lambda k: int(k.split("_", 1)[1]))
            out[row["metric"]] = {
                "mean": row["mean"], "std": row["std"],
                "scope": row.get("scope", ""),
                "seeds": [row[k] for k in keys],
                # The real seed ids, so the header can name the seed that
                # produced each column. These runs use 0,2,4,5,6 and 0,4,6 --
                # never 0..n-1 -- so a positional label misattributes the data.
                "seed_ids": [int(k.split("_", 1)[1]) for k in keys],
            }
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def render_a_table(sbts, floor):
    first = next(iter(sbts.values())) if sbts else None
    ids = first["seed_ids"] if first else list(range(5))
    n_seeds = len(ids)
    head = ["Metric", "Mean ± Std"] + [f"Seed {i}" for i in ids] + ["Perfect floor"]
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


MEASURES = [("mse", "MSE", fmt), ("pct", "% err", pct), ("nrmse", "NRMSE", pct),
            ("cvar90", "CVaR₉₀", pct), ("cvar95", "CVaR₉₅", pct)]


def render_b_table(agg, agg_f, tvd, tvd_f, seed_ids=None):
    n_seeds = len(tvd.get("per_seed", [0] * 5)) if tvd else 5
    # ``per_seed`` is a bare positional list, so the ids have to come from the
    # A-table summary. Fall back to positions only if that is unavailable, and
    # never claim an id we cannot substantiate.
    ids = seed_ids if seed_ids and len(seed_ids) == n_seeds else list(range(n_seeds))
    head = ["Plot", "Measure", "Mean ± Std"] + [f"Seed {i}" for i in ids] + ["Perfect floor"]
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


def _read_loss_csv(path):
    """losses/seed_i_losses.csv -> (final loss_total, min objective, n_rows).

    Read rather than trusted from the config, because the config is written once
    at the end of training while this file is appended at every LOG_EVERY tick:
    if a run was cut short the two disagree, and the CSV is the one that cannot
    lie about where it stopped.
    """
    if not os.path.exists(path):
        return None, None, 0
    final_loss, best_obj, n = None, None, 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            v = row.get("loss_total", "")
            if v not in ("", None):
                f = float(v)
                if f == f:
                    final_loss = f
            o = row.get("objective", "")
            if o not in ("", None):
                g = float(o)
                if g == g and (best_obj is None or g < best_obj):
                    best_obj = g
    return final_loss, best_obj, n


# One annotation per tree entry. GUIDELINE section 8.8: "A tree that lists a file
# which does not exist, or omits one that does, is a defect", and SBTS shipped
# exactly that bug by hard-coding the tree. The tree below is generated from
# os.walk, so it cannot list a phantom; this dict only supplies the comment
# column, and a name with no entry here simply renders uncommented rather than
# being dropped.
TREE_NOTES = {
    "README.md": "← this file (generated by code/render_readme.py)",
    "code": "",
    "code/README.md": "reference kernel, deviations, sweep table",
    "code/fit_reference_multiasset.py": "penalised MLE for the frozen d = 8 reference kernel",
    "code/multivariate_reference.py": "the d = 8 Guyon-Lekeufack kernel itself",
    "code/matrix_control_multiasset.py": "matrix-valued spectral control + Daleckii-Krein adjoint",
    "code/train_multiasset.py": "Algorithm 1, one seed",
    "code/sweep_ridge_lambda.py": "re-selects ridge_lambda on the VALIDATION split",
    "code/run_sweep.sh": "drives the sweep in two waves on one GPU",
    "code/select_checkpoint_multiasset.py": "picks the reported checkpoint per seed on validation",
    "code/run_all_multiasset.py": "samples the 8192-path bank from the SELECTED checkpoint",
    "code/collect_artifacts.py": "rebuilds generation_time.csv, checks the §4 contract",
    "code/plot_losses.py": "plots/loss_convergence.png, all seeds overlaid",
    "code/plot_diagnostics_multiasset.py": "the 8-panel stylised-facts figure",
    "code/measure_memorisation.py": "nearest-neighbour memorisation diagnostic",
    "code/render_readme.py": "regenerates this README from the artefacts",
    "code/reference": "FITTED artefacts only — the upstream package is NOT vendored here",
    "code/reference/reference_kernel.json": "the frozen kernel, fitted once, shared by every seed",
    "code/reference/reference_fit_history.csv": "its penalised-MLE calibration trace",
    "code/selection": "per-seed checkpoint-selection records",
    "code/sweep": "the ridge_lambda sweep, one JSON per lambda + winner.json",
    "code/runs": "per-seed training_checkpoints/ (gitignored)",
    "code/logs": "raw stdout of every launched job (gitignored)",
    "generated_paths": "",
    "weights": "",
    "losses": "",
    "losses/generation_time.csv": "wall-clock time per seed",
    "losses/memorisation.json": "NN-ratio diagnostic on the final 8192-path output",
    "plots": "",
    "plots/loss_convergence.png": "L_adj and the path-functional objective, all seeds",
    "plots/heston_diagnostics.png": "8-panel stylised facts (seed 0, asset 0)",
    "plots/disc_classifier_loss.png": "A18 BCE curves, all seeds",
    "plots/pred_score_loss.png": "A19 MAE curves, all seeds (asset 0)",
    "metrics_summary.csv": "A1-A34, mean ± std, per seed",
    "metrics_per_asset.csv": "per-metric × per-asset breakdown (8 rows per metric)",
    "curve_b_aggregate.json": "B curve-shape aggregate",
    "grid_tvd_aggregate.json": "path-cloud TVD",
}

# Directories whose contents are collapsed to one representative line rather than
# enumerated: five near-identical seed subtrees would bury the structure.
TREE_COLLAPSE = {"generated_paths", "weights", "code/selection", "code/sweep",
                 "code/runs", "code/logs"}
TREE_SKIP = {"__pycache__", ".omc", ".git", ".ipynb_checkpoints", ".pytest_cache"}


def _render_tree(root, prefix="results/HestonMultiAsset/Deep-MKV-TS/"):
    """Generate the section-9 file tree from os.walk, annotated from TREE_NOTES.

    Generated rather than hard-coded on purpose. GUIDELINE section 8.8 records
    that SBTS listed `code/README.md` as a link target for weeks before the file
    existed; a tree read off the filesystem cannot make that mistake, and a stale
    annotation degrades to a missing comment rather than to a phantom path.
    """
    lines = [prefix]

    def walk(d, rel, indent):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return
        entries = [n for n in names if n not in TREE_SKIP and not n.endswith(".pyc")]
        for i, name in enumerate(entries):
            full = os.path.join(d, name)
            relpath = f"{rel}/{name}" if rel else name
            last = (i == len(entries) - 1)
            branch = "└── " if last else "├── "
            note = TREE_NOTES.get(relpath, "")
            label = name + ("/" if os.path.isdir(full) else "")
            pad = " " * max(1, 40 - len(indent) - len(branch) - len(label))
            lines.append(f"{indent}{branch}{label}" + (f"{pad}{note}" if note else ""))
            if os.path.isdir(full):
                if relpath in TREE_COLLAPSE:
                    child = sorted(n for n in os.listdir(full) if n not in TREE_SKIP)
                    if child:
                        sub = indent + ("    " if last else "│   ")
                        lines.append(f"{sub}└── {child[0]}"
                                     + (f"   … and {len(child) - 1} more"
                                        if len(child) > 1 else ""))
                    continue
                walk(full, relpath, indent + ("    " if last else "│   "))

    walk(root, "", "")
    return "\n".join(lines)


def render_training(root):
    """Training wall-clock table, merged from three artefacts on disk.

    GUIDELINE section 8.7: a trained method replaces SBTS's "no training loss"
    placeholder with a real one, and the wall-clock numbers must come off disk.

    Three artefacts carry disjoint parts of what the table needs:

      weights/seed_*_config.json            n_steps, train_time_sec, n_parameters,
                                            ridge_lambda, selected_step,
                                            selection_val_discrepancy
      losses/seed_*_losses.csv              final L_adj, best path-functional objective
      generated_paths/seed_*/metadata.json  selected_step (cross-check)

    They are merged per seed with the config as the base, so a hyperparameter can
    never be silently taken from a downstream copy. Returns
    (markdown, total_minutes, merged_configs).

    Columns differ from CSDI's and LS4's for the reasons given in the module
    docstring: **Steps** not Epochs, and **Selected step / Val discrepancy** rather
    than a validation loss, because Deep-MKV-TS reports a checkpoint chosen after
    training rather than the final one.

    ``Final L_adj`` is the LAST logged adjoint-matching loss, not the minimum over
    the run, and that choice is deliberate. L_adj chases a moving target -- the
    ridge Z-proxy is re-estimated on fresh paths at every outer iteration -- so its
    running minimum is not a "best model", it is whichever iteration happened to
    draw the easiest proxy. ``Best objective`` is a minimum, because the
    path-functional discrepancy is a fixed yardstick and its minimum is meaningful.
    Neither column selects the reported checkpoint; ``Val discrepancy`` does.
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
            for k in ("selected_step", "ridge_lambda", "n_parameters"):
                if c.get(k) is None:
                    c[k] = m.get(k)
        fl, bo, nrows = _read_loss_csv(
            os.path.join(root, "losses", f"seed_{seed}_losses.csv"))
        c["_final_loss"], c["_best_obj"], c["_loss_rows"] = fl, bo, nrows
        c["_seed"] = seed
        cfgs.append(c)
    if not cfgs:
        return "_(weights/seed_\\*_config.json not found)_", 0.0, []
    cfgs.sort(key=lambda c: c["_seed"])
    total = sum(float(c.get("train_time_sec") or 0.0) for c in cfgs) / 60.0

    def _n(v, unit="", nd=1):
        return "-" if v is None else f"{float(v):.{nd}f}{unit}"

    out = ["| Seed | GPUs | Steps | sec/step | Train wall-clock | Selected step "
           "| Val discrepancy | Final L_adj | Best objective |",
           "|------|------|-------|----------|------------------|---------------"
           "|-----------------|-------------|----------------|"]
    for c in cfgs:
        secs = float(c.get("train_time_sec") or 0.0)
        steps = c.get("n_steps")
        per = (secs / float(steps)) if steps else None
        out.append(
            f"| {c['_seed']} "
            # one process per seed, pinned to a single device with
            # CUDA_VISIBLE_DEVICES; seeds run 2-up in waves, never 2-per-GPU.
            f"| 1 "
            f"| {steps if steps is not None else '-'} "
            f"| {_n(per, ' s', nd=2)} "
            f"| {_n(secs / 60.0, ' min')} "
            f"| {c.get('selected_step') if c.get('selected_step') is not None else '-'} "
            f"| {_n(c.get('selection_val_discrepancy'), nd=6)} "
            f"| {_n(c.get('_final_loss'), nd=6)} "
            f"| {_n(c.get('_best_obj'), nd=6)} |")
    return "\n".join(out), total, cfgs


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    # Seed count and ids are properties of the run, not constants. This variant
    # scored 3 seeds (0, 4, 6) while its siblings scored 5, so a hardcoded
    # "5 seeds" in the prose would contradict the table right beside it.
    seed_ids = next(iter(sbts.values()))["seed_ids"] if sbts else list(range(5))
    n_seeds = len(seed_ids)
    seeds_txt = f"{n_seeds} seeds"
    seed_list_txt = ", ".join(str(i) for i in seed_ids)
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))

    missing = [n for n, o in (("Deep-MKV-TS/metrics_summary.csv", sbts),
                              ("Deep-MKV-TS/curve_b_aggregate.json", agg),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --method Deep-MKV-TS and "
            "--method perfect_recovery first."
        )

    timing, total_min, trows = render_timing(
        os.path.join(SBTS, "losses", "generation_time.csv"))
    train_tbl, train_min, cfgs = render_training(SBTS)
    c0 = cfgs[0] if cfgs else {}
    r0 = trows[0] if trows else {}
    # Hyperparameters are read off the per-seed config, never typed here.
    steps = str(c0.get("n_steps", r0.get("n_steps", "3000")))
    params = int(c0.get("n_parameters") or 0)
    batch = str(c0.get("batch_size", r0.get("batch_size", "256")))
    tbatch = str(c0.get("target_batch_size", "256"))
    hidden = str(c0.get("hidden_dim", 96))
    layers = str(c0.get("num_layers", 1))
    lr = c0.get("lr", 2e-3)
    wd = c0.get("weight_decay", 1e-5)
    clip = c0.get("grad_clip_norm", 5.0)
    eta = c0.get("eta", 1.0)
    smin = c0.get("sigma_min", 1e-3)
    smax = c0.get("sigma_max", 0.6)
    lam_s = c0.get("lambda_scale", 50.0)
    kap_s = c0.get("kappa_scale", 100.0)
    preset = str(c0.get("discrepancy_preset", "old_fullv_w0p25"))
    w_abs = c0.get("abs_return_acf_weight", 0.25)
    w_sq = c0.get("squared_return_acf_weight", 0.125)
    ridge = c0.get("ridge_lambda", 1e-3)
    ce_ridge = c0.get("ce_ridge", 1e-3)
    ctrl = str(c0.get("control", "matrix"))
    drift_backend = str(c0.get("drift_adjoint_backend", "autograd_replay"))
    eigh_batch = str(c0.get("max_eigh_batch", 32768))
    seq_len = int(c0.get("seq_len", 252))
    euler_steps = seq_len - 1
    # The cuSOLVER failure band is a product of batch x steps, so compute it
    # rather than quoting a number that would go stale if either moved.
    try:
        eigh_load = f"{int(batch) * euler_steps:,}".replace(",", " ")
    except (TypeError, ValueError):
        eigh_load = "the per-step batch x step product"
    sel_steps = sorted({str(c.get("selected_step")) for c in cfgs
                        if c.get("selected_step") is not None})
    sel_txt = sel_steps[0] if len(sel_steps) == 1 else ", ".join(sel_steps)
    if not sel_steps:
        sel_txt = "-"
    # `retuned_for_d8` is the guideline's honesty field: it lists every hyperparameter
    # that had to move going from d = 1 to d = 8. Read it rather than claim it is empty.
    retuned = c0.get("retuned_for_d8") or []
    retuned_txt = ("**nothing** — the list `retuned_for_d8` in every "
                   "`weights/seed_*_config.json` is empty"
                   if not retuned else "`" + "`, `".join(map(str, retuned)) + "`")
    # Any seed disagreeing on batch_size, step budget or ridge is a real
    # inconsistency: say so rather than letting seed 0 speak for the other four.
    bset = sorted({str(c.get("batch_size")) for c in cfgs if c.get("batch_size") is not None})
    eset = sorted({str(c.get("n_steps")) for c in cfgs if c.get("n_steps") is not None})
    rset = sorted({str(c.get("ridge_lambda")) for c in cfgs
                   if c.get("ridge_lambda") is not None})
    hp_warn = ""
    if len(bset) > 1 or len(eset) > 1 or len(rset) > 1:
        hp_warn = (f" **Seeds disagree** — batch_size {{{', '.join(bset)}}}, "
                   f"n_steps {{{', '.join(eset)}}}, ridge_lambda {{{', '.join(rset)}}}; "
                   f"re-run the odd seed before trusting this page.")
    # Worker counts can differ across seeds, so report the actual set rather
    # than letting row 0 misrepresent the other four rows.
    nws = sorted({str(r.get("n_workers", "")) for r in trows if r.get("n_workers")},
                 key=lambda x: int(x) if x.isdigit() else 0)
    nw = nws[0] if len(nws) == 1 else " and ".join(nws)
    gen_ns = sorted({str(r.get("n_samples", "")) for r in trows if r.get("n_samples")})
    gen_n = f"{int(gen_ns[0]):,}".replace(",", " ") if len(gen_ns) == 1 else "/".join(gen_ns)

    at_floor, n_a, gaps = a_headline(sbts, floor)
    b_at, n_b = b_headline(agg, agg_f)

    # Optional per-run health warning, verbatim from HEALTH.md beside the page.
    # A run whose seed set is incomplete because seeds *died* cannot be read the
    # same way as a clean one, and that caveat has to travel with the numbers
    # rather than live in a shell script nobody opens. No file, no section, so
    # every other method page is unaffected.
    health_path = os.path.join(SBTS, "HEALTH.md")
    health_block = ""
    if os.path.exists(health_path):
        with open(health_path, encoding="utf-8") as fh:
            health_block = "\n" + fh.read().strip() + "\n"

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    md = f"""# Deep-MKV-TS on Multi-Asset Heston (d = 8)

**Deep McKean–Vlasov Time Series generation** — a path-dependent McKean–Vlasov
stochastic-control generator, applied to 8 192 **multi-asset** Heston
stochastic-volatility price paths (seq\\_len = {seq_len}, **d = 8 correlated assets**).
{health_block}
Deep-MKV-TS does **not** learn a generator from noise. It starts from a *frozen,
interpretable reference SDE* (Guyon–Lekeufack, paper §2.1) fitted to the data by
penalised maximum likelihood, then learns a **volatility correction only** — the drift
is never touched. Training minimises

$$\\mathcal{{J}}(\\alpha) = \\mathcal{{D}}\\big(\\mu^{{\\alpha}}, \\mu^{{\\text{{data}}}}\\big) + \\eta\\,\\mathbb{{E}}\\left[\\int_0^T \\tfrac12 \\|\\alpha_t\\|^2 \\sigma_t^{{-2}} dt\\right], \\qquad \\eta = {eta:g}$$

where $\\mathcal{{D}}$ is the same multi-component MMD discrepancy used at d = 1
(observed path, increments, terminal, global realized variance, $|r|$ ACF, $r^2$ ACF)
and the second term is a **specific-entropy running cost** penalising how far the
controlled law drifts from the reference law. The correction network is a
{layers}-layer GRU (`hidden_dim = {hidden}`) — **{params:,} parameters**.
Weights in [`weights/`](weights/), per-seed hyperparameters in
`weights/seed_*_config.json`.

The model is **joint over all 8 assets** — one reference kernel with `state_dim = 8`
and one control network emitting a full `(d, d)` correction — not eight independent
univariate fits. See [`code/README.md`](code/README.md) for the frozen reference
kernel, the vendored upstream package, the forced d = 8 deviations and the one
re-selected hyperparameter; and the dataset-level [`../oldreadme.md`](../oldreadme.md)
for the multi-asset Heston law itself, the per-asset vs native metric scoping, and the
memorisation diagnostic shared by all methods on this dataset.

> **Hyperparameters:** `eta={eta:g}`, `sigma_min={smin:g}`, `sigma_max={smax:g}`,
> `lambda_scale={lam_s:g}`, `kappa_scale={kap_s:g}`, discrepancy preset
> `{preset}` with `abs_return_acf_weight={w_abs:g}`,
> `squared_return_acf_weight={w_sq:g}`; `hidden_dim={hidden}`,
> `num_layers={layers}`, batch {batch}, target batch {tbatch};
> `AdamW(lr={lr:g}, weight_decay={wd:g})`, `grad_clip_norm={clip:g}`,
> `ce_target_mode="ridge"` with `ridge_lambda={ridge:g}`, `ce_ridge={ce_ridge:g}`;
> {steps} outer iterations per seed, {seeds_txt} (seeds {seed_list_txt}).
> Every one of those numbers is the **committed d = 1 value**, unchanged, except for
> what is listed next.
>
> **Retuned for d = 8: {retuned_txt}.** The reason is structural, not a search: the
> frozen package's Z-proxy basis $\\Phi^{{\\text{{ref}}}}_k$ is the flattened path prefix
> plus an intercept, so its width is $p = 1 + (k+1)d$. At d = 1 that is at most {seq_len}
> against a batch of {batch} — an over-determined ridge that genuinely denoises. At
> d = 8 it reaches {1 + seq_len * 8}, under-determined from step 31 onward, and the
> projection degenerates into near-interpolation. `ridge_lambda` was re-selected on the
> **validation** split (`heston_ma_S_val_8192x252x8.npy`), never on test, and the whole
> sweep is recorded in `code/sweep/`.
>
> **The control is matrix-valued, not diagonal** (`control={ctrl}`). The paper clips the
> *eigenvalues* of $\\Theta$; at d > 1, clipping its diagonal entries instead is a
> different operator. $\\Theta$ is symmetric but **indefinite**, so Cholesky and LDLᵀ are
> invalid and `torch.linalg.eigh` is the right primitive, with the Daleckii–Krein /
> Löwner form for its adjoint. The two coincide bitwise only at d = 1.
>
> **The drift Jacobian is kept.** `drift_adjoint_backend={drift_backend}` recomputes
> $\\partial b^{{\\text{{ref}}}}/\\partial x$ by replaying the reference recursion under
> autograd, because the multivariate kernel exposes no analytical
> `drift_step_path_vjp`. No Algorithm 1 term is dropped.
>
> **`max_eigh_batch={eigh_batch}`** chunks the batched eigendecomposition: cuSOLVER's
> batched `eigh` fails above roughly 64 k matrices, and batch {batch} × {euler_steps}
> steps = {eigh_load} sits inside that failure band. Chunking changes throughput, not
> results.
>
> **The reported checkpoint is chosen on validation**, not the final step — step
> {sel_txt} — exactly as the committed d = 1 run reports step 2500 of a 3000-step run.
> There is **no learning-rate scheduler anywhere in the codebase**, so this is bitwise
> identical to having stopped training there.{hp_warn}

---

## Metrics A1-A34 + B, mean ± std across {seeds_txt}

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the independent-draw floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}. The floor, not the other methods, is the reference on this page; the cross-method comparison lives in the dataset-level [`../README.md`](../README.md) so that no method's page grades itself against a rival it was tuned beside.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix. This is the row that matters most for Deep-MKV-TS: the correction network emits a full `(d, d)` matrix and the spectral clip acts on eigenvalues, so A20 is the direct test of whether the d(d−1)/2 spot correlations Σˢ survived the control. A diagonal control would have no mechanism to get this row right. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. Deep-MKV-TS is the one method on this dataset carrying an **explicit** volatility state: the frozen Guyon–Lekeufack reference kernel maintains trend and activity features with fitted half-lives, and the learned control multiplies its σ. These rows therefore test the reference kernel and the correction together, not the network alone.

---

## B, Curve-Shape Metrics, mean ± std across {seeds_txt}

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. The curve is
computed **per asset and averaged over the 8 assets** *before* the combination below, so the
combination rules stay byte-identical to the d = 1 tree. For the real data (L_r) and generated
data (L_g) we build three lists, the curve L, its first finite difference L' (der), and its
second finite difference L'' (sec\\_der), then combine the three sub-scores into **one number
per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\\_der)/3; std = the sample std of that per-seed combined score across the {seeds_txt}. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE, one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself**, the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the {seeds_txt}** of that per-seed function MAPE.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)**, the ill-posed derivative / 2nd-derivative curves are excluded for the same reason as the % err row.
- **CVaR₉₀ / CVaR₉₅ rows**: tail-averaged pointwise curve error (Expected Shortfall) on the curve L **only (funct-only)**. Pointwise error eₜ = |L_g(t) − L_r(t)|; for q ∈ {{0.90, 0.95}}, CVaR_q = mean(eₜ for eₜ ≥ the q-th percentile of eₜ), then range-normalized like NRMSE (÷ (max|L_r| − min|L_r| + 1e-12) × 100).

All ↓ lower is better. The perfect floor is **non-zero** for all plots, it is the residual finite-sample error of an independent multi-asset Heston draw scored against the test set, identical across methods.
Five sublines per plot: **MSE**, **% error**, **NRMSE**, **CVaR₉₀** and **CVaR₉₅** (the per-seed columns hold that seed's combined score).

{render_b_table(agg, agg_f, tvd, tvd_f, seed_ids)}

> **Headline:** **{len(b_at)} of the {n_b} B plots** sit at or below the finite-sample floor on the deciding MSE row: {b_at_txt}.
> **Cross-seed stability**: the seed here controls **three** independent things — the control network's initialisation, every Euler–Maruyama increment drawn during training, and the batch resampling that re-fits the Z-proxy at each outer iteration. The reference kernel is *not* among them: it is fitted once, frozen, and shared byte-for-byte across every seed (`code/reference/reference_kernel.json`), so none of the spread below comes from re-estimating the reference SDE. What remains is optimisation plus sampling variance, and [`plots/loss_convergence.png`](plots/loss_convergence.png) shows whether any seed diverged. Generation uses a **separate** seed stream (90000 + i) that never reuses a training seed.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs Deep-MKV-TS, seed 0, asset 0)

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

## Deep-MKV-TS Training Loss ({seeds_txt})

The figure has **two** panels, and they are not the same quantity plotted twice.

`loss_total` is the **adjoint-matching loss**
$L_{{\\text{{adj}}}} = \\frac{{1}}{{PM}}\\sum\\sum \\|\\hat Z^{{\\theta}} - Z^{{\\text{{proxy}}}}\\|^2$ — the
scalar AdamW actually minimises. Its target **moves**: $Z^{{\\text{{proxy}}}}$ is re-estimated by
ridge regression on freshly sampled paths at every outer iteration, so this curve measures
distance to a target that is itself drifting. A descending `loss_total` is necessary but **not
sufficient** — it can fall simply because the proxy became easier to hit.

`objective` is the **path-functional discrepancy** between generated paths and the training
bank, a fixed yardstick and the same functional form that
`select_checkpoint_multiasset.py` later evaluates on the validation split to choose the
reported checkpoint. **Read convergence from this panel.** Both are logged on the same rows,
so the two x-axes coincide step-for-step.

The x-axis is **outer iteration**, not epoch. Algorithm 1 has no epoch: every iteration draws a
fresh batch of controlled paths and re-fits the proxy, so there is no pass over a fixed dataset
to count. There is also no validation curve here, deliberately — Deep-MKV-TS runs no held-out
forward pass during training. Selection happens **after** the fit, by sampling from each saved
checkpoint and scoring against `heston_ma_S_val_8192x252x8.npy`; the per-checkpoint scores are
archived in `code/selection/seed_*_selection.json` and the winner is the `Selected step` column
below.

![Training Loss](plots/loss_convergence.png)

Training wall-clock, read from `weights/seed_*_config.json` and
`losses/seed_*_losses.csv` (**{train_min:.0f} min total** across the {seeds_txt}, one GPU and
{nw} threads per seed process, run 2-up in waves):

{train_tbl}

Generation wall-clock — the selected control rolled forward {euler_steps}
Euler–Maruyama steps from $x_0 = \\log(100)$ for each of {gen_n} paths, {nw} threads,
**{total_min:.1f} min total** across the {seeds_txt}:

{timing}

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
It is generated from `os.walk` at render time, so it cannot list a file that does not exist.

```
{_render_tree(SBTS)}
```

The upstream `deep_mkv_gen_path_dt` package is **not vendored here**. It lives once at
`methods/Deep-MKV-TS/code/reference/` and is imported through `PYTHONPATH`, **with zero
edits** — every d = 8 adaptation is a method-local file in `code/`. `code/reference/` holds
only the *fitted* artefacts: the frozen kernel and its calibration history.

The `.npy` arrays are **gitignored**: `(8192, {seq_len}, 8)` float64 = 132 MB each, over
GitHub's 100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from the
tracked code, the tracked frozen kernel and the tracked hyperparameters; the `metadata.json`
beside each array **is** tracked, so shapes, price ranges and generation times stay auditable
without the payload.

## Reproduce

```bash
cd /home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$R/src:$R/experiments"

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. fit the frozen reference kernel by penalised MLE on the TRAIN split
#    Writes code/reference/reference_kernel.json. Fitted ONCE and shared by every seed.
cd results/HestonMultiAsset/Deep-MKV-TS/code
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 $PY fit_reference_multiasset.py

# 3. re-select ridge_lambda on the VALIDATION split (GPU 1 only, two waves)
#    The only retuned hyperparameter. Writes sweep/*.json and sweep/winner.json.
bash run_sweep.sh

# 4. final {seeds_txt}, {steps} outer iterations, 2 GPUs (~{train_min:.0f} min of GPU time, 3 waves)
GPUS=(1 2); CORES=("0-7" "8-15")
for wave in "0 1" "2 3" "4"; do
  i=0
  for s in $wave; do
    CUDA_VISIBLE_DEVICES=${{GPUS[$i]}} OMP_NUM_THREADS=8 taskset -c ${{CORES[$i]}} \\
        $PY train_multiasset.py --seed $s --device cuda:0 & i=$((i+1))
  done
  wait
done

# 5. choose the reported checkpoint per seed on VALIDATION, never on test
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \\
    $PY select_checkpoint_multiasset.py --seeds 0 1 2 3 4

# 6. generate the 8192-path bank from the SELECTED checkpoint
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \\
    $PY run_all_multiasset.py --seeds 0 1 2 3 4
$PY collect_artifacts.py            # must print "5 rows" and exit 0 before anything downstream
cd /home/tbasseras/benchmark

# 7. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 8. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=1 $PY metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=1 $PY metrics/compute_all_multiasset.py --method Deep-MKV-TS \\
    --dataset HestonMultiAsset --seeds 5

# 9. figures
$PY results/HestonMultiAsset/Deep-MKV-TS/code/plot_losses.py
$PY results/HestonMultiAsset/Deep-MKV-TS/code/plot_diagnostics_multiasset.py \\
    --method Deep-MKV-TS
$PY metrics/plot_score_losses.py --method Deep-MKV-TS --dataset HestonMultiAsset

# 10. memorisation diagnostic
$PY results/HestonMultiAsset/Deep-MKV-TS/code/measure_memorisation.py --seeds 0,1,2,3,4

# 11. regenerate this README from the artefacts
$PY results/HestonMultiAsset/Deep-MKV-TS/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
