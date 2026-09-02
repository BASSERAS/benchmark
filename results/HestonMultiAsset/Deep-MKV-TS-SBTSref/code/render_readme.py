#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/Deep-MKV-TS-SBTSref/README.md from the artefacts.

The section order, table shape and prose slots are a 1:1 mirror of
`results/HestonMultiAsset/Deep-MKV-TS/README.md`, itself a mirror of the SBTS and
d = 1 pages, so every method page on this dataset displays identically and can be
diffed row-by-row. MULTIASSET_GUIDELINE.md section 8 fixes that layout.

Every line of scoring and table logic from `CATEGORIES` down to `render_training`
is byte-identical to `Deep-MKV-TS/code/render_readme.py`. That is deliberate and it
is checkable, so the check is stated exactly rather than approximately:

    diff Deep-MKV-TS/code/render_readme.py \
         Deep-MKV-TS-SBTSref/code/render_readme.py

should show four things and nothing else -- this module docstring, the *docstring*
of `a_headline` (prose, no logic: it explains why the floor is still the only
reference now that sibling methods exist at d = 8), the tree block, and everything
from `render_memorisation` onward. If that diff ever shows a table helper, a
threshold or a formatter, someone changed how this method is scored relative to its
sibling, which is precisely the thing that must never happen quietly.

What this method is, and why its page cannot reuse the sibling's prose
---------------------------------------------------------------------
Both methods run the same Algorithm 1 and learn the same matrix-valued volatility
correction. They differ in the *reference SDE* the control corrects, and that
difference propagates into almost every claim on the page:

  * The sibling's reference drift is a **parametric Guyon-Lekeufack kernel**, fitted
    once by penalised MLE and frozen into `reference_kernel.json`. After fitting,
    the training set is gone; only the fitted numbers remain.
  * This method's reference drift is the **SBTS Markovian kernel average**

        b^ref_i(X_{1:i}) = sum_m w_m(X_{1:i}) R^m_{i+1} / dt,
        w_m = prod_{l=i-K+1}^{i} Kh(Xtilde^m_l - rt_l)

    over the 8192-path TRAIN bank, with a radial quartic kernel Kh. It is
    nonparametric and has no fitted weights at all: the bank *is* the model, and it
    is read at **every sampling step**, at generation time, not only at fit time.

Three concrete consequences for this file:

  1. There is no `fit_reference_multiasset.py` step in the Reproduce block and no
     `code/reference/` directory in the tree, because there is nothing to fit.
  2. `h`, `markov_order` and `npi` are promoted into the hyperparameter blockquote
     and into the timing table. Two runs with identical network weights and
     different `h` are different models; a page that hid `h` in a config file would
     be describing a model it never names.
  3. The memorisation number has to be read against **both** neighbours -- SBTS
     proper (0.2189 at d = 8) and the sibling -- never against the sibling alone.
     See `code/measure_memorisation.py` for the full argument.

`render_training` is shared with the sibling and the reasons it differs from
CSDI's and LS4's carry over unchanged: Algorithm 1 counts **outer iterations**, not
epochs, because each iteration resamples a fresh batch and re-fits the Z-proxy
ridge regression; the reported checkpoint is **chosen on validation** by
`select_checkpoint_multiasset.py` rather than being the last step; and there is no
validation *loss* to tabulate, because training runs no held-out forward pass.

The one deliberate omission, inherited from the SBTS page, is the Path Shadowing
MC block: PS-MC has not been run for d = 8, and a section with no numbers behind
it would be a lie.

Material that is *about the dataset* -- how the d = 8 metrics are scoped -- lives one
level up in `results/HestonMultiAsset/oldreadme.md`, shared by every method on this
dataset. Material about *this port* -- the SBTS reference kernel, the analytic
weight Jacobian, the matrix-valued spectral control and the S0 rescaling -- lives in
`code/README.md`. The hyperparameter search lives in `code/SWEEP.md`.

Every number here is READ FROM DISK, never typed by hand: the tables *and* the
headline win-counts are computed from the artefacts, so the prose cannot drift
away from the numbers it describes.

Reads
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/metrics_summary.csv
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/curve_b_aggregate.json
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/grid_tvd_aggregate.json
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/generation_time.csv
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/seed_*_losses.csv
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/losses/memorisation.json  (optional)
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/weights/seed_*_config.json
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/generated_paths/seed_*/metadata.json
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv        (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/Deep-MKV-TS-SBTSref/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code/render_readme.py
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Name kept as SBTS so the shared table helpers stay textually diffable against
# SBTS/code/render_readme.py -- the guideline's whole point is that the renderers
# differ only where the layout mandates it.
SBTS = os.path.dirname(HERE)                                  # .../Deep-MKV-TS-SBTSref
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
    Here the reference is the floor and only the floor, even though SBTS and
    Deep-MKV-TS already exist at d = 8: a method page that graded itself against
    the very sibling it was designed to be compared with would be choosing its own
    opponent. The cross-method table in the dataset README is where that
    comparison belongs, computed once, for all methods, by the same code.
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
    "code/README.md": "the reference, the deviations, the four mandatory questions",
    "code/SWEEP.md": "the hyperparameter search, generated from code/sweep/",
    "code/sbts_reference.py": "the SBTS Markovian kernel: b_ref, sigma_ref and db/dx",
    "code/validate_sbts_reference.py": "checks b_ref against a brute-force reimplementation",
    "code/matrix_control_multiasset.py": "matrix-valued spectral control + Daleckii-Krein adjoint",
    "code/eigh_fallback.py": "CPU LAPACK retry when batched cuSOLVER eigh fails",
    "code/train_multiasset.py": "Algorithm 1, one seed",
    "code/sweep_hyperparams.py": "one sweep arm: train short, score on VALIDATION",
    "code/render_sweep.py": "turns code/sweep/*.json into code/SWEEP.md",
    "code/probe_dc.py": "measures whether the drift-correction term changes anything",
    "code/select_checkpoint_multiasset.py": "picks the reported checkpoint per seed on validation",
    "code/run_all_multiasset.py": "samples the 8192-path bank from the SELECTED checkpoint",
    "code/collect_artifacts.py": "rebuilds generation_time.csv, checks the §4 contract",
    "code/plot_losses.py": "plots/loss_convergence.png, 5 seeds overlaid",
    "code/plot_diagnostics_multiasset.py": "the 8-panel stylised-facts figure",
    "code/measure_memorisation.py": "nearest-neighbour memorisation diagnostic",
    "code/render_readme.py": "regenerates this README from the artefacts",
    "code/run_pipeline.sh": "the whole post-training chain, in order, with hard gates",
    "code/selection": "per-seed checkpoint-selection records",
    "code/sweep": "one JSON per sweep arm + incumbent.json",
    "code/runs": "per-seed training_checkpoints/ (gitignored)",
    "generated_paths": "",
    "weights": "",
    "losses": "",
    "losses/generation_time.csv": "wall-clock time per seed, plus h / K / npi / grad mode",
    "losses/memorisation.json": "NN-ratio diagnostic on the final 8192-path output",
    "losses/campaignlogs": "raw stdout of the 5 final-campaign seeds (gitignored)",
    "losses/campaignlogs_detached_aborted": "the aborted first launch, kept as the record",
    "losses/sweeplogs": "raw stdout of every sweep arm (gitignored)",
    "losses/reference_validation.json": "output of code/validate_sbts_reference.py",
    "plots": "",
    "plots/loss_convergence.png": "L_adj and the path-functional objective, 5 seeds",
    "plots/heston_diagnostics.png": "8-panel stylised facts (seed 0, asset 0)",
    "plots/disc_classifier_loss.png": "A18 BCE curves, 5 seeds",
    "plots/pred_score_loss.png": "A19 MAE curves, 5 seeds (asset 0)",
    "metrics_summary.csv": "A1-A34, mean ± std, per seed",
    "metrics_per_asset.csv": "per-metric × per-asset breakdown (8 rows per metric)",
    "curve_b_aggregate.json": "B curve-shape aggregate",
    "grid_tvd_aggregate.json": "path-cloud TVD",
}

# Directories whose contents are collapsed to one representative line rather than
# enumerated: five near-identical seed subtrees would bury the structure.
TREE_COLLAPSE = {"generated_paths", "weights", "code/selection", "code/sweep",
                 "code/runs", "losses/campaignlogs", "losses/sweeplogs",
                 "losses/campaignlogs_detached_aborted"}
TREE_SKIP = {"__pycache__", ".omc", ".git", ".ipynb_checkpoints", ".pytest_cache"}

# Groups of near-identical FILES collapsed to one line, the same way TREE_COLLAPSE
# handles near-identical directories. The greedy sweep emitted one launcher shell
# script per stage; fourteen `run_stage_*.sh` lines would bury the eighteen files
# that actually matter. They are collapsed, not hidden -- the line states how many
# there are, and code/SWEEP.md documents what each stage did.
TREE_FILE_COLLAPSE = [
    ("code", "run_stage_", ".sh", "per-stage sweep launchers (see code/SWEEP.md)"),
    ("losses", "seed_", "_losses.csv", "per-seed training log, one row every 100 steps"),
    ("losses", "timing_", ".log", "reference-kernel throughput probes"),
]


def _collapse_files(rel, entries, isdir):
    """Fold each TREE_FILE_COLLAPSE group in `entries` down to one synthetic name.

    Returns (entries, synthetic) where `synthetic` maps the placeholder name to the
    line to print. Files are only ever folded, never dropped: if a group has one
    member it is left alone, so this can never hide a lone file.
    """
    synthetic = {}
    for parent, pre, suf, note in TREE_FILE_COLLAPSE:
        if rel != parent:
            continue
        group = [n for n in entries
                 if n.startswith(pre) and n.endswith(suf) and not isdir(n)]
        if len(group) < 2:
            continue
        key = f"{pre}*{suf}"
        synthetic[key] = f"({len(group)} files) {note}"
        entries = [n for n in entries if n not in group]
        entries.append(key)
        entries.sort()
    return entries, synthetic


def _render_tree(root, prefix="results/HestonMultiAsset/Deep-MKV-TS-SBTSref/"):
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
        entries, synthetic = _collapse_files(
            rel, entries, lambda n: os.path.isdir(os.path.join(d, n)))
        for i, name in enumerate(entries):
            full = os.path.join(d, name)
            relpath = f"{rel}/{name}" if rel else name
            isdir = name not in synthetic and os.path.isdir(full)
            last = (i == len(entries) - 1)
            branch = "└── " if last else "├── "
            note = synthetic.get(name) or TREE_NOTES.get(relpath, "")
            label = name + ("/" if isdir else "")
            pad = " " * max(1, 40 - len(indent) - len(branch) - len(label))
            lines.append(f"{indent}{branch}{label}" + (f"{pad}{note}" if note else ""))
            if isdir:
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


def render_memorisation(root, ma, euler_steps):
    """The NN-ratio block, with BOTH neighbours read off disk rather than quoted.

    `code/measure_memorisation.py` argues at length that this method's number must
    be read against SBTS proper *and* against the sibling, because its reference
    drift is a live kernel average over the training bank while the sibling's is a
    frozen parametric fit. That argument is worthless if the two comparison numbers
    are hand-typed here and then drift. They are loaded from the neighbours' own
    artefacts; a neighbour without one simply drops out of the sentence.
    """
    mine = read_json(os.path.join(root, "losses", "memorisation.json"))
    if not mine:
        return ""
    peers = []
    for name, label in (("SBTS", "SBTS proper (a kernel average and nothing else)"),
                        ("Deep-MKV-TS", "Deep-MKV-TS (parametric frozen reference)")):
        p = read_json(os.path.join(ma, name, "losses", "memorisation.json"))
        if p.get("nn_ratio") is not None:
            peers.append(f"**{label}: {float(p['nn_ratio']):.4f}**")
    peer_txt = ("  For scale, on byte-identical estimator code: "
                + "; ".join(peers) + ".") if peers else ""
    ratio = float(mine["nn_ratio"])
    std = float(mine.get("nn_ratio_std") or 0.0)
    dup = int(mine.get("n_exact_duplicates") or 0)
    rows = ["| Seed | median NN(gen → train) | ratio vs held-out real | exact duplicates |",
            "|------|------------------------|------------------------|------------------|"]
    for s in mine.get("per_seed", []):
        rows.append(f"| {s['seed']} | {float(s['median_nn_generated']):.6f} "
                    f"| {float(s['nn_ratio']):.4f} | {int(s['n_exact_duplicates'])} |")
    return f"""

---

## Memorisation

`nn_ratio` = median over generated paths of the distance to the nearest **training**
path, divided by the same median computed for genuine **held-out** paths, in
log-return space. 1.0 means the generated paths sit no closer to the training set
than real data from the same law does; below 1.0 means they hug it, and 1/ratio is
how many times closer.

**This method scores {ratio:.4f} ± {std:.4f}** ({dup} exact duplicates across all
seeds).{peer_txt}

{chr(10).join(rows)}

> **Why this row cannot be read like the sibling's.** Deep-MKV-TS carries the
> training set only through fitted parameters. This variant does not: its reference
> drift **is** the SBTS kernel average over the 8192-path training bank, evaluated
> at every one of the {euler_steps} sampling steps, and it returns a convex combination of
> that bank's next returns. The training data is inside the generator at generation
> time, by construction. The concrete degeneracy to watch for is small `h`: as the
> bandwidth shrinks the kernel weights concentrate on one bank path and the drift
> starts replaying that path's increments. `code/SWEEP.md` records the bandwidth
> scan and the occupancy argument for the value actually used.
> The exact-duplicate count is close to uninformative here and must not be read as
> evidence of novelty: a sample is a {euler_steps}-step Euler–Maruyama rollout driven by fresh
> Gaussian increments, so bitwise equality with a training path has probability zero
> whatever the model does. The **ratio** is the number that matters."""


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))

    method = os.path.basename(SBTS)
    missing = [n for n, o in ((f"{method}/metrics_summary.csv", sbts),
                              (f"{method}/curve_b_aggregate.json", agg),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            f"Run metrics/compute_all_multiasset.py --method {method} and "
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
    eigh_batch = str(c0.get("max_eigh_batch", 32768))
    seq_len = int(c0.get("seq_len", 252))
    euler_steps = seq_len - 1
    # ---- the reference, which is NOT in the checkpoint ----------------------- #
    # Two runs with identical network weights and a different h are different
    # models, so these are read off the config and printed, never assumed.
    ref_family = str(c0.get("reference_family", "SBTS-Markovian"))
    ref_h = c0.get("reference_h")
    ref_K = c0.get("reference_markov_order")
    ref_npi = c0.get("reference_npi")
    ref_grad = str(c0.get("reference_weight_grad_mode", "analytic"))
    ref_jl = c0.get("reference_jacobian_lags")
    ref_bank = c0.get("reference_bank_paths")
    ref_split = str(c0.get("reference_bank_split", "train"))
    ref_sigma = c0.get("reference_sigma_constant") or []
    sigma_txt = (", ".join(f"{float(v):.3f}" for v in ref_sigma)
                 if ref_sigma else "see weights/seed_*_config.json")
    bank_txt = (f"{int(ref_bank):,}".replace(",", " ") + "-path") if ref_bank else "full"
    # The Reproduce block prints a literal `train_multiasset.py --h ... --markov-order
    # ...` command line. If any of these four is missing from the config that command
    # would still LOOK runnable while silently naming a different model -- `--h None`
    # is not a bandwidth. Refuse to render rather than publish a recipe that does not
    # reproduce the page. This is the same rule collect_artifacts.py enforces when it
    # hard-fails on seeds that disagree about the reference.
    ref_missing = [k for k, v in (("reference_h", ref_h),
                                  ("reference_markov_order", ref_K),
                                  ("reference_npi", ref_npi),
                                  ("reference_weight_grad_mode", c0.get(
                                      "reference_weight_grad_mode")),
                                  ("reference_jacobian_lags", ref_jl)) if v is None]
    if ref_missing:
        raise SystemExit(
            "ABORT: weights/seed_*_config.json is missing " + ", ".join(ref_missing)
            + ".\nThe SBTS kernel is not stored in the checkpoint -- it is rebuilt from"
            " these knobs plus the price bank, so a page that cannot name them cannot"
            " name the model it is describing, and the Reproduce block would print"
            " `--h None`. Re-run train_multiasset.py, or point this at the right method."
        )
    dc_on = bool(c0.get("allow_drift_correction", False))
    dc_txt = ("**on**" if dc_on else "**off**")
    # A campaign that mixed reference settings would not be five samples of one
    # method. collect_artifacts.py already hard-fails on this; say so here too,
    # because the README is what gets read when the numbers are questioned.
    ref_sets = {(str(c.get("reference_h")), str(c.get("reference_markov_order")),
                 str(c.get("reference_npi")),
                 str(c.get("reference_weight_grad_mode")),
                 str(c.get("reference_jacobian_lags"))) for c in cfgs}
    ref_warn = ("" if len(ref_sets) <= 1 else
                " **Seeds do not share one reference configuration** — "
                f"{len(ref_sets)} distinct (h, K, npi, grad mode) tuples across the "
                "5 seeds. These five banks are not five samples of one method; do "
                "not average them.")
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
    # Renders to "" when losses/memorisation.json is absent, so the page degrades to
    # having no memorisation section rather than to an empty one with a heading.
    memo = render_memorisation(SBTS, MA, euler_steps)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    md = f"""# Deep-MKV-TS with an SBTS reference on Multi-Asset Heston (d = 8)

**Deep McKean–Vlasov Time Series generation, with the Signature/Bridge Transport
Sampler as its reference SDE** — applied to 8 192 **multi-asset** Heston
stochastic-volatility price paths (seq\\_len = {seq_len}, **d = 8 correlated assets**).

Like its sibling [`../Deep-MKV-TS/`](../Deep-MKV-TS/), this method does **not** learn a
generator from noise. It starts from a reference SDE and learns a **volatility
correction only** — the drift is never corrected. Training minimises

$$\\mathcal{{J}}(\\alpha) = \\mathcal{{D}}\\big(\\mu^{{\\alpha}}, \\mu^{{\\text{{data}}}}\\big) + \\eta\\,\\mathbb{{E}}\\left[\\int_0^T \\tfrac12 \\|\\alpha_t\\|^2 \\sigma_t^{{-2}} dt\\right], \\qquad \\eta = {eta:g}$$

where $\\mathcal{{D}}$ is the same multi-component MMD discrepancy used at d = 1
(observed path, increments, terminal, global realized variance, $|r|$ ACF, $r^2$ ACF)
and the second term is a **specific-entropy running cost** penalising how far the
controlled law drifts from the reference law. The correction network is a
{layers}-layer GRU (`hidden_dim = {hidden}`) — **{params:,} parameters**.
Weights in [`weights/`](weights/), per-seed hyperparameters in
`weights/seed_*_config.json`.

**What is different here is the reference, and it changes the character of the
method.** The sibling's reference is a *parametric* Guyon–Lekeufack kernel, fitted once
by penalised MLE and then frozen; after fitting, the training set is gone. This
variant's reference is the **{ref_family} kernel average**

$$b^{{\\text{{ref}}}}_i(X_{{1:i}}) = \\frac{{1}}{{\\Delta t}}\\sum_m w_m(X_{{1:i}})\\, R^m_{{i+1}}, \\qquad w_m = \\prod_{{l=i-K+1}}^{{i}} K_h\\big(\\tilde X^m_l - r_l\\big)$$

over the {bank_txt} **{ref_split}** bank, with a radial quartic kernel
$K_h(u) = (h^2 - \\|u\\|^2)^2 \\mathbf{{1}}_{{\\|u\\|<h}}$. It is nonparametric and has no
fitted weights at all: **the bank is the model**, and it is read at every one of the
{euler_steps} sampling steps, at generation time. That is not a defect — it is what SBTS
*is* — but it does mean the memorisation section below has to be read against SBTS
proper as well as against the sibling, never against the sibling alone.

The reference diffusion is the **SBTS-corrected constant**
$\\sigma^{{\\text{{ref}}}} = \\mathrm{{diag}}({sigma_txt})$: SBTS normalises each asset's
returns by its volatility before kernel-matching and multiplies it back afterwards, so
the diffusion the control corrects is that per-asset scale, constant in time and across
paths.

The model is **joint over all 8 assets** — one reference kernel with `state_dim = 8`
and one control network emitting a full `(d, d)` correction — not eight independent
univariate fits. See [`code/README.md`](code/README.md) for the reference kernel, the
analytic weight Jacobian and the forced d = 8 deviations; [`code/SWEEP.md`](code/SWEEP.md)
for the hyperparameter search and its measured noise floor; and the dataset-level
[`../oldreadme.md`](../oldreadme.md) for the multi-asset Heston law itself and the
per-asset vs native metric scoping.

> **Reference hyperparameters (not stored in the checkpoint):** `h={ref_h}`,
> `markov_order={ref_K}`, `npi={ref_npi}`, `weight_grad_mode={ref_grad}`,
> `jacobian_lags={ref_jl}` (`-1` = all `K` lags enter the adjoint, the exact
> gradient; truncating to the newest lag alone leaves 2 of the {int(ref_K) + 1}
> states in the drift window with no gradient at all and is wrong by 100%%
> relative error against finite differences),
> bank = the {bank_txt} **{ref_split}** split.
> These are printed here because the checkpoint does not contain them: the kernel is
> rebuilt at load time from these four knobs plus the price bank, so two runs with
> identical weights and different `h` are **different models**. `h`, `markov_order` and
> `npi` were all selected on the validation split and are the search's only real wins;
> see `code/SWEEP.md`.{ref_warn}
>
> **Training hyperparameters:** `eta={eta:g}`, `sigma_min={smin:g}`,
> `sigma_max={smax:g}`, `lambda_scale={lam_s:g}`, `kappa_scale={kap_s:g}`, discrepancy
> preset `{preset}` with `abs_return_acf_weight={w_abs:g}`,
> `squared_return_acf_weight={w_sq:g}`; `hidden_dim={hidden}`,
> `num_layers={layers}`, batch {batch}, target batch {tbatch};
> `AdamW(lr={lr:g}, weight_decay={wd:g})`, `grad_clip_norm={clip:g}`,
> `ce_target_mode="ridge"` with `ridge_lambda={ridge:g}`, `ce_ridge={ce_ridge:g}`;
> {steps} outer iterations per seed, 5 seeds.
>
> **The learning rate is not the d = 1 value and could not be.** The reference changed,
> so the loss surface changed with it; `lr` and `ridge_lambda` were re-swept on
> validation. `code/SWEEP.md` also records the **run-to-run noise floor** measured
> before any of those comparisons were believed — several arms that look like wins are
> inside it, and are reported there as ties rather than promoted.
>
> **The weight Jacobian is computed analytically, alongside $b^{{\\text{{ref}}}}$
> itself** (`weight_grad_mode={ref_grad}`). Because
> $\\partial \\log w_m / \\partial r_i = 4\\delta_m/(h^2 - \\|\\delta_m\\|^2)$ with
> $\\delta_m = \\tilde X^m_i - r_i$, the kernel's own weights supply the derivative for
> free during the same pass that computes the average, so no autograd replay is needed.
> This is a **correctness** setting, not a speed one: the alternative mode drops
> $\\partial b^{{\\text{{ref}}}}/\\partial x$ from the backward pass entirely.
> One caveat, stated because it is a real limitation and not a rounding error: only the
> **newest lag** is differentiated, so what is carried is a partial, not the total,
> $\\partial b^{{\\text{{ref}}}}/\\partial X_{{1:i}}$.
>
> **There is no drift-correction term** (`allow_drift_correction` is {dc_txt}). This
> follows the specified Algorithm 1, whose state update is
> $X_{{t_{{k+1}}}} = X_{{t_k}} + b^{{\\text{{ref}}}}\\Delta t_k + \\sigma_{{t_k}}\\sqrt{{\\Delta t_k}}\\,\\varepsilon_{{t_{{k+1}}}}$
> with no additional drift term. `code/probe_dc.py` measured the alternative rather than
> assuming it away.
>
> **The control is matrix-valued, not diagonal** (`control={ctrl}`). The paper clips the
> *eigenvalues* of $\\Theta$; at d > 1, clipping its diagonal entries instead is a
> different operator. $\\Theta$ is symmetric but **indefinite**, so Cholesky and LDLᵀ are
> invalid and `torch.linalg.eigh` is the right primitive, with the Daleckii–Krein /
> Löwner form for its adjoint. The two coincide bitwise only at d = 1.
>
> **`max_eigh_batch={eigh_batch}`** chunks the batched eigendecomposition: cuSOLVER's
> batched `eigh` fails above roughly 64 k matrices, and batch {batch} × {euler_steps}
> steps = {eigh_load} sits inside that failure band. Chunking changes throughput, not
> results, and `code/eigh_fallback.py` counts every CPU LAPACK retry into the config so
> a silent numerical detour cannot go unrecorded.
>
> **The reported checkpoint is chosen on validation**, not the final step — step
> {sel_txt}. There is **no learning-rate scheduler anywhere in the codebase**, so this
> is bitwise identical to having stopped training there.{hp_warn}

---

## Metrics A1-A34 + B, mean ± std across 5 seeds

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
> **A20** *(native d=8)*: error on the full terminal covariance matrix. This is the row that matters most here: the correction network emits a full `(d, d)` matrix and the spectral clip acts on eigenvalues, so A20 is the direct test of whether the d(d−1)/2 spot correlations Σˢ survived the control. A diagonal control would have no mechanism to get this row right. It is also where the two references can be told apart — the SBTS kernel average inherits the bank's cross-asset structure directly rather than through fitted parameters. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. Note what carries the volatility here: the reference σ is a **constant** per-asset scale (the SBTS normalisation), so unlike the sibling there is no fitted volatility state in the reference at all. Everything time-varying in the generated volatility comes from the learned control and from the kernel's path-dependent drift. These rows therefore test the control much more directly than the sibling's do.

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
> **Cross-seed stability**: the seed here controls **three** independent things — the control network's initialisation, every Euler–Maruyama increment drawn during training, and the batch resampling that re-fits the Z-proxy at each outer iteration. The reference is *not* among them, and for a stronger reason than in the sibling: there is nothing to estimate. The kernel has no fitted parameters, and the bank it averages over is the full train split, identical for all 5 seeds. So none of the spread below comes from the reference — what remains is optimisation plus sampling variance, and [`plots/loss_convergence.png`](plots/loss_convergence.png) shows whether any seed diverged. Read that figure before this table: `code/SWEEP.md` records a measured run-to-run noise floor, and a seed spread of the same order is the sampler, not the model. Generation uses a **separate** seed stream (90000 + 1000·seed + chunk) that never reuses a training seed.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs {method}, seed 0, asset 0)

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

## {method} Training Loss (5 seeds)

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
`losses/seed_*_losses.csv` (**{train_min:.0f} min total** across the 5 seeds, one GPU and
{nw} threads per seed process, run 2-up in waves):

{train_tbl}

Generation wall-clock — the selected control rolled forward {euler_steps}
Euler–Maruyama steps from $x_0 = \\log(100)$ for each of {gen_n} paths, {nw} threads,
**{total_min:.1f} min total** across the 5 seeds:

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

![Predictive Score Loss](plots/pred_score_loss.png){memo}

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
edits** — every d = 8 adaptation is a method-local file in `code/`.

There is **no `code/reference/` directory**, and its absence is the structural signature of
this variant. The sibling has one because its reference is parametric and had to be fitted:
`reference_kernel.json` is the fit. Here the reference is nonparametric and there is nothing
to store — the kernel is rebuilt at load time from four numbers (`h`, `markov_order`, `npi`,
`weight_grad_mode`, all tracked in `weights/seed_*_config.json`) plus the train-split price
bank, which lives in `dataset/HestonMultiAsset/` and is shared by every method on this
dataset. `code/sbts_reference.py` is therefore the whole reference: code, not data.

The `.npy` arrays are **gitignored**: `(8192, {seq_len}, 8)` float64 = 132 MB each, over
GitHub's 100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from
the tracked code, the tracked dataset and the tracked hyperparameters; the `metadata.json`
beside each array **is** tracked, so shapes, price ranges, reference knobs and generation
times stay auditable without the payload.

## Reproduce

Steps 5 onwards are exactly what `code/run_pipeline.sh` runs, in that order and with
the same hard gates. Run the script rather than these lines if you want the gates;
they are spelled out here so the chain is readable without opening it.

```bash
cd /home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$R/src:$R/experiments"

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. NO reference-fitting step. The SBTS kernel is nonparametric: it is rebuilt from
#    (h, markov_order, npi, weight_grad_mode) plus the TRAIN price bank at load time.
#    Sanity-check it against a brute-force reimplementation before trusting it:
cd results/HestonMultiAsset/{method}/code
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 $PY validate_sbts_reference.py

# 3. hyperparameter search — greedy, one knob at a time, scored on VALIDATION.
#    Writes sweep/*.json + sweep/incumbent.json; render_sweep.py turns them into SWEEP.md.
#    Measure the run-to-run noise floor FIRST: gaps smaller than it are not results.
bash run_stage_lr_a.sh          # ... and the other per-stage launchers, in order
$PY render_sweep.py

# 4. final 5 seeds, {steps} outer iterations, 2 GPUs, 2 concurrent chains
#    (~{train_min:.0f} min of GPU time). Seeds 0/2/4 on GPU 0, seeds 1/3 on GPU 1.
for s in 0 2 4; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 $PY train_multiasset.py \\
      --seed $s --steps {steps} --lr {lr:g} --ridge-lambda {ridge:g} --h {ref_h} \\
      --markov-order {ref_K} --npi {ref_npi} --weight-grad-mode {ref_grad} \
      --jacobian-lags {ref_jl} --device cuda:0
done &
for s in 1 3; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 $PY train_multiasset.py \\
      --seed $s --steps {steps} --lr {lr:g} --ridge-lambda {ridge:g} --h {ref_h} \\
      --markov-order {ref_K} --npi {ref_npi} --weight-grad-mode {ref_grad} \
      --jacobian-lags {ref_jl} --device cuda:0
done &
wait

# 5. choose the reported checkpoint per seed on VALIDATION, never on test
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \\
    $PY select_checkpoint_multiasset.py --seeds 0 1 2 3 4

# 6. generate the 8192-path bank from the SELECTED checkpoint
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \\
    $PY run_all_multiasset.py --seeds 0 1 2 3 4
$PY collect_artifacts.py            # must print "5 rows" and exit 0 before anything downstream
cd /home/tbasseras/benchmark

# 7. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 8. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=0 $PY metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=0 $PY metrics/compute_all_multiasset.py --method {method} \\
    --dataset HestonMultiAsset --seed-list 0,1,2,3,4

# 9. figures
$PY results/HestonMultiAsset/{method}/code/plot_losses.py
$PY results/HestonMultiAsset/{method}/code/plot_diagnostics_multiasset.py \\
    --method {method}
$PY metrics/plot_score_losses.py --method {method} --dataset HestonMultiAsset

# 10. memorisation diagnostic
$PY results/HestonMultiAsset/{method}/code/measure_memorisation.py --seeds 0,1,2,3,4

# 11. regenerate this README from the artefacts
$PY results/HestonMultiAsset/{method}/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
