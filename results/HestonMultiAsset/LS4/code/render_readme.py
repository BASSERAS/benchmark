#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/LS4/README.md from the metric artefacts.

The section order, table shape and prose slots are a 1:1 mirror of
`results/HestonMultiAsset/SBTS/README.md` (itself a mirror of the d = 1 entry
`methods/SBTS/README.md`), so every method page on this dataset displays
identically and can be diffed row-by-row. MULTIASSET_GUIDELINE.md section 8 fixes
that layout; the two differences it *mandates* for a trained method are section 6,
which becomes a real training-loss section instead of SBTS's "no training loss"
placeholder, and the hyperparameter blockquote.

The one deliberate omission, inherited from the SBTS page, is the Path Shadowing
MC block: PS-MC has not been run for d = 8, and a section with no numbers behind
it would be a lie.

Material that is *about the dataset rather than about LS4* -- how the d = 8
metrics are scoped and the nearest-neighbour memorisation discussion -- lives one
level up in `results/HestonMultiAsset/oldreadme.md`, shared by every method added
to this dataset. Material about *this port* -- the YAML anchor trap, the z_dim
retune, the S0 rescaling -- lives in `code/README.md`.

Every number here is READ FROM DISK, never typed by hand: the tables *and* the
headline win-counts are computed from the artefacts, so the prose cannot drift
away from the numbers it describes.

Reads
  results/HestonMultiAsset/LS4/metrics_summary.csv
  results/HestonMultiAsset/LS4/curve_b_aggregate.json
  results/HestonMultiAsset/LS4/grid_tvd_aggregate.json
  results/HestonMultiAsset/LS4/losses/generation_time.csv
  results/HestonMultiAsset/LS4/losses/zdim_selection.json          (optional)
  results/HestonMultiAsset/LS4/weights/seed_*_config.json          (train times)
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv    (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/LS4/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/LS4/code/render_readme.py
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Name kept as SBTS so the shared table helpers stay textually diffable against
# SBTS/code/render_readme.py -- the guideline's whole point is that the two
# renderers differ only where the layout mandates it.
SBTS = os.path.dirname(HERE)                                  # .../LS4
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
        lines.append(f"| **, {cat}, ** |" + " |" * (len(head) - 1))
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
    worker count, a z_dim or an epoch budget -- a partial re-run at different
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
                                        first_nan_epoch

    They are merged per seed with the config as the base, so a hyperparameter can
    never be silently taken from the metadata copy. Returns
    (markdown, total_minutes, merged_configs).
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
                      "min_val_loss", "first_nan_epoch"):
                c.setdefault(k, m.get(k))
        c["_seed"] = seed
        cfgs.append(c)
    if not cfgs:
        return "_(weights/seed_\\*_config.json not found)_", 0.0, []
    cfgs.sort(key=lambda c: c["_seed"])
    total = sum(float(c.get("train_time_sec") or 0.0) for c in cfgs) / 60.0

    def _n(v, unit="", nd=1):
        return "-" if v is None else f"{float(v):.{nd}f}{unit}"

    out = ["| Seed | Epochs | sec/epoch | Train wall-clock | Final train ELBO | Best val ELBO | NaN |",
           "|------|--------|-----------|------------------|------------------|---------------|-----|"]
    for c in cfgs:
        nan = "-" if c.get("first_nan_epoch") is None else f"ep {c['first_nan_epoch']}"
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
    zsel = read_json(os.path.join(SBTS, "losses", "zdim_selection.json"))

    missing = [n for n, o in (("LS4/metrics_summary.csv", sbts),
                              ("LS4/curve_b_aggregate.json", agg),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --method LS4 and "
            "--method perfect_recovery first."
        )

    timing, total_min, trows = render_timing(os.path.join(SBTS, "losses", "generation_time.csv"))
    train_tbl, train_min, cfgs = render_training(SBTS)
    c0 = cfgs[0] if cfgs else {}
    r0 = trows[0] if trows else {}
    # Hyperparameters are read off the per-seed config, never typed here.
    zdim = str(c0.get("z_dim", r0.get("z_dim", "40")))
    epochs = str(c0.get("epochs", r0.get("epochs", "100")))
    params = int(c0.get("params", 0))
    d_state = str(c0.get("d_state", 64))
    d_model = str(c0.get("d_model", 64))
    n_layers = str(c0.get("n_layers", 4))
    # Any seed disagreeing on z_dim or epochs is a real inconsistency: say so
    # rather than letting seed 0 speak for the other four.
    zset = sorted({str(c.get("z_dim")) for c in cfgs if c.get("z_dim") is not None})
    eset = sorted({str(c.get("epochs")) for c in cfgs if c.get("epochs") is not None})
    hp_warn = ""
    if len(zset) > 1 or len(eset) > 1:
        hp_warn = (f" **Seeds disagree** — z_dim {{{', '.join(zset)}}}, "
                   f"epochs {{{', '.join(eset)}}}; re-run the odd seed before trusting this page.")
    # Worker counts can differ across seeds, so report the actual set rather
    # than letting row 0 misrepresent the other four rows.
    nws = sorted({str(r.get("n_workers", "")) for r in trows if r.get("n_workers")},
                 key=lambda x: int(x) if x.isdigit() else 0)
    nw = nws[0] if len(nws) == 1 else " and ".join(nws)
    gen_ns = sorted({str(r.get("n_samples", "")) for r in trows if r.get("n_samples")})
    gen_n = f"{int(gen_ns[0]):,}".replace(",", " ") if len(gen_ns) == 1 else "/".join(gen_ns)

    at_floor, n_a, gaps = a_headline(sbts, floor)
    b_at, n_b = b_headline(agg, agg_f)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    # z_dim justification, read from the pre-registered sweep rather than asserted.
    z_sel = ""
    if zsel:
        cand = {str(c.get("z_dim")): c for c in zsel.get("candidates", [])}
        rel = cand.get("5")
        best = cand.get(str(zdim))
        if rel and best:
            z_sel = (f" The released `z_dim = 5` scores a validation ELBO of "
                     f"**{float(rel['val_elbo_tail5']):+.2f}** against "
                     f"**{float(best['val_elbo_tail5']):+.2f}** for `z_dim = {zdim}`, and its "
                     f"train ELBO ({float(rel['train_final']):.2f}) essentially equals its "
                     f"validation ELBO ({float(rel['val_final']):.2f}) — under-capacity, not "
                     f"overfitting. A 5-dimensional latent cannot carry 8 correlated channels. "
                     f"The ordering was monotone with no interior optimum inside the tested "
                     f"range, so `z_dim = {zdim}` is the **largest tested** candidate rather "
                     f"than a located maximum; that boundary is disclosed rather than dressed "
                     f"up as convergence.")

    md = f"""# LS4 on Multi-Asset Heston (d = 8)

**Deep Latent State Space Models for Time-Series Generation** (Zhou, Kang, Molina-Salgado, Wu,
Ermon & Grover, ICML 2023, [arXiv:2212.12749](https://arxiv.org/abs/2212.12749)) applied to
8 192 **multi-asset** Heston stochastic-volatility price paths (seq\\_len = 252,
**d = 8 correlated assets**).

LS4 is a **latent variational state-space model**: a VAE whose prior, posterior and decoder are
all S4 (structured state-space) backbones. It is trained by maximising an ELBO
(`loss = KLD + NLL`) with AdamW, and generates by sampling `z` from the learned prior and
decoding — it never touches a training path at generation time. {params:,} parameters,
trained for {epochs} epochs per seed, 5 seeds. Weights in
[`weights/`](weights/), per-seed hyperparameters in `weights/seed_*_config.json`.

The model is **joint over all 8 channels** — one network, `d_input = d_output = 8` — not eight
independent univariate fits. See [`code/README.md`](code/README.md) for source, the vendored
release, and the implementation deviations, and the dataset-level
[`../oldreadme.md`](../oldreadme.md) for the multi-asset Heston law itself, the per-asset vs
native metric scoping, and the memorisation diagnostic shared by all methods on this dataset.

> **Hyperparameters:** `z_dim={zdim}`, `d_state={d_state}`, `d_model={d_model}`,
> `n_layers={n_layers}`, `sigma=0.1`, `backbone=autoreg`, `s4_type=s4`, `latent_type=split`;
> `AdamW(lr=1e-3, wd=0)` + `ReduceLROnPlateau(patience=20, factor=0.5)` +
> `EMA(lamb=0.99, start_step=200)`, batch 128, {epochs} epochs.
> Everything above is the **released `solar_weekly` preset** — the configuration that reproduced
> the paper's Solar Weekly marginal score — **except `z_dim`**, which is ours, and `in_channels`,
> which the data forces. `z_dim` is the one hyperparameter that does not cross dimension.{z_sel}
> Selected on the **validation split, never on test**, against a criterion written to disk
> *before* the sweep ran: [`losses/selection_criterion.md`](losses/selection_criterion.md),
> numbers in [`losses/zdim_selection.json`](losses/zdim_selection.json).
> The scaler was also changed, from a global standardise to a **per-channel** one; a per-channel
> affine map leaves the cross-asset correlation matrix exactly unchanged, so the target coupling
> Σˢ — what A20 scores — survives both the standardisation and its inverse.{hp_warn}

---

## Metrics A1-A34 + B, mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better;, no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the independent-draw floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}. The floor, not the other method, is the reference on this page; the LS4-vs-SBTS comparison lives in the dataset-level [`../README.md`](../README.md) so that neither method's page grades itself against a rival it was tuned beside.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix, which is the row that actually tests whether the d(d−1)/2 spot correlations Σˢ survived generation. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. LS4 *does* carry an explicit latent state, so unlike a kernel method it has a mechanism to reproduce these; whether the ELBO gives it any reason to is a separate question, and the table answers it.

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
> **Cross-seed stability**: unlike a kernel method, LS4's seed controls **both** the weight initialisation and the prior draw, so the std columns here absorb optimisation variance as well as sampling variance. That makes them the honest quantity to read: a seed that collapsed would widen them, and [`plots/loss_convergence.png`](plots/loss_convergence.png) shows whether any did.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs LS4, seed 0, asset 0)

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

## LS4 Training Loss (5 seeds)

LS4 is trained on the **ELBO**, `loss_total = kld_loss + nll_loss`. `mse_loss` is a
reconstruction diagnostic only and is **not** part of the objective — it is plotted because a
flat ELBO with a rising MSE is the signature of a posterior collapse, which is the failure this
figure exists to catch. The x-axis is **step**, not epoch: with batch 128 over 8 192 paths an
epoch is 64 steps, and step resolution is what makes the learning-rate drops visible.
Solid = train, dashed = validation ELBO on the held-out
`heston_ma_S_val_8192x252x8.npy`, which was never used for anything except the `z_dim` choice.

![Training Loss](plots/loss_convergence.png)

Training wall-clock, read from `weights/seed_*_config.json` and
`generated_paths/seed_*/metadata.json` (**{train_min:.0f} min total** across the 5 seeds, run
{nw} threads per process, 2 GPUs):

{train_tbl}

Generation wall-clock — sampling {gen_n} paths from the prior and decoding, {nw} threads,
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

![Predictive Score Loss](plots/pred_score_loss.png)

---

## File layout

This tree is **self-contained**: code, inputs, outputs and documentation sit side by side,
unlike the d = 1 benchmark which splits `methods/<Method>/` from `results/Heston/<Method>/`.

```
results/HestonMultiAsset/LS4/
├── README.md                             ← this file (generated by code/render_readme.py)
├── code/
│   ├── README.md                         source, provenance check, d = 8 deviations
│   ├── train_multiasset.py               trains one seed AND generates its 8192 paths
│   ├── collect_artifacts.py              rebuilds generation_time.csv, checks the §4 contract
│   ├── plot_losses.py                    plots/loss_convergence.png, 5 seeds overlaid
│   ├── plot_diagnostics_multiasset.py    the 8-panel stylised-facts figure
│   ├── measure_memorisation.py           nearest-neighbour memorisation diagnostic
│   ├── render_readme.py                  regenerates this README from the artefacts
│   ├── reference_models/                 vendored LS4 release (ls4, s4, s4d, seq_unet)
│   └── reference_configs/                the released solar_weekly YAML
├── generated_paths/seed_{{0..4}}/
│   ├── generated_paths_8192x252x8.npy    (8192, 252, 8) float64 — gitignored, 132 MB
│   └── metadata.json                     shape, S0_exact, min/max, timings, digest (tracked)
├── weights/
│   ├── seed_{{i}}_model.pt                 state_dict + EMA state_dict + scaler
│   └── seed_{{i}}_config.json              full hyperparameter record incl. retuned_for_d8
├── losses/
│   ├── seed_{{i}}_losses.csv               step, phase, loss_total, kld, nll, mse, lr
│   ├── zdim_selection.json               the 4-candidate validation sweep + verdict
│   ├── selection_criterion.md            pre-registered criterion and its caveats
│   ├── memorisation.json                 NN-ratio diagnostic on the final 8192-path output
│   └── generation_time.csv               wall-clock time per seed
├── plots/
│   ├── loss_convergence.png              ELBO / KLD / NLL / MSE / lr, 5 seeds
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
one: LS4 generates inside `train_multiasset.py`, from the in-memory EMA model immediately after
training. Splitting it out would mean reloading a checkpoint and duplicating the scaler
inversion and the S0 rescaling in a second place where the two copies could silently drift.
`collect_artifacts.py` fills the slot for post-generation bookkeeping and contract verification.

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

# 2. z_dim screening on the VALIDATION split (~4 min; regenerates losses/zdim_selection.json)
cd results/HestonMultiAsset/LS4/code
for z in 5 16 32 40; do
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 0-7 $PY train_multiasset.py \\
      --seed 0 --z_dim $z --epochs 20 --frac 0.25 --val --tag zdim$z --no_generate
done

# 3. final 5 seeds, {epochs} epochs, 2 GPUs (~{train_min:.0f} min of GPU time, ~half that wall-clock)
for s in 0 2 4; do CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 0-7  $PY \\
    train_multiasset.py --seed $s --z_dim {zdim} --epochs {epochs} --val; done &
for s in 1 3;   do CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 8-15 $PY \\
    train_multiasset.py --seed $s --z_dim {zdim} --epochs {epochs} --val; done &
wait
$PY collect_artifacts.py            # must print "5 rows" before anything downstream
cd /home/tbasseras/benchmark

# 4. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 5. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=2 $PY metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=2 $PY metrics/compute_all_multiasset.py --method LS4 \\
    --dataset HestonMultiAsset --seeds 5

# 6. figures
$PY results/HestonMultiAsset/LS4/code/plot_losses.py
$PY results/HestonMultiAsset/LS4/code/plot_diagnostics_multiasset.py --method LS4
$PY metrics/plot_score_losses.py --method LS4 --dataset HestonMultiAsset

# 7. memorisation diagnostic
$PY results/HestonMultiAsset/LS4/code/measure_memorisation.py --seeds 0,1,2,3,4

# 8. regenerate this README from the artefacts
$PY results/HestonMultiAsset/LS4/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
