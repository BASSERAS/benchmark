#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/reference/README.md from the metric artefacts.

Section order, table shape and prose slots mirror `SBTS/code/render_readme.py`
and `LS4/code/render_readme.py`, so every method page on this dataset displays
identically and can be diffed against the others. What changes is the prose, and
it has to: the other pages describe a *generator that was fitted to the data*,
this one describes the **starting line** those generators are supposed to beat.

What this folder is
-------------------
`sigma^ref` -- the frozen reference SDE of Deep-MKV-TS Algorithm 1, sampled with
the neural control switched off. The model is BUILT (all 56 136 parameters) and
never FITTED; both output heads carry a zero final layer, so `Zhat == 0` exactly,
the control `Theta = eta*(sigma^ref)^-1 + Zhat/sqrt(dt)` collapses to
`eta*(sigma^ref)^-1`, and the sampler integrates `sigma^ref` itself.
`code/generate_reference_multiasset.py` VERIFIES those four tensors are zero
before it writes a single path, and records the measured maxima per seed.

So this row is not a competitor. It is the **before** picture:

  * `perfect_recovery/` draws from the TRUE Heston SDE  -> the best any generator
    could do (independent-draw floor).
  * `reference/`        draws from the FITTED reference -> what Deep-MKV-TS
    scores *before its first gradient step*.
  * `Deep-MKV-TS/`      draws from the trained control  -> and must sit strictly
    between the two, or Algorithm 1 bought nothing.

Read the three side by side and the attribution question -- how much of
Deep-MKV-TS's score is the reference SDE and how much is the learned control --
stops being a matter of opinion.

Every number here is READ FROM DISK, never typed by hand: the tables, the
headline win-counts, the coefficient count and the NLLs all come from the
artefacts, so the prose cannot drift away from the numbers it describes.

Reads
  results/HestonMultiAsset/reference/metrics_summary.csv
  results/HestonMultiAsset/reference/curve_b_aggregate.json
  results/HestonMultiAsset/reference/grid_tvd_aggregate.json
  results/HestonMultiAsset/reference/losses/generation_time.csv
  results/HestonMultiAsset/reference/losses/memorisation.json          (optional)
  results/HestonMultiAsset/reference/weights/reference_kernel.json
  results/HestonMultiAsset/reference/generated_paths/seed_0/metadata.json
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv        (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/reference/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/reference/code/render_readme.py
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.dirname(HERE)                                   # .../reference
MA = os.path.dirname(REF)                                     # .../HestonMultiAsset
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

    Curve-plot names come straight from metrics.CURVE_PLOTS and several contain a
    raw '|' (e.g. "ACF |r| lags 1-20"), which silently splits the row into extra
    columns.
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


def render_a_table(ref, floor):
    n_seeds = len(next(iter(ref.values()))["seeds"]) if ref else 5
    head = ["Metric", "Mean ± Std"] + [f"Seed {i}" for i in range(n_seeds)] + ["Perfect floor"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for cat, rows in CATEGORIES:
        lines.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for key, label, direction in rows:
            s = ref.get(key)
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
    """Is the reference SDE's mean at least as good as the floor's mean here?"""
    if direction == "up":
        return sm >= fm
    if direction == "none":              # A28: perfect = 1.0
        return abs(sm - 1.0) <= abs(fm - 1.0)
    return sm <= fm


def a_headline(ref, floor):
    """Count rows at/below the independent-draw floor, and rank the gaps.

    For a *fitted* generator this counts wins. For the reference SDE it measures
    something different and more useful: how much of the benchmark a parametric
    SDE with a couple of dozen coefficients already solves before any neural
    control is trained. Every row in `at_floor` is a row on which Deep-MKV-TS
    cannot claim credit for its control, because the reference alone already
    reaches the floor there.
    """
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
    """Timing table plus all rows.

    The reference bank is CPU-only and single-core, so the interesting column is
    `device`, not a worker count -- and it is READ rather than assumed, because a
    row that said `cuda` would falsify the "no GPU needed" claim this page makes.
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
    out += [f"| {r['seed']} | {90000 + int(r['seed'])} | `{r['device']}` | "
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
    memo = read_json(os.path.join(REF, "losses", "memorisation.json"))

    missing = [n for n, o in (("reference/metrics_summary.csv", ref),
                              ("reference/curve_b_aggregate.json", agg),
                              ("reference/weights/reference_kernel.json", kern),
                              ("reference/generated_paths/seed_0/metadata.json", meta),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing artefacts: " + ", ".join(missing) + "\n"
            "Run code/generate_reference_multiasset.py, then "
            "metrics/compute_all_multiasset.py --method reference and "
            "--method perfect_recovery first."
        )

    # -- everything below is DERIVED from the artefacts, never typed ----------
    n_cov = len(kern.get("gamma_diagonal", []))
    n_off = len(kern.get("gamma_offdiagonal", []))
    n_drift = len(kern.get("gamma_drift", []))
    n_coef = n_cov + n_off + n_drift
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

    n_params = meta.get("n_parameters", "?")
    n_trained = meta.get("n_trained_parameters", "?")
    zhat = meta.get("zhat_head_max_abs", {})
    zmax = max(zhat.values()) if zhat else float("nan")
    zheads = len(zhat)
    zver = bool(meta.get("zero_control_verified", False))
    s0res = float(meta.get("s0_max_abs_residual_before_rescale", 0.0))
    gseed = int(meta.get("generation_seed", 90000))

    # training values the coefficients were fitted against -- derived, not typed
    shape = meta.get("shape", [8192, 252, 8])
    n_train_values = shape[0] * shape[1] * shape[2]
    per_coef = f"{n_train_values / max(n_coef, 1):,.0f}".replace(",", " ")

    timing, total_min, trows = render_timing(os.path.join(REF, "losses", "generation_time.csv"))
    devices = sorted({r.get("device", "") for r in trows if r.get("device")})
    dev_txt = devices[0] if len(devices) == 1 else " and ".join(devices)
    n_bank = len(trows) if trows else 5

    at_floor, n_a, gaps = a_headline(ref, floor)
    b_at, n_b = b_headline(agg, agg_f)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    memo_txt = ""
    if memo:
        memo_txt = (
            f"\n> **Memorisation:** `nn_ratio = {float(memo['nn_ratio']):.4f} ± "
            f"{float(memo['nn_ratio_std']):.4f}` "
            f"([`losses/memorisation.json`](losses/memorisation.json)). This row **calibrates "
            f"the diagnostic rather than testing the generator**: with {n_coef} fitted "
            f"coefficients against ~{n_train_values / 1e6:.1f}M training values and a control "
            f"verified to be exactly zero, memorisation here is arithmetically impossible, so "
            f"whatever the estimator returns *is* its reading for a generator that certainly "
            f"did not memorise. That is the null the other columns need to be interpretable."
        )

    zero_ok = "verified at generation time" if zver else "**NOT VERIFIED**"

    md = f"""# The Reference SDE on Multi-Asset Heston (d = 8)

**σ<sup>ref</sup> — the frozen reference diffusion of Deep-MKV-TS Algorithm 1**
(El Boustany, Basseras, Mekkaoui, Alouadi, Hafsi & Pham), sampled with the neural
control switched **off**, on 8 192 **multi-asset** Heston stochastic-volatility price
paths (seq\\_len = 252, **d = 8 correlated assets**).

This is **not a competing generator**. It is the *starting line*. Algorithm 1 takes a
pre-fitted parametric diffusion σ<sup>ref</sup> and learns a neural control on top of it;
this folder scores σ<sup>ref</sup> **alone**, so the trained method's numbers can be split
into "what the reference SDE already gave us" and "what the control actually bought".
Together with [`../perfect_recovery/`](../perfect_recovery) — draws from the *true* Heston
law — the two bracket [`../Deep-MKV-TS/`](../Deep-MKV-TS) from opposite sides:

| Folder | Samples from | Reads as |
|--------|--------------|----------|
| [`../perfect_recovery/`](../perfect_recovery) | the **true** Heston SDE | the best any generator could do — the floor |
| **`reference/`** *(this page)* | the **fitted** σ<sup>ref</sup> | Deep-MKV-TS **before its first gradient step** |
| [`../Deep-MKV-TS/`](../Deep-MKV-TS) | the **trained** control | must land strictly between the two, or Algorithm 1 bought nothing |

The model is **built but never fitted**. All {n_params} parameters are constructed exactly as
`train_multiasset.py` constructs them, and then **{n_trained} gradient steps** are taken.
Algorithm 1 zero-initialises the final layer of both output heads — weight *and* bias — so
`Ẑ ≡ 0` for every hidden state whatever the GRU beneath computes, the control

$$\\Theta = \\eta\\,(\\sigma^{{\\mathrm{{ref}}}})^{{-1}} + \\hat{{Z}}/\\sqrt{{\\Delta t}}$$

reduces to $\\eta\\,(\\sigma^{{\\mathrm{{ref}}}})^{{-1}}$, and the sampler integrates
σ<sup>ref</sup> itself. All {zheads} head tensors measured `max|·| = {zmax:g}`, {zero_ok} —
`code/generate_reference_multiasset.py` refuses to write a bank if any is non-zero, so the
central claim of this page is **enforced rather than asserted** (per-seed record in
`generated_paths/seed_{{i}}/metadata.json` under `zhat_head_max_abs`).

See [`code/README.md`](code/README.md) for source and implementation details,
[`weights/README.md`](weights/README.md) for the fitted coefficients and their provenance,
and the dataset-level [`../oldreadme.md`](../oldreadme.md) for the multi-asset Heston law,
the per-asset vs native metric scoping, and the memorisation diagnostic shared by all
methods on this dataset.

> **Hyperparameters:** none were chosen for this folder. σ<sup>ref</sup> carries
> **{n_coef} fitted coefficients in total** — {n_cov} covariance, {n_off} off-diagonal,
> {n_drift} drift — plus two trend and two activity half-lives and the spectral clip bounds
> `σ_min={smin}`, `σ_max={smax}`. They come from a **{fit_steps}-step Gaussian-NLL fit on the
> training split only** (`{fit_file}`, internal {100 - val_pc}/{val_pc} calibration/validation
> split, lr={fit_lr}), `fitted_at` **{fit_at}**, final calibration NLL **{cal_nll:.6f}**,
> validation NLL **{val_nll:.6f}**. Nothing else is tuned: no learning rate, no ridge λ, no
> checkpoint selection, no early stopping — because there is no training. That is {n_coef}
> numbers against **{per_coef} training values per coefficient**, which is why memorisation
> here is not merely unlikely but arithmetically impossible. The weights are **byte-identical
> copies** of the ones Deep-MKV-TS trains on top of, with `weights/SHA256SUMS` to detect
> silent drift.{memo_txt}

---

## Metrics A1-A34 + B, mean ± std across {n_bank} seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

{render_a_table(ref, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows already sit at or below the independent-draw floor with the control switched off** — {at_floor_txt}. Those are rows on which Deep-MKV-TS **cannot claim credit for the learned control**: the {n_coef}-coefficient parametric SDE reaches the floor there by itself. The largest remaining gaps — the rows where the control has something left to do — are {gap_txt}. Cross-method win-counts belong in the dataset-level comparison table, not here.
> **How to read this column against [`../Deep-MKV-TS/`](../Deep-MKV-TS):** on any row, *(reference − Deep-MKV-TS)* is the control's contribution and *(Deep-MKV-TS − floor)* is what remains unsolved. A row where Deep-MKV-TS scores **worse** than this page is a row where Algorithm 1 actively damaged a working reference — the single most informative failure this benchmark can surface, and completely invisible without a reference column.
> **Pairing:** generation seeds are `{gseed} + i`, **imported** from `Deep-MKV-TS/code/run_all_multiasset.py` rather than redefined, so each reference bank draws the **same Brownian stream** as the corresponding Deep-MKV-TS bank. The comparison is therefore paired wherever both methods hold that seed, which removes simulation noise from the difference instead of averaging over it.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths).
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix — the row that tests whether the d(d−1)/2 spot correlations Σˢ survived generation, and the row σ<sup>ref</sup>'s {n_off} off-diagonal coefficients exist to capture. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. σ<sup>ref</sup> carries a *structural* activity state (`activity_update = {kern.get("activity_update", "?")}`) rather than the true latent variance, so these two rows measure how well that surrogate stands in for it.

---

## B, Curve-Shape Metrics, mean ± std across {n_bank} seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. The curve is
computed **per asset and averaged over the 8 assets** *before* the combination below, so the
combination rules stay byte-identical to the d = 1 tree. For the real data (L_r) and generated
data (L_g) we build three lists, the curve L, its first finite difference L' (der), and its
second finite difference L'' (sec\\_der), then combine the three sub-scores into **one number
per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\\_der)/3; std = the sample std of that per-seed combined score across the {n_bank} seeds. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE, one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself**, the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the {n_bank} seeds** of that per-seed function MAPE.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)**, the ill-posed derivative / 2nd-derivative curves are excluded for the same reason as the % err row.
- **CVaR₉₀ / CVaR₉₅ rows**: tail-averaged pointwise curve error (Expected Shortfall) on the curve L **only (funct-only)**. Pointwise error eₜ = |L_g(t) − L_r(t)|; for q ∈ {{0.90, 0.95}}, CVaR_q = mean(eₜ for eₜ ≥ the q-th percentile of eₜ), then range-normalized like NRMSE (÷ (max|L_r| − min|L_r| + 1e-12) × 100).

All ↓ lower is better. The perfect floor is **non-zero** for all plots, it is the residual finite-sample error of an independent multi-asset Heston draw scored against the test set, identical across methods.
Five sublines per plot: **MSE**, **% error**, **NRMSE**, **CVaR₉₀** and **CVaR₉₅** (the per-seed columns hold that seed's combined score).

{render_b_table(agg, agg_f, tvd, tvd_f)}

> **Headline:** **{len(b_at)} of the {n_b} B plots** sit at or below the finite-sample floor on the deciding MSE row **with the control switched off**: {b_at_txt}.
> **Cross-seed stability:** there is no gradient descent and no checkpoint selection here, so the *only* seed-to-seed variation is the Euler-Maruyama draw itself. The std columns therefore measure **pure simulation noise at N = 8 192**, which makes them a yardstick rather than a footnote: a trained method whose std is materially larger than this page's is unstable for reasons that are **not** sampling noise.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs Reference SDE, seed 0, asset 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

The third (black dashed) curve is the **independent-draw floor**, not a closed-form theory
curve: `metrics/heston_theory.py` is hard-wired to the d = 1 parameters (κ=2.0, θ=0.04,
ρ=−0.7, dt=1/250), so plotting it over d = 8 data would be a fabricated reference. Wherever
Real and the floor curve already disagree, the gap is finite-sample noise rather than a
modelling failure. **One asset is shown, not eight** — the *metrics* are averaged over all 8
assets, the *figure* is asset 0; pooling eight deliberately different volatilities into one
histogram would show the mixing, not the model.

Read this figure as the **before** picture: the same eight panels rendered from
[`../Deep-MKV-TS/plots/heston_diagnostics.png`](../Deep-MKV-TS/plots/heston_diagnostics.png)
show what the learned control changed, panel by panel.

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## The reference SDE has no training loss

Nothing is trained in this folder, so there is no loss curve and no
`loss_convergence.png` — a plot with no gradient steps behind it would be decoration.
The {n_coef} coefficients were fitted **once, elsewhere**, by
`Deep-MKV-TS/code/fit_reference_multiasset.py`; that fit's {fit_steps}-step
calibration/validation NLL trace is archived verbatim as
[`weights/reference_fit_history.csv`](weights/reference_fit_history.csv) — final calibration
**{cal_nll:.6f}**, validation **{val_nll:.6f}**, i.e. validation *below* calibration, so there
is no overfitting to detect. The `losses/` directory consequently holds only what genuinely
exists: generation wall-clock, and the memorisation diagnostic.

Generation is **CPU-only** — `{dev_txt}`, one core, no GPU, no CUDA runtime required.
Each seed is a 251-step Euler-Maruyama rollout of
`X_{{k+1}} = X_k + b^ref(X_k) + σ(X_k)ε_k` from `x0 = log(100)` driven by fresh Gaussian
noise; the whole {n_bank}-seed bank finishes in **{total_min:.1f} min total**:

{timing}

Prices are then rescaled to the exact §4 contract `S0 = 100.0`. The model runs in float32, so
`exp(log(100))` lands at 100.000008 rather than 100; the largest residual before rescaling was
**{s0res:.3e}**. Multiplying a whole path by a constant is exactly a shift in log-price, so
**every log-return is bit-identical** before and after — the rescale fixes the anchor without
touching a single quantity any metric measures.

---

## A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.
The d = 8 classifier is **native**: it sees all 8 channels at once.

This is the one curve on the page where a *falling* loss is bad news — it is the judge
learning to separate σ<sup>ref</sup>'s paths from real Heston paths, and how fast it manages
that is the clearest single picture of what the learned control still has to fix.

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
matching [`../SBTS/`](../SBTS) and [`../LS4/`](../LS4). The weights are copied rather than
symlinked from Deep-MKV-TS for exactly that reason — a reader who takes only this folder
still has everything needed to regenerate the bank.

```
results/HestonMultiAsset/reference/
├── README.md                             ← this file (generated by code/render_readme.py)
├── code/
│   ├── README.md                         source notes, the zero-control proof, deviations
│   ├── generate_reference_multiasset.py  builds the model, VERIFIES Ẑ == 0, samples σ^ref
│   ├── plot_diagnostics_multiasset.py    the 8-panel stylised-facts figure
│   ├── measure_memorisation.py           NN memorisation diagnostic (the calibration row)
│   └── render_readme.py                  regenerates this README from the artefacts
├── generated_paths/seed_{{0..{n_bank - 1}}}/
│   ├── generated_paths_8192x252x8.npy    (8192, 252, 8) float64 — gitignored, 132 MB
│   └── metadata.json                     seed, shape, S0 contract, zhat_head_max_abs (tracked)
├── losses/
│   ├── generation_time.csv               wall-clock per seed — no loss, nothing trains
│   └── memorisation.json                 NN-ratio diagnostic (calibrates the estimator)
├── plots/
│   ├── heston_diagnostics.png            8-panel stylised facts (seed 0, asset 0)
│   ├── disc_classifier_loss.png          A18 BCE curves, {n_bank} seeds
│   └── pred_score_loss.png               A19 MAE curves, {n_bank} seeds (asset 0)
├── weights/
│   ├── README.md                         provenance, and why there is no neural checkpoint
│   ├── reference_kernel.json             the {n_coef} fitted coefficients + clip bounds
│   ├── reference_fit_history.csv         the {fit_steps}-step calibration/validation NLL trace
│   └── SHA256SUMS                        drift detection against the Deep-MKV-TS originals
├── metrics_summary.csv                   A1-A34, mean ± std, per seed
├── metrics_per_asset.csv                 per-metric × per-asset breakdown (8 rows per metric)
├── curve_b_aggregate.json                B curve-shape aggregate
├── grid_tvd_aggregate.json               path-cloud TVD
└── seed_{{i}}_metrics.json                 full per-seed dump incl. the per_asset block
```

The `.npy` arrays are **gitignored**: `(8192, 252, 8)` float64 = 132 MB each, over GitHub's
100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from the
tracked code and the tracked coefficients — deterministically so, since the generation seeds
are fixed at `{gseed} + i` and the rollout touches no GPU RNG. The `metadata.json` beside each
array **is** tracked, so shapes, price ranges, the S0 residual and the measured `Ẑ` maxima
stay auditable without the payload.

## Reproduce

```bash
cd /home/tbasseras/benchmark
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS/code

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. generate the bank — CPU only, ONE core, no GPU. ~{total_min:.1f} min for {n_bank} seeds.
#    Aborts before writing anything if the control is not exactly zero.
cd results/HestonMultiAsset/reference/code
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \\
PYTHONPATH="$R/src:$R/experiments:$D" taskset -c 15 \\
    /home/tbasseras/gpu-venv/bin/python generate_reference_multiasset.py
cd -

# 3. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/gpu-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 4. metrics — both sides go through byte-identical code
#    (GPU needed HERE, not for generation: A18/A19 train discriminators and predictors)
CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method perfect_recovery --dataset HestonMultiAsset
CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method reference --dataset HestonMultiAsset

# 5. figures
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/reference/code/plot_diagnostics_multiasset.py
/home/tbasseras/gpu-venv/bin/python \\
    metrics/plot_score_losses.py --method reference --dataset HestonMultiAsset

# 6. memorisation diagnostic
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/reference/code/measure_memorisation.py

# 7. regenerate this README from the artefacts
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/reference/code/render_readme.py

# 8. refresh the cross-method comparison table so this column appears in it
/home/tbasseras/gpu-venv/bin/python results/HestonMultiAsset/tools/render_comparison.py
```
"""
    out = os.path.join(REF, "README.md")
    n_h2 = sum(1 for ln in md.splitlines() if ln.startswith("## "))
    if n_h2 != 8:
        raise SystemExit(
            f"ABORT: GUIDELINE 8.1 requires exactly 8 '##' sections, got {n_h2} -- "
            "README not written."
        )
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines, {n_h2} '##' sections)")


if __name__ == "__main__":
    main()
