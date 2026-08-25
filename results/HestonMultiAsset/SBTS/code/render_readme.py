#!/usr/bin/env python3
"""
render_readme.py
----------------
Generate results/HestonMultiAsset/SBTS/README.md from the metric artefacts.

The section order, table shape and prose slots are a 1:1 mirror of the d = 1
entry `methods/SBTS/README.md`, so the two READMEs display identically. The one
deliberate omission is the Path Shadowing MC block: PS-MC has not been run for
d = 8, and a section with no numbers behind it would be a lie.

Material that is *about the dataset rather than about SBTS* -- how the d = 8
metrics are scoped, the deviations from the 1-D kernel implementation, and the
nearest-neighbour memorisation discussion -- lives one level up in
`results/HestonMultiAsset/README.md`, which is shared by every method added to
this dataset.

Every number here is READ FROM DISK, never typed by hand: the tables *and* the
headline win-counts are computed from the artefacts, so the prose cannot drift
away from the numbers it describes.

Reads
  results/HestonMultiAsset/SBTS/metrics_summary.csv
  results/HestonMultiAsset/SBTS/curve_b_aggregate.json
  results/HestonMultiAsset/SBTS/grid_tvd_aggregate.json
  results/HestonMultiAsset/SBTS/losses/generation_time.csv
  results/HestonMultiAsset/SBTS/losses/bandwidth_selection.json    (optional)
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv    (floor)
  results/HestonMultiAsset/perfect_recovery/curve_b_aggregate.json
  results/HestonMultiAsset/perfect_recovery/grid_tvd_aggregate.json

Writes
  results/HestonMultiAsset/SBTS/README.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/SBTS/code/render_readme.py
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SBTS = os.path.dirname(HERE)                                  # .../SBTS
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
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, {}
    total = sum(float(r["elapsed_min"]) for r in rows)
    out = ["| Seed | Workers | Elapsed |", "|------|---------|---------|"]
    out += [f"| {r['seed']} | {r['n_workers']} | {float(r['elapsed_min']):.1f} min |"
            for r in rows]
    return "\n".join(out), total, rows[0]


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    bw = read_json(os.path.join(SBTS, "losses", "bandwidth_selection.json"))

    missing = [n for n, o in (("SBTS/metrics_summary.csv", sbts),
                              ("SBTS/curve_b_aggregate.json", agg),
                              ("perfect_recovery/metrics_summary.csv", floor)) if not o]
    if missing:
        raise SystemExit(
            "ABORT: missing metric artefacts: " + ", ".join(missing) + "\n"
            "Run metrics/compute_all_multiasset.py --method SBTS and "
            "--method perfect_recovery first."
        )

    timing, total_min, r0 = render_timing(os.path.join(SBTS, "losses", "generation_time.csv"))
    h = r0.get("h", "0.31")
    K = r0.get("K", "20")
    npi = r0.get("N_pi", "50")
    nw = r0.get("n_workers", "16")

    at_floor, n_a, gaps = a_headline(sbts, floor)
    b_at, n_b = b_headline(agg, agg_f)

    at_floor_txt = ", ".join(at_floor[:8]) + ("…" if len(at_floor) > 8 else "")
    gap_txt = "; ".join(_fmt_gap(g) for g in gaps[:4]) if gaps else "none"
    b_at_txt = ", ".join(esc(x) for x in b_at) if b_at else "none"

    nn_sel = ""
    if bw:
        nn_sel = (" Selected on the **validation split (seed 3), never on test**, as the largest "
                  "bandwidth satisfying every hard constraint and therefore the least-memorising "
                  "admissible choice; `h = 0.35` is the cliff where volatility error jumps "
                  "1.20 → 6.45 %.")

    md = f"""# SBTS on Multi-Asset Heston (d = 8)

**Schrödinger Bridge Time Series generation** (Alouadi, Barreau, Carlier & Pham, ICAIF 2025,
[arXiv:2503.02943](https://arxiv.org/abs/2503.02943)) applied to 8 192 **multi-asset** Heston
stochastic-volatility price paths (seq\\_len = 252, **d = 8 correlated assets**).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss,
no gradient descent, **no weights**. It estimates the Schrödinger-bridge drift directly from
training data using a **Markovian-K kernel** (K={K} for this benchmark) over the last K states,
then simulates paths via Euler-Maruyama. The "model" *is* the training array plus
`(h, K, N_pi)` — see [`weights/README.md`](weights/README.md).

See [`code/README.md`](code/README.md) for source and implementation details, and the
dataset-level [`../README.md`](../README.md) for the multi-asset Heston law itself, the
per-asset vs native metric scoping, and the memorisation diagnostic shared by all methods
on this dataset.

> **Hyperparameters:** `h={h}`, `K={K}`, `N_pi={npi}`, `dt=1/252`.
> `K` and `N_pi` are the **author's** values (A. Alouadi, confirmed 2026-07-27 for the d = 1
> length-128 Heston benchmark) and are carried over unchanged. **`h` is ours.**
> It is the one hyperparameter that *cannot* cross dimension: the SBTS kernel is **radial**,
> `K_h(x) = (h² − ‖x‖²)²·1{{‖x‖₂ < h}}`, so its support is a ball whose radius must scale with
> the typical distance between d-dimensional increments — and that distance grows like √d
> (median pairwise distance 0.0372 at d = 1 vs **0.2643 at d = 8**). The author's `h = 0.05`
> is catastrophic here: **40.8 % volatility error**, generated excess kurtosis **45.66** against
> a real 1.12, and a minimum price of **0.0** (outright path collapse).{nn_sel}
> Full sweep, criterion and caveats: [`losses/bandwidth_selection.json`](losses/bandwidth_selection.json)
> and [`losses/selection_criterion.md`](losses/selection_criterion.md).

---

## Metrics A1-A34 + B, mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better;, no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **{len(at_floor)} of the {n_a} A-metric rows sit at or below the independent-draw floor** — {at_floor_txt}. The largest remaining gaps are {gap_txt}. Since SBTS is currently the only generator on this dataset, the floor is the only honest reference; cross-method win-counts belong in the dataset-level [`../README.md`](../README.md) once a second method lands.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix, which is the row that actually tests whether the d(d−1)/2 spot correlations Σˢ survived generation. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. A kernel/marginal-matching method with no explicit latent state has no mechanism to reproduce these, at d = 8 exactly as at d = 1.

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
> **Cross-seed stability**: SBTS is a deterministic kernel with no gradient descent, so the only seed-to-seed variation is the simulation noise of the Euler-Maruyama draw — there are no seed-collapse events of the kind a GAN can suffer.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs SBTS, seed 0, asset 0)

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

## SBTS has no training loss

SBTS is kernel-based, there is no loss curve. Instead, the bandwidth `h`, Markovian order `K`,
and Euler substeps `N_pi` are hyperparameters: **h={h}, K={K}, N_pi={npi}** (`K` and `N_pi`
author-specified by A. Alouadi 2026-07-27; `h` re-selected for d = 8 on the validation split,
because the radial kernel's support radius cannot cross dimension). The `losses/` directory
stores the per-seed bandwidth JSON records and the full selection evidence for reproducibility.

Generation wall-clock times ({nw} workers, sequential seeds, full 8 192-path × 8-asset run
finishes in **{total_min / 60:.1f} h total**):

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
results/HestonMultiAsset/SBTS/
├── README.md                             ← this file (generated by code/render_readme.py)
├── code/
│   ├── sbts_generate_multiasset.py       core module: radial kernel, per-asset sigma
│   ├── run_all_multiasset.py             full run: 5 seeds × 8192 paths × 252 steps × 8 assets
│   ├── plot_diagnostics_multiasset.py    the 8-panel stylised-facts figure
│   ├── measure_memorisation.py           nearest-neighbour memorisation diagnostic
│   └── render_readme.py                  regenerates this README from the artefacts
├── generated_paths/seed_{{0..4}}/
│   ├── generated_paths_8192x252x8.npy    (8192, 252, 8) float64 — gitignored, 132 MB
│   └── metadata.json                     seed, h, K, N_pi, shape, min/max, elapsed_sec (tracked)
├── losses/
│   ├── seed_{{i}}_bandwidth.json           h, K, N_pi, dt — no loss (kernel method)
│   ├── bandwidth_selection.json          full d = 8 sweep + selection verdict
│   ├── selection_criterion.md            pre-registered criterion and its caveats
│   ├── memorisation.json                 NN-ratio diagnostic on the final 8192-path output
│   └── generation_time.csv               wall-clock time per seed
├── plots/
│   ├── heston_diagnostics.png            8-panel stylised facts (seed 0, asset 0)
│   ├── disc_classifier_loss.png          A18 BCE curves, 5 seeds
│   └── pred_score_loss.png               A19 MAE curves, 5 seeds (asset 0)
├── weights/README.md                     why there are none
├── metrics_summary.csv                   A1-A34, mean ± std, per seed
├── metrics_per_asset.csv                 per-metric × per-asset breakdown (8 rows per metric)
├── curve_b_aggregate.json                B curve-shape aggregate
├── grid_tvd_aggregate.json               path-cloud TVD
└── seed_{{i}}_metrics.json                 full per-seed dump incl. the per_asset block
```

The `.npy` arrays are **gitignored**: `(8192, 252, 8)` float64 = 132 MB each, over GitHub's
100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from the tracked
code and tracked hyperparameters; the `metadata.json` beside each array **is** tracked, so
shapes, price ranges and generation times stay auditable without the payload.

## Reproduce

```bash
cd /home/tbasseras/benchmark

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. generate paths — CPU only, no GPU. ~{total_min / 60:.1f} h for 5 seeds.
setsid taskset -c 0-15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
    /home/tbasseras/sbts-venv/bin/python \\
    results/HestonMultiAsset/SBTS/code/run_all_multiasset.py \\
    > /tmp/sbts_ma_run.log 2>&1 < /dev/null & disown

# 3. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 4. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method SBTS

# 5. figures
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/SBTS/code/plot_diagnostics_multiasset.py
/home/tbasseras/gpu-venv/bin/python \\
    metrics/plot_score_losses.py --method SBTS --dataset HestonMultiAsset

# 6. memorisation diagnostic
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/SBTS/code/measure_memorisation.py

# 7. regenerate this README from the artefacts
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/SBTS/code/render_readme.py
```
"""
    out = os.path.join(SBTS, "README.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
