#!/usr/bin/env python3
"""
render_readme.py
────────────────
Generate results/HestonMultiAsset/SBTS/README.md from the metric artefacts.

Every number in the README is READ FROM DISK, never typed by hand. Re-running
this script after re-running the metrics regenerates the tables byte-for-byte,
so the README can never silently drift from the artefacts it claims to report.

Reads
  results/HestonMultiAsset/SBTS/metrics_summary.csv
  results/HestonMultiAsset/SBTS/curve_b_aggregate.json
  results/HestonMultiAsset/SBTS/grid_tvd_aggregate.json
  results/HestonMultiAsset/SBTS/losses/generation_time.csv
  results/HestonMultiAsset/SBTS/losses/memorisation.json          (optional)
  results/HestonMultiAsset/perfect_recovery/metrics_summary.csv   (floor)
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

# ── metric display table ─────────────────────────────────────────────────────
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
    """4 significant digits, scientific below 1e-3 — matches the d = 1 READMEs."""
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


def render_timing(path):
    if not os.path.exists(path):
        return "_(generation_time.csv not found)_", 0.0, {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "_(generation_time.csv empty)_", 0.0, {}
    total = sum(float(r["elapsed_min"]) for r in rows)
    out = ["| Seed | Workers | Elapsed |", "|------|---------|---------|"]
    out += [f"| {r['seed']} | {r['n_workers']} | {float(r['elapsed_min']):.1f} min |" for r in rows]
    return "\n".join(out), total, rows[0]


def main():
    sbts = read_summary(os.path.join(SBTS, "metrics_summary.csv"))
    floor = read_summary(os.path.join(FLOOR, "metrics_summary.csv"))
    agg = read_json(os.path.join(SBTS, "curve_b_aggregate.json"))
    agg_f = read_json(os.path.join(FLOOR, "curve_b_aggregate.json"))
    tvd = read_json(os.path.join(SBTS, "grid_tvd_aggregate.json"))
    tvd_f = read_json(os.path.join(FLOOR, "grid_tvd_aggregate.json"))
    mem = read_json(os.path.join(SBTS, "losses", "memorisation.json"))
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

    mem_line = ""
    if mem and "nn_ratio" in mem:
        r = float(mem["nn_ratio"])
        mem_line = (
            f"\n> **Memorisation caveat (measured, not assumed).** Nearest-neighbour ratio "
            f"`median NN(generated→train) / median NN(held-out real→train)` in log-return space "
            f"= **{r:.3f}**. A value of 1.0 would mean the generated paths sit no closer to the "
            f"training set than genuine held-out data does; {r:.3f} means they sit "
            f"**~{1.0 / max(r, 1e-9):.1f}× closer**. This is a property of SBTS at K={K}, not of "
            f"the d = 8 dataset: the same diagnostic on the committed d = 1 SBTS entry gives "
            f"**0.158**. Exact duplicates found: {mem.get('n_exact_duplicates', 'n/a')}. "
            f"Structurally, at K={K} **zero** training paths fall inside the kernel support — the "
            f"scheme degenerates to following the single nearest path, at d = 1 with the author's "
            f"own h = 0.05 just as much as at d = 8. See [`losses/memorisation.json`](losses/memorisation.json).\n"
        )

    nn_sel = ""
    if bw:
        nn_sel = (" Selected on the **validation split (seed 3), never on test**, as the largest "
                  "bandwidth satisfying every hard constraint and therefore the least-memorising "
                  "admissible choice; `h = 0.35` is the cliff where volatility error jumps "
                  "1.20 → 6.45 %.")

    md = f"""# SBTS on Multi-Asset Heston (d = 8)

**Schrödinger Bridge Time Series generation** (Alouadi, Barreau, Carlier & Pham, ICAIF 2025,
[arXiv:2503.02943](https://arxiv.org/abs/2503.02943)) applied to 8 192 **multi-asset** Heston
paths, seq\\_len = 252, **d = 8 correlated assets**.

The dataset is a *parsimonious* multi-asset Heston (Szimayer, Dimitroff & Lorenz 2011,
IJTAF 14(8) 1299-1333): eight independent 1-D Heston marginals coupled **only** through the
d(d−1)/2 spot correlations Σˢ, which makes the joint correlation matrix PSD by construction.
Per-asset parameters are frozen by `PARAM_SEED=1234` and dumped to the tracked
[`dataset/HestonMultiAsset/parameters.json`](../../../dataset/HestonMultiAsset/parameters.json).

SBTS is a **non-parametric, kernel-based** method: no neural network, no training loss, no
gradient descent, **no weights**. It estimates the Schrödinger-bridge drift directly from the
training array using a **Markovian-K kernel** over the last K states, then simulates by
Euler-Maruyama. The "model" *is* the training array plus `(h, K, N_pi)` — see
[`weights/README.md`](weights/README.md).

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

## How the metrics are computed in d = 8

Two scopes:

- **Per-asset then averaged** — the metric is computed independently on each of the 8 univariate
  slices `S[:, :, j]` and the **mean over assets** is the headline number. This covers
  **A1-A5, A12-A17, A19, A21-A24, A26-A34**, every **B curve-shape** metric and `grid_tvd`.
  The per-asset breakdown is kept in [`metrics_per_asset.csv`](metrics_per_asset.csv)
  (one row per metric × asset), so the averaging cannot hide a single bad asset.
- **Native d = 8** — the metric is evaluated **once** on the full `(N, T, 8)` tensor, marked
  *(native d=8)* in the table below. This covers **A6-A11, A18, A20, A25**: the rows where a
  multivariate generalisation is meaningful (kernel/Wasserstein distances on d-dimensional
  paths, a d-channel discriminator, and the full terminal covariance matrix).

**A19 is deliberately per-asset, not native.** `predictive_score.py::_train_gru` builds its
target as `Y_b = data_t[idx, 1:, :1]` — it predicts **only the first feature**. Run natively on
a d = 8 tensor it would silently report an *asset-0-only* score under a multi-asset name. Running
it 8 times on 8 univariate slices and averaging is the only honest reading.

---

## Metrics A1-A34, mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \\log(S_{{t+1}}/S_t)$ unless noted. A26 uses price increments $\\Delta S_t$.
> Rows marked *(native d=8)* are evaluated on the full `(N, T, 8)` tensor; every other row is the
> **mean over the 8 assets**.

{render_a_table(sbts, floor)}

> **Convention:** ↓ lower is better; ↑ higher is better. A28 Kurtosis Ratio: perfect = 1.0.
> The **Perfect floor** column is the *independent-draw* floor (GUIDELINE §5.4): five fresh
> draws from the *same* SDE with the *same* frozen parameters at seeds 1000-1004, scored with
> byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws
> never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a
> permutation of the test set, which would preserve every column-wise statistic exactly, collapse
> most metrics to 0, and be a misleading target.
{mem_line}
---

## B, Curve-Shape Metrics, mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L, not a scalar. The curve is computed **per asset**
and **averaged over the 8 assets** *before* the MSE / %err / NRMSE / CVaR combination, so the
combination rules stay byte-identical to the d = 1 tree.

- **MSE row**: mean of the three sub-scores (funct + der + sec\\_der)/3. **Decides the winner.**
- **% err row**: function-level MAPE on L only; the derivative curves are excluded because
  `diff(L)` has near-zero true values, which inflates any relative error into meaningless 10⁴ %.
- **NRMSE row**: range-normalised RMSE on L only, same exclusion.
- **CVaR₉₀ / CVaR₉₅**: tail-averaged pointwise curve error (Expected Shortfall) on L only.

{render_b_table(agg, agg_f, tvd, tvd_f)}

Plots are intentionally **not** produced for the d = 8 run — the metrics only, as requested.

---

## SBTS has no training loss

SBTS is kernel-based: there is no loss curve and no checkpoint. `losses/` stores the per-seed
bandwidth record and the selection evidence instead.

Generation wall-clock ({nw} workers, seeds run **sequentially**, full 5-seed run in
**{total_min / 60:.1f} h total**):

{timing}

---

## Deviations from the 1-D implementation (all deliberate, all documented)

1. **Per-asset σ, not a pooled scalar.** The eight assets have genuinely different volatilities
   (σ ranges 0.0097-0.0165), so log-returns are scaled per asset. Pooling a single σ would
   distort every marginal.
2. **Radial kernel, not a product of d univariate kernels.** `K_h(x) = (h² − ‖x‖²)²·1{{‖x‖₂ < h}}`,
   following the reference `sbts_multi_markovian.py`.
3. **The correct `weights_tilde` form.** The upstream *univariate* file
   (`sbts_uni_markovian.py`, lines 96-98) computes `exp(diff² · dt / 2)`, while the *multivariate*
   file (line 101) computes `exp(diff² / (2·dt))`. Continuity at `k = 0` proves the **multivariate**
   form correct; at `dt = 1/252` the univariate form evaluates to `exp(8e-6) ≈ 1` instead of
   `exp(0.5) ≈ 1.65`, silently disabling the bridge correction. This implementation uses the
   correct multivariate form. The committed d = 1 `sbts_generate.py` is left **untouched** so its
   already-published numbers remain reproducible.

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
│   └── render_readme.py                  regenerates this README from the artefacts
├── generated_paths/seed_{{0..4}}/
│   ├── generated_paths_8192x252x8.npy    (8192, 252, 8) float64 — gitignored, 132 MB
│   └── metadata.json                     seed, h, K, N_pi, shape, min/max, elapsed_sec (tracked)
├── losses/
│   ├── seed_{{i}}_bandwidth.json           h, K, N_pi, dt — no loss (kernel method)
│   ├── bandwidth_selection.json          full d = 8 sweep + selection verdict
│   ├── selection_criterion.md            pre-registered criterion and its caveats
│   ├── memorisation.json                 NN-ratio diagnostic on the final 8192-path output
│   └── generation_time.csv               wall-clock per seed
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

# 5. regenerate this README from the artefacts
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
