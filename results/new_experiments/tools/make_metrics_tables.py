"""Render README section 2 (the A1-A34 + B curve-shape suite) from ``metrics_summary.csv``.

Deliberately SEPARATE from ``aggregate_pdf_metrics.py``, which renders section 1 (the
protocol PDF's own evaluator metrics). The two suites answer different questions and the
README must not blend them: section 1 says "did the model reproduce the DGP structure the
protocol targets", section 2 says "how does it score on the benchmark's standard battery".

Method-neutral: the metric list comes from the CSV, so a method that produces the same
CSV schema tabulates without touching this file.

A33/A34 are dropped in both new experiments (they need the latent sigma path, which is not
part of the generator's output contract here). Their cells are blank in the CSV and render
as ``n/a`` -- never as 0, which would silently flatter the method.

Usage:
  python make_metrics_tables.py --model-dir ../experiment_A/LS4 \
      --floor-dir ../experiment_A/perfect_floor --label LS4 --table A
"""
import os
import csv
import argparse
import numpy as np

SEED_COLS = [f"seed_{i}" for i in range(5)]

GROUPS = [
    ("Fat Tail", ["A1_kurtosis_error", "A2_abs_r_q95_error", "A3_abs_r_q99_error",
                  "A4_tail_qq_error", "A5_hill_tail_index_error"]),
    ("Distribution", ["A6_path_mmd2", "A7_terminal_mmd2", "A8_increment_mmd2",
                      "A9_volatility_mmd", "A10_terminal_swd", "A11_path_swd",
                      "A12_rv_law_loss", "A13_mean_path_rmse", "A14_ks_logreturns",
                      "A15_skewness_error", "A16_qq_rmse", "A17_terminal_ks"]),
    ("Adversarial", ["A18_disc_score_gru", "A18_disc_score_mlp"]),
    ("Predictive", ["A19_pred_score_gru", "A19_pred_score_mlp"]),
    ("Temporal", ["A20_cov_error", "A21_acf_abs", "A22_acf_sq",
                  "A23_acf_lag1_abs_error", "A24_acf_lag1_sq_error"]),
    ("Volatility", ["A25_mean_rmse", "A26_std_error", "A27_logreturn_std_error",
                    "A28_kurtosis_ratio", "A29_sigma_mean_error", "A30_vol_path_rmse",
                    "A31_rolling_vol_ks", "A32_vol_of_vol_error"]),
    ("Heston-specific (dropped)", ["A33_sigma_corr", "A34_sigma_rmse"]),
]

PRETTY = {
    "A1_kurtosis_error": "A1 Kurtosis Error ↓",
    "A2_abs_r_q95_error": "A2 |r| q95 Error ↓",
    "A3_abs_r_q99_error": "A3 |r| q99 Error ↓",
    "A4_tail_qq_error": "A4 Tail QQ Error ↓",
    "A5_hill_tail_index_error": "A5 Hill Tail Index Error ↓",
    "A6_path_mmd2": "A6 Path MMD² ↓",
    "A7_terminal_mmd2": "A7 Terminal MMD² ↓",
    "A8_increment_mmd2": "A8 Increment MMD² ↓",
    "A9_volatility_mmd": "A9 Volatility MMD ↓",
    "A10_terminal_swd": "A10 Terminal SWD ↓",
    "A11_path_swd": "A11 Path SWD ↓",
    "A12_rv_law_loss": "A12 RV Law Loss ↓",
    "A13_mean_path_rmse": "A13 Mean Path RMSE ↓",
    "A14_ks_logreturns": "A14 KS Log-returns ↓",
    "A15_skewness_error": "A15 Skewness Error ↓",
    "A16_qq_rmse": "A16 QQ RMSE ↓",
    "A17_terminal_ks": "A17 Terminal KS ↓",
    "A18_disc_score_gru": "A18 Discriminative (GRU) ↓",
    "A18_disc_score_mlp": "A18 Discriminative (MLP) ↓",
    "A19_pred_score_gru": "A19 Predictive (GRU) ↓",
    "A19_pred_score_mlp": "A19 Predictive (MLP) ↓",
    "A20_cov_error": "A20 Covariance Error ↓",
    "A21_acf_abs": "A21 ACF |r| ↓",
    "A22_acf_sq": "A22 ACF r² ↓",
    "A23_acf_lag1_abs_error": "A23 ACF lag-1 |r| Error ↓",
    "A24_acf_lag1_sq_error": "A24 ACF lag-1 r² Error ↓",
    "A25_mean_rmse": "A25 Mean RMSE ↓",
    "A26_std_error": "A26 Std Error ↓",
    "A27_logreturn_std_error": "A27 Log-return Std Error ↓",
    "A28_kurtosis_ratio": "A28 Kurtosis Ratio → 1",
    "A29_sigma_mean_error": "A29 Sigma Mean Error ↓",
    "A30_vol_path_rmse": "A30 Vol Path RMSE ↓",
    "A31_rolling_vol_ks": "A31 Rolling Vol KS ↓",
    "A32_vol_of_vol_error": "A32 Vol-of-vol Error ↓",
    "A33_sigma_corr": "A33 Sigma Correlation (dropped)",
    "A34_sigma_rmse": "A34 Sigma RMSE (dropped)",
}

PLOTS = [
    ("log_ret_hist", "Log-return histogram"),
    ("qq_plot", "QQ plot"),
    ("acf_abs_r", "ACF of |log-returns|"),
    ("acf_sq_r", "ACF of squared log-returns"),
    ("roll_vol_hist", "Rolling volatility histogram"),
    ("tail_surv", "Tail survival"),
]


def load(model_dir):
    """metric -> [5 per-seed values]; blank cells (dropped metrics) become None."""
    path = os.path.join(model_dir, "metrics_summary.csv")
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            vals = []
            for c in SEED_COLS:
                raw = (row.get(c) or "").strip()
                vals.append(float(raw) if raw not in ("", "nan", "None") else None)
            out[row["metric"]] = vals
    return out


def stats(vals):
    """mean, sample std (ddof=1) over the seeds that actually produced a number."""
    good = [v for v in vals if v is not None]
    if not good:
        return None, None
    a = np.asarray(good, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def g(x, digits=6):
    return "n/a" if x is None else f"{x:.{digits}g}"


def ms(vals):
    m, s = stats(vals)
    return "n/a" if m is None else f"{m:.6g} ± {s:.3g}"


def combined_mse(src, stem):
    """Per-seed (funct + der + sec_der)/3 -- the row that ranks methods on curve shape."""
    parts = [src.get(f"B_{stem}_{k}") for k in ("funct", "der", "sec_der")]
    if any(p is None for p in parts):
        return [None] * 5
    out = []
    for i in range(5):
        trio = [p[i] for p in parts]
        out.append(None if any(t is None for t in trio) else sum(trio) / 3.0)
    return out


def mean_of(src, keys):
    """Per-seed mean over several CSV rows (used for the % / NRMSE / CVaR sublines)."""
    parts = [src.get(k) for k in keys]
    if any(p is None for p in parts):
        return [None] * 5
    out = []
    for i in range(5):
        trio = [p[i] for p in parts]
        out.append(None if any(t is None for t in trio) else sum(trio) / len(trio))
    return out


def emit_row(label, vals, floor_vals, has_floor):
    # Metric names contain literal pipes (A2 |r| q95). Unescaped they split the markdown
    # row into extra columns and the whole table renders misaligned on GitHub.
    label = label.replace("|r|", r"\|r\|").replace("|log-returns|", r"\|log-returns\|")
    cells = [g(v) for v in vals]
    row = f"| {label} | {ms(vals)} | " + " | ".join(cells) + " |"
    if has_floor:
        row += f" {ms(floor_vals)} |" if floor_vals else " n/a |"
    return row


def table_a(model, floor, label):
    has_floor = bool(floor)
    head = f"| Metric | {label} (mean ± std) | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |"
    align = "|---|---|---|---|---|---|---|"
    if has_floor:
        head += " Perfect floor |"
        align += "---|"
    lines = [head, align]
    for gname, keys in GROUPS:
        lines.append(f"| **{gname}** | | | | | | |" + (" |" if has_floor else ""))
        for k in keys:
            if k not in model and k not in floor:
                continue
            lines.append(emit_row(PRETTY.get(k, k), model.get(k, [None] * 5),
                                  floor.get(k), has_floor))
    return "\n".join(lines)


def table_b(model, floor, label):
    has_floor = bool(floor)
    head = f"| Plot | Measure | {label} (mean ± std) | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |"
    align = "|---|---|---|---|---|---|---|---|"
    if has_floor:
        head += " Perfect floor |"
        align += "---|"
    lines = [head, align]

    if "grid_tvd" in model or "grid_tvd" in floor:
        mv = [None if v is None else 100.0 * v for v in model.get("grid_tvd", [None] * 5)]
        fv = floor.get("grid_tvd")
        fv = None if fv is None else [None if v is None else 100.0 * v for v in fv]
        lines.append(emit_row("**Grid TVD (%)** | —", mv, fv, has_floor))

    for stem, pretty in PLOTS:
        rows = [
            ("MSE (funct/der/sec-der avg)", combined_mse(model, stem),
             combined_mse(floor, stem) if has_floor else None),
            ("% error", mean_of(model, [f"B_{stem}_{k}_pct" for k in ("funct", "der", "sec_der")]),
             mean_of(floor, [f"B_{stem}_{k}_pct" for k in ("funct", "der", "sec_der")]) if has_floor else None),
            ("NRMSE", mean_of(model, [f"B_{stem}_{k}_nrmse" for k in ("funct", "der", "sec_der")]),
             mean_of(floor, [f"B_{stem}_{k}_nrmse" for k in ("funct", "der", "sec_der")]) if has_floor else None),
            ("CVaR₉₀", model.get(f"B_{stem}_funct_cvar90", [None] * 5),
             floor.get(f"B_{stem}_funct_cvar90") if has_floor else None),
            ("CVaR₉₅", model.get(f"B_{stem}_funct_cvar95", [None] * 5),
             floor.get(f"B_{stem}_funct_cvar95") if has_floor else None),
        ]
        for i, (measure, vals, fvals) in enumerate(rows):
            name = f"**{pretty}**" if i == 0 else ""
            lines.append(emit_row(f"{name} | {measure}", vals, fvals, has_floor))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--floor-dir", default=None)
    ap.add_argument("--label", default="LS4")
    ap.add_argument("--table", choices=("A", "B"), required=True)
    a = ap.parse_args()

    model = load(a.model_dir)
    floor = load(a.floor_dir) if a.floor_dir else {}
    print((table_a if a.table == "A" else table_b)(model, floor, a.label))


if __name__ == "__main__":
    main()
