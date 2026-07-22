#!/usr/bin/env python3
"""
render_tables.py
────────────────
Reads every scored method's on-disk results and emits the family-grouped
comparison tables (A1-A34, B curve-shape, PS-MC) as GitHub-renderable HTML,
plus derives the win-counts. NOTHING is hand-transcribed — every number is
read from disk here and pasted verbatim into the READMEs.

Family super-columns (fixed order):
  GAN                 : TimeGAN, COSCI-GAN
  Diffusion           : Diffusion-TS, CSDI
  VAE                 : TimeVAE, TimeVQVAE, LS4
  Schrödinger Bridge  : SBTS
  Fourier Flow        : Fourier Flow

Usage
─────
    python metrics/render_tables.py            # prints all blocks to stdout
    python metrics/render_tables.py --which A  # only the A table
    python metrics/render_tables.py --which B
    python metrics/render_tables.py --which PS
"""
import argparse, json, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)
DATASET    = "Heston"

# (family label, [ (disk_dir, display_name), ... ])
FAMILIES = [
    ("GAN",                [("TimeGAN", "TimeGAN"),   ("COSCI-GAN", "COSCI-GAN")]),
    ("Diffusion",          [("DiffusionTS", "Diffusion-TS"), ("CSDI", "CSDI")]),
    ("VAE",                [("TimeVAE", "TimeVAE"), ("TimeVQVAE", "TimeVQVAE"), ("LS4", "LS4")]),
    ("Schrödinger Bridge", [("SBTS", "SBTS")]),
    ("Fourier Flow",       [("FourierFlow", "Fourier Flow")]),
]
# flat ordered list of disk dirs / display names
METHOD_DIRS  = [d for _, ms in FAMILIES for d, _ in ms]
METHOD_NAMES = [n for _, ms in FAMILIES for _, n in ms]

# ── A metric metadata: (row_label, json_key, direction) ─────────────────────
#   direction: "lo" lower-better, "hi" higher-better, "ratio" target=1
A_ROWS = [
    ("— Fat Tail —", None, None),
    ("A1 Kurtosis Error ↓",          "A1_kurtosis_error",       "lo"),
    ("A2 \\|r\\| q95 Error ↓",       "A2_abs_r_q95_error",      "lo"),
    ("A3 \\|r\\| q99 Error ↓",       "A3_abs_r_q99_error",      "lo"),
    ("A4 Tail QQ Error ↓",           "A4_tail_qq_error",        "lo"),
    ("A5 Hill Tail Index Error ↓",   "A5_hill_tail_index_error","lo"),
    ("— Distribution —", None, None),
    ("A6 Path MMD² ↓",               "A6_path_mmd2",            "lo"),
    ("A7 Terminal MMD² ↓",           "A7_terminal_mmd2",        "lo"),
    ("A8 Increment MMD² ↓",          "A8_increment_mmd2",       "lo"),
    ("A9 Volatility MMD ↓",          "A9_volatility_mmd",       "lo"),
    ("A10 Terminal SWD ↓",           "A10_terminal_swd",        "lo"),
    ("A11 Path SWD ↓",               "A11_path_swd",            "lo"),
    ("A12 RV Law Loss ↓",            "A12_rv_law_loss",         "lo"),
    ("A13 Mean Path RMSE ↓",         "A13_mean_path_rmse",      "lo"),
    ("A14 KS Log-returns ↓",         "A14_ks_logreturns",       "lo"),
    ("A15 Skewness Error ↓",         "A15_skewness_error",      "lo"),
    ("A16 QQ RMSE (300-pt) ↓",       "A16_qq_rmse",             "lo"),
    ("A17 Terminal Price KS ↓",      "A17_terminal_ks",         "lo"),
    ("— Adversarial —", None, None),
    ("A18 Disc Score GRU ↓",         "A18_disc_score_gru",      "lo"),
    ("A18 Disc Score MLP ↓",         "A18_disc_score_mlp",      "lo"),
    ("— Predictive —", None, None),
    ("A19 Pred Score GRU ↓",         "A19_pred_score_gru",      "lo"),
    ("A19 Pred Score MLP ↓",         "A19_pred_score_mlp",      "lo"),
    ("— Temporal —", None, None),
    ("A20 Covariance Error ↓",       "A20_cov_error",           "lo"),
    ("A21 ACF \\|r\\| Error (lags) ↓","A21_acf_abs",            "lo"),
    ("A22 ACF r² Error (lags) ↓",    "A22_acf_sq",              "lo"),
    ("A23 ACF \\|r\\| Lag-1 Error ↓","A23_acf_lag1_abs_error",  "lo"),
    ("A24 ACF r² Lag-1 Error ↓",     "A24_acf_lag1_sq_error",   "lo"),
    ("— Vol —", None, None),
    ("A25 Mean RMSE ↓",              "A25_mean_rmse",           "lo"),
    ("A26 Return Std Error ↓",       "A26_std_error",           "lo"),
    ("A27 Log-Return Std Error ↓",   "A27_logreturn_std_error", "lo"),
    ("A28 Kurtosis Ratio (→ 1)",     "A28_kurtosis_ratio",      "ratio"),
    ("A29 Sigma Mean Error ↓",       "A29_sigma_mean_error",    "lo"),
    ("A30 Cross-Sect. Vol Path RMSE ↓","A30_vol_path_rmse",     "lo"),
    ("A31 Rolling Vol KS (w=5) ↓",   "A31_rolling_vol_ks",      "lo"),
    ("A32 Vol-of-Vol Error ↓",       "A32_vol_of_vol_error",    "lo"),
    ("— Heston Spec —", None, None),
    ("A33 Teacher-Sigma Corr ↑",     "A33_sigma_corr",          "hi"),
    ("A34 Teacher-Sigma RMSE ↓",     "A34_sigma_rmse",          "lo"),
]

CURVE_PLOTS = [
    ("B_log_ret_hist", "Log-return histogram"),
    ("B_qq_plot",      "QQ plot"),
    ("B_acf_abs_r",    "ACF \\|r\\| lags 1–20"),
    ("B_acf_sq_r",     "ACF r² lags 1–20"),
    ("B_roll_vol_hist","Rolling vol histogram"),
    ("B_tail_surv",    "Tail survival"),
]


# ── formatting ──────────────────────────────────────────────────────────────
def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    if x == 0.0:
        return "0"
    a = abs(x)
    if a < 1e-3 or a >= 1e5:
        return f"{x:.2e}"
    # 4 significant figures
    from math import log10, floor
    digits = 3 - int(floor(log10(a)))
    digits = max(0, digits)
    s = f"{x:.{digits}f}"
    return s


def ms(mean, std):
    return f"{fmt(mean)} ± {fmt(std)}"


# ── data loading ────────────────────────────────────────────────────────────
def results_dir(method):
    return os.path.join(REPO, "results", DATASET, method)


def load_a_stats(method):
    """Return {key: (mean, std)} across the 5 seed JSONs."""
    rdir = results_dir(method)
    docs = []
    for seed in range(5):
        p = os.path.join(rdir, f"seed_{seed}_metrics.json")
        with open(p) as f:
            docs.append(json.load(f))
    out = {}
    for _, key, _ in A_ROWS:
        if key is None:
            continue
        vals = [d[key] for d in docs if key in d and d[key] is not None
                and not (isinstance(d[key], float) and np.isnan(d[key]))]
        if vals:
            out[key] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            out[key] = (None, None)
    return out


def load_curve(method):
    p = os.path.join(results_dir(method), "curve_b_aggregate.json")
    with open(p) as f:
        return json.load(f)


def load_psmc(method):
    p = os.path.join(results_dir(method), "path_shadowing", "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ── winner logic ────────────────────────────────────────────────────────────
def winner_idx(means, direction):
    """index of winning method among means (list, None allowed)."""
    idx = [i for i, m in enumerate(means) if m is not None]
    if not idx:
        return None
    if direction == "lo":
        return min(idx, key=lambda i: means[i])
    if direction == "hi":
        return max(idx, key=lambda i: means[i])
    if direction == "ratio":
        return min(idx, key=lambda i: abs(means[i] - 1.0))
    return None


# ── HTML header rows (shared) ───────────────────────────────────────────────
def header_html(first_col, extra_cols):
    """extra_cols: list of trailing <th rowspan=2> labels (e.g. Perfect, Winner)."""
    sup = [f'    <th rowspan="2">{first_col}</th>']
    for fam, methods in FAMILIES:
        n = len(methods)
        if n == 1:
            sup.append(f'    <th>{fam}</th>')
        else:
            sup.append(f'    <th colspan="{n}">{fam}</th>')
    for c in extra_cols:
        sup.append(f'    <th rowspan="2">{c}</th>')
    # sub row: every method column in order
    sub = [f'    <th>{name}</th>' for name in METHOD_NAMES]
    lines = ["<table>", "<thead>", "  <tr>"]
    lines += sup
    lines += ["  </tr>", "  <tr>"]
    lines += sub
    lines += ["  </tr>", "</thead>", "<tbody>"]
    return "\n".join(lines)


def cell(text, bold=False):
    return f"<td>{'<b>'+text+'</b>' if bold else text}</td>"


# ── A table ─────────────────────────────────────────────────────────────────
def render_A():
    stats = {m: load_a_stats(m) for m in METHOD_DIRS}
    wins = {m: 0 for m in METHOD_NAMES}
    total = 0
    ncol = 1 + len(METHOD_DIRS) + 1
    out = [header_html("Metric", ["Winner"])]
    for label, key, direction in A_ROWS:
        if key is None:  # category separator
            out.append(f'  <tr><td colspan="{ncol}"><b>{label}</b></td></tr>')
            continue
        means = [stats[m][key][0] for m in METHOD_DIRS]
        stds  = [stats[m][key][1] for m in METHOD_DIRS]
        wi = winner_idx(means, direction)
        total += 1
        win_name = METHOD_NAMES[wi] if wi is not None else "—"
        if wi is not None:
            wins[win_name] += 1
        row = [f'  <tr><td>{label}</td>']
        for i in range(len(METHOD_DIRS)):
            row.append(cell(ms(means[i], stds[i]), bold=(i == wi)))
        row.append(f'<td><b>{win_name}</b></td></tr>')
        out.append("".join(row))
    out += ["</tbody>", "</table>"]
    return "\n".join(out), wins, total


# ── B table ─────────────────────────────────────────────────────────────────
def render_B():
    curves = {m: load_curve(m) for m in METHOD_DIRS}
    perfect = None
    pp = os.path.join(REPO, "methods", "perfect_recovery", "results", "curve_b_aggregate.json")
    if os.path.exists(pp):
        perfect = json.load(open(pp))
    wins = {m: 0 for m in METHOD_NAMES}
    ncol = 2 + len(METHOD_DIRS) + 2  # Plot,Measure + methods + Perfect + Winner
    sup = ['  <tr>', '    <th rowspan="2">Plot</th>', '    <th rowspan="2">Measure</th>']
    for fam, methods in FAMILIES:
        n = len(methods)
        sup.append(f'    <th colspan="{n}">{fam}</th>' if n > 1 else f'    <th>{fam}</th>')
    sup += ['    <th rowspan="2">Perfect</th>', '    <th rowspan="2">Winner</th>', '  </tr>']
    sub = ['  <tr>'] + [f'    <th>{n}</th>' for n in METHOD_NAMES] + ['  </tr>']
    out = ["<table>", "<thead>"] + sup + sub + ["</thead>", "<tbody>"]

    for prefix, name in CURVE_PLOTS:
        mse_means = [curves[m][prefix]["mse"]["mean"] for m in METHOD_DIRS]
        wi = winner_idx(mse_means, "lo")
        win_name = METHOD_NAMES[wi] if wi is not None else "—"
        if wi is not None:
            wins[win_name] += 1
        pf = perfect[prefix] if perfect else None
        for mi, measure in enumerate(("mse", "pct", "nrmse")):
            row = ['  <tr>']
            if mi == 0:
                row.append(f'<td rowspan="3"><b>{name}</b></td>')
            meas_label = {"mse": "MSE", "pct": "% err", "nrmse": "NRMSE"}[measure]
            row.append(f'<td>{meas_label}</td>')
            for i, m in enumerate(METHOD_DIRS):
                d = curves[m][prefix][measure]
                txt = ms(d["mean"], d["std"])
                if measure in ("pct", "nrmse"):
                    txt = f"{fmt(d['mean'])}% ± {fmt(d['std'])}%"
                row.append(cell(txt, bold=(measure == "mse" and i == wi)))
            if pf is not None:
                pv = pf[measure]["mean"]
                row.append(f'<td>{fmt(pv)}</td>')
            else:
                row.append('<td>0</td>')
            if mi == 0:
                row.append(f'<td rowspan="3"><b>{win_name}</b></td>')
            row.append('</tr>')
            out.append("".join(row))
    out += ["</tbody>", "</table>"]
    return "\n".join(out), wins


# ── PS-MC table ─────────────────────────────────────────────────────────────
def psmc_crps(summary, h):
    """Return (mean, std) of CRPS at horizon h, uniform-weight variant."""
    key = f"h{h}_CRPS_uniform"
    if summary is None or key not in summary:
        return (None, None)
    return (summary[key]["mean"], summary[key]["std"])


def render_PS():
    sums = {m: load_psmc(m) for m in METHOD_DIRS}
    base = None
    for m in METHOD_DIRS:
        if sums[m] is not None:
            base = sums[m]["baseline"]
            break
    wins = {m: 0 for m in METHOD_NAMES}
    out = [header_html("Metric", ["RW baseline", "Winner"])]
    for h in (32, 64):
        means = [psmc_crps(sums[m], h)[0] for m in METHOD_DIRS]
        stds  = [psmc_crps(sums[m], h)[1] for m in METHOD_DIRS]
        wi = winner_idx(means, "lo")
        win_name = METHOD_NAMES[wi] if wi is not None else "—"
        if wi is not None:
            wins[win_name] += 1
        row = [f'  <tr><td>PS-MC CRPS H={h} ↓</td>']
        for i in range(len(METHOD_DIRS)):
            row.append(cell(ms(means[i], stds[i]), bold=(i == wi)))
        bval = fmt(base[f"CRPS_h{h}"]) if base else "—"
        row.append(f'<td>{bval}</td><td><b>{win_name}</b></td></tr>')
        out.append("".join(row))
    out += ["</tbody>", "</table>"]
    return "\n".join(out), wins


# ── per-method markdown (for results/Heston/<M>/README.md) ──────────────────
def load_a_seeds(method):
    """Return {key: [v0..v4]} per-seed values for a method."""
    rdir = results_dir(method)
    docs = []
    for seed in range(5):
        with open(os.path.join(rdir, f"seed_{seed}_metrics.json")) as f:
            docs.append(json.load(f))
    out = {}
    for _, key, _ in A_ROWS:
        if key is None:
            continue
        out[key] = [d.get(key) for d in docs]
    return out


def perfect_floor_a():
    """Mean across the 5 independent-draw perfect-recovery seeds, per A key."""
    pdir = os.path.join(REPO, "methods", "perfect_recovery", "results")
    docs = []
    for seed in range(5):
        with open(os.path.join(pdir, f"seed_{seed}_metrics.json")) as f:
            docs.append(json.load(f))
    out = {}
    for _, key, _ in A_ROWS:
        if key is None:
            continue
        vals = [d[key] for d in docs if key in d and d[key] is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def perfect_floor_curve():
    pp = os.path.join(REPO, "methods", "perfect_recovery", "results", "curve_b_aggregate.json")
    return json.load(open(pp)) if os.path.exists(pp) else None


def render_method_A_md(method):
    """Per-seed A table (markdown) for one method."""
    stats = load_a_stats(method)
    seeds = load_a_seeds(method)
    floor = perfect_floor_a()
    lines = ["| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |",
             "|--------|-----------|--------|--------|--------|--------|--------|---------------|"]
    for label, key, _ in A_ROWS:
        if key is None:
            lines.append(f"| **{label}** | | | | | | | |")
            continue
        m, s = stats[key]
        sv = seeds[key]
        cells = " | ".join(fmt(v) for v in sv)
        lines.append(f"| {label} | {ms(m, s)} | {cells} | {fmt(floor[key])} |")
    return "\n".join(lines)


def render_method_B_md(method):
    """Per-method B table (markdown): MSE/%err/NRMSE per-seed + mean±std + floor."""
    cur = load_curve(method)
    pf = perfect_floor_curve()
    lines = ["| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |",
             "|------|---------|-----------|--------|--------|--------|--------|--------|---------------|"]
    for prefix, name in CURVE_PLOTS:
        for mi, measure in enumerate(("mse", "pct", "nrmse")):
            d = cur[prefix][measure]
            lab = {"mse": "MSE", "pct": "% err", "nrmse": "NRMSE"}[measure]
            ps = d.get("per_seed", [None] * 5)
            if measure == "mse":
                val = ms(d["mean"], d["std"])
                cells = " | ".join(fmt(v) for v in ps)
                fv = fmt(pf[prefix][measure]["mean"]) if pf else "—"
            else:
                val = f"{fmt(d['mean'])}% ± {fmt(d['std'])}%"
                cells = " | ".join(f"{fmt(v)}%" for v in ps)
                fv = f"{fmt(pf[prefix][measure]['mean'])}%" if pf else "—"
            plot_cell = f"**{name}**" if mi == 0 else ""
            lines.append(f"| {plot_cell} | {lab} | {val} | {cells} | {fv} |")
    return "\n".join(lines)


def render_method_PS_md(method):
    s = load_psmc(method)
    if s is None:
        return "_(no path-shadowing results)_"
    base = s["baseline"]
    lines = ["| Metric | Value (mean ± std) | RW baseline |",
             "|--------|--------------------|-------------|"]
    for h in (32, 64):
        m, sd = psmc_crps(s, h)
        lines.append(f"| PS-MC CRPS H={h} ↓ | {ms(m, sd)} | {fmt(base[f'CRPS_h{h}'])} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B", "PS", "all"], default="all")
    ap.add_argument("--method", default=None,
                    help="disk dir of a single method; prints per-method markdown tables")
    args = ap.parse_args()

    if args.method:
        print("<!-- ===== PER-METHOD A TABLE ===== -->")
        print(render_method_A_md(args.method))
        print("\n<!-- ===== PER-METHOD B TABLE ===== -->")
        print(render_method_B_md(args.method))
        print("\n<!-- ===== PER-METHOD PS-MC TABLE ===== -->")
        print(render_method_PS_md(args.method))
        return

    if args.which in ("A", "all"):
        html, wins, total = render_A()
        print("<!-- ===== A1-A34 TABLE ===== -->")
        print(html)
        print(f"\n<!-- A win-counts (of {total}): " +
              ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1])) + " -->\n")
    if args.which in ("B", "all"):
        html, wins = render_B()
        print("<!-- ===== B CURVE-SHAPE TABLE ===== -->")
        print(html)
        print("\n<!-- B win-counts (MSE, of 6): " +
              ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1]) if v) + " -->\n")
    if args.which in ("PS", "all"):
        html, wins = render_PS()
        print("<!-- ===== PS-MC TABLE ===== -->")
        print(html)
        print("\n<!-- PS-MC win-counts: " +
              ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1]) if v) + " -->\n")


if __name__ == "__main__":
    main()
