#!/usr/bin/env python3
"""Structural test for ``render_readme.py`` -- MULTIASSET_GUIDELINE.md section 8.9.

The guideline is blunt about why this exists: a generated page that is wrong is
worse than a hand-written one, because it will be regenerated wrong forever. SBTS
shipped a tree listing ``code/README.md`` for weeks before that file existed, and
nobody noticed, because nothing checked.

None of CSDI, LS4 or SBTS ships such a test -- this is the first. It runs the real
renderer against synthetic, SCHEMA-EXACT fixtures (a temporary tree, never the real
artefacts) and asserts the seven properties section 8.9 enumerates:

  1. it writes a README at all;
  2. every markdown table is rectangular, counting ``|`` and ignoring escaped ``\\|``;
  3. exactly 8 ``##`` headings, in the section 8.1 order;
  4. no ``Path Shadowing`` / ``PS-MC`` / ``path_shadowing`` string anywhere;
  5. every image link resolves to a file that exists;
  6. ``*(native d=8)*`` on exactly 10 lines BEGINNING WITH ``|`` -- the tag also
     appears in prose, and a naive ``grep -c`` reports 14 and calls the page broken;
  7. the headline win-counts are arithmetically consistent with the fixture inputs.

Check 7 is the one with teeth, and the fixture is built to make it bite. Two metrics
are NOT "lower is better", and getting either wrong silently inverts the headline
while leaving it looking plausible:

  ``A33_sigma_corr``      direction ``up``   -- fixture gives method 0.9 vs floor 0.5,
                          a WIN under ``up`` and a LOSS under a naive ``down``.
  ``A28_kurtosis_ratio``  direction ``none`` -- perfect value is 1.0, scored |x - 1|.
                          Fixture gives method 1.2 vs floor 0.5: |0.2| <= |0.5| is a
                          WIN, while a naive ``down`` reads 1.2 <= 0.5 and calls it a
                          loss.

So a renderer that collapsed every row to "smaller is better" would score 2 instead
of 4 and this test would fail. The direction table itself is pinned separately, in
``test_direction_table``, so the traps cannot be defused by editing CATEGORIES.

What this canNOT check is link resolution against the REAL tree -- the fixture has no
``code/README.md``, and its ``plots/*.png`` are fabricated. Section 8.9 prescribes a
separate shell step for that, printed by ``main`` on success.

Usage:
    /home/tbasseras/gpu-venv/bin/python \\
        results/HestonMultiAsset/Deep-MKV-TS/code/test_render_readme.py
Exit code 0 = all checks pass. Non-zero = at least one failed; every failure is
printed with its measured value, never just "assertion failed".
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERER = os.path.join(HERE, "render_readme.py")

SEEDS = [0, 1, 2, 3, 4]

# The 10 rows the guideline tags `*(native d=8)*`: A6-A11 (six), A18 (two rows,
# GRU and MLP), A20 and A25. Everything else is a per-asset mean.
NATIVE = {
    "A6_path_mmd2", "A7_terminal_mmd2", "A8_increment_mmd2", "A9_volatility_mmd",
    "A10_terminal_swd", "A11_path_swd",
    "A18_disc_score_gru", "A18_disc_score_mlp",
    "A20_cov_error", "A25_mean_rmse",
}

# (method_mean, floor_mean) overrides. Everything not named here is (2.0, 1.0),
# a clear loss on a lower-is-better row.
WINS = {
    "A6_path_mmd2":       (0.5, 1.0),  # plain win, direction down
    "A20_cov_error":      (1.0, 1.0),  # exact tie counts as at-floor  (<=)
    "A33_sigma_corr":     (0.9, 0.5),  # TRAP 1: win only under direction "up"
    "A28_kurtosis_ratio": (1.2, 0.5),  # TRAP 2: win only under direction "none"
}
EXPECTED_A_WINS = len(WINS)

B_PLOTS = [
    ("terminal_hist", "Terminal Price Histogram"),
    ("logret_hist",   "Log-Return Histogram"),
    ("acf_abs",       "ACF |r|"),
    ("acf_sq",        "ACF r^2"),
    ("vol_cluster",   "Volatility Clustering"),
    ("qq_plot",       "QQ Plot"),
]
B_WINNERS = {"terminal_hist", "acf_sq"}
EXPECTED_B_WINS = len(B_WINNERS)

EXPECTED_HEADINGS = [
    "Metrics A1-A34 + B, mean ± std across 5 seeds",
    "B, Curve-Shape Metrics, mean ± std across 5 seeds",
    "Stylised Facts Diagnostic (Multi-Asset Heston vs Deep-MKV-TS, seed 0, asset 0)",
    "Deep-MKV-TS Training Loss (5 seeds)",
    "A18, Discriminative Classifier Training Loss",
    "A19, Predictive Score Training Loss (TSTR)",
    "File layout",
    "Reproduce",
]

FORBIDDEN = ["Path Shadowing", "PS-MC", "path_shadowing"]

PLOT_FILES = ["heston_diagnostics.png", "loss_convergence.png",
              "disc_classifier_loss.png", "pred_score_loss.png"]


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------
def load_renderer():
    spec = importlib.util.spec_from_file_location("render_readme_under_test", RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric_keys(mod):
    """[(key, direction)] flattened from the renderer's own CATEGORIES."""
    return [(k, d) for _cat, rows in mod.CATEGORIES for k, _lbl, d in rows]


def write_summary(path, keys, pick):
    """metrics_summary.csv. `pick` maps key -> mean; every seed repeats it."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "mean", "std", "scope"] + [f"seed_{i}" for i in SEEDS])
        for key in keys:
            m = pick(key)
            scope = "native" if key in NATIVE else "per_asset"
            w.writerow([key, f"{m:.6f}", "0.010000", scope]
                       + [f"{m:.6f}" for _ in SEEDS])


def b_block(name, mse):
    per = [mse] * len(SEEDS)
    blk = {"name": name}
    for mk in ("mse", "pct", "nrmse", "cvar90", "cvar95"):
        blk[mk] = {"mean": mse, "std": 0.01, "per_seed": per}
    return blk


def build_fixture(mod):
    """A temporary method tree + floor tree, schema-exact, nothing real touched."""
    tmp = tempfile.mkdtemp(prefix="dmkv_render_test_")
    root = os.path.join(tmp, "Deep-MKV-TS")
    floor = os.path.join(tmp, "perfect_recovery")
    for d in (root, floor,
              os.path.join(root, "losses"), os.path.join(root, "weights"),
              os.path.join(root, "plots"), os.path.join(root, "code")):
        os.makedirs(d, exist_ok=True)

    keys = [k for k, _ in metric_keys(mod)]
    write_summary(os.path.join(root, "metrics_summary.csv"), keys,
                  lambda k: WINS.get(k, (2.0, 1.0))[0])
    write_summary(os.path.join(floor, "metrics_summary.csv"), keys,
                  lambda k: WINS.get(k, (2.0, 1.0))[1])

    agg = {p: b_block(n, 0.5 if p in B_WINNERS else 2.0) for p, n in B_PLOTS}
    agg_f = {p: b_block(n, 1.0) for p, n in B_PLOTS}
    with open(os.path.join(root, "curve_b_aggregate.json"), "w") as fh:
        json.dump(agg, fh)
    with open(os.path.join(floor, "curve_b_aggregate.json"), "w") as fh:
        json.dump(agg_f, fh)

    with open(os.path.join(root, "grid_tvd_aggregate.json"), "w") as fh:
        json.dump({"mean": 0.12, "std": 0.004, "per_seed": [0.12] * len(SEEDS)}, fh)
    with open(os.path.join(floor, "grid_tvd_aggregate.json"), "w") as fh:
        json.dump({"mean": 0.10, "std": 0.003, "per_seed": [0.10] * len(SEEDS)}, fh)

    with open(os.path.join(root, "losses", "generation_time.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "seed", "n_samples", "T", "d", "n_steps", "selected_step",
            "ridge_lambda", "batch_size", "n_workers", "elapsed_sec", "elapsed_min"])
        w.writeheader()
        for s in SEEDS:
            w.writerow({"seed": s, "n_samples": 8192, "T": 252, "d": 8,
                        "n_steps": 3000, "selected_step": 2500,
                        "ridge_lambda": 1.0, "batch_size": 256, "n_workers": 8,
                        "elapsed_sec": 120.0, "elapsed_min": 2.0})

    for s in SEEDS:
        with open(os.path.join(root, "losses", f"seed_{s}_losses.csv"),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "step", "phase", "loss_total", "objective", "grad_norm",
                "control_rms", "elapsed_sec"])
            w.writeheader()
            for step in range(0, 3000, 100):
                w.writerow({"step": step, "phase": "train",
                            "loss_total": 1.0 / (step + 1),
                            "objective": 0.5 / (step + 1),
                            "grad_norm": 1.0, "control_rms": 0.1,
                            "elapsed_sec": float(step)})
        with open(os.path.join(root, "weights", f"seed_{s}_config.json"), "w") as fh:
            json.dump({
                "method": "Deep-MKV-TS", "seed": s, "d": 8, "seq_len": 252,
                "n_steps": 3000, "selected_step": 2500,
                "selection_val_discrepancy": 0.1207937,
                "n_parameters": 123456, "batch_size": 256,
                "target_batch_size": 256, "hidden_dim": 96, "num_layers": 1,
                "lr": 2e-3, "weight_decay": 1e-5, "grad_clip_norm": 5.0,
                "eta": 1.0, "sigma_min": 1e-3, "sigma_max": 0.6,
                "lambda_scale": 50.0, "kappa_scale": 100.0,
                "discrepancy_preset": "old_fullv_w0p25",
                "abs_return_acf_weight": 0.25, "squared_return_acf_weight": 0.125,
                "ridge_lambda": 1.0, "ce_ridge": 1e-3,
                "control": "matrix", "drift_adjoint_backend": "autograd_replay",
                "path_derivative_backend": "autograd_replay",
                "differentiable_sigma": True, "max_eigh_batch": 32768,
                "retuned_for_d8": ["ridge_lambda"],
                "train_time_sec": 18720.0, "gpu": "A100-SXM4-80GB",
                "date": "2026-08-26",
            }, fh)
        gd = os.path.join(root, "generated_paths", f"seed_{s}")
        os.makedirs(gd, exist_ok=True)
        with open(os.path.join(gd, "metadata.json"), "w") as fh:
            json.dump({"method": "Deep-MKV-TS", "seed": s,
                       "shape": [8192, 252, 8], "dtype": "float64",
                       "selected_step": 2500, "ridge_lambda": 1.0,
                       "n_parameters": 123456, "s0_rescaled": True}, fh)

    for name in PLOT_FILES:
        open(os.path.join(root, "plots", name), "wb").close()
    return tmp, root, floor


# ---------------------------------------------------------------------------
# the seven checks
# ---------------------------------------------------------------------------
def pipes(line):
    """Column count of a markdown row, with escaped \\| not counted."""
    return line.replace("\\|", "").count("|")


def table_blocks(lines):
    """Contiguous runs of lines starting with '|', as (start_index, [lines])."""
    blocks, cur, start = [], [], 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("|"):
            if not cur:
                start = i
            cur.append(ln)
        elif cur:
            blocks.append((start, cur))
            cur = []
    if cur:
        blocks.append((start, cur))
    return blocks


def check(md, root, mod, failures):
    def fail(n, msg):
        failures.append(f"CHECK {n} FAILED: {msg}")

    lines = md.splitlines()

    # 2 -- rectangular tables
    for start, blk in table_blocks(lines):
        widths = {pipes(ln) for ln in blk}
        if len(widths) != 1:
            counts = {}
            for ln in blk:
                counts.setdefault(pipes(ln), []).append(ln[:70])
            fail(2, f"ragged table at line {start + 1}: column counts {sorted(widths)}\n"
                    + "\n".join(f"    {w}: {v[0]}" for w, v in sorted(counts.items())))

    # 3 -- exactly 8 '##' headings, in order
    heads = [ln[3:].strip() for ln in lines
             if ln.startswith("## ") and not ln.startswith("###")]
    if len(heads) != 8:
        fail(3, f"expected 8 '##' headings, found {len(heads)}: {heads}")
    elif heads != EXPECTED_HEADINGS:
        for got, want in zip(heads, EXPECTED_HEADINGS):
            if got != want:
                fail(3, f"heading order/text: got {got!r}, want {want!r}")

    # 4 -- forbidden strings
    for s in FORBIDDEN:
        if s in md:
            fail(4, f"forbidden string {s!r} present")

    # 5 -- image links resolve
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        if rel.startswith("http"):
            continue
        if not os.path.exists(os.path.join(root, rel)):
            fail(5, f"image link does not resolve: {rel}")

    # 6 -- native tag on exactly 10 TABLE ROWS (prose mentions do not count)
    rows = [ln for ln in lines if ln.startswith("|") and "*(native d=8)*" in ln]
    if len(rows) != 10:
        fail(6, f"'*(native d=8)*' on {len(rows)} table rows, expected 10")
    native_prefixes = {k.split("_")[0] for k in NATIVE}
    for ln in rows:
        label = ln.split("|")[1].strip()
        prefix = label.split()[0] if label.split() else ""
        if prefix and prefix not in native_prefixes:
            fail(6, f"native tag on non-native row: {label}")

    # 7 -- headline arithmetic
    m = re.search(r"\*\*(\d+) of the (\d+) A-metric rows", md)
    if not m:
        fail(7, "A headline sentence not found")
    else:
        got, total = int(m.group(1)), int(m.group(2))
        n_keys = len(metric_keys(mod))
        if total != n_keys:
            fail(7, f"A headline denominator {total}, fixture supplies {n_keys} rows")
        if got != EXPECTED_A_WINS:
            fail(7, f"A headline says {got} rows at/below floor, fixture built "
                    f"{EXPECTED_A_WINS} (A33 needs direction 'up', A28 needs 'none')")
    m = re.search(r"\*\*(\d+) of the (\d+) B plots\*\*", md)
    if not m:
        fail(7, "B headline sentence not found")
    else:
        got, total = int(m.group(1)), int(m.group(2))
        if total != len(B_PLOTS):
            fail(7, f"B headline denominator {total}, expected {len(B_PLOTS)}")
        if got != EXPECTED_B_WINS:
            fail(7, f"B headline says {got}, fixture built {EXPECTED_B_WINS}")


def test_direction_table(mod, failures):
    """Pin the two directions check 7's traps depend on.

    Without this, someone could 'fix' a failing check 7 by flipping A33 to `down`
    and the fixture would quietly stop discriminating.
    """
    d = dict(metric_keys(mod))
    for key, want in (("A33_sigma_corr", "up"), ("A28_kurtosis_ratio", "none")):
        if d.get(key) != want:
            failures.append(
                f"DIRECTION TABLE FAILED: {key} is {d.get(key)!r}, must be {want!r} "
                f"(GUIDELINE 8.9 check 7)")


def main():
    mod = load_renderer()
    failures = []
    test_direction_table(mod, failures)

    tmp, root, floor = build_fixture(mod)
    try:
        mod.SBTS, mod.FLOOR = root, floor
        mod.MA = os.path.dirname(root)
        mod.main()

        out = os.path.join(root, "README.md")
        # 1 -- it writes a README at all
        if not os.path.exists(out):
            failures.append("CHECK 1 FAILED: renderer wrote no README.md")
        else:
            with open(out) as fh:
                md = fh.read()
            if not md.strip():
                failures.append("CHECK 1 FAILED: README.md is empty")
            else:
                print(f"rendered {len(md.splitlines())} lines from fixtures")
                check(md, root, mod, failures)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nAll 7 structural checks pass (GUIDELINE section 8.9), plus the "
          "direction-table pin.")
    print("Still to run against the REAL tree (fixtures cannot cover it):")
    print("  cd results/HestonMultiAsset/Deep-MKV-TS && "
          "grep -o '](\\([^)h][^)]*\\))' README.md | sed 's/^](//;s/)$//' | sort -u "
          "| while read -r p; do [ -e \"$p\" ] || echo \"BROKEN: $p\"; done")


if __name__ == "__main__":
    main()
