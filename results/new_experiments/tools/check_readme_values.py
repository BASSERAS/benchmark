#!/usr/bin/env python
"""Verify that every numeric table in a method README still matches the artefacts.

Why this file exists
--------------------
The READMEs assert, in prose, that their tables are "emitted verbatim by
``tools/aggregate_pdf_metrics.py``" and ``tools/make_metrics_tables.py``. That
claim is only worth something if somebody checks it. Nothing did.

The failure mode this catches is specific and realistic: a README is edited by
hand -- to fix a typo, to reword a verdict, to correct a sigma multiple -- and a
digit inside a table cell moves at the same time; or a table goes stale because
the metrics were recomputed and only *some* of the document was refreshed. Both
leave a README that reads perfectly and reports a number no artefact supports.

How it works
------------
For each generated block, re-run the generator that is supposed to have produced
it, locate the corresponding table in the README by the key of its first data
row, and compare **cell by cell**:

  block                     generator                                README anchor
  ------------------------- ---------------------------------------- ------------------
  1   PDF metrics (test)    aggregate_pdf_metrics --subdir           first metric key
                              pdf_metrics
  1.3 validation side       aggregate_pdf_metrics --subdir           same keys
                              pdf_metrics_validation
  2.1 A1-A34 battery        make_metrics_tables --table A            "A1 ..."
  2.2 B curve-shape         make_metrics_tables --table B            "B ..."

Comparison is on the **rendered text of each cell**, not on a re-parsed float.
That is deliberate: the README's contract is that the bytes came from the
generator, so a cell differing only in formatting is still a defect -- it means
somebody retyped it.

Both experiments carry the same metric keys in the test and validation tables, so
the two aggregator blocks are matched positionally in document order (test first,
validation second), not by key.

Usage
-----
    python check_readme_values.py \
        --root  results/new_experiments/experiment_A/LS4 \
        --floor results/new_experiments/experiment_A/perfect_floor \
        --experiment A

Exit codes: 0 every generated cell in the README matches a freshly generated one,
1 otherwise. All mismatches are printed, not just the first.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
PATTERN = {"A": "*_drawdown_memory.json", "B": "*_heston_mixture.json"}

# --- section 1.3 is a hand-curated DIGEST, not a generated block -------------
# It shows a handful of primary metrics side by side, test column vs validation
# column, drawn from two different aggregator runs. Its row labels are prose, not
# metric keys, so the mapping has to be written down explicitly. Every entry below
# was read off the aggregator's own output, not inferred from the label text.
#
# ``None`` marks a DERIVED row that no evaluator emits -- it is checked separately
# against its definition rather than against a single key.
SECTION_1_3 = {
    "A": {
        "`early_hit_rate_error`": "`errors.early_hit_rate_error`",
        "`future_rv_hit_gap_error`": "`errors.future_rv_hit_gap_error`",
        "`early_history_incremental_r2_error`": "`errors.early_history_incremental_r2_error`",
        "`future_rv_wasserstein`": "`errors.future_rv_wasserstein`",
        "`terminal_log_price_ks`": "`errors.terminal_log_price_ks`",
    },
    "B": {
        "Regime proportion TVD": "`mixture_fidelity.regime_proportion_tvd`",
        "W̄_param (derived)": None,
        "Realized-volatility Wasserstein":
            "`observable_fidelity.realized_volatility_wasserstein`",
        "Leverage-curve RMSE": "`observable_fidelity.leverage_curve_rmse_lags_0_20`",
    },
}

# W_bar_param is the unweighted mean of the three support-normalised Wassersteins
# (PDF 3.3). Recomputed from the per-seed JSONs, never copied from a table.
# --- section 1.1 is the other hand-assembled digest -------------------------
# Four rows, five columns: the first three come from the test aggregate, the last
# is DERIVED (model mean / floor mean, rendered to two significant figures). It is
# the table that decides each experiment, so it gets checked too.
SECTION_1_1 = {
    "A": {
        "`early_hit_rate_error`": "`errors.early_hit_rate_error`",
        "`future_rv_hit_gap_error`": "`errors.future_rv_hit_gap_error`",
        "`early_history_incremental_r2_error`": "`errors.early_history_incremental_r2_error`",
        "`future_rv_wasserstein`": "`errors.future_rv_wasserstein`",
    },
    "B": {
        "Regime proportion TVD": "`mixture_fidelity.regime_proportion_tvd`",
        "**W̄_param** *(derived — see below)*": None,
        "Realized-volatility Wasserstein":
            "`observable_fidelity.realized_volatility_wasserstein`",
        "Leverage-curve RMSE (lags 0–20)":
            "`observable_fidelity.leverage_curve_rmse_lags_0_20`",
    },
}

WBAR_COMPONENTS = [
    "mixture_fidelity.parameters.theta.support_normalized_wasserstein",
    "mixture_fidelity.parameters.xi.support_normalized_wasserstein",
    "mixture_fidelity.parameters.rho.support_normalized_wasserstein",
]


GENERATOR_ERRORS: list[str] = []


def run(cmd: list[str]) -> list[str]:
    """Run a generator; return only its markdown table rows.

    A failing generator is recorded as a failure and reported at the end, rather
    than aborting the process. Exiting here would discard every mismatch already
    found in the earlier blocks, which is the opposite of useful when a README has
    both a bad cell and a missing artefact.
    """
    out = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if out.returncode != 0:
        tail = (out.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        GENERATOR_ERRORS.append(
            f"generator exited {out.returncode}: {os.path.basename(cmd[0])} "
            f"{' '.join(cmd[1:])}\n      {tail}")
        return []
    return [ln for ln in out.stdout.splitlines() if ln.lstrip().startswith("|")]


def cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def is_sep(row: str) -> bool:
    """True for a markdown |---|---| separator row."""
    return all(set(c) <= set("-: ") and c for c in cells(row))


def data_rows(rows: list[str]) -> list[str]:
    """Drop the header row and the separator row; keep the body."""
    body = [r for r in rows if not is_sep(r)]
    return body[1:] if body else []


def readme_tables(readme: list[str]) -> list[list[str]]:
    """Every contiguous run of markdown table rows in the README, in document order."""
    tables, cur = [], []
    for ln in readme:
        if ln.lstrip().startswith("|"):
            cur.append(ln)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def match_table(tables: list[list[str]], want_keys: list[str], skip: set[int]) -> int:
    """Index of the first unconsumed README table whose data keys equal want_keys."""
    for i, t in enumerate(tables):
        if i in skip:
            continue
        got = [cells(r)[0] for r in data_rows(t)]
        if got == want_keys:
            return i
    return -1


def compare(name: str, generated: list[str], tables: list[list[str]],
            skip: set[int]) -> list[str]:
    """Compare one generated block against its README copy. Returns failure strings."""
    gen = data_rows(generated)
    if not gen:
        return [f"{name}: generator produced no data rows"]
    want = [cells(r)[0] for r in gen]

    idx = match_table(tables, want, skip)
    if idx < 0:
        # fall back to matching on the first key alone, so we can report *what* differs
        loose = [i for i, t in enumerate(tables) if i not in skip and data_rows(t)
                 and cells(data_rows(t)[0])[0] == want[0]]
        if not loose:
            return [f"{name}: no README table starts with row {want[0]!r} "
                    f"({len(want)} rows expected)"]
        idx = loose[0]
        got = [cells(r)[0] for r in data_rows(tables[idx])]
        miss = [k for k in want if k not in got]
        extra = [k for k in got if k not in want]
        fails = []
        for k in miss:
            fails.append(f"{name}: row {k!r} is generated but ABSENT from the README")
        for k in extra:
            fails.append(f"{name}: row {k!r} is in the README but NOT generated "
                         f"(stale or hand-added)")
        skip.add(idx)
        return fails or [f"{name}: row order differs between generator and README"]

    skip.add(idx)
    rm = data_rows(tables[idx])
    fails, checked = [], 0
    for g, r in zip(gen, rm):
        gc, rc = cells(g), cells(r)
        if len(gc) != len(rc):
            fails.append(f"{name}: row {gc[0]!r} has {len(rc)} cells in the README, "
                         f"{len(gc)} generated")
            continue
        for j, (a, b) in enumerate(zip(gc, rc)):
            checked += 1
            if a != b:
                fails.append(f"{name}: row {gc[0]!r} col {j}: "
                             f"README has {b!r}, generator emits {a!r}")
    print(f"  {name}: {len(gen)} rows, {checked} cells compared, {len(fails)} mismatch(es)")
    return fails


def parse_generated(rows: list[str]) -> dict[str, list[str]]:
    """key -> all rendered cells, from a generator's table rows.

    Aggregator column order: [key, 'mean ± std', 'CI half-width', 'floor mean ± std'].
    """
    return {cells(r)[0]: cells(r) for r in data_rows(rows)}


def wbar_cell(root: str, subdir: str, pattern: str) -> str:
    """Recompute the derived W̄_param cell exactly as the aggregator would render it.

    PDF 3.3: unweighted mean of the three support-normalised Wassersteins, computed
    per seed and only then aggregated -- not the mean of the three aggregated means,
    which would give the same mean but the wrong std.
    """
    import glob
    import json

    import numpy as np

    per_seed = []
    for p in sorted(glob.glob(os.path.join(root, subdir, pattern))):
        d = json.load(open(p, encoding="utf-8"))
        vals = []
        for key in WBAR_COMPONENTS:
            node = d
            for part in key.split("."):
                node = node[part]
            vals.append(float(node))
        per_seed.append(sum(vals) / len(vals))
    a = np.asarray(per_seed, dtype=float)
    return f"{a.mean():.6g} ± {a.std(ddof=1):.3g}"


def check_section_1_1(experiment: str, root: str, floor: str, pattern: str,
                      test_rows: list[str], tables: list[list[str]],
                      skip: set[int]) -> list[str]:
    """Verify the PRIMARY panel — the table that actually decides each experiment.

    Columns 1-3 must equal the test aggregate cell for cell. Column 4 (``× floor``)
    is derived: model mean / floor mean, rendered to two significant figures. It is
    checked by recomputation, because a stale ratio next to fresh operands is the
    least visible way for this table to go wrong.
    """
    mapping = SECTION_1_1[experiment]
    test = parse_generated(test_rows)

    idx = -1
    for i, t in enumerate(tables):
        if i in skip:
            continue
        header = [c for c in t if not is_sep(c)]
        if header and "× floor" in header[0]:
            idx = i
            break
    if idx < 0:
        return ["section 1.1: no README table whose header has a `× floor` column"]
    skip.add(idx)

    unknown = [cells(r)[0] for r in data_rows(tables[idx]) if cells(r)[0] not in mapping]
    if unknown:
        return [f"section 1.1: row label {u!r} is not in the label map; add it to "
                f"SECTION_1_1[{experiment!r}]" for u in unknown]

    fails, checked = [], 0
    for r in data_rows(tables[idx]):
        c = cells(r)
        label, key = c[0], mapping[c[0]]
        if key is None:                       # derived W̄_param row
            want = ["", wbar_cell(root, "pdf_metrics", pattern), None,
                    wbar_cell(floor, "pdf_metrics", pattern)]
        else:
            if key not in test:
                fails.append(f"section 1.1: {label} maps to {key}, absent from the "
                             f"aggregator output")
                continue
            want = test[key]
        for j in (1, 2, 3):
            if want[j] is None:               # CI half-width of a derived row: skip
                continue
            checked += 1
            if c[j] != want[j]:
                fails.append(f"section 1.1: {label} col {j}: README has {c[j]!r}, "
                             f"aggregator emits {want[j]!r}")
        # column 4: the derived ratio
        checked += 1
        model = float(want[1].split("±")[0])
        floor_mean = float(want[3].split("±")[0])
        got = c[4].replace("*", "").replace("×", "").strip()
        exp_ratio = model / floor_mean
        if abs(float(got) - exp_ratio) > 0.05 * exp_ratio:
            fails.append(f"section 1.1: {label} '× floor' is {c[4]!r}, but "
                         f"{model:.6g}/{floor_mean:.6g} = {exp_ratio:.3g}")
    print(f"  section 1.1 PRIMARY panel: {len(data_rows(tables[idx]))} rows, {checked} "
          f"cells checked (incl. derived ratios), {len(fails)} mismatch(es)")
    return fails


def check_section_1_3(experiment: str, root: str, pattern: str,
                      test_rows: list[str], val_rows: list[str],
                      tables: list[list[str]], skip: set[int]) -> list[str]:
    """Verify the hand-curated section-1.3 digest against both aggregator runs.

    This table is NOT emitted by any generator -- it is assembled by hand from two
    runs, with prose row labels. So it is checked by explicit mapping (SECTION_1_3)
    rather than by whole-table equality: column 1 must equal the test aggregate,
    column 2 must equal the validation aggregate, for every row present.
    """
    mapping = SECTION_1_3[experiment]
    test = parse_generated(test_rows)
    val = parse_generated(val_rows)

    # Match on the HEADER signature, not on row-label overlap. Section 1.1's primary
    # panel shares four of experiment A's five row labels but lays its columns out
    # differently (col 2 is a CI half-width, not the validation value), so a
    # label-subset match silently compares the wrong columns. Measured: it did.
    idx = -1
    for i, t in enumerate(tables):
        if i in skip:
            continue
        header = [c for c in t if not is_sep(c)]
        if header and "disc.npy" in header[0]:
            idx = i
            break
    if idx < 0:
        return ["section 1.3: no README table whose header mentions `disc.npy` "
                "(expected the validation-vs-test digest)"]
    skip.add(idx)

    unknown = [cells(r)[0] for r in data_rows(tables[idx]) if cells(r)[0] not in mapping]
    if unknown:
        return [f"section 1.3: row label {u!r} is not in the label map; add it to "
                f"SECTION_1_3[{experiment!r}] with the metric key it displays"
                for u in unknown]

    fails, checked = [], 0
    for r in data_rows(tables[idx]):
        c = cells(r)
        label = c[0]
        key = mapping[label]
        if key is None:                       # derived row
            want_test = wbar_cell(root, "pdf_metrics", pattern)
            want_val = wbar_cell(root, "pdf_metrics_validation", pattern)
        else:
            if key not in test or key not in val:
                fails.append(f"section 1.3: {label} maps to {key}, absent from the "
                             f"aggregator output")
                continue
            want_test, want_val = test[key][1], val[key][1]
        for col, (got, want, side) in enumerate(
                ((c[1], want_test, "test"), (c[2], want_val, "validation")), start=1):
            checked += 1
            if got != want:
                fails.append(f"section 1.3: {label} ({side} column): README has {got!r}, "
                             f"recomputed {want!r}")
    print(f"  section 1.3 digest: {len(data_rows(tables[idx]))} rows, {checked} cells "
          f"cross-checked against both runs, {len(fails)} mismatch(es)")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="method dir, e.g. .../experiment_A/LS4")
    ap.add_argument("--floor", required=True, help="perfect_floor dir, same experiment")
    ap.add_argument("--experiment", required=True, choices=["A", "B"])
    a = ap.parse_args()

    readme_path = os.path.join(a.root, "README.md")
    if not os.path.isfile(readme_path):
        print(f"FAIL  no README at {readme_path}")
        return 1
    tables = readme_tables(open(readme_path, encoding="utf-8").read().splitlines())
    pat = PATTERN[a.experiment]
    agg = os.path.join(TOOLS, "aggregate_pdf_metrics.py")
    tab = os.path.join(TOOLS, "make_metrics_tables.py")
    base = ["--model-dir", a.root, "--floor-dir", a.floor, "--pattern", pat, "--label", "LS4",
            "--exclude-prefix", "configuration", "oracle_gate"]

    print(f"Checking {readme_path} (experiment {a.experiment})")
    print(f"  {len(tables)} markdown tables found in the README")
    fails: list[str] = []
    skip: set[int] = set()

    test_rows = run([agg] + base + ["--subdir", "pdf_metrics"])
    fails += compare("PDF metrics (test)", test_rows, tables, skip)
    fails += check_section_1_1(a.experiment, a.root, a.floor, pat, test_rows, tables, skip)

    val_dir = os.path.join(a.root, "pdf_metrics_validation")
    if os.path.isdir(val_dir):
        val_rows = run([agg] + base + ["--subdir", "pdf_metrics_validation"])
        # The full validation table is NOT reproduced in the README -- section 1.3
        # shows a curated digest of it instead. Check that digest against both runs.
        fails += check_section_1_3(a.experiment, a.root, pat,
                                   test_rows, val_rows, tables, skip)
    else:
        fails.append(f"no pdf_metrics_validation/ under {a.root} -- PDF 1.4 requires the "
                     f"validation scoring pass alongside the test pass")

    for t in ("A", "B"):
        fails += compare(f"battery table {t}",
                         run([tab, "--model-dir", a.root, "--floor-dir", a.floor,
                              "--label", "LS4", "--table", t]), tables, skip)

    fails += GENERATOR_ERRORS
    if fails:
        print(f"\nFAIL  {len(fails)} problem(s):\n")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nPASS  every generated cell in this README matches a freshly generated one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
