#!/usr/bin/env python
"""Preflight for Experiment B: refuse to score unless the oracle gate actually passed.

Why this file exists
--------------------
PDF §3.2 makes the 8-regime validation accuracy >= 0.90 a **gate**: below it, no
Experiment B number means anything, because the instrument that produces every
mixture metric is not good enough to be trusted. The canonical scripts enforce
that gate exactly once, at fit time, in a different process -- and they leave two
holes behind. Both were read from the vendored source, not inferred:

**Gap 1 -- artefacts survive a failed gate.**
``protocol/experiments/scripts/fit_heston_mixture_oracle.py`` writes the model at
line 79 and the report at lines 80-83, and only *then* checks the accuracy and
raises at lines 85-89::

    79  joblib.dump(classifier, args.output_dir / "oracle.joblib")
    80  (args.output_dir / "gate_report.json").write_text(...)
    ...
    85  if report["eight_regime_accuracy"] < float(args.minimum_accuracy):
    86      raise RuntimeError(...)

A failed fit therefore leaves a fully loadable ``oracle.joblib`` and a fully
formed ``gate_report.json`` on disk. Nothing downstream can tell them apart from
a passing pair. **The presence of the blob is not evidence the gate passed.**

**Gap 2 -- the evaluator never re-checks.**
``evaluate_heston_parameter_mixture.py:232`` does::

    "oracle_gate": json.loads(args.oracle_gate_report.read_text(encoding="utf-8"))

It copies the report into its output payload and asserts nothing about it. So a
sub-0.90 oracle scores silently, and the failure is visible only to a reader who
opens the JSON and compares a number to 0.90 by eye.

Neither script may be edited -- PDF §7 item 7 requires the canonical scripts to
run unchanged. So the check goes *around* them, here, as a mandatory step in the
documented run order.

What it verifies
----------------
1. ``eight_regime_accuracy >= --minimum-accuracy`` (default 0.90, the PDF value).
2. The report is internally consistent: the confusion matrix is 8x8, its row sums
   equal ``validation_true_counts``, its column sums equal
   ``validation_predicted_counts``, its total equals ``validation_paths``, and
   ``trace / total`` reproduces the reported accuracy. A hand-edited accuracy
   field will not survive this -- the trace has to move with it.
3. ``configuration.minimum_accuracy`` is not *below* the PDF's 0.90, so a laxer
   local threshold cannot masquerade as the protocol's gate.
4. Staleness (per Gap 1): if ``oracle.joblib`` is newer than ``gate_report.json``,
   the report may describe a different forest than the one that will be loaded.
   That is exactly the state a failed refit leaves behind.

Usage
-----
    python check_oracle_gate.py \
        --gate-report ../../../dataset/Heston/new_experiments/experiment_B/oracle/gate_report.json \
        --oracle      ../../../dataset/Heston/new_experiments/experiment_B/oracle/oracle.joblib

Run this **before** every ``evaluate_heston_parameter_mixture.py`` invocation.
Exit codes: 0 gate passed and report is coherent, 1 otherwise (all problems are
reported, not just the first).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PDF_MINIMUM_ACCURACY = 0.90
N_REGIMES = 8


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gate-report", required=True,
                    help="experiment_B/oracle/gate_report.json")
    ap.add_argument("--oracle", default=None,
                    help="experiment_B/oracle/oracle.joblib; enables the staleness check")
    ap.add_argument("--minimum-accuracy", type=float, default=PDF_MINIMUM_ACCURACY,
                    help=f"PDF 3.2 gate, default {PDF_MINIMUM_ACCURACY}")
    a = ap.parse_args()

    if not os.path.isfile(a.gate_report):
        print(f"FAIL  missing gate report: {a.gate_report}")
        print("      Refit with: fit_heston_mixture_oracle.py --workers 16 (NOT the "
              "default 24 -- this machine caps at 16 physical cores)")
        return 1

    r = json.load(open(a.gate_report))
    fails: list[str] = []

    # --- 1. the gate itself
    acc = float(r["eight_regime_accuracy"])
    if acc < a.minimum_accuracy:
        fails.append(f"GATE FAILED: eight_regime_accuracy {acc!r} < {a.minimum_accuracy}. "
                     "No Experiment B metric computed against this oracle is valid.")

    # --- 2. internal coherence: a hand-edited accuracy cannot fake the trace
    cm = r.get("confusion_matrix")
    if not (isinstance(cm, list) and len(cm) == N_REGIMES
            and all(isinstance(row, list) and len(row) == N_REGIMES for row in cm)):
        fails.append(f"confusion_matrix is not {N_REGIMES}x{N_REGIMES}")
    else:
        total = sum(sum(row) for row in cm)
        trace = sum(cm[i][i] for i in range(N_REGIMES))
        n_val = int(r["validation_paths"])
        if total != n_val:
            fails.append(f"confusion_matrix sums to {total}, validation_paths says {n_val}")
        row_sums = [sum(row) for row in cm]
        col_sums = [sum(cm[i][j] for i in range(N_REGIMES)) for j in range(N_REGIMES)]
        if row_sums != list(r["validation_true_counts"]):
            fails.append(f"row sums {row_sums} != validation_true_counts "
                         f"{r['validation_true_counts']}")
        if col_sums != list(r["validation_predicted_counts"]):
            fails.append(f"column sums {col_sums} != validation_predicted_counts "
                         f"{r['validation_predicted_counts']}")
        if total and abs(trace / total - acc) > 1e-12:
            fails.append(f"trace/total = {trace}/{total} = {trace / total!r} does not "
                         f"reproduce eight_regime_accuracy {acc!r}")

    # --- 3. a laxer local threshold must not pass for the PDF's
    cfg_min = float(r.get("configuration", {}).get("minimum_accuracy", PDF_MINIMUM_ACCURACY))
    if cfg_min < PDF_MINIMUM_ACCURACY:
        fails.append(f"the fit was run with --minimum-accuracy {cfg_min} < the PDF's "
                     f"{PDF_MINIMUM_ACCURACY}; its 'pass' is not the protocol's pass")

    # --- 4. staleness (Gap 1)
    if a.oracle:
        if not os.path.isfile(a.oracle):
            print(f"NOTE  {a.oracle} is absent (it is gitignored). Refit locally before "
                  "scoring; the gate report alone does not let you score.")
        elif os.path.getmtime(a.oracle) > os.path.getmtime(a.gate_report) + 1.0:
            fails.append(f"STALE: {os.path.basename(a.oracle)} is newer than the gate "
                         "report, so the report may describe a different forest. This is "
                         "the exact state a failed refit leaves behind (Gap 1). Refit and "
                         "re-run this check.")

    if fails:
        print(f"FAIL  oracle gate preflight: {len(fails)} problem(s)\n")
        for f in fails:
            print(f"  {f}")
        return 1

    print(f"PASS  oracle gate: eight_regime_accuracy = {acc!r} >= {a.minimum_accuracy} "
          f"(margin +{acc - a.minimum_accuracy:.6f})")
    print(f"      per-parameter: {r['parameter_state_accuracy']}")
    print(f"      confusion matrix coherent over {r['validation_paths']} validation paths; "
          f"trace reproduces the accuracy exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
