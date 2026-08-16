#!/usr/bin/env python3
"""Aggregate the 4-seed Deep-MKV-TS Heston reproduction into a paper-vs-ours table.

Reads runs/seed_{0,1,3,4}/{reference,volatility_only_online_mp}/metrics.json
produced by the vendored run_matched_control_synthetic_validation.py and emits
results/paper_vs_ours.json + results/PAPER_VS_OURS.md.

The five reported quantities are exactly the paper's Heston table columns; each
is read from the upstream metric implementation, none is recomputed here.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from statistics import median

PR_ROOT = Path(__file__).resolve().parent.parent
RUNS = PR_ROOT / "runs"
RESULTS = PR_ROOT / "results"
SEEDS = (0, 1, 3, 4)
# The campaign trains 3000 steps and reports the step-2500 checkpoint; that is
# how the paper's stated K=2500 is realised. The frozen reference arm has no
# training trajectory, so it is read from its single metrics.json.
ARMS = {
    "Reference": "reference/metrics.json",
    "Deep-MKV-TS": "volatility_only_online_mp/checkpoint_evaluations/step_2500/metrics.json",
}

# Columns as printed in the paper's Heston table (lower is better throughout).
COLUMNS = [
    ("SWD", "path_swd_normalized"),
    ("RV W1", "realized_volatility_w1_normalized"),
    ("|r| ACF", "absolute_return_acf_rmse"),
    ("Early-future", "prefix_future_rv_correlation_error"),
    ("MDD W1", "maximum_drawdown_w1_normalized"),
]

# Deep-MKV-TS paper, Heston table (medians over four seeds).
PAPER = {
    "Reference": [0.060, 0.089, 0.068, 0.214, 0.059],
    "Deep-MKV-TS": [0.062, 0.014, 0.016, 0.018, 0.022],
}

# The published Table 1 was generated with the `local_gaussian` reference (a
# per-step ridge on the raw complete prefix). Paper Section 2.1 instead
# describes the price-only Guyon-Lekeufack model fitted by Gaussian
# quasi-likelihood on 80% of the training paths, and the author confirmed
# Guyon-Lekeufack is the correct reference. We therefore run Guyon-Lekeufack,
# which is a materially stronger starting point, so our Reference row is
# expected to sit well below the published one. Scoring that row against the
# published one would flag a difference we introduced deliberately, so it is
# reported for context and excluded from the pass count.
SCORED_ROWS = ("Deep-MKV-TS",)
REFERENCE_NOTE = (
    "The Reference row is informational, not a reproduction target: the "
    "published row came from the `local_gaussian` reference, while this "
    "reimplementation uses the Guyon-Lekeufack reference that paper Section 2.1 "
    "describes. Only the Deep-MKV-TS row is scored against the paper."
)

# A metric counts as reproduced when it lands within 25% of the paper value, or
# within 0.005 absolute, whichever band is wider. The absolute floor keeps the
# near-zero columns from failing on rounding alone: the paper prints three
# decimals, so 0.014 carries +/-0.0005 of quantisation error before any
# seed-to-seed variation.
REL_TOL = 0.25
ABS_TOL = 0.005


def md_cell(text: str) -> str:
    """Escape pipes so a metric name like `|r| ACF` cannot split a table cell.

    The paper's third column is literally named with pipes around the r. Emitted
    raw, it adds two phantom columns and desynchronises every row below it.
    """
    return text.replace("|", "\\|")


def paper_metrics(metrics: dict) -> dict[str, float]:
    """Pull the five paper columns out of one upstream metrics.json."""
    path_law = metrics["path_law"]
    memory = metrics["conditional_path_memory"]
    out = {key: float(path_law[key]) for _, key in COLUMNS if key in path_law}
    out["prefix_future_rv_correlation_error"] = abs(
        float(memory["target"]["prefix_future_rv_correlation"])
        - float(memory["generated"]["prefix_future_rv_correlation"])
    )
    return out


def collect() -> dict[str, dict]:
    table: dict[str, dict] = {}
    for label, arm in ARMS.items():
        per_seed: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            path = RUNS / f"seed_{seed}" / arm
            if not path.is_file():
                raise FileNotFoundError(f"missing {path}; run run_reproduction.sh first")
            per_seed[seed] = paper_metrics(json.loads(path.read_text()))
        table[label] = {
            "per_seed": per_seed,
            "median": {
                key: median(per_seed[s][key] for s in SEEDS) for _, key in COLUMNS
            },
        }
    return table


def verdict(ours: float, paper: float) -> tuple[bool, float]:
    band = max(ABS_TOL, REL_TOL * abs(paper))
    return abs(ours - paper) <= band, ours - paper


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = collect()

    passes: list[bool] = []
    lines = [
        "# Deep-MKV-TS, paper reimplementation on Heston",
        "",
        "Medians over the paper's four training seeds (0, 1, 3, 4), 8192 paths per split,",
        "evaluated against `dataset/Heston/heston_S_disc_8192x128.npy`. Lower is better.",
        "",
        f"Tolerance: within {REL_TOL:.0%} of the paper value or {ABS_TOL} absolute, whichever is wider.",
        "",
        REFERENCE_NOTE,
        "",
        "| Row | Metric | Ours (median, 4 seeds) | Paper | Delta | Verdict |",
        "|-----|--------|-----------------------:|------:|------:|:-------:|",
    ]
    for label in ARMS:
        scored = label in SCORED_ROWS
        for idx, (name, key) in enumerate(COLUMNS):
            ours = table[label]["median"][key]
            ref = PAPER[label][idx]
            ok, delta = verdict(ours, ref)
            if scored:
                passes.append(ok)
                mark = "match" if ok else "MISS"
            else:
                mark = "better" if delta < 0 else "worse"
                mark = f"context ({mark})"
            lines.append(
                f"| {label} | {md_cell(name)} | {ours:.4f} | {ref:.3f} | "
                f"{delta:+.4f} | {mark} |"
            )

    lines += ["", "## Per-seed values", ""]
    for label in ARMS:
        lines += [
            f"### {label}",
            "",
            "| Seed | " + " | ".join(md_cell(n) for n, _ in COLUMNS) + " |",
            "|------|" + "|".join(["------:"] * len(COLUMNS)) + "|",
        ]
        for seed in SEEDS:
            row = table[label]["per_seed"][seed]
            lines.append(
                f"| {seed} | " + " | ".join(f"{row[k]:.4f}" for _, k in COLUMNS) + " |"
            )
        lines.append("")

    n_ok = sum(passes)
    lines += [
        f"**Overall: {n_ok}/{len(passes)} Deep-MKV-TS metrics within tolerance "
        f"(rows scored: {', '.join(SCORED_ROWS)}).**",
        "",
    ]

    (RESULTS / "PAPER_VS_OURS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "paper_vs_ours.json").write_text(
        json.dumps(
            {
                "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "seeds": list(SEEDS),
                "columns": [name for name, _ in COLUMNS],
                "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
                "paper": PAPER,
                "ours": {
                    label: {
                        "median": table[label]["median"],
                        "per_seed": {
                            str(s): table[label]["per_seed"][s] for s in SEEDS
                        },
                    }
                    for label in ARMS
                },
                "metrics_within_tolerance": n_ok,
                "metrics_total": len(passes),
                "scored_rows": list(SCORED_ROWS),
                "reference_note": REFERENCE_NOTE,
                "reference_kind": "guyon_lekeufack_structural_likelihood",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
