#!/usr/bin/env python3
"""Rank Deep-MKV-TS hyperparameter trials scored on the validation splits.

Every trial under runs/hpsearch/<tag>/ was evaluated against
dataset/Heston/heston_S_valdisc_8192x128.npy (via BENCHMARK_HESTON_EVAL), never
against heston_S_disc_8192x128.npy. Only the single winning tag may later be
re-run on `disc`, exactly once, per GUIDELINE 11 (selection_scope=validation_only).

Score is the mean over the five paper columns of ours/paper; < 1 means the trial
beats the published Deep-MKV-TS row on average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_paper_table import COLUMNS, PAPER, md_cell, paper_metrics

PR_ROOT = Path(__file__).resolve().parent.parent
HPSEARCH = PR_ROOT / "runs" / "hpsearch"
ARM = "volatility_only_online_mp/checkpoint_evaluations/step_2500"

TARGET = dict(zip((key for _, key in COLUMNS), PAPER["Deep-MKV-TS"]))


def load_trials(root: Path) -> dict[str, dict[str, float]]:
    trials: dict[str, dict[str, float]] = {}
    for metrics_path in sorted(root.glob(f"*/{ARM}/metrics.json")):
        tag = metrics_path.parents[3].name
        trials[tag] = paper_metrics(json.loads(metrics_path.read_text()))
    return trials


def score(row: dict[str, float]) -> float:
    """Mean ratio to the published Deep-MKV-TS row; lower is better, <1 beats it."""
    return sum(row[key] / TARGET[key] for _, key in COLUMNS) / len(COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HPSEARCH)
    args = parser.parse_args()

    trials = load_trials(args.root)
    if not trials:
        raise SystemExit(f"no completed trials under {args.root}")

    ranked = sorted(trials.items(), key=lambda kv: score(kv[1]))

    lines = [
        "# Deep-MKV-TS hyperparameter search (scored on heston_S_valdisc)",
        "",
        "Score = mean over the five paper columns of ours/paper. Lower is better;",
        "below 1.000 means the trial beats the published Deep-MKV-TS row on average.",
        "",
        "| | " + " | ".join(md_cell(n) for n, _ in COLUMNS) + " |",
        "|---|" + "|".join(["---:"] * len(COLUMNS)) + "|",
        "| Paper | " + " | ".join(f"{TARGET[k]:.3f}" for _, k in COLUMNS) + " |",
        "",
        "| Rank | Trial | " + " | ".join(md_cell(n) for n, _ in COLUMNS) + " | Score |",
        "|---|---|" + "|".join(["---:"] * (len(COLUMNS) + 1)) + "|",
    ]
    for rank, (tag, row) in enumerate(ranked, start=1):
        cells = " | ".join(f"{row[key]:.4f}" for _, key in COLUMNS)
        lines.append(f"| {rank} | {tag} | {cells} | {score(row):.3f} |")

    best_tag, best_row = ranked[0]
    lines += [
        "",
        f"**Winner: `{best_tag}` (score {score(best_row):.3f}).**",
        "",
        "Selected on validation only. Score it on `disc` exactly once to produce the",
        "paper-vs-ours table; do not re-enter the search after touching `disc`.",
    ]

    out_md = args.root / "RANKING.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.root / "ranking.json").write_text(
        json.dumps(
            {
                "scored_split": "heston_S_valdisc_8192x128.npy",
                "columns": [name for name, _ in COLUMNS],
                "paper_target": TARGET,
                "winner": best_tag,
                "trials": {
                    tag: {"metrics": row, "score": score(row)} for tag, row in ranked
                },
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
