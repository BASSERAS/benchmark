#!/usr/bin/env python3
"""Aggregate frozen delayed-memory reference and solver controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "abs_return_acf_rmse_lags_1_50",
    "future_rv_wasserstein",
    "future_rv_hit_gap_error",
    "early_history_incremental_r2_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 3, 4))
    parser.add_argument(
        "--group",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH_TEMPLATE"),
        required=True,
        help="Repeat for each label; PATH_TEMPLATE must contain {seed}.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    sources: dict[str, str] = {}
    for label, template in args.group:
        if "{seed}" not in template:
            raise ValueError(f"path template for {label!r} must contain {{seed}}")
        label_rows: dict[str, dict[str, float]] = {}
        for seed in seeds:
            path = Path(template.format(seed=seed))
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors = payload["errors"]
            label_rows[str(seed)] = {
                metric: float(errors[metric]) for metric in METRICS
            }
        per_seed[label] = label_rows
        aggregate[label] = {
            metric: summarize(
                [label_rows[str(seed)][metric] for seed in seeds]
            )
            for metric in METRICS
        }
        sources[label] = template
    payload = {
        "seeds": list(seeds),
        "metric_direction": "lower_is_better",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
