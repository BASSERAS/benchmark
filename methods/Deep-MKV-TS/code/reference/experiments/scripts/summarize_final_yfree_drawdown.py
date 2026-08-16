#!/usr/bin/env python3
"""Aggregate the frozen four-seed ES drawdown-risk evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO_ROOT / "runs" / "es_drawdown_risk_yfree_test_20260808"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "es_drawdown_risk_yfree_aggregate_20260808"
SEEDS = (0, 1, 3, 4)
METRICS = (
    "leverage",
    "violation",
    "violation_excess",
    "realized_leveraged_drawdown",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_808)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            metric: np.asarray(payload[metric], dtype=np.float64).reshape(-1)
            for metric in METRICS
        }


def _method_paths(run_root: Path) -> dict[str, list[Path]]:
    return {
        "Deep MKV": [
            run_root / f"deep_seed{seed}" / "test" / "path_metrics.npz"
            for seed in SEEDS
        ],
        "Fitted reference": [
            run_root / f"reference_seed{seed}" / "test" / "path_metrics.npz"
            for seed in SEEDS
        ],
        "SBTS": [run_root / "sbts" / "test" / "path_metrics.npz"],
        "Moving-block bootstrap": [
            run_root / "moving_block" / "test" / "path_metrics.npz"
        ],
        "Whole-session bootstrap": [
            run_root / "whole_session" / "test" / "path_metrics.npz"
        ],
        "Historical neighbor": [
            run_root / "historical_neighbor" / "test" / "path_metrics.npz"
        ],
    }


def _moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    generator: np.random.Generator,
) -> np.ndarray:
    count = int(values.size)
    blocks = (count + block_length - 1) // block_length
    starts = generator.integers(
        0,
        count - block_length + 1,
        size=(replicates, blocks),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)
    return values[indices[:, :count]].mean(axis=1)


def _interval(
    values: np.ndarray,
    *,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, float]:
    bootstrap = _moving_block_means(
        values,
        block_length=int(args.block_length),
        replicates=int(args.replicates),
        generator=np.random.default_rng(int(args.seed) + seed_offset),
    )
    return {
        "estimate": float(values.mean()),
        "ci_lower": float(np.quantile(bootstrap, 0.025)),
        "ci_upper": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_standard_error": float(bootstrap.std(ddof=0)),
    }


def main() -> None:
    args = _parse_args()
    paths = _method_paths(args.run_root.resolve())
    values: dict[str, dict[str, np.ndarray]] = {}
    for method, files in paths.items():
        if not all(path.is_file() for path in files):
            raise FileNotFoundError(f"missing input for {method}: {files}")
        loaded = [_load(path) for path in files]
        values[method] = {
            metric: np.stack([row[metric] for row in loaded]).mean(axis=0)
            for metric in METRICS
        }

    methods: dict[str, dict[str, object]] = {}
    counter = 0
    for method, metric_values in values.items():
        methods[method] = {
            "bank_count": len(paths[method]),
            "metrics": {},
        }
        for metric, array in metric_values.items():
            methods[method]["metrics"][metric] = _interval(
                array,
                args=args,
                seed_offset=counter,
            )
            counter += 1

    paired = {}
    for comparison in values:
        if comparison == "Deep MKV":
            continue
        paired[comparison] = {}
        for metric in METRICS:
            paired[comparison][metric] = _interval(
                values["Deep MKV"][metric] - values[comparison][metric],
                args=args,
                seed_offset=counter,
            )
            counter += 1

    report = json.loads(
        (args.run_root.resolve() / "report.json").read_text(encoding="utf-8")
    )
    payload = {
        "scope": {
            "index": "ES",
            "test_sessions": int(values["Deep MKV"]["leverage"].size),
            "deep_and_reference_seeds": list(SEEDS),
            "safety_calibration": "validation only, separately by generated bank",
            "test_used_for_selection": False,
            "risk_budget": float(report["base_config"]["risk_budget"]),
            "drawdown_quantile": float(report["base_config"]["drawdown_quantile"]),
        },
        "bootstrap": {
            "replicates": int(args.replicates),
            "block_length_sessions": int(args.block_length),
            "confidence_level": 0.95,
            "seed": int(args.seed),
        },
        "methods": methods,
        "paired_deep_minus_comparison": paired,
    }

    lines = [
        "# Final ES drawdown-risk aggregate",
        "",
        "Each Deep MKV and fitted-reference entry first averages the four frozen "
        "generated-bank seeds for each test session. Safety factors were fitted "
        "on validation and frozen before the 123 chronological test sessions were read.",
        "",
        "| Method | Avg. leverage | Violation rate | Mean excess | Realized drawdown |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = methods[method]["metrics"]
        lines.append(
            f"| {method} | {row['leverage']['estimate']:.3f} | "
            f"{100*row['violation']['estimate']:.2f}% | "
            f"{row['violation_excess']['estimate']:.6f} | "
            f"{row['realized_leveraged_drawdown']['estimate']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Deep MKV differences",
            "",
            "Entries are Deep MKV minus the comparison with 95% moving-session-block intervals. Positive leverage favors Deep MKV; negative risk quantities favor Deep MKV.",
            "",
            "| Comparison | Leverage | Violation rate | Mean excess |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, row in paired.items():
        leverage = row["leverage"]
        violation = row["violation"]
        excess = row["violation_excess"]
        lines.append(
            f"| {comparison} | {leverage['estimate']:.3f} "
            f"[{leverage['ci_lower']:.3f}, {leverage['ci_upper']:.3f}] | "
            f"{100*violation['estimate']:.2f} pp "
            f"[{100*violation['ci_lower']:.2f}, {100*violation['ci_upper']:.2f}] | "
            f"{excess['estimate']:.6f} "
            f"[{excess['ci_lower']:.6f}, {excess['ci_upper']:.6f}] |"
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "AGGREGATE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = "\n".join(lines) + "\n"
    (output_root / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
