#!/usr/bin/env python3
"""Aggregate the frozen four-index real-data panel used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


INDICES = ("ES", "NQ", "RTY", "YM")
SEEDS = (0, 1, 3, 4)
LAW_METRICS = (
    "return_qq_rmse_normalized",
    "abs_return_acf_error_rms",
    "realized_volatility_wasserstein_normalized",
    "maximum_drawdown_wasserstein_normalized",
)
SHADOW_METRICS = (
    "prefix_distance_mean",
    "realized_volatility_crps",
    "realized_volatility_coverage_90",
    "realized_volatility_band_width_90",
    "cumulative_return_crps",
    "increment_crps",
)


def parse_index_root(value: str) -> tuple[str, Path]:
    try:
        index, root = value.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected INDEX=PATH") from exc
    if index not in INDICES:
        raise argparse.ArgumentTypeError(f"unsupported index {index!r}")
    return index, Path(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deep-index-root",
        action="append",
        type=parse_index_root,
        required=True,
        help="Repeat as INDEX=PATH; PATH contains seed_<SEED> directories.",
    )
    parser.add_argument("--sbts-root", type=Path, required=True)
    parser.add_argument("--moving-block-root", type=Path)
    parser.add_argument("--whole-session-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mean_sd(rows: list[dict[str, float]], metrics: tuple[str, ...]) -> dict:
    result = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    return result


def method_summary(index_root: Path) -> dict:
    primary_law = []
    primary_shadow = []
    normal_shadow = []
    stress_shadow = []
    for seed in SEEDS:
        primary = load_json(
            index_root / f"seed_{seed}" / "primary_2026_holdout" / "metrics.json"
        )["phases"]["test"]
        event = load_json(
            index_root / f"seed_{seed}" / "us_iran_2026_war" / "metrics.json"
        )["phases"]
        primary_law.append(primary["law_metrics"])
        primary_shadow.append(primary["path_shadowing_metrics"])
        normal_shadow.append(event["matched_prewar"]["path_shadowing_metrics"])
        stress_shadow.append(event["war"]["path_shadowing_metrics"])
    return {
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "primary_law": mean_sd(primary_law, LAW_METRICS),
        "primary_shadow": mean_sd(primary_shadow, SHADOW_METRICS),
        "matched_normal_shadow": mean_sd(normal_shadow, SHADOW_METRICS),
        "stress_shadow": mean_sd(stress_shadow, SHADOW_METRICS),
    }


def one_seed_summary(index_root: Path) -> dict:
    primary = load_json(
        index_root / "seed_0" / "primary_2026_holdout" / "metrics.json"
    )["phases"]["test"]
    event = load_json(
        index_root / "seed_0" / "us_iran_2026_war" / "metrics.json"
    )["phases"]
    return {
        "seed_count": 1,
        "seeds": [0],
        "primary_law": mean_sd([primary["law_metrics"]], LAW_METRICS),
        "primary_shadow": mean_sd(
            [primary["path_shadowing_metrics"]], SHADOW_METRICS
        ),
        "matched_normal_shadow": mean_sd(
            [event["matched_prewar"]["path_shadowing_metrics"]], SHADOW_METRICS
        ),
        "stress_shadow": mean_sd(
            [event["war"]["path_shadowing_metrics"]], SHADOW_METRICS
        ),
    }


def main() -> None:
    args = parse_args()
    deep_roots = dict(args.deep_index_root)
    if set(deep_roots) != set(INDICES):
        raise ValueError(f"provide exactly one Deep root for each of {INDICES}")

    payload = {"indices": {}, "metric_direction": "lower_except_coverage"}
    for index in INDICES:
        methods = {
            "Deep MKV": method_summary(deep_roots[index]),
            "SBTS": method_summary(args.sbts_root / index),
        }
        if args.moving_block_root is not None:
            methods["Moving-block bootstrap"] = one_seed_summary(
                args.moving_block_root / index
            )
        if args.whole_session_root is not None:
            methods["Whole-session bootstrap"] = one_seed_summary(
                args.whole_session_root / index
            )
        payload["indices"][index] = methods

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
