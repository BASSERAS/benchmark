#!/usr/bin/env python3
"""Aggregate a frozen four-seed delayed-memory Deep MKV validation campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


SEEDS = (0, 1, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed0-evaluation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_sd": float(stdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else run_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: dict[str, object] = {}
    groups: dict[str, list[float]] = {}
    metrics: dict[str, list[float]] = {}
    wall_times: list[float] = []
    peak_memory: list[float] = []
    complete = 0
    passed = 0
    for seed in SEEDS:
        path = (
            args.seed0_evaluation.resolve()
            if seed == 0 and args.seed0_evaluation is not None
            else run_root / f"seed_{seed}" / "SELECTED_METHOD_VALIDATION.json"
        )
        if not path.is_file():
            per_seed[str(seed)] = {"status": "missing", "path": str(path)}
            continue
        payload = read_json(path)
        gate = payload["gate"]
        seed_groups = {
            name: float(values["median_gap_closed"])
            for name, values in payload["groups"].items()
        }
        seed_metrics = {
            name: float(value)
            for name, value in payload["candidate_gap_closed"].items()
            if value is not None and math.isfinite(float(value))
        }
        for name, value in seed_groups.items():
            groups.setdefault(name, []).append(value)
        for name, value in seed_metrics.items():
            metrics.setdefault(name, []).append(value)
        candidate_resources = payload["arms"][
            "generic_ce_online_source_volatility_only"
        ]["training_resources"]
        wall = float(
            candidate_resources.get(
                "training_wall_seconds",
                candidate_resources.get("wall_seconds", math.nan),
            )
        )
        memory = float(
            candidate_resources.get(
                "peak_gpu_allocated_gb",
                float(
                    candidate_resources.get(
                        "cuda_incremental_peak_allocated_bytes", math.nan
                    )
                )
                / 2**30,
            )
        )
        if math.isfinite(wall):
            wall_times.append(wall)
        if math.isfinite(memory):
            peak_memory.append(memory)
        advance = bool(gate["advance"])
        per_seed[str(seed)] = {
            "status": "complete",
            "gate_pass": advance,
            "groups": seed_groups,
            "metric_gap_closed": seed_metrics,
            "checks": gate["checks"],
            "scale_tail": payload["arms"][
                "generic_ce_online_source_volatility_only"
            ]["scale_tail"],
            "history_safety": payload["arms"][
                "generic_ce_online_source_volatility_only"
            ].get("history_safety"),
            "evaluation": str(path),
        }
        complete += 1
        passed += int(advance)

    result = {
        "seeds": list(SEEDS),
        "complete_count": complete,
        "failure_or_missing_count": len(SEEDS) - complete,
        "gate_pass_count": passed,
        "all_complete_seeds_pass": complete == len(SEEDS) and passed == len(SEEDS),
        "per_seed": per_seed,
        "group_gap_closed": {name: stats(values) for name, values in groups.items()},
        "metric_gap_closed": {name: stats(values) for name, values in metrics.items()},
        "resources": {
            "wall_seconds": stats(wall_times) if wall_times else None,
            "peak_gpu_allocated_gb": stats(peak_memory) if peak_memory else None,
        },
        "test_split_used": False,
    }
    json_path = output_dir / "AGGREGATE.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Selected online volatility-only delayed-memory campaign",
        "",
        f"Completed seeds: **{complete}/4**. Gate passes: **{passed}/4**.",
        "",
        "| Seed | Status | Distribution gap closed | Dependence gap closed | Local calibration |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = per_seed[str(seed)]
        if row["status"] != "complete":
            lines.append(f"| {seed} | missing | — | — | — |")
            continue
        seed_groups = row["groups"]
        lines.append(
            f"| {seed} | {'pass' if row['gate_pass'] else 'fail'} | "
            f"{100.0 * seed_groups['distribution']:.1f}% | "
            f"{100.0 * seed_groups['dependence']:.1f}% | "
            f"{100.0 * seed_groups['local_calibration']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Failures, saturation, nonfinite outcomes, and missing seeds remain in the denominator.",
            "No protected test split was used.",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
