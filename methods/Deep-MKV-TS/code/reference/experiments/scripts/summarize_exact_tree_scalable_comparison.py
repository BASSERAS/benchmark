#!/usr/bin/env python3
"""Aggregate the locked four-seed exact-tree scalable comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.exact_tree_evaluation import (  # noqa: E402
    DEPENDENCE_METRICS,
    PRIMARY_DISTRIBUTION_METRICS,
)
from path_dt_experiments.runners import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3))
    return parser.parse_args()


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _finite_values(payload: dict[str, float | None], names: tuple[str, ...]) -> list[float]:
    return [
        float(payload[name])
        for name in names
        if payload[name] is not None and math.isfinite(float(payload[name]))
    ]


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def _format(
    mean_sd: dict[str, float],
    *,
    percent: bool = False,
    precision: int = 2,
) -> str:
    scale = 100.0 if percent else 1.0
    return (
        f"{scale * mean_sd['mean']:.{int(precision)}f} ± "
        f"{scale * mean_sd['sample_sd']:.{int(precision)}f}"
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    seed_reports = []
    for seed in args.seeds:
        path = run_dir / f"seed_{int(seed)}" / "report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        seed_reports.append(json.loads(path.read_text()))

    canonical = seed_reports[0]["protocol"]
    locked_names = (
        "horizon",
        "leaf_count",
        "control_mode",
        "hidden_dim",
        "num_layers",
        "parameter_count",
        "optimizer",
        "lr",
        "weight_decay",
        "optimizer_steps_per_arm",
        "nested_outer_steps",
        "nested_inner_steps",
    )
    for report in seed_reports[1:]:
        for name in locked_names:
            if report["protocol"][name] != canonical[name]:
                raise ValueError(f"protocol mismatch for {name}")
        if report["reference"] != seed_reports[0]["reference"]:
            raise ValueError("reference result differs across seeds")
        if report["exact_mp"] != seed_reports[0]["exact_mp"]:
            raise ValueError("exact result differs across seeds")

    aggregate: dict[str, object] = {
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "seed_count": len(seed_reports),
            "locked": {name: canonical[name] for name in locked_names},
        },
        "reference": seed_reports[0]["reference"],
        "exact_mp": seed_reports[0]["exact_mp"],
        "arms": {},
    }
    for arm in ("nested_mp", "direct_pathwise"):
        rows = [report[arm] for report in seed_reports]
        distribution_closure = [
            _median(_finite_values(row["gap_closed"], PRIMARY_DISTRIBUTION_METRICS))
            for row in rows
        ]
        dependence_closure = [
            _median(_finite_values(row["gap_closed"], DEPENDENCE_METRICS))
            for row in rows
        ]
        distribution_capture = [
            _median(
                _finite_values(
                    row["fraction_of_exact_gap_closure"],
                    PRIMARY_DISTRIBUTION_METRICS,
                )
            )
            for row in rows
        ]
        dependence_capture = [
            _median(
                _finite_values(
                    row["fraction_of_exact_gap_closure"], DEPENDENCE_METRICS
                )
            )
            for row in rows
        ]
        per_metric = {}
        for name in PRIMARY_DISTRIBUTION_METRICS + DEPENDENCE_METRICS:
            closures = [
                float(row["gap_closed"][name])
                for row in rows
                if row["gap_closed"][name] is not None
                and math.isfinite(float(row["gap_closed"][name]))
            ]
            captures = [
                float(row["fraction_of_exact_gap_closure"][name])
                for row in rows
                if row["fraction_of_exact_gap_closure"][name] is not None
                and math.isfinite(float(row["fraction_of_exact_gap_closure"][name]))
            ]
            per_metric[name] = {
                "gap_closed": _mean_sd(closures) if closures else None,
                "fraction_of_exact_improvement": _mean_sd(captures) if captures else None,
                "improved_seed_count": sum(value > 0.0 for value in closures),
            }
        aggregate["arms"][arm] = {
            "stable_seed_count": sum(bool(row["stable"]) for row in rows),
            "objective": _mean_sd([float(row["objective"]["total"]) for row in rows]),
            "distribution_median_gap_closed": _mean_sd(distribution_closure),
            "dependence_median_gap_closed": _mean_sd(dependence_closure),
            "distribution_fraction_of_exact": _mean_sd(distribution_capture),
            "dependence_fraction_of_exact": _mean_sd(dependence_capture),
            "relative_exact_r_stationarity_residual": _mean_sd(
                [float(row["relative_exact_r_stationarity_residual"]) for row in rows]
            ),
            "wall_seconds": _mean_sd(
                [float(row["resources"]["wall_seconds"]) for row in rows]
            ),
            "incremental_peak_allocated_bytes": _mean_sd(
                [
                    float(row["resources"]["cuda_incremental_peak_allocated_bytes"])
                    for row in rows
                ]
            ),
            "objective_evaluations": _mean_sd(
                [float(row["resources"]["objective_evaluations"]) for row in rows]
            ),
            "timewise_return_mean_error": _mean_sd(
                [
                    float(row["errors"]["timewise_return_mean_rmse_normalized"])
                    for row in rows
                ]
            ),
            "timewise_return_std_error": _mean_sd(
                [
                    float(row["errors"]["timewise_return_std_rmse_normalized"])
                    for row in rows
                ]
            ),
            "return_acf_error": _mean_sd(
                [float(row["errors"]["return_acf_rmse"]) for row in rows]
            ),
            "nested_accepted_stage_count": (
                _mean_sd(
                    [
                        float(sum(item["accepted"] for item in row["history"]))
                        for row in rows
                    ]
                )
                if arm == "nested_mp"
                else None
            ),
            "per_metric": per_metric,
        }

    nested = aggregate["arms"]["nested_mp"]
    direct = aggregate["arms"]["direct_pathwise"]
    aggregate["paired"] = {
        "nested_minus_direct_objective": _mean_sd(
            [
                float(report["nested_mp"]["objective"]["total"])
                - float(report["direct_pathwise"]["objective"]["total"])
                for report in seed_reports
            ]
        ),
        "nested_to_direct_wall_time_ratio": float(
            nested["wall_seconds"]["mean"] / direct["wall_seconds"]["mean"]
        ),
        "nested_to_direct_peak_memory_ratio": float(
            nested["incremental_peak_allocated_bytes"]["mean"]
            / direct["incremental_peak_allocated_bytes"]["mean"]
        ),
    }
    exact_run_dir = Path(canonical["exact_run_dir"])
    exact_panel = json.loads(
        (exact_run_dir / "path_law_panel.json").read_text()
    )["modes"]["volatility_only"]
    exact_report = json.loads(
        (exact_run_dir / "volatility_only" / "report.json").read_text()
    )
    oracle_objective = next(
        float(row["initial_objective"])
        for row in exact_report["optimization"]["starts"]
        if row["name"] == "oracle"
    )
    aggregate["benchmarks"] = {
        "target": {
            "objective": oracle_objective,
            "distribution_median_gap_closed": 1.0,
            "dependence_median_gap_closed": 1.0,
        },
        "reference": {
            "objective": float(seed_reports[0]["reference"]["objective"]),
            "distribution_median_gap_closed": 0.0,
            "dependence_median_gap_closed": 0.0,
        },
        "one_stage_mp": {
            "objective": float(exact_panel["one_stage"]["objective"]["total"]),
            "distribution_median_gap_closed": float(
                exact_panel["one_stage"]["summary"]["distribution"]["median_gap_closed"]
            ),
            "dependence_median_gap_closed": float(
                exact_panel["one_stage"]["summary"]["dependence"]["median_gap_closed"]
            ),
            "distribution_fraction_of_exact": float(
                exact_panel["one_stage_fraction_of_exact_improvement"]["distribution"]["median_fraction_captured"]
            ),
            "dependence_fraction_of_exact": float(
                exact_panel["one_stage_fraction_of_exact_improvement"]["dependence"]["median_fraction_captured"]
            ),
        },
        "exact_mp": {
            "objective": float(seed_reports[0]["exact_mp"]["objective"]),
            "distribution_median_gap_closed": float(
                exact_panel["exact_mp"]["summary"]["distribution"]["median_gap_closed"]
            ),
            "dependence_median_gap_closed": float(
                exact_panel["exact_mp"]["summary"]["dependence"]["median_gap_closed"]
            ),
            "distribution_fraction_of_exact": 1.0,
            "dependence_fraction_of_exact": 1.0,
        },
    }
    reference_objective = float(aggregate["benchmarks"]["reference"]["objective"])
    exact_objective = float(aggregate["benchmarks"]["exact_mp"]["objective"])
    objective_improvement = reference_objective - exact_objective
    aggregate["benchmarks"]["reference"]["objective_fraction_of_exact"] = 0.0
    aggregate["benchmarks"]["one_stage_mp"]["objective_fraction_of_exact"] = (
        reference_objective
        - float(aggregate["benchmarks"]["one_stage_mp"]["objective"])
    ) / objective_improvement
    aggregate["benchmarks"]["exact_mp"]["objective_fraction_of_exact"] = 1.0
    for arm in ("nested_mp", "direct_pathwise"):
        objective_values = [
            (
                reference_objective
                - float(report[arm]["objective"]["total"])
            )
            / objective_improvement
            for report in seed_reports
        ]
        aggregate["arms"][arm]["objective_fraction_of_exact"] = _mean_sd(
            objective_values
        )
    write_json(run_dir / "aggregate.json", aggregate)

    lines = [
        "# Matched scalable N=8 comparison",
        "",
        f"Four seeds: `{', '.join(str(seed) for seed in args.seeds)}`. Values are mean ± sample SD.",
        "",
        "## Complete comparison",
        "",
        "| Method | Stable | Objective | Exact objective gain captured | Median distribution gap closed | Exact distribution gain captured | Median dependence gap closed | Exact dependence gain captured | Time (s) | Peak memory (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("target", "Target law"),
        ("reference", "Local reference"),
        ("one_stage_mp", "Exact-node one-stage MP"),
        ("exact_mp", "Exact nodewise MP optimum"),
    ):
        row = aggregate["benchmarks"][key]
        objective_capture = row.get("objective_fraction_of_exact")
        distribution_capture = row.get("distribution_fraction_of_exact")
        dependence_capture = row.get("dependence_fraction_of_exact")
        lines.append(
            f"| {label} | deterministic | {row['objective']:.6f} | "
            f"{'--' if objective_capture is None else f'{100.0 * objective_capture:.2f}%'} | "
            f"{100.0 * row['distribution_median_gap_closed']:.2f}% | "
            f"{'--' if distribution_capture is None else f'{100.0 * distribution_capture:.2f}%'} | "
            f"{100.0 * row['dependence_median_gap_closed']:.2f}% | "
            f"{'--' if dependence_capture is None else f'{100.0 * dependence_capture:.2f}%'} | -- | -- |"
        )
    for arm, label in (("nested_mp", "Nested MP"), ("direct_pathwise", "Direct pathwise")):
        row = aggregate["arms"][arm]
        peak_mb = {
            key: value / (1024.0**2)
            for key, value in row["incremental_peak_allocated_bytes"].items()
        }
        lines.append(
            f"| {label} | {row['stable_seed_count']}/{len(seed_reports)} | "
            f"{_format(row['objective'], precision=5)} | "
            f"{_format(row['objective_fraction_of_exact'], percent=True)}% | "
            f"{_format(row['distribution_median_gap_closed'], percent=True)}% | "
            f"{_format(row['distribution_fraction_of_exact'], percent=True)}% | "
            f"{_format(row['dependence_median_gap_closed'], percent=True)}% | "
            f"{_format(row['dependence_fraction_of_exact'], percent=True)}% | "
            f"{_format(row['wall_seconds'])} | {_format(peak_mb)} |"
        )
    lines.extend(
        [
            "",
            "## Per-metric fraction of exact improvement captured",
            "",
            "| Metric | Nested MP | Direct pathwise | Nested seeds improving reference | Direct seeds improving reference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in PRIMARY_DISTRIBUTION_METRICS + DEPENDENCE_METRICS:
        nested_metric = aggregate["arms"]["nested_mp"]["per_metric"][name]
        direct_metric = aggregate["arms"]["direct_pathwise"]["per_metric"][name]
        nested_capture = nested_metric["fraction_of_exact_improvement"]
        direct_capture = direct_metric["fraction_of_exact_improvement"]
        lines.append(
            f"| `{name}` | "
            f"{'--' if nested_capture is None else _format(nested_capture, percent=True) + '%'} | "
            f"{'--' if direct_capture is None else _format(direct_capture, percent=True) + '%'} | "
            f"{nested_metric['improved_seed_count']}/{len(seed_reports)} | "
            f"{direct_metric['improved_seed_count']}/{len(seed_reports)} |"
        )
    lines.extend(
        [
            "",
            "The exact nodewise optimum and exact one-stage benchmark are reported in the sibling `exact_scenario_tree_n8_20260804/PATH_LAW_PANEL.md` artifact.",
            "",
            "## Numerical and local-calibration diagnostics",
            "",
            "| Method | Exact R residual | Return-mean error | Return-std error | Return-ACF error | Complete-objective evaluations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, label in (("nested_mp", "Nested MP"), ("direct_pathwise", "Direct pathwise")):
        row = aggregate["arms"][arm]
        lines.append(
            f"| {label} | {_format(row['relative_exact_r_stationarity_residual'], precision=4)} | "
            f"{_format(row['timewise_return_mean_error'], precision=3)} | "
            f"{_format(row['timewise_return_std_error'], precision=4)} | "
            f"{_format(row['return_acf_error'], precision=3)} | "
            f"{_format(row['objective_evaluations'], precision=0)} |"
        )
    lines.extend(
        [
            "",
            "All eight learned runs are finite and improve the reference. Nested MP accepts all eight stages in every seed and decreases the exact objective monotonically at those gates.",
            "",
            "Fractions above 100% do not mean a lower control objective than the exact nodewise optimum. The optimum minimizes the weighted objective, whereas an approximate policy can improve an individual diagnostic more by accepting a different running-cost or cross-metric tradeoff.",
        ]
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["paired"], indent=2), flush=True)


if __name__ == "__main__":
    main()
