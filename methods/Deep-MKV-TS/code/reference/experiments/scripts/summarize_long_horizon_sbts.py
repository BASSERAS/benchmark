#!/usr/bin/env python3
"""Aggregate the locked SBTS/MP/direct delayed-memory horizon comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from evaluate_long_horizon_sbts import evaluate_bank  # noqa: E402


METHODS = (
    "reference",
    "sbts",
    "volatility_only_nested_mp",
    "direct_pathwise",
)
EXISTING_ARMS = {
    "reference": "generic_reference_only",
    "volatility_only_nested_mp": "generic_ce_nested_volatility_only",
    "direct_pathwise": "direct_bptt_volatility_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _closure(reference: float, candidate: float) -> float | None:
    if not math.isfinite(reference) or reference <= 1e-12 or not math.isfinite(candidate):
        return None
    return float((reference - candidate) / reference)


def _group_closure(
    reference: dict[str, float],
    candidate: dict[str, float],
    metric_names: list[str],
) -> dict[str, object]:
    values = [
        value
        for name in metric_names
        if (value := _closure(reference[name], candidate[name])) is not None
    ]
    return {
        "metric_count": len(values),
        "improved_count": sum(value > 0.0 for value in values),
        "median_gap_closed": float(median(values)) if values else None,
        "mean_gap_closed": float(np.mean(values)) if values else None,
    }


def _existing_resource(manifest: dict[str, object]) -> dict[str, float]:
    resources = manifest["training_resources"]
    return {
        "training_wall_seconds": float(resources["training_wall_seconds"]),
        "cuda_incremental_peak_allocated_bytes": float(
            resources["cuda_incremental_peak_allocated_bytes"]
        ),
    }


def _failure_reason(seed_dir: Path, arm: str) -> str:
    log = seed_dir.parent / f"seed_{seed_dir.name.removeprefix('seed_')}.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace").lower()
        if (
            "drawdown groups must both be non-empty" in text
            and "sigma=0.6000 grad=0" in text
        ):
            return "volatility_saturation_empty_conditional_group"
        if "out of memory" in text:
            return "out_of_memory"
        if "nonfinite" in text or "non-finite" in text or "nan" in text:
            return "nonfinite_or_nan"
        if "failed" in text or "traceback" in text:
            return "runtime_failure"
    return f"missing_or_invalid_{arm}_artifact"


def _write_report(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Long-horizon delayed-memory comparison",
        "",
        "Validation data (`disc.npy`) only; every failed or missing seed remains in completion counts.",
        "",
        "## Decisive reference-relative table",
        "",
        "Cells are available-run means of the per-seed median reference-relative gap closure: distribution / dependence / path memory. Reference is zero by definition; completion denominators appear below.",
        "",
        "| Horizon | Reference | SBTS | Vol-only nested MP | Direct pathwise |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in (128, 256, 512):
        aggregate = payload["horizons"][str(horizon)]["aggregate"]  # type: ignore[index]
        cells = {"reference": "0.0% / 0.0% / 0.0%"}
        for method in METHODS[1:]:
            row = aggregate[method]
            if not row.get("group_gap_closed"):
                cells[method] = "--"
                continue
            groups = row["group_gap_closed"]
            cells[method] = " / ".join(
                f"{100.0 * float(groups[group]['mean']):+.1f}%"
                for group in ("distribution", "dependence", "path_memory")
            )
        lines.append(
            f"| {horizon} | {cells['reference']} | {cells['sbts']} | "
            f"{cells['volatility_only_nested_mp']} | {cells['direct_pathwise']} |"
        )

    lines.extend([
        "",
        "## Raw path and memory metrics",
        "",
        "Successful-run mean (sample SD). Lower is better; completion denominators appear below.",
        "",
        "| N | Method | Path MMD | Path SWD | Abs-ACF | Sq-ACF | Delay gap | Early R2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    labels = {
        "reference": "Reference",
        "sbts": "SBTS",
        "volatility_only_nested_mp": "Vol-only nested MP",
        "direct_pathwise": "Direct pathwise",
    }
    metric_names = (
        "path_mmd", "path_swd_normalized", "absolute_return_acf_rmse",
        "squared_return_acf_rmse", "future_rv_hit_gap_error",
        "early_history_incremental_r2_error",
    )
    for horizon in (128, 256, 512):
        aggregate = payload["horizons"][str(horizon)]["aggregate"]  # type: ignore[index]
        for method in METHODS:
            row = aggregate[method]
            if not row.get("metrics"):
                values = ["--"] * len(metric_names)
            else:
                values = [
                    f"{row['metrics'][name]['mean']:.4f} ({row['metrics'][name]['sample_sd']:.4f})"
                    for name in metric_names
                ]
            lines.append(
                f"| {horizon} | {labels[method]} | " + " | ".join(values) + " |"
            )

    lines.extend([
        "",
        "## Completion and resources",
        "",
        "| N | Method | Complete | Failed/missing | Wall min | Peak GPU GB | Failure reasons |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    for horizon in (128, 256, 512):
        aggregate = payload["horizons"][str(horizon)]["aggregate"]  # type: ignore[index]
        for method in METHODS:
            row = aggregate[method]
            resources = row.get("training_resources")
            wall = "--" if not resources else f"{resources['training_wall_seconds']['mean'] / 60.0:.2f}"
            gpu = "--" if not resources else f"{resources['cuda_incremental_peak_allocated_bytes']['mean'] / 2**30:.2f}"
            reasons = ", ".join(row.get("failure_reasons", [])) or "--"
            lines.append(
                f"| {horizon} | {labels[method]} | {row['complete_count']}/4 | "
                f"{row['failure_or_missing_count']}/4 | {wall} | {gpu} | {reasons} |"
            )

    lines.extend([
        "",
        "## Broad distribution details",
        "",
        "Successful-run mean (sample SD); lower is better.",
        "",
        "| N | Method | Terminal W1 | Increment W1 | RV W1 | Drawdown W1 | QQ RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    distribution_names = (
        "terminal_w1_normalized", "increment_w1_normalized",
        "realized_volatility_w1_normalized", "maximum_drawdown_w1_normalized",
        "return_qq_rmse_normalized",
    )
    for horizon in (128, 256, 512):
        aggregate = payload["horizons"][str(horizon)]["aggregate"]  # type: ignore[index]
        for method in METHODS:
            row = aggregate[method]
            values = (
                ["--"] * len(distribution_names)
                if not row.get("metrics")
                else [
                    f"{row['metrics'][name]['mean']:.4f} ({row['metrics'][name]['sample_sd']:.4f})"
                    for name in distribution_names
                ]
            )
            lines.append(
                f"| {horizon} | {labels[method]} | " + " | ".join(values) + " |"
            )

    lines.extend([
        "",
        "## Additional dependence, memory, and local calibration",
        "",
        "Successful-run mean (sample SD); lower is better.",
        "",
        "| N | Method | Return ACF | Early corr. | Hit-rate | Local mean | Local std |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    additional_names = (
        "return_acf_rmse", "early_hit_future_rv_correlation_error",
        "early_hit_rate_error", "timewise_return_mean_rmse_normalized",
        "timewise_return_std_rmse_normalized",
    )
    for horizon in (128, 256, 512):
        aggregate = payload["horizons"][str(horizon)]["aggregate"]  # type: ignore[index]
        for method in METHODS:
            row = aggregate[method]
            values = (
                ["--"] * len(additional_names)
                if not row.get("metrics")
                else [
                    f"{row['metrics'][name]['mean']:.4f} ({row['metrics'][name]['sample_sd']:.4f})"
                    for name in additional_names
                ]
            )
            lines.append(
                f"| {horizon} | {labels[method]} | " + " | ".join(values) + " |"
            )

    lines.extend([
        "",
        "## Protocol and decision boundary",
        "",
        f"- N=128 reproduction gate: {'PASS' if payload['n128_reproduction_gate_passed'] else 'FAIL'}.",
        "- Path MMD and SWD use exactly the locked horizon/seed subsamples for every method.",
        "- No final test array was loaded.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = _read(args.protocol.resolve())
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("comparison protocol is not validation-only")
    run_root = args.run_root.resolve()
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    device = torch.device(args.device)
    horizon_payloads: dict[str, object] = {}

    for horizon in (128, 256, 512):
        horizon_protocol = protocol["horizons"][str(horizon)]
        dataset_root = Path(horizon_protocol["dataset_root"])
        existing_root = Path(horizon_protocol["existing_method_run_root"])
        existing_summary = _read(Path(horizon_protocol["existing_summary"]))
        per_seed: dict[str, object] = {}
        for seed in seeds:
            seed_rows: dict[str, object] = {}
            existing_seed = existing_summary["per_seed"][str(seed)]
            for method, arm in EXISTING_ARMS.items():
                status = str(existing_seed[arm]["status"])
                arm_dir = existing_root / f"seed_{seed}" / arm
                if status != "complete":
                    seed_rows[method] = {
                        "status": "failed_or_missing",
                        "failure_reason": _failure_reason(existing_root / f"seed_{seed}", arm),
                    }
                    continue
                manifest = _read(arm_dir / "training_manifest.json")
                if bool(manifest.get("test_split_used", True)):
                    raise RuntimeError(f"{arm_dir} does not certify validation-only use")
                bank = arm_dir / f"generated_paths_8192x{horizon}.npy"
                evaluation = evaluate_bank(
                    dataset_root=dataset_root,
                    generated_path=bank,
                    sequence_length=horizon,
                    seed=seed,
                    subsample_size=int(protocol["evaluation"]["subsample_size"]),
                    swd_projections=int(protocol["evaluation"]["swd_projections"]),
                    device=device,
                )
                seed_rows[method] = {
                    "status": "complete",
                    "metrics": evaluation["metrics"],
                    "training_resources": _existing_resource(manifest),
                }

            sbts_dir = run_root / f"n{horizon}" / f"seed_{seed}" / "sbts"
            if (sbts_dir / "COMPLETE.json").is_file():
                complete = _read(sbts_dir / "COMPLETE.json")
                if bool(complete.get("test_split_used", True)):
                    raise RuntimeError(f"{sbts_dir} does not certify validation-only use")
                metrics_payload = _read(sbts_dir / "validation_metrics.json")
                seed_rows["sbts"] = {
                    "status": "complete",
                    "metrics": metrics_payload["metrics"],
                    "training_resources": metrics_payload["training_resources"],
                }
            else:
                failure_path = sbts_dir / "FAILURE.json"
                failure = _read(failure_path) if failure_path.is_file() else {}
                seed_rows["sbts"] = {
                    "status": "failed_or_missing",
                    "failure_reason": str(failure.get("error_type", "missing_artifact")),
                }

            reference = seed_rows["reference"]
            if reference["status"] != "complete":
                raise RuntimeError(f"reference is incomplete at N={horizon}, seed={seed}")
            reference_metrics = reference["metrics"]
            for method in METHODS[1:]:
                row = seed_rows[method]
                if row["status"] != "complete":
                    continue
                row["gap_closed"] = {
                    name: _closure(float(reference_metrics[name]), float(row["metrics"][name]))
                    for name in reference_metrics
                }
                row["group_gap_closed"] = {
                    group: _group_closure(
                        reference_metrics,
                        row["metrics"],
                        list(names),
                    )
                    for group, names in protocol["evaluation"]["metric_groups"].items()
                }
            per_seed[str(seed)] = seed_rows

        aggregate: dict[str, object] = {}
        for method in METHODS:
            complete_rows = [
                per_seed[str(seed)][method]
                for seed in seeds
                if per_seed[str(seed)][method]["status"] == "complete"
            ]
            failures = [
                str(per_seed[str(seed)][method].get("failure_reason", "unknown"))
                for seed in seeds
                if per_seed[str(seed)][method]["status"] != "complete"
            ]
            row: dict[str, object] = {
                "complete_count": len(complete_rows),
                "failure_or_missing_count": len(seeds) - len(complete_rows),
                "failure_reasons": failures,
            }
            if complete_rows:
                metric_names = tuple(complete_rows[0]["metrics"])
                row["metrics"] = {
                    name: _summary([float(item["metrics"][name]) for item in complete_rows])
                    for name in metric_names
                }
                row["training_resources"] = {
                    name: _summary(
                        [float(item["training_resources"][name]) for item in complete_rows]
                    )
                    for name in (
                        "training_wall_seconds",
                        "cuda_incremental_peak_allocated_bytes",
                    )
                }
            closure_rows = [item for item in complete_rows if "gap_closed" in item]
            if closure_rows:
                row["gap_closed"] = {
                    name: _summary(
                        [float(item["gap_closed"][name]) for item in closure_rows]
                    )
                    for name in closure_rows[0]["gap_closed"]
                    if all(item["gap_closed"][name] is not None for item in closure_rows)
                }
                row["group_gap_closed"] = {
                    group: _summary(
                        [
                            float(item["group_gap_closed"][group]["median_gap_closed"])
                            for item in closure_rows
                        ]
                    )
                    for group in protocol["evaluation"]["metric_groups"]
                }
            aggregate[method] = row
        horizon_payloads[str(horizon)] = {
            "per_seed": per_seed,
            "aggregate": aggregate,
        }

    gate_path = run_root / "n128" / "seed_0" / "sbts" / "REPRODUCTION_GATE.json"
    gate = _read(gate_path)
    payload = {
        "protocol": str(args.protocol.resolve()),
        "test_split_used": False,
        "seeds": list(seeds),
        "horizons": horizon_payloads,
        "n128_reproduction_gate_passed": bool(gate["pass"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(payload, args.output.with_name("REPORT.md"))


if __name__ == "__main__":
    main()
