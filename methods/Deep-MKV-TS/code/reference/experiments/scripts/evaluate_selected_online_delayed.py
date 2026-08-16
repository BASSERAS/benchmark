#!/usr/bin/env python3
"""Evaluate one selected online volatility-only delayed-memory run on validation data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from scripts.summarize_volatility_only_nested_scaling import (  # noqa: E402
    DEPENDENCE_METRICS,
    LOCAL_CALIBRATION_METRICS,
    PRIMARY_DISTRIBUTION_METRICS,
    _group_closure,
    _path_metrics,
)


REFERENCE_ARM = "generic_reference_only"
CANDIDATE_ARM = "generic_ce_online_source_volatility_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--candidate-evaluation-step",
        type=int,
        default=None,
        help="Evaluate an intermediate online checkpoint instead of the final model.",
    )
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path} must be a rank-two path bank")
    return values


def scale_tail(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    target_log = np.log(target / target[:, :1])
    generated_log = np.log(generated / generated[:, :1])
    target_returns = np.diff(target_log, axis=1)
    generated_returns = np.diff(generated_log, axis=1)
    target_rv = np.sqrt(np.sum(target_returns**2, axis=1))
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))
    return {
        "nonfinite_path_fraction": float(1.0 - np.isfinite(generated).mean()),
        "nonpositive_price_fraction": float(np.mean(generated <= 0.0)),
        "return_std_ratio": float(
            generated_returns.std() / max(target_returns.std(), 1e-12)
        ),
        "realized_volatility_mean_ratio": float(
            generated_rv.mean() / max(target_rv.mean(), 1e-12)
        ),
        "realized_volatility_std_ratio": float(
            generated_rv.std() / max(target_rv.std(), 1e-12)
        ),
        "absolute_return_q99_ratio": float(
            np.quantile(np.abs(generated_returns), 0.99)
            / max(np.quantile(np.abs(target_returns), 0.99), 1e-12)
        ),
        "absolute_return_q999_ratio": float(
            np.quantile(np.abs(generated_returns), 0.999)
            / max(np.quantile(np.abs(target_returns), 0.999), 1e-12)
        ),
    }


def history_safety(path: Path, *, maximum_step: int | None = None) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if maximum_step is not None:
        rows = [
            row for row in rows if int(float(row.get("step", 0.0))) <= int(maximum_step)
        ]
    caps = [
        float(row["sigma_max_active_fraction"])
        for row in rows
        if "sigma_max_active_fraction" in row
    ]
    nonfinite = [
        float(row.get("sigma_nonfinite_fraction", 0.0))
        for row in rows
    ]
    return {
        "logged_rows": float(len(rows)),
        "sigma_cap_fraction_final": float(caps[-1]) if caps else math.nan,
        "sigma_cap_fraction_max": float(max(caps)) if caps else math.nan,
        "sigma_nonfinite_fraction_max": float(max(nonfinite)) if nonfinite else 0.0,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    target_all = load_prices(dataset_root / "disc.npy")
    if int(args.subsample_size) > len(target_all):
        raise ValueError("subsample-size exceeds the validation bank")
    rng = np.random.default_rng(
        20260805 + 1000 * int(args.sequence_length) + int(args.seed)
    )
    target_indices = rng.choice(
        len(target_all), int(args.subsample_size), replace=False
    )
    generated_indices = rng.choice(8192, int(args.subsample_size), replace=False)
    target = target_all[target_indices]
    rows: dict[str, dict[str, object]] = {}
    for arm in (REFERENCE_ARM, CANDIDATE_ARM):
        arm_dir = run_dir / arm
        evaluation_dir = (
            arm_dir
            / "checkpoint_evaluations"
            / f"step_{int(args.candidate_evaluation_step):04d}"
            if arm == CANDIDATE_ARM
            and args.candidate_evaluation_step is not None
            else arm_dir
        )
        bank_path = evaluation_dir / f"generated_paths_8192x{int(args.sequence_length)}.npy"
        evaluation_path = evaluation_dir / "validation_metrics.json"
        manifest_path = arm_dir / "training_manifest.json"
        for required in (bank_path, evaluation_path, manifest_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        manifest = read_json(manifest_path)
        if bool(manifest.get("test_split_used", True)):
            raise RuntimeError(f"{manifest_path} does not certify validation isolation")
        bank = load_prices(bank_path)
        if len(bank) != 8192:
            raise ValueError(f"{bank_path} must contain exactly 8192 paths")
        evaluation = read_json(evaluation_path)
        metrics = _path_metrics(
            target,
            bank[generated_indices],
            device=torch.device(str(args.device)),
            swd_projections=int(args.swd_projections),
            swd_seed=20260805 + int(args.seed),
        )
        metrics.update(
            {
                "future_rv_hit_gap_error": float(
                    evaluation["errors"]["future_rv_hit_gap_error"]  # type: ignore[index]
                ),
                "early_hit_future_rv_correlation_error": abs(
                    float(
                        evaluation["generated_memory"][  # type: ignore[index]
                            "early_hit_future_rv_correlation"
                        ]
                    )
                    - float(
                        evaluation["target_memory"][  # type: ignore[index]
                            "early_hit_future_rv_correlation"
                        ]
                    )
                ),
            }
        )
        rows[arm] = {
            "metrics": metrics,
            "scale_tail": scale_tail(target_all, bank),
            "training_resources": manifest["training_resources"],
            "manifest": {
                "solver": manifest["solver"],
                "admissible_control": manifest["admissible_control"],
                "training_stage": manifest.get("training_stage"),
                "test_split_used": manifest["test_split_used"],
            },
        }
        if arm == CANDIDATE_ARM:
            rows[arm]["history_safety"] = history_safety(
                arm_dir / "training_history.jsonl",
                maximum_step=args.candidate_evaluation_step,
            )

    reference = rows[REFERENCE_ARM]["metrics"]
    candidate = rows[CANDIDATE_ARM]["metrics"]
    groups = {
        "distribution": _group_closure(
            reference, candidate, PRIMARY_DISTRIBUTION_METRICS  # type: ignore[arg-type]
        ),
        "dependence": _group_closure(
            reference, candidate, DEPENDENCE_METRICS  # type: ignore[arg-type]
        ),
        "local_calibration": _group_closure(
            reference, candidate, LOCAL_CALIBRATION_METRICS  # type: ignore[arg-type]
        ),
    }
    scale = rows[CANDIDATE_ARM]["scale_tail"]
    history = rows[CANDIDATE_ARM]["history_safety"]
    checks = {
        "finite_positive_paths": (
            float(scale["nonfinite_path_fraction"]) == 0.0  # type: ignore[index]
            and float(scale["nonpositive_price_fraction"]) == 0.0  # type: ignore[index]
            and float(history["sigma_nonfinite_fraction_max"]) == 0.0  # type: ignore[index]
        ),
        # The delayed-volatility target itself has a high-volatility regime and
        # the established valid nested baseline uses the 0.6 cap on roughly
        # ten percent of terminal controls.  Here the failure mode is regime
        # collapse (near-universal saturation), not any cap contact at all.
        "terminal_sigma_cap_at_most_20pct": float(
            history["sigma_cap_fraction_final"]  # type: ignore[index]
        )
        <= 0.20,
        "maximum_logged_sigma_cap_at_most_25pct": float(
            history["sigma_cap_fraction_max"]  # type: ignore[index]
        )
        <= 0.25,
        "return_std_ratio_0p8_1p2": 0.8
        <= float(scale["return_std_ratio"])  # type: ignore[index]
        <= 1.2,
        "rv_mean_ratio_0p8_1p2": 0.8
        <= float(scale["realized_volatility_mean_ratio"])  # type: ignore[index]
        <= 1.2,
        "rv_std_ratio_0p5_1p5": 0.5
        <= float(scale["realized_volatility_std_ratio"])  # type: ignore[index]
        <= 1.5,
        "distribution_median_improves": float(groups["distribution"]["median_gap_closed"])  # type: ignore[arg-type,index]
        > 0.0,
        "dependence_median_improves": float(groups["dependence"]["median_gap_closed"])  # type: ignore[arg-type,index]
        > 0.0,
        "local_calibration_degradation_at_most_10pct": float(
            groups["local_calibration"]["median_gap_closed"]  # type: ignore[arg-type,index]
        )
        >= -0.10,
    }
    payload = {
        "protocol": {
            "decision_split": "disc.npy",
            "test_split_used": False,
            "sequence_length": int(args.sequence_length),
            "seed": int(args.seed),
            "candidate_evaluation_step": args.candidate_evaluation_step,
            "subsample_size": int(args.subsample_size),
        },
        "arms": rows,
        "candidate_gap_closed": {
            name: (
                (float(reference[name]) - float(candidate[name]))
                / float(reference[name])
                if float(reference[name]) > 1e-12
                else None
            )
            for name in reference
        },
        "groups": groups,
        "gate": {"checks": checks, "advance": all(checks.values())},
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "SELECTED_METHOD_VALIDATION.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"groups": groups, "gate": payload["gate"]}, indent=2))


if __name__ == "__main__":
    main()
