#!/usr/bin/env python3
"""Aggregate the locked seed-0 cross-task volatility-only validation screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.real_data_protocol import (  # noqa: E402
    normalized_prices_to_log_paths,
    real_data_law_metrics,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingFeatureConfig,
    evaluate_path_shadowing,
)
from summarize_volatility_only_nested_scaling import (  # noqa: E402
    DEPENDENCE_METRICS,
    LOCAL_CALIBRATION_METRICS,
    PRIMARY_DISTRIBUTION_METRICS,
    _group_closure,
    _path_metrics,
)


TASKS = ("heston", "mixture", "ES", "NQ", "RTY", "YM")
REFERENCE_ARM = "generic_reference_only"
MP_ARM = "generic_ce_nested_volatility_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--swd-projections", type=int, default=256)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _arm_metrics(
    *,
    task: str,
    arm_dir: Path,
    target: np.ndarray,
    fake_indices: np.ndarray,
    swd_projections: int,
) -> tuple[dict[str, float], dict[str, object]]:
    bank_path = arm_dir / "generated_paths_8192x128.npy"
    manifest_path = arm_dir / "training_manifest.json"
    evaluation_path = arm_dir / "validation_metrics.json"
    for path in (bank_path, manifest_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _read_json(manifest_path)
    if bool(manifest.get("test_split_used", True)):
        raise RuntimeError(f"{manifest_path} does not certify validation-only use")
    generated = np.asarray(np.load(bank_path, allow_pickle=False), dtype=np.float64)
    metrics = _path_metrics(
        target,
        generated[fake_indices],
        device=torch.device("cpu"),
        swd_projections=int(swd_projections),
        swd_seed=20260804 + sum(ord(character) for character in task),
    )
    evaluation = _read_json(evaluation_path)
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
    return metrics, manifest


def _closure(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or reference <= 1e-12:
        raise ValueError("reference error must be positive and finite")
    return float((reference - candidate) / reference)


def _real_metrics(
    target_prices: np.ndarray,
    bank_path: Path,
) -> dict[str, object]:
    generated_prices = np.asarray(
        np.load(bank_path, allow_pickle=False), dtype=np.float64
    )
    target_log = normalized_prices_to_log_paths(target_prices)
    generated_log = normalized_prices_to_log_paths(generated_prices)
    law = real_data_law_metrics(generated_log, target_log)
    shadow = evaluate_path_shadowing(
        real_paths=target_log,
        generated_paths=generated_log,
        prefix_length=65,
        future_horizon=32,
        top_k=256,
        feature_config=ShadowingFeatureConfig(),
        search_backend="auto",
    )
    return {"law": law, "shadowing": shadow.metrics}


def main() -> None:
    args = parse_args()
    screen_root = args.screen_root.resolve()
    rows: dict[str, object] = {}
    for task in tuple(str(value) for value in args.tasks):
        dataset_root = screen_root / "datasets" / task
        validation = np.asarray(
            np.load(dataset_root / "disc.npy", allow_pickle=False),
            dtype=np.float64,
        )
        count = min(int(args.subsample_size), len(validation))
        generator = np.random.default_rng(
            20260804 + sum(ord(character) for character in task)
        )
        target_indices = generator.choice(len(validation), count, replace=False)
        fake_indices = generator.choice(8192, count, replace=False)
        target = validation[target_indices]
        task_dir = screen_root / "tasks" / task
        reference, reference_manifest = _arm_metrics(
            task=task,
            arm_dir=task_dir / REFERENCE_ARM,
            target=target,
            fake_indices=fake_indices,
            swd_projections=int(args.swd_projections),
        )
        candidate, candidate_manifest = _arm_metrics(
            task=task,
            arm_dir=task_dir / MP_ARM,
            target=target,
            fake_indices=fake_indices,
            swd_projections=int(args.swd_projections),
        )
        gap_closed = {
            name: _closure(reference[name], candidate[name])
            for name in reference
        }
        groups = {
            "distribution": _group_closure(
                reference, candidate, PRIMARY_DISTRIBUTION_METRICS
            ),
            "dependence": _group_closure(
                reference, candidate, DEPENDENCE_METRICS
            ),
            "local_calibration": _group_closure(
                reference, candidate, LOCAL_CALIBRATION_METRICS
            ),
        }
        gate = {
            "distribution_improves": bool(
                float(groups["distribution"]["median_gap_closed"]) > 0.0
            ),
            "dependence_not_materially_worse": bool(
                float(groups["dependence"]["median_gap_closed"]) >= -0.10
            ),
            "local_calibration_degradation_at_most_ten_percent": bool(
                float(groups["local_calibration"]["median_gap_closed"]) >= -0.10
            ),
        }
        gate["pass"] = all(gate.values())
        row: dict[str, object] = {
            "validation_paths": int(len(validation)),
            "metric_subsample_paths": count,
            "reference_metrics": reference,
            "candidate_metrics": candidate,
            "gap_closed": gap_closed,
            "group_gap_closed": groups,
            "gate": gate,
            "training_resources": candidate_manifest["training_resources"],
            "admissible_control": candidate_manifest["admissible_control"],
            "initialization": candidate_manifest["initialization"],
            "reference_initialization": reference_manifest["initialization"],
        }
        if task in {"ES", "NQ", "RTY", "YM"}:
            row["real_data"] = {
                "reference": _real_metrics(
                    validation,
                    task_dir / REFERENCE_ARM / "generated_paths_8192x128.npy",
                ),
                "candidate": _real_metrics(
                    validation,
                    task_dir / MP_ARM / "generated_paths_8192x128.npy",
                ),
            }
        rows[task] = row
    passed = [task for task, row in rows.items() if bool(row["gate"]["pass"])]  # type: ignore[index]
    payload = {
        "protocol": {
            "model_seed": 0,
            "solver": "finite_objective_gated_nested_mp",
            "admissible_control": "volatility_only_alpha_fixed_zero",
            "objective": "old_fullv_w0p25_plus_joint_prefix_future_rv",
            "tasks": list(args.tasks),
            "decision_split": "validation_only",
            "test_split_used": False,
            "screen_gate": {
                "distribution_group_median_gap_closed": "> 0",
                "dependence_group_median_gap_closed": ">= -0.10",
                "local_group_median_gap_closed": ">= -0.10",
            },
        },
        "tasks": rows,
        "decision": {
            "passed_tasks": passed,
            "failed_tasks": [task for task in args.tasks if task not in passed],
            "pass_count": len(passed),
            "task_count": len(tuple(args.tasks)),
            "expand_to_required_seeds": len(passed) == len(tuple(args.tasks)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
