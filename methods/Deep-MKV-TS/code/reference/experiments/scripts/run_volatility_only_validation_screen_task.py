#!/usr/bin/env python3
"""Run one task from the locked seed-0 volatility-only validation screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("heston", "mixture", "ES", "NQ", "RTY", "YM")


@dataclass(frozen=True)
class LockedProtocol:
    model_seed: int = 0
    joint_training_steps: int = 1600
    nested_outer_steps: int = 8
    nested_inner_steps: int = 200
    batch_size: int = 256
    target_batch_size: int = 256
    nested_outer_batch_size: int = 1024
    nested_outer_target_batch_size: int = 1024
    nested_inner_batch_size: int = 256
    nested_trust_fraction: float = 1.0
    unused_proximal_validation_paths: int = 128
    bank_size: int = 8192
    sample_batch_size: int = 2048
    eta: float = 1.0
    sigma_min: float = 1e-3
    sigma_max: float = 0.6
    gradient_clipping: bool = False
    objective: str = "old_fullv_w0p25_plus_joint_prefix_future_rv"
    conditional_expectation_basis: str = "generic_causal"
    reference_selection: str = "common_train_only_causal_anchor_grid"
    test_split_used: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = LockedProtocol()
    screen_root = args.screen_root.resolve()
    dataset_root = screen_root / "datasets" / str(args.task)
    run_dir = screen_root / "tasks" / str(args.task)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "locked_protocol.json"
    payload = {
        **asdict(protocol),
        "task": str(args.task),
        "device": str(args.device),
        "dataset_root": str(dataset_root),
        "arms": [
            "generic_reference_only",
            "generic_ce_nested_volatility_only",
        ],
        "decision_split": "disc.npy",
        "test_filename_is_validation_alias": True,
    }
    protocol_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "experiments"
            / "scripts"
            / "run_drawdown_memory_deep_mkv_ablation.py"
        ),
        "--dataset-root",
        str(dataset_root),
        "--run-dir",
        str(run_dir),
        "--device",
        str(args.device),
        "--seed",
        str(protocol.model_seed),
        "--arms",
        "generic_reference_only",
        "generic_ce_nested_volatility_only",
        "--source-training-steps",
        "400",
        "--joint-training-steps",
        str(protocol.joint_training_steps),
        "--batch-size",
        str(protocol.batch_size),
        "--target-batch-size",
        str(protocol.target_batch_size),
        "--bank-size",
        str(protocol.bank_size),
        "--sample-batch-size",
        str(protocol.sample_batch_size),
        "--eta",
        str(protocol.eta),
        "--sigma-min",
        str(protocol.sigma_min),
        "--sigma-max",
        str(protocol.sigma_max),
        "--drawdown-threshold-quantiles",
        "0.2",
        "0.3",
        "0.4",
        "--drawdown-memory-half-lives",
        "20",
        "60",
        "--drawdown-memory-lags",
        "32",
        "--drawdown-gate-temperature",
        "0.002",
        "--reference-crossfit-folds",
        "3",
        "--reference-variance-ridges",
        "1e-5",
        "1e-4",
        "1e-3",
        "--nested-outer-steps",
        str(protocol.nested_outer_steps),
        "--nested-inner-steps",
        str(protocol.nested_inner_steps),
        "--nested-outer-batch-size",
        str(protocol.nested_outer_batch_size),
        "--nested-outer-target-batch-size",
        str(protocol.nested_outer_target_batch_size),
        "--nested-inner-batch-size",
        str(protocol.nested_inner_batch_size),
        "--nested-trust-fraction",
        str(protocol.nested_trust_fraction),
        "--proximal-validation-paths",
        str(protocol.unused_proximal_validation_paths),
        "--generic-reference",
        "--calibrate-reference",
        "--log-every",
        "25",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
