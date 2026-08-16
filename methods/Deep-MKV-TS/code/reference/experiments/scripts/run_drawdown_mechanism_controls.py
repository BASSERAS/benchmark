#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    REPO_ROOT
    / "runs"
    / "drawdown_memory_seed0_comparison_20260730"
    / "data"
)
ARMS = ("generic_ce_nested", "direct_bptt")
AVAILABLE_ARMS = (*ARMS, "generic_ce_online")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen delayed-memory mechanism-control seed and score "
            "both selected arms on the untouched test split."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=AVAILABLE_ARMS, default=ARMS)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    source_design_path = args.source_run_dir / "design.json"
    source_checkpoint = args.source_run_dir / "source_model_checkpoint.pt"
    if not source_design_path.exists() or not source_checkpoint.exists():
        raise FileNotFoundError(
            f"frozen source artifacts are missing under {args.source_run_dir}"
        )
    source_design = json.loads(source_design_path.read_text(encoding="utf-8"))
    expected_quantiles = [0.2, 0.3, 0.4]
    if source_design.get("drawdown_threshold_quantiles") != expected_quantiles:
        raise ValueError("source run does not use the frozen threshold quantiles")

    run_dir = args.run_dir.resolve()
    source_run_dir = args.source_run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed_identifier": int(args.seed),
        "device": str(args.device),
        "arms": list(args.arms),
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "shared_source_between_arms": True,
        "independent_source_across_seed_identifiers": True,
        "frozen_configuration": {
            "source_training_steps": 400,
            "joint_training_steps": 1600,
            "batch_size": 256,
            "target_batch_size": 256,
            "drawdown_threshold_quantiles": expected_quantiles,
            "drawdown_memory_half_lives": [20, 60],
            "drawdown_memory_lags": [32],
            "drawdown_gate_temperature": 0.002,
            "reference_crossfit_folds": 3,
            "reference_variance_ridges": [1e-5, 1e-4, 1e-3],
            "nested_outer_steps": 8,
            "nested_inner_steps": 200,
            "nested_trust_fraction": 1.0,
            "gradient_clip_norm": None,
        },
        "selection_split": "disc.npy",
        "final_split": "test.npy",
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "run_drawdown_memory_deep_mkv_ablation.py"),
            "--dataset-root",
            str(DATA_ROOT),
            "--run-dir",
            str(run_dir),
            "--source-run-dir",
            str(source_run_dir),
            "--device",
            str(args.device),
            "--seed",
            str(int(args.seed)),
            "--arms",
            *args.arms,
            "--source-training-steps",
            "400",
            "--joint-training-steps",
            "1600",
            "--batch-size",
            "256",
            "--target-batch-size",
            "256",
            "--bank-size",
            "8192",
            "--sample-batch-size",
            "8192",
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
            "8",
            "--nested-inner-steps",
            "200",
            "--nested-outer-batch-size",
            "1024",
            "--nested-outer-target-batch-size",
            "1024",
            "--nested-inner-batch-size",
            "256",
            "--nested-trust-fraction",
            "1.0",
            "--log-every",
            "25",
        ]
    )

    evaluator = REPO_ROOT / "experiments" / "scripts" / "evaluate_drawdown_memory.py"
    for arm in args.arms:
        arm_dir = run_dir / arm
        run(
            [
                sys.executable,
                str(evaluator),
                "--train-data",
                str(DATA_ROOT / "train.npy"),
                "--test-data",
                str(DATA_ROOT / "test.npy"),
                "--generated-data",
                str(arm_dir / "generated_paths_8192x128.npy"),
                "--dataset-manifest",
                str(DATA_ROOT / "manifest.json"),
                "--output",
                str(arm_dir / "test_metrics.json"),
            ]
        )


if __name__ == "__main__":
    main()
