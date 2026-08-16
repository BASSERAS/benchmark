#!/usr/bin/env python3
"""Run one seed of the generic-reference solver and resolution comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARMS = ("generic_ce_online", "generic_ce_nested", "direct_bptt")
VOLATILITY_ONLY_COMPARISON_ARMS = (
    "generic_reference_only",
    "generic_ce_online_volatility_only",
    "generic_ce_online_source_volatility_only",
    "generic_ce_nested_reference",
    "generic_ce_nested_volatility_only",
    "direct_bptt_volatility_only",
)
AVAILABLE_ARMS = (*DEFAULT_ARMS, *VOLATILITY_ONLY_COMPARISON_ARMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument(
        "--arms", nargs="+", choices=AVAILABLE_ARMS, default=DEFAULT_ARMS
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Stop after the frozen validation evaluation performed by the arm "
            "runner; do not read or evaluate test.npy."
        ),
    )
    parser.add_argument("--source-training-steps", type=int, default=400)
    parser.add_argument(
        "--online-checkpoint-steps", type=int, nargs="*", default=()
    )
    parser.add_argument("--joint-training-steps", type=int, default=1600)
    parser.add_argument("--nested-outer-batch-size", type=int, default=1024)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--reference-kind",
        choices=("causal", "guyon_lekeufack_structural_likelihood"),
        default="causal",
    )
    parser.add_argument("--adjoint-weight", type=float, default=1.0)
    parser.add_argument("--adjoint-noise-weight", type=float, default=1.0)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    if int(configuration["sequence_length"]) != int(args.sequence_length):
        raise ValueError("dataset sequence length does not match the requested run")
    expected_scale = float(args.sequence_length) / 128.0
    expected = {
        "dt": (1.0 / 250.0) / expected_scale,
        "response_delay": int(round(32 * expected_scale)),
        "memory_half_life": float(60 * expected_scale),
        "split_index": int(round(64 * expected_scale)),
        "history_cutoff": int(round(40 * expected_scale)),
        "recent_window": int(round(20 * expected_scale)),
        "future_horizon": int(round(32 * expected_scale)),
    }
    for key, value in expected.items():
        actual = configuration[key]
        if isinstance(value, float):
            if abs(float(actual) - value) > 1e-12:
                raise ValueError(f"dataset {key}={actual!r}, expected {value!r}")
        elif int(actual) != value:
            raise ValueError(f"dataset {key}={actual!r}, expected {value!r}")

    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": int(args.seed),
        "device": str(args.device),
        "sequence_length": int(args.sequence_length),
        "temporal_scale": expected_scale,
        "dataset_root": str(dataset_root),
        "source_run_dir": (
            str(args.source_run_dir.resolve()) if args.source_run_dir else None
        ),
        "arms": list(args.arms),
        "online_checkpoint_steps": [
            int(value) for value in args.online_checkpoint_steps
        ],
        "shared_reference_initialization_objective_and_update_budget": True,
        "generic_reference_and_projection_basis": True,
        "reference_kind": str(args.reference_kind),
        "training_resource_scope": "joint solver fit only",
        "decision_split": "disc.npy" if args.validation_only else "test.npy",
        "test_split_used": not bool(args.validation_only),
        "adjoint_weight": float(args.adjoint_weight),
        "adjoint_noise_weight": float(args.adjoint_noise_weight),
    }
    (run_dir / "scaling_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_drawdown_memory_deep_mkv_ablation.py"),
        "--dataset-root",
        str(dataset_root),
        "--run-dir",
        str(run_dir),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--arms",
        *args.arms,
        "--source-training-steps",
        str(int(args.source_training_steps)),
        "--joint-training-steps",
        str(int(args.joint_training_steps)),
        "--batch-size",
        "256",
        "--target-batch-size",
        "256",
        "--bank-size",
        "8192",
        "--sample-batch-size",
        "2048",
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
        str(int(args.nested_outer_batch_size)),
        "--lambda-scale",
        str(float(args.lambda_scale)),
        "--kappa-scale",
        str(float(args.kappa_scale)),
        "--eta",
        str(float(args.eta)),
        "--reference-kind",
        str(args.reference_kind),
        "--adjoint-weight",
        str(float(args.adjoint_weight)),
        "--adjoint-noise-weight",
        str(float(args.adjoint_noise_weight)),
        "--nested-outer-target-batch-size",
        str(int(args.nested_outer_batch_size)),
        "--nested-inner-batch-size",
        "256",
        "--nested-trust-fraction",
        "1.0",
        "--generic-reference",
        "--scale-temporal-features",
        "--log-every",
        "25",
    ]
    if args.source_run_dir is not None:
        command.extend(["--source-run-dir", str(args.source_run_dir.resolve())])
    if args.online_checkpoint_steps:
        command.extend(
            [
                "--online-checkpoint-steps",
                *(str(int(value)) for value in args.online_checkpoint_steps),
            ]
        )
    run(command)

    if args.validation_only:
        return

    evaluator = REPO_ROOT / "experiments" / "scripts" / "evaluate_drawdown_memory.py"
    for arm in args.arms:
        arm_dir = run_dir / arm
        run(
            [
                sys.executable,
                str(evaluator),
                "--train-data",
                str(dataset_root / "train.npy"),
                "--test-data",
                str(dataset_root / "test.npy"),
                "--generated-data",
                str(
                    arm_dir
                    / f"generated_paths_8192x{int(args.sequence_length)}.npy"
                ),
                "--dataset-manifest",
                str(manifest_path),
                "--output",
                str(arm_dir / "test_metrics.json"),
            ]
        )


if __name__ == "__main__":
    main()
