#!/usr/bin/env python3
"""Run one dimensionless-state real-data paper replication lane."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
DEFAULT_ROOT = REPO_ROOT / "runs" / "real_dimensionless_4seed_20260801"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        choices=("ES", "NQ", "RTY", "YM"),
        default="RTY",
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 3, 4), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def common_command(
    *, phase: str, index: str, seed: int, device: str, run_dir: Path
) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "experiments/scripts/run_real_data_tuning_candidate.py"),
        "--phase", phase,
        "--run-dir", str(run_dir),
        "--protocol", "primary_2026_holdout",
        "--objective-protocol", "real_scale_corrected_v3_family_adjoint_rms",
        "--index", index,
        "--seed", str(seed),
        "--device", device,
        "--dtype", "float32",
        "--state-unit-scaling", "train_return_volatility",
        "--steps", "1500",
        "--solver", "nested",
        "--nested-outer-steps", "3",
        "--nested-inner-steps", "500",
        "--nested-outer-batch-size", "64",
        "--nested-outer-target-batch-size", "496",
        "--nested-outer-backward-replicates", "1",
        "--nested-outer-target-estimator", "nested_branches",
        "--nested-conditional-branches", "64",
        "--nested-conditional-query-batch-size", "16384",
        "--nested-population-batch-size", "256",
        "--nested-line-search-batch-size", "512",
        "--nested-inner-batch-size", "64",
        "--nested-probe-paths", "512",
        "--nested-relaxation-mode", "adjoint_blocks",
        "--nested-backtrack-factor", "0.5",
        "--nested-objective-tolerance", "0",
        "--nested-block-coordinate-passes", "2",
        "--lr", "0.00025",
        "--ridge-lambda", "10",
        "--ce-target-mode", "ridge",
        "--r-ce-basis", "causal_compact",
        "--p-ce-basis", "causal_compact",
        "--r-ce-normalization", "train_fixed",
        "--r-ce-time-mode", "pooled",
        "--adjoint-network", "causal_residual",
        "--ce-crossfit-folds", "1",
        "--noise-target-control-variate", "timewise",
        "--noise-target-estimator", "score",
        "--stein-probes", "16",
        "--preconditioner-batches", "8",
        "--reference-rv-spread-ratio-min", "0.4",
        "--reference-feature-basis", "compact",
        "--lambda-scale", "1",
        "--eta", "0.5",
        "--running-cost", "gaussian_relative_entropy",
        "--lambda-x", "64",
        "--lambda-v", "16",
        "--joint-weight", "0.25",
        "--return-discrepancy", "mmd",
        "--global-rv-discrepancy", "mmd",
        "--discrepancy-stack", "full",
        "--joint-block-multiplier", "1",
        "--grad-clip-norm", "0",
        "--bank-size", "4096",
        "--sample-batch-size", "4096",
        "--screen-each-outer",
        "--log-every", "50",
    ]


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    seed_root = root / args.index / f"seed_{args.seed}"
    source_root = seed_root / "source"
    joint_root = seed_root / "joint"
    matched_root = seed_root / "matched8192"
    source_checkpoint = source_root / "outer_002_checkpoint.pt"
    joint_checkpoint = joint_root / "outer_003_checkpoint.pt"

    if not source_checkpoint.exists():
        command = common_command(
            phase="source",
            index=args.index,
            seed=args.seed,
            device=args.device,
            run_dir=source_root,
        )
        command.extend(
            [
                "--nested-max-backtracks", "6",
                "--nested-block-trust-fraction", "1",
                "--conditional-claim-mode", "none",
            ]
        )
        run(command)

    if not joint_checkpoint.exists():
        command = common_command(
            phase="joint",
            index=args.index,
            seed=args.seed,
            device=args.device,
            run_dir=joint_root,
        )
        command.extend(
            [
                "--source-checkpoint", str(source_checkpoint),
                "--source-checkpoint-topology", "causal_residual",
                "--source-checkpoint-r-ce-basis", "causal_compact",
                "--source-checkpoint-stack-depth", "1",
                "--nested-max-backtracks", "8",
                "--nested-block-trust-fraction", "0.5",
                "--conditional-claim-mode", "correlation",
            ]
        )
        run(command)

    bank = matched_root / "validation_bank.npy"
    if not bank.exists():
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_real_data_candidate_bank.py"),
                "--candidate-run-dir", str(joint_root),
                "--checkpoint", str(joint_checkpoint),
                "--run-dir", str(matched_root),
                "--device", args.device,
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(70_000 + args.seed),
            ]
        )

    generated_link = (
        root / "generated" / "primary_2026_holdout" / args.index
        / f"seed_{args.seed}.npy"
    )
    generated_link.parent.mkdir(parents=True, exist_ok=True)
    if not generated_link.exists():
        generated_link.symlink_to(
            os.path.relpath(bank, start=generated_link.parent)
        )

    run(
        [
            str(PYTHON),
            str(REPO_ROOT / "experiments/scripts/run_real_data_evaluation.py"),
            "--generated-root", str(root / "generated"),
            "--output-root", str(root / "protocol"),
            "--model-name", "Deep-MKV-dimensionless",
            "--index", args.index,
            "--seed", str(args.seed),
            "--protocol", "primary_2026_holdout",
            "--protocol", "us_iran_2026_war",
        ]
    )


if __name__ == "__main__":
    main()
