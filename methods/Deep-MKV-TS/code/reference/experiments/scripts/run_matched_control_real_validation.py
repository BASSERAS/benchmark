#!/usr/bin/env python3
"""Run a canonical real-data matched screen or final volatility-only arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CAUSAL_WEIGHT = {"ES": 0.55, "NQ": 0.55, "RTY": 0.65, "YM": 0.75}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=tuple(CAUSAL_WEIGHT), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--final-zero-drift-only",
        action="store_true",
        help=(
            "Run only nested volatility-only MP with an identically zero "
            "controlled drift."
        ),
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate an already completed final zero-drift joint checkpoint.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _common(
    *,
    index: str,
    phase: str,
    run_dir: Path,
    device: str,
    mode: str,
    seed: int,
    use_reference_drift: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_real_data_tuning_candidate.py"),
        "--phase", phase,
        "--run-dir", str(run_dir),
        "--protocol", "primary_2026_holdout",
        "--objective-protocol", "real_scale_corrected_v3_family_adjoint_rms",
        "--index", index,
        "--seed", str(int(seed)),
        "--device", device,
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
        "--nested-conditional-antithetic",
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
        "--reference-causal-weight", str(CAUSAL_WEIGHT[index]),
        "--lambda-scale", "1",
        "--eta", "0.5",
        "--running-cost", "gaussian_relative_entropy",
        "--drift-control-mode", mode,
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
    if bool(use_reference_drift):
        command.append("--use-reference-drift")
    return command


def _train_arm(
    *,
    index: str,
    arm_root: Path,
    device: str,
    mode: str,
    seed: int,
    use_reference_drift: bool,
    frozen_from: Path | None,
) -> None:
    source = arm_root / "source"
    joint = arm_root / "joint"
    source_command = _common(
        index=index,
        phase="source",
        run_dir=source,
        device=device,
        mode=mode,
        seed=seed,
        use_reference_drift=use_reference_drift,
    )
    source_command.extend(
        [
            "--nested-max-backtracks", "6",
            "--nested-block-trust-fraction", "1",
            "--conditional-claim-mode", "none",
        ]
    )
    if frozen_from is not None:
        source_command.extend(
            [
                "--frozen-discrepancy-normalization-manifest",
                str(frozen_from / "source" / "candidate_manifest.json"),
            ]
        )
    _run(source_command)
    source_checkpoint = source / "outer_002_checkpoint.pt"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)

    joint_command = _common(
        index=index,
        phase="joint",
        run_dir=joint,
        device=device,
        mode=mode,
        seed=seed,
        use_reference_drift=use_reference_drift,
    )
    joint_command.extend(
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
    if frozen_from is not None:
        joint_command.extend(
            [
                "--frozen-discrepancy-normalization-manifest",
                str(frozen_from / "joint" / "candidate_manifest.json"),
            ]
        )
    _run(joint_command)
    if not (joint / "outer_003_checkpoint.pt").is_file():
        raise FileNotFoundError(joint / "outer_003_checkpoint.pt")


def _evaluate(
    *,
    candidate: Path,
    output: Path,
    checkpoint: Path,
    device: str,
    bank_seed: int,
    reference: bool = False,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "evaluate_real_data_candidate_bank.py"),
        "--candidate-run-dir", str(candidate),
        "--checkpoint", str(checkpoint),
        "--run-dir", str(output),
        "--device", device,
        "--bank-size", "8192",
        "--sample-batch-size", "4096",
        "--bank-seed", str(int(bank_seed)),
    ]
    if reference:
        command.append("--reference-only")
    _run(command)


def main() -> None:
    args = parse_args()
    locked = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if bool(locked.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol does not permit this run")
    if not bool(args.final_zero_drift_only) and int(args.seed) != 0:
        raise ValueError("the matched two-arm screen permits seed zero only")
    if bool(args.evaluation_only) and not bool(args.final_zero_drift_only):
        raise ValueError("evaluation-only is restricted to the final zero-drift arm")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    volatility_only = root / "volatility_only_nested_mp"
    bank_seed = 70000 + int(args.seed)
    if bool(args.final_zero_drift_only):
        if not bool(args.evaluation_only):
            _train_arm(
                index=args.index,
                arm_root=volatility_only,
                device=args.device,
                mode="volatility_only",
                seed=int(args.seed),
                use_reference_drift=False,
                frozen_from=None,
            )
        vol_joint = volatility_only / "joint"
        if not (vol_joint / "outer_003_checkpoint.pt").is_file():
            raise FileNotFoundError(vol_joint / "outer_003_checkpoint.pt")
        if not (vol_joint / "candidate_manifest.json").is_file():
            raise FileNotFoundError(vol_joint / "candidate_manifest.json")
        _evaluate(
            candidate=vol_joint,
            output=root / "reference",
            checkpoint=vol_joint / "outer_003_checkpoint.pt",
            device=args.device,
            bank_seed=bank_seed,
            reference=True,
        )
        _evaluate(
            candidate=vol_joint,
            output=volatility_only / "validation_8192",
            checkpoint=vol_joint / "outer_003_checkpoint.pt",
            device=args.device,
            bank_seed=bank_seed,
        )
        arms = ["volatility_only_nested_mp"]
    else:
        full = root / "full_control_nested_mp"
        _train_arm(
            index=args.index,
            arm_root=full,
            device=args.device,
            mode="full",
            seed=0,
            use_reference_drift=True,
            frozen_from=None,
        )
        _train_arm(
            index=args.index,
            arm_root=volatility_only,
            device=args.device,
            mode="volatility_only",
            seed=0,
            use_reference_drift=True,
            frozen_from=full,
        )
        full_joint = full / "joint"
        vol_joint = volatility_only / "joint"
        _evaluate(
            candidate=full_joint,
            output=root / "reference",
            checkpoint=full_joint / "outer_003_checkpoint.pt",
            device=args.device,
            bank_seed=bank_seed,
            reference=True,
        )
        _evaluate(
            candidate=full_joint,
            output=full / "validation_8192",
            checkpoint=full_joint / "outer_003_checkpoint.pt",
            device=args.device,
            bank_seed=bank_seed,
        )
        _evaluate(
            candidate=vol_joint,
            output=volatility_only / "validation_8192",
            checkpoint=vol_joint / "outer_003_checkpoint.pt",
            device=args.device,
            bank_seed=bank_seed,
        )
        arms = ["full_control_nested_mp", "volatility_only_nested_mp"]
    (root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "index": args.index,
                "seed": int(args.seed),
                "selection_scope": "validation_only",
                "test_split_loaded": False,
                "arms": arms,
                "physical_drift": (
                    "zero_controlled_drift"
                    if bool(args.final_zero_drift_only)
                    else "fitted_reference_plus_optional_correction"
                ),
                "reference_causal_weight": CAUSAL_WEIGHT[args.index],
                "conditional_target_estimator": "nested_branches_64_antithetic",
                "evaluation_only_recovery": bool(args.evaluation_only),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
