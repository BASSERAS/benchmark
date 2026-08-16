#!/usr/bin/env python3
"""Run one resumable GPU lane of the remaining ICAIF experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
MIXTURE_ROOT = REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
MIXTURE_OUTPUT = REPO_ROOT / "runs" / "paper_mixture_independent_4seed_20260801"
REAL_OUTPUT = REPO_ROOT / "runs" / "paper_real_4seed_4index_20260801"
REAL_GENERATED = (
    REPO_ROOT / "runs" / "real_data_protocol" / "generated"
    / "deep_mkv_stacked_corr_o3"
)
QUEUE_STATUS_ROOT = REPO_ROOT / "runs" / "paper_remaining_queues_20260801"

LANES: dict[int, tuple[tuple[object, ...], ...]] = {
    0: (
        ("mixture", 1),
        ("real", "ES", 0),
        ("real", "NQ", 0),
        ("real", "RTY", 0),
        ("real", "YM", 0),
    ),
    1: (
        ("mixture", 3),
        ("real", "ES", 3),
        ("real", "NQ", 3),
        ("real", "RTY", 3),
        ("real", "YM", 3),
    ),
    2: (
        ("mixture", 4),
        ("real", "ES", 4),
        ("real", "NQ", 4),
        ("real", "RTY", 4),
        ("real", "YM", 4),
    ),
    3: (
        ("real", "NQ", 1),
        ("real", "RTY", 1),
        ("real", "YM", 1),
    ),
    # Acceleration lane for a freed GPU.  The primary lanes will detect and
    # reuse these artifacts when they eventually reach the same late tasks.
    4: (
        ("real", "YM", 0),
        ("real", "YM", 3),
        ("real", "YM", 4),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--device", help="Defaults to cuda:<lane>.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"[{timestamp()}] RUN {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def mixture_source_command(*, seed: int, device: str, run_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "experiments/scripts/run_basseras_heston_path_shadowing.py"),
        "--benchmark-root", "/home/samer/scenarios/BASSERAS-benchmark",
        "--custom-dataset-root", str(MIXTURE_ROOT / "data"),
        "--run-dir", str(run_dir),
        "--stage", "train",
        "--device", device,
        "--dtype", "float32",
        "--seed", str(seed),
        "--data-layout", "benchmark_8192",
        "--state-representation", "log_price",
        "--source-training-steps", "2000",
        "--joint-training-steps", "1",
        "--batch-size", "256",
        "--target-batch-size", "256",
        "--hidden-dim", "96",
        "--num-layers", "1",
        "--lr", "0.002",
        "--weight-decay", "0.00001",
        "--lambda-scale", "50",
        "--kappa-scale", "50",
        "--ridge-lambda", "0.001",
        "--reference-ridge", "0.001",
        "--reference-variance-shrinkage", "0.1",
        "--reference-log-variance-bias-correction", "0.6",
        "--reference-kind", "local",
        "--eta", "1",
        "--sigma-min", "0.001",
        "--sigma-max", "0.6",
        "--grad-clip-norm", "5",
        "--log-every", "100",
    ]


def run_mixture(*, seed: int, device: str, dry_run: bool) -> None:
    seed_root = MIXTURE_OUTPUT / f"seed_{seed}"
    source_root = seed_root / "source"
    source_checkpoint = source_root / "source_model_checkpoint.pt"
    if not source_checkpoint.exists():
        source_root.mkdir(parents=True, exist_ok=True)
        run(
            mixture_source_command(seed=seed, device=device, run_dir=source_root),
            dry_run=dry_run,
        )
    else:
        print(f"[{timestamp()}] REUSE {source_checkpoint}", flush=True)

    arm = "summary_online_uncapped_causal_ce"
    arm_root = seed_root / arm
    validation = arm_root / "validation_metrics.json"
    if not validation.exists():
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/run_heston_mixture_deep_mkv_ablation.py"),
                "--experiment-root", str(MIXTURE_ROOT),
                "--source-run-dir", str(source_root),
                "--run-dir", str(seed_root),
                "--device", device,
                "--seed", str(seed),
                "--arms", arm,
                "--training-steps", "1600",
                "--batch-size", "256",
                "--target-batch-size", "256",
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--hidden-dim", "96",
                "--num-layers", "1",
                "--lr", "0.002",
                "--weight-decay", "0.00001",
                "--ridge-lambda", "0.001",
                "--summary-weight", "1",
                "--sigma-min", "0.001",
                "--sigma-max", "0.6",
                "--eta", "1",
                "--log-every", "100",
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{timestamp()}] REUSE {validation}", flush=True)

    generated = arm_root / "generated_paths_8192x128.npy"
    test_metrics = seed_root / "test_metrics.json"
    if not test_metrics.exists():
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_heston_parameter_mixture.py"),
                "--train-data", str(MIXTURE_ROOT / "data/train.npy"),
                "--test-data", str(MIXTURE_ROOT / "data/test.npy"),
                "--generated-data", str(generated),
                "--oracle", str(MIXTURE_ROOT / "oracle/oracle.joblib"),
                "--oracle-gate-report", str(MIXTURE_ROOT / "oracle/gate_report.json"),
                "--output", str(test_metrics),
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{timestamp()}] REUSE {test_metrics}", flush=True)


def common_real_command(
    *, phase: str, index: str, seed: int, device: str, run_dir: Path
) -> list[str]:
    command = [
        str(PYTHON),
        str(REPO_ROOT / "experiments/scripts/run_real_data_tuning_candidate.py"),
        "--phase", phase,
        "--run-dir", str(run_dir),
        "--protocol", "primary_2026_holdout",
        "--objective-protocol", "real_scale_corrected_v3_family_adjoint_rms",
        "--index", index,
        "--seed", str(seed),
        "--device", device,
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
    return command


def ensure_relative_symlink(*, target: Path, link: Path, dry_run: bool) -> None:
    print(f"[{timestamp()}] LINK {link} -> {target}", flush=True)
    if dry_run:
        return
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing bank link: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, start=link.parent))


def run_real(*, index: str, seed: int, device: str, dry_run: bool) -> None:
    root = REAL_OUTPUT / index / f"seed_{seed}"
    source_root = root / "source"
    joint_root = root / "joint"
    matched_root = root / "matched8192"

    source_checkpoint = source_root / "outer_002_checkpoint.pt"
    if not source_checkpoint.exists():
        command = common_real_command(
            phase="source", index=index, seed=seed, device=device, run_dir=source_root
        )
        command.extend(
            [
                "--nested-max-backtracks", "6",
                "--nested-block-trust-fraction", "1",
                "--conditional-claim-mode", "none",
            ]
        )
        run(command, dry_run=dry_run)
    else:
        print(f"[{timestamp()}] REUSE {source_checkpoint}", flush=True)

    joint_checkpoint = joint_root / "outer_003_checkpoint.pt"
    if not joint_checkpoint.exists():
        command = common_real_command(
            phase="joint", index=index, seed=seed, device=device, run_dir=joint_root
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
        run(command, dry_run=dry_run)
    else:
        print(f"[{timestamp()}] REUSE {joint_checkpoint}", flush=True)

    bank = matched_root / "validation_bank.npy"
    metrics = matched_root / "metrics.json"
    if not bank.exists() or not metrics.exists():
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_real_data_candidate_bank.py"),
                "--candidate-run-dir", str(joint_root),
                "--checkpoint", str(joint_checkpoint),
                "--run-dir", str(matched_root),
                "--device", device,
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(70_000 + seed),
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{timestamp()}] REUSE {bank}", flush=True)

    link = (
        REAL_GENERATED / "primary_2026_holdout" / index / f"seed_{seed}.npy"
    )
    ensure_relative_symlink(target=bank, link=link, dry_run=dry_run)

    protocol_root = (
        REPO_ROOT / "runs" / "real_data_protocol"
        / "Deep-MKV-Gen-stacked-corr-o3" / index / f"seed_{seed}"
    )
    expected = [
        protocol_root / name / "metrics.json"
        for name in ("primary_2026_holdout", "us_iran_2026_war")
    ]
    if not all(path.exists() for path in expected):
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/run_real_data_evaluation.py"),
                "--generated-root", str(REAL_GENERATED),
                "--model-name", "Deep-MKV-Gen-stacked-corr-o3",
                "--index", index,
                "--seed", str(seed),
                "--protocol", "primary_2026_holdout",
                "--protocol", "us_iran_2026_war",
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{timestamp()}] REUSE {protocol_root}", flush=True)


def main() -> None:
    args = parse_args()
    device = args.device or f"cuda:{args.lane}"
    print(
        f"[{timestamp()}] START lane={args.lane} device={device} "
        f"tasks={LANES[args.lane]}",
        flush=True,
    )
    status_path = QUEUE_STATUS_ROOT / f"lane_{args.lane}.json"
    completed: list[list[object]] = []
    if not args.dry_run:
        write_status(
            status_path,
            {
                "status": "running",
                "lane": args.lane,
                "device": device,
                "started_utc": timestamp(),
                "tasks": [list(task) for task in LANES[args.lane]],
                "completed": completed,
            },
        )
    try:
        for task in LANES[args.lane]:
            if task[0] == "mixture":
                run_mixture(seed=int(task[1]), device=device, dry_run=args.dry_run)
            elif task[0] == "real":
                run_real(
                    index=str(task[1]),
                    seed=int(task[2]),
                    device=device,
                    dry_run=args.dry_run,
                )
            else:  # pragma: no cover - static task table
                raise ValueError(task)
            completed.append(list(task))
            if not args.dry_run:
                write_status(
                    status_path,
                    {
                        "status": "running",
                        "lane": args.lane,
                        "device": device,
                        "tasks": [list(item) for item in LANES[args.lane]],
                        "completed": completed,
                        "updated_utc": timestamp(),
                    },
                )
    except BaseException as error:
        if not args.dry_run:
            write_status(
                status_path,
                {
                    "status": "failed",
                    "lane": args.lane,
                    "device": device,
                    "tasks": [list(item) for item in LANES[args.lane]],
                    "completed": completed,
                    "failed_utc": timestamp(),
                    "error": repr(error),
                },
            )
        raise
    if not args.dry_run:
        write_status(
            status_path,
            {
                "status": "complete",
                "lane": args.lane,
                "device": device,
                "tasks": [list(item) for item in LANES[args.lane]],
                "completed": completed,
                "completed_utc": timestamp(),
            },
        )
    print(f"[{timestamp()}] COMPLETE lane={args.lane}", flush=True)


if __name__ == "__main__":
    main()
