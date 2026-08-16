#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RUNNER = (
    REPO_ROOT / "experiments" / "scripts" / "run_real_data_tuning_candidate.py"
)


JOINT_VARIANTS: tuple[dict[str, float | str], ...] = (
    {
        "identifier": "center",
        "lr": 2.5e-4,
        "joint_weight": 0.25,
        "eta": 2.0,
    },
    {
        "identifier": "lr1e-4",
        "lr": 1e-4,
        "joint_weight": 0.25,
        "eta": 2.0,
    },
    {
        "identifier": "lr5e-4",
        "lr": 5e-4,
        "joint_weight": 0.25,
        "eta": 2.0,
    },
    {
        "identifier": "joint_weight0.1",
        "lr": 2.5e-4,
        "joint_weight": 0.10,
        "eta": 2.0,
    },
    {
        "identifier": "joint_weight0.5",
        "lr": 2.5e-4,
        "joint_weight": 0.50,
        "eta": 2.0,
    },
    {
        "identifier": "eta1",
        "lr": 2.5e-4,
        "joint_weight": 0.25,
        "eta": 1.0,
    },
    {
        "identifier": "eta4",
        "lr": 2.5e-4,
        "joint_weight": 0.25,
        "eta": 4.0,
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one shard of the validation-only joint objective-balance sweep."
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_tuning" / "source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_tuning" / "joint",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("shard-index must lie in [0, num-shards)")
    source_summary_path = args.source_root / "summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError(
            f"source summary is not ready: {source_summary_path}"
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    selected_identifier = source_summary.get("selected_identifier")
    if not selected_identifier:
        raise RuntimeError("the source sweep did not produce an eligible candidate")
    selected_rows = [
        row
        for row in source_summary["candidates"]
        if row["identifier"] == selected_identifier
    ]
    if len(selected_rows) != 1:
        raise RuntimeError("selected source candidate is ambiguous")
    selected = selected_rows[0]
    source_training = selected["configuration"]["training"]
    source_lambda = float(source_training["lambda_scale"])
    source_ridge = float(source_training["ridge_lambda"])
    jobs = [
        (variant, seed)
        for variant in JOINT_VARIANTS
        for seed in (1, 2)
    ]
    selected_jobs = [
        job
        for index, job in enumerate(jobs)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    sweep_manifest = {
        "selection_scope": "training_and_validation_only",
        "test_or_event_data_seen": False,
        "phase": "joint",
        "seeds": [1, 2],
        "steps": int(args.steps),
        "bank_size": int(args.bank_size),
        "grad_clip_norm": 50.0,
        "selected_source_identifier": selected_identifier,
        "source_lambda_scale": source_lambda,
        "source_ridge_lambda": source_ridge,
        "variants": list(JOINT_VARIANTS),
    }
    (args.output_root / "sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for variant, seed in selected_jobs:
        identifier = str(variant["identifier"])
        run_dir = args.output_root / identifier / f"seed_{seed}"
        manifest = run_dir / "candidate_manifest.json"
        if manifest.is_file():
            print(f"skip completed {identifier}/seed_{seed}", flush=True)
            continue
        source_checkpoint = Path(selected["checkpoints"][str(seed)])
        command = [
            sys.executable,
            str(CANDIDATE_RUNNER),
            "--phase",
            "joint",
            "--run-dir",
            str(run_dir),
            "--source-checkpoint",
            str(source_checkpoint),
            "--seed",
            str(seed),
            "--device",
            str(args.device),
            "--steps",
            str(args.steps),
            "--lr",
            str(variant["lr"]),
            "--ridge-lambda",
            str(source_ridge),
            "--lambda-scale",
            str(source_lambda),
            "--eta",
            str(variant["eta"]),
            "--joint-weight",
            str(variant["joint_weight"]),
            "--grad-clip-norm",
            "50",
            "--bank-size",
            str(args.bank_size),
            "--sample-batch-size",
            str(args.bank_size),
            "--log-every",
            "50",
        ]
        print(f"start {identifier}/seed_{seed} on {args.device}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"complete shard {args.shard_index}/{args.num_shards}", flush=True)


if __name__ == "__main__":
    main()
