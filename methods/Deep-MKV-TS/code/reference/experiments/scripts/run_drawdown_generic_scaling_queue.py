#!/usr/bin/env python3
"""Run the three-resolution generic solver comparison for one model seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    128: REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data",
    256: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n256",
    512: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n512",
}
GENERIC_SOURCE_ROOT = REPO_ROOT / "runs" / "paper_drawdown_fully_generic_4seed_20260801"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=(128, 256, 512))
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / f"seed_{int(args.seed)}_queue_status.json"
    status: dict[str, object] = {
        "seed": int(args.seed),
        "device": str(args.device),
        "started_at": timestamp(),
        "resolutions": {},
    }
    write_status(status_path, status)
    failures = 0
    for sequence_length in tuple(int(value) for value in args.sequence_lengths):
        if sequence_length not in DATASETS:
            raise ValueError(f"unsupported sequence length {sequence_length}")
        row = {
            "status": "running",
            "started_at": timestamp(),
        }
        status["resolutions"][str(sequence_length)] = row  # type: ignore[index]
        write_status(status_path, status)
        command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "run_drawdown_generic_solver_scaling.py"),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--sequence-length",
            str(sequence_length),
            "--dataset-root",
            str(DATASETS[sequence_length]),
            "--run-dir",
            str(run_root / f"n{sequence_length}" / f"seed_{int(args.seed)}"),
        ]
        if sequence_length == 128:
            command.extend(
                [
                    "--source-run-dir",
                    str(GENERIC_SOURCE_ROOT / f"seed_{int(args.seed)}"),
                ]
            )
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures += 1
            row.update(
                {
                    "status": "failed",
                    "completed_at": timestamp(),
                    "returncode": int(error.returncode),
                }
            )
        else:
            row.update(
                {
                    "status": "complete",
                    "completed_at": timestamp(),
                    "returncode": 0,
                }
            )
        write_status(status_path, status)
    status["completed_at"] = timestamp()
    status["failure_count"] = int(failures)
    write_status(status_path, status)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
