#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
SOURCE_ROOT = REPO_ROOT / "runs" / "real_data_tuning" / "source"
JOINT_ROOT = REPO_ROOT / "runs" / "real_data_tuning" / "joint"
UNCAPPED_ROOT = REPO_ROOT / "runs" / "real_data_tuning" / "uncapped"
LOG_ROOT = REPO_ROOT / "runs" / "real_data_tuning" / "logs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the source screen, select on validation, run the joint "
            "screen, and confirm its selected candidate without clipping."
        )
    )
    parser.add_argument(
        "--devices",
        nargs=3,
        default=("cuda:0", "cuda:1", "cuda:2"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    return parser.parse_args()


def _manifest_count(root: Path) -> int:
    return sum(1 for _ in root.glob("*/seed_*/candidate_manifest.json"))


def _wait_for_count(
    root: Path,
    *,
    expected: int,
    poll_seconds: int,
    deadline: float,
) -> None:
    while True:
        count = _manifest_count(root)
        print(f"waiting for {root.name}: {count}/{expected}", flush=True)
        if count == expected:
            return
        if count > expected:
            raise RuntimeError(f"{root} contains more than {expected} manifests")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {root}")
        time.sleep(int(poll_seconds))


def _summarize(root: Path, *, phase: str, expected: int) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "summarize_real_data_tuning.py"),
            "--root",
            str(root),
            "--phase",
            phase,
            "--expected-candidates",
            str(expected),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return json.loads((root / "summary.json").read_text(encoding="utf-8"))


def _run_joint_shards(devices: list[str]) -> None:
    processes: list[tuple[subprocess.Popen, object]] = []
    for shard, device in enumerate(devices):
        log_path = LOG_ROOT / f"joint_shard{shard}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPTS / "run_real_data_joint_stability_sweep.py"),
                "--device",
                device,
                "--shard-index",
                str(shard),
                "--num-shards",
                str(len(devices)),
            ],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle))
    failures: list[int] = []
    for process, handle in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0:
            failures.append(returncode)
    if failures:
        raise RuntimeError(f"joint sweep shards failed: {failures}")


def _selected_row(summary: dict) -> dict:
    identifier = summary.get("selected_identifier")
    rows = [
        row
        for row in summary["candidates"]
        if row["identifier"] == identifier
    ]
    if len(rows) != 1:
        raise RuntimeError("validation selection did not identify one candidate")
    return rows[0]


def _run_uncapped_confirmation(
    *,
    source_summary: dict,
    joint_summary: dict,
    devices: list[str],
) -> None:
    source = _selected_row(source_summary)
    selected = _selected_row(joint_summary)
    training = selected["configuration"]["training"]
    identifier = str(selected["identifier"])
    processes: list[tuple[subprocess.Popen, object]] = []
    for position, seed in enumerate((1, 2)):
        run_dir = UNCAPPED_ROOT / identifier / f"seed_{seed}"
        manifest = run_dir / "candidate_manifest.json"
        if manifest.is_file():
            continue
        log_path = LOG_ROOT / f"uncapped_seed{seed}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPTS / "run_real_data_tuning_candidate.py"),
                "--phase",
                "joint",
                "--run-dir",
                str(run_dir),
                "--source-checkpoint",
                str(source["checkpoints"][str(seed)]),
                "--seed",
                str(seed),
                "--device",
                devices[position],
                "--steps",
                str(training["num_steps"]),
                "--lr",
                str(training["lr"]),
                "--ridge-lambda",
                str(training["ridge_lambda"]),
                "--lambda-scale",
                str(training["lambda_scale"]),
                "--eta",
                str(selected["configuration"]["eta"]),
                "--joint-weight",
                str(selected["configuration"]["joint_weight"]),
                "--grad-clip-norm",
                "0",
                "--bank-size",
                "4096",
                "--sample-batch-size",
                "4096",
                "--log-every",
                "50",
            ],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle))
    failures: list[int] = []
    for process, handle in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0:
            failures.append(returncode)
    if failures:
        raise RuntimeError(f"uncapped confirmation failed: {failures}")


def main() -> None:
    args = _parse_args()
    if int(args.poll_seconds) < 5 or float(args.timeout_hours) <= 0.0:
        raise ValueError("invalid polling configuration")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 3600.0 * float(args.timeout_hours)
    _wait_for_count(
        SOURCE_ROOT,
        expected=18,
        poll_seconds=int(args.poll_seconds),
        deadline=deadline,
    )
    source_summary = _summarize(SOURCE_ROOT, phase="source", expected=9)
    if not source_summary.get("selected_identifier"):
        raise RuntimeError("no source candidate passed every stability gate")
    print(
        f"selected source {source_summary['selected_identifier']}",
        flush=True,
    )
    _run_joint_shards(list(args.devices))
    joint_summary = _summarize(JOINT_ROOT, phase="joint", expected=7)
    if not joint_summary.get("selected_identifier"):
        raise RuntimeError("no joint candidate passed every stability gate")
    print(
        f"selected joint {joint_summary['selected_identifier']}",
        flush=True,
    )
    _run_uncapped_confirmation(
        source_summary=source_summary,
        joint_summary=joint_summary,
        devices=list(args.devices),
    )
    uncapped_summary = _summarize(
        UNCAPPED_ROOT,
        phase="joint",
        expected=1,
    )
    print(
        "uncapped confirmation selected="
        f"{uncapped_summary['selected_identifier']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
