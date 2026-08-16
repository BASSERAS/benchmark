#!/usr/bin/env python3
"""Generate and evaluate the remaining real-data financial baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
SEEDS = (0, 1, 3, 4)
INDICES = ("ES", "NQ", "RTY", "YM")
PROTOCOLS = ("primary_2026_holdout", "us_iran_2026_war")
STATUS_PATH = (
    REPO_ROOT / "runs" / "paper_remaining_queues_20260801" / "baselines.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    print(f"[{stamp}] RUN {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def evaluated(model: str, index: str, seed: int) -> bool:
    root = REPO_ROOT / "runs" / "real_data_protocol" / model / index / f"seed_{seed}"
    return all((root / protocol / "metrics.json").exists() for protocol in PROTOCOLS)


def evaluation_command(*, index: str, seed: int, model: str) -> list[str]:
    generated = REPO_ROOT / "runs" / "real_data_protocol" / "generated" / "sbts"
    return [
        str(PYTHON),
        str(REPO_ROOT / "experiments/scripts/run_real_data_evaluation.py"),
        "--generated-root", str(generated),
        "--model-name", model,
        "--index", index,
        "--seed", str(seed),
        "--protocol", PROTOCOLS[0],
        "--protocol", PROTOCOLS[1],
    ]


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(
            json.dumps({"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
            encoding="utf-8",
        )
    generated = REPO_ROOT / "runs" / "real_data_protocol" / "generated" / "sbts"
    for index in INDICES:
        for seed in SEEDS:
            bank = generated / "primary_2026_holdout" / index / f"seed_{seed}.npy"
            if not bank.exists():
                run(
                    [
                        str(PYTHON),
                        str(REPO_ROOT / "experiments/scripts/run_real_data_sbts.py"),
                        "--protocol", "primary_2026_holdout",
                        "--index", index,
                        "--seed", str(seed),
                        "--bank-size", "8192",
                        "--workers", "32",
                    ],
                    dry_run=args.dry_run,
                )
            if not evaluated("SBTS", index, seed):
                run(
                    evaluation_command(index=index, seed=seed, model="SBTS"),
                    dry_run=args.dry_run,
                )

    for baseline, model in (
        ("moving_block", "moving_block_bootstrap"),
        ("whole_session", "whole_session_bootstrap"),
    ):
        for index in INDICES:
            if evaluated(model, index, 0):
                continue
            run(
                [
                    str(PYTHON),
                    str(REPO_ROOT / "experiments/scripts/run_real_data_evaluation.py"),
                    "--baseline", baseline,
                    "--model-name", model,
                    "--index", index,
                    "--seed", "0",
                    "--protocol", PROTOCOLS[0],
                    "--protocol", PROTOCOLS[1],
                ],
                dry_run=args.dry_run,
            )
    if not args.dry_run:
        STATUS_PATH.write_text(
            json.dumps({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
