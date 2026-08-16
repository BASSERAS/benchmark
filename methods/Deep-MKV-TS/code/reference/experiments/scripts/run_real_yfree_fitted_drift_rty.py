#!/usr/bin/env python3
"""Run one seed of the frozen Y-free RTY small-cap transfer diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from run_real_yfree_fitted_drift_queue import (  # noqa: E402
    REFERENCE_SUMMARY,
    completed,
    run_logged,
    train_command,
)


RUN_ROOT = REPO_ROOT / "runs" / "real_yweight0_fitted_drift_rty_4seed_20260808"
PROTOCOL = RUN_ROOT / "LOCKED_PROTOCOL.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 3, 4), required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def score_command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "score_selected_real_candidate.py"),
        "--run-dir", str(run_dir),
        "--reference-summary", str(REFERENCE_SUMMARY),
        "--protocol", str(PROTOCOL),
    ]


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol must forbid test access")
    run_dir = RUN_ROOT / "RTY" / f"seed_{int(args.seed)}"
    if completed(run_dir, index="RTY", seed=int(args.seed)):
        print(f"SKIP complete RTY seed={args.seed}", flush=True)
        return
    print(f"START RTY seed={args.seed} device={args.device}", flush=True)
    run_logged(
        train_command(
            index="RTY",
            seed=int(args.seed),
            device=str(args.device),
            run_dir=run_dir,
        ),
        log_path=run_dir / "run.log",
        mode="w",
    )
    run_logged(
        score_command(run_dir),
        log_path=run_dir / "run.log",
        mode="a",
    )
    if not completed(run_dir, index="RTY", seed=int(args.seed)):
        raise RuntimeError(f"completion audit failed for RTY seed={args.seed}")
    print(f"DONE RTY seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
