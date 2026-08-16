#!/usr/bin/env python3
"""Run one centrally assigned lane of the locked final validation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_SEEDS = {"heston": (0, 1, 2, 3), "mixture": (0, 1, 3, 4)}
REAL_LANES = {
    "real_es_nq": ("ES", "NQ"),
    "real_rty_ym": ("RTY", "YM"),
    # Dedicated lanes allow the second index in each original serial lane to
    # use an otherwise idle GPU. Existing queue processes retain their loaded
    # lane definition and are not interrupted by adding these choices.
    "real_nq": ("NQ",),
    "real_ym": ("YM",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        choices=("heston", "mixture", *REAL_LANES),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> None:
    print("RUN " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _require_clean_target(target: Path) -> bool:
    complete = target / "COMPLETE.json"
    if complete.is_file():
        print(f"SKIP complete target {target}", flush=True)
        return False
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            f"refusing to resume nonempty incomplete target without audit: {target}"
        )
    return True


def _real_target_state(target: Path) -> str:
    if (target / "COMPLETE.json").is_file():
        print(f"SKIP complete target {target}", flush=True)
        return "complete"
    joint = target / "volatility_only_nested_mp" / "joint"
    if (joint / "outer_003_checkpoint.pt").is_file() and (
        joint / "candidate_manifest.json"
    ).is_file():
        print(f"RECOVER evaluation for trained target {target}", flush=True)
        return "evaluate"
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            f"refusing incomplete real target without a final checkpoint: {target}"
        )
    return "train"


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    locked = json.loads(protocol_path.read_text(encoding="utf-8"))
    final = locked.get("final_method", {})
    if bool(locked.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol authorizes protected split access")
    if final.get("solver") != "objective_safeguarded_nested_mp":
        raise RuntimeError("locked protocol does not select nested MP")
    if final.get("controlled_drift") != "identically_zero":
        raise RuntimeError("locked protocol does not select zero controlled drift")

    run_root = args.run_root.resolve()
    if args.lane in SYNTHETIC_SEEDS:
        task = str(args.lane)
        for seed in SYNTHETIC_SEEDS[task]:
            target = run_root / "synthetic" / task / f"seed_{seed}"
            if not args.dry_run and not _require_clean_target(target):
                continue
            command = [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "scripts"
                    / "run_matched_control_synthetic_validation.py"
                ),
                "--task",
                task,
                "--run-dir",
                str(target),
                "--device",
                args.device,
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                str(seed),
                "--final-zero-drift-only",
            ]
            _run(command, dry_run=bool(args.dry_run))
            if not args.dry_run:
                complete = {
                    "task": task,
                    "seed": seed,
                    "method": "nested_volatility_only_zero_drift",
                    "selection_scope": "validation_only",
                    "test_split_loaded": False,
                }
                (target / "COMPLETE.json").write_text(
                    json.dumps(complete, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return

    for index in REAL_LANES[str(args.lane)]:
        for seed in (0, 1, 3, 4):
            target = run_root / "real" / index / f"seed_{seed}"
            state = "train" if bool(args.dry_run) else _real_target_state(target)
            if state == "complete":
                continue
            command = [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "scripts"
                    / "run_matched_control_real_validation.py"
                ),
                "--index",
                index,
                "--run-dir",
                str(target),
                "--device",
                args.device,
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                str(seed),
                "--final-zero-drift-only",
            ]
            if state == "evaluate":
                command.append("--evaluation-only")
            _run(command, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
