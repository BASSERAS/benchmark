#!/usr/bin/env python3
"""Run one locked seed-zero Heston targeted-feature arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARMS = {"acf_focused", "conditional_focused", "combined"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(sorted(EXPECTED_ARMS)), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("the locked experiment must remain validation-only")
    if int(protocol.get("seed", -1)) != 0:
        raise RuntimeError("the locked experiment must use seed zero")
    arms = protocol.get("arms", {})
    if set(arms) != EXPECTED_ARMS:
        raise RuntimeError("the protocol arm set does not match this runner")

    arm = str(args.arm)
    weights = arms[arm]
    run_dir = args.run_root.resolve() / arm / "seed_0"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty run directory {run_dir}")

    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_matched_control_synthetic_validation.py"),
        "--task", "heston",
        "--run-dir", str(run_dir),
        "--device", str(args.device),
        "--protocol-manifest", str(protocol_path),
        "--seed", "0",
        "--bank-size", "8192",
        "--sample-batch-size", "2048",
        "--lambda-scale", "50",
        "--kappa-scale", "100",
        "--eta", "1",
        "--reference-kind", "local_gaussian",
        "--joint-volatility-weight", "0",
        "--source-joint-volatility-weight", str(weights["source_joint_volatility_weight"]),
        "--abs-return-acf-weight", str(weights["abs_return_acf_weight"]),
        "--squared-return-acf-weight", str(weights["squared_return_acf_weight"]),
        "--source-only",
        "--source-steps", "3000",
        "--source-checkpoint-steps", "500", "1000", "1500", "2000", "2500", "3000",
        "--solver", "online",
        "--fitted-reference-drift-only",
        "--adjoint-weight", "0",
        "--adjoint-noise-weight", "1",
        "--lr", "0.002",
        "--grad-clip-norm", "5",
        "--path-derivative-backend", "autograd",
        "--drift-adjoint-backend", "analytical_reference",
    ]
    print("RUN " + " ".join(command), flush=True)
    if not bool(args.dry_run):
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
