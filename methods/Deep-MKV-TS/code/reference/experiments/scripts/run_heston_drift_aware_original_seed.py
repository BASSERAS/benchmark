#!/usr/bin/env python3
"""Run one seed of the locked drift-aware original-objective Heston campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (0, 1, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(int(seed) for seed in protocol.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError("the protocol seed set does not match this launcher")
    if not bool(protocol.get("multiseed_expansion_authorized", False)):
        raise RuntimeError("the protocol does not authorize multi-seed expansion")
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("the locked campaign must remain validation-only")

    run_dir = args.run_root.resolve() / f"seed_{int(args.seed)}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing nonempty run directory {run_dir}")
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_matched_control_synthetic_validation.py"),
        "--task", "heston",
        "--run-dir", str(run_dir),
        "--device", str(args.device),
        "--protocol-manifest", str(protocol_path),
        "--seed", str(int(args.seed)),
        "--bank-size", "8192",
        "--sample-batch-size", "2048",
        "--lambda-scale", "50",
        "--kappa-scale", "100",
        "--eta", "1",
        "--reference-kind", "local_gaussian",
        "--joint-volatility-weight", "0",
        "--source-joint-volatility-weight", "0",
        "--abs-return-acf-weight", "0.25",
        "--squared-return-acf-weight", "0.125",
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
