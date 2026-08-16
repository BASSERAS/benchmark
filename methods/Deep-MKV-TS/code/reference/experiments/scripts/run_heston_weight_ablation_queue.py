#!/usr/bin/env python3
"""Run the locked two-seed Heston discrepancy-pressure ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "relative_volatility_half": {
        "lambda_scale": 50.0,
        "kappa_scale": 25.0,
        "joint_volatility_weight": 0.5,
    },
    "all_discrepancy_half": {
        "lambda_scale": 25.0,
        "kappa_scale": 25.0,
        "joint_volatility_weight": 1.0,
    },
}
SEEDS = (1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("the locked ablation must be validation-only")
    if tuple(int(seed) for seed in protocol["seeds"]) != SEEDS:
        raise RuntimeError("the locked ablation seed set does not match the queue")
    if protocol["variants"] != VARIANTS:
        raise RuntimeError("the locked ablation variants do not match the queue")

    run_root = args.run_root.resolve()
    for variant, weights in VARIANTS.items():
        for seed in SEEDS:
            target = run_root / variant / f"seed_{seed}"
            complete = target / "COMPLETE.json"
            if complete.is_file():
                print(f"SKIP complete target {target}", flush=True)
                continue
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f"refusing nonempty incomplete target {target}")
            command = [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "scripts"
                    / "run_matched_control_synthetic_validation.py"
                ),
                "--task",
                "heston",
                "--run-dir",
                str(target),
                "--device",
                str(args.device),
                "--protocol-manifest",
                str(protocol_path),
                "--seed",
                str(seed),
                "--lambda-scale",
                str(weights["lambda_scale"]),
                "--kappa-scale",
                str(weights["kappa_scale"]),
                "--joint-volatility-weight",
                str(weights["joint_volatility_weight"]),
                "--final-zero-drift-only",
            ]
            print("RUN " + " ".join(command), flush=True)
            if bool(args.dry_run):
                continue
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            complete.write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "weights": weights,
                        "method": "nested_volatility_only_zero_drift",
                        "selection_scope": "validation_only",
                        "test_split_loaded": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
