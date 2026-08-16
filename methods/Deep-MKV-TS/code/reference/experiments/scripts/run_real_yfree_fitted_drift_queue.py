#!/usr/bin/env python3
"""Run one GPU lane of the frozen Y-free real-futures validation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT / "runs" / "real_yweight0_fitted_drift_3index_4seed_20260808"
)
PROTOCOL = RUN_ROOT / "LOCKED_PROTOCOL.json"
REFERENCE_SUMMARY = (
    REPO_ROOT / "runs" / "real_reference_ablation_4seed_20260802" / "summary.json"
)
REFERENCE_WEIGHT = {"ES": 0.55, "NQ": 0.55, "RTY": 0.65, "YM": 0.75}
LANES = {
    0: (("ES", 0), ("ES", 4), ("YM", 3)),
    1: (("ES", 1), ("NQ", 0), ("YM", 4)),
    2: (("ES", 3), ("NQ", 1), ("YM", 0)),
    3: (("NQ", 3), ("NQ", 4), ("YM", 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def completed(run_dir: Path, *, index: str, seed: int) -> bool:
    manifest_path = run_dir / "candidate_manifest.json"
    bank_path = run_dir / "validation_screen_bank.npy"
    gate_path = run_dir / "FROZEN_GATE.json"
    if not (manifest_path.is_file() and bank_path.is_file() and gate_path.is_file()):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = manifest.get("training", {})
    return (
        str(manifest.get("index")) == index
        and int(manifest.get("seed", -1)) == int(seed)
        and str(manifest.get("solver")) == "online"
        and str(manifest.get("drift_control_mode")) == "volatility_only"
        and bool(manifest.get("use_reference_drift")) is True
        and float(training.get("adjoint_weight", -1.0)) == 0.0
        and float(training.get("adjoint_noise_weight", -1.0)) == 1.0
        and bool(manifest.get("test_or_event_data_seen", True)) is False
    )


def train_command(
    *, index: str, seed: int, device: str, run_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "run_real_data_tuning_candidate.py"),
        "--phase", "source",
        "--run-dir", str(run_dir),
        "--protocol", "primary_2026_holdout",
        "--objective-protocol", "real_scale_corrected_v3_family_adjoint_rms",
        "--index", index,
        "--seed", str(seed),
        "--device", device,
        "--state-unit-scaling", "train_return_volatility",
        "--steps", "400",
        "--adjoint-weight", "0",
        "--adjoint-noise-weight", "1",
        "--solver", "online",
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
        "--reference-causal-weight", str(REFERENCE_WEIGHT[index]),
        "--lambda-scale", "1",
        "--eta", "2",
        "--running-cost", "gaussian_relative_entropy",
        "--use-reference-drift",
        "--drift-control-mode", "volatility_only",
        "--lambda-x", "50",
        "--lambda-v", "100",
        "--joint-weight", "0.25",
        "--return-discrepancy", "mmd",
        "--global-rv-discrepancy", "mmd",
        "--acf-discrepancy", "mmd",
        "--conditional-claim-mode", "none",
        "--discrepancy-stack", "full",
        "--joint-block-multiplier", "1",
        "--grad-clip-norm", "0",
        "--sigma-cap-quantile", "0.99",
        "--sigma-cap-std-multiple", "3",
        "--bank-size", "8192",
        "--sample-batch-size", "4096",
        "--log-every", "50",
    ]


def score_command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "score_selected_real_candidate.py"),
        "--run-dir", str(run_dir),
        "--reference-summary", str(REFERENCE_SUMMARY),
        "--protocol", str(PROTOCOL),
    ]


def run_logged(command: list[str], *, log_path: Path, mode: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open(mode, encoding="utf-8") as stream:
        stream.write("RUN " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol must forbid test access")
    for index, seed in LANES[int(args.lane)]:
        run_dir = RUN_ROOT / index / f"seed_{seed}"
        if completed(run_dir, index=index, seed=seed):
            print(f"SKIP complete {index} seed={seed}", flush=True)
            continue
        print(f"START {index} seed={seed} device={args.device}", flush=True)
        run_logged(
            train_command(
                index=index,
                seed=seed,
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
        if not completed(run_dir, index=index, seed=seed):
            raise RuntimeError(f"completion audit failed for {index} seed={seed}")
        print(f"DONE {index} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
