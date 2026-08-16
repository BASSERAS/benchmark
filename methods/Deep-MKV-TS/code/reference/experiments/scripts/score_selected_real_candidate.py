#!/usr/bin/env python3
"""Apply the frozen validation gate to one selected real-data candidate.

This script deliberately reads only the candidate's validation artifacts and the
previously computed reference-ablation summary.  It never loads ``test.npy`` or
an event/crisis split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CRPS_NAMES = (
    "cumulative_return_crps",
    "increment_crps",
    "realized_volatility_crps",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("runs/real_reference_ablation_4seed_20260802/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "runs/selected_deep_mkv_online_completion_20260805/real/"
            "FROZEN_PROTOCOL.json"
        ),
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError(f"expected a two-dimensional price bank, got {prices.shape}")
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("price bank must be finite and strictly positive")
    return np.diff(np.log(prices), axis=1)


def score_candidate(
    *,
    run_dir: Path,
    reference_summary: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_summary.read_text(encoding="utf-8"))

    if bool(manifest.get("test_or_event_data_seen", True)):
        raise RuntimeError("candidate is not validation-isolated")

    generated = np.load(run_dir / "validation_screen_bank.npy")
    validation = np.load(run_dir / "adapter_dataset" / "disc.npy")
    generated_returns = _returns(generated)
    validation_returns = _returns(validation)
    generated_abs = np.abs(generated_returns).reshape(-1)
    validation_abs = np.abs(validation_returns).reshape(-1)

    law = manifest["validation_law_metrics"]
    shadow = manifest["validation_shadowing_metrics"]
    index = str(manifest["index"])
    seed = int(manifest["seed"])
    reference_metrics = reference["per_seed"][index]["reference"][str(seed)]

    q99_ratio = float(
        np.quantile(generated_abs, 0.99)
        / max(float(np.quantile(validation_abs, 0.99)), 1e-15)
    )
    q999_ratio = float(
        np.quantile(generated_abs, 0.999)
        / max(float(np.quantile(validation_abs, 0.999)), 1e-15)
    )
    target_kurtosis = float(law["target_excess_kurtosis"])
    generated_kurtosis = float(law["generated_excess_kurtosis"])
    kurtosis_limit = max(2.5 * target_kurtosis, target_kurtosis + 10.0)
    bounds = protocol["scale_tail_bounds"]

    crps_improvements = {
        name: float(reference_metrics[name]) - float(shadow[name])
        for name in CRPS_NAMES
    }
    checks = {
        "validation_isolated": not bool(manifest["test_or_event_data_seen"]),
        "finite": bool(np.isfinite(generated).all())
        and all(
            math.isfinite(float(value))
            for value in (
                law["return_std_ratio"],
                law["realized_volatility_mean_ratio"],
                law["realized_volatility_std_ratio"],
                q99_ratio,
                q999_ratio,
                generated_kurtosis,
            )
        ),
        "return_std_ratio": float(bounds["return_std_ratio"][0])
        <= float(law["return_std_ratio"])
        <= float(bounds["return_std_ratio"][1]),
        "rv_mean_ratio": float(bounds["realized_volatility_mean_ratio"][0])
        <= float(law["realized_volatility_mean_ratio"])
        <= float(bounds["realized_volatility_mean_ratio"][1]),
        "rv_std_ratio": float(bounds["realized_volatility_std_ratio"][0])
        <= float(law["realized_volatility_std_ratio"])
        <= float(bounds["realized_volatility_std_ratio"][1]),
        "absolute_return_q99_ratio": q99_ratio
        <= float(bounds["absolute_return_q99_ratio_max"]),
        "absolute_return_q999_ratio": q999_ratio
        <= float(bounds["absolute_return_q999_ratio_max"]),
        "excess_kurtosis": generated_kurtosis <= kurtosis_limit,
        "all_three_shadowing_crps_improve": all(
            improvement > 0.0 for improvement in crps_improvements.values()
        ),
    }
    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "index": index,
        "seed": seed,
        "selection_scope": "validation_only",
        "test_or_event_data_seen": False,
        "checks": checks,
        "passes_frozen_gate": all(checks.values()),
        "law_metrics": {
            "return_std_ratio": float(law["return_std_ratio"]),
            "realized_volatility_mean_ratio": float(
                law["realized_volatility_mean_ratio"]
            ),
            "realized_volatility_std_ratio": float(
                law["realized_volatility_std_ratio"]
            ),
            "absolute_return_q99_ratio": q99_ratio,
            "absolute_return_q999_ratio": q999_ratio,
            "generated_excess_kurtosis": generated_kurtosis,
            "target_excess_kurtosis": target_kurtosis,
            "excess_kurtosis_limit": kurtosis_limit,
        },
        "candidate_crps": {name: float(shadow[name]) for name in CRPS_NAMES},
        "reference_crps": {
            name: float(reference_metrics[name]) for name in CRPS_NAMES
        },
        "crps_improvements": crps_improvements,
    }
    (run_dir / "FROZEN_GATE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = _parse_args()
    payload = score_candidate(
        run_dir=args.run_dir,
        reference_summary=args.reference_summary,
        protocol_path=args.protocol,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pass and not bool(payload["passes_frozen_gate"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
