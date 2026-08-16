#!/usr/bin/env python3
"""Aggregate frozen-gate results for the selected real-data campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


SEEDS = (0, 1, 3, 4)
INDICES = ("ES", "NQ", "RTY", "YM")
LAW_NAMES = (
    "return_std_ratio",
    "realized_volatility_mean_ratio",
    "realized_volatility_std_ratio",
    "absolute_return_q99_ratio",
    "absolute_return_q999_ratio",
    "generated_excess_kurtosis",
)
CRPS_NAMES = (
    "cumulative_return_crps",
    "increment_crps",
    "realized_volatility_crps",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path(
            "runs/selected_deep_mkv_online_completion_20260805/real"
        ),
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=INDICES,
        default=INDICES,
    )
    parser.add_argument(
        "--layout",
        choices=("legacy_selected", "flat"),
        default="legacy_selected",
        help=(
            "Use the historical selected-campaign directory names or the "
            "flat INDEX/seed_SEED layout used by final matched campaigns."
        ),
    )
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_sd": float(stdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _gate_path(root: Path, index: str, seed: int, *, layout: str) -> Path:
    if layout == "flat":
        return root / index / f"seed_{seed}" / "FROZEN_GATE.json"
    if index == "ES" and seed == 0:
        return root / "ES_early_eta2_capq99" / "FROZEN_GATE.json"
    if seed == 0:
        return root / f"transfer_{index}_seed0" / "FROZEN_GATE.json"
    return root / f"{index}_4seed" / f"seed_{seed}" / "FROZEN_GATE.json"


def main() -> None:
    args = _parse_args()
    root = args.real_root.resolve()
    result: dict[str, Any] = {
        "method": "online source-stage volatility-only Deep MKV",
        "selection_scope": "validation_only",
        "test_or_event_data_seen": False,
        "indices": {},
    }
    panel: list[dict[str, Any]] = []
    selected_indices = tuple(str(value) for value in args.indices)
    for index in selected_indices:
        seeds: dict[str, Any] = {}
        complete_payloads: list[dict[str, Any]] = []
        for seed in SEEDS:
            path = _gate_path(root, index, seed, layout=str(args.layout))
            if not path.is_file():
                seeds[str(seed)] = {"status": "missing", "path": str(path)}
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            seeds[str(seed)] = {"status": "complete", **payload}
            complete_payloads.append(payload)
            panel.append(payload)
        law_summary = {
            name: _stats(
                [float(payload["law_metrics"][name]) for payload in complete_payloads]
            )
            for name in LAW_NAMES
            if complete_payloads
        }
        crps_summary = {
            name: _stats(
                [float(payload["crps_improvements"][name]) for payload in complete_payloads]
            )
            for name in CRPS_NAMES
            if complete_payloads
        }
        result["indices"][index] = {
            "complete_count": len(complete_payloads),
            "pass_count": sum(
                bool(payload["passes_frozen_gate"])
                for payload in complete_payloads
            ),
            "seeds": seeds,
            "law_metrics": law_summary,
            "crps_improvements": crps_summary,
        }
    result["panel"] = {
        "complete_count": len(panel),
        "pass_count": sum(bool(payload["passes_frozen_gate"]) for payload in panel),
        "crps_improvements": {
            name: _stats(
                [float(payload["crps_improvements"][name]) for payload in panel]
            )
            for name in CRPS_NAMES
            if panel
        },
    }
    output = root / "REAL_SUMMARY.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Selected Deep MKV real-data validation campaign",
        "",
        "No protected test or event split was loaded.",
        "",
        "| Index | Complete | Passed | Cum. CRPS improvement | Inc. CRPS improvement | RV CRPS improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index in selected_indices:
        row = result["indices"][index]
        improvements = row["crps_improvements"]
        values = [
            improvements[name]["mean"] if name in improvements else float("nan")
            for name in CRPS_NAMES
        ]
        lines.append(
            f"| {index} | {row['complete_count']}/4 | {row['pass_count']}/4 | "
            f"{values[0]:.6g} | {values[1]:.6g} | {values[2]:.6g} |"
        )
    lines.extend(
        [
            "",
            "A positive CRPS improvement means the selected method is better than the matched fitted reference.",
            "",
        ]
    )
    (root / "REAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
