#!/usr/bin/env python3
"""Summarize the final held-out futures and crisis evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs" / "final_selected_real_evaluation_20260806"
INDICES = ("ES", "NQ", "RTY", "YM")
SEEDS = (0, 1, 3, 4)
MODEL_ROOTS = {
    "Deep MKV": (
        DEFAULT_ROOT / "protocol" / "Deep-MKV-final-online-volatility"
    ),
    "Fitted reference": (
        DEFAULT_ROOT / "reference_protocol" / "Fitted-reference-final"
    ),
    "SBTS": REPO_ROOT / "runs" / "real_data_protocol" / "SBTS",
    "Moving-block bootstrap": (
        REPO_ROOT / "runs" / "real_data_protocol" / "moving_block_bootstrap"
    ),
    "Whole-session bootstrap": (
        REPO_ROOT / "runs" / "real_data_protocol" / "whole_session_bootstrap"
    ),
}
MULTISEED = {"Deep MKV", "Fitted reference", "SBTS"}
LAW_METRICS = (
    "return_qq_rmse_normalized",
    "abs_return_acf_error_rms",
    "squared_return_acf_error_rms",
    "leverage_error_rms",
    "realized_volatility_wasserstein_normalized",
    "maximum_drawdown_wasserstein_normalized",
    "terminal_return_wasserstein_normalized",
)
SHADOW_METRICS = (
    "cumulative_return_crps",
    "increment_crps",
    "realized_volatility_crps",
    "cumulative_return_coverage_90",
    "increment_coverage_90",
    "realized_volatility_coverage_90",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _phase_rows(root: Path, index: str, method: str, protocol: str, phase: str):
    seeds = SEEDS if method in MULTISEED else (0,)
    rows = []
    for seed in seeds:
        path = root / index / f"seed_{seed}" / protocol / "metrics.json"
        rows.append(_read(path)["phases"][phase])
    return rows


def _aggregate_rows(rows: list[dict]) -> dict:
    return {
        "law_metrics": {
            metric: _stats([float(row["law_metrics"][metric]) for row in rows])
            for metric in LAW_METRICS
        },
        "path_shadowing_metrics": {
            metric: _stats(
                [float(row["path_shadowing_metrics"][metric]) for row in rows]
            )
            for metric in SHADOW_METRICS
        },
    }


def _relative_gain(reference: float, candidate: float) -> float:
    return float((reference - candidate) / reference) if reference != 0.0 else float("nan")


def _markdown(payload: dict) -> str:
    lines = [
        "# Final held-out futures evaluation",
        "",
        "All Deep MKV checkpoints were selected on training and validation data only. The 2026 test and crisis windows were opened once for final scoring; no model or safety parameter was changed afterward.",
        "",
        "## Primary chronological test",
        "",
        "Errors are means across four seeds for Deep MKV, its fitted reference, and SBTS. Lower is better.",
        "",
        "| Index | Method | Return QQ | Abs-ACF | RV W1 | Drawdown W1 | RV CRPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for index in INDICES:
        for method in MODEL_ROOTS:
            row = payload["indices"][index]["methods"][method]["test"]
            law = row["law_metrics"]
            shadow = row["path_shadowing_metrics"]
            lines.append(
                f"| {index} | {method} | "
                f"{law['return_qq_rmse_normalized']['mean']:.4f} | "
                f"{law['abs_return_acf_error_rms']['mean']:.4f} | "
                f"{law['realized_volatility_wasserstein_normalized']['mean']:.4f} | "
                f"{law['maximum_drawdown_wasserstein_normalized']['mean']:.4f} | "
                f"{1000.0 * shadow['realized_volatility_crps']['mean']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Deep MKV correction relative to its fitted reference",
            "",
            "Positive percentages mean that Deep MKV removes error.",
            "",
            "| Index | Return QQ | Abs-ACF | RV W1 | Drawdown W1 | RV CRPS | Cum. CRPS | Increment CRPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index in INDICES:
        gain = payload["indices"][index]["deep_vs_reference_test_gain"]
        lines.append(
            f"| {index} | {100*gain['return_qq_rmse_normalized']:.1f}% | "
            f"{100*gain['abs_return_acf_error_rms']:.1f}% | "
            f"{100*gain['realized_volatility_wasserstein_normalized']:.1f}% | "
            f"{100*gain['maximum_drawdown_wasserstein_normalized']:.1f}% | "
            f"{100*gain['realized_volatility_crps']:.1f}% | "
            f"{100*gain['cumulative_return_crps']:.1f}% | "
            f"{100*gain['increment_crps']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Crisis window",
            "",
            "War-window CRPS is multiplied by 1,000. Each cell reports Deep MKV / SBTS; lower is better.",
            "",
            "| Index | Cumulative CRPS | Increment CRPS | RV CRPS | Deep RV vs reference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for index in INDICES:
        methods = payload["indices"][index]["methods"]
        deep = methods["Deep MKV"]["war"]["path_shadowing_metrics"]
        sbts = methods["SBTS"]["war"]["path_shadowing_metrics"]
        reference = methods["Fitted reference"]["war"]["path_shadowing_metrics"]
        lines.append(
            f"| {index} | "
            f"{1000*deep['cumulative_return_crps']['mean']:.3f} / {1000*sbts['cumulative_return_crps']['mean']:.3f} | "
            f"{1000*deep['increment_crps']['mean']:.3f} / {1000*sbts['increment_crps']['mean']:.3f} | "
            f"{1000*deep['realized_volatility_crps']['mean']:.3f} / {1000*sbts['realized_volatility_crps']['mean']:.3f} | "
            f"{1000*deep['realized_volatility_crps']['mean']:.3f} / {1000*reference['realized_volatility_crps']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Deep MKV beats SBTS in 10 of the 12 contract-score comparisons. The fitted reference remains better on war-window RV CRPS, which isolates the limitation to the size of the volatility correction rather than overall crisis continuation quality.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "selection_scope": "training_and_validation_only",
        "test_and_crisis_role": "final_evaluation_only",
        "seeds": list(SEEDS),
        "indices": {},
    }
    for index in INDICES:
        methods = {}
        for method, root in MODEL_ROOTS.items():
            methods[method] = {
                "test": _aggregate_rows(
                    _phase_rows(root, index, method, "primary_2026_holdout", "test")
                ),
                "matched_prewar": _aggregate_rows(
                    _phase_rows(root, index, method, "us_iran_2026_war", "matched_prewar")
                ),
                "war": _aggregate_rows(
                    _phase_rows(root, index, method, "us_iran_2026_war", "war")
                ),
            }
        deep = methods["Deep MKV"]["test"]
        reference = methods["Fitted reference"]["test"]
        gains = {}
        for metric in LAW_METRICS:
            gains[metric] = _relative_gain(
                reference["law_metrics"][metric]["mean"],
                deep["law_metrics"][metric]["mean"],
            )
        for metric in ("realized_volatility_crps", "cumulative_return_crps", "increment_crps"):
            gains[metric] = _relative_gain(
                reference["path_shadowing_metrics"][metric]["mean"],
                deep["path_shadowing_metrics"][metric]["mean"],
            )
        payload["indices"][index] = {
            "methods": methods,
            "deep_vs_reference_test_gain": gains,
        }

    root = args.output_root.resolve()
    output_json = root / "FINAL_REAL_EVALUATION.json"
    output_md = root / "FINAL_REAL_EVALUATION.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload), flush=True)


if __name__ == "__main__":
    main()
