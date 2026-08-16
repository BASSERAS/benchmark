#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize two-seed validation-only real-data tuning candidates."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("source", "joint"), required=True)
    parser.add_argument("--expected-candidates", type=int)
    return parser.parse_args()


def _nested(payload: dict[str, Any], *keys: str) -> float:
    value: Any = payload
    for key in keys:
        value = value[key]
    return float(value)


def _legacy_history_is_finite(manifest_path: Path) -> bool:
    """Validate runs written before the raw gradient diagnostic was renamed."""
    history_path = manifest_path.with_name("history.jsonl")
    if not history_path.is_file():
        return False
    required = (
        "train_total_loss",
        "sigma_mean",
        "sigma_q99",
        "sigma_max_active_fraction",
        "grad_norm",
        "grad_clip_scale",
        "grad_clip_active",
    )
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return False
    for record in records:
        if not all(
            name in record and math.isfinite(float(record[name]))
            for name in required
        ):
            return False
        gradient_fraction = record.get(
            "gradient_nonfinite_fraction",
            record.get("rolling_gradient_nonfinite_fraction_max"),
        )
        if gradient_fraction is None or not math.isfinite(
            float(gradient_fraction)
        ):
            return False
        if float(gradient_fraction) != 0.0:
            return False
    return True


def _seed_gate(
    payload: dict[str, Any], *, manifest_path: Path
) -> dict[str, bool]:
    stability = payload["stability"]
    bank = payload["bank_diagnostics"]
    law = payload["validation_law_metrics"]
    gradient_fraction = float(stability["gradient_nonfinite_fraction_max"])
    finite = bool(stability["all_logged_quantities_finite"]) and (
        math.isfinite(gradient_fraction) and gradient_fraction == 0.0
    )
    if not finite and not math.isfinite(gradient_fraction):
        finite = _legacy_history_is_finite(manifest_path)
    return {
        "finite": finite,
        "sigma_trajectory": float(stability["sigma_mean_max"]) <= 0.05
        and float(stability["sigma_cap_fraction_max"]) <= 0.01,
        "clipping_not_dominant": float(stability["clip_active_fraction"]) <= 0.25
        and float(stability["clip_scale_min"]) >= 0.10,
        "no_explosive_bank_tail": (
            float(bank["generated_rv_max_to_validation_max"]) <= 3.0
            and float(bank["generated_price_min"]) >= 90.0
            and float(bank["generated_price_max"]) <= 110.0
        ),
        "central_scale_plausible": (
            0.65 <= float(law["return_std_ratio"]) <= 1.35
            and 0.60 <= float(law["realized_volatility_mean_ratio"]) <= 1.40
            and 0.55 <= float(law["realized_volatility_q90_ratio"]) <= 1.45
        ),
    }


def _seed_metrics(payload: dict[str, Any]) -> dict[str, float]:
    law = payload["validation_law_metrics"]
    shadow = payload["validation_shadowing_metrics"]
    stability = payload["stability"]
    bank = payload["bank_diagnostics"]
    return {
        "return_qq": float(law["return_qq_rmse_normalized"]),
        "abs_acf": float(law["abs_return_acf_error_rms"]),
        "squared_acf": float(law["squared_return_acf_error_rms"]),
        "rv_wasserstein": float(
            law["realized_volatility_wasserstein_normalized"]
        ),
        "drawdown_wasserstein": float(
            law["maximum_drawdown_wasserstein_normalized"]
        ),
        "return_std_ratio": float(law["return_std_ratio"]),
        "rv_mean_ratio": float(law["realized_volatility_mean_ratio"]),
        "rv_q90_ratio": float(law["realized_volatility_q90_ratio"]),
        "rv_std_ratio": float(law["realized_volatility_std_ratio"]),
        "rv_crps": float(shadow["realized_volatility_crps"]),
        "rv_coverage90": float(
            shadow["realized_volatility_coverage_90"]
        ),
        "rv_width90": float(
            shadow["realized_volatility_band_width_90"]
        ),
        "cumulative_crps": float(shadow["cumulative_return_crps"]),
        "cumulative_coverage90": float(
            shadow["cumulative_return_coverage_90"]
        ),
        "prefix_distance": float(shadow["prefix_distance_mean"]),
        "clip_active_fraction": float(stability["clip_active_fraction"]),
        "clip_scale_min": float(stability["clip_scale_min"]),
        "sigma_mean_max": float(stability["sigma_mean_max"]),
        "grad_norm_max": float(stability["grad_norm_max"]),
        "rv_max_ratio": float(
            bank["generated_rv_max_to_validation_max"]
        ),
        "fraction_above_real_rv_q90": float(
            bank["generated_fraction_above_validation_rv_q90"]
        ),
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/seed_*/candidate_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload["phase"]) != str(args.phase):
            continue
        identifier = path.parents[1].name
        grouped.setdefault(identifier, {})[int(payload["seed"])] = payload
    if args.expected_candidates is not None and len(grouped) != int(
        args.expected_candidates
    ):
        raise RuntimeError(
            f"found {len(grouped)} candidates, expected {args.expected_candidates}"
        )
    rows: list[dict[str, Any]] = []
    for identifier, seeds in sorted(grouped.items()):
        if set(seeds) != {1, 2}:
            raise RuntimeError(
                f"{identifier} has seeds {sorted(seeds)}, expected [1, 2]"
            )
        gates = {
            seed: _seed_gate(
                payload,
                manifest_path=root
                / identifier
                / f"seed_{seed}"
                / "candidate_manifest.json",
            )
            for seed, payload in seeds.items()
        }
        seed_metrics = {
            seed: _seed_metrics(payload) for seed, payload in seeds.items()
        }
        metric_names = tuple(seed_metrics[1])
        aggregated = {
            name: _aggregate(
                [seed_metrics[seed][name] for seed in (1, 2)]
            )
            for name in metric_names
        }
        rows.append(
            {
                "identifier": identifier,
                "passes_all_gates": all(
                    all(seed_gate.values()) for seed_gate in gates.values()
                ),
                "gates": {str(seed): gate for seed, gate in gates.items()},
                "metrics_by_seed": {
                    str(seed): metrics
                    for seed, metrics in seed_metrics.items()
                },
                "aggregate": aggregated,
                "configuration": {
                    "eta": float(seeds[1]["eta"]),
                    "joint_weight": float(seeds[1]["joint_weight"]),
                    "training": seeds[1]["training"],
                },
                "checkpoints": {
                    str(seed): seeds[seed]["checkpoint"]
                    for seed in (1, 2)
                },
            }
        )

    eligible = [row for row in rows if bool(row["passes_all_gates"])]
    score_components = (
        "return_qq",
        "abs_acf",
        "squared_acf",
        "rv_wasserstein",
        "drawdown_wasserstein",
        "rv_crps",
        "cumulative_crps",
    )
    if eligible:
        medians = {
            name: sorted(
                float(row["aggregate"][name]["mean"]) for row in eligible
            )[len(eligible) // 2]
            for name in score_components
        }
        for row in eligible:
            normalized = [
                float(row["aggregate"][name]["mean"])
                / max(float(medians[name]), 1e-12)
                for name in score_components
            ]
            coverage_penalty = abs(
                float(row["aggregate"]["rv_coverage90"]["mean"]) - 0.90
            ) / 0.10
            scale_penalty = (
                abs(float(row["aggregate"]["return_std_ratio"]["mean"]) - 1.0)
                + abs(float(row["aggregate"]["rv_mean_ratio"]["mean"]) - 1.0)
                + abs(float(row["aggregate"]["rv_q90_ratio"]["mean"]) - 1.0)
            )
            row["validation_selection_score"] = float(
                mean(normalized) + 0.25 * coverage_penalty + 0.25 * scale_penalty
            )
        eligible.sort(
            key=lambda row: float(row["validation_selection_score"])
        )
    selected = eligible[0] if eligible else None
    payload = {
        "selection_scope": "training_and_validation_only",
        "test_or_event_data_seen": False,
        "phase": str(args.phase),
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "score_components": list(score_components),
        "selected_identifier": (
            None if selected is None else selected["identifier"]
        ),
        "candidates": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Real-data {args.phase} tuning summary",
        "",
        "Selection uses training and validation only. Test and event windows are unseen.",
        "",
        "| Candidate | Gate | Score | Clip fraction | RV q90 ratio | RV CRPS | RV coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            not bool(item["passes_all_gates"]),
            float(item.get("validation_selection_score", math.inf)),
            str(item["identifier"]),
        ),
    ):
        aggregate = row["aggregate"]
        score = row.get("validation_selection_score")
        lines.append(
            "| {identifier} | {gate} | {score} | {clip:.3f} | "
            "{q90:.3f} | {crps:.6f} | {coverage:.3f} |".format(
                identifier=row["identifier"],
                gate="pass" if row["passes_all_gates"] else "fail",
                score="--" if score is None else f"{float(score):.4f}",
                clip=float(aggregate["clip_active_fraction"]["mean"]),
                q90=float(aggregate["rv_q90_ratio"]["mean"]),
                crps=float(aggregate["rv_crps"]["mean"]),
                coverage=float(aggregate["rv_coverage90"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            f"Selected: `{payload['selected_identifier']}`",
            "",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"summarized {len(rows)} candidates; "
        f"eligible={len(eligible)} selected={payload['selected_identifier']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
