#!/usr/bin/env python3
"""Aggregate the frozen online volatility-only Heston and mixture campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 3, 4)
GROUPS = {
    "distribution": (
        "path_swd_normalized",
        "terminal_w1_normalized",
        "increment_w1_normalized",
        "realized_volatility_w1_normalized",
        "maximum_drawdown_w1_normalized",
        "return_qq_rmse_normalized",
    ),
    "dependence": (
        "return_acf_rmse",
        "absolute_return_acf_rmse",
        "squared_return_acf_rmse",
    ),
    "local_calibration": (
        "timewise_return_mean_rmse_normalized",
        "timewise_return_std_rmse_normalized",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=REPO_ROOT / "runs" / "selected_deep_mkv_online_completion_20260805",
    )
    parser.add_argument(
        "--heston-run-root",
        type=Path,
        default=None,
        help=(
            "Optional four-seed Heston run root. When supplied, seeds 1/3/4 "
            "are read from seed_<seed> below this root."
        ),
    )
    parser.add_argument(
        "--heston-seed0-checkpoint-step",
        type=int,
        default=3000,
        help="Globally selected seed-zero Heston trajectory checkpoint.",
    )
    parser.add_argument(
        "--mixture-run-root",
        type=Path,
        default=None,
        help=(
            "Optional four-seed mixture run root. When supplied, every seed "
            "is read from seed_<seed> below this root."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_sd": float(stdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def closure(reference: float, candidate: float) -> float | None:
    if not math.isfinite(reference) or reference <= 1e-12 or not math.isfinite(candidate):
        return None
    return (reference - candidate) / reference


def roots(
    campaign: Path,
    task: str,
    seed: int,
    *,
    heston_run_root: Path | None,
    heston_seed0_checkpoint_step: int,
    mixture_run_root: Path | None,
) -> tuple[Path, Path]:
    if (
        seed == 0
        and task == "heston"
        and heston_run_root is not None
        and (heston_run_root / "seed_0").is_dir()
    ):
        root = heston_run_root / "seed_0"
        return root / "reference", root / "volatility_only_online_mp"
    if seed == 0 and task == "heston":
        root = REPO_ROOT / "runs" / "heston_online_fitted_drift_volatility_only_50_100_trajectory_seed0_20260805"
        candidate = root / "volatility_only_online_mp"
        if int(heston_seed0_checkpoint_step) != 3000:
            candidate = (
                candidate
                / "checkpoint_evaluations"
                / f"step_{int(heston_seed0_checkpoint_step):04d}"
            )
        return root / "reference", candidate
    if task == "mixture" and mixture_run_root is not None:
        root = mixture_run_root / f"seed_{seed}"
        return root / "reference", root / "volatility_only_online_mp"
    if seed == 0 and task == "mixture":
        root = REPO_ROOT / "runs" / "mixture_online_zero_drift_eta1p5_50_100_trajectory_seed0_20260805"
        return root / "reference", root / "volatility_only_online_mp" / "checkpoint_evaluations" / "step_2500"
    root = (
        heston_run_root / f"seed_{seed}"
        if task == "heston" and heston_run_root is not None
        else campaign / task / f"seed_{seed}"
    )
    return root / "reference", root / "volatility_only_online_mp"


def memory_errors(metrics: dict[str, object]) -> dict[str, float]:
    memory = metrics["conditional_path_memory"]
    generated = memory["generated"]
    target = memory["target"]
    return {
        "prefix_future_rv_correlation_error": abs(
            float(generated["prefix_future_rv_correlation"])
            - float(target["prefix_future_rv_correlation"])
        ),
        "prefix_high_low_future_rv_gap_error": abs(
            float(generated["prefix_high_low_future_rv_gap"])
            - float(target["prefix_high_low_future_rv_gap"])
        ),
    }


def main() -> None:
    args = parse_args()
    campaign = args.campaign_root.resolve()
    heston_run_root = (
        args.heston_run_root.resolve() if args.heston_run_root is not None else None
    )
    mixture_run_root = (
        args.mixture_run_root.resolve() if args.mixture_run_root is not None else None
    )
    result: dict[str, object] = {
        "seeds": list(SEEDS),
        "tasks": {},
        "selection": {
            "heston_seed0_checkpoint_step": int(args.heston_seed0_checkpoint_step),
            "heston_run_root": (
                str(heston_run_root) if heston_run_root is not None else None
            ),
            "mixture_run_root": (
                str(mixture_run_root) if mixture_run_root is not None else None
            ),
        },
    }
    for task in ("heston", "mixture"):
        per_seed: dict[str, object] = {}
        metric_values: dict[str, list[float]] = {}
        gap_values: dict[str, list[float]] = {}
        group_values: dict[str, list[float]] = {
            **{name: [] for name in GROUPS},
            "conditional_memory": [],
        }
        mixture_gap_values: dict[str, list[float]] = {}
        complete = 0
        for seed in SEEDS:
            reference_root, candidate_root = roots(
                campaign,
                task,
                seed,
                heston_run_root=heston_run_root,
                heston_seed0_checkpoint_step=int(
                    args.heston_seed0_checkpoint_step
                ),
                mixture_run_root=mixture_run_root,
            )
            required = (
                reference_root / "metrics.json",
                candidate_root / "metrics.json",
                candidate_root / "control_diagnostics.json",
            )
            if task == "mixture":
                required = (
                    *required,
                    reference_root / "mixture_metrics.json",
                    candidate_root / "mixture_metrics.json",
                )
            if not all(path.is_file() for path in required):
                per_seed[str(seed)] = {
                    "status": "missing",
                    "missing": [str(path) for path in required if not path.is_file()],
                }
                continue
            reference = read_json(reference_root / "metrics.json")
            candidate = read_json(candidate_root / "metrics.json")
            controls = read_json(candidate_root / "control_diagnostics.json")
            reference_path = reference["path_law"]
            candidate_path = candidate["path_law"]
            seed_groups: dict[str, float] = {}
            for group, names in GROUPS.items():
                values = []
                for name in names:
                    value = float(candidate_path[name])
                    gap = closure(float(reference_path[name]), value)
                    metric_values.setdefault(name, []).append(value)
                    if gap is not None:
                        gap_values.setdefault(name, []).append(gap)
                        values.append(gap)
                seed_groups[group] = float(median(values)) if values else math.nan
                if values:
                    group_values[group].append(seed_groups[group])
            reference_memory = memory_errors(reference)
            candidate_memory = memory_errors(candidate)
            memory_gaps = [
                gap
                for name in reference_memory
                if (gap := closure(reference_memory[name], candidate_memory[name]))
                is not None
            ]
            seed_groups["conditional_memory"] = (
                float(median(memory_gaps)) if memory_gaps else math.nan
            )
            if memory_gaps:
                group_values["conditional_memory"].append(
                    seed_groups["conditional_memory"]
                )
            scale = candidate["full_bank_scale_tail"]
            safety = {
                "finite": (
                    float(scale["nonfinite_path_fraction"]) == 0.0
                    and float(controls["nonfinite_path_fraction"]) == 0.0
                    and float(controls["nonfinite_control_fraction"]) == 0.0
                ),
                "sigma_cap_at_most_1pct": float(
                    controls["sigma_max_active_fraction"]
                ) <= 0.01,
                "return_std_ratio_0p8_1p2": 0.8
                <= float(scale["return_std_ratio"])
                <= 1.2,
                "rv_mean_ratio_0p8_1p2": 0.8
                <= float(scale["realized_volatility_mean_ratio"])
                <= 1.2,
                "rv_std_ratio_0p5_1p5": 0.5
                <= float(scale["realized_volatility_std_ratio"])
                <= 1.5,
                "absolute_return_q99_ratio_0p75_1p25": 0.75
                <= float(scale["absolute_return_q99_ratio"])
                <= 1.25,
                "absolute_return_q999_ratio_0p65_1p35": 0.65
                <= float(scale["absolute_return_q999_ratio"])
                <= 1.35,
                "no_gross_excess_kurtosis_inflation": float(
                    scale["generated_excess_kurtosis"]
                )
                <= max(
                    1.5 * float(scale["target_excess_kurtosis"]),
                    float(scale["target_excess_kurtosis"]) + 1.0,
                ),
            }
            mixture_fidelity: dict[str, object] | None = None
            if task == "mixture":
                reference_mixture = read_json(
                    reference_root / "mixture_metrics.json"
                )["mixture_fidelity"]
                candidate_mixture = read_json(
                    candidate_root / "mixture_metrics.json"
                )["mixture_fidelity"]
                mixture_errors = {
                    "regime_proportion_tvd": (
                        float(reference_mixture["regime_proportion_tvd"]),
                        float(candidate_mixture["regime_proportion_tvd"]),
                    ),
                    **{
                        f"{parameter}_support_normalized_wasserstein": (
                            float(
                                reference_mixture["parameters"][parameter][  # type: ignore[index]
                                    "support_normalized_wasserstein"
                                ]
                            ),
                            float(
                                candidate_mixture["parameters"][parameter][  # type: ignore[index]
                                    "support_normalized_wasserstein"
                                ]
                            ),
                        )
                        for parameter in ("rho", "theta", "xi")
                    },
                }
                gaps = {
                    name: closure(reference_value, candidate_value)
                    for name, (reference_value, candidate_value) in mixture_errors.items()
                }
                for name, gap in gaps.items():
                    if gap is not None:
                        mixture_gap_values.setdefault(name, []).append(gap)
                mixture_fidelity = {
                    "reference_candidate_errors": {
                        name: {
                            "reference": reference_value,
                            "candidate": candidate_value,
                            "gap_closed": gaps[name],
                        }
                        for name, (reference_value, candidate_value) in mixture_errors.items()
                    },
                    "median_gap_closed": float(
                        median(value for value in gaps.values() if value is not None)
                    ),
                }
            per_seed[str(seed)] = {
                "status": "complete",
                "group_gap_closed": seed_groups,
                "conditional_memory_errors": candidate_memory,
                "full_bank_scale_tail": scale,
                "controls": controls,
                "safety": {"pass": all(safety.values()), "checks": safety},
                "mixture_fidelity": mixture_fidelity,
            }
            complete += 1
        task_result = {
            "complete_count": complete,
            "failure_or_missing_count": len(SEEDS) - complete,
            "per_seed": per_seed,
            "metrics": {name: summary(values) for name, values in metric_values.items()},
            "gap_closed": {name: summary(values) for name, values in gap_values.items()},
            "group_gap_closed": {
                name: summary(values) for name, values in group_values.items() if values
            },
            "mixture_fidelity_gap_closed": {
                name: summary(values) for name, values in mixture_gap_values.items()
            },
        }
        result["tasks"][task] = task_result
    output = campaign / "SYNTHETIC_SUMMARY.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
