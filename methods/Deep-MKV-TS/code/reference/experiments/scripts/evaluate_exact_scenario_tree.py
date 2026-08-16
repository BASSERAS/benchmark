#!/usr/bin/env python3
"""Evaluate exact and one-stage MP laws against the exact local reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.exact_scenario_tree import (  # noqa: E402
    ExactBinaryTreeProblem,
    ExactTreeConfig,
)
from path_dt_experiments.exact_tree_evaluation import (  # noqa: E402
    DEPENDENCE_METRICS,
    LOCAL_CALIBRATION_METRICS,
    PRIMARY_DISTRIBUTION_METRICS,
    exact_tree_metric_errors,
    fraction_of_exact_improvement_captured,
    gap_closed,
    summarize_gap_closure,
)
from path_dt_experiments.runners import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--swd-projections", type=int, default=1024)
    parser.add_argument("--swd-seed", type=int, default=20260804)
    return parser.parse_args()


def _problem(run_dir: Path, mode: str, device: torch.device) -> tuple[ExactBinaryTreeProblem, torch.Tensor]:
    report = json.loads((run_dir / mode / "report.json").read_text())
    configuration = report["configuration"]
    protocol = report["protocol"]
    config = ExactTreeConfig(
        horizon=int(protocol["horizon"]),
        dt=float(configuration["dt"]),
        sigma_low=float(configuration["sigma_low"]),
        sigma_high=float(configuration["sigma_high"]),
        sigma_min=float(configuration["sigma_bounds"][0]),
        sigma_max=float(configuration["sigma_bounds"][1]),
        drawdown_threshold=float(configuration["drawdown_threshold"]),
        history_cutoff=int(configuration["history_cutoff"]),
        response_delay=int(configuration["response_delay"]),
        eta=float(configuration["eta"]),
        lambda_scale=float(configuration["lambda_scale"]),
        bandwidths=tuple(float(value) for value in configuration["bandwidths"]),
    )
    problem = ExactBinaryTreeProblem(
        config,
        mode=mode,
        device=device,
        dtype=torch.float64,
    )
    saved = torch.load(run_dir / mode / "solution.pt", map_location=device, weights_only=True)
    variables = saved["variables"].to(device=device, dtype=torch.float64)
    return problem, variables


def _one_stage(problem: ExactBinaryTreeProblem) -> tuple[torch.Tensor, dict[str, float]]:
    reference = problem.reference_start()
    proposal = problem.hamiltonian_best_response(reference)
    direction = proposal - reference

    differentiable_reference = reference.detach().clone().requires_grad_(True)
    reference_value = problem.objective(differentiable_reference)
    reference_gradient = torch.autograd.grad(
        reference_value, differentiable_reference
    )[0]
    directional_derivative = float(
        torch.sum(reference_gradient * direction).detach().cpu().item()
    )

    def objective(fraction: float) -> float:
        variables = reference + float(fraction) * direction
        return float(problem.objective(variables).detach().cpu().item())

    fractions = sorted({0.0, *(2.0 ** (-power) for power in range(0, 41))})
    grid_values = [objective(fraction) for fraction in fractions]
    best_index = int(np.argmin(grid_values))
    candidates = [(float(grid_values[best_index]), float(fractions[best_index]))]
    optimizer_success = True
    if 0 < best_index < len(fractions) - 1:
        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(fractions[best_index - 1], fractions[best_index + 1]),
            options={"xatol": 1e-14, "maxiter": 500},
        )
        optimizer_success = bool(result.success)
        candidates.append((float(result.fun), float(result.x)))
    candidates.extend(
        (objective(endpoint), endpoint) for endpoint in (0.0, 1.0)
    )
    accepted_value, fraction = min(candidates, key=lambda item: (item[0], item[1]))
    variables = reference + fraction * direction
    return variables, {
        "accepted_fraction": fraction,
        "reference_objective": objective(0.0),
        "full_proposal_objective": objective(1.0),
        "accepted_objective": accepted_value,
        "directional_derivative_at_reference": directional_derivative,
        "logarithmic_grid_minimum_fraction": float(fractions[best_index]),
        "logarithmic_grid_minimum_objective": float(grid_values[best_index]),
        "optimizer_success": optimizer_success,
    }


def _running_cost(problem: ExactBinaryTreeProblem, variables: torch.Tensor) -> dict[str, float]:
    _, components = problem.objective_components(variables)
    return {
        key: float(value.detach().cpu().item())
        for key, value in components.items()
    }


def _candidate_payload(
    problem: ExactBinaryTreeProblem,
    variables: torch.Tensor,
    reference_errors: dict[str, float],
    *,
    swd_projections: int,
    swd_seed: int,
) -> dict[str, object]:
    errors = exact_tree_metric_errors(
        problem,
        problem.solution(variables),
        swd_projections=swd_projections,
        swd_seed=swd_seed,
    )
    closure = gap_closed(reference_errors, errors)
    return {
        "errors": errors,
        "gap_closed": closure,
        "summary": {
            "distribution": summarize_gap_closure(
                closure, PRIMARY_DISTRIBUTION_METRICS
            ),
            "dependence": summarize_gap_closure(closure, DEPENDENCE_METRICS),
            "local_calibration": summarize_gap_closure(
                closure, LOCAL_CALIBRATION_METRICS
            ),
        },
        "objective": _running_cost(problem, variables),
    }


def _format_percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}%"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    report: dict[str, object] = {
        "protocol": {
            "reference_comparison": "fraction of exact reference error removed",
            "one_stage_definition": (
                "one exact adjoint evaluation at the reference, bounded Hamiltonian "
                "best-response direction, and exact-population scalar objective gate on [0,1]"
            ),
            "swd_projections": int(args.swd_projections),
            "swd_seed": int(args.swd_seed),
            "lower_is_better_for_every_error": True,
        },
        "modes": {},
    }
    markdown_rows = []
    for mode in ("full", "volatility_only"):
        problem, exact_variables = _problem(run_dir, mode, device)
        reference_variables = problem.reference_start()
        reference_errors = exact_tree_metric_errors(
            problem,
            problem.solution(reference_variables),
            swd_projections=int(args.swd_projections),
            swd_seed=int(args.swd_seed),
        )
        one_stage_variables, stage = _one_stage(problem)
        exact = _candidate_payload(
            problem,
            exact_variables,
            reference_errors,
            swd_projections=int(args.swd_projections),
            swd_seed=int(args.swd_seed),
        )
        one_stage = _candidate_payload(
            problem,
            one_stage_variables,
            reference_errors,
            swd_projections=int(args.swd_projections),
            swd_seed=int(args.swd_seed),
        )
        report["modes"][mode] = {
            "reference": {
                "errors": reference_errors,
                "objective": _running_cost(problem, reference_variables),
            },
            "one_stage": one_stage | {"stage": stage},
            "exact_mp": exact,
            "one_stage_fraction_of_exact_improvement": {
                "distribution": fraction_of_exact_improvement_captured(
                    one_stage["gap_closed"],
                    exact["gap_closed"],
                    PRIMARY_DISTRIBUTION_METRICS,
                ),
                "dependence": fraction_of_exact_improvement_captured(
                    one_stage["gap_closed"],
                    exact["gap_closed"],
                    DEPENDENCE_METRICS,
                ),
            },
        }
        label = "Drift + volatility" if mode == "full" else "Volatility only"
        for candidate_name, candidate in (("One-stage", one_stage), ("Exact MP", exact)):
            summary = candidate["summary"]
            markdown_rows.append(
                (
                    f"{label}: {candidate_name}",
                    summary["distribution"],
                    summary["dependence"],
                    stage["accepted_fraction"] if candidate_name == "One-stage" else None,
                )
            )

    write_json(run_dir / "path_law_panel.json", report)
    lines = [
        "# Exact N=8 path-law correction panel",
        "",
        "Every percentage is the fraction of the local reference error removed; negative values mean deterioration.",
        "",
        "| Candidate | Distribution metrics improved | Median distribution gap closed | Dependence metrics improved | Median dependence gap closed | One-stage fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, distribution, dependence, fraction in markdown_rows:
        lines.append(
            f"| {label} | {distribution['improved_count']}/{distribution['metric_count']} | "
            f"{_format_percent(distribution['median_gap_closed'])} | "
            f"{dependence['improved_count']}/{dependence['metric_count']} | "
            f"{_format_percent(dependence['median_gap_closed'])} | "
            f"{'--' if fraction is None else f'{fraction:.6g}'} |"
        )
    lines.extend(
        [
            "",
            "## Per-metric gap closure",
            "",
            "| Metric | Full one-stage | Full exact | Vol-only one-stage | Vol-only exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    modes = report["modes"]
    metric_names = list(PRIMARY_DISTRIBUTION_METRICS + DEPENDENCE_METRICS + LOCAL_CALIBRATION_METRICS)
    for metric in metric_names:
        values = [
            modes["full"]["one_stage"]["gap_closed"][metric],
            modes["full"]["exact_mp"]["gap_closed"][metric],
            modes["volatility_only"]["one_stage"]["gap_closed"][metric],
            modes["volatility_only"]["exact_mp"]["gap_closed"][metric],
        ]
        lines.append(
            f"| `{metric}` | " + " | ".join(_format_percent(value) for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## One-stage share of exact improvement",
            "",
            "The ratio is computed only where exact MP positively closes the reference gap.",
            "",
            "| Control problem | Distribution median | Dependence median |",
            "|---|---:|---:|",
        ]
    )
    for mode, label in (
        ("full", "Drift + volatility"),
        ("volatility_only", "Volatility only"),
    ):
        captured = modes[mode]["one_stage_fraction_of_exact_improvement"]
        lines.append(
            f"| {label} | "
            f"{_format_percent(captured['distribution']['median_fraction_captured'])} | "
            f"{_format_percent(captured['dependence']['median_fraction_captured'])} |"
        )
    lines.extend(
        [
            "",
            "## Objective and local-calibration preservation",
            "",
            "| Candidate | Total objective | Running cost | Timewise return-mean error | Timewise return-std error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, label in (
        ("full", "Full"),
        ("volatility_only", "Vol-only"),
    ):
        for candidate_key, candidate_label in (
            ("reference", "Reference"),
            ("one_stage", "One-stage"),
            ("exact_mp", "Exact MP"),
        ):
            candidate = modes[mode][candidate_key]
            objective = candidate["objective"]
            errors = candidate["errors"]
            running = float(objective["drift_cost"]) + float(objective["volatility_cost"])
            lines.append(
                f"| {label}: {candidate_label} | {objective['total']:.6g} | "
                f"{running:.6g} | {errors['timewise_return_mean_rmse_normalized']:.6g} | "
                f"{errors['timewise_return_std_rmse_normalized']:.6g} |"
            )
    lines.extend(
        [
            "",
            "The target has zero error by construction. Gap closure is undefined for return ACF and the two local-calibration metrics because the local reference already matches them to numerical precision; their absolute candidate errors are reported instead.",
            "",
            "A scalable nested-MP row is intentionally absent from this exact panel: evaluating a different horizon or sampled target would not be a matched comparison. It requires a separate `N=8` approximation experiment against this same enumerated tree.",
        ]
    )
    (run_dir / "PATH_LAW_PANEL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(run_dir / "PATH_LAW_PANEL.md")}, indent=2))


if __name__ == "__main__":
    main()
