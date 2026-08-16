#!/usr/bin/env python3
"""Deploy exact GARCH-tree conditional adjoints without a neural estimator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.garch_tree_evaluation import (  # noqa: E402
    GARCH_DEPENDENCE_METRICS,
    GARCH_DISTRIBUTION_METRICS,
    GARCH_LOCAL_CALIBRATION_METRICS,
    gap_closed,
    garch_tree_metric_errors,
    summarize_gap_closure,
)
from path_dt_experiments.garch_tree_oracle import GarchTreeConfig  # noqa: E402
from path_dt_experiments.garch_tree_production import (  # noqa: E402
    ExactGarchTreeMPResponse,
    ProductionGarchTreeProblem,
    exact_garch_tree_mp_response,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-outer-steps", type=int, default=50)
    parser.add_argument("--max-backtracks", type=int, default=24)
    parser.add_argument("--fixed-point-tol", type=float, default=1e-6)
    parser.add_argument("--objective-atol", type=float, default=1e-12)
    parser.add_argument("--objective-rtol", type=float, default=1e-12)
    parser.add_argument("--swd-projections", type=int, default=512)
    return parser.parse_args()


def _objective(
    problem: ProductionGarchTreeProblem, variables: torch.Tensor
) -> float:
    return float(problem.objective(variables).detach().cpu().item())


def _components(
    problem: ProductionGarchTreeProblem, variables: torch.Tensor
) -> dict[str, float]:
    _, components = problem.objective_components(variables)
    return {
        name: float(value.detach().cpu().item())
        for name, value in components.items()
    }


def _gain_fraction(reference: float, exact: float, candidate: float) -> float:
    denominator = float(reference) - float(exact)
    if denominator <= 1e-14:
        return math.nan
    return (float(reference) - float(candidate)) / denominator


def _state_payload(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    *,
    reference_errors: dict[str, float],
    reference_objective: float,
    exact_objective: float,
    exact_variables: torch.Tensor,
    swd_projections: int,
) -> dict[str, object]:
    point = variables.detach().to(device=problem.device, dtype=problem.dtype)
    solution = problem.solution(point)
    errors = garch_tree_metric_errors(
        problem,
        solution,
        swd_projections=int(swd_projections),
    )
    closure = gap_closed(reference_errors, errors)
    objective = _components(problem, point)
    response = exact_garch_tree_mp_response(problem, point)
    sigma = torch.exp(point)
    return {
        "objective": objective,
        "objective_gain_fraction_of_exact": _gain_fraction(
            reference_objective,
            exact_objective,
            float(objective["total"]),
        ),
        "errors": errors,
        "gap_closed": closure,
        "summary": {
            "distribution": summarize_gap_closure(
                closure, GARCH_DISTRIBUTION_METRICS
            ),
            "dependence": summarize_gap_closure(
                closure, GARCH_DEPENDENCE_METRICS
            ),
            "local_calibration": summarize_gap_closure(
                closure, GARCH_LOCAL_CALIBRATION_METRICS
            ),
        },
        "stationarity": {
            "exact_ce_log_sigma_response_rms": response.metrics[
                "response_log_sigma_rms"
            ],
            "distance_to_exact_log_sigma_rms": float(
                (point - exact_variables).double().pow(2).mean().sqrt().item()
            ),
        },
        "safety": {
            "sigma_min": float(sigma.min().item()),
            "sigma_max": float(sigma.max().item()),
            "lower_bound_activity": float(
                (
                    sigma
                    <= float(problem.config.sigma_min) * (1.0 + 1e-6)
                )
                .double()
                .mean()
                .item()
            ),
            "upper_bound_activity": float(
                (
                    sigma
                    >= float(problem.config.sigma_max) * (1.0 - 1e-6)
                )
                .double()
                .mean()
                .item()
            ),
        },
    }


def _load_problem(
    args: argparse.Namespace,
) -> tuple[
    ProductionGarchTreeProblem,
    torch.Tensor,
    dict[str, object],
]:
    exact_dir = args.exact_run_dir.resolve()
    report = json.loads((exact_dir / "exact_report.json").read_text())
    protocol = report["protocol"]
    configuration = report["configuration"]
    config = GarchTreeConfig(
        horizon=int(configuration["horizon"]),
        dt=float(configuration["dt"]),
        long_run_sigma=float(configuration["long_run_sigma"]),
        arch=float(configuration["arch"]),
        garch=float(configuration["garch"]),
        sigma_min=float(configuration["sigma_bounds"][0]),
        sigma_max=float(configuration["sigma_bounds"][1]),
        eta=float(protocol["eta"]),
        lambda_scale=float(protocol["lambda_scale"]),
    )
    device = torch.device(args.device)
    problem = ProductionGarchTreeProblem(
        config,
        lambda_scale=float(protocol["lambda_scale"]),
        kappa_scale=float(protocol["kappa_scale"]),
        device=device,
        dtype=torch.float64,
    )
    exact = torch.load(
        exact_dir / "exact.pt", map_location=device, weights_only=True
    )["exact_variables"].to(device=device, dtype=torch.float64)
    return problem, exact, report


def run(args: argparse.Namespace) -> None:
    if int(args.max_outer_steps) < 1:
        raise ValueError("max_outer_steps must be positive")
    if int(args.max_backtracks) < 1:
        raise ValueError("max_backtracks must be positive")
    if float(args.fixed_point_tol) <= 0.0:
        raise ValueError("fixed_point_tol must be positive")

    run_dir = ensure_run_dir(args.run_dir.resolve())
    problem, exact, exact_report = _load_problem(args)
    reference = problem.reference_start()
    reference_objective = _objective(problem, reference)
    exact_objective = _objective(problem, exact)
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )

    current = reference.detach().clone()
    first_raw_response: ExactGarchTreeMPResponse | None = None
    first_accepted: torch.Tensor | None = None
    history: list[dict[str, float]] = []
    status = "maximum_outer_steps"
    started = time.perf_counter()

    for outer in range(int(args.max_outer_steps)):
        response = exact_garch_tree_mp_response(problem, current)
        if first_raw_response is None:
            first_raw_response = response
        current_objective = _objective(problem, current)
        raw_objective = _objective(problem, response.variables)
        residual = float(response.metrics["response_log_sigma_rms"])
        row = {
            "outer_step": float(outer),
            "objective_before": current_objective,
            "raw_full_response_objective": raw_objective,
            "fixed_point_residual_before": residual,
            "raw_response_sigma_min": float(
                response.metrics["response_sigma_min"]
            ),
            "raw_response_sigma_max": float(
                response.metrics["response_sigma_max"]
            ),
            "raw_response_lower_bound_activity": float(
                response.metrics["response_lower_bound_activity"]
            ),
            "raw_response_upper_bound_activity": float(
                response.metrics["response_upper_bound_activity"]
            ),
            "accepted": 0.0,
            "accepted_relaxation": 0.0,
            "objective_after": current_objective,
        }
        if residual <= float(args.fixed_point_tol):
            status = "fixed_point_tolerance"
            history.append(row)
            break

        required_decrease = max(
            float(args.objective_atol),
            float(args.objective_rtol) * max(abs(current_objective), 1.0),
        )
        accepted: torch.Tensor | None = None
        accepted_objective = current_objective
        accepted_relaxation = 0.0
        direction = response.variables - current
        for backtrack in range(int(args.max_backtracks)):
            relaxation = 2.0 ** (-backtrack)
            candidate = current + relaxation * direction
            candidate_objective = _objective(problem, candidate)
            if (
                math.isfinite(candidate_objective)
                and candidate_objective
                <= current_objective - required_decrease
            ):
                accepted = candidate.detach()
                accepted_objective = candidate_objective
                accepted_relaxation = relaxation
                break

        if accepted is None:
            status = "objective_gate_rejection"
            history.append(row)
            print(
                f"outer {outer}: rejected; J={current_objective:.9g}; "
                f"raw J={raw_objective:.9g}; residual={residual:.6g}",
                flush=True,
            )
            break

        if first_accepted is None:
            first_accepted = accepted.detach().clone()
        row.update(
            {
                "accepted": 1.0,
                "accepted_relaxation": float(accepted_relaxation),
                "objective_after": float(accepted_objective),
                "objective_improvement": float(
                    current_objective - accepted_objective
                ),
            }
        )
        history.append(row)
        current = accepted
        print(
            f"outer {outer}: J {current_objective:.9g} -> "
            f"{accepted_objective:.9g}; tau={accepted_relaxation:.6g}; "
            f"raw J={raw_objective:.9g}; residual={residual:.6g}",
            flush=True,
        )

    if first_raw_response is None:
        raise RuntimeError("the exact-CE loop performed no evaluation")
    if first_accepted is None:
        first_accepted = reference.detach().clone()

    elapsed = time.perf_counter() - started
    initial_payload = _state_payload(
        problem,
        reference,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    raw_payload = _state_payload(
        problem,
        first_raw_response.variables,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    first_payload = _state_payload(
        problem,
        first_accepted,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    final_payload = _state_payload(
        problem,
        current,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    exact_payload = _state_payload(
        problem,
        exact,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    report = {
        "protocol": {
            "role": "strict exact-conditional-expectation MP deployment oracle",
            "source_exact_run_dir": str(args.exact_run_dir.resolve()),
            "physical_drift": "identically_zero",
            "controlled_channel": "volatility_only",
            "lambda_scale": float(problem.lambda_scale),
            "kappa_scale": float(problem.kappa_scale),
            "eta": float(problem.config.eta),
            "conditional_expectations": (
                "complete descendant enumeration at every current-law node"
            ),
            "deployment": "direct production Hamiltonian control map",
            "neural_network": "none",
            "regression_loss": "none",
            "refresh": "exactly recomputed after every accepted law update",
            "relaxation": (
                "largest dyadic log-volatility step passing strict objective decrease"
            ),
            "comparison_models": "none",
        },
        "configuration": exact_report["configuration"],
        "termination": {
            "status": status,
            "accepted_updates": int(sum(row["accepted"] for row in history)),
            "evaluated_outer_steps": len(history),
            "wall_seconds": float(elapsed),
            "fixed_point_tolerance": float(args.fixed_point_tol),
            "max_backtracks": int(args.max_backtracks),
        },
        "reference": initial_payload,
        "one_raw_full_mp_response": raw_payload,
        "one_objective_gated_mp_response": first_payload,
        "repeated_objective_gated_mp": final_payload,
        "exact_nodewise_optimum": exact_payload,
        "first_response_audit": first_raw_response.metrics,
        "history": history,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "reference_variables": reference.detach().cpu(),
            "first_raw_response_variables": first_raw_response.variables.detach().cpu(),
            "first_accepted_variables": first_accepted.detach().cpu(),
            "final_variables": current.detach().cpu(),
            "exact_variables": exact.detach().cpu(),
        },
        run_dir / "solutions.pt",
    )

    first_summary = first_payload["summary"]
    final_summary = final_payload["summary"]
    (run_dir / "SUMMARY.md").write_text(
        "# Exact-CE direct-deployment GARCH-tree audit\n\n"
        "Every conditional adjoint is computed by complete descendant "
        "enumeration and injected directly into the production volatility-only "
        "Hamiltonian map. No GRU, regression, or Monte Carlo estimator is used.\n\n"
        f"- Reference objective: {reference_objective:.8g}\n"
        f"- Exact nodewise optimum: {exact_objective:.8g}\n"
        f"- Raw full first MP response: "
        f"{raw_payload['objective']['total']:.8g}\n"
        f"- Objective-gated first response: "
        f"{first_payload['objective']['total']:.8g}\n"
        f"- Final repeated response: {final_payload['objective']['total']:.8g}\n"
        f"- First response exact gain captured: "
        f"{first_payload['objective_gain_fraction_of_exact']:.4%}\n"
        f"- Final exact gain captured: "
        f"{final_payload['objective_gain_fraction_of_exact']:.4%}\n"
        f"- First response distribution metrics improved: "
        f"{first_summary['distribution']['improved_count']}/"
        f"{first_summary['distribution']['metric_count']}\n"
        f"- First response dependence metrics improved: "
        f"{first_summary['dependence']['improved_count']}/"
        f"{first_summary['dependence']['metric_count']}\n"
        f"- Final distribution metrics improved: "
        f"{final_summary['distribution']['improved_count']}/"
        f"{final_summary['distribution']['metric_count']}\n"
        f"- Final dependence metrics improved: "
        f"{final_summary['dependence']['improved_count']}/"
        f"{final_summary['dependence']['metric_count']}\n"
        f"- Termination: {status}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "accepted_updates": report["termination"]["accepted_updates"],
                "reference_objective": reference_objective,
                "raw_first_response_objective": raw_payload["objective"]["total"],
                "accepted_first_response_objective": first_payload["objective"]["total"],
                "final_objective": final_payload["objective"]["total"],
                "exact_objective": exact_objective,
                "first_gain_fraction": first_payload[
                    "objective_gain_fraction_of_exact"
                ],
                "final_gain_fraction": final_payload[
                    "objective_gain_fraction_of_exact"
                ],
                "final_stationarity": final_payload["stationarity"],
                "final_distribution": final_summary["distribution"],
                "final_dependence": final_summary["dependence"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
