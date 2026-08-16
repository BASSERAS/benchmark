#!/usr/bin/env python3
"""Trust-region Newton solve of the exact GARCH-tree MP stationarity system."""

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

from path_dt_experiments.exact_garch_stationarity import (  # noqa: E402
    ExactGarchStationarityEvaluation,
    exact_garch_mp_stationarity,
    finite_difference_stationarity_jacobian,
)
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
    ProductionGarchTreeProblem,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-solutions",
        type=Path,
        help=(
            "Optional checkpoint containing final_variables or current_variables. "
            "The reference remains unchanged and is used only for reporting."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-newton-steps", type=int, default=12)
    parser.add_argument("--max-trust-trials", type=int, default=12)
    parser.add_argument("--jacobian-epsilon", type=float, default=1e-4)
    parser.add_argument("--initial-radius", type=float, default=0.25)
    parser.add_argument("--maximum-radius", type=float, default=1.0)
    parser.add_argument("--minimum-radius", type=float, default=1e-7)
    parser.add_argument("--residual-tolerance", type=float, default=1e-3)
    parser.add_argument("--objective-atol", type=float, default=1e-12)
    parser.add_argument("--objective-rtol", type=float, default=1e-10)
    parser.add_argument("--objective-envelope-fraction", type=float, default=0.05)
    parser.add_argument("--swd-projections", type=int, default=512)
    return parser.parse_args()


def _load_problem(
    args: argparse.Namespace,
) -> tuple[ProductionGarchTreeProblem, torch.Tensor, dict[str, object]]:
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


def _objective(
    problem: ProductionGarchTreeProblem, variables: torch.Tensor
) -> float:
    return float(problem.objective(variables).detach().cpu().item())


def _lm_step(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    *,
    radius_rms: float,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Regularized least-squares Newton step inside an RMS trust region."""

    matrix = jacobian.double()
    right = residual.double()
    dimension = int(right.numel())
    radius_l2 = float(radius_rms) * math.sqrt(float(dimension))
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    projected = u.transpose(0, 1) @ right
    cutoff = singular.max().clamp_min(1e-30) * 1e-12

    def candidate(mu: float) -> torch.Tensor:
        if mu == 0.0:
            gain = torch.where(
                singular > cutoff,
                1.0 / singular.clamp_min(cutoff),
                torch.zeros_like(singular),
            )
        else:
            gain = singular / (singular.pow(2) + float(mu))
        return -(vh.transpose(0, 1) @ (gain * projected))

    step = candidate(0.0)
    mu = 0.0
    if float(torch.linalg.vector_norm(step).item()) > radius_l2:
        low = 0.0
        high = max(float(singular.max().item()) ** 2 * 1e-12, 1e-18)
        while float(torch.linalg.vector_norm(candidate(high)).item()) > radius_l2:
            high *= 10.0
            if high > 1e30:
                raise RuntimeError("failed to bracket the LM trust-region shift")
        for _ in range(80):
            middle = 0.5 * (low + high)
            if float(torch.linalg.vector_norm(candidate(middle)).item()) > radius_l2:
                low = middle
            else:
                high = middle
        mu = high
        step = candidate(mu)
    linear_residual = right + matrix @ step
    predicted_reduction = 0.5 * float(
        (right.pow(2).sum() - linear_residual.pow(2).sum()).item()
    )
    return step.to(jacobian), predicted_reduction, {
        "lm_shift": float(mu),
        "step_rms": float(step.pow(2).mean().sqrt().item()),
        "linearized_residual_rms": float(
            linear_residual.pow(2).mean().sqrt().item()
        ),
        "maximum_singular_value": float(singular.max().item()),
        "minimum_singular_value": float(singular.min().item()),
    }


def _metric_payload(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    *,
    evaluation: ExactGarchStationarityEvaluation,
    reference_errors: dict[str, float],
    reference_objective: float,
    exact_objective: float,
    exact_variables: torch.Tensor,
    swd_projections: int,
) -> dict[str, object]:
    point = variables.detach().to(problem.device, problem.dtype)
    objective = _objective(problem, point)
    errors = garch_tree_metric_errors(
        problem,
        problem.solution(point),
        swd_projections=int(swd_projections),
    )
    closure = gap_closed(reference_errors, errors)
    sigma = torch.exp(point)
    denominator = reference_objective - exact_objective
    return {
        "objective": objective,
        "objective_gain_fraction_of_exact": (
            (reference_objective - objective) / denominator
            if denominator > 1e-14
            else math.nan
        ),
        "stationarity": evaluation.metrics,
        "distance_to_exact_log_sigma_rms": float(
            (point - exact_variables).double().pow(2).mean().sqrt().item()
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


def run(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.objective_envelope_fraction) < 1.0:
        raise ValueError("objective_envelope_fraction must lie in [0, 1)")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    problem, exact, exact_report = _load_problem(args)
    reference = problem.reference_start()
    current = reference.detach().clone()
    if args.initial_solutions is not None:
        initial_path = args.initial_solutions.resolve()
        payload = torch.load(
            initial_path,
            map_location=problem.device,
            weights_only=True,
        )
        if "final_variables" in payload:
            current = payload["final_variables"]
        elif "current_variables" in payload:
            current = payload["current_variables"]
        else:
            raise KeyError(
                f"{initial_path} contains neither final_variables nor current_variables"
            )
        current = current.to(device=problem.device, dtype=problem.dtype).detach()
        if current.shape != reference.shape:
            raise ValueError("initial variables have the wrong shape")
    initial = current.detach().clone()
    reference_objective = _objective(problem, reference)
    current_objective = _objective(problem, current)
    exact_objective = _objective(problem, exact)
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )
    lower = math.log(float(problem.config.sigma_min))
    upper = math.log(float(problem.config.sigma_max))
    radius = float(args.initial_radius)
    history: list[dict[str, object]] = []
    operator_evaluations = 0
    status = "maximum_newton_steps"
    started = time.perf_counter()

    reference_evaluation = exact_garch_mp_stationarity(
        problem,
        reference,
        gradient_audit=True,
    )
    initial_evaluation = exact_garch_mp_stationarity(
        problem,
        current,
        gradient_audit=True,
    )
    operator_evaluations += 2
    current_evaluation = initial_evaluation
    best_objective = current_objective

    for iteration in range(int(args.max_newton_steps)):
        residual = current_evaluation.projected_residual
        residual_rms = float(residual.double().pow(2).mean().sqrt().item())
        if residual_rms <= float(args.residual_tolerance):
            status = "residual_tolerance"
            break
        print(
            f"Newton {iteration}: building {int(residual.numel())}x"
            f"{int(residual.numel())} exact MP Jacobian; "
            f"J={current_objective:.9g}; residual={residual_rms:.6g}; "
            f"radius={radius:.6g}",
            flush=True,
        )

        def progress(done: int, total: int) -> None:
            if done == total or done % 32 == 0:
                print(
                    f"  Jacobian columns {done}/{total}",
                    flush=True,
                )

        jacobian = finite_difference_stationarity_jacobian(
            problem,
            current,
            epsilon=float(args.jacobian_epsilon),
            progress=progress,
        )
        operator_evaluations += 1 + 2 * int(residual.numel())
        base_disagreement = float(
            torch.linalg.vector_norm(
                jacobian.base_residual.double() - residual.double()
            ).item()
        )
        if base_disagreement > 1e-9:
            raise RuntimeError("Jacobian base residual changed unexpectedly")

        accepted = False
        trial_rows: list[dict[str, object]] = []
        for trial in range(int(args.max_trust_trials)):
            newton_step, _, newton_metrics = _lm_step(
                jacobian.value,
                residual,
                radius_rms=radius,
            )
            projected_gradient = torch.where(
                residual == 0.0,
                torch.zeros_like(
                    current_evaluation.weighted_objective_gradient
                ),
                current_evaluation.weighted_objective_gradient,
            )
            gradient_rms = float(
                projected_gradient.double().pow(2).mean().sqrt().item()
            )
            proposals = [("newton", newton_step, newton_metrics)]
            if gradient_rms > 0.0:
                cauchy_step = -projected_gradient * (radius / gradient_rms)
                proposals.append(
                    (
                        "cauchy",
                        cauchy_step,
                        {
                            "lm_shift": 0.0,
                            "step_rms": float(radius),
                            "linearized_residual_rms": 0.0,
                            "maximum_singular_value": float(
                                newton_metrics["maximum_singular_value"]
                            ),
                            "minimum_singular_value": float(
                                newton_metrics["minimum_singular_value"]
                            ),
                        },
                    )
                )

            for direction_name, step, step_metrics in proposals:
                candidate = (current + step).clamp(
                    min=lower, max=upper
                ).detach()
                actual_step = candidate - current
                linear_residual = residual + jacobian.value @ actual_step
                step_metrics = dict(step_metrics)
                step_metrics["linearized_residual_rms"] = float(
                    linear_residual.double().pow(2).mean().sqrt().item()
                )
                predicted_reduction = 0.5 * float(
                    (
                        residual.double().pow(2).sum()
                        - linear_residual.double().pow(2).sum()
                    ).item()
                )
                candidate_evaluation = exact_garch_mp_stationarity(
                    problem,
                    candidate,
                    gradient_audit=False,
                )
                operator_evaluations += 1
                candidate_residual = candidate_evaluation.projected_residual
                actual_reduction = 0.5 * float(
                    (
                        residual.double().pow(2).sum()
                        - candidate_residual.double().pow(2).sum()
                    ).item()
                )
                ratio = (
                    actual_reduction / predicted_reduction
                    if predicted_reduction > 0.0
                    else -math.inf
                )
                candidate_objective = _objective(problem, candidate)
                objective_tolerance = max(
                    float(args.objective_atol),
                    float(args.objective_rtol)
                    * max(abs(current_objective), 1.0),
                )
                objective_ceiling = best_objective + float(
                    args.objective_envelope_fraction
                ) * max(reference_objective - best_objective, 0.0)
                residual_improved = actual_reduction > 0.0
                objective_safe = (
                    candidate_objective
                    <= objective_ceiling + objective_tolerance
                )
                trial_row = {
                    "trial": float(trial),
                    "direction": direction_name,
                    "radius": float(radius),
                    "candidate_objective": float(candidate_objective),
                    "objective_ceiling": float(objective_ceiling),
                    "candidate_residual_rms": float(
                        candidate_residual.double()
                        .pow(2)
                        .mean()
                        .sqrt()
                        .item()
                    ),
                    "predicted_merit_reduction": float(predicted_reduction),
                    "actual_merit_reduction": float(actual_reduction),
                    "trust_ratio": float(ratio),
                    "objective_safe": float(objective_safe),
                    "residual_improved": float(residual_improved),
                    "actual_step_rms": float(
                        actual_step.double().pow(2).mean().sqrt().item()
                    ),
                    **step_metrics,
                }
                trial_rows.append(trial_row)
                print(
                    f"  trial {trial} {direction_name}: "
                    f"radius={radius:.6g}; J={candidate_objective:.9g}; "
                    f"residual={trial_row['candidate_residual_rms']:.6g}; "
                    f"ratio={ratio:.4g}; objective_safe={objective_safe}",
                    flush=True,
                )
                ratio_safe = ratio > 1e-4 or direction_name == "cauchy"
                if residual_improved and objective_safe and ratio_safe:
                    accepted = True
                    current = candidate
                    current_evaluation = candidate_evaluation
                    current_objective = candidate_objective
                    best_objective = min(best_objective, candidate_objective)
                    if (
                        direction_name == "newton"
                        and ratio > 0.75
                        and float(
                            actual_step.double()
                            .pow(2)
                            .mean()
                            .sqrt()
                            .item()
                        )
                        >= 0.8 * radius
                    ):
                        radius = min(
                            2.0 * radius, float(args.maximum_radius)
                        )
                    elif direction_name == "newton" and ratio < 0.25:
                        radius = max(
                            0.5 * radius, float(args.minimum_radius)
                        )
                    break
            if accepted:
                break
            radius = max(0.25 * radius, float(args.minimum_radius))
            if radius <= float(args.minimum_radius):
                break

        history.append(
            {
                "iteration": int(iteration),
                "accepted": bool(accepted),
                "objective_after": float(current_objective),
                "residual_rms_after": float(
                    current_evaluation.projected_residual.double()
                    .pow(2)
                    .mean()
                    .sqrt()
                    .item()
                ),
                "radius_after": float(radius),
                "jacobian": jacobian.metrics,
                "trials": trial_rows,
            }
        )
        torch.save(
            {
                "current_variables": current.detach().cpu(),
                "history": history,
            },
            run_dir / "progress.pt",
        )
        write_json(
            run_dir / "progress.json",
            {
                "status": "running" if accepted else "trust_region_rejection",
                "operator_evaluations": int(operator_evaluations),
                "history": history,
            },
        )
        if not accepted:
            status = "trust_region_rejection"
            break

    elapsed = time.perf_counter() - started
    final_evaluation = exact_garch_mp_stationarity(
        problem,
        current,
        gradient_audit=True,
    )
    exact_evaluation = exact_garch_mp_stationarity(
        problem,
        exact,
        gradient_audit=True,
    )
    operator_evaluations += 2
    reference_payload = _metric_payload(
        problem,
        reference,
        evaluation=reference_evaluation,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    final_payload = _metric_payload(
        problem,
        current,
        evaluation=final_evaluation,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    exact_payload = _metric_payload(
        problem,
        exact,
        evaluation=exact_evaluation,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    initial_payload = _metric_payload(
        problem,
        initial,
        evaluation=initial_evaluation,
        reference_errors=reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        exact_variables=exact,
        swd_projections=int(args.swd_projections),
    )
    report = {
        "protocol": {
            "role": "exact conditional-adjoint MP stationarity Newton ceiling",
            "source_exact_run_dir": str(args.exact_run_dir.resolve()),
            "physical_drift": "identically_zero",
            "controlled_channel": "volatility_only",
            "lambda_scale": float(problem.lambda_scale),
            "kappa_scale": float(problem.kappa_scale),
            "eta": float(problem.config.eta),
            "residual": "projected pointwise volatility Hamiltonian KKT residual",
            "conditional_expectations": "complete descendant enumeration",
            "jacobian": "central finite differences of refreshed exact MP residual",
            "step": "SVD Levenberg-Marquardt trust-region Newton",
            "objective_role": "acceptance safety gate only",
            "objective_filter": (
                "residual must decrease; objective remains inside a locked "
                "best-to-reference envelope"
            ),
            "objective_envelope_fraction": float(
                args.objective_envelope_fraction
            ),
            "exact_optimum_role": "evaluation only; never read by the solver update",
            "initial_solutions": (
                None
                if args.initial_solutions is None
                else str(args.initial_solutions.resolve())
            ),
        },
        "configuration": exact_report["configuration"],
        "termination": {
            "status": status,
            "accepted_steps": int(sum(bool(row["accepted"]) for row in history)),
            "operator_evaluations": int(operator_evaluations),
            "wall_seconds": float(elapsed),
            "residual_tolerance": float(args.residual_tolerance),
        },
        "reference": reference_payload,
        "initial": initial_payload,
        "newton": final_payload,
        "exact_nodewise_optimum": exact_payload,
        "history": history,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "reference_variables": reference.detach().cpu(),
            "newton_variables": current.detach().cpu(),
            "exact_variables": exact.detach().cpu(),
        },
        run_dir / "solutions.pt",
    )
    final_summary = final_payload["summary"]
    (run_dir / "SUMMARY.md").write_text(
        "# Exact GARCH-tree MP stationarity Newton audit\n\n"
        "The solver uses exact conditional adjoints and the volatility "
        "Hamiltonian stationarity residual. The objective is used only as an "
        "acceptance safety gate, and the exact optimum is used only for final "
        "evaluation.\n\n"
        f"- Termination: {status}\n"
        f"- Accepted Newton steps: {report['termination']['accepted_steps']}\n"
        f"- Exact MP operator evaluations: {operator_evaluations}\n"
        f"- Reference objective: {reference_objective:.9g}\n"
        f"- Newton objective: {final_payload['objective']:.9g}\n"
        f"- Exact objective: {exact_objective:.9g}\n"
        f"- Exact gain captured: "
        f"{final_payload['objective_gain_fraction_of_exact']:.6%}\n"
        f"- Newton stationarity residual: "
        f"{final_evaluation.metrics['projected_residual_rms']:.6g}\n"
        f"- Exact-optimum stationarity residual: "
        f"{exact_evaluation.metrics['projected_residual_rms']:.6g}\n"
        f"- Distribution metrics improved: "
        f"{final_summary['distribution']['improved_count']}/"
        f"{final_summary['distribution']['metric_count']}\n"
        f"- Dependence metrics improved: "
        f"{final_summary['dependence']['improved_count']}/"
        f"{final_summary['dependence']['metric_count']}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "accepted_steps": report["termination"]["accepted_steps"],
                "operator_evaluations": operator_evaluations,
                "reference_objective": reference_objective,
                "newton_objective": final_payload["objective"],
                "exact_objective": exact_objective,
                "gain_fraction": final_payload[
                    "objective_gain_fraction_of_exact"
                ],
                "newton_residual": final_evaluation.metrics[
                    "projected_residual_rms"
                ],
                "exact_residual": exact_evaluation.metrics[
                    "projected_residual_rms"
                ],
                "distance_to_exact": final_payload[
                    "distance_to_exact_log_sigma_rms"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
