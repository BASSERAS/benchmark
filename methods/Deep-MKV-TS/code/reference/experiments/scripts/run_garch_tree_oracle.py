#!/usr/bin/env python3
"""Run the exact and learned stages of the finite GARCH-tree oracle."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar
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
from path_dt_experiments.garch_tree_online import (  # noqa: E402
    CausalGarchAdjointPolicy,
    fit_direct_garch_objective,
    fit_online_garch_mp,
    policy_from_state_dict,
    rollout_garch_policy,
)
from path_dt_experiments.garch_tree_oracle import (  # noqa: E402
    ExactGarchTreeProblem,
    GarchTreeConfig,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("exact", "seed", "summarize"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="*", default=(0, 1, 3, 4))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--long-run-sigma", type=float, default=0.25)
    parser.add_argument("--arch", type=float, default=0.35)
    parser.add_argument("--garch", type=float, default=0.60)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.70)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--swd-projections", type=int, default=512)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> GarchTreeConfig:
    return GarchTreeConfig(
        horizon=int(args.horizon),
        dt=1.0 / float(args.horizon),
        long_run_sigma=float(args.long_run_sigma),
        arch=float(args.arch),
        garch=float(args.garch),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        eta=float(args.eta),
        lambda_scale=float(args.lambda_scale),
        bandwidths=(0.5, 1.0, 2.0),
    )


def _config_from_report(report: dict[str, object]) -> GarchTreeConfig:
    config = report["configuration"]
    return GarchTreeConfig(
        horizon=int(config["horizon"]),
        dt=float(config["dt"]),
        long_run_sigma=float(config["long_run_sigma"]),
        arch=float(config["arch"]),
        garch=float(config["garch"]),
        sigma_min=float(config["sigma_bounds"][0]),
        sigma_max=float(config["sigma_bounds"][1]),
        eta=float(config["eta"]),
        lambda_scale=float(config["lambda_scale"]),
        bandwidths=tuple(float(value) for value in config["bandwidths"]),
        innovations=tuple(float(value) for value in config["innovations"]),
    )


def _projected_gradient(
    variables: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    projected = np.asarray(gradient, dtype=np.float64).copy()
    for index, (value, derivative, (lower, upper)) in enumerate(
        zip(variables, gradient, bounds)
    ):
        if value <= float(lower) + tolerance and derivative > 0.0:
            projected[index] = 0.0
        if value >= float(upper) - tolerance and derivative < 0.0:
            projected[index] = 0.0
    return projected


def _components(
    problem: ExactGarchTreeProblem, variables: torch.Tensor
) -> dict[str, float]:
    _, values = problem.objective_components(variables)
    return {
        key: float(value.detach().cpu().item()) for key, value in values.items()
    }


def _ideal_one_stage(
    problem: ExactGarchTreeProblem,
) -> tuple[torch.Tensor, dict[str, float]]:
    reference = problem.reference_start()
    proposal = problem.hamiltonian_best_response(reference)
    direction = proposal - reference
    differentiable = reference.detach().clone().requires_grad_(True)
    reference_value = problem.objective(differentiable)
    gradient = torch.autograd.grad(reference_value, differentiable)[0]
    directional_derivative = float(
        torch.sum(gradient * direction).detach().cpu().item()
    )

    def objective(fraction: float) -> float:
        point = reference + float(fraction) * direction
        return float(problem.objective(point).detach().cpu().item())

    fractions = sorted({0.0, *(2.0 ** (-power) for power in range(0, 31))})
    values = [objective(fraction) for fraction in fractions]
    best_index = int(np.argmin(values))
    candidates = [(float(values[best_index]), float(fractions[best_index]))]
    if 0 < best_index < len(fractions) - 1:
        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(fractions[best_index - 1], fractions[best_index + 1]),
            options={"xatol": 1e-13, "maxiter": 300},
        )
        candidates.append((float(result.fun), float(result.x)))
    candidates.extend((objective(value), value) for value in (0.0, 1.0))
    accepted, fraction = min(candidates, key=lambda item: (item[0], item[1]))
    return reference + float(fraction) * direction, {
        "accepted_fraction": float(fraction),
        "reference_objective": objective(0.0),
        "full_proposal_objective": objective(1.0),
        "accepted_objective": float(accepted),
        "directional_derivative_at_reference": directional_derivative,
    }


def _metric_payload(
    problem: ExactGarchTreeProblem,
    variables: torch.Tensor,
    reference_errors: dict[str, float],
    *,
    swd_projections: int,
) -> dict[str, object]:
    errors = garch_tree_metric_errors(
        problem,
        problem.solution(variables),
        swd_projections=int(swd_projections),
    )
    closure = gap_closed(reference_errors, errors)
    return {
        "objective": _components(problem, variables),
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
    }


def _objective_gain_fraction(
    reference: float, exact: float, candidate: float
) -> float | None:
    attainable = float(reference) - float(exact)
    if not math.isfinite(attainable) or attainable <= 1e-14:
        return None
    return (float(reference) - float(candidate)) / attainable


def run_exact(args: argparse.Namespace) -> None:
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    config = _config_from_args(args)
    problem = ExactGarchTreeProblem(
        config, device=device, dtype=torch.float64
    )
    bounds = problem.bounds()
    generator = torch.Generator(device="cpu").manual_seed(20260805)
    reference = problem.reference_start()
    target = problem.target_start()
    starts = {
        "reference": reference,
        "target": target,
        "midpoint": 0.5 * (reference + target),
    }
    for index in range(2):
        perturbation = 0.05 * torch.randn(
            problem.variable_count,
            generator=generator,
            dtype=torch.float64,
        ).to(device)
        starts[f"perturbed_{index}"] = torch.maximum(
            torch.minimum(
                target + perturbation,
                torch.full_like(target, bounds[0][1]),
            ),
            torch.full_like(target, bounds[0][0]),
        )

    evaluations = 0

    def objective_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        point = torch.tensor(
            values, device=device, dtype=torch.float64, requires_grad=True
        )
        objective = problem.objective(point)
        gradient = torch.autograd.grad(objective, point)[0]
        return (
            float(objective.detach().cpu().item()),
            gradient.detach().cpu().double().numpy(),
        )

    results = []
    for name, start in starts.items():
        start_np = start.detach().cpu().double().numpy()
        initial, _ = objective_and_gradient(start_np)
        print(f"exact GARCH tree: optimizing from {name}", flush=True)
        result = minimize(
            objective_and_gradient,
            start_np,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(args.max_iterations),
                "gtol": 1e-9,
                "ftol": 1e-14,
                "maxls": 80,
                "maxcor": 30,
            },
        )
        projected = _projected_gradient(result.x, result.jac, bounds)
        row = {
            "name": name,
            "initial_objective": float(initial),
            "final_objective": float(result.fun),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "projected_gradient_rms": float(np.sqrt(np.mean(projected**2))),
            "projected_gradient_max": float(np.max(np.abs(projected))),
            "variables": result.x,
        }
        results.append(row)
        print(
            f"  J={result.fun:.9g}, projected max={row['projected_gradient_max']:.3g}",
            flush=True,
        )

    selected = min(
        results,
        key=lambda row: (
            float(row["final_objective"]),
            float(row["projected_gradient_max"]),
            str(row["name"]),
        ),
    )
    exact_variables = torch.tensor(
        selected["variables"], device=device, dtype=torch.float64
    )
    audit = problem.exact_adjoint_audit(exact_variables)
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )
    exact_payload = _metric_payload(
        problem,
        exact_variables,
        reference_errors,
        swd_projections=int(args.swd_projections),
    )
    one_stage_variables, stage = _ideal_one_stage(problem)
    one_stage_payload = _metric_payload(
        problem,
        one_stage_variables,
        reference_errors,
        swd_projections=int(args.swd_projections),
    )
    reference_objective = float(problem.objective(reference).item())
    exact_objective = float(problem.objective(exact_variables).item())
    one_stage_objective = float(problem.objective(one_stage_variables).item())
    report = {
        "protocol": {
            "purpose": "exact-population solver oracle, not a generative benchmark",
            "complete_tree": True,
            "equal_branch_probabilities": True,
            "conditional_expectations": "exact nodewise averages",
            "neural_network_used_for_exact_solution": False,
            "sampling_used_for_exact_solution": False,
            "control": "volatility_only",
            "physical_drift": "zero",
            "reference": "constant variance matched to the target unconditional variance",
            "optimizer": "multi-start reduced-space L-BFGS-B with exact autograd gradients",
            "independent_mp_gradient_verification": True,
        },
        "configuration": {
            "horizon": int(config.horizon),
            "dt": float(config.dt),
            "branching": int(config.branching),
            "leaf_count": int(config.leaf_count),
            "decision_nodes": int(config.decision_nodes),
            "long_run_sigma": float(config.long_run_sigma),
            "arch": float(config.arch),
            "garch": float(config.garch),
            "omega": float(config.omega),
            "innovations": list(config.innovations),
            "sigma_bounds": [float(config.sigma_min), float(config.sigma_max)],
            "eta": float(config.eta),
            "lambda_scale": float(config.lambda_scale),
            "bandwidths": list(config.bandwidths),
            "reference_sigma": problem.target.reference_sigma.detach().cpu().tolist(),
        },
        "optimization": {
            "starts": [
                {key: value for key, value in row.items() if key != "variables"}
                for row in results
            ],
            "selected_start": str(selected["name"]),
            "total_objective_evaluations": int(evaluations),
        },
        "reference": {
            "objective": _components(problem, reference),
            "errors": reference_errors,
        },
        "ideal_one_stage_mp": one_stage_payload
        | {
            "stage": stage,
            "objective_gain_fraction_of_exact": _objective_gain_fraction(
                reference_objective, exact_objective, one_stage_objective
            ),
        },
        "exact_tree_optimum": exact_payload,
        "exact_mp_audit": {
            key: value
            for key, value in audit.items()
            if key not in {"P_levels", "R_levels"}
        },
        "decision": {
            "stationarity_verified": bool(
                float(audit["volatility_log_best_response_rms"]) <= 1e-5
                and float(audit["independent_adjoint_gradient_relative_error"])
                <= 1e-8
            ),
            "strictly_interior": bool(
                float(audit["sigma_lower_activity"]) == 0.0
                and float(audit["sigma_upper_activity"]) == 0.0
                and float(audit["minimum_sigma_bound_margin"]) > 1e-4
            ),
            "all_starts_same_objective_within_1e_7": bool(
                max(float(row["final_objective"]) for row in results)
                - min(float(row["final_objective"]) for row in results)
                <= 1e-7
            ),
        },
    }
    write_json(run_dir / "exact_report.json", report)
    torch.save(
        {
            "exact_variables": exact_variables.detach().cpu(),
            "one_stage_variables": one_stage_variables.detach().cpu(),
        },
        run_dir / "exact_solutions.pt",
    )
    (run_dir / "EXACT_SUMMARY.md").write_text(
        "# Exact GARCH-tree oracle\n\n"
        f"- Horizon / branches / leaves: `{config.horizon}` / "
        f"`{config.branching}` / `{config.leaf_count}`\n"
        f"- Reference objective: `{reference_objective:.8g}`\n"
        f"- Exact-tree objective: `{exact_objective:.8g}`\n"
        f"- Ideal one-stage objective: `{one_stage_objective:.8g}`\n"
        f"- Ideal one-stage share of exact objective gain: "
        f"`{report['ideal_one_stage_mp']['objective_gain_fraction_of_exact']:.2%}`\n"
        f"- Exact MP stationarity verified: "
        f"`{report['decision']['stationarity_verified']}`\n"
        f"- Exact solution interior: `{report['decision']['strictly_interior']}`\n"
        f"- All starts agree within 1e-7: "
        f"`{report['decision']['all_starts_same_objective_within_1e_7']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


def _policy_payload(
    problem: ExactGarchTreeProblem,
    policy: CausalGarchAdjointPolicy,
    fit,
    reference_errors: dict[str, float],
    *,
    reference_objective: float,
    exact_objective: float,
    swd_projections: int,
) -> dict[str, object]:
    solution, p_prediction, r_prediction = rollout_garch_policy(problem, policy)
    errors = garch_tree_metric_errors(
        problem,
        solution,
        swd_projections=int(swd_projections),
    )
    closure = gap_closed(reference_errors, errors)
    objective = {
        key: float(value.detach().cpu().item())
        for key, value in problem.objective_components_for_solution(solution)[1].items()
    }
    p_target, r_target = problem.conditional_adjoint_targets_for_solution(solution)

    def residual(predictions, targets) -> float:
        error = 0.0
        energy = 0.0
        for level, (prediction, target) in enumerate(zip(predictions, targets)):
            probability = problem.level_node_probability(level)
            error += probability * float(
                torch.sum((prediction.detach() - target).double().pow(2)).item()
            )
            energy += probability * float(torch.sum(target.double().pow(2)).item())
        return math.sqrt(error / max(energy, 1e-24))

    return {
        "stable": bool(fit.stable),
        "objective": objective,
        "objective_gain_fraction_of_exact": _objective_gain_fraction(
            reference_objective, exact_objective, float(objective["total"])
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
        "relative_p_residual": residual(p_prediction, p_target),
        "relative_r_residual": residual(r_prediction, r_target),
        "history": list(fit.history),
        "resources": fit.resources,
    }


def run_seed(args: argparse.Namespace) -> None:
    if args.seed is None:
        raise ValueError("--seed is required for the seed phase")
    run_dir = args.run_dir.resolve()
    exact_report = json.loads((run_dir / "exact_report.json").read_text())
    config = _config_from_report(exact_report)
    device = torch.device(args.device)
    problem = ExactGarchTreeProblem(config, device=device, dtype=torch.float64)
    saved = torch.load(
        run_dir / "exact_solutions.pt", map_location=device, weights_only=True
    )
    exact_variables = saved["exact_variables"].to(device=device, dtype=torch.float64)
    reference_variables = problem.reference_start()
    reference_objective = float(problem.objective(reference_variables).item())
    exact_objective = float(problem.objective(exact_variables).item())
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference_variables),
        swd_projections=int(args.swd_projections),
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    initial = CausalGarchAdjointPolicy(
        hidden_dim=int(args.hidden_dim), num_layers=int(args.num_layers)
    ).to(device=device, dtype=torch.float64)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }

    print(f"GARCH tree seed {args.seed}: online one-stage MP", flush=True)
    online_fit = fit_online_garch_mp(
        problem,
        initial,
        steps=int(args.steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
    )
    online_policy = policy_from_state_dict(
        online_fit.state_dict,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        device=device,
        dtype=torch.float64,
    )
    online = _policy_payload(
        problem,
        online_policy,
        online_fit,
        reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        swd_projections=int(args.swd_projections),
    )
    online_state = {
        name: value.detach().cpu().clone()
        for name, value in online_fit.state_dict.items()
    }
    del online_policy, online_fit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"GARCH tree seed {args.seed}: direct pathwise", flush=True)
    initial.load_state_dict(initial_state)
    direct_fit = fit_direct_garch_objective(
        problem,
        initial,
        steps=int(args.steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
    )
    direct_policy = policy_from_state_dict(
        direct_fit.state_dict,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        device=device,
        dtype=torch.float64,
    )
    direct = _policy_payload(
        problem,
        direct_policy,
        direct_fit,
        reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        swd_projections=int(args.swd_projections),
    )
    direct_state = {
        name: value.detach().cpu().clone()
        for name, value in direct_fit.state_dict.items()
    }

    seed_dir = ensure_run_dir(run_dir / f"seed_{int(args.seed)}")
    report = {
        "protocol": {
            "seed": int(args.seed),
            "device": str(device),
            "shared_initialization": True,
            "same_causal_gru_and_hamiltonian_control_map": True,
            "online_targets": "exactly refreshed after every policy update",
            "P_R_loss": "equal-weight exact-population mean squared error",
            "checkpoint_selection": "lowest exact-population objective on the fixed 25-step grid",
            "steps_per_arm": int(args.steps),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip_norm": float(args.grad_clip_norm),
        },
        "reference": {
            "objective": reference_objective,
            "errors": reference_errors,
        },
        "exact_tree_optimum": {
            "objective": exact_objective,
        },
        "online_one_stage_mp": online,
        "direct_pathwise": direct,
    }
    write_json(seed_dir / "report.json", report)
    torch.save(
        {
            "initial_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in initial_state.items()
            },
            "online_state_dict": online_state,
            "direct_state_dict": direct_state,
        },
        seed_dir / "policies.pt",
    )
    print(
        json.dumps(
            {
                name: {
                    "objective": row["objective"]["total"],
                    "fraction_exact": row["objective_gain_fraction_of_exact"],
                    "distribution": row["summary"]["distribution"],
                    "dependence": row["summary"]["dependence"],
                    "seconds": row["resources"]["wall_seconds"],
                }
                for name, row in (
                    ("online_one_stage_mp", online),
                    ("direct_pathwise", direct),
                )
            },
            indent=2,
        ),
        flush=True,
    )


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_summarize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    exact = json.loads((run_dir / "exact_report.json").read_text())
    reports = []
    for seed in args.seeds:
        path = run_dir / f"seed_{int(seed)}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text()))
    aggregate: dict[str, object] = {}
    for arm in ("online_one_stage_mp", "direct_pathwise"):
        rows = [report[arm] for report in reports]
        aggregate[arm] = {
            "complete_count": len(rows),
            "stable_count": sum(bool(row["stable"]) for row in rows),
            "objective": _aggregate(
                [float(row["objective"]["total"]) for row in rows]
            ),
            "objective_gain_fraction_of_exact": _aggregate(
                [float(row["objective_gain_fraction_of_exact"]) for row in rows]
            ),
            "distribution_median_gap_closed": _aggregate(
                [float(row["summary"]["distribution"]["median_gap_closed"]) for row in rows]
            ),
            "dependence_median_gap_closed": _aggregate(
                [float(row["summary"]["dependence"]["median_gap_closed"]) for row in rows]
            ),
            "wall_seconds": _aggregate(
                [float(row["resources"]["wall_seconds"]) for row in rows]
            ),
            "relative_p_residual": _aggregate(
                [float(row["relative_p_residual"]) for row in rows]
            ),
            "relative_r_residual": _aggregate(
                [float(row["relative_r_residual"]) for row in rows]
            ),
        }
    online_rows = [report["online_one_stage_mp"] for report in reports]
    locked_gate = {
        "exact_stationarity_verified": bool(
            exact["decision"]["stationarity_verified"]
        ),
        "exact_solution_interior": bool(exact["decision"]["strictly_interior"]),
        "all_exact_starts_agree": bool(
            exact["decision"]["all_starts_same_objective_within_1e_7"]
        ),
        "online_stable_seed_count": sum(
            bool(row["stable"]) for row in online_rows
        ),
        "online_mean_exact_objective_gain_fraction": float(
            aggregate["online_one_stage_mp"][
                "objective_gain_fraction_of_exact"
            ]["mean"]
        ),
        "online_positive_distribution_closure_every_seed": all(
            float(row["summary"]["distribution"]["median_gap_closed"]) > 0.0
            for row in online_rows
        ),
        "online_positive_dependence_closure_every_seed": all(
            float(row["summary"]["dependence"]["median_gap_closed"]) > 0.0
            for row in online_rows
        ),
    }
    locked_gate["passed"] = bool(
        locked_gate["exact_stationarity_verified"]
        and locked_gate["exact_solution_interior"]
        and locked_gate["all_exact_starts_agree"]
        and int(locked_gate["online_stable_seed_count"]) == len(reports)
        and float(locked_gate["online_mean_exact_objective_gain_fraction"])
        >= 0.5
        and locked_gate["online_positive_distribution_closure_every_seed"]
        and locked_gate["online_positive_dependence_closure_every_seed"]
    )
    summary = {
        "protocol": {
            "seeds": [int(seed) for seed in args.seeds],
            "selection": "exact-population checkpoint selection, no sampling or test split",
        },
        "exact": {
            "reference_objective": exact["reference"]["objective"]["total"],
            "exact_objective": exact["exact_tree_optimum"]["objective"]["total"],
            "ideal_one_stage_objective": exact["ideal_one_stage_mp"]["objective"]["total"],
            "ideal_one_stage_fraction": exact["ideal_one_stage_mp"]["objective_gain_fraction_of_exact"],
            "decision": exact["decision"],
        },
        "aggregate": aggregate,
        "locked_gate": locked_gate,
    }
    write_json(run_dir / "SUMMARY.json", summary)
    lines = [
        "# GARCH tree oracle experiment",
        "",
        "The target is a zero-drift, three-branch GARCH process. The local reference matches its unconditional variance but has no conditional volatility feedback.",
        "",
        f"- Horizon: `{exact['configuration']['horizon']}`",
        f"- Leaves: `{exact['configuration']['leaf_count']}`",
        f"- Reference objective: `{summary['exact']['reference_objective']:.6g}`",
        f"- Exact-tree objective: `{summary['exact']['exact_objective']:.6g}`",
        f"- Ideal one-stage share of exact objective gain: `{summary['exact']['ideal_one_stage_fraction']:.1%}`",
        f"- Locked gate passed: `{locked_gate['passed']}`",
        "",
        "| Learned method | Stable seeds | Exact objective gain captured | Distribution gap closed | Dependence gap closed | Time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("online_one_stage_mp", "Online one-stage MP"),
        ("direct_pathwise", "Direct pathwise"),
    ):
        row = aggregate[arm]
        lines.append(
            f"| {label} | {row['stable_count']}/{row['complete_count']} | "
            f"{row['objective_gain_fraction_of_exact']['mean']:.1%} ± "
            f"{row['objective_gain_fraction_of_exact']['sample_sd']:.1%} | "
            f"{row['distribution_median_gap_closed']['mean']:.1%} ± "
            f"{row['distribution_median_gap_closed']['sample_sd']:.1%} | "
            f"{row['dependence_median_gap_closed']['mean']:.1%} ± "
            f"{row['dependence_median_gap_closed']['sample_sd']:.1%} | "
            f"{row['wall_seconds']['mean']:.1f} s |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.phase == "exact":
        run_exact(args)
    elif args.phase == "seed":
        run_seed(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
