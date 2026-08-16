#!/usr/bin/env python3
"""Solve the complete short-horizon binary scenario-tree control problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.exact_scenario_tree import (  # noqa: E402
    ExactBinaryTreeProblem,
    ExactTreeConfig,
    build_delayed_memory_target,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402
from deep_mkv_gen_path_dt.reference import (  # noqa: E402
    fit_local_gaussian_reference_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "volatility_only"), required=True)
    parser.add_argument(
        "--fixed-drift",
        choices=("zero", "fitted_reference"),
        default="zero",
        help=(
            "Physical drift used by volatility-only controls.  The fitted "
            "reference arm estimates drift from a finite sample of target leaves "
            "and freezes its values at the exact decision nodes."
        ),
    )
    parser.add_argument("--drift-fit-seed", type=int, default=0)
    parser.add_argument("--drift-fit-paths", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-9)
    parser.add_argument("--function-tolerance", type=float, default=1e-14)
    parser.add_argument("--skip-kkt-conditioning", action="store_true")
    return parser.parse_args()


def _config(horizon: int) -> ExactTreeConfig:
    horizon = int(horizon)
    history_cutoff = max(2, horizon // 4)
    response_delay = max(1, horizon // 4)
    return ExactTreeConfig(
        horizon=horizon,
        dt=1.0 / float(horizon),
        sigma_low=0.18,
        sigma_high=0.45,
        sigma_min=0.03,
        sigma_max=0.80,
        drawdown_threshold=0.08,
        history_cutoff=history_cutoff,
        response_delay=response_delay,
        eta=4.0,
        lambda_scale=50.0,
        bandwidths=(0.5, 1.0, 2.0),
    )


def _projected_gradient(
    variables: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    projected = np.asarray(gradient, dtype=np.float64).copy()
    for index, (value, derivative, bound) in enumerate(
        zip(variables, gradient, bounds)
    ):
        lower, upper = bound
        if lower is not None and value <= float(lower) + tolerance and derivative > 0.0:
            projected[index] = 0.0
        if upper is not None and value >= float(upper) - tolerance and derivative < 0.0:
            projected[index] = 0.0
    return projected


def _memory_comparison(problem, solution) -> dict[str, object]:
    target_solution = problem.target_solution()
    reference_solution = problem.solution(problem.reference_start())
    target = problem.delayed_memory_metrics(target_solution)
    reference = problem.delayed_memory_metrics(reference_solution)
    generated = problem.delayed_memory_metrics(solution)
    errors = {
        key: abs(float(generated[key]) - float(target[key]))
        for key in target
    }
    gap_denominator = float(target["future_rv_hit_gap"]) - float(
        reference["future_rv_hit_gap"]
    )
    recovered = (
        (float(generated["future_rv_hit_gap"]) - float(reference["future_rv_hit_gap"]))
        / gap_denominator
        if abs(gap_denominator) > 1e-12
        else float("nan")
    )
    return {
        "target": target,
        "reference": reference,
        "generated": generated,
        "absolute_errors": errors,
        "future_rv_hit_gap_fraction_recovered_from_reference": recovered,
    }


def _fit_nodewise_reference_drift(
    config: ExactTreeConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    num_paths: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    if int(num_paths) < 2:
        raise ValueError("drift-fit-paths must be at least two")
    target = build_delayed_memory_target(config, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        int(target.leaf_paths.shape[0]),
        (int(num_paths),),
        generator=generator,
        device="cpu",
    ).to(device=device)
    sampled_paths = target.leaf_paths.index_select(0, indices)[..., None]
    grid = DiscreteTimeGrid(
        T=float(config.horizon) * float(config.dt),
        num_steps=int(config.horizon),
    )
    kernel = fit_local_gaussian_reference_kernel(
        target_paths=sampled_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    levels = []
    horizon = int(config.horizon)
    for step in range(horizon):
        descendant_count = 2 ** (horizon - step)
        prefixes = target.leaf_paths[::descendant_count, : step + 1, None]
        drift, _ = kernel.evaluate(prefixes, step_index=step)
        levels.append(drift[:, 0].detach())
    flattened = torch.cat(levels)
    return tuple(levels), {
        "seed": int(seed),
        "sampled_paths": int(num_paths),
        "ridge": 1e-3,
        "sampling": "uniform target leaves with replacement",
        "evaluation": "fitted causal drift frozen at exact target-tree nodes",
        "coefficient_summary": kernel.summary(),
        "node_drift_rms": float(flattened.double().pow(2).mean().sqrt().item()),
        "node_drift_mean": float(flattened.double().mean().item()),
        "node_drift_max_abs": float(flattened.double().abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    if str(args.fixed_drift) != "zero" and str(args.mode) != "volatility_only":
        raise ValueError("fitted_reference drift is only defined for volatility_only mode")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    config = _config(int(args.horizon))
    device = torch.device(args.device)
    fixed_drift_levels = None
    drift_fit = None
    if str(args.fixed_drift) == "fitted_reference":
        fixed_drift_levels, drift_fit = _fit_nodewise_reference_drift(
            config,
            device=device,
            dtype=torch.float64,
            seed=int(args.drift_fit_seed),
            num_paths=int(args.drift_fit_paths),
        )
    problem = ExactBinaryTreeProblem(
        config,
        mode=str(args.mode),
        fixed_drift_levels=fixed_drift_levels,
        device=device,
        dtype=torch.float64,
    )
    starts = {
        "reference": problem.reference_start(),
        "oracle": problem.oracle_start(),
        "midpoint": 0.5 * (problem.reference_start() + problem.oracle_start()),
    }
    bounds = problem.bounds()
    evaluations = 0

    def objective_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        tensor = torch.tensor(
            values,
            device=problem.device,
            dtype=problem.dtype,
            requires_grad=True,
        )
        objective = problem.objective(tensor)
        gradient = torch.autograd.grad(objective, tensor)[0]
        return (
            float(objective.detach().cpu().item()),
            gradient.detach().cpu().double().numpy(),
        )

    results = []
    for name, start in starts.items():
        print(f"{args.mode} N={args.horizon}: optimizing from {name}", flush=True)
        start_np = start.detach().cpu().double().numpy()
        initial_value, _ = objective_and_gradient(start_np)
        result = minimize(
            objective_and_gradient,
            start_np,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(args.max_iterations),
                "gtol": float(args.gradient_tolerance),
                "ftol": float(args.function_tolerance),
                "maxls": 80,
                "maxcor": 30,
            },
        )
        projected = _projected_gradient(result.x, result.jac, bounds)
        results.append(
            {
                "name": name,
                "result": result,
                "initial_objective": initial_value,
                "projected_gradient_rms": float(np.sqrt(np.mean(projected**2))),
                "projected_gradient_max": float(np.max(np.abs(projected))),
            }
        )
        print(
            f"  J={result.fun:.9g}, success={result.success}, "
            f"projected max={np.max(np.abs(projected)):.3g}",
            flush=True,
        )
    selected = min(
        results,
        key=lambda row: (
            float(row["result"].fun),
            float(row["projected_gradient_max"]),
            str(row["name"]),
        ),
    )
    variables = torch.tensor(
        selected["result"].x,
        device=problem.device,
        dtype=problem.dtype,
    )
    solution = problem.solution(variables)
    _, components = problem.objective_components(variables)
    adjoint = problem.exact_adjoint_audit(variables)
    memory = _memory_comparison(problem, solution)
    full_requires_drift = str(args.mode) == "full"
    stationarity_passed = bool(
        float(adjoint["volatility_log_best_response_rms"]) <= 1e-5
        and (
            not full_requires_drift
            or float(adjoint["drift_stationarity_rms"]) <= 1e-5
        )
        and float(adjoint["independent_adjoint_gradient_relative_error"]) <= 1e-8
    )
    interior = bool(
        float(adjoint["sigma_lower_activity"]) == 0.0
        and float(adjoint["sigma_upper_activity"]) == 0.0
        and float(adjoint["minimum_sigma_bound_margin"]) > 1e-4
    )
    conditioning = None
    print("computing the exact reduced-objective Hessian", flush=True)
    reduced_hessian = problem.reduced_hessian_conditioning(variables)
    if not bool(args.skip_kkt_conditioning):
        print("constructing and conditioning the independent full KKT matrix", flush=True)
        cpu_problem = ExactBinaryTreeProblem(
            config,
            mode=str(args.mode),
            fixed_drift_levels=(
                None
                if fixed_drift_levels is None
                else tuple(value.detach().cpu() for value in fixed_drift_levels)
            ),
            device="cpu",
            dtype=torch.float64,
        )
        conditioning = cpu_problem.full_kkt_conditioning(variables.detach().cpu())

    start_reports = []
    for row in results:
        result = row["result"]
        start_reports.append(
            {
                "name": row["name"],
                "initial_objective": float(row["initial_objective"]),
                "final_objective": float(result.fun),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "projected_gradient_rms": float(row["projected_gradient_rms"]),
                "projected_gradient_max": float(row["projected_gradient_max"]),
            }
        )
    report = {
        "protocol": {
            "exact_complete_binary_tree": True,
            "horizon": int(config.horizon),
            "leaf_count": int(config.leaf_count),
            "decision_node_count": int(config.decision_nodes),
            "control_mode": str(args.mode),
            "fixed_physical_drift": str(args.fixed_drift),
            "binary_noise_probabilities": [0.5, 0.5],
            "conditional_expectations": "exact child-probability averages",
            "neural_network_used": False,
            "regression_used": bool(drift_fit is not None),
            "nested_monte_carlo_used": False,
            "sampling_bank_used": bool(drift_fit is not None),
            "optimizer": "simultaneous reduced-space L-BFGS-B with exact autograd gradient",
            "independent_mp_verification": True,
        },
        "configuration": {
            "dt": float(config.dt),
            "sigma_low": float(config.sigma_low),
            "sigma_high": float(config.sigma_high),
            "sigma_bounds": [float(config.sigma_min), float(config.sigma_max)],
            "drawdown_threshold": float(config.drawdown_threshold),
            "history_cutoff": int(config.history_cutoff),
            "response_delay": int(config.response_delay),
            "response_start": int(config.response_start),
            "eta": float(config.eta),
            "lambda_scale": float(config.lambda_scale),
            "bandwidths": list(config.bandwidths),
            "fitted_time_marginal_reference_sigma": problem.target.reference_sigma.detach().cpu().tolist(),
            "fitted_reference_drift": drift_fit,
        },
        "optimization": {
            "starts": start_reports,
            "selected_start": str(selected["name"]),
            "total_objective_evaluations": int(evaluations),
        },
        "primal": {
            key: float(value.detach().cpu().item()) for key, value in components.items()
        },
        "exact_mp_audit": {
            key: value
            for key, value in adjoint.items()
            if key not in {"P_levels", "R_levels"}
        },
        "delayed_memory": memory,
        "conditioning": conditioning,
        "reduced_hessian": reduced_hessian,
        "decision": {
            "stationarity_verified": stationarity_passed,
            "stationary_solution_strictly_away_from_sigma_bounds": interior,
            "strict_local_minimum_in_reduced_control_space": bool(
                float(reduced_hessian["minimum_eigenvalue"]) > 1e-8
            ),
        },
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "variables": variables.detach().cpu(),
            "alpha_levels": tuple(value.detach().cpu() for value in solution.alpha_levels),
            "sigma_levels": tuple(value.detach().cpu() for value in solution.sigma_levels),
            "state_levels": tuple(value.detach().cpu() for value in solution.state_levels),
            "leaf_paths": solution.leaf_paths.detach().cpu(),
            "P_levels": adjoint["P_levels"],
            "R_levels": adjoint["R_levels"],
        },
        run_dir / "solution.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Exact finite-tree MP solve\n\n"
        f"- Mode: `{args.mode}`\n"
        f"- Fixed physical drift: `{args.fixed_drift}`\n"
        f"- Horizon / leaves / decision nodes: `{config.horizon}` / "
        f"`{config.leaf_count}` / `{config.decision_nodes}`\n"
        f"- Selected start: `{selected['name']}`\n"
        f"- Objective: `{components['total'].item():.9g}`\n"
        f"- MMD2: `{components['mmd2'].item():.9g}`\n"
        f"- Drift MP residual RMS: `{adjoint['drift_stationarity_rms']:.6g}`\n"
        f"- Volatility MP residual RMS: `{adjoint['volatility_log_best_response_rms']:.6g}`\n"
        f"- Independent adjoint/gradient relative error: "
        f"`{adjoint['independent_adjoint_gradient_relative_error']:.3g}`\n"
        f"- Sigma lower/upper activity: `{adjoint['sigma_lower_activity']:.2%}` / "
        f"`{adjoint['sigma_upper_activity']:.2%}`\n"
        f"- Stationarity verified: `{stationarity_passed}`\n"
        f"- Strictly interior sigma solution: `{interior}`\n"
        f"- Future-RV hit-gap error: "
        f"`{memory['absolute_errors']['future_rv_hit_gap']:.6g}`\n"
        f"- Future-RV hit-gap fraction recovered from reference: "
        f"`{memory['future_rv_hit_gap_fraction_recovered_from_reference']:.2%}`\n"
        f"- Reduced-Hessian minimum eigenvalue: "
        f"`{reduced_hessian['minimum_eigenvalue']:.6g}`\n"
        + (
            f"- Full KKT condition number: `{conditioning['condition_number']:.6g}`\n"
            if conditioning is not None
            else "- Full KKT conditioning: `skipped`\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
