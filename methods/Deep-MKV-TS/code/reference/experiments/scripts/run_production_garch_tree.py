#!/usr/bin/env python3
"""Check the production Deep MKV solver against a finite GARCH-tree oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPTrainingConfig,
    FitReplayBank,
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
    ExactGarchTreeConditionalTargetProvider,
    ProductionGarchTreeProblem,
    build_production_garch_components,
    enumerated_garch_noise,
    production_model_on_exact_tree,
)
from path_dt_experiments.garch_tree_guyon import (  # noqa: E402
    GuyonProductionGarchTreeProblem,
)
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("exact", "deep-mkv"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--swd-projections", type=int, default=512)
    parser.add_argument(
        "--reference-kind",
        choices=("constant", "guyon"),
        default="constant",
    )
    parser.add_argument("--adjoint-weight", type=float, default=1.0)
    parser.add_argument("--adjoint-noise-weight", type=float, default=1.0)
    parser.add_argument(
        "--exact-tree-replay",
        action="store_true",
        help="train on the complete enumerated tree with direct GRU targets",
    )
    parser.add_argument(
        "--oracle-ce-targets",
        action="store_true",
        help="replace only the GRU P/R labels by exact nodewise conditional targets",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> GarchTreeConfig:
    return GarchTreeConfig(
        horizon=int(args.horizon),
        dt=1.0 / float(args.horizon),
        long_run_sigma=0.25,
        arch=0.35,
        garch=0.60,
        sigma_min=0.05,
        sigma_max=0.70,
        eta=float(args.eta),
        lambda_scale=float(args.lambda_scale),
    )


def make_problem(
    args: argparse.Namespace,
    config: GarchTreeConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ProductionGarchTreeProblem:
    problem_type = (
        GuyonProductionGarchTreeProblem
        if str(args.reference_kind) == "guyon"
        else ProductionGarchTreeProblem
    )
    return problem_type(
        config,
        lambda_scale=float(args.lambda_scale),
        kappa_scale=float(args.kappa_scale),
        device=device,
        dtype=dtype,
    )


def _components(
    problem: ProductionGarchTreeProblem, variables: torch.Tensor
) -> dict[str, float]:
    _, components = problem.objective_components(variables)
    return {
        name: float(value.detach().cpu().item())
        for name, value in components.items()
    }


def _projected_gradient(
    point: np.ndarray,
    gradient: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    result = np.asarray(gradient, dtype=np.float64).copy()
    for index, (value, derivative, (lower, upper)) in enumerate(
        zip(point, gradient, bounds)
    ):
        if value <= lower + 1e-10 and derivative > 0.0:
            result[index] = 0.0
        if value >= upper - 1e-10 and derivative < 0.0:
            result[index] = 0.0
    return result


def _gain_fraction(reference: float, exact: float, candidate: float) -> float:
    denominator = reference - exact
    return (reference - candidate) / denominator if denominator > 1e-14 else math.nan


def _metric_payload(
    problem: ProductionGarchTreeProblem,
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


def run_exact(args: argparse.Namespace) -> None:
    run_dir = ensure_run_dir(args.run_dir.resolve())
    config = make_config(args)
    device = torch.device(args.device)
    problem = make_problem(
        args, config, device=device, dtype=torch.float64
    )
    reference = problem.reference_start()
    target = problem.target_start()
    starts = {
        "reference": reference,
        "target": target,
        "midpoint": 0.5 * (reference + target),
    }
    bounds = problem.bounds()
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
            gradient.detach().cpu().numpy(),
        )

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for name, start in starts.items():
        print(f"production oracle: optimizing from {name}", flush=True)
        result = minimize(
            objective_and_gradient,
            start.detach().cpu().numpy(),
            jac=True,
            method="L-BFGS-B",
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
            "objective": float(result.fun),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "projected_gradient_max": float(np.max(np.abs(projected))),
            "variables": result.x,
        }
        rows.append(row)
        print(
            f"  J={result.fun:.9g}; projected gradient max="
            f"{row['projected_gradient_max']:.3g}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    selected = min(
        rows,
        key=lambda row: (
            float(row["objective"]),
            float(row["projected_gradient_max"]),
        ),
    )
    exact = torch.tensor(
        selected["variables"], device=device, dtype=torch.float64
    )
    audit = problem.exact_adjoint_audit(exact)
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )
    exact_payload = _metric_payload(
        problem,
        exact,
        reference_errors,
        swd_projections=int(args.swd_projections),
    )
    reference_objective = float(problem.objective(reference).item())
    exact_objective = float(problem.objective(exact).item())
    report = {
        "protocol": {
            "role": "finite-tree mathematical and solver oracle",
            "production_discrepancy": True,
            "control": "volatility_only",
            "physical_drift": "identically_zero",
            "reference_kind": str(args.reference_kind),
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "eta": float(args.eta),
            "tree_noise": "three-point equal-probability quadrature",
            "conditional_expectations": "exact nodewise averages",
            "comparison_models": "none",
        },
        "configuration": {
            "horizon": int(config.horizon),
            "dt": float(config.dt),
            "branching": int(config.branching),
            "leaves": int(config.leaf_count),
            "decision_nodes": int(config.decision_nodes),
            "long_run_sigma": float(config.long_run_sigma),
            "arch": float(config.arch),
            "garch": float(config.garch),
            "sigma_bounds": [float(config.sigma_min), float(config.sigma_max)],
        },
        "optimization": {
            "selected_start": str(selected["name"]),
            "total_evaluations": int(evaluations),
            "wall_seconds": float(elapsed),
            "starts": [
                {key: value for key, value in row.items() if key != "variables"}
                for row in rows
            ],
        },
        "reference": {
            "objective": _components(problem, reference),
            "errors": reference_errors,
        },
        "exact_tree_optimum": exact_payload,
        "exact_mp_audit": {
            key: value
            for key, value in audit.items()
            if key not in {"P_levels", "R_levels"}
        },
    }
    write_json(run_dir / "exact_report.json", report)
    torch.save({"exact_variables": exact.detach().cpu()}, run_dir / "exact.pt")
    print(
        json.dumps(
            {
                "reference_objective": reference_objective,
                "exact_objective": exact_objective,
                "adjoint_gradient_relative_error": audit[
                    "independent_adjoint_gradient_relative_error"
                ],
                "volatility_fixed_point_residual": audit[
                    "volatility_log_best_response_rms"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def _solution_payload(
    problem: ProductionGarchTreeProblem,
    model,
    reference_errors: dict[str, float],
    *,
    reference_objective: float,
    exact_objective: float,
    swd_projections: int,
) -> dict[str, object]:
    solution, replay = production_model_on_exact_tree(model, problem)
    variables = problem.pack(solution.sigma_levels)
    payload = _metric_payload(
        problem,
        variables,
        reference_errors,
        swd_projections=int(swd_projections),
    )
    objective = float(payload["objective"]["total"])
    sigma = torch.cat(solution.sigma_levels)

    noise = enumerated_garch_noise(
        problem.config, device=model.device, dtype=model.dtype
    )
    with torch.no_grad():
        rollout = model.rollout(
            x0=torch.zeros(
                problem.config.leaf_count,
                1,
                device=model.device,
                dtype=model.dtype,
            ),
            noise=noise,
        )
    target_p, target_r = problem.conditional_adjoint_targets_for_solution(solution)

    def relative_residual(prediction: torch.Tensor, targets) -> float:
        error = 0.0
        energy = 0.0
        branching = int(problem.config.branching)
        horizon = int(problem.config.horizon)
        for level, target in enumerate(targets):
            nodes = branching**level
            descendants = branching ** (horizon - level)
            node_prediction = prediction[:, level, 0].reshape(
                nodes, descendants
            )[:, 0].double()
            probability = problem.level_node_probability(level)
            error += probability * float(
                torch.sum((node_prediction - target.double()).pow(2)).item()
            )
            energy += probability * float(torch.sum(target.double().pow(2)).item())
        return math.sqrt(error / max(energy, 1e-24))

    payload.update(
        {
            "objective_gain_fraction_of_exact": _gain_fraction(
                reference_objective, exact_objective, objective
            ),
            "relative_p_residual": relative_residual(
                rollout.expected_adjoint_next, target_p
            ),
            "relative_r_residual": relative_residual(
                rollout.expected_adjoint_noise_next, target_r
            ),
            "safety": {
                "sigma_min": float(sigma.min().item()),
                "sigma_max": float(sigma.max().item()),
                "lower_bound_activity": float(
                    (sigma <= problem.config.sigma_min * (1.0 + 1e-6))
                    .double()
                    .mean()
                    .item()
                ),
                "upper_bound_activity": float(
                    (sigma >= problem.config.sigma_max * (1.0 - 1e-6))
                    .double()
                    .mean()
                    .item()
                ),
            },
            "replay": replay,
        }
    )
    return payload


def run_deep_mkv(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    exact_report = json.loads((run_dir / "exact_report.json").read_text())
    config = make_config(args)
    locked = exact_report["protocol"]
    for key, value in (
        ("lambda_scale", float(args.lambda_scale)),
        ("kappa_scale", float(args.kappa_scale)),
        ("eta", float(args.eta)),
    ):
        if float(locked[key]) != value:
            raise ValueError(f"{key} differs from the exact-tree protocol")
    if int(exact_report["configuration"]["horizon"]) != int(config.horizon):
        raise ValueError("horizon differs from the exact-tree protocol")
    if str(locked.get("reference_kind", "constant")) != str(args.reference_kind):
        raise ValueError("reference_kind differs from the exact-tree protocol")

    device = torch.device(args.device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    problem = make_problem(
        args, config, device=device, dtype=torch.float64
    )
    if isinstance(problem, GuyonProductionGarchTreeProblem):
        model = problem.build_model(
            hidden_dim=int(args.hidden_dim),
            seed=int(args.seed),
            dtype=torch.float32,
        )
    else:
        model, _, _ = build_production_garch_components(
            config,
            lambda_scale=float(args.lambda_scale),
            kappa_scale=float(args.kappa_scale),
            hidden_dim=int(args.hidden_dim),
            seed=int(args.seed),
            device=device,
            dtype=torch.float32,
        )
    reference = problem.reference_start()
    exact = torch.load(
        run_dir / "exact.pt", map_location=device, weights_only=True
    )["exact_variables"].to(device=device, dtype=torch.float64)
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )
    reference_objective = float(problem.objective(reference).item())
    exact_objective = float(problem.objective(exact).item())
    initial = _solution_payload(
        problem,
        model,
        reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        swd_projections=int(args.swd_projections),
    )
    best_checkpoint = copy.deepcopy(model.checkpoint_state())
    best_objective = float(initial["objective"]["total"])
    best_step = 0
    validation_history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "exact_tree_objective": best_objective,
            "zero_drift_max_abs": 0.0,
            "selection_eligible": 1.0,
        }
    ]

    def callback(current_model, metrics: dict[str, float]) -> None:
        nonlocal best_checkpoint, best_objective, best_step
        solution, replay = production_model_on_exact_tree(current_model, problem)
        objective = float(
            problem.objective_components_for_solution(solution)[0]
            .detach()
            .cpu()
            .item()
        )
        step = int(metrics["step"])
        validation_history.append(
            {
                "step": float(step),
                "exact_tree_objective": objective,
                "zero_drift_max_abs": float(replay["zero_drift_max_abs"]),
                "selection_eligible": float(
                    step % int(args.log_every) == 0 or step == int(args.steps)
                ),
            }
        )
        selection_eligible = (
            step % int(args.log_every) == 0 or step == int(args.steps)
        )
        if (
            selection_eligible
            and math.isfinite(objective)
            and objective < best_objective
        ):
            best_objective = objective
            best_step = step
            best_checkpoint = copy.deepcopy(current_model.checkpoint_state())
        print(
            f"Deep MKV step {step}: tree J={objective:.8g}; "
            f"best={best_objective:.8g} at {best_step}",
            flush=True,
        )

    replay_bank = None
    if bool(args.exact_tree_replay):
        full_indices = torch.arange(config.leaf_count, dtype=torch.long)
        replay_bank = FitReplayBank(
            source_indices=full_indices,
            target_indices=full_indices,
            noise=enumerated_garch_noise(
                config, device=device, dtype=model.dtype
            ),
        )
    if bool(args.oracle_ce_targets) and replay_bank is None:
        raise ValueError("--oracle-ce-targets requires --exact-tree-replay")
    conditional_target_provider = (
        ExactGarchTreeConditionalTargetProvider(problem)
        if bool(args.oracle_ce_targets)
        else None
    )
    training = DiscreteMPTrainingConfig(
        batch_size=(config.leaf_count if replay_bank is not None else 256),
        target_batch_size=(config.leaf_count if replay_bank is not None else 256),
        num_steps=int(args.steps),
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=float(args.lambda_scale),
        adjoint_weight=float(args.adjoint_weight),
        adjoint_noise_weight=float(args.adjoint_noise_weight),
        ce_target_mode=("direct" if replay_bank is not None else "ridge"),
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner=("timewise" if replay_bank is not None else "none"),
        preconditioner_batches=(1 if replay_bank is not None else 8),
        noise_target_control_variate="none",
        grad_clip_norm=5.0,
        log_every=int(args.log_every),
        seed=int(args.seed),
        observed_only=False,
    )
    target = problem.target.leaf_paths.unsqueeze(-1).float()
    started = time.perf_counter()
    fit = model.fit(
        target,
        training=training,
        log_callback=callback,
        replay_bank=replay_bank,
        conditional_target_provider=conditional_target_provider,
    )
    training_seconds = time.perf_counter() - started
    final_checkpoint = copy.deepcopy(model.checkpoint_state())
    final = _solution_payload(
        problem,
        model,
        reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        swd_projections=int(args.swd_projections),
    )
    model.load_checkpoint_state(best_checkpoint)
    selected = _solution_payload(
        problem,
        model,
        reference_errors,
        reference_objective=reference_objective,
        exact_objective=exact_objective,
        swd_projections=int(args.swd_projections),
    )
    selected_solution, _ = production_model_on_exact_tree(model, problem)
    report = {
        "protocol": {
            "method": "production DiscreteMPModel.fit",
            "label": "Deep MKV",
            "seed": int(args.seed),
            "physical_drift": "identically_zero",
            "controlled_channel": "volatility_only",
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "eta": float(args.eta),
            "reference_kind": str(args.reference_kind),
            "adjoint_weight": float(args.adjoint_weight),
            "adjoint_noise_weight": float(args.adjoint_noise_weight),
            "steps": int(args.steps),
            "hidden_dim": int(args.hidden_dim),
            "conditional_estimator": (
                "exact nodewise oracle CE labels distilled into the causal GRU"
                if conditional_target_provider is not None
                else "direct nonlinear causal GRU on complete tree"
                if replay_bank is not None
                else "production raw-prefix ridge"
            ),
            "loss": "production P/R adjoint consistency loss",
            "training_noise": (
                "complete three-point tree quadrature"
                if replay_bank is not None
                else "Gaussian"
            ),
            "tree_evaluation_noise": "same three-point quadrature as exact oracle",
            "exact_tree_replay": bool(replay_bank is not None),
            "oracle_ce_targets": bool(conditional_target_provider is not None),
            "checkpoint_selection": "lowest exact-tree objective on locked 100-step grid",
            "comparison_models": "none",
        },
        "reference": exact_report["reference"],
        "exact_tree_optimum": exact_report["exact_tree_optimum"],
        "initial_deep_mkv": initial,
        "selected_deep_mkv": selected,
        "final_deep_mkv": final,
        "selection": {
            "selected_step": int(best_step),
            "selected_objective": float(best_objective),
            "training_wall_seconds": float(training_seconds),
        },
        "fit_final_metrics": fit.final_metrics,
    }
    seed_dir = ensure_run_dir(run_dir / f"seed_{int(args.seed)}")
    write_json(seed_dir / "report.json", report)
    write_history_jsonl(seed_dir / "fit_history.jsonl", fit.history)
    write_history_jsonl(
        seed_dir / "tree_validation_history.jsonl", validation_history
    )
    torch.save(
        {
            "selected_checkpoint": best_checkpoint,
            "final_checkpoint": final_checkpoint,
        },
        seed_dir / "checkpoints.pt",
    )
    torch.save(
        {
            "reference_variables": problem.reference_start().detach().cpu(),
            "exact_variables": exact.detach().cpu(),
            "selected_variables": problem.pack(
                selected_solution.sigma_levels
            ).detach().cpu(),
            "selected_leaf_paths": selected_solution.leaf_paths.detach().cpu(),
        },
        seed_dir / "tree_comparison.pt",
    )
    summary = selected["summary"]
    (seed_dir / "SUMMARY.md").write_text(
        "# Production Deep MKV on the GARCH tree\n\n"
        "This run uses the repository's production `DiscreteMPModel.fit`; it is "
        "not the earlier tree-specific training surrogate.\n\n"
        f"- Physical drift: exactly zero\n"
        f"- Controlled channel: volatility only\n"
        f"- Discrepancy weights: {args.lambda_scale:g} / {args.kappa_scale:g}\n"
        f"- Entropy coefficient: eta = {args.eta:g}\n"
        f"- Selected checkpoint: step {best_step}\n"
        f"- Exact objective gain captured: "
        f"{selected['objective_gain_fraction_of_exact']:.2%}\n"
        f"- Distribution metrics improved: "
        f"{summary['distribution']['improved_count']}/"
        f"{summary['distribution']['metric_count']}\n"
        f"- Dependence metrics improved: "
        f"{summary['dependence']['improved_count']}/"
        f"{summary['dependence']['metric_count']}\n"
        f"- Maximum absolute drift: "
        f"{selected['replay']['zero_drift_max_abs']:.3g}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_step": best_step,
                "objective_gain_fraction_of_exact": selected[
                    "objective_gain_fraction_of_exact"
                ],
                "distribution": summary["distribution"],
                "dependence": summary["dependence"],
                "safety": selected["safety"],
                "zero_drift": selected["replay"]["zero_drift_max_abs"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.phase == "exact":
        run_exact(args)
    else:
        run_deep_mkv(args)


if __name__ == "__main__":
    main()
