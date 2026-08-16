#!/usr/bin/env python3
"""Run one matched N=8 nested-MP versus direct-training seed."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys

import numpy as np
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
    gap_closed,
    summarize_gap_closure,
)
from path_dt_experiments.exact_tree_scalable import (  # noqa: E402
    CausalVolatilityPolicy,
    fit_direct_exact_objective,
    fit_nested_exact_mp,
    policy_from_state_dict,
    rollout_causal_policy,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--outer-steps", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--swd-projections", type=int, default=1024)
    parser.add_argument("--swd-seed", type=int, default=20260804)
    return parser.parse_args()


def _problem(exact_run_dir: Path, device: torch.device) -> tuple[ExactBinaryTreeProblem, torch.Tensor]:
    mode_dir = exact_run_dir / "volatility_only"
    report = json.loads((mode_dir / "report.json").read_text())
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
        mode="volatility_only",
        device=device,
        dtype=torch.float64,
    )
    saved = torch.load(mode_dir / "solution.pt", map_location=device, weights_only=True)
    exact_variables = saved["variables"].to(device=device, dtype=torch.float64)
    return problem, exact_variables


def _parameter_count(policy: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters())


def _arm_report(
    problem: ExactBinaryTreeProblem,
    exact_errors: dict[str, float],
    reference_errors: dict[str, float],
    policy: CausalVolatilityPolicy,
    fit,
    *,
    swd_projections: int,
    swd_seed: int,
) -> dict[str, object]:
    solution, predicted_r = rollout_causal_policy(problem, policy)
    _, objective = problem.objective_components_for_solution(solution)
    errors = exact_tree_metric_errors(
        problem,
        solution,
        swd_projections=int(swd_projections),
        swd_seed=int(swd_seed),
    )
    closure = gap_closed(reference_errors, errors)
    exact_closure = gap_closed(reference_errors, exact_errors)
    fraction_exact = {}
    for name, value in closure.items():
        attainable = exact_closure[name]
        fraction_exact[name] = (
            None
            if value is None or attainable is None or float(attainable) <= 0.0
            else float(value) / float(attainable)
        )
    target_p, target_r = problem.conditional_adjoint_targets_for_solution(solution)
    del target_p
    weighted_r_error = 0.0
    weighted_r_energy = 0.0
    for level, (prediction, target) in enumerate(zip(predicted_r, target_r)):
        probability = 2.0 ** (-level)
        weighted_r_error += probability * float(
            torch.sum((prediction.detach() - target).double().pow(2)).item()
        )
        weighted_r_energy += probability * float(
            torch.sum(target.double().pow(2)).item()
        )
    relative_r_residual = (weighted_r_error / max(weighted_r_energy, 1e-24)) ** 0.5
    return {
        "stable": bool(fit.stable),
        "objective": {
            key: float(value.detach().cpu().item()) for key, value in objective.items()
        },
        "errors": errors,
        "gap_closed": closure,
        "fraction_of_exact_gap_closure": fraction_exact,
        "summary": {
            "distribution": summarize_gap_closure(
                closure, PRIMARY_DISTRIBUTION_METRICS
            ),
            "dependence": summarize_gap_closure(closure, DEPENDENCE_METRICS),
            "local_calibration": summarize_gap_closure(
                closure, LOCAL_CALIBRATION_METRICS
            ),
        },
        "relative_exact_r_stationarity_residual": relative_r_residual,
        "resources": fit.resources,
        "history": list(fit.history),
    }


def main() -> None:
    args = parse_args()
    if int(args.steps) != int(args.outer_steps) * int(args.inner_steps):
        raise ValueError("direct steps must equal nested outer_steps * inner_steps")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    exact_run_dir = args.exact_run_dir.resolve()
    device = torch.device(args.device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    problem, exact_variables = _problem(exact_run_dir, device)
    initial = CausalVolatilityPolicy(
        hidden_dim=int(args.hidden_dim), num_layers=int(args.num_layers)
    ).to(device=device, dtype=torch.float64)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }
    reference_solution = problem.solution(problem.reference_start())
    exact_solution = problem.solution(exact_variables)
    reference_errors = exact_tree_metric_errors(
        problem,
        reference_solution,
        swd_projections=int(args.swd_projections),
        swd_seed=int(args.swd_seed),
    )
    exact_errors = exact_tree_metric_errors(
        problem,
        exact_solution,
        swd_projections=int(args.swd_projections),
        swd_seed=int(args.swd_seed),
    )

    print(f"seed {args.seed}: nested MP", flush=True)
    nested = fit_nested_exact_mp(
        problem,
        initial,
        outer_steps=int(args.outer_steps),
        inner_steps=int(args.inner_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    nested_policy = policy_from_state_dict(
        nested.state_dict,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        device=device,
        dtype=torch.float64,
    )
    nested_report = _arm_report(
        problem,
        exact_errors,
        reference_errors,
        nested_policy,
        nested,
        swd_projections=int(args.swd_projections),
        swd_seed=int(args.swd_seed),
    )
    nested_state = {
        name: value.detach().cpu().clone()
        for name, value in nested.state_dict.items()
    }
    del nested_policy
    del nested
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"seed {args.seed}: direct pathwise", flush=True)
    initial.load_state_dict(initial_state)
    direct = fit_direct_exact_objective(
        problem,
        initial,
        steps=int(args.steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    direct_policy = policy_from_state_dict(
        direct.state_dict,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        device=device,
        dtype=torch.float64,
    )
    direct_report = _arm_report(
        problem,
        exact_errors,
        reference_errors,
        direct_policy,
        direct,
        swd_projections=int(args.swd_projections),
        swd_seed=int(args.swd_seed),
    )
    direct_state = {
        name: value.detach().cpu().clone()
        for name, value in direct.state_dict.items()
    }

    report = {
        "protocol": {
            "seed": int(args.seed),
            "device": str(device),
            "exact_run_dir": str(exact_run_dir),
            "horizon": int(problem.config.horizon),
            "leaf_count": int(problem.config.leaf_count),
            "control_mode": "volatility_only",
            "shared_initial_state": True,
            "same_policy_class": True,
            "same_reference_objective_and_hamiltonian_map": True,
            "sampling_used": False,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "parameter_count": _parameter_count(initial),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "optimizer_steps_per_arm": int(args.steps),
            "nested_outer_steps": int(args.outer_steps),
            "nested_inner_steps": int(args.inner_steps),
            "checkpoint_selection": "lowest exact-population objective encountered",
        },
        "reference": {
            "errors": reference_errors,
            "objective": float(problem.objective(problem.reference_start()).item()),
        },
        "exact_mp": {
            "errors": exact_errors,
            "gap_closed": gap_closed(reference_errors, exact_errors),
            "objective": float(problem.objective(exact_variables).item()),
        },
        "nested_mp": nested_report,
        "direct_pathwise": direct_report,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "initial_state_dict": initial_state,
            "nested_state_dict": nested_state,
            "direct_state_dict": direct_state,
        },
        run_dir / "policies.pt",
    )
    summary = {
        arm: {
            "stable": report[arm]["stable"],
            "objective": report[arm]["objective"]["total"],
            "distribution": report[arm]["summary"]["distribution"],
            "dependence": report[arm]["summary"]["dependence"],
            "wall_seconds": report[arm]["resources"]["wall_seconds"],
            "peak_bytes": report[arm]["resources"]["cuda_incremental_peak_allocated_bytes"],
        }
        for arm in ("nested_mp", "direct_pathwise")
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
