#!/usr/bin/env python3
"""Run a frozen-CE-fit / gated-law-update loop on the exact GARCH tree."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.frozen_garch_ce_gru import (  # noqa: E402
    FrozenExactCEFitConfig,
    collapse_leaf_controls_to_variables,
    fit_frozen_exact_ce_gru,
    frozen_ce_fit_metrics,
    objective_gated_exact_ce_update,
)
from path_dt_experiments.garch_tree_evaluation import (  # noqa: E402
    GARCH_DEPENDENCE_METRICS,
    GARCH_DISTRIBUTION_METRICS,
    gap_closed,
    garch_tree_metric_errors,
    summarize_gap_closure,
)
from path_dt_experiments.garch_tree_oracle import GarchTreeConfig  # noqa: E402
from path_dt_experiments.garch_tree_production import (  # noqa: E402
    ProductionGarchTreeProblem,
    exact_garch_tree_mp_response,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--oracle-trajectory-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-outer-steps", type=int, default=200)
    parser.add_argument("--fit-max-steps", type=int, default=3000)
    parser.add_argument("--fit-min-steps", type=int, default=200)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-log-every", type=int, default=100)
    parser.add_argument("--ce-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--fixed-point-tolerance", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swd-projections", type=int, default=512)
    return parser.parse_args()


def _load_problem(
    args: argparse.Namespace,
) -> tuple[ProductionGarchTreeProblem, torch.Tensor, torch.Tensor, dict[str, object]]:
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
    oracle = torch.load(
        args.oracle_trajectory_run_dir.resolve() / "solutions.pt",
        map_location=device,
        weights_only=True,
    )["final_variables"].to(device=device, dtype=torch.float64)
    return problem, exact, oracle, report


def _objective(problem: ProductionGarchTreeProblem, point: torch.Tensor) -> float:
    return float(problem.objective(point).detach().cpu().item())


def _direction_metrics(
    current: torch.Tensor,
    learned: torch.Tensor,
    exact: torch.Tensor,
) -> dict[str, float]:
    learned_direction = (learned - current).double()
    exact_direction = (exact - current).double()
    learned_norm = torch.linalg.vector_norm(learned_direction)
    exact_norm = torch.linalg.vector_norm(exact_direction).clamp_min(1e-30)
    cosine = torch.sum(learned_direction * exact_direction) / (
        learned_norm * exact_norm
    ).clamp_min(1e-30)
    return {
        "control_direction_cosine": float(cosine.item()),
        "control_direction_amplitude_ratio": float((learned_norm / exact_norm).item()),
        "learned_vs_exact_proposal_rms": float(
            (learned.double() - exact.double()).pow(2).mean().sqrt().item()
        ),
    }


def _cpu_state_dict(model: object) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.network.state_dict().items()
    }


def _save_checkpoint(
    run_dir: Path,
    *,
    current: torch.Tensor,
    network_state: dict[str, torch.Tensor] | None,
    outer_steps_completed: int,
    p_scale: torch.Tensor | None,
    r_scale: torch.Tensor | None,
) -> None:
    torch.save(
        {
            "current_variables": current.detach().cpu(),
            "network_state_dict": network_state,
            "outer_steps_completed": int(outer_steps_completed),
            "p_scale": None if p_scale is None else p_scale.detach().cpu(),
            "r_scale": None if r_scale is None else r_scale.detach().cpu(),
        },
        run_dir / "checkpoint.pt",
    )


def run(args: argparse.Namespace) -> None:
    if int(args.max_outer_steps) < 1:
        raise ValueError("max_outer_steps must be positive")
    if float(args.ce_relative_tolerance) <= 0.0:
        raise ValueError("ce_relative_tolerance must be positive")
    if float(args.fixed_point_tolerance) <= 0.0:
        raise ValueError("fixed_point_tolerance must be positive")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    problem, exact_optimum, oracle_final, exact_report = _load_problem(args)
    fit_config = FrozenExactCEFitConfig(
        steps=int(args.fit_max_steps),
        batch_size=int(args.fit_batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.fit_log_every),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        minimum_scale=1e-3,
        minimum_steps=int(args.fit_min_steps),
        early_stop_relative_error=float(args.ce_relative_tolerance),
    )
    reference = problem.reference_start()
    current = reference.detach().clone()
    reference_objective = _objective(problem, reference)
    exact_objective = _objective(problem, exact_optimum)
    oracle_objective = _objective(problem, oracle_final)
    network_state: dict[str, torch.Tensor] | None = None
    last_p_scale: torch.Tensor | None = None
    last_r_scale: torch.Tensor | None = None
    history: list[dict[str, float]] = []
    status = "maximum_outer_steps"
    started = time.perf_counter()

    for outer in range(int(args.max_outer_steps)):
        exact_response = exact_garch_tree_mp_response(problem, current)
        exact_residual_before = float(
            exact_response.metrics["response_log_sigma_rms"]
        )
        objective_before = _objective(problem, current)
        if exact_residual_before <= float(args.fixed_point_tolerance):
            status = "fixed_point_tolerance"
            break

        fit = fit_frozen_exact_ce_gru(
            problem,
            current,
            config=fit_config,
            initial_network_state=network_state,
        )
        fit_metrics = frozen_ce_fit_metrics(fit)
        network_state = _cpu_state_dict(fit.model)
        last_p_scale = fit.p_scale
        last_r_scale = fit.r_scale
        ce_passed = (
            fit_metrics["p_relative_error"] <= float(args.ce_relative_tolerance)
            and fit_metrics["r_relative_error"] <= float(args.ce_relative_tolerance)
        )
        learned_controls = problem.control_map.controls_from_moments(
            paths=fit.paths.to(problem.device, problem.dtype),
            expected_adjoint_next=fit.prediction_p.to(problem.device, problem.dtype),
            expected_adjoint_noise_next=fit.prediction_r.to(problem.device, problem.dtype),
        )
        learned_proposal, learned_consensus = collapse_leaf_controls_to_variables(
            problem, learned_controls
        )
        exact_proposal = exact_response.variables
        _, exact_gate = objective_gated_exact_ce_update(
            problem, current, exact_proposal
        )
        learned_candidate, learned_gate = objective_gated_exact_ce_update(
            problem, current, learned_proposal
        )
        oracle_step_gain = float(exact_gate["objective_improvement"])
        learned_step_gain = float(learned_gate["objective_improvement"])
        update_applied = ce_passed and float(learned_gate["accepted"]) == 1.0
        row = {
            "outer_step": float(outer),
            "objective_before": objective_before,
            "exact_response_residual_before": exact_residual_before,
            "ce_gate_passed": float(ce_passed),
            "fit_selected_step": float(fit.selected_step),
            "fit_selected_loss": float(fit.selected_loss),
            "p_relative_error": float(fit_metrics["p_relative_error"]),
            "r_relative_error": float(fit_metrics["r_relative_error"]),
            "p_correlation": float(fit_metrics["p_correlation"]),
            "r_correlation": float(fit_metrics["r_correlation"]),
            "p_amplitude_ratio": float(fit_metrics["p_amplitude_ratio"]),
            "r_amplitude_ratio": float(fit_metrics["r_amplitude_ratio"]),
            **_direction_metrics(current, learned_proposal, exact_proposal),
            "learned_node_consensus_max_abs": float(learned_consensus),
            "exact_gate_relaxation": float(exact_gate["relaxation"]),
            "learned_gate_accepted": float(learned_gate["accepted"]),
            "update_applied": float(update_applied),
            "learned_gate_relaxation": float(learned_gate["relaxation"]),
            "exact_one_step_objective_gain": oracle_step_gain,
            "learned_one_step_objective_gain": learned_step_gain,
            "learned_fraction_of_exact_one_step_gain": (
                learned_step_gain / oracle_step_gain
                if oracle_step_gain > 1e-30
                else math.nan
            ),
            "objective_after": float(learned_gate["objective_after"]),
            "fit_evaluations": float(len(fit.history)),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if not ce_passed:
            status = "ce_accuracy_gate_failed"
            break
        if float(learned_gate["accepted"]) != 1.0:
            status = "objective_gate_rejected"
            break
        current = learned_candidate.detach()
        _save_checkpoint(
            run_dir,
            current=current,
            network_state=network_state,
            outer_steps_completed=outer + 1,
            p_scale=last_p_scale,
            r_scale=last_r_scale,
        )
        write_json(
            run_dir / "progress.json",
            {
                "status": "running",
                "history": history,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )

    final_response = exact_garch_tree_mp_response(problem, current)
    final_objective = _objective(problem, current)
    denominator = reference_objective - exact_objective
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )
    final_errors = garch_tree_metric_errors(
        problem,
        problem.solution(current),
        swd_projections=int(args.swd_projections),
    )
    closure = gap_closed(reference_errors, final_errors)
    report: dict[str, object] = {
        "status": status,
        "protocol": {
            "solver": "two-timescale frozen-law GRU fit followed by one objective-gated update",
            "conditional_targets": "exact nodewise enumeration",
            "network_warm_start": True,
            "optimizer_reset_each_outer_step": True,
            "control": "volatility_only",
            "physical_drift": "identically_zero",
            "fit_config": asdict(fit_config),
            "fixed_point_tolerance": float(args.fixed_point_tolerance),
            "underlying_exact_protocol": exact_report["protocol"],
        },
        "outer_steps_completed": int(sum(row["update_applied"] for row in history)),
        "history": history,
        "objective": {
            "reference": reference_objective,
            "final": final_objective,
            "oracle_exact_ce_trajectory_final": oracle_objective,
            "exact_nodewise_optimum": exact_objective,
            "fraction_of_exact_objective_gain": (
                (reference_objective - final_objective) / denominator
                if denominator > 1e-30
                else math.nan
            ),
            "remaining_gap_to_exact_optimum": final_objective - exact_objective,
        },
        "stationarity": {
            "final_exact_ce_log_sigma_response_rms": float(
                final_response.metrics["response_log_sigma_rms"]
            ),
            "distance_to_exact_optimum_log_sigma_rms": float(
                (current - exact_optimum).double().pow(2).mean().sqrt().item()
            ),
            "distance_to_oracle_trajectory_log_sigma_rms": float(
                (current - oracle_final).double().pow(2).mean().sqrt().item()
            ),
        },
        "path_law": {
            "errors": final_errors,
            "gap_closed": closure,
            "distribution": summarize_gap_closure(
                closure, GARCH_DISTRIBUTION_METRICS
            ),
            "dependence": summarize_gap_closure(
                closure, GARCH_DEPENDENCE_METRICS
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    write_json(run_dir / "report.json", report)
    _save_checkpoint(
        run_dir,
        current=current,
        network_state=network_state,
        outer_steps_completed=int(report["outer_steps_completed"]),
        p_scale=last_p_scale,
        r_scale=last_r_scale,
    )
    summary = (
        "# Exact-tree two-timescale Deep MKV loop\n\n"
        f"- Status: `{status}`\n"
        f"- Accepted outer updates: {report['outer_steps_completed']}\n"
        f"- Objective: {reference_objective:.8g} -> {final_objective:.8g}\n"
        f"- Exact optimum: {exact_objective:.8g}\n"
        f"- Exact objective gain captured: {report['objective']['fraction_of_exact_objective_gain']:.6%}\n"
        f"- Final exact-CE response residual: {report['stationarity']['final_exact_ce_log_sigma_response_rms']:.8g}\n"
        f"- Distribution metrics improved: {report['path_law']['distribution']['improved_count']}/7\n"
        f"- Dependence metrics improved: {report['path_law']['dependence']['improved_count']}/4\n"
        f"- Runtime: {report['runtime_seconds']:.1f} seconds\n"
    )
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
