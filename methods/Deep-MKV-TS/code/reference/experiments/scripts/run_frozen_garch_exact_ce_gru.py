#!/usr/bin/env python3
"""Fit the production adjoint GRU to exact CEs on one frozen GARCH law."""

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
    reconstruct_exact_ce_law,
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
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", choices=("reference", "intermediate", "near"), required=True)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--trajectory-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--minimum-scale", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intermediate-updates", type=int, default=3)
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


def _objective(problem: ProductionGarchTreeProblem, point: torch.Tensor) -> float:
    return float(problem.objective(point).detach().cpu().item())


def _law_variables(
    args: argparse.Namespace,
    problem: ProductionGarchTreeProblem,
) -> tuple[torch.Tensor, dict[str, object]]:
    if args.law == "reference":
        return problem.reference_start(), {"source": "reference_start"}
    if args.law == "intermediate":
        point, history = reconstruct_exact_ce_law(
            problem, accepted_updates=int(args.intermediate_updates)
        )
        return point, {
            "source": "reconstructed_exact_ce_trajectory",
            "accepted_updates": int(args.intermediate_updates),
            "trajectory": list(history),
        }
    payload = torch.load(
        args.trajectory_run_dir.resolve() / "solutions.pt",
        map_location=problem.device,
        weights_only=True,
    )
    return payload["final_variables"].to(problem.device, problem.dtype), {
        "source": "locked_exact_ce_trajectory_final",
        "trajectory_run_dir": str(args.trajectory_run_dir.resolve()),
    }


def _vector_comparison(
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
        "learned_vs_exact_proposal_log_sigma_rms": float(
            (learned.double() - exact.double()).pow(2).mean().sqrt().item()
        ),
        "learned_direction_cosine_with_exact": float(cosine.item()),
        "learned_direction_amplitude_ratio": float((learned_norm / exact_norm).item()),
        "exact_response_log_sigma_rms": float(
            exact_direction.pow(2).mean().sqrt().item()
        ),
        "learned_response_log_sigma_rms": float(
            learned_direction.pow(2).mean().sqrt().item()
        ),
    }


def _state_errors(
    problem: ProductionGarchTreeProblem,
    point: torch.Tensor,
    *,
    reference_errors: dict[str, float],
    projections: int,
) -> dict[str, object]:
    errors = garch_tree_metric_errors(
        problem,
        problem.solution(point),
        swd_projections=int(projections),
    )
    closure = gap_closed(reference_errors, errors)
    return {
        "errors": errors,
        "gap_closed": closure,
        "distribution_gap_closed": summarize_gap_closure(
            closure, GARCH_DISTRIBUTION_METRICS
        ),
        "dependence_gap_closed": summarize_gap_closure(
            closure, GARCH_DEPENDENCE_METRICS
        ),
    }


def run(args: argparse.Namespace) -> None:
    run_dir = ensure_run_dir(args.run_dir.resolve())
    problem, exact_optimum, exact_report = _load_problem(args)
    current, law_source = _law_variables(args, problem)
    reference = problem.reference_start()
    reference_objective = _objective(problem, reference)
    exact_optimum_objective = _objective(problem, exact_optimum)
    current_objective = _objective(problem, current)
    fit_config = FrozenExactCEFitConfig(
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        minimum_scale=float(args.minimum_scale),
    )

    started = time.perf_counter()
    fit = fit_frozen_exact_ce_gru(problem, current, config=fit_config)
    response = exact_garch_tree_mp_response(problem, current)
    learned_controls = problem.control_map.controls_from_moments(
        paths=fit.paths.to(problem.device, problem.dtype),
        expected_adjoint_next=fit.prediction_p.to(problem.device, problem.dtype),
        expected_adjoint_noise_next=fit.prediction_r.to(problem.device, problem.dtype),
    )
    learned_proposal, learned_consensus = collapse_leaf_controls_to_variables(
        problem, learned_controls
    )
    exact_proposal, exact_consensus = collapse_leaf_controls_to_variables(
        problem, response.controls
    )
    exact_candidate, exact_gate = objective_gated_exact_ce_update(
        problem, current, exact_proposal
    )
    learned_candidate, learned_gate = objective_gated_exact_ce_update(
        problem, current, learned_proposal
    )
    oracle_improvement = float(exact_gate["objective_improvement"])
    learned_improvement = float(learned_gate["objective_improvement"])
    gain_fraction = (
        learned_improvement / oracle_improvement
        if oracle_improvement > 1e-30
        else math.nan
    )
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(reference),
        swd_projections=int(args.swd_projections),
    )

    report: dict[str, object] = {
        "status": "complete",
        "law": args.law,
        "law_source": law_source,
        "protocol": {
            "purpose": "production-GRU representation audit on a frozen exact tree law",
            "conditional_targets": "exact nodewise enumeration",
            "law_refresh_during_fit": False,
            "deployment": "production volatility-only Hamiltonian map",
            "selection": "lowest full-tree normalized P-plus-R MSE",
            "fit_config": asdict(fit_config),
            "exact_run_dir": str(args.exact_run_dir.resolve()),
            "underlying_exact_protocol": exact_report["protocol"],
        },
        "objectives": {
            "reference": reference_objective,
            "current_frozen_law": current_objective,
            "exact_global_optimum": exact_optimum_objective,
            "exact_raw_proposal": _objective(problem, exact_proposal),
            "learned_raw_proposal": _objective(problem, learned_proposal),
            "exact_gated": exact_gate,
            "learned_gated": learned_gate,
            "learned_fraction_of_exact_one_step_improvement": gain_fraction,
        },
        "conditional_expectation_fit": frozen_ce_fit_metrics(fit),
        "control_comparison": {
            **_vector_comparison(current, learned_proposal, exact_proposal),
            "learned_node_consensus_max_abs": float(learned_consensus),
            "exact_node_consensus_max_abs": float(exact_consensus),
        },
        "path_law": {
            "current": _state_errors(
                problem,
                current,
                reference_errors=reference_errors,
                projections=int(args.swd_projections),
            ),
            "exact_gated": _state_errors(
                problem,
                exact_candidate,
                reference_errors=reference_errors,
                projections=int(args.swd_projections),
            ),
            "learned_gated": _state_errors(
                problem,
                learned_candidate,
                reference_errors=reference_errors,
                projections=int(args.swd_projections),
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "fingerprint": fit.fingerprint,
    }
    write_json(run_dir / "report.json", report)
    write_history_jsonl(run_dir / "history.jsonl", list(fit.history))
    torch.save(
        {
            "network_state_dict": fit.model.network.state_dict(),
            "p_scale": fit.p_scale.cpu(),
            "r_scale": fit.r_scale.cpu(),
            "current_variables": current.detach().cpu(),
            "exact_proposal_variables": exact_proposal.detach().cpu(),
            "learned_proposal_variables": learned_proposal.detach().cpu(),
            "exact_gated_variables": exact_candidate.detach().cpu(),
            "learned_gated_variables": learned_candidate.detach().cpu(),
        },
        run_dir / "checkpoint.pt",
    )
    p = report["conditional_expectation_fit"]
    o = report["objectives"]
    c = report["control_comparison"]
    summary = (
        f"# Frozen exact-CE GRU audit: {args.law}\n\n"
        f"The production GRU was fitted on exact conditional-adjoint labels while "
        f"the generated law was held fixed. No law refresh occurred during training.\n\n"
        f"| Quantity | Result |\n|---|---:|\n"
        f"| Selected step | {int(p['selected_step'])} |\n"
        f"| P relative error | {p['p_relative_error']:.6f} |\n"
        f"| R relative error | {p['r_relative_error']:.6f} |\n"
        f"| P explained energy | {p['p_explained_energy']:.6f} |\n"
        f"| R explained energy | {p['r_explained_energy']:.6f} |\n"
        f"| R correlation | {p['r_correlation']:.6f} |\n"
        f"| R amplitude ratio | {p['r_amplitude_ratio']:.6f} |\n"
        f"| Learned/exact control-direction cosine | {c['learned_direction_cosine_with_exact']:.6f} |\n"
        f"| Learned/exact direction amplitude | {c['learned_direction_amplitude_ratio']:.6f} |\n"
        f"| Exact one-step objective improvement | {o['exact_gated']['objective_improvement']:.8g} |\n"
        f"| Learned one-step objective improvement | {o['learned_gated']['objective_improvement']:.8g} |\n"
        f"| Learned fraction of exact one-step gain | {o['learned_fraction_of_exact_one_step_improvement']:.6f} |\n"
    )
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
