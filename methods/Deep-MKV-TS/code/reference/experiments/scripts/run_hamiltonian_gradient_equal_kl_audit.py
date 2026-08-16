#!/usr/bin/env python3
"""Equal-KL audit of full, raw-gradient, and KL-mirror MP moves."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.hamiltonian_gradient_audit import (  # noqa: E402
    EqualKLBudgetConfig,
    HamiltonianDirectionPolicy,
    audit_direction_evaluation,
    calibrate_direction_to_kl_budget,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
    rollout_policy_on_bank,
    time_block_bounds,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_lifted_collocation_probe import (  # noqa: E402
    DEFAULT_ADJOINT_ARTIFACTS,
    _load_adjoint_network,
)
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


DEFAULT_BUDGET_PROVENANCE = (
    REPO_ROOT
    / "runs"
    / "lifted_multiradius_derivative_audit_seed0_20260803"
    / "signed_derivative_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS)
    parser.add_argument("--budget-provenance", type=Path, default=DEFAULT_BUDGET_PROVENANCE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_121)
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--target-paths", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4_096)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "raw", "kl"),
        default=("full", "raw", "kl"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _provenance_budgets(payload: dict[str, object]) -> tuple[float, ...]:
    surfaces = payload.get("surfaces")
    if not isinstance(surfaces, dict) or not isinstance(surfaces.get("fit"), dict):
        raise ValueError("budget provenance is missing fit surfaces")
    fit = surfaces["fit"]
    rows = []
    for epsilon in ("0.00625", "0.0125", "0.025"):
        row = fit[epsilon]["log_sigma_block_2"]["plus"]
        rows.append(float(row["law_transition_kl_mean"]))
    return tuple(rows)


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Hamiltonian-gradient equal-KL audit",
        "",
        "No candidate was deployed and no adjoint was refit.",
        "",
        f"- Methods: `{report['methods']}`",
        f"- Mean-KL budgets: `{report['config']['mean_budgets']}`",
        f"- Fit/confirmation bank fingerprints differ: `{report['fit_bank_fingerprint'] != report['confirmation_bank_fingerprint']}`",
        "",
        "| Method | Coordinate | Budget | Fit dJ | Confirm dJ | Fit refreshed vol KKT | Confirm refreshed vol KKT | Fit KL | Confirm KL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        for row in report["results"][method]:
            coordinate = f"{row['signal']}_{row['block_index']}"
            lines.append(
                f"| {method} | {coordinate} | {row['target_mean_kl']:.3g} | "
                f"{row['fit']['objective_change']:.4g} | "
                f"{row['confirmation']['objective_change']:.4g} | "
                f"{row['fit']['refreshed_target_volatility_kkt_rms']:.4g} | "
                f"{row['confirmation']['refreshed_target_volatility_kkt_rms']:.4g} | "
                f"{row['fit']['transition_kl_mean']:.3g} | "
                f"{row['confirmation']['transition_kl_mean']:.3g} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if int(args.paths) < 1 or int(args.target_paths) < 1:
        raise ValueError("paths and target-paths must be positive")
    if int(args.branches) < 2 or int(args.branches) % 2 != 0:
        raise ValueError("branches must be an even integer of at least two")
    if int(args.query_batch_size) < int(args.paths) * int(args.branches):
        raise ValueError("query-batch-size must hold one complete branch population")
    methods = tuple(dict.fromkeys(str(value) for value in args.methods))
    continuation = args.continuation_run_dir.resolve()
    adjoint_artifacts = args.adjoint_artifacts.resolve()
    budget_provenance_path = args.budget_provenance.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    provenance = _load_json(budget_provenance_path)
    budgets = _provenance_budgets(provenance)
    config = EqualKLBudgetConfig(mean_budgets=budgets)

    model, target_pool, checkpoint_training, continuation_report, design = (
        _build_frozen_model(
            continuation=continuation,
            device=torch.device(args.device),
            model_seed=int(args.model_seed),
        )
    )
    stages = continuation_report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("continuation report has no stages")
    terminal = stages[-1]
    operator_beta = float(terminal["beta_target"])
    full_lambda_scale = float(continuation_report["final_lambda_scale"])
    full_training = replace(
        checkpoint_training,
        lambda_scale=full_lambda_scale,
        grad_clip_norm=None,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
    )
    dataset = design.get("dataset")
    context = ScenarioDataContext(
        initial_state_pool=target_pool.detach().cpu(),
        target_path_pool=target_pool.detach().cpu(),
        dataset_id=(
            json.dumps(dataset, sort_keys=True)
            if isinstance(dataset, dict)
            else str(dataset)
        ),
    )
    adjoint_network = _load_adjoint_network(model=model, artifacts_path=adjoint_artifacts)
    model.network.eval()
    actor_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    critic_policy = AdjointNetworkControlPolicy(model=model, network=adjoint_network)
    model_hash = model_checkpoint_sha256(model)
    actor_hash = network_sha256(model.network)
    critic_hash = network_sha256(adjoint_network)
    fit_bank, confirmation_bank = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.paths),
        target_count=int(args.target_paths),
        num_branches=int(args.branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.seed),
    )
    baseline_evaluations = {}
    baseline_audits = {}
    baseline_artifacts = {}
    for bank_name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        print(f"building {bank_name} baseline", flush=True)
        evaluation = rollout_policy_on_bank(
            model=model,
            policy=actor_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        baseline_evaluations[bank_name] = evaluation
        audit, tensors = audit_direction_evaluation(
            model=model,
            evaluation=evaluation,
            actor_policy=actor_policy,
            critic_policy=critic_policy,
            baseline_objective=float(evaluation.objective["complete_objective_value"]),
        )
        baseline_audits[bank_name] = audit
        baseline_artifacts[bank_name] = tensors

    blocks = time_block_bounds(num_steps=int(model.grid.num_steps), num_blocks=3)
    results: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    artifacts: dict[str, object] = {
        "baseline": baseline_artifacts,
        "results": {method: {} for method in methods},
    }
    for method in methods:
        for signal in ("alpha", "log_sigma"):
            for block_index in range(len(blocks)):
                for budget_index, budget in enumerate(config.mean_budgets):
                    coordinate = f"{signal}_block_{block_index}_budget_{budget_index}"
                    print(f"calibrating {method} {coordinate}", flush=True)
                    parameter, calibration = calibrate_direction_to_kl_budget(
                        model=model,
                        actor_policy=actor_policy,
                        critic_policy=critic_policy,
                        data_context=context,
                        fit_bank=fit_bank,
                        method=method,
                        signal=signal,
                        block_index=block_index,
                        blocks=blocks,
                        budget=float(budget),
                        config=config,
                    )
                    policy = HamiltonianDirectionPolicy(
                        model=model,
                        actor_policy=actor_policy,
                        critic_policy=critic_policy,
                        method=method,
                        signal=signal,
                        block_index=block_index,
                        blocks=blocks,
                        parameter=parameter,
                    )
                    bank_rows = {}
                    for bank_name, bank in (
                        ("fit", fit_bank),
                        ("confirmation", confirmation_bank),
                    ):
                        print(f"evaluating {method} {coordinate} {bank_name}", flush=True)
                        evaluation = rollout_policy_on_bank(
                            model=model,
                            policy=policy,
                            bank=bank,
                            data_context=context,
                            full_training=full_training,
                            query_batch_size=int(args.query_batch_size),
                        )
                        row, tensors = audit_direction_evaluation(
                            model=model,
                            evaluation=evaluation,
                            actor_policy=actor_policy,
                            critic_policy=critic_policy,
                            baseline_objective=float(
                                baseline_evaluations[bank_name].objective[
                                    "complete_objective_value"
                                ]
                            ),
                        )
                        row["mean_budget_ratio"] = float(
                            row["transition_kl_mean"] / float(budget)
                        )
                        row["q95_ceiling_passed"] = bool(
                            float(row["transition_kl_q95"])
                            <= float(budget) * float(config.q95_multiplier)
                        )
                        row["maximum_ceiling_passed"] = bool(
                            float(row["transition_kl_max"])
                            <= float(budget) * float(config.maximum_multiplier)
                        )
                        bank_rows[bank_name] = row
                        artifacts["results"][method].setdefault(coordinate, {})[
                            bank_name
                        ] = tensors
                    results[method].append(
                        {
                            "method": method,
                            "signal": signal,
                            "block_index": block_index,
                            "budget_index": budget_index,
                            "target_mean_kl": float(budget),
                            "selected_parameter": parameter,
                            "calibration": calibration,
                            "fit": bank_rows["fit"],
                            "confirmation": bank_rows["confirmation"],
                        }
                    )

    if model_checkpoint_sha256(model) != model_hash:
        raise RuntimeError("equal-KL audit mutated the model")
    if network_sha256(model.network) != actor_hash:
        raise RuntimeError("equal-KL audit mutated the actor")
    if network_sha256(adjoint_network) != critic_hash:
        raise RuntimeError("equal-KL audit mutated the critic")
    report = {
        "protocol": {
            "diagnostic_only": True,
            "candidate_deployed": False,
            "adjoint_refit": False,
            "same_fitted_P_R_for_all_directions": True,
            "fit_bank_calibrates_parameter": True,
            "confirmation_parameter_rescaled": False,
            "exact_gaussian_transition_kl": True,
            "dual_deployed_and_refreshed_kkt": True,
            "test_paths_used": False,
        },
        "methods": list(methods),
        "config": dict(config.__dict__),
        "budget_provenance": str(budget_provenance_path),
        "operator_beta": operator_beta,
        "effective_lambda_scale": operator_beta * full_lambda_scale,
        "blocks": [list(block) for block in blocks],
        "baseline": baseline_audits,
        "results": results,
        "fit_bank_fingerprint": fit_bank.fingerprint,
        "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
        "nonmutation": {
            "model_unchanged": True,
            "actor_unchanged": True,
            "critic_unchanged": True,
        },
    }
    write_json(run_dir / "equal_kl_report.json", report)
    torch.save(artifacts, run_dir / "equal_kl_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    print(f"wrote equal-KL audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
