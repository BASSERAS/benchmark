#!/usr/bin/env python3
"""Run the single permitted residual-Newton cycle for volatility-only MP."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    DirectRidgeRAdjointPolicy,
)
from path_dt_experiments.extragradient_solver import _sample_inputs  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    ScenarioDataContext,
    build_fit_confirmation_banks,
    rollout_causal_policy,
    rollout_policy_on_bank,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    _time_blocks,
)
from path_dt_experiments.volatility_only_solver import (  # noqa: E402
    VolatilityOnlyBlockwiseRPolicy,
)
from run_direct_ridge_r_spectral_matrix import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_volatility_only_first_operator_audit import (  # noqa: E402
    _restrict_to_volatility,
)
from run_volatility_only_r_spectral_matrix import (  # noqa: E402
    DEFAULT_FIRST_OPERATOR_RUN,
    _operator_coordinates_r,
)


DEFAULT_SPECTRAL_RUN = REPO_ROOT / "runs" / "volatility_only_r_spectral_20260803"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spectral-run-dir", type=Path, default=DEFAULT_SPECTRAL_RUN)
    parser.add_argument(
        "--first-operator-run-dir", type=Path, default=DEFAULT_FIRST_OPERATOR_RUN
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-pool-paths", type=int, default=3072)
    parser.add_argument("--operator-fit-paths", type=int, default=512)
    parser.add_argument("--operator-validation-paths", type=int, default=512)
    parser.add_argument("--operator-target-paths", type=int, default=1024)
    parser.add_argument("--operator-replicates", type=int, default=3)
    parser.add_argument("--audit-paths", type=int, default=128)
    parser.add_argument("--audit-target-paths", type=int, default=512)
    parser.add_argument("--audit-branches", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--lm-relative-cutoff", type=float, default=0.01)
    parser.add_argument("--trust-paths", type=int, default=512)
    parser.add_argument("--mean-kl-budget", type=float, default=0.02)
    parser.add_argument("--q95-kl-budget", type=float, default=0.10)
    parser.add_argument("--max-kl-budget", type=float, default=1.50)
    parser.add_argument("--maximum-objective-relative-increase", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2_609_001)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        dataset_root=args.dataset_root,
        signal_run_dir=args.signal_run_dir,
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )


def _coordinate_residual(theta: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    residual = theta.double() - target.double()
    return {
        "theta": theta.tolist(),
        "target": target.tolist(),
        "residual": residual.tolist(),
        "norm": float(torch.linalg.vector_norm(residual).item()),
    }


def _nested_metrics(*, model, evaluation) -> dict[str, object]:
    target_controls = model.control_map.controls_from_moments(
        paths=evaluation.paths,
        expected_adjoint_next=evaluation.target_p,
        expected_adjoint_noise_next=evaluation.target_r,
    )
    state_dim = int(evaluation.paths.shape[-1])
    alpha = evaluation.controls[..., :state_dim]
    sigma_residual = (
        torch.log(evaluation.controls[..., state_dim:])
        - torch.log(target_controls[..., state_dim:])
    )
    r_residual = evaluation.policy_r - evaluation.target_r
    p_residual = evaluation.policy_p - evaluation.target_p

    def rms(value: torch.Tensor) -> float:
        return float(torch.mean(value.detach().double().pow(2)).sqrt().item())

    return {
        "objective": evaluation.objective,
        "safety": evaluation.safety,
        "R_residual_rms": rms(r_residual),
        "R_relative_residual": rms(r_residual) / max(rms(evaluation.target_r), 1e-12),
        "P_diagnostic_residual_rms": rms(p_residual),
        "log_volatility_target_consensus_rms": rms(sigma_residual),
        "maximum_absolute_drift": float(alpha.detach().abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    spectral = json.loads(
        (args.spectral_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    if spectral["gate"]["next_phase"] != "one_residual_newton_cycle":
        raise RuntimeError("the locked spectrum did not authorize a Newton cycle")
    model, paths = _construct_model(_model_args(args))
    _restrict_to_volatility(model)
    checkpoint_path = (
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=model.device, weights_only=False
    )
    p_network = copy.deepcopy(model.network)
    p_network.load_state_dict(checkpoint["state_dict"])
    p_network.eval()
    projection_path = (
        args.first_operator_run_dir.resolve() / "residual_aware_ridge_r.pt"
    )
    projection = torch.load(projection_path, map_location="cpu", weights_only=False)
    candidate = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=projection
    )
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    dimension = len(blocks)
    fit_pool = paths[: int(args.fit_pool_paths)]
    validation_pool = paths[int(args.fit_pool_paths) :]
    jacobian_t = torch.tensor(spectral["mean_matrix"], dtype=torch.float64)

    t0_rows = []
    for index in range(int(args.operator_replicates)):
        print(f"T(0) replicate {index + 1}/{args.operator_replicates}", flush=True)
        row, _ = _operator_coordinates_r(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            candidate_policy=candidate,
            theta_r=torch.zeros(dimension, dtype=torch.float64),
            blocks=blocks,
            fit_count=int(args.operator_fit_paths),
            validation_count=int(args.operator_validation_paths),
            target_count=int(args.operator_target_paths),
            seed=int(args.seed) + index * 10_000,
        )
        t0_rows.append(row)
    t0_stack = torch.stack(t0_rows)
    t0 = t0_stack.mean(dim=0)
    jacobian_f = torch.eye(dimension, dtype=torch.float64) - jacobian_t
    u, singular_values, vh = torch.linalg.svd(jacobian_f)
    regularization = float(args.lm_relative_cutoff) * float(singular_values.max().item())
    direction = vh.transpose(0, 1) @ (
        singular_values
        / (singular_values.pow(2) + regularization * regularization)
        * (u.transpose(0, 1) @ t0)
    )
    predicted = -t0 + jacobian_f @ direction
    predicted_ratio = float(
        torch.linalg.vector_norm(predicted)
        / torch.linalg.vector_norm(t0).clamp_min(torch.finfo(torch.float64).eps)
    )
    base_policy = VolatilityOnlyBlockwiseRPolicy(
        model=model,
        candidate_policy=candidate,
        theta_r=torch.zeros(dimension, dtype=torch.float64),
        blocks=blocks,
    )
    trust_x0, trust_noise = _sample_inputs(
        model=model,
        initial_state_pool=validation_pool,
        count=int(args.trust_paths),
        seed=int(args.seed) + 50_000,
    )
    with torch.no_grad():
        base_law = rollout_causal_policy(
            model=model, policy=base_policy, x0=trust_x0, noise=trust_noise
        )

    def trust(scale: float) -> tuple[bool, dict[str, float]]:
        theta = float(scale) * direction
        policy = VolatilityOnlyBlockwiseRPolicy(
            model=model, candidate_policy=candidate, theta_r=theta, blocks=blocks
        )
        with torch.no_grad():
            law = rollout_causal_policy(
                model=model, policy=policy, x0=trust_x0, noise=trust_noise
            )
        kl = model.control_map.transition_kernel_kl(
            controls=law.controls, current_controls=base_law.controls
        ).detach()
        metrics = {
            "scale": float(scale),
            "mean_kl": float(kl.mean().item()),
            "q95_kl": float(torch.quantile(kl.reshape(-1), 0.95).item()),
            "max_kl": float(kl.max().item()),
            "maximum_absolute_drift": float(
                law.controls[..., : int(model.architecture.state_dim)]
                .abs()
                .max()
                .item()
            ),
            "controls_finite": float(bool(torch.isfinite(law.controls).all().item())),
        }
        passed = bool(
            metrics["controls_finite"] == 1.0
            and metrics["maximum_absolute_drift"] == 0.0
            and metrics["mean_kl"] <= float(args.mean_kl_budget)
            and metrics["q95_kl"] <= float(args.q95_kl_budget)
            and metrics["max_kl"] <= float(args.max_kl_budget)
        )
        return passed, metrics

    full_passed, trust_metrics = trust(1.0)
    if full_passed:
        accepted_scale = 1.0
    else:
        lower, upper = 0.0, 1.0
        trust_metrics = trust(0.0)[1]
        for _ in range(24):
            midpoint = 0.5 * (lower + upper)
            passed, metrics = trust(midpoint)
            if passed:
                lower, trust_metrics = midpoint, metrics
            else:
                upper = midpoint
        accepted_scale = lower
    theta = float(accepted_scale) * direction
    newton_policy = VolatilityOnlyBlockwiseRPolicy(
        model=model, candidate_policy=candidate, theta_r=theta, blocks=blocks
    )
    print(
        f"Newton direction={direction.tolist()}, KL scale={accepted_scale:.6g}, "
        f"theta={theta.tolist()}",
        flush=True,
    )

    projected_rows = []
    for index in range(int(args.operator_replicates)):
        seed = int(args.seed) + 100_000 + index * 10_000
        print(f"fresh-projection acceptance replicate {index + 1}", flush=True)
        before_target, _ = _operator_coordinates_r(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            candidate_policy=candidate,
            theta_r=torch.zeros_like(theta),
            blocks=blocks,
            fit_count=int(args.operator_fit_paths),
            validation_count=int(args.operator_validation_paths),
            target_count=int(args.operator_target_paths),
            seed=seed,
        )
        after_target, _ = _operator_coordinates_r(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            candidate_policy=candidate,
            theta_r=theta,
            blocks=blocks,
            fit_count=int(args.operator_fit_paths),
            validation_count=int(args.operator_validation_paths),
            target_count=int(args.operator_target_paths),
            seed=seed,
        )
        before = _coordinate_residual(torch.zeros_like(theta), before_target)
        after = _coordinate_residual(theta, after_target)
        projected_rows.append(
            {
                "seed": seed,
                "before": before,
                "after": after,
                "relative_improvement": (
                    float(before["norm"]) - float(after["norm"])
                )
                / max(float(before["norm"]), 1e-12),
            }
        )

    context = ScenarioDataContext(
        initial_state_pool=paths.detach().cpu(),
        target_path_pool=paths.detach().cpu(),
        dataset_id=json.dumps(
            {
                "path": str(args.dataset_root.resolve()),
                "split": "train.npy deterministic prefix",
                "count": int(args.calibration_paths),
                "admissible_control": "alpha=0, sigma bounded",
            },
            sort_keys=True,
        ),
    )
    fit_bank, confirmation_bank = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 200_000,
    )
    nested = {}
    for label, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        print(f"complete nested audit {label}", flush=True)
        before_evaluation = rollout_policy_on_bank(
            model=model,
            policy=base_policy,
            bank=bank,
            data_context=context,
            full_training=model.training,
            query_batch_size=int(args.query_batch_size),
        )
        after_evaluation = rollout_policy_on_bank(
            model=model,
            policy=newton_policy,
            bank=bank,
            data_context=context,
            full_training=model.training,
            query_batch_size=int(args.query_batch_size),
        )
        nested[label] = {
            "before": _nested_metrics(model=model, evaluation=before_evaluation),
            "after": _nested_metrics(model=model, evaluation=after_evaluation),
        }

    checks = {
        f"fresh_projection_replicate_{index}_residual_reduced": (
            float(row["after"]["norm"]) < float(row["before"]["norm"])
        )
        for index, row in enumerate(projected_rows)
    }
    for label, row in nested.items():
        before, after = row["before"], row["after"]
        checks[f"{label}_R_residual_reduced"] = (
            float(after["R_residual_rms"]) < float(before["R_residual_rms"])
        )
        checks[f"{label}_volatility_consensus_reduced"] = (
            float(after["log_volatility_target_consensus_rms"])
            < float(before["log_volatility_target_consensus_rms"])
        )
        initial_objective = float(before["objective"]["complete_objective_value"])
        terminal_objective = float(after["objective"]["complete_objective_value"])
        checks[f"{label}_objective_safe"] = (
            terminal_objective
            <= initial_objective
            + float(args.maximum_objective_relative_increase)
            * max(abs(initial_objective), 1e-12)
        )
        checks[f"{label}_drift_exactly_zero"] = (
            float(after["maximum_absolute_drift"]) == 0.0
        )
        checks[f"{label}_controls_safe"] = (
            float(after["safety"]["controls_finite"]) == 1.0
        )
    checks["nontrivial_newton_step"] = (
        float(torch.linalg.vector_norm(theta).item()) >= 1e-4
    )
    gate_passed = all(checks.values())
    gate = {
        "checks": checks,
        "passed": gate_passed,
        "next_phase": (
            "terminal_cycle_succeeded"
            if gate_passed
            else "stop_solver_development_current_formulation"
        ),
    }
    report = {
        "protocol": {
            "single_permitted_residual_newton_cycle": True,
            "alpha_identically_zero": True,
            "P_regressed_but_excluded_from_control": True,
            "fresh_direct_R_projection_in_each_operator_evaluation": True,
            "no_additional_solver_variant_if_gate_fails": True,
            "test_paths_used": False,
        },
        "linear_system": {
            "T0_replicates": t0_stack.tolist(),
            "T0_mean": t0.tolist(),
            "T0_sd": t0_stack.std(dim=0, unbiased=True).tolist(),
            "jacobian_T": jacobian_t.tolist(),
            "singular_values_I_minus_JT": singular_values.tolist(),
            "condition_number_I_minus_JT": float(
                (singular_values.max() / singular_values.min()).item()
            ),
            "LM_regularization": regularization,
            "unscaled_direction": direction.tolist(),
            "predicted_residual_ratio": predicted_ratio,
        },
        "trust_region": {
            "accepted_scale": float(accepted_scale),
            "accepted_theta": theta.tolist(),
            "metrics": trust_metrics,
            "budgets": {
                "mean_kl": float(args.mean_kl_budget),
                "q95_kl": float(args.q95_kl_budget),
                "max_kl": float(args.max_kl_budget),
            },
        },
        "fresh_projection_residuals": projected_rows,
        "nested_fit_confirmation": nested,
        "scenario_banks": {
            "fit": fit_bank.fingerprint,
            "confirmation": confirmation_bank.fingerprint,
        },
        "gate": gate,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "jacobian_T": jacobian_t,
            "T0_replicates": t0_stack,
            "direction": direction,
            "theta": theta,
        },
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Terminal volatility-only residual-Newton cycle\n\n"
        f"- T(0) mean: `{t0.tolist()}`\n"
        f"- Unscaled Newton direction: `{direction.tolist()}`\n"
        f"- Accepted scale: `{accepted_scale:.6g}`\n"
        f"- Accepted coordinates: `{theta.tolist()}`\n"
        f"- Predicted residual ratio: `{predicted_ratio:.3%}`\n"
        f"- Fresh-projection residual improvements: "
        f"`{[round(row['relative_improvement'], 4) for row in projected_rows]}`\n"
        f"- Gate passed: `{gate_passed}`\n"
        f"- Next phase: `{gate['next_phase']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
