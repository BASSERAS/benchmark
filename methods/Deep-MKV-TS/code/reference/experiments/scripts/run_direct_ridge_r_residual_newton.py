#!/usr/bin/env python3
"""Take one locked simultaneous Newton step on the six-dimensional MP residual."""

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
    BlockwiseAdjointInterpolationPolicy,
    DirectRidgeRAdjointPolicy,
)
from path_dt_experiments.extragradient_solver import _sample_inputs  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    rollout_causal_policy,
    rollout_policy_on_bank,
)
from path_dt_experiments.terminal_fixed_point_diagnostic import _time_blocks  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_direct_ridge_r_spectral_matrix import _operator_coordinates  # noqa: E402
from run_residual_volatility_reduced_spectral_audit import _construct_model  # noqa: E402
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)


DEFAULT_DIRECT_RUN = REPO_ROOT / "runs" / "direct_ridge_r_audit_seed0_20260803"
DEFAULT_SPECTRAL_RUN = REPO_ROOT / "runs" / "direct_ridge_r_spectral_20260803"
DEFAULT_T0_ROOT = REPO_ROOT / "runs" / "direct_ridge_r_newton_20260803"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "runs" / "residual_volatility_first_operator_corrected_parts_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--direct-run-dir", type=Path, default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--spectral-run-dir", type=Path, default=DEFAULT_SPECTRAL_RUN)
    parser.add_argument("--t0-root", type=Path, default=DEFAULT_T0_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-pool-paths", type=int, default=3072)
    parser.add_argument("--operator-fit-paths", type=int, default=512)
    parser.add_argument("--operator-validation-paths", type=int, default=512)
    parser.add_argument("--operator-target-paths", type=int, default=1024)
    parser.add_argument("--audit-paths", type=int, default=64)
    parser.add_argument("--audit-target-paths", type=int, default=512)
    parser.add_argument("--audit-branches", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--lm-relative-cutoff", type=float, default=0.01)
    parser.add_argument("--trust-paths", type=int, default=256)
    parser.add_argument("--mean-kl-budget", type=float, default=0.02)
    parser.add_argument("--q95-kl-budget", type=float, default=0.10)
    parser.add_argument("--max-kl-budget", type=float, default=1.50)
    parser.add_argument("--maximum-objective-relative-increase", type=float, default=0.01)
    parser.add_argument("--confirmation-replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2_608_801)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _coordinate_residual(theta: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    residual = theta.double() - target.double()
    return {
        "theta": theta.tolist(),
        "target_coordinates": target.tolist(),
        "residual": residual.tolist(),
        "total_norm": float(torch.linalg.vector_norm(residual).item()),
        "p_norm": float(torch.linalg.vector_norm(residual[:3]).item()),
        "r_norm": float(torch.linalg.vector_norm(residual[3:]).item()),
    }


def _exact_bank_metrics(*, model, evaluation) -> dict[str, object]:
    target_controls = model.control_map.controls_from_moments(
        paths=evaluation.paths,
        expected_adjoint_next=evaluation.target_p,
        expected_adjoint_noise_next=evaluation.target_r,
    )
    state_dim = int(model.architecture.state_dim)
    drift = evaluation.controls[..., :state_dim] - target_controls[..., :state_dim]
    log_sigma = torch.log(evaluation.controls[..., state_dim:]) - torch.log(
        target_controls[..., state_dim:]
    )
    p = evaluation.policy_p - evaluation.target_p
    r = evaluation.policy_r - evaluation.target_r

    def rms(value: torch.Tensor) -> float:
        return float(torch.mean(value.detach().double().pow(2)).sqrt().item())

    drift_rms = rms(drift)
    log_sigma_rms = rms(log_sigma)
    return {
        "objective": evaluation.objective,
        "safety": evaluation.safety,
        "p_residual_rms": rms(p),
        "r_residual_rms": rms(r),
        "p_relative_residual": rms(p) / max(rms(evaluation.target_p), 1e-12),
        "r_relative_residual": rms(r) / max(rms(evaluation.target_r), 1e-12),
        "drift_target_control_consensus_rms": drift_rms,
        "log_volatility_target_control_consensus_rms": log_sigma_rms,
        "combined_target_control_consensus": math.sqrt(
            drift_rms * drift_rms + log_sigma_rms * log_sigma_rms
        ),
    }


def main() -> None:
    args = parse_args()
    if int(args.confirmation_replicates) < 1:
        raise ValueError("confirmation-replicates must be positive")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model_args = argparse.Namespace(
        device=args.device,
        dataset_root=args.dataset_root,
        signal_run_dir=args.signal_run_dir,
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )
    model, paths = _construct_model(model_args)
    checkpoint = torch.load(
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt",
        map_location=model.device,
        weights_only=False,
    )
    p_network = copy.deepcopy(model.network)
    p_network.load_state_dict(checkpoint["state_dict"])
    p_network.eval()
    for module in p_network.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters()
    projection = torch.load(
        args.direct_run_dir.resolve() / "residual_aware_ridge.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    candidate_policy = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=projection
    )
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    dimension = 2 * len(blocks)

    spectral = json.loads(
        (args.spectral_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    jacobian_t = torch.tensor(spectral["mean_matrix"], dtype=torch.float64)
    t0_rows = []
    t0_paths = []
    for index in range(4):
        path = args.t0_root.resolve() / f"t0_{index}" / "report.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        t0_rows.append(torch.tensor(row["coordinates"], dtype=torch.float64))
        t0_paths.append(str(path))
    t0_stack = torch.stack(t0_rows)
    t0 = t0_stack.mean(dim=0)
    identity = torch.eye(dimension, dtype=torch.float64)
    jacobian_f = identity - jacobian_t
    u, singular_values, vh = torch.linalg.svd(jacobian_f)
    regularization = float(args.lm_relative_cutoff) * float(singular_values.max().item())
    newton_direction = vh.transpose(0, 1) @ (
        singular_values
        / (singular_values.pow(2) + regularization * regularization)
        * (u.transpose(0, 1) @ t0)
    )
    predicted_residual = -t0 + jacobian_f @ newton_direction
    print(
        f"regularized Newton direction {newton_direction.tolist()} with predicted "
        f"residual ratio {torch.linalg.vector_norm(predicted_residual).item() / torch.linalg.vector_norm(t0).item():.4f}",
        flush=True,
    )

    fit_pool = paths[: int(args.fit_pool_paths)]
    validation_pool = paths[int(args.fit_pool_paths) :]
    trust_x0, trust_noise = _sample_inputs(
        model=model,
        initial_state_pool=validation_pool,
        count=int(args.trust_paths),
        seed=int(args.seed),
    )
    with torch.no_grad():
        trust_base = rollout_causal_policy(
            model=model, policy=base_policy, x0=trust_x0, noise=trust_noise
        )

    def trust(scale: float) -> tuple[bool, dict[str, float]]:
        theta = float(scale) * newton_direction
        policy = BlockwiseAdjointInterpolationPolicy(
            model=model,
            base_policy=base_policy,
            candidate_policy=candidate_policy,
            theta=theta,
            blocks=blocks,
        )
        with torch.no_grad():
            law = rollout_causal_policy(
                model=model, policy=policy, x0=trust_x0, noise=trust_noise
            )
        kl = model.control_map.transition_kernel_kl(
            controls=law.controls, current_controls=trust_base.controls
        ).detach()
        sigma = law.controls[..., int(model.architecture.state_dim) :]
        metrics = {
            "scale": float(scale),
            "mean_kl": float(kl.mean().item()),
            "q95_kl": float(torch.quantile(kl.reshape(-1), 0.95).item()),
            "max_kl": float(kl.max().item()),
            "sigma_lower_activity": float(
                torch.mean((sigma <= float(model.control_map.sigma_min) * (1.0 + 1e-6)).float()).item()
            ),
            "sigma_upper_activity": float(
                torch.mean((sigma >= float(model.control_map.sigma_max) * (1.0 - 1e-6)).float()).item()
            ),
            "controls_finite": float(bool(torch.isfinite(law.controls).all().item())),
        }
        passed = bool(
            metrics["controls_finite"] == 1.0
            and metrics["mean_kl"] <= float(args.mean_kl_budget)
            and metrics["q95_kl"] <= float(args.q95_kl_budget)
            and metrics["max_kl"] <= float(args.max_kl_budget)
        )
        return passed, metrics

    full_passed, full_trust = trust(1.0)
    if full_passed:
        accepted_scale = 1.0
        trust_metrics = full_trust
    else:
        lower, upper = 0.0, 1.0
        trust_metrics = trust(0.0)[1]
        for _ in range(24):
            midpoint = 0.5 * (lower + upper)
            passed, metrics = trust(midpoint)
            if passed:
                lower = midpoint
                trust_metrics = metrics
            else:
                upper = midpoint
        accepted_scale = lower
    theta = float(accepted_scale) * newton_direction
    newton_policy = BlockwiseAdjointInterpolationPolicy(
        model=model,
        base_policy=base_policy,
        candidate_policy=candidate_policy,
        theta=theta,
        blocks=blocks,
    )
    print(f"KL-globalized scale {accepted_scale:.6g}; theta={theta.tolist()}", flush=True)

    operator_rows = []
    operator_seeds = [int(args.seed) + 10_000 * index for index in range(int(args.confirmation_replicates) + 1)]
    for index, seed in enumerate(operator_seeds):
        label = "fit" if index == 0 else f"confirmation_{index}"
        print(f"operator residual {label}", flush=True)
        zero_coordinates, _ = _operator_coordinates(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            base_policy=base_policy,
            candidate_policy=candidate_policy,
            theta=torch.zeros(dimension, dtype=torch.float64),
            blocks=blocks,
            fit_count=int(args.operator_fit_paths),
            validation_count=int(args.operator_validation_paths),
            target_count=int(args.operator_target_paths),
            seed=seed,
        )
        candidate_coordinates, _ = _operator_coordinates(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            base_policy=base_policy,
            candidate_policy=candidate_policy,
            theta=theta,
            blocks=blocks,
            fit_count=int(args.operator_fit_paths),
            validation_count=int(args.operator_validation_paths),
            target_count=int(args.operator_target_paths),
            seed=seed,
        )
        before = _coordinate_residual(torch.zeros_like(theta), zero_coordinates)
        after = _coordinate_residual(theta, candidate_coordinates)
        operator_rows.append(
            {
                "label": label,
                "seed": seed,
                "before": before,
                "after": after,
                "relative_improvements": {
                    key: (
                        float(before[key]) - float(after[key])
                    ) / max(float(before[key]), 1e-12)
                    for key in ("total_norm", "p_norm", "r_norm")
                },
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
            },
            sort_keys=True,
        ),
    )
    audit_fit, audit_confirmation = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 100_000,
    )
    exact_rows = {}
    for label, bank in (("fit", audit_fit), ("confirmation", audit_confirmation)):
        print(f"nested-target acceptance bank {label}", flush=True)
        base_evaluation = rollout_policy_on_bank(
            model=model,
            policy=base_policy,
            bank=bank,
            data_context=context,
            full_training=model.training,
            query_batch_size=int(args.query_batch_size),
        )
        candidate_evaluation = rollout_policy_on_bank(
            model=model,
            policy=newton_policy,
            bank=bank,
            data_context=context,
            full_training=model.training,
            query_batch_size=int(args.query_batch_size),
        )
        before = _exact_bank_metrics(model=model, evaluation=base_evaluation)
        after = _exact_bank_metrics(model=model, evaluation=candidate_evaluation)
        exact_rows[label] = {"before": before, "after": after}

    projected_checks = {}
    for row in operator_rows:
        for component in ("total_norm", "p_norm", "r_norm"):
            projected_checks[f"{row['label']}_{component}_reduced"] = bool(
                float(row["after"][component]) < float(row["before"][component])
            )
    exact_checks = {}
    for label, row in exact_rows.items():
        before, after = row["before"], row["after"]
        exact_checks[f"{label}_drift_consensus_reduced"] = bool(
            after["drift_target_control_consensus_rms"]
            < before["drift_target_control_consensus_rms"]
        )
        exact_checks[f"{label}_volatility_consensus_reduced"] = bool(
            after["log_volatility_target_control_consensus_rms"]
            < before["log_volatility_target_control_consensus_rms"]
        )
        initial_objective = float(before["objective"]["complete_objective_value"])
        terminal_objective = float(after["objective"]["complete_objective_value"])
        exact_checks[f"{label}_objective_safe"] = bool(
            terminal_objective
            <= initial_objective
            + float(args.maximum_objective_relative_increase)
            * max(abs(initial_objective), 1e-12)
        )
        exact_checks[f"{label}_controls_safe"] = bool(
            float(after["safety"]["controls_finite"]) == 1.0
        )
    gate = {
        "passed": all(projected_checks.values()) and all(exact_checks.values()),
        "projected_residual_checks": projected_checks,
        "exact_nested_checks": exact_checks,
        "next_phase": (
            "recompute_operator_and_expand_coarse_space"
            if all(projected_checks.values()) and all(exact_checks.values())
            else "stop_local_root_solver"
        ),
    }
    report = {
        "protocol": {
            "single_simultaneous_residual_newton_step": True,
            "network_or_adjoint_loss_changed": False,
            "direct_ridge_R_operator_preserved": True,
            "jacobian_reused_once": True,
            "independent_confirmation_replicates": int(args.confirmation_replicates),
            "test_paths_used": False,
        },
        "inputs": {
            "spectral_report": str(args.spectral_run_dir.resolve() / "report.json"),
            "t0_reports": t0_paths,
        },
        "linear_system": {
            "t0_replicates": t0_stack.tolist(),
            "t0_mean": t0.tolist(),
            "t0_sd": t0_stack.std(dim=0, unbiased=True).tolist(),
            "jacobian_t": jacobian_t.tolist(),
            "singular_values_i_minus_jt": singular_values.tolist(),
            "condition_number_i_minus_jt": float(
                (singular_values.max() / singular_values.min()).item()
            ),
            "lm_regularization": regularization,
            "unscaled_newton_direction": newton_direction.tolist(),
            "predicted_residual": predicted_residual.tolist(),
            "predicted_residual_ratio": float(
                torch.linalg.vector_norm(predicted_residual).item()
                / torch.linalg.vector_norm(t0).item()
            ),
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
        "projected_operator_residual": operator_rows,
        "exact_nested_acceptance": exact_rows,
        "scenario_banks": {
            "fit": audit_fit.fingerprint,
            "confirmation": audit_confirmation.fingerprint,
        },
        "gate": gate,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "jacobian_t": jacobian_t,
            "t0_replicates": t0_stack,
            "newton_direction": newton_direction,
            "accepted_theta": theta,
        },
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Simultaneous projected-residual Newton audit\n\n"
        f"- T(0) mean: `{t0.tolist()}`\n"
        f"- LM direction: `{newton_direction.tolist()}`\n"
        f"- Accepted KL scale: `{accepted_scale:.6g}`\n"
        f"- Predicted residual ratio: `{report['linear_system']['predicted_residual_ratio']:.3%}`\n"
        f"- Gate passed: `{gate['passed']}`\n"
        f"- Next phase: `{gate['next_phase']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
