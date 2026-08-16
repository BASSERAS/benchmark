#!/usr/bin/env python3
"""Phase-1 curvature-matched specific-entropy versus Gaussian-KL audit."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    GaussianRelativeEntropyDiagonalControl,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
    rollout_policy_on_bank,
)
from path_dt_experiments.lifted_poll_solver import (  # noqa: E402
    AdjointRefitConfig,
    refit_adjoint_network,
)
from path_dt_experiments.running_cost_curvature_audit import (  # noqa: E402
    curvature_matched_gaussian_eta,
    curvature_phase_one_gate,
    fixed_law_cost_audit,
    log_volatility_curvature_at_reference,
    target_agreement,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--network-init-seed", type=int, default=2_608_301)
    parser.add_argument("--refit-seed", type=int, default=2_608_311)
    parser.add_argument("--audit-seed", type=int, default=2_608_321)
    parser.add_argument("--refit-paths", type=int, default=512)
    parser.add_argument("--refit-target-paths", type=int, default=1_024)
    parser.add_argument("--refit-branches", type=int, default=8)
    parser.add_argument("--audit-paths", type=int, default=64)
    parser.add_argument("--audit-target-paths", type=int, default=256)
    parser.add_argument("--audit-branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4_096)
    parser.add_argument("--refit-inner-batch-size", type=int, default=128)
    parser.add_argument("--refit-max-steps", type=int, default=2_000)
    parser.add_argument("--refit-eval-every", type=int, default=50)
    parser.add_argument("--refit-patience", type=int, default=12)
    return parser.parse_args()


def _cost_model(
    base_model: DiscreteMPModel,
    *,
    control,
) -> DiscreteMPModel:
    model = copy.copy(base_model)
    model.control_map = control
    model.running_cost = control.running_cost
    return model


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Curvature-matched running-cost audit",
        "",
        "Phase 1 fixes the physical law and trains fresh adjoints from a common",
        "initialization. Gaussian KL replaces only the volatility cost.",
        "",
        f"- Specific log-volatility curvature: `{report['curvature']['specific_entropy']:.6g}`",
        f"- Gaussian log-volatility curvature: `{report['curvature']['gaussian_relative_entropy']:.6g}`",
        f"- Phase-1 gate passed: `{report['phase_one_gate']['passed']}`",
        f"- Next phase: `{report['phase_one_gate']['next_phase']}`",
        "",
        "| Bank | Cost | P CE | R CE | log-vol move | upper bound | KL q95 | sens. q95 | target KKT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in ("fit", "confirmation"):
        for cost in ("specific_entropy", "gaussian_relative_entropy"):
            row = report["audit"][cost][bank]
            lines.append(
                f"| {bank} | {cost} | {row['p_ce_residual_rms']:.4g} | "
                f"{row['r_ce_residual_rms']:.4g} | "
                f"{row['log_volatility_move_rms']:.4g} | "
                f"{row['induced_sigma_upper_activity']:.2%} | "
                f"{row['transition_kl']['q95']:.4g} | "
                f"{row['induced_fitted_sensitivity_abs']['q95']:.4g} | "
                f"{row['induced_target_kkt_rms']:.4g} |"
            )
    lines.extend(
        [
            "",
            "## Gaussian/specific ratios",
            "",
            "| Bank | Sensitivity q95 | KL q95 | KL max | Upper activity | Move RMS | P CE | R CE | Target KKT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        row = report["phase_one_gate"]["ratios_gaussian_over_specific"][bank]
        lines.append(
            f"| {bank} | {row['sensitivity_q95']:.3f} | "
            f"{row['transition_kl_q95']:.3f} | {row['transition_kl_max']:.3f} | "
            f"{row['upper_activity']:.3f} | {row['log_volatility_move_rms']:.3f} | "
            f"{row['p_ce_residual_rms']:.3f} | {row['r_ce_residual_rms']:.3f} | "
            f"{row['induced_target_kkt_rms']:.3f} |"
        )
    lines.extend(["", "## Locked checks", ""])
    for name, passed in report["phase_one_gate"]["checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(
        [
            "",
            "## Fixed-law mathematical target agreement",
            "",
            "| Bank | Signal | Relative difference | Cosine |",
            "|---|---|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        for signal in ("p", "r"):
            row = report["target_agreement"][bank][signal]
            lines.append(
                f"| {bank} | {signal.upper()} | "
                f"{row['relative_difference']:.3e} | {row['cosine']:.12f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for paths, targets, branches, label in (
        (args.refit_paths, args.refit_target_paths, args.refit_branches, "refit"),
        (args.audit_paths, args.audit_target_paths, args.audit_branches, "audit"),
    ):
        if int(paths) < 1 or int(targets) < 1:
            raise ValueError(f"{label} paths and targets must be positive")
        if int(branches) < 2 or int(branches) % 2 != 0:
            raise ValueError(f"{label} branches must be an even integer")
        if int(args.query_batch_size) < int(paths) * int(branches):
            raise ValueError(f"query-batch-size is too small for {label}")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    base_model, target_pool, checkpoint_training, continuation_report, design = (
        _build_frozen_model(
            continuation=args.continuation_run_dir.resolve(),
            device=torch.device(args.device),
            model_seed=int(args.model_seed),
        )
    )
    if not isinstance(base_model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("curvature audit requires the specific-entropy source model")
    specific_control = base_model.control_map
    if specific_control.sigma_max is None:
        raise ValueError("curvature audit requires bounded volatility")
    gaussian_control = GaussianRelativeEntropyDiagonalControl(
        dt=float(specific_control.dt),
        reference_kernel=specific_control.reference_kernel,
        eta=curvature_matched_gaussian_eta(float(specific_control.eta)),
        drift_penalty=1.0,
        sigma_min=float(specific_control.sigma_min),
        sigma_max=float(specific_control.sigma_max),
    )
    models = {
        "specific_entropy": _cost_model(base_model, control=specific_control),
        "gaussian_relative_entropy": _cost_model(
            base_model, control=gaussian_control
        ),
    }
    curvature = {
        name: log_volatility_curvature_at_reference(model.control_map)
        for name, model in models.items()
    }
    if abs(curvature["specific_entropy"] - curvature["gaussian_relative_entropy"]) > 1e-12:
        raise RuntimeError("volatility costs are not curvature matched")
    stages = continuation_report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("continuation report has no terminal stage")
    operator_beta = float(stages[-1]["beta_target"])
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
    fixed_law_policy = AdjointNetworkControlPolicy(
        model=base_model, network=base_model.network
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(args.network_init_seed))
        common_initial_network = DiscreteMPModel._make_network(base_model.architecture)
    common_initial_network.to(device=base_model.device, dtype=base_model.dtype)
    initial_hash = network_sha256(common_initial_network)
    source_hash = model_checkpoint_sha256(base_model)
    source_network_hash = network_sha256(base_model.network)
    refit_fit_bank, refit_validation_bank = build_fit_confirmation_banks(
        model=base_model,
        context=context,
        path_count=int(args.refit_paths),
        target_count=int(args.refit_target_paths),
        num_branches=int(args.refit_branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.refit_seed),
    )
    audit_fit_bank, audit_confirmation_bank = build_fit_confirmation_banks(
        model=base_model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.audit_seed),
    )
    refits = {}
    training_artifacts = {}
    fixed_law_fingerprints = {}
    for cost_name, model in models.items():
        print(f"building {cost_name} training targets", flush=True)
        fit = rollout_policy_on_bank(
            model=model,
            policy=fixed_law_policy,
            bank=refit_fit_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        print(f"building {cost_name} validation targets", flush=True)
        validation = rollout_policy_on_bank(
            model=model,
            policy=fixed_law_policy,
            bank=refit_validation_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        fixed_law_fingerprints[cost_name] = {
            "fit_paths": float(fit.paths.detach().double().sum().item()),
            "fit_controls": float(fit.controls.detach().double().sum().item()),
            "validation_paths": float(validation.paths.detach().double().sum().item()),
            "validation_controls": float(validation.controls.detach().double().sum().item()),
        }

        def progress(message: str, *, label: str = cost_name) -> None:
            print(f"{label}: {message}", flush=True)

        result = refit_adjoint_network(
            model=model,
            initial_network=common_initial_network,
            fit_paths=fit.paths,
            fit_target_p=fit.target_p,
            fit_target_r=fit.target_r,
            validation_paths=validation.paths,
            validation_target_p=validation.target_p,
            validation_target_r=validation.target_r,
            training=full_training,
            config=AdjointRefitConfig(
                inner_batch_size=int(args.refit_inner_batch_size),
                max_steps=int(args.refit_max_steps),
                eval_every=int(args.refit_eval_every),
                patience=int(args.refit_patience),
                seed=int(args.refit_seed) + 10_000,
            ),
            progress_callback=progress,
        )
        refits[cost_name] = result
        training_artifacts[cost_name] = {
            "fit_paths": fit.paths.detach().cpu(),
            "fit_target_p": fit.target_p.detach().cpu(),
            "fit_target_r": fit.target_r.detach().cpu(),
            "validation_paths": validation.paths.detach().cpu(),
            "validation_target_p": validation.target_p.detach().cpu(),
            "validation_target_r": validation.target_r.detach().cpu(),
        }
        torch.save(
            {
                "cost": cost_name,
                "state_dict": result.state_dict,
                "refit_report": result.report,
                "common_initial_network_sha256": initial_hash,
            },
            run_dir / f"fresh_adjoint_{cost_name}.pt",
        )
    if fixed_law_fingerprints["specific_entropy"] != fixed_law_fingerprints[
        "gaussian_relative_entropy"
    ]:
        raise RuntimeError("cost variants did not train on the same fixed physical law")

    audit_report = {name: {} for name in models}
    audit_artifacts = {name: {} for name in models}
    audit_law_tensors = {}
    for cost_name, model in models.items():
        policy = AdjointNetworkControlPolicy(
            model=model, network=refits[cost_name].network
        )
        for bank_name, bank in (
            ("fit", audit_fit_bank),
            ("confirmation", audit_confirmation_bank),
        ):
            print(f"auditing {cost_name} {bank_name}", flush=True)
            evaluation = rollout_policy_on_bank(
                model=model,
                policy=fixed_law_policy,
                bank=bank,
                data_context=context,
                full_training=full_training,
                query_batch_size=int(args.query_batch_size),
            )
            key = bank_name
            if key not in audit_law_tensors:
                audit_law_tensors[key] = {
                    "paths": evaluation.paths.detach().cpu(),
                    "controls": evaluation.controls.detach().cpu(),
                }
            else:
                torch.testing.assert_close(
                    evaluation.paths.detach().cpu(),
                    audit_law_tensors[key]["paths"],
                )
                torch.testing.assert_close(
                    evaluation.controls.detach().cpu(),
                    audit_law_tensors[key]["controls"],
                )
            row, tensors = fixed_law_cost_audit(
                model=model,
                evaluation=evaluation,
                adjoint_policy=policy,
            )
            row["conditional_target_metadata"] = evaluation.metadata[
                "conditional_target"
            ]
            audit_report[cost_name][bank_name] = row
            audit_artifacts[cost_name][bank_name] = tensors
    gate = curvature_phase_one_gate(audit_report)
    target_comparison = {}
    for bank, artifact_key in (("fit", "fit"), ("confirmation", "validation")):
        target_comparison[bank] = {
            signal: target_agreement(
                training_artifacts["specific_entropy"][
                    f"{artifact_key}_target_{signal}"
                ],
                training_artifacts["gaussian_relative_entropy"][
                    f"{artifact_key}_target_{signal}"
                ],
            )
            for signal in ("p", "r")
        }
    report = {
        "protocol": {
            "diagnostic_phase": 1,
            "fixed_physical_law": True,
            "fresh_adjoint_per_cost": True,
            "common_network_initialization": True,
            "common_minibatch_sequence": True,
            "complete_running_cost_sources_recomputed": True,
            "volatility_component_only_changed": True,
            "drift_cost_unchanged": True,
            "discrepancy_reference_bounds_architecture_and_loss_unchanged": True,
            "test_paths_used": False,
            "phase_two_is_conditional": True,
        },
        "operator_beta": operator_beta,
        "effective_lambda_scale": operator_beta * full_lambda_scale,
        "curvature": curvature,
        "specific_eta": float(specific_control.eta),
        "gaussian_eta": float(gaussian_control.eta),
        "common_initial_network_sha256": initial_hash,
        "bank_fingerprints": {
            "refit_fit": refit_fit_bank.fingerprint,
            "refit_validation": refit_validation_bank.fingerprint,
            "audit_fit": audit_fit_bank.fingerprint,
            "audit_confirmation": audit_confirmation_bank.fingerprint,
        },
        "refit": {
            name: result.report for name, result in refits.items()
        },
        "audit": audit_report,
        "target_agreement": target_comparison,
        "phase_one_gate": gate,
        "fixed_law_fingerprints": fixed_law_fingerprints,
    }
    if model_checkpoint_sha256(base_model) != source_hash:
        raise RuntimeError("running-cost audit mutated the source model")
    if network_sha256(base_model.network) != source_network_hash:
        raise RuntimeError("running-cost audit mutated the source actor")
    if network_sha256(common_initial_network) != initial_hash:
        raise RuntimeError("running-cost audit mutated the common initialization")
    report["nonmutation"] = {
        "source_model_unchanged": True,
        "source_actor_unchanged": True,
        "common_initial_network_unchanged": True,
    }
    write_json(run_dir / "phase_one_report.json", report)
    torch.save(
        {
            "training": training_artifacts,
            "audit": audit_artifacts,
            "fixed_law": audit_law_tensors,
        },
        run_dir / "phase_one_tensors.pt",
    )
    (run_dir / "SUMMARY.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(f"wrote phase-1 curvature audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
