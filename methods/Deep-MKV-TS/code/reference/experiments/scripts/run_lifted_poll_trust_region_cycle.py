#!/usr/bin/env python3
"""One-update poll trust region followed by an unchanged-loss adjoint refit."""

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
    FrozenLiftedMerit,
    PollTrustRegionConfig,
    evaluate_complete_lifted_cycle,
    poll_one_law_update,
    refit_adjoint_network,
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


DEFAULT_TERMINAL_DIAGNOSTIC = (
    REPO_ROOT / "runs" / "terminal_fixed_point_diagnostic_seed0_20260803"
    / "diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS)
    parser.add_argument(
        "--terminal-diagnostic", type=Path, default=DEFAULT_TERMINAL_DIAGNOSTIC
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=2_608_081)
    parser.add_argument("--refit-seed", type=int, default=2_608_091)
    parser.add_argument("--audit-seed", type=int, default=2_608_101)
    parser.add_argument("--poll-paths", type=int, default=64)
    parser.add_argument("--poll-target-paths", type=int, default=256)
    parser.add_argument("--poll-branches", type=int, default=8)
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


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _estimator_noise_variances(diagnostic: dict[str, object]) -> tuple[float, float]:
    residual = diagnostic.get("residual_estimator")
    if not isinstance(residual, dict):
        raise ValueError("terminal diagnostic is missing residual_estimator")
    base = residual.get("base_policy")
    if not isinstance(base, dict):
        raise ValueError("terminal diagnostic is missing base_policy")
    values = []
    for key in ("p", "r"):
        row = base.get(key)
        if not isinstance(row, dict) or "estimator_noise_rms" not in row:
            raise ValueError(f"terminal diagnostic is missing {key} estimator noise")
        rms = float(row["estimator_noise_rms"])
        values.append(rms * rms)
    return values[0], values[1]


def _summary_markdown(report: dict[str, object]) -> str:
    poll = report["poll"]
    lines = [
        "# Lifted poll trust-region cycle",
        "",
        f"- Operator beta: `{report['operator_beta']:.10g}`",
        f"- Effective lambda: `{report['effective_lambda_scale']:.10g}`",
        f"- Accepted law update: `{poll['accepted']}`",
        f"- Confirmation attempts: `{poll['confirmation_attempts']}`",
        "",
        "## Poll decisions",
        "",
        "| Radius | Fit selection | Confirmation | Accepted |",
        "|---:|---|---|---|",
    ]
    for row in poll["radii"]:
        gate = row.get("confirmation_gate")
        gate_label = "not evaluated" if gate is None else str(gate["passed"])
        lines.append(
            f"| {row['radius']:.6g} | {row['fit_selected_coordinate']} | "
            f"{gate_label} | {row['accepted']} |"
        )
    if not bool(poll["accepted"]):
        lines.extend(
            [
                "",
                "No update was confirmed. The protocol terminated at the radius floor without refitting or modifying a checkpoint.",
            ]
        )
        return "\n".join(lines) + "\n"
    accepted = poll["accepted_candidate"]
    lines.extend(
        [
            "",
            f"Accepted coordinate: `{accepted['coordinate']}` at radius `{accepted['radius']:.6g}`.",
            "",
            "## Adjoint refit",
            "",
            f"- Best step: `{report['adjoint_refit']['best_step']}`",
            f"- Initial validation loss: `{report['adjoint_refit']['initial']['validation']['total']:.6g}`",
            f"- Selected validation loss: `{report['adjoint_refit']['selected']['validation']['total']:.6g}`",
            "",
            "## Fresh complete-cycle audit",
            "",
            "| Bank | Merit improvement | P | R | Drift consensus | Log-vol consensus |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    cycle = report["complete_cycle"]
    for bank in ("fit", "confirmation"):
        row = cycle[bank]["complete_cycle"]
        components = row["component_relative_improvements"]
        lines.append(
            f"| {bank} | {row['merit_relative_improvement']:.2%} | "
            f"{components['p']:.2%} | {components['r']:.2%} | "
            f"{components['alpha']:.2%} | {components['log_sigma']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Complete-cycle success: `{cycle['cycle_success']}`.",
            "",
            "The accepted law and refitted adjoint are saved separately; the source checkpoint was not overwritten.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for count, branches, label in (
        (args.poll_paths, args.poll_branches, "poll"),
        (args.refit_paths, args.refit_branches, "refit"),
        (args.audit_paths, args.audit_branches, "audit"),
    ):
        if int(count) < 1 or int(branches) < 2 or int(branches) % 2 != 0:
            raise ValueError(f"{label} paths must be positive and branches even")
        if int(args.query_batch_size) < int(count) * int(branches):
            raise ValueError(f"query-batch-size is too small for the {label} bank")

    continuation = args.continuation_run_dir.resolve()
    adjoint_artifacts = args.adjoint_artifacts.resolve()
    terminal_diagnostic_path = args.terminal_diagnostic.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    terminal_diagnostic = _load_json(terminal_diagnostic_path)
    p_noise_variance, r_noise_variance = _estimator_noise_variances(
        terminal_diagnostic
    )
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
    if not isinstance(terminal, dict) or bool(terminal.get("accepted")):
        raise ValueError("poll cycle expects the first terminal rejected stage")
    operator_beta = float(terminal["beta_target"])
    full_lambda_scale = float(continuation_report["final_lambda_scale"])
    effective_lambda = operator_beta * full_lambda_scale
    if abs(effective_lambda - float(terminal["lambda_scale"])) > 1e-10:
        raise RuntimeError("continuation report scales the discrepancy inconsistently")
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
    base_law_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    pre_refit_adjoint_policy = AdjointNetworkControlPolicy(
        model=model, network=adjoint_network
    )
    model_hash = model_checkpoint_sha256(model)
    law_hash = network_sha256(model.network)
    adjoint_hash = network_sha256(adjoint_network)

    poll_fit, poll_confirmation = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.poll_paths),
        target_count=int(args.poll_target_paths),
        num_branches=int(args.poll_branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.selection_seed),
    )

    def progress(message: str) -> None:
        print(message, flush=True)

    poll_result = poll_one_law_update(
        model=model,
        law_policy=base_law_policy,
        adjoint_policy=pre_refit_adjoint_policy,
        data_context=context,
        fit_bank=poll_fit,
        confirmation_bank=poll_confirmation,
        full_training=full_training,
        query_batch_size=int(args.query_batch_size),
        p_estimator_noise_variance=p_noise_variance,
        r_estimator_noise_variance=r_noise_variance,
        config=PollTrustRegionConfig(),
        progress_callback=progress,
    )
    report: dict[str, object] = {
        "protocol": {
            "maximum_accepted_law_updates": 1,
            "source_checkpoint_overwritten": False,
            "test_paths_used": False,
            "poll_selection_and_confirmation_banks_independent": True,
            "post_refit_audit_banks_fresh": True,
        },
        "continuation_run_dir": str(continuation),
        "adjoint_artifacts": str(adjoint_artifacts),
        "terminal_diagnostic": str(terminal_diagnostic_path),
        "operator_beta": operator_beta,
        "full_lambda_scale": full_lambda_scale,
        "effective_lambda_scale": effective_lambda,
        "estimator_noise_variances": {
            "p": p_noise_variance,
            "r": r_noise_variance,
        },
        "poll": poll_result.report,
    }
    artifacts: dict[str, object] = {"poll": poll_result.artifacts}
    if poll_result.accepted_policy is not None:
        accepted = poll_result.report["accepted_candidate"]
        accepted_law_payload = {
            "protocol": "blockwise_physical_control_interpolation",
            "operator_beta": operator_beta,
            "dataset_fingerprint": context.fingerprint,
            "accepted_candidate": accepted,
            "base_law_network_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.network.state_dict().items()
            },
            "proposal_adjoint_network_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in adjoint_network.state_dict().items()
            },
        }
        torch.save(accepted_law_payload, run_dir / "accepted_law.pt")

        refit_fit_bank, refit_validation_bank = build_fit_confirmation_banks(
            model=model,
            context=context,
            path_count=int(args.refit_paths),
            target_count=int(args.refit_target_paths),
            num_branches=int(args.refit_branches),
            antithetic=True,
            beta=operator_beta,
            seed=int(args.refit_seed),
        )
        progress("building refreshed adjoint fit targets")
        refit_fit = rollout_policy_on_bank(
            model=model,
            policy=poll_result.accepted_policy,
            bank=refit_fit_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        progress("building refreshed adjoint validation targets")
        refit_validation = rollout_policy_on_bank(
            model=model,
            policy=poll_result.accepted_policy,
            bank=refit_validation_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        refit_result = refit_adjoint_network(
            model=model,
            initial_network=adjoint_network,
            fit_paths=refit_fit.paths,
            fit_target_p=refit_fit.target_p,
            fit_target_r=refit_fit.target_r,
            validation_paths=refit_validation.paths,
            validation_target_p=refit_validation.target_p,
            validation_target_r=refit_validation.target_r,
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
        report["adjoint_refit"] = refit_result.report
        torch.save(
            {
                "state_dict": refit_result.state_dict,
                "accepted_law_file": "accepted_law.pt",
                "operator_beta": operator_beta,
                "training": refit_result.report,
            },
            run_dir / "refitted_adjoint.pt",
        )
        artifacts["refit"] = {
            "fit_paths": refit_fit.paths.detach().cpu(),
            "fit_target_p": refit_fit.target_p.detach().cpu(),
            "fit_target_r": refit_fit.target_r.detach().cpu(),
            "validation_paths": refit_validation.paths.detach().cpu(),
            "validation_target_p": refit_validation.target_p.detach().cpu(),
            "validation_target_r": refit_validation.target_r.detach().cpu(),
        }
        post_refit_policy = AdjointNetworkControlPolicy(
            model=model, network=refit_result.network
        )
        audit_fit, audit_confirmation = build_fit_confirmation_banks(
            model=model,
            context=context,
            path_count=int(args.audit_paths),
            target_count=int(args.audit_target_paths),
            num_branches=int(args.audit_branches),
            antithetic=True,
            beta=operator_beta,
            seed=int(args.audit_seed),
        )
        merit = FrozenLiftedMerit.from_baseline(
            poll_result.artifacts["baseline"]["fit"],
            p_estimator_noise_variance=p_noise_variance,
            r_estimator_noise_variance=r_noise_variance,
        )
        cycle_report, cycle_artifacts = evaluate_complete_lifted_cycle(
            model=model,
            base_law_policy=base_law_policy,
            accepted_law_policy=poll_result.accepted_policy,
            pre_refit_adjoint_policy=pre_refit_adjoint_policy,
            post_refit_adjoint_policy=post_refit_policy,
            data_context=context,
            fit_bank=audit_fit,
            confirmation_bank=audit_confirmation,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
            merit=merit,
        )
        report["complete_cycle"] = cycle_report
        artifacts["complete_cycle"] = cycle_artifacts

    if model_checkpoint_sha256(model) != model_hash:
        raise RuntimeError("source model changed during poll cycle")
    if network_sha256(model.network) != law_hash:
        raise RuntimeError("source law network changed during poll cycle")
    if network_sha256(adjoint_network) != adjoint_hash:
        raise RuntimeError("source adjoint network changed during poll cycle")
    report["nonmutation"] = {
        "source_model_unchanged": True,
        "source_law_network_unchanged": True,
        "source_adjoint_network_unchanged": True,
    }
    write_json(run_dir / "poll_cycle_report.json", report)
    torch.save(artifacts, run_dir / "poll_cycle_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    print(f"wrote lifted poll trust-region cycle to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
