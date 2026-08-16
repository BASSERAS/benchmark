#!/usr/bin/env python3
"""Run the locked early-volatility full-control law/adjoint cycle."""

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

from path_dt_experiments.early_volatility_cycle import (  # noqa: E402
    locked_early_volatility_proposal,
    strict_complete_cycle_decision,
)
from path_dt_experiments.hamiltonian_gradient_audit import (  # noqa: E402
    audit_direction_evaluation,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    BlockwiseControlPerturbationPolicy,
    ScenarioDataContext,
    _bank_consensus_metrics,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
    rollout_policy_on_bank,
    time_block_bounds,
)
from path_dt_experiments.lifted_poll_solver import (  # noqa: E402
    AdjointRefitConfig,
    FrozenLiftedMerit,
    evaluate_complete_lifted_cycle,
    refit_adjoint_network,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_lifted_collocation_probe import (  # noqa: E402
    DEFAULT_ADJOINT_ARTIFACTS,
    _load_adjoint_network,
)
from run_lifted_poll_trust_region_cycle import (  # noqa: E402
    DEFAULT_TERMINAL_DIAGNOSTIC,
    _estimator_noise_variances,
)
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


DEFAULT_EQUAL_KL_RUN = (
    REPO_ROOT / "runs" / "hamiltonian_gradient_equal_kl_full_seed0_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS)
    parser.add_argument("--terminal-diagnostic", type=Path, default=DEFAULT_TERMINAL_DIAGNOSTIC)
    parser.add_argument("--equal-kl-run-dir", type=Path, default=DEFAULT_EQUAL_KL_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--precheck-seed", type=int, default=2_608_121)
    parser.add_argument("--refit-seed", type=int, default=2_608_131)
    parser.add_argument("--audit-seed", type=int, default=2_608_141)
    parser.add_argument("--precheck-paths", type=int, default=64)
    parser.add_argument("--precheck-target-paths", type=int, default=256)
    parser.add_argument("--precheck-branches", type=int, default=8)
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
    parser.add_argument("--minimum-merit-improvement", type=float, default=1e-3)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _direction_row(
    *,
    model,
    actor_policy,
    critic_policy,
    base_evaluation,
    moved_evaluation,
    budget: float,
    q95_multiplier: float,
    maximum_multiplier: float,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline, baseline_tensors = audit_direction_evaluation(
        model=model,
        evaluation=base_evaluation,
        actor_policy=actor_policy,
        critic_policy=critic_policy,
        baseline_objective=float(base_evaluation.objective["complete_objective_value"]),
    )
    moved, moved_tensors = audit_direction_evaluation(
        model=model,
        evaluation=moved_evaluation,
        actor_policy=actor_policy,
        critic_policy=critic_policy,
        baseline_objective=float(base_evaluation.objective["complete_objective_value"]),
    )
    moved["refreshed_volatility_kkt_change"] = float(
        moved["refreshed_target_volatility_kkt_rms"]
    ) - float(baseline["refreshed_target_volatility_kkt_rms"])
    moved["mean_budget_ratio"] = float(moved["transition_kl_mean"]) / float(budget)
    moved["q95_ceiling_passed"] = bool(
        float(moved["transition_kl_q95"]) <= float(budget) * float(q95_multiplier)
    )
    moved["maximum_ceiling_passed"] = bool(
        float(moved["transition_kl_max"])
        <= float(budget) * float(maximum_multiplier)
    )
    return {"baseline": baseline, "moved": moved}, {
        "baseline": baseline_tensors,
        "moved": moved_tensors,
    }


def _summary_markdown(report: dict[str, object]) -> str:
    decision = report.get("decision")
    lines = [
        "# Early-volatility complete cycle",
        "",
        f"- Locked block: `early volatility`",
        f"- Locked mean-KL budget: `{report['proposal']['target_mean_kl']:.8g}`",
        f"- Full-control coefficient: `{report['proposal']['coefficient']:.8g}`",
        "- Source checkpoint overwritten: `False`",
        "",
        "## Direction checks",
        "",
        "| Bank | Stage | dJ | d refreshed-vol KKT | KL/budget | q95 safe | max safe |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    stages = ["precheck"]
    if "fresh_audit" in report:
        stages.append("fresh_audit")
    for stage in stages:
        for bank in ("fit", "confirmation"):
            row = report[stage][bank]["moved"]
            lines.append(
                f"| {bank} | {stage} | {row['objective_change']:.6g} | "
                f"{row['refreshed_volatility_kkt_change']:.6g} | "
                f"{row['mean_budget_ratio']:.3f} | "
                f"{row['q95_ceiling_passed']} | {row['maximum_ceiling_passed']} |"
            )
    if not isinstance(decision, dict):
        lines.extend(["", "The locked precheck failed; no adjoint was refit."])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "## Post-refit complete lifted merit",
            "",
            "| Bank | Merit improvement | P | R | Drift consensus | Log-vol consensus |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        row = report["complete_cycle"][bank]["complete_cycle"]
        components = row["component_relative_improvements"]
        lines.append(
            f"| {bank} | {row['merit_relative_improvement']:.2%} | "
            f"{components['p']:.2%} | {components['r']:.2%} | "
            f"{components['alpha']:.2%} | {components['log_sigma']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Strict cycle success: `{decision['passed']}`.",
            f"Decision: `{decision['interpretation']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for paths, targets, branches, label in (
        (args.precheck_paths, args.precheck_target_paths, args.precheck_branches, "precheck"),
        (args.refit_paths, args.refit_target_paths, args.refit_branches, "refit"),
        (args.audit_paths, args.audit_target_paths, args.audit_branches, "audit"),
    ):
        if int(paths) < 1 or int(targets) < 1:
            raise ValueError(f"{label} path counts must be positive")
        if int(branches) < 2 or int(branches) % 2 != 0:
            raise ValueError(f"{label} branches must be a positive even integer")
        if int(args.query_batch_size) < int(paths) * int(branches):
            raise ValueError(f"query-batch-size is too small for {label}")

    run_dir = ensure_run_dir(args.run_dir.resolve())
    equal_kl_dir = args.equal_kl_run_dir.resolve()
    equal_kl_report = _load_json(equal_kl_dir / "equal_kl_report.json")
    proposal = locked_early_volatility_proposal(equal_kl_report, budget_index=1)
    equal_config = equal_kl_report["config"]
    q95_multiplier = float(equal_config["q95_multiplier"])
    maximum_multiplier = float(equal_config["maximum_multiplier"])
    terminal_diagnostic = _load_json(args.terminal_diagnostic.resolve())
    p_noise_variance, r_noise_variance = _estimator_noise_variances(terminal_diagnostic)
    model, target_pool, checkpoint_training, continuation_report, design = (
        _build_frozen_model(
            continuation=args.continuation_run_dir.resolve(),
            device=torch.device(args.device),
            model_seed=int(args.model_seed),
        )
    )
    stages = continuation_report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("continuation report has no terminal stage")
    terminal = stages[-1]
    operator_beta = float(terminal["beta_target"])
    if abs(operator_beta - float(equal_kl_report["operator_beta"])) > 1e-12:
        raise ValueError("equal-KL proposal addresses a different operator beta")
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
        dataset_id=json.dumps(dataset, sort_keys=True) if isinstance(dataset, dict) else str(dataset),
    )
    adjoint_network = _load_adjoint_network(
        model=model, artifacts_path=args.adjoint_artifacts.resolve()
    )
    model.network.eval()
    actor_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    critic_policy = AdjointNetworkControlPolicy(model=model, network=adjoint_network)
    blocks = time_block_bounds(num_steps=int(model.grid.num_steps), num_blocks=3)
    candidate_policy = BlockwiseControlPerturbationPolicy(
        model=model,
        law_policy=actor_policy,
        adjoint_policy=critic_policy,
        blocks=blocks,
        alpha_coefficients=(0.0, 0.0, 0.0),
        log_sigma_coefficients=(float(proposal.coefficient), 0.0, 0.0),
    )
    hashes = {
        "model": model_checkpoint_sha256(model),
        "actor": network_sha256(model.network),
        "critic": network_sha256(adjoint_network),
    }
    report: dict[str, object] = {
        "protocol": {
            "prespecified_single_coordinate": True,
            "coordinate": "early_volatility_full_control",
            "budget_selected_before_cycle": True,
            "unchanged_adjoint_loss": True,
            "refit_train_validation_and_final_audit_banks_distinct": True,
            "source_checkpoint_overwritten": False,
            "candidate_deployed_to_source": False,
            "test_paths_used": False,
        },
        "proposal": dict(proposal.__dict__),
        "operator_beta": operator_beta,
        "effective_lambda_scale": operator_beta * full_lambda_scale,
        "source_equal_kl_run": str(equal_kl_dir),
    }
    artifacts: dict[str, object] = {}

    pre_fit, pre_confirmation = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.precheck_paths),
        target_count=int(args.precheck_target_paths),
        num_branches=int(args.precheck_branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.precheck_seed),
    )
    if pre_fit.fingerprint != equal_kl_report["fit_bank_fingerprint"]:
        raise RuntimeError("locked fit bank does not reproduce equal-KL provenance")
    if pre_confirmation.fingerprint != equal_kl_report["confirmation_bank_fingerprint"]:
        raise RuntimeError("locked confirmation bank does not reproduce equal-KL provenance")
    precheck = {}
    precheck_artifacts = {}
    for bank_name, bank in (("fit", pre_fit), ("confirmation", pre_confirmation)):
        print(f"precheck {bank_name}", flush=True)
        base = rollout_policy_on_bank(
            model=model,
            policy=actor_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        moved = rollout_policy_on_bank(
            model=model,
            policy=candidate_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        precheck[bank_name], precheck_artifacts[bank_name] = _direction_row(
            model=model,
            actor_policy=actor_policy,
            critic_policy=critic_policy,
            base_evaluation=base,
            moved_evaluation=moved,
            budget=proposal.target_mean_kl,
            q95_multiplier=q95_multiplier,
            maximum_multiplier=maximum_multiplier,
        )
    report["precheck"] = precheck
    artifacts["precheck"] = precheck_artifacts
    locked_precheck_passed = all(
        float(precheck[bank]["moved"]["objective_change"]) < 0.0
        and float(precheck[bank]["moved"]["refreshed_volatility_kkt_change"]) < 0.0
        and bool(precheck[bank]["moved"]["q95_ceiling_passed"])
        and bool(precheck[bank]["moved"]["maximum_ceiling_passed"])
        and float(
            precheck[bank]["moved"]["candidate_control_safety"]["controls_finite"]
        )
        == 1.0
        and float(
            precheck[bank]["moved"]["candidate_control_safety"][
                "specific_entropy_denominator_margin_min"
            ]
        )
        > 0.0
        for bank in ("fit", "confirmation")
    )
    report["locked_precheck_passed"] = locked_precheck_passed
    if not locked_precheck_passed:
        report["decision"] = None
    else:
        torch.save(
            {
                "protocol": "locked_early_volatility_full_control",
                "proposal": dict(proposal.__dict__),
                "base_law_network_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.network.state_dict().items()
                },
                "proposal_adjoint_network_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in adjoint_network.state_dict().items()
                },
            },
            run_dir / "candidate_law.pt",
        )
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
        print("building refit training targets", flush=True)
        refit_fit = rollout_policy_on_bank(
            model=model,
            policy=candidate_policy,
            bank=refit_fit_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        print("building refit validation targets", flush=True)
        refit_validation = rollout_policy_on_bank(
            model=model,
            policy=candidate_policy,
            bank=refit_validation_bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )

        def progress(message: str) -> None:
            print(message, flush=True)

        refit = refit_adjoint_network(
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
        report["adjoint_refit"] = refit.report
        torch.save(
            {
                "state_dict": refit.state_dict,
                "candidate_law_file": "candidate_law.pt",
                "proposal": dict(proposal.__dict__),
                "training": refit.report,
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
        post_refit_policy = AdjointNetworkControlPolicy(model=model, network=refit.network)
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
        fresh_direction = {}
        fresh_artifacts = {}
        fresh_base_fit = None
        for bank_name, bank in (("fit", audit_fit), ("confirmation", audit_confirmation)):
            print(f"fresh audit direction {bank_name}", flush=True)
            base = rollout_policy_on_bank(
                model=model,
                policy=actor_policy,
                bank=bank,
                data_context=context,
                full_training=full_training,
                query_batch_size=int(args.query_batch_size),
            )
            moved = rollout_policy_on_bank(
                model=model,
                policy=candidate_policy,
                bank=bank,
                data_context=context,
                full_training=full_training,
                query_batch_size=int(args.query_batch_size),
            )
            fresh_direction[bank_name], fresh_artifacts[bank_name] = _direction_row(
                model=model,
                actor_policy=actor_policy,
                critic_policy=critic_policy,
                base_evaluation=base,
                moved_evaluation=moved,
                budget=proposal.target_mean_kl,
                q95_multiplier=q95_multiplier,
                maximum_multiplier=maximum_multiplier,
            )
            if bank_name == "fit":
                fresh_base_fit = base
        assert fresh_base_fit is not None
        _, merit_baseline_artifacts = _bank_consensus_metrics(
            model=model,
            evaluation=fresh_base_fit,
            adjoint_policy=critic_policy,
        )
        merit = FrozenLiftedMerit.from_baseline(
            merit_baseline_artifacts,
            p_estimator_noise_variance=p_noise_variance,
            r_estimator_noise_variance=r_noise_variance,
        )
        complete_cycle, cycle_artifacts = evaluate_complete_lifted_cycle(
            model=model,
            base_law_policy=actor_policy,
            accepted_law_policy=candidate_policy,
            pre_refit_adjoint_policy=critic_policy,
            post_refit_adjoint_policy=post_refit_policy,
            data_context=context,
            fit_bank=audit_fit,
            confirmation_bank=audit_confirmation,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
            merit=merit,
        )
        decision = strict_complete_cycle_decision(
            precheck={bank: precheck[bank]["moved"] for bank in ("fit", "confirmation")},
            fresh_direction={
                bank: fresh_direction[bank]["moved"] for bank in ("fit", "confirmation")
            },
            complete_cycle=complete_cycle,
            minimum_merit_relative_improvement=float(args.minimum_merit_improvement),
        )
        report["fresh_audit"] = fresh_direction
        report["complete_cycle"] = complete_cycle
        report["decision"] = decision
        artifacts["fresh_audit"] = fresh_artifacts
        artifacts["complete_cycle"] = cycle_artifacts

    if model_checkpoint_sha256(model) != hashes["model"]:
        raise RuntimeError("cycle mutated the source model")
    if network_sha256(model.network) != hashes["actor"]:
        raise RuntimeError("cycle mutated the source actor")
    if network_sha256(adjoint_network) != hashes["critic"]:
        raise RuntimeError("cycle mutated the source critic")
    report["nonmutation"] = {
        "source_model_unchanged": True,
        "source_actor_unchanged": True,
        "source_critic_unchanged": True,
    }
    write_json(run_dir / "cycle_report.json", report)
    torch.save(artifacts, run_dir / "cycle_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(f"wrote early-volatility cycle to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
