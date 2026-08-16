#!/usr/bin/env python3
"""Run the locked prefix/population/conditional-future target-channel audit."""

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

from path_dt_experiments.conditional_target_channel_audit import (  # noqa: E402
    factorial_conditional_targets_on_bank,
    locked_channel_decision,
    summarize_factorial_bank,
)
from path_dt_experiments.early_volatility_cycle import (  # noqa: E402
    locked_early_volatility_proposal,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    BlockwiseControlPerturbationPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
    time_block_bounds,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_early_volatility_complete_cycle import DEFAULT_EQUAL_KL_RUN  # noqa: E402
from run_lifted_collocation_probe import (  # noqa: E402
    DEFAULT_ADJOINT_ARTIFACTS,
    _load_adjoint_network,
)
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS)
    parser.add_argument("--equal-kl-run-dir", type=Path, default=DEFAULT_EQUAL_KL_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--audit-seed", type=int, default=2_608_411)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--target-paths", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=2_048)
    parser.add_argument("--dominance-threshold", type=float, default=0.50)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Conditional-target channel decomposition",
        "",
        "The audit applies the locked early-volatility perturbation and evaluates",
        "all eight prefix/population/conditional-future refresh combinations.",
        "No policy, adjoint, discrepancy, or checkpoint is updated.",
        "",
        f"- Proposal coefficient: `{report['proposal']['coefficient']:.8g}`",
        f"- Mean-KL provenance budget: `{report['proposal']['target_mean_kl']:.8g}`",
        f"- Dominant channel: `{report['decision']['dominant_channel']}`",
        f"- Dominant discrepancy family: `{report['decision']['dominant_discrepancy_family']}`",
        f"- Locked next experiment: `{report['decision']['next_experiment']}`",
        "",
        "## Channel attribution",
        "",
        "| Bank | Channel | Joint share | P share | R share | Log-vol share | Joint alignment |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bank in ("fit", "confirmation"):
        for channel, row in report["banks"][bank]["channels"].items():
            lines.append(
                f"| {bank} | {channel} | {row['magnitude_share']:.2%} | "
                f"{report['banks'][bank]['head_channels'][channel]['p']['magnitude_share']:.2%} | "
                f"{report['banks'][bank]['head_channels'][channel]['r']['magnitude_share']:.2%} | "
                f"{report['banks'][bank]['control_channels'][channel]['log_volatility']['magnitude_share']:.2%} | "
                f"{row['alignment_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Discrepancy-family attribution",
            "",
            "| Bank | Family | Magnitude share | Alignment fraction | Cosine |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        for family, row in sorted(report["banks"][bank]["families"].items()):
            lines.append(
                f"| {bank} | {family} | {row['magnitude_share']:.2%} | "
                f"{row['alignment_fraction']:.3f} | {row['cosine_to_total']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Selected factorial cells",
            "",
            "| Bank | Cell | Joint change RMS | Cosine to full change | Target KKT |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        cells = report["banks"][bank]["cells"]
        for cell in ("q0_m0_c0", "q1_m0_c0", "q0_m1_c0", "q0_m0_c1", "q1_m1_c1"):
            row = cells[cell]
            lines.append(
                f"| {bank} | {cell} | {row['joint_change_rms']:.5g} | "
                f"{row['cosine_to_complete_target_change']:.3f} | "
                f"{row['target_kkt_rms_at_cell_law']:.5g} |"
            )
    lines.extend(["", "## Complete control response", ""])
    for bank in ("fit", "confirmation"):
        row = report["banks"][bank]["control_reversal"]
        lines.append(
            f"- {bank}: target/law drift cosine "
            f"`{row['target_vs_law_drift_cosine']:.3f}`; target/law log-volatility "
            f"cosine `{row['target_vs_law_log_volatility_cosine']:.3f}`."
        )
    lines.extend(["", "## Numerical checks", ""])
    for bank in ("fit", "confirmation"):
        reconstruction = report["banks"][bank]["reconstruction"]
        metadata = report["banks"][bank]["metadata"]
        lines.append(
            f"- {bank}: Shapley error `{reconstruction['shapley_relative_error']:.3e}`, "
            f"component error `{reconstruction['component_relative_error']:.3e}`, "
            f"prefix error `{max(metadata['prefix_max_abs_errors'].values()):.3e}`."
        )
    lines.extend(["", "## Interpretation", ""])
    for bank in ("fit", "confirmation"):
        bank_report = report["banks"][bank]
        query_population = sum(
            bank_report["channels"][name]["magnitude_share"]
            for name in ("query_prefix", "population_law")
        )
        volatility_system = sum(
            bank_report["families"][name]["magnitude_share"]
            for name in ("realized_volatility", "prefix_future")
        )
        lines.append(
            f"- {bank}: query plus population channels account for "
            f"`{query_population:.2%}` of channel magnitude; realized-volatility "
            f"plus prefix--future discrepancies account for `{volatility_system:.2%}` "
            f"of component magnitude; running cost accounts for "
            f"`{bank_report['families']['running_cost']['magnitude_share']:.2%}`."
        )
    lines.extend(
        [
            "",
            "No channel or separately declared discrepancy family clears the locked",
            "50% dominance rule on both banks. Law mixing would address only the",
            "population part of a distributed prefix--population response, so the",
            "audit does not authorize a mixture-of-laws solver experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if int(args.paths) < 1 or int(args.target_paths) < 1:
        raise ValueError("paths and target-paths must be positive")
    if int(args.branches) < 2 or int(args.branches) % 2 != 0:
        raise ValueError("branches must be a positive even integer")
    if int(args.query_batch_size) < int(args.paths) * int(args.branches):
        raise ValueError("query-batch-size must hold a complete branch population")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    equal_kl_dir = args.equal_kl_run_dir.resolve()
    equal_kl_report = _load_json(equal_kl_dir / "equal_kl_report.json")
    proposal = locked_early_volatility_proposal(equal_kl_report, budget_index=1)
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
    beta = float(stages[-1]["beta_target"])
    if abs(beta - float(equal_kl_report["operator_beta"])) > 1e-12:
        raise ValueError("locked perturbation and audited operator beta differ")
    full_training = replace(
        checkpoint_training,
        lambda_scale=float(continuation_report["final_lambda_scale"]),
        grad_clip_norm=None,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
    )
    model.training = full_training
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
    critic_network = _load_adjoint_network(
        model=model, artifacts_path=args.adjoint_artifacts.resolve()
    )
    model.network.eval()
    critic_network.eval()
    baseline_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    critic_policy = AdjointNetworkControlPolicy(model=model, network=critic_network)
    blocks = time_block_bounds(num_steps=int(model.grid.num_steps), num_blocks=3)
    candidate_policy = BlockwiseControlPerturbationPolicy(
        model=model,
        law_policy=baseline_policy,
        adjoint_policy=critic_policy,
        blocks=blocks,
        alpha_coefficients=(0.0, 0.0, 0.0),
        log_sigma_coefficients=(float(proposal.coefficient), 0.0, 0.0),
    )
    hashes_before = {
        "model": model_checkpoint_sha256(model),
        "actor": network_sha256(model.network),
        "critic": network_sha256(critic_network),
    }
    fit_bank, confirmation_bank = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.paths),
        target_count=int(args.target_paths),
        num_branches=int(args.branches),
        antithetic=True,
        beta=beta,
        seed=int(args.audit_seed),
    )
    bank_reports = {}
    bank_artifacts = {}
    for name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        print(f"factorial channel audit {name}", flush=True)
        result = factorial_conditional_targets_on_bank(
            model=model,
            baseline_policy=baseline_policy,
            candidate_policy=candidate_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        bank_reports[name] = summarize_factorial_bank(model=model, result=result)
        bank_artifacts[name] = {
            "cells": result.cells,
            "baseline_paths": result.baseline_paths,
            "baseline_controls": result.baseline_controls,
            "candidate_paths": result.candidate_paths,
            "candidate_controls": result.candidate_controls,
            "component_families": result.component_families,
            "metadata": result.metadata,
            "bank": {
                "initial_indices": bank.initial_indices,
                "target_indices": bank.target_indices,
                "base_noise": bank.base_noise,
            },
        }
    decision = locked_channel_decision(
        bank_reports, dominance_threshold=float(args.dominance_threshold)
    )
    hashes_after = {
        "model": model_checkpoint_sha256(model),
        "actor": network_sha256(model.network),
        "critic": network_sha256(critic_network),
    }
    if hashes_after != hashes_before:
        raise RuntimeError("channel audit mutated a model or network")
    report = {
        "protocol": {
            "posthoc_train_only": True,
            "policy_updates": False,
            "adjoint_refits": False,
            "factorial_switches": [
                "query_prefix_law",
                "population_law_in_lions_derivative",
                "conditional_future_continuation_law",
            ],
            "complete_factorial_cells": 8,
            "direct_antithetic_branch_average": True,
            "invented_projection_regression": False,
            "common_random_numbers_within_bank": True,
            "fit_confirmation_independent": True,
            "componentwise_weighted_lions_sources": True,
            "locked_dominance_threshold": float(args.dominance_threshold),
        },
        "proposal": dict(proposal.__dict__),
        "operator_beta": beta,
        "effective_lambda_scale": beta * float(full_training.lambda_scale),
        "source_equal_kl_run": str(equal_kl_dir),
        "bank_fingerprints": {
            "fit": fit_bank.fingerprint,
            "confirmation": confirmation_bank.fingerprint,
        },
        "banks": bank_reports,
        "decision": decision,
        "nonmutation": {"passed": True, "before": hashes_before, "after": hashes_after},
    }
    write_json(run_dir / "channel_audit_report.json", report)
    torch.save(bank_artifacts, run_dir / "channel_audit_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(f"wrote channel decomposition to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
