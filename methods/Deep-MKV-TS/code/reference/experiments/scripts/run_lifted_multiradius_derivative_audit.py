#!/usr/bin/env python3
"""Locked no-update multiradius signed-derivative audit."""

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
    MultiRadiusDerivativeAuditConfig,
    ScenarioDataContext,
    audit_multiradius_signed_derivatives,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_lifted_collocation_probe import (  # noqa: E402
    DEFAULT_ADJOINT_ARTIFACTS,
    _load_adjoint_network,
)
from run_lifted_directional_active_set_audit import (  # noqa: E402
    DEFAULT_REPLAY_PROBE,
    _locked_probe_checks,
)
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS)
    parser.add_argument("--replay-probe", type=Path, default=DEFAULT_REPLAY_PROBE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_041)
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--target-paths", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4_096)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=(0.025, 0.0125, 0.00625),
    )
    parser.add_argument("--projected-kkt-gamma", type=float, default=1.0)
    parser.add_argument("--minimum-stable-cosine", type=float, default=0.8)
    parser.add_argument(
        "--maximum-stable-relative-difference", type=float, default=0.5
    )
    return parser.parse_args()


def _aggregate_rows(report: dict[str, object]) -> list[str]:
    decision = report["decision"]
    convergence = decision["small_radius_convergence"]
    agreement = decision["smallest_radius_bank_agreement"]
    rows = [
        "| Signed residual system | Radius cosine min | Radius relative difference max | Radius passes | Bank cosine min | Bank relative difference max | Bank passes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in decision["systems"]:
        within = convergence[system]
        between = agreement[system]
        rows.append(
            f"| {system} | {within['minimum_cosine_similarity']:.6g} | "
            f"{within['maximum_relative_vector_difference']:.6g} | "
            f"{within['passing_count']}/{within['count']} | "
            f"{between['minimum_cosine_similarity']:.6g} | "
            f"{between['maximum_relative_vector_difference']:.6g} | "
            f"{between['passing_count']}/{between['count']} |"
        )
    return rows


def _small_radius_rows(report: dict[str, object]) -> list[str]:
    epsilons = report["config"]["epsilons"]
    larger = f"{float(epsilons[-2]):.10g}"
    smaller = f"{float(epsilons[-1]):.10g}"
    pair_index = len(epsilons) - 2
    agreement = report["bank_agreement"][smaller]
    rows = [
        "| Coordinate | Side | Fit target-KKT radius cos | Confirm target-KKT radius cos | Target-KKT bank cos | Lifted bank cos |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for coordinate in agreement:
        for side in ("minus", "plus"):
            fit = report["radius_convergence"]["fit"][coordinate][side][
                "target_projected_kkt_residual"
            ][pair_index]
            confirmation = report["radius_convergence"]["confirmation"][
                coordinate
            ][side]["target_projected_kkt_residual"][pair_index]
            target_bank = agreement[coordinate][side][
                "target_projected_kkt_residual"
            ]
            lifted_bank = agreement[coordinate][side]["lifted_consensus"]
            rows.append(
                f"| {coordinate} | {side} | "
                f"{fit['cosine_similarity']:.6g} | "
                f"{confirmation['cosine_similarity']:.6g} | "
                f"{target_bank['cosine_similarity']:.6g} | "
                f"{lifted_bank['cosine_similarity']:.6g} |"
            )
    rows.insert(
        0,
        f"Small-radius convergence compares `{larger}` with `{smaller}`; bank agreement is evaluated at `{smaller}`.",
    )
    rows.insert(1, "")
    return rows


def _summary_markdown(report: dict[str, object]) -> str:
    decision = report["decision"]
    lines = [
        "# Lifted multiradius signed-derivative audit",
        "",
        "No candidate, solver step, or diagnostic policy was deployed.",
        "",
        f"- Operator beta: `{report['operator_beta']:.10g}`",
        f"- Effective lambda: `{report['effective_lambda_scale']:.10g}`",
        f"- Radii: `{report['config']['epsilons']}`",
        f"- Locked stability gate: cosine >= `{report['config']['minimum_stable_cosine']}` "
        f"and relative vector difference <= `{report['config']['maximum_stable_relative_difference']}`",
        f"- One-sided columns stable: `{decision['one_sided_columns_stable']}`",
        f"- Diagnostic recommendation: `{decision['recommended_next_solver']}`",
        "",
        "## Aggregate stability",
        "",
        *_aggregate_rows(report),
        "",
        "## Small-radius signed-column cosines",
        "",
        *_small_radius_rows(report),
        "",
        "Full signed sample columns are compared across radii within each bank. Across independent banks, the comparison uses time-aligned signed mean profiles rather than unrelated pathwise coordinates.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if int(args.paths) < 1 or int(args.target_paths) < 1:
        raise ValueError("paths and target-paths must be positive")
    if int(args.branches) < 2 or int(args.branches) % 2 != 0:
        raise ValueError("branches must be an even integer of at least two")
    if int(args.query_batch_size) < int(args.paths) * int(args.branches):
        raise ValueError("query-batch-size must hold one complete branch population")

    continuation = args.continuation_run_dir.resolve()
    adjoint_artifacts = args.adjoint_artifacts.resolve()
    replay_probe_path = args.replay_probe.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    with replay_probe_path.open("r", encoding="utf-8") as handle:
        replay_probe = json.load(handle)

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
        raise ValueError("audit expects the first terminal rejected stage")
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
    _locked_probe_checks(
        probe=replay_probe,
        operator_beta=operator_beta,
        model_seed=int(args.model_seed),
        seed=int(args.seed),
        paths=int(args.paths),
        target_paths=int(args.target_paths),
        branches=int(args.branches),
        query_batch_size=int(args.query_batch_size),
        dataset_fingerprint=context.fingerprint,
        fit_bank_fingerprint=fit_bank.fingerprint,
        confirmation_bank_fingerprint=confirmation_bank.fingerprint,
    )

    adjoint_network = _load_adjoint_network(model=model, artifacts_path=adjoint_artifacts)
    model.network.eval()
    law_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    adjoint_policy = AdjointNetworkControlPolicy(model=model, network=adjoint_network)
    checkpoint_before = model_checkpoint_sha256(model)
    law_before = network_sha256(model.network)
    adjoint_before = network_sha256(adjoint_network)
    result = audit_multiradius_signed_derivatives(
        model=model,
        law_policy=law_policy,
        adjoint_policy=adjoint_policy,
        data_context=context,
        fit_bank=fit_bank,
        confirmation_bank=confirmation_bank,
        full_training=full_training,
        query_batch_size=int(args.query_batch_size),
        config=MultiRadiusDerivativeAuditConfig(
            epsilons=tuple(float(value) for value in args.epsilons),
            num_time_blocks=3,
            projected_kkt_gamma=float(args.projected_kkt_gamma),
            minimum_stable_cosine=float(args.minimum_stable_cosine),
            maximum_stable_relative_difference=float(
                args.maximum_stable_relative_difference
            ),
        ),
    )
    if model_checkpoint_sha256(model) != checkpoint_before:
        raise RuntimeError("model checkpoint changed during derivative audit")
    if network_sha256(model.network) != law_before:
        raise RuntimeError("law policy changed during derivative audit")
    if network_sha256(adjoint_network) != adjoint_before:
        raise RuntimeError("adjoint policy changed during derivative audit")

    result.report.update(
        {
            "continuation_run_dir": str(continuation),
            "adjoint_artifacts": str(adjoint_artifacts),
            "locked_replay_probe": str(replay_probe_path),
            "checkpoint_beta": float(continuation_report["accepted_beta"]),
            "operator_beta": operator_beta,
            "full_lambda_scale": full_lambda_scale,
            "effective_lambda_scale": effective_lambda,
            "model_seed": int(args.model_seed),
            "probe_seed": int(args.seed),
            "paths_per_bank": int(args.paths),
            "target_paths_per_bank": int(args.target_paths),
            "conditional_branches": int(args.branches),
            "query_batch_size": int(args.query_batch_size),
            "adjoint_source": "terminal fixed-point diagnostic best candidate",
        }
    )
    write_json(run_dir / "signed_derivative_report.json", result.report)
    torch.save(result.artifacts, run_dir / "signed_derivative_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(result.report), encoding="utf-8"
    )
    print(f"wrote multiradius derivative audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
