#!/usr/bin/env python3
"""No-update replay and residual audit for lifted law--adjoint consensus."""

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

from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    audit_lifted_consensus,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


DEFAULT_ADJOINT_ARTIFACTS = (
    REPO_ROOT / "runs" / "terminal_fixed_point_diagnostic_seed0_20260803"
    / "artifacts.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument(
        "--adjoint-artifacts", type=Path, default=DEFAULT_ADJOINT_ARTIFACTS
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_041)
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--target-paths", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4_096)
    return parser.parse_args()


def _load_adjoint_network(
    *,
    model,
    artifacts_path: Path,
) -> torch.nn.Module:
    payload = torch.load(
        artifacts_path, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict):
        raise ValueError("adjoint artifact must contain a dictionary")
    values = payload.get("best_candidate_network_values")
    if not isinstance(values, dict):
        raise ValueError(
            "adjoint artifact is missing best_candidate_network_values"
        )
    network = copy.deepcopy(model.network)
    named = dict(network.named_parameters())
    if set(values) != set(named):
        missing = sorted(set(named) - set(values))
        unexpected = sorted(set(values) - set(named))
        raise ValueError(
            f"adjoint parameter mismatch: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in named.items():
            value = values[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"adjoint parameter {name} is not a tensor")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    for module in network.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()
    network.eval()
    return network


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Lifted law--adjoint replay probe",
        "",
        "No law update, Gauss--Newton step, dual update, or acceptance decision was performed.",
        "",
        f"- Operator beta: `{report['operator_beta']:.10g}`",
        f"- Effective lambda: `{report['effective_lambda_scale']:.10g}`",
        f"- Dataset SHA-256: `{report['dataset_fingerprint']}`",
        f"- Fit-bank replay gate: `{report['fit_replay']['passed']}` "
        f"(`{report['fit_replay']['required_mode']}`)",
        f"- Model checkpoint unchanged: `{report['nonmutation']['model_unchanged']}`",
        f"- Law policy unchanged: `{report['nonmutation']['law_policy_unchanged']}`",
        f"- Adjoint policy unchanged: `{report['nonmutation']['adjoint_policy_unchanged']}`",
        "",
        "| Quantity | Fit bank | Confirmation bank |",
        "|---|---:|---:|",
    ]
    fit = report["fit"]
    confirmation = report["confirmation"]
    rows = (
        ("P CE residual", "p_ce_residual_rms"),
        ("R CE residual", "r_ce_residual_rms"),
        ("Drift consensus", "drift_consensus_rms"),
        ("Log-volatility consensus", "log_volatility_consensus_rms"),
        (r"$J_\beta$", "complete_objective_value"),
        ("Transition KL", "transition_kl_mean"),
    )
    for label, key in rows:
        lines.append(f"| {label} | {fit[key]:.6g} | {confirmation[key]:.6g} |")
    for label, safety_group, safety_key in (
        (
            "Accepted denominator margin",
            "accepted_control_safety",
            "specific_entropy_denominator_margin_min",
        ),
        (
            "Adjoint denominator margin",
            "adjoint_control_safety",
            "specific_entropy_denominator_margin_min",
        ),
        (
            "Accepted upper-bound activity",
            "accepted_control_safety",
            "sigma_upper_bound_activity",
        ),
        (
            "Adjoint upper-bound activity",
            "adjoint_control_safety",
            "sigma_upper_bound_activity",
        ),
    ):
        lines.append(
            f"| {label} | {fit[safety_group][safety_key]:.6g} | "
            f"{confirmation[safety_group][safety_key]:.6g} |"
        )
    lines.extend(
        [
            "",
            "The transition KL compares the adjoint-induced proposal with the accepted law on identical prefixes.",
        ]
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

    continuation = args.continuation_run_dir.resolve()
    adjoint_artifacts = args.adjoint_artifacts.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model, target_paths, checkpoint_training, continuation_report, design = (
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
        raise ValueError("probe expects the first terminal rejected stage")
    operator_beta = float(terminal["beta_target"])
    full_lambda_scale = float(continuation_report["final_lambda_scale"])
    full_training = replace(
        checkpoint_training,
        lambda_scale=full_lambda_scale,
        grad_clip_norm=None,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
    )
    effective_lambda = full_lambda_scale * operator_beta
    if abs(effective_lambda - float(terminal["lambda_scale"])) > 1e-10:
        raise RuntimeError("continuation report scales the discrepancy inconsistently")

    dataset = design.get("dataset")
    dataset_identity = (
        json.dumps(dataset, sort_keys=True)
        if isinstance(dataset, dict)
        else str(dataset)
    )
    context = ScenarioDataContext(
        initial_state_pool=target_paths.detach().cpu(),
        target_path_pool=target_paths.detach().cpu(),
        dataset_id=dataset_identity,
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
    adjoint_network = _load_adjoint_network(
        model=model, artifacts_path=adjoint_artifacts
    )
    model.network.eval()
    law_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    adjoint_policy = AdjointNetworkControlPolicy(
        model=model, network=adjoint_network
    )
    checkpoint_before = model_checkpoint_sha256(model)
    law_before = network_sha256(model.network)
    adjoint_before = network_sha256(adjoint_network)
    result = audit_lifted_consensus(
        model=model,
        law_policy=law_policy,
        adjoint_policy=adjoint_policy,
        data_context=context,
        fit_bank=fit_bank,
        confirmation_bank=confirmation_bank,
        full_training=full_training,
        query_batch_size=int(args.query_batch_size),
    )
    if model_checkpoint_sha256(model) != checkpoint_before:
        raise RuntimeError("model checkpoint changed during the probe")
    if network_sha256(model.network) != law_before:
        raise RuntimeError("law policy changed during the probe")
    if network_sha256(adjoint_network) != adjoint_before:
        raise RuntimeError("adjoint policy changed during the probe")

    result.report.update(
        {
            "continuation_run_dir": str(continuation),
            "adjoint_artifacts": str(adjoint_artifacts),
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
    write_json(run_dir / "probe.json", result.report)
    torch.save(result.artifacts, run_dir / "residual_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(result.report), encoding="utf-8"
    )
    print(f"wrote lifted collocation replay probe to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
