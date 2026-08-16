#!/usr/bin/env python3
"""Locked symmetric active-set/KKT audit for the lifted MP operator."""

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
    DirectionalActiveSetAuditConfig,
    ScenarioDataContext,
    audit_directional_active_set_surface,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
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


DEFAULT_REPLAY_PROBE = (
    REPO_ROOT / "runs" / "lifted_collocation_probe_seed0_20260803" / "probe.json"
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
    parser.add_argument("--epsilon", type=float, default=0.025)
    parser.add_argument("--projected-kkt-gamma", type=float, default=1.0)
    parser.add_argument("--max-active-set-flip", type=float, default=0.05)
    parser.add_argument("--max-kkt-slope-asymmetry", type=float, default=0.25)
    return parser.parse_args()


def _locked_probe_checks(
    *,
    probe: dict[str, object],
    operator_beta: float,
    model_seed: int,
    seed: int,
    paths: int,
    target_paths: int,
    branches: int,
    query_batch_size: int,
    dataset_fingerprint: str,
    fit_bank_fingerprint: str,
    confirmation_bank_fingerprint: str,
) -> None:
    expected = {
        "operator_beta": float(operator_beta),
        "model_seed": int(model_seed),
        "probe_seed": int(seed),
        "paths_per_bank": int(paths),
        "target_paths_per_bank": int(target_paths),
        "conditional_branches": int(branches),
        "query_batch_size": int(query_batch_size),
        "dataset_fingerprint": str(dataset_fingerprint),
        "fit_bank_fingerprint": str(fit_bank_fingerprint),
        "confirmation_bank_fingerprint": str(confirmation_bank_fingerprint),
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = probe.get(key)
        if isinstance(expected_value, float):
            matched = actual is not None and abs(float(actual) - expected_value) <= 1e-12
        else:
            matched = actual == expected_value
        if not matched:
            mismatches[key] = {"expected": expected_value, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "directional audit does not reproduce the locked replay probe: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _summary_markdown(report: dict[str, object]) -> str:
    decision = report["decision"]
    baseline = report["baseline"]
    lines = [
        "# Lifted directional active-set audit",
        "",
        "This is a no-update symmetric surface audit. No candidate was deployed.",
        "",
        f"- Operator beta: `{report['operator_beta']:.10g}`",
        f"- Effective lambda: `{report['effective_lambda_scale']:.10g}`",
        f"- Perturbation epsilon: `{report['config']['epsilon']:.6g}`",
        f"- Maximum active-set flip fraction: `{decision['maximum_active_set_flip_fraction']:.6g}`",
        f"- Maximum KKT slope asymmetry: `{decision['maximum_kkt_slope_asymmetry']:.6g}`",
        f"- Central Jacobian supported: `{decision['central_jacobian_supported']}`",
        f"- Recommended next linearization: `{decision['recommended_next_linearization']}`",
        "",
        "## Baseline constrained residuals",
        "",
        "| Quantity | Fit bank | Confirmation bank |",
        "|---|---:|---:|",
    ]
    baseline_rows = (
        ("P CE residual", "p_ce_residual_rms"),
        ("R CE residual", "r_ce_residual_rms"),
        ("Drift consensus", "drift_consensus_rms"),
        ("Log-volatility consensus", "log_volatility_consensus_rms"),
        ("Adjoint projected-KKT residual", "adjoint_projected_kkt_residual_rms"),
        ("Conditional-target projected-KKT residual", "target_projected_kkt_residual_rms"),
        (r"$J_\beta$", "complete_objective_value"),
        ("Candidate upper-bound activity", "candidate_sigma_upper_activity"),
    )
    for label, key in baseline_rows:
        lines.append(
            f"| {label} | {baseline['fit'][key]:.6g} | "
            f"{baseline['confirmation'][key]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Symmetric directional response",
            "",
            "Central slopes use the same base noises and independently regenerated candidate laws and conditional targets.",
            "",
            "| Coordinate | Bank | KKT slope | Target-KKT slope | Objective slope | Max active flip |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for coordinate in report["surfaces"]["fit"]:
        for bank_name in ("fit", "confirmation"):
            row = report["surfaces"][bank_name][coordinate]
            slopes = row["slopes"]
            maximum_flip = max(
                float(row[side][key])
                for side in ("minus", "plus")
                for key in (
                    "adjoint_target_flip_fraction",
                    "conditional_target_flip_fraction",
                    "actual_control_flip_fraction",
                )
            )
            lines.append(
                f"| {coordinate} | {bank_name} | "
                f"{slopes['adjoint_projected_kkt_residual_rms']['central_slope']:.6g} | "
                f"{slopes['target_projected_kkt_residual_rms']['central_slope']:.6g} | "
                f"{slopes['complete_objective_value']['central_slope']:.6g} | "
                f"{maximum_flip:.6g} |"
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
    result = audit_directional_active_set_surface(
        model=model,
        law_policy=law_policy,
        adjoint_policy=adjoint_policy,
        data_context=context,
        fit_bank=fit_bank,
        confirmation_bank=confirmation_bank,
        full_training=full_training,
        query_batch_size=int(args.query_batch_size),
        config=DirectionalActiveSetAuditConfig(
            num_time_blocks=3,
            epsilon=float(args.epsilon),
            projected_kkt_gamma=float(args.projected_kkt_gamma),
            maximum_active_set_flip_fraction=float(args.max_active_set_flip),
            maximum_kkt_slope_asymmetry=float(args.max_kkt_slope_asymmetry),
        ),
    )
    if model_checkpoint_sha256(model) != checkpoint_before:
        raise RuntimeError("model checkpoint changed during directional audit")
    if network_sha256(model.network) != law_before:
        raise RuntimeError("law policy changed during directional audit")
    if network_sha256(adjoint_network) != adjoint_before:
        raise RuntimeError("adjoint policy changed during directional audit")

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
    write_json(run_dir / "directional_surface.json", result.report)
    torch.save(result.artifacts, run_dir / "directional_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(result.report), encoding="utf-8"
    )
    print(f"wrote directional active-set audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
