#!/usr/bin/env python3
"""Audit the terminal zero-drift, volatility-only MP first operator."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    DirectRidgeRAdjointPolicy,
    GenericCausalFeatureMap,
    ResidualAwareCausalFeatureMap,
    select_ridge_projection,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    ScenarioDataContext,
    build_fit_confirmation_banks,
    build_scenario_collocation_bank,
    policy_moments_on_paths,
    rollout_policy_on_bank,
)
from path_dt_experiments.residual_volatility_gate import (  # noqa: E402
    first_operator_alignment,
)
from path_dt_experiments.running_cost_curvature_audit import (  # noqa: E402
    fixed_law_cost_audit,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.volatility_only_solver import (  # noqa: E402
    VolatilityOnlyPDiagnosticReferencePolicy,
)
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)
from run_direct_ridge_r_spectral_matrix import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-paths", type=int, default=512)
    parser.add_argument("--fit-target-paths", type=int, default=1024)
    parser.add_argument("--fit-branches", type=int, default=16)
    parser.add_argument("--validation-paths", type=int, default=256)
    parser.add_argument("--validation-target-paths", type=int, default=512)
    parser.add_argument("--validation-branches", type=int, default=16)
    parser.add_argument("--audit-paths", type=int, default=128)
    parser.add_argument("--audit-target-paths", type=int, default=512)
    parser.add_argument("--audit-branches", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2_608_801)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    parser.add_argument(
        "--ridge-grid", nargs="+", type=float, default=(1e-5, 1e-4, 1e-3, 1e-2)
    )
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


def _restrict_to_volatility(model) -> None:
    old = model.control_map
    restricted = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=float(old.dt),
        eta=float(old.eta),
        reference_kernel=old.reference_kernel,
        sigma_min=float(old.sigma_min),
        sigma_max=None if old.sigma_max is None else float(old.sigma_max),
        denominator_eps=float(old.denominator_eps),
    )
    model.control_map = restricted
    model.running_cost = restricted.running_cost


def _audit_policy(*, model, evaluation, policy) -> dict[str, object]:
    row, _ = fixed_law_cost_audit(
        model=model,
        evaluation=evaluation,
        adjoint_policy=policy,
    )
    with torch.no_grad():
        p, r, controls = policy_moments_on_paths(
            model=model, policy=policy, paths=evaluation.paths
        )
        shifted = model.control_map.controls_from_moments(
            paths=evaluation.paths,
            expected_adjoint_next=p + 1000.0,
            expected_adjoint_noise_next=r,
        )
    row.update(
        first_operator_alignment(
            model=model,
            evaluation=evaluation,
            fitted_controls=controls,
        )
    )
    state_dim = int(evaluation.paths.shape[-1])
    row.update(
        {
            "maximum_absolute_drift": float(
                controls[..., :state_dim].detach().abs().max().item()
            ),
            "p_shift_control_maximum_absolute_difference": float(
                (shifted - controls).detach().abs().max().item()
            ),
            "r_projection_explained_energy": float(
                1.0
                - torch.mean((r - evaluation.target_r).double().pow(2))
                / torch.mean(
                    (
                        evaluation.target_r
                        - evaluation.target_r.mean(dim=0, keepdim=True)
                    ).double().pow(2)
                ).clamp_min(torch.finfo(torch.float64).eps)
            ),
        }
    )
    return row


def _gate(audit: dict[str, dict[str, object]]) -> dict[str, object]:
    checks = {}
    for bank in ("fit", "confirmation"):
        row = audit[bank]
        checks[f"{bank}_drift_exactly_zero"] = (
            float(row["maximum_absolute_drift"]) == 0.0
        )
        checks[f"{bank}_P_is_control_invariant"] = (
            float(row["p_shift_control_maximum_absolute_difference"]) == 0.0
        )
        checks[f"{bank}_R_has_positive_explained_energy"] = (
            float(row["r_projection_explained_energy"]) > 0.0
        )
        checks[f"{bank}_volatility_move_is_nontrivial"] = (
            float(row["log_volatility_move_rms"]) > 1e-5
        )
        checks[f"{bank}_direction_alignment_is_positive"] = (
            float(row["log_volatility_direction_cosine"]) > 0.0
        )
    fit_move = float(audit["fit"]["log_volatility_move_rms"])
    confirmation_move = float(audit["confirmation"]["log_volatility_move_rms"])
    checks["cross_bank_movement_agrees_within_35_percent"] = (
        abs(fit_move - confirmation_move)
        / max(abs(fit_move), abs(confirmation_move), 1e-12)
        <= 0.35
    )
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "next_phase": "three_dimensional_R_spectrum" if passed else "stop",
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model, target_paths = _construct_model(_model_args(args))
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
    reference_policy = VolatilityOnlyPDiagnosticReferencePolicy(
        model=model, p_network=p_network
    )
    context = ScenarioDataContext(
        initial_state_pool=target_paths.detach().cpu(),
        target_path_pool=target_paths.detach().cpu(),
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
    fit_bank = build_scenario_collocation_bank(
        model=model,
        context=context,
        path_count=int(args.fit_paths),
        target_count=int(args.fit_target_paths),
        num_branches=int(args.fit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed),
        label="volatility-only-ridge-fit",
    )
    validation_bank = build_scenario_collocation_bank(
        model=model,
        context=context,
        path_count=int(args.validation_paths),
        target_count=int(args.validation_target_paths),
        num_branches=int(args.validation_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 10_000,
        label="volatility-only-ridge-validation",
    )
    audit_fit_bank, audit_confirmation_bank = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 20_000,
    )
    print("constructing volatility-only ridge fit targets", flush=True)
    fit = rollout_policy_on_bank(
        model=model,
        policy=reference_policy,
        bank=fit_bank,
        data_context=context,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
    )
    validation = rollout_policy_on_bank(
        model=model,
        policy=reference_policy,
        bank=validation_bank,
        data_context=context,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
    )
    signal = torch.load(
        args.signal_run_dir.resolve() / "selected_residual_objective.pt",
        map_location="cpu",
        weights_only=False,
    )
    residual = signal["selected_discrepancy"]
    feature_map = ResidualAwareCausalFeatureMap(
        generic_features=GenericCausalFeatureMap(model.ce_estimator).generic_features,
        reference_kernel=signal["generic_reference"],
        dt=float(model.grid.dt),
        num_steps=int(model.grid.num_steps),
        split_index=int(residual.split_index),
        variance_floor=float(residual.prefix_variance_floor),
    )
    projection, selection = select_ridge_projection(
        feature_map=feature_map,
        fit_paths=fit.paths,
        fit_targets=fit.target_r,
        validation_paths=validation.paths,
        validation_targets=validation.target_r,
        ridge_grid=args.ridge_grid,
    )
    projection_path = run_dir / "residual_aware_ridge_r.pt"
    torch.save(projection, projection_path)
    candidate_policy = DirectRidgeRAdjointPolicy(
        model=model,
        p_network=p_network,
        r_projection=projection,
    )
    audit = {}
    for name, bank in (
        ("fit", audit_fit_bank),
        ("confirmation", audit_confirmation_bank),
    ):
        print(f"auditing {name} bank", flush=True)
        evaluation = rollout_policy_on_bank(
            model=model,
            policy=reference_policy,
            bank=bank,
            data_context=context,
            full_training=model.training,
            query_batch_size=int(args.query_batch_size),
        )
        audit[name] = _audit_policy(
            model=model,
            evaluation=evaluation,
            policy=candidate_policy,
        )
        audit[name]["bank_fingerprint"] = bank.fingerprint
    gate = _gate(audit)
    report = {
        "protocol": {
            "terminal_solver_experiment": True,
            "admissible_drift": "alpha identically zero",
            "controlled_coordinate": "bounded diagonal volatility only",
            "P_role": "regressed costate diagnostic; excluded from forward control",
            "R_role": "direct residual-aware causal ridge deployed through Hamiltonian map",
            "objective": "residual conditional volatility",
            "volatility_eta": 4.0,
            "fresh_R_projection_under_restricted_problem": True,
            "test_paths_used": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training": asdict(model.training),
        "scenario_banks": {
            "fit": fit_bank.fingerprint,
            "validation": validation_bank.fingerprint,
            "audit_fit": audit_fit_bank.fingerprint,
            "audit_confirmation": audit_confirmation_bank.fingerprint,
        },
        "ridge_selection": selection,
        "audit": audit,
        "gate": gate,
        "artifacts": {
            "P_checkpoint": str(checkpoint_path),
            "R_projection": str(projection_path.resolve()),
        },
    }
    write_json(run_dir / "report.json", report)
    (run_dir / "SUMMARY.md").write_text(
        "# Volatility-only MP first-operator audit\n\n"
        f"- Gate passed: `{gate['passed']}`\n"
        f"- Selected ridge: `{selection['selected_ridge']}`\n"
        f"- Fit/confirmation R explained energy: "
        f"`{audit['fit']['r_projection_explained_energy']:.4f}` / "
        f"`{audit['confirmation']['r_projection_explained_energy']:.4f}`\n"
        f"- Fit/confirmation log-volatility movement: "
        f"`{audit['fit']['log_volatility_move_rms']:.6g}` / "
        f"`{audit['confirmation']['log_volatility_move_rms']:.6g}`\n"
        f"- Maximum drift: `{max(audit['fit']['maximum_absolute_drift'], audit['confirmation']['maximum_absolute_drift']):.3g}`\n"
        f"- Maximum control change after adding 1000 to P: "
        f"`{max(audit['fit']['p_shift_control_maximum_absolute_difference'], audit['confirmation']['p_shift_control_maximum_absolute_difference']):.3g}`\n"
        f"- Next phase: `{gate['next_phase']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
