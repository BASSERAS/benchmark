#!/usr/bin/env python3
"""Audit direct causal-ridge deployment of the residual-objective R adjoint."""

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

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    DirectRidgeRAdjointPolicy,
    GenericCausalFeatureMap,
    ResidualAwareCausalFeatureMap,
    replicated_target_metrics,
    select_ridge_projection,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    LawEvaluation,
    ScenarioCollocationBank,
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
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
    _build_discrepancies,
    _load_paths,
)


DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "runs" / "residual_volatility_first_operator_corrected_parts_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-paths", type=int, default=256)
    parser.add_argument("--fit-target-paths", type=int, default=1024)
    parser.add_argument("--fit-branches", type=int, default=16)
    parser.add_argument("--validation-paths", type=int, default=128)
    parser.add_argument("--validation-target-paths", type=int, default=512)
    parser.add_argument("--validation-branches", type=int, default=16)
    parser.add_argument("--audit-paths", type=int, default=64)
    parser.add_argument("--audit-target-paths", type=int, default=512)
    parser.add_argument("--audit-branches", type=int, default=32)
    parser.add_argument("--replicate-branches", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2_608_501)
    parser.add_argument(
        "--ridge-grid", nargs="+", type=float, default=(1e-5, 1e-4, 1e-3, 1e-2)
    )
    return parser.parse_args()


def _matching_prefix_replication(
    *, template: ScenarioCollocationBank, independent: ScenarioCollocationBank, label: str
) -> ScenarioCollocationBank:
    return ScenarioCollocationBank(
        initial_indices=template.initial_indices,
        target_indices=independent.target_indices,
        base_noise=template.base_noise,
        branch_seed=independent.branch_seed,
        branch_stream_offset=independent.branch_stream_offset,
        num_branches=independent.num_branches,
        antithetic=independent.antithetic,
        beta=template.beta,
        dataset_fingerprint=template.dataset_fingerprint,
        label=label,
    )


def _replace_targets(evaluation: LawEvaluation, *, target_p: torch.Tensor, target_r: torch.Tensor) -> LawEvaluation:
    return LawEvaluation(
        paths=evaluation.paths,
        controls=evaluation.controls,
        policy_p=evaluation.policy_p,
        policy_r=evaluation.policy_r,
        target_p=target_p,
        target_r=target_r,
        objective=evaluation.objective,
        safety=evaluation.safety,
        metadata=evaluation.metadata,
    )


def _policy_audit(*, model: DiscreteMPModel, evaluation: LawEvaluation, policy: object) -> dict[str, object]:
    row, _ = fixed_law_cost_audit(
        model=model, evaluation=evaluation, adjoint_policy=policy
    )
    with torch.no_grad():
        _, _, controls = policy_moments_on_paths(
            model=model, policy=policy, paths=evaluation.paths
        )
    row.update(
        first_operator_alignment(
            model=model, evaluation=evaluation, fitted_controls=controls
        )
    )
    return row


def _datewise_mean_prediction(fit_target: torch.Tensor, eval_target: torch.Tensor) -> torch.Tensor:
    return fit_target.mean(dim=0, keepdim=True).expand_as(eval_target)


def _gate(
    *, current: dict[str, dict[str, object]], candidate: dict[str, dict[str, object]]
) -> dict[str, object]:
    thresholds = {
        "minimum_r_explained_energy": 0.0,
        "maximum_target_control_consensus_ratio": 1.0,
        "minimum_move_ratio": 0.25,
        "maximum_move_ratio": 0.80,
        "maximum_upper_activity_ratio": 0.80,
        "minimum_upper_activity_drop": 0.05,
        "minimum_alignment": 0.10,
        "maximum_cross_bank_move_gap": 0.35,
        "maximum_cross_bank_upper_gap": 0.10,
    }
    checks: dict[str, bool] = {}
    ratios: dict[str, object] = {}
    for bank in ("fit", "confirmation"):
        baseline = current[bank]
        row = candidate[bank]
        move_ratio = float(row["log_volatility_move_rms"]) / max(
            float(baseline["log_volatility_move_rms"]), torch.finfo(torch.float64).eps
        )
        consensus_ratio = float(row["fitted_target_log_volatility_consensus_rms"]) / max(
            float(baseline["fitted_target_log_volatility_consensus_rms"]),
            torch.finfo(torch.float64).eps,
        )
        upper = float(row["induced_sigma_upper_activity"])
        baseline_upper = float(baseline["induced_sigma_upper_activity"])
        upper_ratio = upper / max(baseline_upper, torch.finfo(torch.float64).eps)
        ratios[bank] = {
            "log_volatility_move": move_ratio,
            "target_control_consensus": consensus_ratio,
            "upper_activity": upper_ratio,
        }
        checks[f"{bank}_r_explained"] = float(row["r_ce_explained_energy"]) >= 0.0
        checks[f"{bank}_consensus_better_than_current"] = (
            consensus_ratio <= thresholds["maximum_target_control_consensus_ratio"]
        )
        checks[f"{bank}_movement_nontrivial_and_reduced"] = (
            thresholds["minimum_move_ratio"] <= move_ratio <= thresholds["maximum_move_ratio"]
        )
        checks[f"{bank}_upper_activity_reduced"] = (
            upper_ratio <= thresholds["maximum_upper_activity_ratio"]
            or upper <= baseline_upper - thresholds["minimum_upper_activity_drop"]
        )
        checks[f"{bank}_alignment_positive"] = (
            float(row["log_volatility_direction_cosine"])
            >= thresholds["minimum_alignment"]
        )
    fit_move = float(candidate["fit"]["log_volatility_move_rms"])
    confirmation_move = float(candidate["confirmation"]["log_volatility_move_rms"])
    checks["cross_bank_move_stable"] = (
        abs(fit_move - confirmation_move) / max(abs(fit_move), abs(confirmation_move), 1e-12)
        <= thresholds["maximum_cross_bank_move_gap"]
    )
    fit_upper = float(candidate["fit"]["induced_sigma_upper_activity"])
    confirmation_upper = float(candidate["confirmation"]["induced_sigma_upper_activity"])
    checks["cross_bank_upper_activity_stable"] = (
        abs(fit_upper - confirmation_upper) <= thresholds["maximum_cross_bank_upper_gap"]
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ratios_over_current_objective": ratios,
        "thresholds": thresholds,
        "next_phase": "reduced_spectral_audit" if all(checks.values()) else "stop_direct_ridge_deployment",
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Direct causal-ridge R audit",
        "",
        "Two high-branch R targets were reconstructed on identical held-out prefixes.",
        "Ridge hyperparameters were selected on a separate validation bank.",
        "",
        "## Learnability ceiling",
        "",
        "| Predictor | Explained energy | Residual RMS |",
        "|---|---:|---:|",
    ]
    for name, row in report["learnability_ceiling"]["predictors"].items():
        lines.append(
            f"| {name} | {row['prediction_explained_energy']:.3f} | "
            f"{row['prediction_residual_rms']:.4g} |"
        )
    replication = report["learnability_ceiling"]["replication"]
    lines.extend(
        [
            "",
            f"Replicate cosine: `{replication['replicate_cosine']:.3f}`; "
            f"replicate-difference RMS: `{replication['replicate_difference_rms']:.4g}`.",
            "",
            "## First operator",
            "",
            "| Bank | Policy | R explained | log-vol move | upper bound | target-control log-vol | alignment |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bank in ("fit", "confirmation"):
        for policy, by_bank in report["first_operator"].items():
            row = by_bank[bank]
            lines.append(
                f"| {bank} | {policy} | {row['r_ce_explained_energy']:.3f} | "
                f"{row['log_volatility_move_rms']:.4g} | "
                f"{row['induced_sigma_upper_activity']:.2%} | "
                f"{row['fitted_target_log_volatility_consensus_rms']:.4g} | "
                f"{row['log_volatility_direction_cosine']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Locked gate passed: `{report['gate']['passed']}`.",
            f"Next phase: `{report['gate']['next_phase']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    target_paths, manifest = _load_paths(
        args.dataset_root.resolve(), count=int(args.calibration_paths), device=device
    )
    configuration = manifest["configuration"]
    num_steps = int(configuration["sequence_length"]) - 1
    grid = DiscreteTimeGrid(T=float(num_steps) * float(configuration["dt"]), num_steps=num_steps)
    signal = torch.load(
        args.signal_run_dir.resolve() / "selected_residual_objective.pt",
        map_location="cpu",
        weights_only=False,
    )
    residual_discrepancy = signal["selected_discrepancy"]
    reference = signal["generic_reference"]
    discrepancies = _build_discrepancies(
        target_paths=target_paths,
        grid=grid,
        configuration=configuration,
        residual_discrepancy=residual_discrepancy,
        requested_variants=(
            "current_objective_eta1",
            "residual_conditional_eta4",
        ),
    )
    architecture = DiscreteMPArchitectureConfig(
        state_dim=1,
        noise_dim=1,
        hidden_dim=96,
        num_layers=1,
        adjoint_dim=1,
        noise_adjoint_dim=1,
    )
    ce = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
    )
    training = DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=2000,
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=50.0,
        ce_target_mode="ridge",
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
        grad_clip_norm=None,
        log_every=100,
        seed=0,
        observed_only=False,
    )

    def build_model(*, name: str, eta: float) -> DiscreteMPModel:
        control = SpecificEntropyDiagonalControl(
            dt=float(grid.dt),
            eta=float(eta),
            reference_kernel=reference,
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        model = DiscreteMPModel(
            architecture=architecture,
            grid=grid,
            dynamics_step=DriftVolatilityEulerStep(dt=float(grid.dt)),
            control_map=control,
            discrepancy=discrepancies[name],
            running_cost=control.running_cost,
            ce_estimator=ce,
            noise_ce_estimator=ce,
            training=training,
            init_seed=2_608_401,
        )
        model.network.to(device=device, dtype=torch.float32)
        return model

    current_model = build_model(name="current_objective_eta1", eta=1.0)
    residual_model = build_model(name="residual_conditional_eta4", eta=4.0)
    current_network = copy.deepcopy(current_model.network)
    residual_network = copy.deepcopy(residual_model.network)
    current_checkpoint = torch.load(
        args.checkpoint_root.resolve()
        / "current"
        / "fresh_adjoint_current_objective_eta1.pt",
        map_location=device,
        weights_only=False,
    )
    residual_checkpoint = torch.load(
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt",
        map_location=device,
        weights_only=False,
    )
    current_network.load_state_dict(current_checkpoint["state_dict"])
    residual_network.load_state_dict(residual_checkpoint["state_dict"])
    current_network.eval()
    residual_network.eval()
    law_policy = AdjointNetworkControlPolicy(
        model=residual_model, network=residual_model.network
    )
    context = ScenarioDataContext(
        initial_state_pool=target_paths.detach().cpu(),
        target_path_pool=target_paths.detach().cpu(),
        dataset_id=json.dumps(
            {
                "path": str(args.dataset_root.resolve()),
                "split": "train.npy deterministic prefix",
                "count": int(args.calibration_paths),
            },
            sort_keys=True,
        ),
    )

    fit_bank = build_scenario_collocation_bank(
        model=residual_model,
        context=context,
        path_count=int(args.fit_paths),
        target_count=int(args.fit_target_paths),
        num_branches=int(args.fit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed),
        label="ridge-fit",
    )
    validation_bank = build_scenario_collocation_bank(
        model=residual_model,
        context=context,
        path_count=int(args.validation_paths),
        target_count=int(args.validation_target_paths),
        num_branches=int(args.validation_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 10_000,
        label="ridge-validation",
    )
    audit_fit_bank, audit_confirmation_bank = build_fit_confirmation_banks(
        model=residual_model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 20_000,
    )
    replicate_source = build_scenario_collocation_bank(
        model=residual_model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.replicate_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 30_000,
        label="heldout-replication-source",
    )
    high_branch_template = ScenarioCollocationBank(
        initial_indices=audit_fit_bank.initial_indices,
        target_indices=audit_fit_bank.target_indices,
        base_noise=audit_fit_bank.base_noise,
        branch_seed=audit_fit_bank.branch_seed,
        branch_stream_offset=audit_fit_bank.branch_stream_offset,
        num_branches=int(args.replicate_branches),
        antithetic=True,
        beta=1.0,
        dataset_fingerprint=audit_fit_bank.dataset_fingerprint,
        label="heldout-replication-left",
    )
    high_branch_replica = _matching_prefix_replication(
        template=high_branch_template,
        independent=replicate_source,
        label="heldout-replication-right",
    )

    print("constructing ridge fit and validation targets", flush=True)
    fit_eval = rollout_policy_on_bank(
        model=residual_model,
        policy=law_policy,
        bank=fit_bank,
        data_context=context,
        full_training=training,
        query_batch_size=int(args.query_batch_size),
    )
    validation_eval = rollout_policy_on_bank(
        model=residual_model,
        policy=law_policy,
        bank=validation_bank,
        data_context=context,
        full_training=training,
        query_batch_size=int(args.query_batch_size),
    )
    generic_map = GenericCausalFeatureMap(ce)
    residual_map = ResidualAwareCausalFeatureMap(
        generic_features=ce,
        reference_kernel=reference,
        dt=float(grid.dt),
        num_steps=int(grid.num_steps),
        split_index=int(configuration["split_index"]),
        variance_floor=float(residual_discrepancy.prefix_variance_floor),
    )
    projections = {}
    selection = {}
    for name, feature_map in (("generic_ridge", generic_map), ("residual_aware_ridge", residual_map)):
        print(f"fitting {name}", flush=True)
        projection, selection_report = select_ridge_projection(
            feature_map=feature_map,
            fit_paths=fit_eval.paths,
            fit_targets=fit_eval.target_r,
            validation_paths=validation_eval.paths,
            validation_targets=validation_eval.target_r,
            ridge_grid=args.ridge_grid,
        )
        projections[name] = projection
        selection[name] = selection_report
        torch.save(projection, run_dir / f"{name}.pt")

    print("constructing two high-branch heldout target replications", flush=True)
    heldout_left = rollout_policy_on_bank(
        model=residual_model,
        policy=law_policy,
        bank=high_branch_template,
        data_context=context,
        full_training=training,
        query_batch_size=int(args.query_batch_size),
    )
    heldout_right = rollout_policy_on_bank(
        model=residual_model,
        policy=law_policy,
        bank=high_branch_replica,
        data_context=context,
        full_training=training,
        query_batch_size=int(args.query_batch_size),
    )
    torch.testing.assert_close(heldout_left.paths, heldout_right.paths, rtol=0.0, atol=0.0)
    torch.testing.assert_close(heldout_left.controls, heldout_right.controls, rtol=0.0, atol=0.0)
    target_consensus = 0.5 * (heldout_left.target_r + heldout_right.target_r)
    with torch.no_grad():
        _, gru_prediction, _ = policy_moments_on_paths(
            model=residual_model,
            policy=AdjointNetworkControlPolicy(model=residual_model, network=residual_network),
            paths=heldout_left.paths,
        )
    predictions = {
        "datewise_mean": _datewise_mean_prediction(fit_eval.target_r, target_consensus),
        "current_gru": gru_prediction,
        **{
            name: projection.predict_all(heldout_left.paths)
            for name, projection in projections.items()
        },
    }
    predictor_metrics = {
        name: replicated_target_metrics(
            prediction=prediction,
            target_left=heldout_left.target_r,
            target_right=heldout_right.target_r,
        )
        for name, prediction in predictions.items()
    }
    replication = replicated_target_metrics(
        prediction=target_consensus,
        target_left=heldout_left.target_r,
        target_right=heldout_right.target_r,
    )

    print("auditing first operator on independent banks", flush=True)
    policies = {
        "current_objective_gru": (
            current_model,
            AdjointNetworkControlPolicy(model=current_model, network=current_network),
        ),
        "residual_eta4_gru": (
            residual_model,
            AdjointNetworkControlPolicy(model=residual_model, network=residual_network),
        ),
        "residual_eta4_generic_ridge_r": (
            residual_model,
            DirectRidgeRAdjointPolicy(
                model=residual_model,
                p_network=residual_network,
                r_projection=projections["generic_ridge"],
            ),
        ),
        "residual_eta4_residual_aware_ridge_r": (
            residual_model,
            DirectRidgeRAdjointPolicy(
                model=residual_model,
                p_network=residual_network,
                r_projection=projections["residual_aware_ridge"],
            ),
        ),
    }
    first_operator: dict[str, dict[str, object]] = {name: {} for name in policies}
    bank_fingerprints = {
        "ridge_fit": fit_bank.fingerprint,
        "ridge_validation": validation_bank.fingerprint,
        "audit_fit": audit_fit_bank.fingerprint,
        "audit_confirmation": audit_confirmation_bank.fingerprint,
        "heldout_left": high_branch_template.fingerprint,
        "heldout_right": high_branch_replica.fingerprint,
    }
    for bank_name, bank in (("fit", audit_fit_bank), ("confirmation", audit_confirmation_bank)):
        eta_evaluation = rollout_policy_on_bank(
            model=residual_model,
            policy=law_policy,
            bank=bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
        )
        current_evaluation = rollout_policy_on_bank(
            model=current_model,
            policy=AdjointNetworkControlPolicy(model=current_model, network=current_model.network),
            bank=bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
        )
        torch.testing.assert_close(eta_evaluation.paths, current_evaluation.paths)
        for name, (model, policy) in policies.items():
            evaluation = current_evaluation if name == "current_objective_gru" else eta_evaluation
            first_operator[name][bank_name] = _policy_audit(
                model=model, evaluation=evaluation, policy=policy
            )

    gate = _gate(
        current=first_operator["current_objective_gru"],
        candidate=first_operator["residual_eta4_residual_aware_ridge_r"],
    )
    report = {
        "protocol": {
            "seed": int(args.seed),
            "test_paths_used": False,
            "same_prefix_high_branch_replication": True,
            "ridge_selected_on_separate_validation_bank": True,
            "P_representation": "fresh residual-eta4 GRU P head",
            "R_representation": "frozen directly deployed causal ridge",
            "control_source": "unchanged specific-entropy Hamiltonian map",
            "volatility_eta": 4.0,
            "residual_objective": True,
        },
        "effective_settings": {
            "cli": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "training": asdict(training),
        },
        "dataset_fingerprint": context.fingerprint,
        "scenario_banks": bank_fingerprints,
        "ridge_selection": selection,
        "feature_dimensions": {
            "generic": len(generic_map.feature_names),
            "residual_aware": len(residual_map.feature_names),
            "residual_aware_names": list(residual_map.feature_names),
        },
        "learnability_ceiling": {
            "replication": replication,
            "predictors": predictor_metrics,
        },
        "first_operator": first_operator,
        "gate": gate,
    }
    write_json(run_dir / "report.json", report)
    (run_dir / "SUMMARY.md").write_text(_markdown(report), encoding="utf-8")
    torch.save(
        {
            "target_left": heldout_left.target_r.detach().cpu(),
            "target_right": heldout_right.target_r.detach().cpu(),
            "predictions": {key: value.detach().cpu() for key, value in predictions.items()},
        },
        run_dir / "learnability_tensors.pt",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
