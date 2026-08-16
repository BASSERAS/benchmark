#!/usr/bin/env python3
"""Run the locked first-operator gate for residual conditional volatility."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    network_sha256,
    policy_moments_on_paths,
    rollout_policy_on_bank,
)
from path_dt_experiments.lifted_poll_solver import (  # noqa: E402
    AdjointRefitConfig,
    refit_adjoint_network,
)
from path_dt_experiments.residual_volatility_gate import (  # noqa: E402
    first_operator_alignment,
    residual_phase_one_gate,
)
from path_dt_experiments.running_cost_curvature_audit import (  # noqa: E402
    fixed_law_cost_audit,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


DEFAULT_DATASET = (
    REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"
)
DEFAULT_SIGNAL_RUN = (
    REPO_ROOT
    / "runs"
    / "residual_volatility_signal_derivative_drift_corrected_seed0_20260803"
)
VARIANTS = (
    "current_objective_eta1",
    "residual_conditional_eta1",
    "residual_conditional_eta4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--refit-paths", type=int, default=512)
    parser.add_argument("--refit-target-paths", type=int, default=1024)
    parser.add_argument("--refit-branches", type=int, default=8)
    parser.add_argument("--audit-paths", type=int, default=64)
    parser.add_argument("--audit-target-paths", type=int, default=256)
    parser.add_argument("--audit-branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--refit-inner-batch-size", type=int, default=128)
    parser.add_argument("--refit-max-steps", type=int, default=2000)
    parser.add_argument("--refit-eval-every", type=int, default=50)
    parser.add_argument("--refit-patience", type=int, default=12)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    parser.add_argument("--refit-seed", type=int, default=2_608_411)
    parser.add_argument("--audit-seed", type=int, default=2_608_421)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=VARIANTS
    )
    return parser.parse_args()


def _load_paths(
    root: Path, *, count: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prices = np.asarray(np.load(root / "train.npy"), dtype=np.float64)[: int(count)]
    if int(prices.shape[0]) != int(count):
        raise ValueError("calibration-paths exceeds the available training paths")
    values = np.log(prices / prices[:, :1])
    return (
        torch.from_numpy(values).unsqueeze(-1).to(
            device=device, dtype=torch.float32
        ),
        manifest,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_discrepancies(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    configuration: dict[str, object],
    residual_discrepancy: object,
    requested_variants: tuple[str, ...] = VARIANTS,
) -> dict[str, CompositePathFunctionalDiscrepancy]:
    base = build_heston_discrepancy(
        num_steps=int(grid.num_steps),
        include_acf=True,
        acf_lags=(1, 2, 5, 10),
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    requested = tuple(dict.fromkeys(str(value) for value in requested_variants))
    invalid = set(requested) - set(VARIANTS)
    if invalid:
        raise ValueError(f"unknown residual-volatility variants: {sorted(invalid)}")
    current = None
    if "current_objective_eta1" in requested:
        joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
            target_paths,
            grid=grid,
            split_index=int(configuration["split_index"]),
            future_horizon=int(configuration["future_horizon"]),
            prefix_windows=(16, 32, 64),
            recent_return_window=16,
            bandwidths=(0.5, 1.0, 2.0),
        )
        current = CompositePathFunctionalDiscrepancy(
            blocks=tuple(base.blocks)
            + (
                WeightedDiscrepancy(
                    name="volatility_law_joint_prefix_future_rv",
                    weight=1.0,
                    discrepancy=joint,
                ),
            )
        )
    residual_base = tuple(
        block
        for block in base.blocks
        if block.name != "volatility_law_global_rv"
    )
    if len(residual_base) != len(base.blocks) - 1:
        raise RuntimeError("the current objective has no unique global-RV block")
    residual = CompositePathFunctionalDiscrepancy(
        blocks=residual_base
        + (
            WeightedDiscrepancy(
                name="volatility_law_reference_residual_conditional_rv",
                weight=1.0,
                discrepancy=residual_discrepancy,
            ),
        )
    )
    variants = {
        "residual_conditional_eta1": residual,
        "residual_conditional_eta4": residual,
    }
    if current is not None:
        variants["current_objective_eta1"] = current
    return {name: variants[name] for name in requested}


def _summary_markdown(report: dict[str, object]) -> str:
    gate = report.get("phase_one_gate")
    lines = [
        "# Residual-volatility first-operator audit",
        "",
        "All variants start from the exact zero-adjoint causal reference at full",
        "discrepancy strength. Each adjoint is independently refitted with the",
        "unchanged normalized P/R loss.",
        "",
        f"- Gate passed: `{None if gate is None else gate['passed']}`",
        f"- Next phase: `{'aggregate_partial_runs' if gate is None else gate['next_phase']}`",
        "",
        "| Bank | Variant | P CE rel. | R CE rel. | R explained | log-vol move | upper bound | target-control log-vol |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in ("fit", "confirmation"):
        for name in report["audit"]:
            row = report["audit"][name][bank]
            lines.append(
                f"| {bank} | {name} | {row['p_ce_relative_residual']:.3f} | "
                f"{row['r_ce_relative_residual']:.3f} | "
                f"{row['r_ce_explained_energy']:.3f} | "
                f"{row['log_volatility_move_rms']:.4g} | "
                f"{row['induced_sigma_upper_activity']:.2%} | "
                f"{row['fitted_target_log_volatility_consensus_rms']:.4g} |"
            )
    if gate is not None:
        lines.extend(["", "## Locked gate", ""])
        for name, details in gate["variants"].items():
            lines.append(f"### {name}: `{details['passed']}`")
            lines.append("")
            for check, passed in details["checks"].items():
                lines.append(f"- {check}: `{passed}`")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = ensure_run_dir(args.run_dir.resolve())
    target_paths, manifest = _load_paths(
        args.dataset_root.resolve(),
        count=int(args.calibration_paths),
        device=device,
    )
    configuration = manifest["configuration"]
    if not isinstance(configuration, dict):
        raise TypeError("dataset manifest configuration must be a mapping")
    num_steps = int(configuration["sequence_length"]) - 1
    grid = DiscreteTimeGrid(
        T=float(num_steps) * float(configuration["dt"]), num_steps=num_steps
    )
    signal_path = args.signal_run_dir.resolve() / "selected_residual_objective.pt"
    signal = torch.load(signal_path, map_location="cpu", weights_only=False)
    residual_discrepancy = signal["selected_discrepancy"]
    reference = signal["generic_reference"]
    if reference is not residual_discrepancy.reference_kernel:
        raise RuntimeError("selected discrepancy and control do not share a reference")
    requested_variants = tuple(dict.fromkeys(str(value) for value in args.variants))
    discrepancies = _build_discrepancies(
        target_paths=target_paths,
        grid=grid,
        configuration=configuration,
        residual_discrepancy=residual_discrepancy,
        requested_variants=requested_variants,
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
        seed=int(args.seed),
        observed_only=False,
    )
    models: dict[str, DiscreteMPModel] = {}
    for name, discrepancy in discrepancies.items():
        eta = 4.0 if name.endswith("eta4") else 1.0
        control = SpecificEntropyDiagonalControl(
            dt=float(grid.dt),
            eta=eta,
            reference_kernel=reference,
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        model = DiscreteMPModel(
            architecture=architecture,
            grid=grid,
            dynamics_step=DriftVolatilityEulerStep(dt=float(grid.dt)),
            control_map=control,
            discrepancy=discrepancy,
            running_cost=control.running_cost,
            ce_estimator=ce,
            noise_ce_estimator=ce,
            training=training,
            init_seed=int(args.network_init_seed),
        )
        model.network.to(device=device, dtype=torch.float32)
        models[name] = model
    common_initial_network = next(iter(models.values())).network
    initial_hash = network_sha256(common_initial_network)
    for model in models.values():
        if network_sha256(model.network) != initial_hash:
            raise RuntimeError("variants did not receive the common zero-adjoint initialization")
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
    bank_model = next(iter(models.values()))
    refit_fit_bank, refit_validation_bank = build_fit_confirmation_banks(
        model=bank_model,
        context=context,
        path_count=int(args.refit_paths),
        target_count=int(args.refit_target_paths),
        num_branches=int(args.refit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.refit_seed),
    )
    audit_fit_bank, audit_confirmation_bank = build_fit_confirmation_banks(
        model=bank_model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.audit_seed),
    )
    law_policy = AdjointNetworkControlPolicy(
        model=bank_model, network=common_initial_network
    )
    refits = {}
    law_fingerprints = {}
    for index, (name, model) in enumerate(models.items()):
        print(f"{name}: constructing fixed-law fit targets", flush=True)
        fit = rollout_policy_on_bank(
            model=model,
            policy=law_policy,
            bank=refit_fit_bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
        )
        print(f"{name}: constructing fixed-law validation targets", flush=True)
        validation = rollout_policy_on_bank(
            model=model,
            policy=law_policy,
            bank=refit_validation_bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
        )
        law_fingerprints[name] = {
            "fit_paths": float(fit.paths.detach().double().sum().item()),
            "fit_controls": float(fit.controls.detach().double().sum().item()),
            "validation_paths": float(validation.paths.detach().double().sum().item()),
            "validation_controls": float(validation.controls.detach().double().sum().item()),
        }

        def progress(message: str, *, label: str = name) -> None:
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
            training=training,
            config=AdjointRefitConfig(
                inner_batch_size=int(args.refit_inner_batch_size),
                max_steps=int(args.refit_max_steps),
                eval_every=int(args.refit_eval_every),
                patience=int(args.refit_patience),
                seed=int(args.refit_seed) + 10_000,
            ),
            progress_callback=progress,
        )
        refits[name] = result
        torch.save(
            {
                "variant": name,
                "state_dict": result.state_dict,
                "refit_report": result.report,
                "common_initial_network_sha256": initial_hash,
                "objective_blocks": [
                    {"name": block.name, "weight": float(block.weight)}
                    for block in model.discrepancy.blocks
                ],
            },
            run_dir / f"fresh_adjoint_{name}.pt",
        )
    if len({json.dumps(value, sort_keys=True) for value in law_fingerprints.values()}) != 1:
        raise RuntimeError("variants did not use the same exact reference law")

    audit: dict[str, dict[str, object]] = {name: {} for name in models}
    shared_laws: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, model in models.items():
        adjoint_policy = AdjointNetworkControlPolicy(
            model=model, network=refits[name].network
        )
        for bank_name, bank in (
            ("fit", audit_fit_bank),
            ("confirmation", audit_confirmation_bank),
        ):
            print(f"{name}: auditing {bank_name} bank", flush=True)
            evaluation = rollout_policy_on_bank(
                model=model,
                policy=law_policy,
                bank=bank,
                data_context=context,
                full_training=training,
                query_batch_size=int(args.query_batch_size),
            )
            if bank_name in shared_laws:
                torch.testing.assert_close(evaluation.paths, shared_laws[bank_name][0])
                torch.testing.assert_close(evaluation.controls, shared_laws[bank_name][1])
            else:
                shared_laws[bank_name] = (
                    evaluation.paths.detach().clone(),
                    evaluation.controls.detach().clone(),
                )
            row, tensors = fixed_law_cost_audit(
                model=model,
                evaluation=evaluation,
                adjoint_policy=adjoint_policy,
            )
            with torch.no_grad():
                _, _, fitted_controls = policy_moments_on_paths(
                    model=model,
                    policy=adjoint_policy,
                    paths=evaluation.paths,
                )
            row.update(
                first_operator_alignment(
                    model=model,
                    evaluation=evaluation,
                    fitted_controls=fitted_controls,
                )
            )
            epsilon = torch.finfo(torch.float64).eps
            row["p_ce_relative_residual"] = float(
                row["p_ce_residual_rms"]
            ) / max(float(row["p_target_rms"]), epsilon)
            row["r_ce_relative_residual"] = float(
                row["r_ce_residual_rms"]
            ) / max(float(row["r_target_rms"]), epsilon)
            row["bank_fingerprint"] = evaluation.metadata["bank_fingerprint"]
            row["dataset_fingerprint"] = evaluation.metadata[
                "dataset_fingerprint"
            ]
            row["conditional_target_metadata"] = evaluation.metadata[
                "conditional_target"
            ]
            audit[name][bank_name] = row
            torch.save(
                tensors,
                run_dir / f"audit_tensors_{name}_{bank_name}.pt",
            )
    gate = residual_phase_one_gate(audit) if set(models) == set(VARIANTS) else None
    report = {
        "protocol": {
            "seed": int(args.seed),
            "beta": 1.0,
            "lambda_scale": 50.0,
            "running_cost_scaled": False,
            "exact_zero_adjoint_reference": True,
            "fresh_adjoint_per_variant": True,
            "common_initialization": True,
            "unchanged_p_r_loss": True,
            "test_paths_used": False,
            "signal_artifact": str(signal_path),
            "signal_artifact_sha256": _sha256_file(signal_path),
        },
        "effective_settings": {
            "cli": {
                name: (
                    [str(item) for item in value]
                    if isinstance(value, (list, tuple))
                    else str(value)
                    if isinstance(value, Path)
                    else value
                )
                for name, value in vars(args).items()
            },
            "training": asdict(training),
            "architecture": asdict(architecture),
        },
        "scenario_banks": {
            "refit_fit": refit_fit_bank.fingerprint,
            "refit_validation": refit_validation_bank.fingerprint,
            "audit_fit": audit_fit_bank.fingerprint,
            "audit_confirmation": audit_confirmation_bank.fingerprint,
        },
        "dataset_fingerprint": context.fingerprint,
        "objective_design": {
            name: {
                "eta": float(model.control_map.eta),
                "blocks": [
                    {"name": block.name, "weight": float(block.weight)}
                    for block in model.discrepancy.blocks
                ],
            }
            for name, model in models.items()
        },
        "common_initial_network_sha256": initial_hash,
        "law_fingerprints": law_fingerprints,
        "audit": audit,
        "phase_one_gate": gate,
    }
    write_json(run_dir / "report.json", report)
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
