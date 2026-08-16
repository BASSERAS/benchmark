#!/usr/bin/env python3
"""Frozen noise/Jacobian/projection audit at the continuation branch barrier."""

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

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_causal_volatility_reference_kernel,
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
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    TerminalFixedPointDiagnosticConfig,
    audit_terminal_fixed_point,
)
from run_drawdown_memory_deep_mkv_ablation import _load_dataset  # noqa: E402


DEFAULT_CONTINUATION = (
    REPO_ROOT / "runs" / "drawdown_continuation_transition_kl_seed0_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_031)
    parser.add_argument("--residual-replicates", type=int, default=12)
    parser.add_argument("--common-evaluation-paths", type=int, default=1_024)
    parser.add_argument("--residual-fit-paths", type=int, default=1_024)
    parser.add_argument("--residual-target-batch-size", type=int, default=2_048)
    parser.add_argument("--projection-fit-paths", type=int, default=2_048)
    parser.add_argument("--projection-validation-paths", type=int, default=1_024)
    parser.add_argument("--projection-target-batch-size", type=int, default=2_048)
    parser.add_argument("--projection-restarts", type=int, default=3)
    parser.add_argument("--projection-inner-batch-size", type=int, default=256)
    parser.add_argument("--projection-max-steps", type=int, default=2_000)
    parser.add_argument("--projection-eval-every", type=int, default=50)
    parser.add_argument("--projection-patience", type=int, default=12)
    parser.add_argument("--time-blocks", type=int, default=3)
    parser.add_argument("--jacobian-replicates", type=int, default=3)
    parser.add_argument("--jacobian-fit-paths", type=int, default=512)
    parser.add_argument("--jacobian-validation-paths", type=int, default=512)
    parser.add_argument("--jacobian-target-batch-size", type=int, default=1_024)
    parser.add_argument("--finite-difference-step", type=float, default=0.05)
    parser.add_argument(
        "--secondary-finite-difference-step", type=float, default=0.025
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _build_frozen_model(
    *,
    continuation: Path,
    device: torch.device,
    model_seed: int,
) -> tuple[
    DiscreteMPModel,
    torch.Tensor,
    DiscreteMPTrainingConfig,
    dict[str, object],
    dict[str, object],
]:
    design = _load_json(continuation / "design.json")
    arm_dir = continuation / "generic_ce_continuation_extragradient"
    continuation_report = _load_json(arm_dir / "validated_nested_report.json")
    checkpoint = torch.load(
        arm_dir / "model_checkpoint.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("model checkpoint must contain a dictionary")
    dataset = design.get("dataset")
    if not isinstance(dataset, dict) or "train" not in dataset:
        raise ValueError("continuation design is missing the dataset")
    dataset_root = Path(str(dataset["train"])).resolve().parent
    target_paths, manifest, _ = _load_dataset(
        dataset_root,
        device=device,
        dtype=torch.float32,
    )
    configuration = manifest["configuration"]
    num_steps = int(configuration["sequence_length"]) - 1
    grid = DiscreteTimeGrid(
        T=num_steps * float(configuration["dt"]),
        num_steps=num_steps,
    )
    temporal = design.get("temporal_scaling")
    if not isinstance(temporal, dict):
        raise ValueError("continuation design is missing temporal scaling")
    if str(design.get("reference_basis")) != "generic_causal":
        raise ValueError("terminal diagnostic expects the generic causal reference")
    rolling = tuple(int(value) for value in temporal["rolling_windows"])
    ewma = tuple(int(value) for value in temporal["ewma_half_lives"])
    downside = tuple(int(value) for value in temporal["downside_windows"])
    acf_lags = tuple(int(value) for value in temporal["acf_lags"])
    prefix_windows = tuple(
        int(value) for value in temporal["joint_prefix_windows"]
    )
    recent_window = int(temporal["joint_recent_return_window"])
    reference_config = design.get("reference")
    if not isinstance(reference_config, dict):
        raise ValueError("continuation design is missing reference metadata")
    reference = fit_causal_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        rolling_windows=rolling,
        ewma_half_lives=ewma,
        downside_windows=downside,
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
        drift_ridge=1e-3,
        variance_ridges=(1e-5, 1e-4, 1e-3),
        crossfit_folds=int(reference_config["reference_crossfit_folds"]),
        crossfit_seed=1701 + int(model_seed),
        sigma_min=float(reference_config["reference_sigma_min"]),
        sigma_max=float(reference_config["reference_sigma_max"]),
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=float(reference_config["reference_sigma_min"]),
        sigma_max=float(reference_config["reference_sigma_max"]),
    )
    ce = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=rolling,
        ewma_half_lives=ewma,
        downside_windows=downside,
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
    )
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        acf_lags=acf_lags,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=int(configuration["split_index"]),
        future_horizon=int(configuration["future_horizon"]),
        prefix_windows=prefix_windows,
        recent_return_window=recent_window,
        bandwidths=(0.5, 1.0, 2.0),
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
        )
    )
    architecture_payload = checkpoint.get("architecture")
    training_payload = checkpoint.get("training")
    if not isinstance(architecture_payload, dict) or not isinstance(
        training_payload, dict
    ):
        raise ValueError("checkpoint is missing architecture or training")
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(**architecture_payload),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        ce_estimator=ce,
        init_seed=int(model_seed),
    )
    model.network.to(device=device, dtype=torch.float32)
    model.load_checkpoint_state(checkpoint)
    checkpoint_training = DiscreteMPTrainingConfig(**training_payload)
    return model, target_paths, checkpoint_training, continuation_report, design


def _summary_markdown(report: dict[str, object]) -> str:
    residual = report["residual_estimator"]["base_policy"]  # type: ignore[index]
    projection = report["projection_floor"]  # type: ignore[assignment]
    jacobian = report["local_jacobian"]  # type: ignore[assignment]
    lines = [
        "# Terminal fixed-point diagnostic",
        "",
        f"- Frozen checkpoint beta: `{report['checkpoint_beta']:.10g}`",
        f"- Diagnosed operator beta: `{report['operator_beta']:.10g}`",
        "",
        "## Repeated conditional-target estimates",
        "",
        "| Signal | policy bias | estimator noise | noise share of residual MSE | residual CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("p", "P"), ("r", "R")):
        row = residual[key]
        lines.append(
            f"| {label} | {row['relative_policy_bias']:.6g} | "
            f"{row['relative_estimator_noise']:.6g} | "
            f"{row['estimator_noise_fraction_of_expected_residual_mse']:.2%} | "
            f"{row['replicate_relative_residual_cv']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen GRU projection",
            "",
            "| Policy | combined dimensionless residual |",
            "|---|---:|",
            f"| Frozen checkpoint | {projection['base_policy']['combined']['dimensionless_policy_bias_norm']:.6g} |",
            f"| Best frozen-target GRU | {projection['best_frozen_target_gru']['combined']['dimensionless_policy_bias_norm']:.6g} |",
            "",
            "## Reduced local Jacobian",
            "",
            f"- Spectral radius of mean `J_T`: `{jacobian['mean_matrix_spectrum']['spectral_radius_T']:.6g}`",
            f"- Replicate range: `[{jacobian['replicate_spectral_radius_min']:.6g}, {jacobian['replicate_spectral_radius_max']:.6g}]`",
            f"- Jacobian replication-noise ratio: `{jacobian['jacobian_replication_noise_frobenius_ratio']:.6g}`",
            f"- Step-halving relative change: `{jacobian['finite_difference_step_relative_change']:.6g}`",
            "",
            "The Jacobian is restricted to the declared residual-informed P/R time-block subspace; it is not the full neural-operator spectrum.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    continuation = args.continuation_run_dir.resolve()
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
        raise ValueError("terminal continuation stage must be rejected")
    checkpoint_beta = float(continuation_report["accepted_beta"])
    operator_beta = float(terminal["beta_target"])
    operator_training = replace(
        checkpoint_training,
        lambda_scale=float(terminal["lambda_scale"]),
        grad_clip_norm=None,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
    )
    config = TerminalFixedPointDiagnosticConfig(
        residual_replicates=int(args.residual_replicates),
        common_evaluation_paths=int(args.common_evaluation_paths),
        residual_fit_paths=int(args.residual_fit_paths),
        residual_target_batch_size=int(args.residual_target_batch_size),
        projection_fit_paths=int(args.projection_fit_paths),
        projection_validation_paths=int(args.projection_validation_paths),
        projection_target_batch_size=int(args.projection_target_batch_size),
        projection_restarts=int(args.projection_restarts),
        projection_inner_batch_size=int(args.projection_inner_batch_size),
        projection_max_steps=int(args.projection_max_steps),
        projection_eval_every=int(args.projection_eval_every),
        projection_patience=int(args.projection_patience),
        time_blocks=int(args.time_blocks),
        jacobian_replicates=int(args.jacobian_replicates),
        jacobian_fit_paths=int(args.jacobian_fit_paths),
        jacobian_validation_paths=int(args.jacobian_validation_paths),
        jacobian_target_batch_size=int(args.jacobian_target_batch_size),
        finite_difference_step=float(args.finite_difference_step),
        secondary_finite_difference_step=float(
            args.secondary_finite_difference_step
        ),
        seed=int(args.seed),
    )

    def progress(message: str) -> None:
        print(message, flush=True)

    result = audit_terminal_fixed_point(
        model=model,
        target_paths=target_paths,
        checkpoint_training=checkpoint_training,
        operator_training=operator_training,
        checkpoint_beta=checkpoint_beta,
        operator_beta=operator_beta,
        config=config,
        progress_callback=progress,
    )
    result.report["continuation_run_dir"] = str(continuation)
    result.report["dataset"] = design["dataset"]
    write_json(run_dir / "diagnostic.json", result.report)
    torch.save(result.artifacts, run_dir / "artifacts.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(result.report), encoding="utf-8"
    )
    print(f"wrote terminal fixed-point diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
