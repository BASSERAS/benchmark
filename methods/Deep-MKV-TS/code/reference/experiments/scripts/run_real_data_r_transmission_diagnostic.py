#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import DiscreteMPTrainingConfig  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import sample_standard_normals  # noqa: E402
from path_dt_experiments.hamiltonian_regime import (  # noqa: E402
    REGIME_NAMES,
    audit_specific_entropy_hamiltonian_by_regime,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
)
from run_real_data_tuning_candidate import (  # noqa: E402
    _component_args,
    _configure_r_ce_estimator,
)


PRICE_BLOCKS = frozenset({"path_law_observed_path", "path_law_returns"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen real-data audit of the volatility discrepancy -> R -> sigma "
            "transmission chain at a completed family-normalized checkpoint."
        )
    )
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-paths", type=int, default=384)
    parser.add_argument("--projection-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=920_031)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument("--future-horizon", type=int, default=32)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().double().pow(2))).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float(torch.dot(left_centered, right_centered).div(denominator).item())


def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().double().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.01, 0.05, 0.50, 0.95, 0.99), dtype=flat.dtype),
    )
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(flat.pow(2))).item()),
        "q01": float(quantiles[0].item()),
        "q05": float(quantiles[1].item()),
        "q50": float(quantiles[2].item()),
        "q95": float(quantiles[3].item()),
        "q99": float(quantiles[4].item()),
    }


def _restore_normalized_discrepancy(
    discrepancy: CompositePathFunctionalDiscrepancy,
    normalization: Mapping[str, Any],
) -> CompositePathFunctionalDiscrepancy:
    weights: dict[str, float] = {}
    families = normalization.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("manifest is missing family discrepancy normalization")
    for family in families.values():
        if not isinstance(family, Mapping) or not isinstance(family.get("blocks"), Mapping):
            raise ValueError("invalid family discrepancy normalization")
        for name, block in family["blocks"].items():
            if not isinstance(block, Mapping):
                raise ValueError(f"invalid normalization for block {name}")
            weights[str(name)] = float(block["final_family_scaled_weight"])
    original = {str(block.name): block for block in discrepancy.blocks}
    if set(original) != set(weights):
        raise ValueError("manifest normalization blocks do not match reconstructed discrepancy")
    return CompositePathFunctionalDiscrepancy(
        blocks=tuple(
            WeightedDiscrepancy(
                name=name,
                weight=weights[name],
                discrepancy=original[name].discrepancy,
            )
            for name in original
        )
    )


def _isolated_discrepancies(
    discrepancy: CompositePathFunctionalDiscrepancy,
) -> dict[str, CompositePathFunctionalDiscrepancy]:
    blocks = tuple(discrepancy.blocks)
    price = tuple(block for block in blocks if str(block.name) in PRICE_BLOCKS)
    volatility = tuple(block for block in blocks if str(block.name) not in PRICE_BLOCKS)
    if not price or not volatility:
        raise ValueError("expected nonempty price and volatility discrepancy families")
    result = {
        "price_law": CompositePathFunctionalDiscrepancy(blocks=price),
        "volatility_law": CompositePathFunctionalDiscrepancy(blocks=volatility),
    }
    result.update(
        {
            str(block.name): CompositePathFunctionalDiscrepancy(blocks=(block,))
            for block in volatility
        }
    )
    return result


def _denominator_summary(
    *,
    artifacts: Mapping[str, torch.Tensor],
    eta: float,
    dt: float,
) -> dict[str, object]:
    sigma_ref = artifacts["sigma_reference"].double()
    resistance = float(eta) / sigma_ref
    sqrt_dt = math.sqrt(float(dt))
    regimes = artifacts["regime_indices"].long()
    result: dict[str, object] = {}
    for name, r_values in (
        ("projected_target", artifacts["target_r"].double()),
        ("fitted_network", artifacts["fitted_r"].double()),
    ):
        forcing = r_values / sqrt_dt
        denominator = resistance + forcing
        ratio = forcing.abs() / resistance.clamp_min(torch.finfo(torch.float64).eps)

        def summarize(mask: torch.Tensor | None) -> dict[str, object]:
            used_resistance = resistance if mask is None else resistance[mask]
            used_forcing = forcing if mask is None else forcing[mask]
            used_denominator = denominator if mask is None else denominator[mask]
            used_ratio = ratio if mask is None else ratio[mask]
            return {
                "eta_over_sigma_reference": _quantile_summary(used_resistance),
                "r_over_sqrt_dt": _quantile_summary(used_forcing),
                "absolute_forcing_to_resistance": _quantile_summary(used_ratio),
                "denominator": _quantile_summary(used_denominator),
                "denominator_nonpositive_fraction": float(
                    (used_denominator <= 0.0).double().mean().item()
                ),
                "r_requests_higher_sigma_fraction": float(
                    (used_forcing < 0.0).double().mean().item()
                ),
            }

        result[name] = {
            "all": summarize(None),
            "by_prefix_volatility_regime": {
                regime_name: summarize(regimes == regime_index)
                for regime_index, regime_name in enumerate(REGIME_NAMES)
            },
        }
    return result


def _block_cancellation_summary(
    artifacts: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    names = sorted(
        name.removesuffix("_r")
        for name in artifacts
        if name.startswith("volatility_law_")
        and name != "volatility_law_r"
        and name.endswith("_r")
    )
    components = {name: artifacts[f"{name}_r"].double() for name in names}
    if not components:
        raise ValueError("no isolated volatility-block R components were saved")
    component_sum = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    volatility_family = artifacts["volatility_law_r"].double()
    full = artifacts["target_r"].double()
    sum_component_rms = sum(_rms(value) for value in components.values())
    quadrature_rms = math.sqrt(sum(_rms(value) ** 2 for value in components.values()))
    return {
        "blocks": {
            name: {
                "rms": _rms(value),
                "correlation_with_volatility_family_r": _correlation(
                    value, volatility_family
                ),
                "correlation_with_complete_r": _correlation(value, full),
            }
            for name, value in components.items()
        },
        "volatility_family_rms": _rms(volatility_family),
        "sum_of_block_rms": sum_component_rms,
        "quadrature_block_rms": quadrature_rms,
        "rms_of_block_sum": _rms(component_sum),
        "block_sum_to_sum_of_rms_ratio": _rms(component_sum)
        / max(sum_component_rms, torch.finfo(torch.float64).eps),
        "block_sum_to_quadrature_rms_ratio": _rms(component_sum)
        / max(quadrature_rms, torch.finfo(torch.float64).eps),
        "reconstruction_relative_error": _rms(component_sum - volatility_family)
        / max(_rms(volatility_family), torch.finfo(torch.float64).eps),
        "volatility_family_to_complete_r_correlation": _correlation(
            volatility_family, full
        ),
    }


def _compact_regime_rows(summary: Mapping[str, Any]) -> list[dict[str, object]]:
    rows = []
    for regime_name in REGIME_NAMES:
        regime = summary["regimes"][regime_name]
        window = regime["future_control_window"]
        controls = window["controls"]
        hamiltonian = window["hamiltonian_balance"]
        r_signal = window["r"]
        rows.append(
            {
                "regime": regime_name,
                "count": int(regime["count"]),
                "future_rv_std_ratio": float(
                    regime["future_law"]["generated_to_target_rv_std_ratio"]
                ),
                "future_rv_iqr_ratio": float(
                    regime["future_law"]["generated_to_target_rv_iqr_ratio"]
                ),
                "r_fitted_to_target_rms_ratio": float(
                    r_signal["fitted_to_target_rms_ratio"]
                ),
                "r_relative_rmse": float(r_signal["relative_rmse"]),
                "r_fitted_target_correlation": r_signal["fitted_target_correlation"],
                "running_to_discrepancy_r_ratio": float(
                    r_signal["running_to_discrepancy_rms_ratio"]
                ),
                "running_discrepancy_r_correlation": r_signal[
                    "running_discrepancy_correlation"
                ],
                "sigma_signal_transmission_fraction": controls[
                    "sigma_signal_transmission_fraction"
                ],
                "sigma_current_oracle_correction_correlation": controls[
                    "sigma_current_oracle_correction_correlation"
                ],
                "sigma_to_reference_ratio_mean": float(
                    controls["sigma_to_reference_ratio"]["mean"]
                ),
                "oracle_sigma_to_reference_ratio_mean": float(
                    controls["oracle_sigma_to_reference_ratio"]["mean"]
                ),
                "target_r_requests_higher_sigma_fraction": float(
                    controls["target_r_requests_higher_sigma_fraction"]
                ),
                "sigma_cap_fraction": float(controls["sigma_cap_fraction"]),
                "oracle_sigma_cap_fraction": float(
                    controls["oracle_sigma_cap_fraction"]
                ),
                "target_sigma_stationarity_rms": float(
                    hamiltonian["target_sigma_stationarity_rms"]
                ),
                "fitted_sigma_stationarity_rms": float(
                    hamiltonian["fitted_sigma_stationarity_rms"]
                ),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    if int(args.audit_paths) < 24:
        raise ValueError("audit-paths must be at least 24")
    if int(args.projection_folds) < 2:
        raise ValueError("projection-folds must be at least two for held-out diagnostics")
    if int(args.future_horizon) < 1 or int(args.split_step) + int(args.future_horizon) > NUM_STEPS:
        raise ValueError("future horizon must fit after split step")
    if int(args.prefix_window) < 1 or int(args.prefix_window) > int(args.split_step):
        raise ValueError("prefix window must fit before split step")

    candidate_dir = args.candidate_run_dir.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    manifest = _read_json(candidate_dir / "candidate_manifest.json")
    if manifest.get("objective_protocol") != "real_scale_corrected_v3_family_adjoint_rms":
        raise ValueError("diagnostic requires a v3 family-normalized candidate")
    adapter_dataset = candidate_dir / "adapter_dataset"
    device = torch.device(args.device)
    dtype = torch.float32
    target_paths, transform, preprocessing = _load_training_data(
        REPO_ROOT,
        device=device,
        dtype=dtype,
        state_representation="log_price",
        custom_dataset_root=adapter_dataset,
        model_dt=1.0 / NUM_STEPS,
    )
    train_prices = np.load(adapter_dataset / "train.npy").astype(np.float64)
    reference_sigma_std_floor = (
        0.25
        * float(np.diff(np.log(train_prices), axis=1).std(ddof=0))
        / math.sqrt(1.0 / NUM_STEPS)
    )
    component_namespace = argparse.Namespace(
        objective_protocol=str(manifest["objective_protocol"]),
        eta=float(manifest["eta"]),
        running_cost=str(manifest.get("running_cost", "specific_entropy")),
        joint_weight=float(manifest["joint_weight"]),
        reference_feature_basis=str(
            manifest.get("reference_feature_basis", "compact")
        ),
        reference_causal_weight=manifest.get("reference_causal_weight"),
        discrepancy_acf_lags=tuple(
            int(value)
            for value in manifest.get(
                "discrepancy_acf_lags", (1, 2, 5, 10)
            )
        ),
    )
    component_args = _component_args(
        component_namespace,
        reference_sigma_std_floor=reference_sigma_std_floor,
        admissible_sigma_max=float(manifest.get("admissible_sigma_max", 0.6)),
    )
    grid, architecture, control, _, joint = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    discrepancy = _restore_normalized_discrepancy(
        joint,
        manifest["discrepancy_normalization"],
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        device=device,
        dtype=dtype,
        seed=int(manifest["seed"]),
    )
    estimator_args = argparse.Namespace(
        r_ce_basis=str(manifest.get("r_ce_basis", "shared_flat_prefix")),
        ridge_lambda=float(manifest["training"]["ridge_lambda"]),
    )
    _configure_r_ce_estimator(model, estimator_args)
    training = replace(
        DiscreteMPTrainingConfig(**manifest["training"]),
        batch_size=min(int(args.audit_paths), int(target_paths.shape[0])),
        target_batch_size=min(int(args.audit_paths), int(target_paths.shape[0])),
        num_steps=1,
        ce_crossfit_folds=int(args.projection_folds),
        grad_clip_norm=None,
        log_every=1,
        seed=int(args.seed),
    )
    audit_count = min(int(args.audit_paths), int(target_paths.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    indices = torch.randperm(int(target_paths.shape[0]), generator=generator)[:audit_count]
    target = target_paths.index_select(0, indices.to(device)).to(device=device, dtype=dtype)
    x0 = target[:, 0, :]
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=audit_count,
        noise_dim=int(model.architecture.noise_dim),
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1,
    )
    isolated = _isolated_discrepancies(discrepancy)

    source_metadata = manifest.get("source_checkpoint")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("candidate manifest is missing source checkpoint provenance")
    source_checkpoint = Path(str(source_metadata["path"]))
    checkpoint = Path(str(manifest["checkpoint"]))
    candidate_checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    model.load_checkpoint_state(candidate_checkpoint_state)
    frozen_noise_baseline = (
        None
        if model._noise_target_timewise_baseline is None
        else model._noise_target_timewise_baseline.detach().clone()
    )
    print("auditing source policy with the candidate objective", flush=True)
    model.load_checkpoint_state(
        torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    )
    # The control variate is a frozen, train-only expectation estimated before
    # the candidate fit.  Reuse it for both policies so this remains a paired
    # policy audit rather than an estimator change.
    model._noise_target_timewise_baseline = frozen_noise_baseline
    source = audit_specific_entropy_hamiltonian_by_regime(
        model=model,
        target_paths=target,
        x0=x0,
        noise=noise,
        training=training,
        split_step=int(args.split_step),
        prefix_window=int(args.prefix_window),
        future_horizon=int(args.future_horizon),
        isolated_discrepancies=isolated,
    )

    print("auditing trained candidate on the same frozen problem", flush=True)
    model.load_checkpoint_state(candidate_checkpoint_state)
    candidate = audit_specific_entropy_hamiltonian_by_regime(
        model=model,
        target_paths=target,
        x0=x0,
        noise=noise,
        training=training,
        split_step=int(args.split_step),
        prefix_window=int(args.prefix_window),
        future_horizon=int(args.future_horizon),
        isolated_discrepancies=isolated,
        baseline_rollout=source.rollout,
    )

    dt = float(model.grid.dt)
    eta = float(control.eta)
    payload: dict[str, object] = {
        "protocol": {
            "frozen_checkpoint_no_retraining": True,
            "same_target_paths_initial_states_and_noise_for_source_and_candidate": True,
            "cross_fitted_projection": True,
            "complete_path_dependent_maximum_principle": True,
            "discrepancy_weights_restored_from_candidate_manifest": True,
            "r_estimator_restored_from_candidate_manifest": True,
            "noise_control_variate_frozen_and_shared_across_policies": True,
        },
        "candidate_run_dir": str(candidate_dir),
        "candidate_checkpoint": str(checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "config": {
            **vars(args),
            "candidate_run_dir": str(candidate_dir),
            "run_dir": str(run_dir),
            "audit_paths_used": audit_count,
            "training": asdict(training),
            "preprocessing": preprocessing,
            "eta": eta,
            "dt": dt,
            "lambda_x": float(manifest["lambda_x"]),
            "lambda_v": float(manifest["lambda_v"]),
        },
        "source": source.summary,
        "candidate": candidate.summary,
        "compact_regime_comparison": {
            "source": _compact_regime_rows(source.summary),
            "candidate": _compact_regime_rows(candidate.summary),
        },
        "denominator_balance": {
            "source": _denominator_summary(
                artifacts=source.artifacts, eta=eta, dt=dt
            ),
            "candidate": _denominator_summary(
                artifacts=candidate.artifacts, eta=eta, dt=dt
            ),
        },
        "volatility_block_r_cancellation": {
            "source": _block_cancellation_summary(source.artifacts),
            "candidate": _block_cancellation_summary(candidate.artifacts),
        },
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {"source": source.artifacts, "candidate": candidate.artifacts},
        run_dir / "diagnostic_artifacts.pt",
    )
    for law, rows in payload["compact_regime_comparison"].items():
        for row in rows:
            print(
                "law={law} regime={regime} rv_std_ratio={rv:.3f} "
                "R_fit/target={ratio:.3f} R_corr={corr} sigma_tx={tx}".format(
                    law=law,
                    regime=row["regime"],
                    rv=float(row["future_rv_std_ratio"]),
                    ratio=float(row["r_fitted_to_target_rms_ratio"]),
                    corr=(
                        "n/a"
                        if row["r_fitted_target_correlation"] is None
                        else f"{float(row['r_fitted_target_correlation']):.3f}"
                    ),
                    tx=(
                        "n/a"
                        if row["sigma_signal_transmission_fraction"] is None
                        else f"{float(row['sigma_signal_transmission_fraction']):.3f}"
                    ),
                ),
                flush=True,
            )
    print(f"wrote R transmission diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
