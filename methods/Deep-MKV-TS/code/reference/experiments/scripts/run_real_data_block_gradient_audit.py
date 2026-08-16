#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import DiscreteMPTrainingConfig  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import (  # noqa: E402
    derive_stream_seed,
    sample_standard_normals,
)
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
)
from run_real_data_tuning_candidate import _component_args  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit each real-data discrepancy block at a frozen source checkpoint, "
            "including empirical Lions sources, raw/projected P/R targets, and "
            "the induced neural target-forcing gradient."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.detach().double().pow(2).mean()).item())


def _maximum(value: torch.Tensor) -> float:
    return float(value.detach().double().abs().max().item())


def _gradient_norm(
    scalar: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> float:
    gradients = torch.autograd.grad(
        scalar,
        parameters,
        allow_unused=True,
    )
    squared_norm = sum(
        float(gradient.detach().double().pow(2).sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(squared_norm)


def _forcing_gradient_norm(
    *,
    model,
    generated_paths: torch.Tensor,
    projected_p: torch.Tensor,
    projected_r: torch.Tensor,
    training: DiscreteMPTrainingConfig,
) -> float:
    moments = model.network(generated_paths.detach())
    target_p = model._normalize_adjoint_target(
        projected_p.detach(),
        noise_adjoint=False,
    )
    target_r = model._normalize_adjoint_target(
        projected_r.detach(),
        noise_adjoint=True,
    )
    # Only the target-dependent linear term is block-additive.  The prediction
    # squared term is common to the complete regression loss and is therefore
    # intentionally excluded from per-block attribution.
    forcing = -2.0 * (
        float(training.adjoint_weight)
        * torch.mean(moments.expected_adjoint_next * target_p)
        + float(training.adjoint_noise_weight)
        * torch.mean(moments.expected_adjoint_noise_next * target_r)
    )
    parameters = tuple(parameter for parameter in model.network.parameters() if parameter.requires_grad)
    return _gradient_norm(forcing, parameters)


def _project_targets(
    *,
    model,
    generated_paths: torch.Tensor,
    noise: torch.Tensor,
    raw_p: torch.Tensor,
    training: DiscreteMPTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_r = model._adjoint_noise_target(
        pathwise_adjoint_target=raw_p,
        moments_noise_target_dim=int(
            model._default_noise_adjoint_dim(model.architecture)
        ),
        noise=noise,
    )
    with torch.no_grad():
        projected_p, _ = model._project_pathwise_target(
            frozen_generated_path=generated_paths.detach(),
            pathwise_target=raw_p.detach(),
            training=training,
        )
        projected_r, _ = model._project_pathwise_target(
            frozen_generated_path=generated_paths.detach(),
            pathwise_target=raw_r.detach(),
            training=training,
            noise_adjoint=True,
        )
    return raw_r.detach(), projected_p.detach(), projected_r.detach()


def _one_batch(
    *,
    model,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    seed: int,
) -> dict[str, dict[str, float]]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source_indices = model._sample_indices(
        size=int(target_paths.shape[0]),
        count=int(training.batch_size),
        generator=cpu_generator,
    )
    target_indices = model._sample_indices(
        size=int(target_paths.shape[0]),
        count=int(training.target_batch_size),
        generator=cpu_generator,
    )
    x0 = target_paths.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = target_paths.index_select(0, target_indices.to(model.device))
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(training.batch_size),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(seed), stream_offset=91),
    )
    with torch.no_grad():
        rollout = model.rollout(x0=x0, noise=noise)
    if not isinstance(model.discrepancy, CompositePathFunctionalDiscrepancy):
        raise ValueError("the audit requires a composite discrepancy")

    payload: dict[str, dict[str, float]] = {}
    block_p: list[torch.Tensor] = []
    for block in tuple(model.discrepancy.blocks):
        single = CompositePathFunctionalDiscrepancy(blocks=(block,))
        generated_req = rollout.paths.detach().clone().requires_grad_(True)
        result = single(
            generated_paths=generated_req,
            target_paths=target_batch,
            grid=model.grid,
        )
        empirical_gradient = torch.autograd.grad(
            float(training.lambda_scale) * result.value,
            generated_req,
        )[0].detach()
        lions_source = float(generated_req.shape[0]) * empirical_gradient
        raw_p, _ = model._path_functional_adjoint_targets(
            generated_path=rollout.paths,
            target_path=target_batch,
            training=training,
            discrepancy=single,
            include_running_cost=False,
        )
        raw_r, projected_p, projected_r = _project_targets(
            model=model,
            generated_paths=rollout.paths,
            noise=noise,
            raw_p=raw_p,
            training=training,
        )
        block_p.append(raw_p.detach())
        payload[str(block.name)] = {
            "weight": float(block.weight),
            "discrepancy_value": float(result.value.detach().item()),
            "lions_source_rms": _rms(lions_source),
            "lions_source_max_abs": _maximum(lions_source),
            "raw_p_rms": _rms(raw_p),
            "raw_p_max_abs": _maximum(raw_p),
            "raw_r_rms": _rms(raw_r),
            "raw_r_max_abs": _maximum(raw_r),
            "projected_p_rms": _rms(projected_p),
            "projected_r_rms": _rms(projected_r),
            "target_forcing_parameter_gradient_norm": _forcing_gradient_norm(
                model=model,
                generated_paths=rollout.paths,
                projected_p=projected_p,
                projected_r=projected_r,
                training=training,
            ),
        }

    full_p, _ = model._path_functional_adjoint_targets(
        generated_path=rollout.paths,
        controls=rollout.controls,
        target_path=target_batch,
        training=training,
        include_running_cost=True,
    )
    discrepancy_p = torch.stack(block_p, dim=0).sum(dim=0)
    running_p = full_p.detach() - discrepancy_p
    for name, raw_p in (
        ("__running_cost__", running_p),
        ("__complete_objective__", full_p.detach()),
    ):
        raw_r, projected_p, projected_r = _project_targets(
            model=model,
            generated_paths=rollout.paths,
            noise=noise,
            raw_p=raw_p,
            training=training,
        )
        payload[name] = {
            "weight": 1.0,
            "discrepancy_value": math.nan,
            "lions_source_rms": math.nan,
            "lions_source_max_abs": math.nan,
            "raw_p_rms": _rms(raw_p),
            "raw_p_max_abs": _maximum(raw_p),
            "raw_r_rms": _rms(raw_r),
            "raw_r_max_abs": _maximum(raw_r),
            "projected_p_rms": _rms(projected_p),
            "projected_r_rms": _rms(projected_r),
            "target_forcing_parameter_gradient_norm": _forcing_gradient_norm(
                model=model,
                generated_paths=rollout.paths,
                projected_p=projected_p,
                projected_r=projected_r,
                training=training,
            ),
        }
    return payload


def _aggregate(
    records: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for block_name, metrics in record.items():
            for metric_name, value in metrics.items():
                values[block_name][metric_name].append(float(value))
    result: dict[str, dict[str, float]] = {}
    for block_name, metrics in values.items():
        result[block_name] = {}
        for metric_name, items in metrics.items():
            array = np.asarray(items, dtype=np.float64)
            finite = array[np.isfinite(array)]
            result[block_name][f"{metric_name}_mean"] = (
                math.nan if finite.size == 0 else float(finite.mean())
            )
            result[block_name][f"{metric_name}_max"] = (
                math.nan if finite.size == 0 else float(finite.max())
            )
    return result


def main() -> None:
    args = _parse_args()
    if int(args.batches) < 1 or int(args.batch_size) < 2:
        raise ValueError("batches must be positive and batch-size must be at least two")
    device = torch.device(args.device)
    target_paths, transform, preprocessing = _load_training_data(
        REPO_ROOT,
        device=device,
        dtype=torch.float32,
        state_representation="log_price",
        custom_dataset_root=args.adapter_dataset,
        model_dt=1.0 / NUM_STEPS,
    )
    train_prices = np.load(args.adapter_dataset / "train.npy").astype(np.float64)
    # The component helper only needs this floor for train-only reference gating.
    reference_sigma_std_floor = (
        0.25
        * float(np.diff(np.log(train_prices), axis=1).std())
        / math.sqrt(1.0 / NUM_STEPS)
    )
    component_namespace = argparse.Namespace(
        objective_protocol="real_scale_corrected_v1",
        eta=1.0,
        joint_weight=0.25,
        discrepancy_acf_lags=(1, 2, 5, 10),
    )
    component_args = _component_args(
        component_namespace,
        reference_sigma_std_floor=reference_sigma_std_floor,
    )
    grid, architecture, control, _, discrepancy = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_checkpoint_state(checkpoint)
    training = DiscreteMPTrainingConfig(
        batch_size=int(args.batch_size),
        target_batch_size=int(args.batch_size),
        num_steps=1,
        lr=2.5e-4,
        weight_decay=1e-5,
        lambda_scale=5.0,
        ce_target_mode="ridge",
        ridge_lambda=1e-2,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=None,
        log_every=1,
        seed=int(args.seed),
        observed_only=False,
    )
    records = []
    for batch_index in range(int(args.batches)):
        print(f"audit batch {batch_index + 1}/{args.batches}", flush=True)
        records.append(
            _one_batch(
                model=model,
                target_paths=target_paths,
                training=training,
                seed=int(args.seed) + 10_000 * batch_index,
            )
        )
    output: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "adapter_dataset": str(args.adapter_dataset.resolve()),
        "seed": int(args.seed),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "preprocessing": preprocessing,
        "blocks": _aggregate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
