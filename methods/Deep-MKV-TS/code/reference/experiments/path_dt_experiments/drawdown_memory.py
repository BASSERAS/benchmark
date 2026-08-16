from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.ce import fit_ridge
from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancyResult
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _strictly_increasing_positive_ints(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or any(value < 1 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"{name} must be strictly increasing positive integers")
    return result


def _positive_floats(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not math.isfinite(value) or value <= 0.0 for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct positive finite floats")
    return result


def _validate_paths(paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> None:
    if paths.ndim != 3 or int(paths.shape[-1]) != 1:
        raise ValueError("drawdown-memory discrepancy requires paths of shape (B, N + 1, 1)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")
    if int(paths.shape[0]) < 2:
        raise ValueError("drawdown-memory discrepancy requires at least two paths")


def _memory_observables(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    history_cutoffs: tuple[int, ...],
    future_horizons: tuple[int, ...],
    recent_volatility_windows: tuple[int, ...],
    drawdown_thresholds: tuple[float, ...],
    gate_temperature: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current controls, soft past-event gates, and future log RV."""

    _validate_paths(paths, grid=grid)
    split = int(split_index)
    if split < max(recent_volatility_windows):
        raise ValueError("recent_volatility_windows must fit before split_index")
    if max(history_cutoffs) > split:
        raise ValueError("history_cutoffs must not exceed split_index")
    if split + max(future_horizons) > int(grid.num_steps):
        raise ValueError("future_horizons must fit after split_index")

    values = paths[..., 0]
    returns = values[:, 1:] - values[:, :-1]
    scaled_returns = returns / math.sqrt(float(grid.dt))
    running_max = torch.cummax(values, dim=1).values
    drawdowns = running_max - values
    eps_squared = float(eps) ** 2

    recent_log_volatility = [
        torch.log(
            torch.sqrt(
                scaled_returns[:, split - window : split]
                .pow(2)
                .mean(dim=1)
                + eps_squared
            )
        )
        for window in recent_volatility_windows
    ]
    controls = torch.stack(
        [
            *recent_log_volatility,
            drawdowns[:, split],
            values[:, split],
        ],
        dim=1,
    )

    gates = torch.stack(
        [
            torch.sigmoid(
                (
                    drawdowns[:, : cutoff + 1].amax(dim=1)
                    - float(threshold)
                )
                / float(gate_temperature)
            )
            for cutoff in history_cutoffs
            for threshold in drawdown_thresholds
        ],
        dim=1,
    )
    future_log_volatility = torch.stack(
        [
            torch.log(
                torch.sqrt(
                    scaled_returns[:, split : split + horizon]
                    .pow(2)
                    .mean(dim=1)
                    + eps_squared
                )
            )
            for horizon in future_horizons
        ],
        dim=1,
    )
    return controls, gates, future_log_volatility


def _quadratic_design(standardized_controls: torch.Tensor) -> torch.Tensor:
    if standardized_controls.ndim != 2:
        raise ValueError("standardized_controls must have shape (B, p)")
    width = int(standardized_controls.shape[1])
    interactions = [
        standardized_controls[:, left] * standardized_controls[:, right]
        for left in range(width)
        for right in range(left + 1, width)
    ]
    columns = [
        torch.ones(
            int(standardized_controls.shape[0]),
            1,
            device=standardized_controls.device,
            dtype=standardized_controls.dtype,
        ),
        standardized_controls,
        standardized_controls.pow(2),
    ]
    if interactions:
        columns.append(torch.stack(interactions, dim=1))
    return torch.cat(columns, dim=1)


@dataclass(frozen=True)
class FixedTargetResidualDrawdownMemoryMoments:
    """Cross-fitted conditional drawdown-memory moment discrepancy.

    Nuisance regressions remove what current volatility, current drawdown, and
    current state already explain.  The compared moment is the standardized
    cross moment between the remaining early-drawdown signal and future
    realized volatility.  All nuisance fits and scales are frozen from the
    complete training target, so generated-path gradients act only through the
    generated law.
    """

    control_mean: torch.Tensor
    control_scale: torch.Tensor
    gate_coefficients: torch.Tensor
    future_coefficients: torch.Tensor
    gate_residual_scale: torch.Tensor
    future_residual_scale: torch.Tensor
    target_cross_moments: torch.Tensor
    split_index: int
    history_cutoffs: tuple[int, ...]
    future_horizons: tuple[int, ...]
    recent_volatility_windows: tuple[int, ...]
    drawdown_thresholds: tuple[float, ...]
    gate_temperature: float
    ridge: float
    crossfit_folds: int
    eps: float = 1e-6
    name: str = "residual_drawdown_memory_moments"

    def __post_init__(self) -> None:
        tensor_fields = (
            "control_mean",
            "control_scale",
            "gate_coefficients",
            "future_coefficients",
            "gate_residual_scale",
            "future_residual_scale",
            "target_cross_moments",
        )
        for name in tensor_fields:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite tensor")
            object.__setattr__(self, name, value.detach().cpu().clone())
        if tuple(self.control_mean.shape) != tuple(self.control_scale.shape):
            raise ValueError("control_mean and control_scale shapes must match")
        if self.control_mean.ndim != 2 or int(self.control_mean.shape[0]) != 1:
            raise ValueError("control statistics must have shape (1, p)")
        if not torch.all(self.control_scale > 0.0):
            raise ValueError("control_scale must be positive")
        if self.gate_residual_scale.ndim != 2 or int(
            self.gate_residual_scale.shape[0]
        ) != 1:
            raise ValueError("gate_residual_scale must have shape (1, gates)")
        if self.future_residual_scale.ndim != 2 or int(
            self.future_residual_scale.shape[0]
        ) != 1:
            raise ValueError("future_residual_scale must have shape (1, horizons)")
        if not torch.all(self.gate_residual_scale > 0.0) or not torch.all(
            self.future_residual_scale > 0.0
        ):
            raise ValueError("residual scales must be positive")
        expected_moment_shape = (
            int(self.gate_residual_scale.shape[1]),
            int(self.future_residual_scale.shape[1]),
        )
        if tuple(self.target_cross_moments.shape) != expected_moment_shape:
            raise ValueError("target_cross_moments shape does not match residual scales")

    def _residuals(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        controls, gates, future = _memory_observables(
            paths,
            grid=grid,
            split_index=int(self.split_index),
            history_cutoffs=self.history_cutoffs,
            future_horizons=self.future_horizons,
            recent_volatility_windows=self.recent_volatility_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            gate_temperature=float(self.gate_temperature),
            eps=float(self.eps),
        )
        mean = self.control_mean.to(device=paths.device, dtype=paths.dtype)
        scale = self.control_scale.to(device=paths.device, dtype=paths.dtype)
        design = _quadratic_design((controls - mean) / scale)
        gate_prediction = design @ self.gate_coefficients.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        future_prediction = design @ self.future_coefficients.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        return gates - gate_prediction, future - future_prediction

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        _validate_paths(generated_paths, grid=grid)
        _validate_paths(target_paths, grid=grid)
        generated_gate_residual, generated_future_residual = self._residuals(
            generated_paths,
            grid=grid,
        )
        centered_gate = generated_gate_residual - generated_gate_residual.mean(
            dim=0,
            keepdim=True,
        )
        centered_future = (
            generated_future_residual
            - generated_future_residual.mean(dim=0, keepdim=True)
        )
        cross_moments = torch.mean(
            centered_gate.unsqueeze(2) * centered_future.unsqueeze(1),
            dim=0,
        )
        gate_scale = self.gate_residual_scale.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        future_scale = self.future_residual_scale.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        standardized = cross_moments / (
            gate_scale.transpose(0, 1) * future_scale
        )
        target = self.target_cross_moments.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        difference = standardized - target
        value = torch.mean(difference.pow(2))
        metrics: dict[str, float] = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_rmse": float(
                torch.sqrt(torch.mean(difference.detach().pow(2))).item()
            ),
            f"{self.name}_moment_count": float(difference.numel()),
            f"{self.name}_split_index": float(self.split_index),
            f"{self.name}_crossfit_folds": float(self.crossfit_folds),
            f"{self.name}_target_abs_mean": float(
                target.detach().abs().mean().item()
            ),
            f"{self.name}_generated_abs_mean": float(
                standardized.detach().abs().mean().item()
            ),
        }
        for gate_index in range(int(standardized.shape[0])):
            for horizon_index in range(int(standardized.shape[1])):
                suffix = f"g{gate_index}_h{self.future_horizons[horizon_index]}"
                metrics[f"{self.name}_generated_{suffix}"] = float(
                    standardized[gate_index, horizon_index].detach().item()
                )
                metrics[f"{self.name}_target_{suffix}"] = float(
                    target[gate_index, horizon_index].detach().item()
                )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "split_index": int(self.split_index),
            "history_cutoffs": list(self.history_cutoffs),
            "future_horizons": list(self.future_horizons),
            "recent_volatility_windows": list(self.recent_volatility_windows),
            "drawdown_thresholds": list(self.drawdown_thresholds),
            "gate_temperature": float(self.gate_temperature),
            "ridge": float(self.ridge),
            "crossfit_folds": int(self.crossfit_folds),
            "moment_count": int(self.target_cross_moments.numel()),
            "target_cross_moments": self.target_cross_moments.tolist(),
        }


def fit_fixed_target_residual_drawdown_memory_moments(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int = 64,
    history_cutoffs: Sequence[int] = (32, 40, 48),
    future_horizons: Sequence[int] = (16, 32, 48),
    recent_volatility_windows: Sequence[int] = (10, 20, 32),
    drawdown_thresholds: Sequence[float],
    gate_temperature: float = 0.005,
    ridge: float = 1e-3,
    crossfit_folds: int = 5,
    crossfit_seed: int = 1907,
    eps: float = 1e-6,
) -> FixedTargetResidualDrawdownMemoryMoments:
    """Fit immutable nuisance regressions and target moments on training data."""

    _validate_paths(target_paths, grid=grid)
    cutoffs = _strictly_increasing_positive_ints(
        "history_cutoffs",
        history_cutoffs,
    )
    horizons = _strictly_increasing_positive_ints(
        "future_horizons",
        future_horizons,
    )
    windows = _strictly_increasing_positive_ints(
        "recent_volatility_windows",
        recent_volatility_windows,
    )
    thresholds = _positive_floats("drawdown_thresholds", drawdown_thresholds)
    temperature = float(gate_temperature)
    ridge_value = float(ridge)
    eps_value = float(eps)
    folds = int(crossfit_folds)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("gate_temperature must be a positive finite float")
    if not math.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be a nonnegative finite float")
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps must be a positive finite float")
    if folds < 2 or folds > int(target_paths.shape[0]):
        raise ValueError("crossfit_folds must lie between 2 and the target path count")

    working = target_paths.detach().to(dtype=torch.float64)
    controls, gates, future = _memory_observables(
        working,
        grid=grid,
        split_index=int(split_index),
        history_cutoffs=cutoffs,
        future_horizons=horizons,
        recent_volatility_windows=windows,
        drawdown_thresholds=thresholds,
        gate_temperature=temperature,
        eps=eps_value,
    )
    control_mean = controls.mean(dim=0, keepdim=True)
    control_scale = controls.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(eps_value)
    design = _quadratic_design((controls - control_mean) / control_scale)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(crossfit_seed))
    path_count = int(target_paths.shape[0])
    permutation = torch.randperm(path_count, generator=generator)
    fold_ids_cpu = torch.empty(path_count, dtype=torch.long)
    fold_ids_cpu[permutation] = torch.arange(path_count) % folds
    fold_ids = fold_ids_cpu.to(device=working.device)
    gate_oof = torch.empty_like(gates)
    future_oof = torch.empty_like(future)
    for fold in range(folds):
        train = fold_ids != fold
        valid = fold_ids == fold
        gate_coefficients = fit_ridge(
            design=design[train],
            target=gates[train],
            ridge=ridge_value,
        )
        future_coefficients = fit_ridge(
            design=design[train],
            target=future[train],
            ridge=ridge_value,
        )
        gate_oof[valid] = gates[valid] - design[valid] @ gate_coefficients
        future_oof[valid] = future[valid] - design[valid] @ future_coefficients

    centered_gate = gate_oof - gate_oof.mean(dim=0, keepdim=True)
    centered_future = future_oof - future_oof.mean(dim=0, keepdim=True)
    gate_scale = centered_gate.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(eps_value)
    future_scale = centered_future.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    ).clamp_min(eps_value)
    target_cross_moments = torch.mean(
        centered_gate.unsqueeze(2) * centered_future.unsqueeze(1),
        dim=0,
    ) / (gate_scale.transpose(0, 1) * future_scale)

    final_gate_coefficients = fit_ridge(
        design=design,
        target=gates,
        ridge=ridge_value,
    )
    final_future_coefficients = fit_ridge(
        design=design,
        target=future,
        ridge=ridge_value,
    )
    return FixedTargetResidualDrawdownMemoryMoments(
        control_mean=control_mean,
        control_scale=control_scale,
        gate_coefficients=final_gate_coefficients,
        future_coefficients=final_future_coefficients,
        gate_residual_scale=gate_scale,
        future_residual_scale=future_scale,
        target_cross_moments=target_cross_moments,
        split_index=int(split_index),
        history_cutoffs=cutoffs,
        future_horizons=horizons,
        recent_volatility_windows=windows,
        drawdown_thresholds=thresholds,
        gate_temperature=temperature,
        ridge=ridge_value,
        crossfit_folds=folds,
        eps=eps_value,
    )


__all__ = [
    "FixedTargetResidualDrawdownMemoryMoments",
    "fit_fixed_target_residual_drawdown_memory_moments",
]
