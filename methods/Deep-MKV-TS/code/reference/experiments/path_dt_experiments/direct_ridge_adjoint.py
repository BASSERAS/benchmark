from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.ce import fit_ridge
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import (
    AdjointNetworkControlPolicy,
    PolicyStepResult,
)


def _rms(value: torch.Tensor) -> float:
    return float(torch.mean(value.detach().double().pow(2)).sqrt().item())


def _time_basis(*, num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Early/middle/late continuous hat basis, with rows summing to one."""

    if int(num_steps) < 2:
        raise ValueError("time-smoothed projection requires at least two steps")
    time = torch.linspace(0.0, 1.0, int(num_steps), device=device, dtype=dtype)
    early = (1.0 - 2.0 * time).clamp(min=0.0, max=1.0)
    middle = (1.0 - torch.abs(2.0 * time - 1.0)).clamp(min=0.0, max=1.0)
    late = (2.0 * time - 1.0).clamp(min=0.0, max=1.0)
    return torch.stack((early, middle, late), dim=1)


def _window_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    """Causal mean at decision t using increments strictly before t."""

    batch, steps, width = values.shape
    cumulative = torch.cat(
        (
            torch.zeros(batch, 1, width, device=values.device, dtype=values.dtype),
            torch.cumsum(values, dim=1),
        ),
        dim=1,
    )
    result = torch.zeros_like(values)
    for step in range(1, steps):
        left = max(0, step - int(window))
        result[:, step, :] = (cumulative[:, step, :] - cumulative[:, left, :]) / float(
            step - left
        )
    return result


@dataclass(frozen=True)
class ResidualAwareFeatureState:
    previous_drift: torch.Tensor
    previous_sigma: torch.Tensor
    standardized_history: torch.Tensor
    innovation_square_history: torch.Tensor
    predicted_variance_history: torch.Tensor
    running_max: torch.Tensor


@dataclass(frozen=True)
class ResidualAwareCausalFeatureMap:
    """Observable features for projecting the volatility adjoint.

    The reference is frozen.  Its drift and volatility are nevertheless
    re-evaluated on each prefix, so the features describe innovations relative
    to the law that performs local calibration.
    """

    generic_features: object
    reference_kernel: object
    dt: float
    num_steps: int
    split_index: int
    variance_floor: float
    recent_windows: tuple[int, ...] = (5, 20)

    def __post_init__(self) -> None:
        if not callable(getattr(self.generic_features, "features_all", None)):
            raise TypeError("generic_features must expose features_all")
        if not callable(getattr(self.generic_features, "features_at", None)):
            raise TypeError("generic_features must expose features_at")
        if not callable(getattr(self.reference_kernel, "evaluate_all", None)):
            raise TypeError("reference_kernel must expose evaluate_all")
        if float(self.dt) <= 0.0 or int(self.num_steps) < 2:
            raise ValueError("dt and num_steps must be positive")
        if not 0 < int(self.split_index) < int(self.num_steps):
            raise ValueError("split_index must be inside the decision grid")
        if not math.isfinite(float(self.variance_floor)) or float(self.variance_floor) <= 0.0:
            raise ValueError("variance_floor must be positive and finite")
        windows = tuple(int(value) for value in self.recent_windows)
        if not windows or any(value < 1 for value in windows):
            raise ValueError("recent_windows must contain positive integers")
        object.__setattr__(self, "recent_windows", windows)

    @property
    def feature_names(self) -> tuple[str, ...]:
        generic = tuple(getattr(self.generic_features, "feature_names"))
        return generic + (
            "reference_sigma",
            "last_standardized_innovation",
            *(f"recent_standardized_innovation_mean_{w}" for w in self.recent_windows),
            *(f"recent_standardized_innovation_square_mean_{w}" for w in self.recent_windows),
            "accumulated_standardized_variance_residual",
            "prefix_reference_residual_log_variance",
            "reference_scaled_drawdown",
            "distance_from_prefix_future_split",
        )

    def features_all(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.ndim != 3 or int(paths.shape[1]) != int(self.num_steps) + 1:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[-1]) != 1:
            raise ValueError("residual-aware projection currently requires d=1")
        generic = self.generic_features.features_all(paths)
        drift, sigma = self.reference_kernel.evaluate_all(paths)
        increments = paths[:, 1:, :] - paths[:, :-1, :]
        innovations = increments - float(self.dt) * drift
        predicted_variance = sigma.pow(2) * float(self.dt)
        standardized = innovations / torch.sqrt(
            predicted_variance.clamp_min(float(self.variance_floor))
        )
        known = torch.zeros_like(standardized)
        known[:, 1:, :] = standardized[:, :-1, :]
        columns = [sigma, known]
        columns.extend(_window_mean(standardized, window) for window in self.recent_windows)
        columns.extend(
            _window_mean(standardized.pow(2), window) for window in self.recent_windows
        )

        residual = standardized.pow(2) - 1.0
        residual_cumulative = torch.cat(
            (
                torch.zeros(
                    int(paths.shape[0]), 1, 1, device=paths.device, dtype=paths.dtype
                ),
                torch.cumsum(residual, dim=1),
            ),
            dim=1,
        )
        decisions = torch.arange(
            int(self.num_steps), device=paths.device, dtype=paths.dtype
        ).view(1, -1, 1)
        accumulated = residual_cumulative[:, :-1, :] / torch.sqrt(decisions.clamp_min(1.0))
        accumulated[:, 0, :] = 0.0
        columns.append(accumulated)

        innovation_sq_cumulative = torch.cat(
            (
                torch.zeros_like(residual_cumulative[:, :1, :]),
                torch.cumsum(innovations.pow(2), dim=1),
            ),
            dim=1,
        )
        predicted_cumulative = torch.cat(
            (
                torch.zeros_like(residual_cumulative[:, :1, :]),
                torch.cumsum(predicted_variance, dim=1),
            ),
            dim=1,
        )
        prefix_ratio = torch.zeros_like(standardized)
        window = max(self.recent_windows)
        for step in range(1, int(self.num_steps)):
            left = max(0, step - int(window))
            realized = (
                innovation_sq_cumulative[:, step, :]
                - innovation_sq_cumulative[:, left, :]
            )
            predicted = (
                predicted_cumulative[:, step, :]
                - predicted_cumulative[:, left, :]
            )
            prefix_ratio[:, step, :] = torch.log(
                (realized + float(self.variance_floor))
                / (predicted + float(self.variance_floor))
            )
        columns.append(prefix_ratio)

        decision_paths = paths[:, :-1, :]
        running_max = torch.cummax(decision_paths, dim=1).values
        drawdown = running_max - decision_paths
        scale = torch.sqrt(
            predicted_cumulative[:, :-1, :].clamp_min(float(self.variance_floor))
        )
        columns.append(drawdown / scale)
        distance = (
            decisions - float(self.split_index)
        ) / float(self.num_steps)
        columns.append(distance.expand(int(paths.shape[0]), -1, -1))
        return torch.cat((generic, *columns), dim=2)

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor:
        step = int(prefixes.shape[1]) - 1
        if not 0 <= step < int(self.num_steps):
            raise ValueError("prefix length does not identify a valid decision step")
        padding = prefixes[:, -1:, :].expand(
            -1, int(self.num_steps) + 1 - int(prefixes.shape[1]), -1
        )
        padded = torch.cat((prefixes, padding), dim=1)
        return self.features_all(padded)[:, step, :]

    def step_features(
        self,
        prefixes: torch.Tensor,
        *,
        step_index: int,
        state: ResidualAwareFeatureState | None,
    ) -> tuple[torch.Tensor, ResidualAwareFeatureState]:
        """Return exact current features while updating a causal feature state."""

        step = int(step_index)
        if int(prefixes.shape[1]) != step + 1:
            raise ValueError("prefix length and step_index disagree")
        batch = int(prefixes.shape[0])
        width = int(prefixes.shape[-1])
        if width != 1:
            raise ValueError("residual-aware projection currently requires d=1")
        if step == 0:
            empty = torch.empty(
                batch, 0, width, device=prefixes.device, dtype=prefixes.dtype
            )
            standardized = empty
            innovation_square = empty
            predicted_variance = empty
            running_max = prefixes[:, -1, :]
        else:
            if state is None:
                raise ValueError("noninitial residual feature step requires state")
            innovation = (
                prefixes[:, -1, :]
                - prefixes[:, -2, :]
                - float(self.dt) * state.previous_drift
            )
            variance = state.previous_sigma.pow(2) * float(self.dt)
            current_standardized = innovation / torch.sqrt(
                variance.clamp_min(float(self.variance_floor))
            )
            standardized = torch.cat(
                (state.standardized_history, current_standardized.unsqueeze(1)), dim=1
            )
            innovation_square = torch.cat(
                (state.innovation_square_history, innovation.pow(2).unsqueeze(1)), dim=1
            )
            predicted_variance = torch.cat(
                (state.predicted_variance_history, variance.unsqueeze(1)), dim=1
            )
            running_max = torch.maximum(state.running_max, prefixes[:, -1, :])

        generic = self.generic_features.features_at(prefixes)
        drift, sigma = self.reference_kernel.evaluate(prefixes, step_index=step)
        if step == 0:
            zero = torch.zeros(batch, width, device=prefixes.device, dtype=prefixes.dtype)
            last = zero
            rolling_mean = [zero for _ in self.recent_windows]
            rolling_square = [zero for _ in self.recent_windows]
            accumulated = zero
            prefix_ratio = zero
            variance_sum = zero
        else:
            last = standardized[:, -1, :]
            rolling_mean = [
                standardized[:, -min(step, window) :, :].mean(dim=1)
                for window in self.recent_windows
            ]
            rolling_square = [
                standardized[:, -min(step, window) :, :].pow(2).mean(dim=1)
                for window in self.recent_windows
            ]
            accumulated = (standardized.pow(2) - 1.0).sum(dim=1) / math.sqrt(
                float(step)
            )
            window = min(step, max(self.recent_windows))
            realized = innovation_square[:, -window:, :].sum(dim=1)
            predicted = predicted_variance[:, -window:, :].sum(dim=1)
            prefix_ratio = torch.log(
                (realized + float(self.variance_floor))
                / (predicted + float(self.variance_floor))
            )
            variance_sum = predicted_variance.sum(dim=1)
        drawdown = running_max - prefixes[:, -1, :]
        scaled_drawdown = drawdown / torch.sqrt(
            variance_sum.clamp_min(float(self.variance_floor))
        )
        distance = torch.full_like(
            last, (float(step) - float(self.split_index)) / float(self.num_steps)
        )
        features = torch.cat(
            (
                generic,
                sigma,
                last,
                *rolling_mean,
                *rolling_square,
                accumulated,
                prefix_ratio,
                scaled_drawdown,
                distance,
            ),
            dim=1,
        )
        return features, ResidualAwareFeatureState(
            previous_drift=drift,
            previous_sigma=sigma,
            standardized_history=standardized,
            innovation_square_history=innovation_square,
            predicted_variance_history=predicted_variance,
            running_max=running_max,
        )


@dataclass(frozen=True)
class GenericCausalFeatureMap:
    generic_features: object

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(getattr(self.generic_features, "feature_names"))

    def features_all(self, paths: torch.Tensor) -> torch.Tensor:
        return self.generic_features.features_all(paths)

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor:
        return self.generic_features.features_at(prefixes)


@dataclass(frozen=True)
class TimeSmoothedRidgeProjection:
    feature_map: object
    feature_center: torch.Tensor
    feature_scale: torch.Tensor
    coefficients: torch.Tensor
    ridge: float
    num_steps: int

    @classmethod
    def fit(
        cls,
        *,
        feature_map: object,
        paths: torch.Tensor,
        targets: torch.Tensor,
        ridge: float,
        eps: float = 1e-6,
    ) -> "TimeSmoothedRidgeProjection":
        if paths.ndim != 3 or targets.ndim != 3:
            raise ValueError("paths and targets must have shapes (B,N+1,d) and (B,N,m)")
        if int(paths.shape[0]) != int(targets.shape[0]):
            raise ValueError("paths and targets must share batch size")
        if int(paths.shape[1]) != int(targets.shape[1]) + 1:
            raise ValueError("targets must have one entry per decision time")
        if float(ridge) < 0.0:
            raise ValueError("ridge must be nonnegative")
        raw = feature_map.features_all(paths).detach()
        center = raw.mean(dim=(0, 1), keepdim=True)
        scale = raw.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(float(eps))
        design = cls._design_from_standardized((raw - center) / scale)
        flattened = design.reshape(-1, int(design.shape[-1])).double()
        response = targets.detach().reshape(-1, int(targets.shape[-1])).double()
        # fit_ridge uses a sum-of-squares convention. Scale by observation
        # count so ridge has the meaning of a mean-square penalty.
        coefficients = fit_ridge(
            design=flattened,
            target=response,
            ridge=float(ridge) * float(flattened.shape[0]),
        )
        return cls(
            feature_map=feature_map,
            feature_center=center.detach().cpu(),
            feature_scale=scale.detach().cpu(),
            coefficients=coefficients.detach().cpu(),
            ridge=float(ridge),
            num_steps=int(targets.shape[1]),
        )

    @staticmethod
    def _design_from_standardized(features: torch.Tensor) -> torch.Tensor:
        basis = _time_basis(
            num_steps=int(features.shape[1]), device=features.device, dtype=features.dtype
        ).unsqueeze(0).expand(int(features.shape[0]), -1, -1)
        interactions = features.unsqueeze(-1) * basis.unsqueeze(2)
        return torch.cat((basis, interactions.flatten(start_dim=2)), dim=2)

    def predict_all(self, paths: torch.Tensor) -> torch.Tensor:
        raw = self.feature_map.features_all(paths)
        center = self.feature_center.to(device=raw.device, dtype=raw.dtype)
        scale = self.feature_scale.to(device=raw.device, dtype=raw.dtype)
        design = self._design_from_standardized((raw - center) / scale)
        coefficients = self.coefficients.to(device=raw.device, dtype=raw.dtype)
        return design @ coefficients

    def predict_at(self, prefixes: torch.Tensor, *, step_index: int) -> torch.Tensor:
        raw = self.feature_map.features_at(prefixes)
        return self.predict_features_at(raw, step_index=step_index)

    def predict_features_at(
        self, raw: torch.Tensor, *, step_index: int
    ) -> torch.Tensor:
        center = self.feature_center[:, 0, :].to(device=raw.device, dtype=raw.dtype)
        scale = self.feature_scale[:, 0, :].to(device=raw.device, dtype=raw.dtype)
        normalized = (raw - center) / scale
        basis = _time_basis(
            num_steps=int(self.num_steps), device=raw.device, dtype=raw.dtype
        )[int(step_index)].view(1, -1).expand(int(raw.shape[0]), -1)
        design = torch.cat((basis, (normalized.unsqueeze(-1) * basis.unsqueeze(1)).flatten(1)), dim=1)
        return design @ self.coefficients.to(device=raw.device, dtype=raw.dtype)

    def diagnostics(self, *, paths: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        prediction = self.predict_all(paths)
        residual = prediction - targets
        centered = targets - targets.mean(dim=0, keepdim=True)
        variance = float(torch.mean(centered.detach().double().pow(2)).item())
        return {
            "residual_rms": _rms(residual),
            "relative_residual": _rms(residual) / max(_rms(targets), torch.finfo(torch.float64).eps),
            "explained_energy": 1.0 - float(
                torch.mean(residual.detach().double().pow(2)).item()
            ) / max(variance, torch.finfo(torch.float64).eps),
        }


def select_ridge_projection(
    *,
    feature_map: object,
    fit_paths: torch.Tensor,
    fit_targets: torch.Tensor,
    validation_paths: torch.Tensor,
    validation_targets: torch.Tensor,
    ridge_grid: Sequence[float],
) -> tuple[TimeSmoothedRidgeProjection, dict[str, object]]:
    candidates = []
    for ridge in tuple(float(value) for value in ridge_grid):
        projection = TimeSmoothedRidgeProjection.fit(
            feature_map=feature_map,
            paths=fit_paths,
            targets=fit_targets,
            ridge=ridge,
        )
        validation = projection.diagnostics(paths=validation_paths, targets=validation_targets)
        candidates.append((float(validation["residual_rms"]), ridge, projection, validation))
    _, selected_ridge, selected, validation = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, {
        "selected_ridge": float(selected_ridge),
        "validation": validation,
        "ridge_grid": [
            {
                "ridge": float(ridge),
                "validation": metrics,
            }
            for _, ridge, _, metrics in candidates
        ],
    }


@dataclass(frozen=True)
class DirectRidgeRAdjointPolicy:
    """Use the existing network P head and deploy a causal ridge R directly."""

    model: DiscreteMPModel
    p_network: torch.nn.Module
    r_projection: TimeSmoothedRidgeProjection

    def initial_state(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[object, ResidualAwareFeatureState | None]:
        network_state = AdjointNetworkControlPolicy(
            model=self.model, network=self.p_network
        ).initial_state(batch_size=batch_size, device=device, dtype=dtype)
        return network_state, None

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: tuple[object, ResidualAwareFeatureState | None],
    ) -> PolicyStepResult:
        network_result = AdjointNetworkControlPolicy(
            model=self.model, network=self.p_network
        ).control_step(x_prefix=x_prefix, step_index=step_index, state=state[0])
        p = network_result.expected_adjoint_next
        if p is None:
            raise RuntimeError("P network did not expose its adjoint moment")
        step_features = getattr(self.r_projection.feature_map, "step_features", None)
        if callable(step_features):
            raw_features, feature_state = step_features(
                x_prefix,
                step_index=int(step_index),
                state=state[1],
            )
            r = self.r_projection.predict_features_at(
                raw_features, step_index=int(step_index)
            )
        else:
            feature_state = state[1]
            r = self.r_projection.predict_at(x_prefix, step_index=int(step_index))
        control = self.model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p,
                expected_adjoint_noise_next=r,
                parameters=self.model.parameters,
            )
        )
        if control.ndim == 1:
            control = control.unsqueeze(-1)
        return PolicyStepResult(
            control=control,
            state=(network_result.state, feature_state),
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


@dataclass(frozen=True)
class BlockwiseAdjointInterpolationPolicy:
    """Interpolate base/candidate adjoint outputs in fixed temporal blocks."""

    model: DiscreteMPModel
    base_policy: object
    candidate_policy: object
    theta: torch.Tensor
    blocks: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if tuple(self.theta.shape) != (2 * len(self.blocks),):
            raise ValueError("theta must contain P and R coordinates for every block")
        if not self.blocks or self.blocks[0][0] != 0 or self.blocks[-1][1] != int(
            self.model.grid.num_steps
        ):
            raise ValueError("blocks must cover the decision grid")
        if any(right <= left for left, right in self.blocks):
            raise ValueError("blocks must be nonempty")

    def initial_state(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[object, object]:
        return (
            self.base_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
            self.candidate_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
        )

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: tuple[object, object],
    ) -> PolicyStepResult:
        base = self.base_policy.control_step(
            x_prefix=x_prefix, step_index=int(step_index), state=state[0]
        )
        candidate = self.candidate_policy.control_step(
            x_prefix=x_prefix, step_index=int(step_index), state=state[1]
        )
        if (
            base.expected_adjoint_next is None
            or base.expected_adjoint_noise_next is None
            or candidate.expected_adjoint_next is None
            or candidate.expected_adjoint_noise_next is None
        ):
            raise TypeError("interpolated policies must expose both adjoint moments")
        block = next(
            index
            for index, (left, right) in enumerate(self.blocks)
            if left <= int(step_index) < right
        )
        count = len(self.blocks)
        tau_p = self.theta[block].to(base.expected_adjoint_next)
        tau_r = self.theta[count + block].to(base.expected_adjoint_noise_next)
        p = base.expected_adjoint_next + tau_p * (
            candidate.expected_adjoint_next - base.expected_adjoint_next
        )
        r = base.expected_adjoint_noise_next + tau_r * (
            candidate.expected_adjoint_noise_next
            - base.expected_adjoint_noise_next
        )
        control = self.model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p,
                expected_adjoint_noise_next=r,
                parameters=self.model.parameters,
            )
        )
        if control.ndim == 1:
            control = control.unsqueeze(-1)
        return PolicyStepResult(
            control=control,
            state=(base.state, candidate.state),
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


def replicated_target_metrics(
    *, prediction: torch.Tensor, target_left: torch.Tensor, target_right: torch.Tensor
) -> dict[str, float]:
    if tuple(target_left.shape) != tuple(target_right.shape) or tuple(prediction.shape) != tuple(target_left.shape):
        raise ValueError("prediction and replicated targets must have identical shapes")
    consensus = 0.5 * (target_left + target_right)
    centered = consensus - consensus.mean(dim=0, keepdim=True)
    denominator = float(torch.mean(centered.detach().double().pow(2)).item())
    residual = prediction - consensus
    replicate_difference = target_left - target_right
    left_centered = target_left - target_left.mean(dim=0, keepdim=True)
    right_centered = target_right - target_right.mean(dim=0, keepdim=True)
    cosine_denominator = torch.linalg.vector_norm(left_centered.double()) * torch.linalg.vector_norm(
        right_centered.double()
    )
    return {
        "prediction_rms": _rms(prediction),
        "target_consensus_rms": _rms(consensus),
        "prediction_residual_rms": _rms(residual),
        "prediction_explained_energy": 1.0 - float(
            torch.mean(residual.detach().double().pow(2)).item()
        ) / max(denominator, torch.finfo(torch.float64).eps),
        "replicate_difference_rms": _rms(replicate_difference),
        "replicate_cosine": float(
            torch.dot(left_centered.double().reshape(-1), right_centered.double().reshape(-1))
            / cosine_denominator.clamp_min(torch.finfo(torch.float64).eps)
        ),
    }


__all__ = [
    "BlockwiseAdjointInterpolationPolicy",
    "DirectRidgeRAdjointPolicy",
    "GenericCausalFeatureMap",
    "ResidualAwareCausalFeatureMap",
    "ResidualAwareFeatureState",
    "TimeSmoothedRidgeProjection",
    "replicated_target_metrics",
    "select_ridge_projection",
]
