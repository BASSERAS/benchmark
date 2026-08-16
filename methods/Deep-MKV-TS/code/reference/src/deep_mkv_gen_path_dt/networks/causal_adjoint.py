from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn

from deep_mkv_gen_path_dt.causal_reference import (
    CausalPathFeatureRidgeConditionalExpectation,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.networks.adjoint import AdjointMoments


class CausalFeatureMap(Protocol):
    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def features_all(self, paths: torch.Tensor) -> torch.Tensor: ...

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor: ...


class IncrementCausalFeatureMap:
    """Small generic adapted basis with an inexpensive online evaluation."""

    feature_names = (
        "normalized_time",
        "current_log_price",
        "last_return",
        "absolute_last_return",
        "last_realized_variance",
    )

    def __init__(self, *, num_steps: int, dt: float) -> None:
        if int(num_steps) < 1 or float(dt) <= 0.0:
            raise ValueError("num_steps and dt must be positive")
        self.num_steps = int(num_steps)
        self.dt = float(dt)

    def features_all(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.ndim != 3 or tuple(paths.shape[1:]) != (
            self.num_steps + 1,
            1,
        ):
            raise ValueError("paths must have shape (B, N + 1, 1)")
        batch = int(paths.shape[0])
        values = paths[..., 0]
        points = values[:, :-1]
        returns = values[:, 1:] - values[:, :-1]
        last_return = torch.cat(
            [values.new_zeros((batch, 1)), returns[:, :-1]], dim=1
        )
        time = torch.arange(
            self.num_steps, device=values.device, dtype=values.dtype
        ).unsqueeze(0).expand(batch, -1)
        time = time / float(max(1, self.num_steps - 1))
        return torch.stack(
            (
                time,
                points,
                last_return,
                last_return.abs(),
                last_return.pow(2) / self.dt,
            ),
            dim=-1,
        )

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor:
        if prefixes.ndim != 3 or int(prefixes.shape[2]) != 1:
            raise ValueError("prefixes must have shape (B, n + 1, 1)")
        step = int(prefixes.shape[1]) - 1
        if not 0 <= step < self.num_steps:
            raise ValueError("prefix length is outside the feature grid")
        current = prefixes[:, -1, 0]
        last_return = (
            torch.zeros_like(current)
            if step == 0
            else current - prefixes[:, -2, 0]
        )
        time = torch.full_like(
            current, float(step) / float(max(1, self.num_steps - 1))
        )
        return torch.stack(
            (
                time,
                current,
                last_return,
                last_return.abs(),
                last_return.pow(2) / self.dt,
            ),
            dim=-1,
        )


class CompactVolatilityCausalFeatureMap:
    """Generic volatility basis with an exact linear-time streaming update.

    The features are the same observable ``causal_compact`` features used by
    the R conditional-expectation projection.  They contain no future values
    or latent state.  Streaming sufficient statistics avoid recomputing every
    prefix during a rollout.
    """

    rolling_windows = (5, 10, 20, 32)
    ewma_half_lives = (5, 20, 60)
    downside_windows = (10, 32)

    def __init__(self, *, grid: DiscreteTimeGrid) -> None:
        self.grid = grid
        self.num_steps = int(grid.num_steps)
        self.dt = float(grid.dt)
        self._batch_feature_map = CausalPathFeatureRidgeConditionalExpectation(
            grid=grid,
            ridge=0.0,
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
        )
        self.feature_names = tuple(self._batch_feature_map.feature_names)
        self._decays = tuple(
            math.exp(-math.log(2.0) / float(half_life))
            for half_life in self.ewma_half_lives
        )

    def features_all(self, paths: torch.Tensor) -> torch.Tensor:
        return self._batch_feature_map.features_all(paths)

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor:
        # Direct diagnostics retain the generic contract. Rollouts use the
        # streaming method below and never recompute the complete prefix.
        return self._batch_feature_map.features_at(prefixes)

    def features_step(
        self,
        x_t: torch.Tensor,
        *,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None,
        step_index: int,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != 1:
            raise ValueError("x_t must have shape (B, 1)")
        step = int(step_index)
        if not 0 <= step < self.num_steps:
            raise ValueError("step_index is outside the feature grid")
        current = x_t[:, 0]
        batch = int(current.shape[0])
        zeros = torch.zeros_like(current)
        if state is None:
            if step != 0:
                raise ValueError("streaming feature state is required after step zero")
            previous = current
            running_max = current
            maximum_drawdown = zeros
            history = current.new_empty((batch, 0))
            ewma_numerator = current.new_zeros(
                (batch, len(self.ewma_half_lives))
            )
            ewma_denominator = current.new_zeros(
                (batch, len(self.ewma_half_lives))
            )
            last_return = zeros
        else:
            (
                previous,
                running_max,
                maximum_drawdown,
                history,
                ewma_numerator,
                ewma_denominator,
            ) = state
            if any(int(value.shape[0]) != batch for value in state):
                raise ValueError("streaming feature state batch size changed")
            last_return = current - previous
            history = torch.cat([history, last_return.unsqueeze(1)], dim=1)
            history = history[:, -max(self.rolling_windows) :]
            running_max = torch.maximum(running_max, current)
            current_drawdown = running_max - current
            maximum_drawdown = torch.maximum(maximum_drawdown, current_drawdown)
            decay = current.new_tensor(self._decays).unsqueeze(0)
            ewma_numerator = (
                decay * ewma_numerator + last_return.pow(2).unsqueeze(1)
            )
            ewma_denominator = decay * ewma_denominator + 1.0

        current_drawdown = running_max - current
        time = torch.full_like(
            current,
            float(step) / float(max(1, self.num_steps - 1)),
        )
        features = [
            time,
            current,
            -current_drawdown,
            maximum_drawdown,
            last_return,
            last_return.abs(),
            last_return.pow(2) / self.dt,
        ]

        def trailing_mean(values: torch.Tensor, window: int) -> torch.Tensor:
            if int(values.shape[1]) == 0:
                return zeros
            usable = min(int(window), int(values.shape[1]))
            return values[:, -usable:].mean(dim=1)

        for window in (5, 20):
            features.append(trailing_mean(history, window))
        squared_history = history.pow(2)
        for window in self.rolling_windows:
            features.append(trailing_mean(squared_history, window) / self.dt)
        ewma_values = (
            ewma_numerator
            / ewma_denominator.clamp_min(torch.finfo(current.dtype).eps)
            / self.dt
        )
        features.extend(ewma_values.unbind(dim=1))
        downside_history = torch.relu(-history).pow(2)
        for window in self.downside_windows:
            features.append(trailing_mean(downside_history, window) / self.dt)

        result = torch.stack(features, dim=1)
        if int(result.shape[1]) != len(self.feature_names):
            raise RuntimeError("streaming compact feature dimension is inconsistent")
        return result, (
            current,
            running_max,
            maximum_drawdown,
            history,
            ewma_numerator,
            ewma_denominator,
        )


def _zero_init_final_linear(module: nn.Sequential) -> None:
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("the final residual head layer must be linear")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


class CausalFeatureAdjointResidualNetwork(nn.Module):
    """Adapted P/R residual on standardized observable prefix features.

    The feature map is causal and contains no future or latent state.  The
    zero-initialized output layers make the residual identically zero before
    training, so wrapping an existing generator preserves its law exactly.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int,
        noise_adjoint_dim: int,
        feature_map: CausalFeatureMap,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_dim: int = 96,
        num_layers: int = 1,
        standardized_feature_clip: float | None = 4.0,
    ) -> None:
        super().__init__()
        if min(
            int(state_dim),
            int(adjoint_dim),
            int(noise_adjoint_dim),
            int(hidden_dim),
            int(num_layers),
        ) < 1:
            raise ValueError("causal residual dimensions must be positive")
        if int(state_dim) != 1:
            raise ValueError("the current causal feature map is univariate")
        if feature_center.ndim != 2:
            raise ValueError("feature_center must have shape (N, f)")
        if tuple(feature_scale.shape) != tuple(feature_center.shape):
            raise ValueError("feature_scale must match feature_center")
        if int(feature_center.shape[1]) != len(feature_map.feature_names):
            raise ValueError("feature statistics do not match the feature map")
        if not bool(torch.isfinite(feature_center).all()):
            raise ValueError("feature_center must be finite")
        if not bool(torch.isfinite(feature_scale).all()) or bool(
            (feature_scale <= 0.0).any()
        ):
            raise ValueError("feature_scale must be finite and positive")
        if standardized_feature_clip is not None and float(
            standardized_feature_clip
        ) <= 0.0:
            raise ValueError("standardized_feature_clip must be positive")

        self.state_dim = int(state_dim)
        self.adjoint_dim = int(adjoint_dim)
        self.noise_adjoint_dim = int(noise_adjoint_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.feature_map = feature_map
        self.feature_names = tuple(str(name) for name in feature_map.feature_names)
        self.standardized_feature_clip = (
            None
            if standardized_feature_clip is None
            else float(standardized_feature_clip)
        )
        self.register_buffer("feature_center", feature_center.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        feature_dim = int(feature_center.shape[1])
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.expected_adjoint_next_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(adjoint_dim)),
        )
        self.expected_adjoint_noise_next_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(noise_adjoint_dim)),
        )
        _zero_init_final_linear(self.expected_adjoint_next_head)
        _zero_init_final_linear(self.expected_adjoint_noise_next_head)

    def _standardize(
        self, features: torch.Tensor, *, step_index: int | None = None
    ) -> torch.Tensor:
        if step_index is None:
            center = self.feature_center.unsqueeze(0).to(features)
            scale = self.feature_scale.unsqueeze(0).to(features)
        else:
            if not 0 <= int(step_index) < int(self.feature_center.shape[0]):
                raise ValueError("causal residual step exceeds the feature grid")
            center = self.feature_center[int(step_index)].unsqueeze(0).to(features)
            scale = self.feature_scale[int(step_index)].unsqueeze(0).to(features)
        standardized = (features - center) / scale
        if self.standardized_feature_clip is not None:
            clip = float(self.standardized_feature_clip)
            standardized = clip * torch.tanh(standardized / clip)
        return standardized

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3 or int(x_path.shape[1]) < 2:
            raise ValueError("x_path must have shape (B, N + 1, d)")
        if int(x_path.shape[2]) != self.state_dim:
            raise ValueError("x_path state dimension does not match the residual")
        features = self.feature_map.features_all(x_path)
        if int(features.shape[1]) != int(self.feature_center.shape[0]):
            raise ValueError("feature sequence does not match the stored grid")
        output, _ = self.gru(self._standardize(features))
        return AdjointMoments(
            expected_adjoint_next=self.expected_adjoint_next_head(output),
            expected_adjoint_noise_next=self.expected_adjoint_noise_next_head(output),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[torch.Tensor | None, object, int] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, object, int],
    ]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            gru_hidden = None
            feature_state = None
            step_index = 0
        else:
            gru_hidden, feature_state, step_index = hidden
        streaming = getattr(self.feature_map, "features_step", None)
        if callable(streaming):
            features, next_feature_state = streaming(
                x_t,
                state=feature_state,
                step_index=int(step_index),
            )
        else:
            if feature_state is None:
                prefix = x_t.unsqueeze(1)
            else:
                if not isinstance(feature_state, torch.Tensor):
                    raise TypeError("non-streaming feature state must be a prefix tensor")
                prefix = torch.cat([feature_state, x_t.unsqueeze(1)], dim=1)
            features = self.feature_map.features_at(prefix)
            next_feature_state = prefix
        normalized = self._standardize(features, step_index=int(step_index))
        output, next_gru_hidden = self.gru(normalized.unsqueeze(1), gru_hidden)
        encoded = output[:, -1, :]
        return (
            self.expected_adjoint_next_head(encoded),
            self.expected_adjoint_noise_next_head(encoded),
            (next_gru_hidden, next_feature_state, int(step_index) + 1),
        )


class CausalRegimeGatedAdjointResidualNetwork(nn.Module):
    """Zero-initialized causal P/R experts with smooth volatility gates.

    Three otherwise symmetric experts share a causal encoder and are mixed by
    train-law rolling-volatility tercile gates.  The gates use only the current
    observable prefix.  They therefore add no latent state or future
    information; their purpose is to keep an adapted correction learned in one
    volatility regime from being broadcast indiscriminately to the others.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int,
        noise_adjoint_dim: int,
        feature_map: CausalFeatureMap,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        gate_feature_index: int,
        gate_thresholds: torch.Tensor,
        gate_temperature: torch.Tensor,
        hidden_dim: int = 96,
        num_layers: int = 1,
        standardized_feature_clip: float | None = 4.0,
    ) -> None:
        super().__init__()
        if min(
            int(state_dim),
            int(adjoint_dim),
            int(noise_adjoint_dim),
            int(hidden_dim),
            int(num_layers),
        ) < 1:
            raise ValueError("causal regime residual dimensions must be positive")
        if int(state_dim) != 1:
            raise ValueError("the current causal regime feature map is univariate")
        if feature_center.ndim != 2 or tuple(feature_scale.shape) != tuple(
            feature_center.shape
        ):
            raise ValueError("feature moments must have matching shape (N, f)")
        feature_dim = int(feature_center.shape[1])
        gate_index = int(gate_feature_index)
        if not 0 <= gate_index < feature_dim:
            raise ValueError("gate_feature_index is outside the causal feature map")
        num_steps = int(feature_center.shape[0])
        if tuple(gate_thresholds.shape) != (num_steps, 2):
            raise ValueError("gate_thresholds must have shape (N, 2)")
        if tuple(gate_temperature.shape) != (num_steps,):
            raise ValueError("gate_temperature must have shape (N,)")
        if not bool(torch.isfinite(gate_thresholds).all().item()) or not bool(
            torch.isfinite(gate_temperature).all().item()
        ):
            raise ValueError("gate parameters must be finite")
        if bool((gate_temperature <= 0.0).any().item()):
            raise ValueError("gate_temperature must be strictly positive")
        if bool((gate_thresholds[:, 1] < gate_thresholds[:, 0]).any().item()):
            raise ValueError("gate thresholds must be ordered")
        if standardized_feature_clip is not None and float(
            standardized_feature_clip
        ) <= 0.0:
            raise ValueError("standardized_feature_clip must be positive")

        self.state_dim = int(state_dim)
        self.adjoint_dim = int(adjoint_dim)
        self.noise_adjoint_dim = int(noise_adjoint_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.feature_map = feature_map
        self.feature_names = tuple(str(name) for name in feature_map.feature_names)
        self.gate_feature_index = gate_index
        self.gate_feature_name = self.feature_names[gate_index]
        self.standardized_feature_clip = (
            None
            if standardized_feature_clip is None
            else float(standardized_feature_clip)
        )
        self.register_buffer("feature_center", feature_center.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.register_buffer("gate_thresholds", gate_thresholds.detach().clone())
        self.register_buffer("gate_temperature", gate_temperature.detach().clone())
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )

        def zero_head(output_dim: int) -> nn.Sequential:
            head = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Tanh(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )
            _zero_init_final_linear(head)
            return head

        self.expected_adjoint_next_heads = nn.ModuleList(
            [zero_head(int(adjoint_dim)) for _ in range(3)]
        )
        self.expected_adjoint_noise_next_heads = nn.ModuleList(
            [zero_head(int(noise_adjoint_dim)) for _ in range(3)]
        )

    def _standardize(
        self,
        features: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        if step_index is None:
            center = self.feature_center.unsqueeze(0).to(features)
            scale = self.feature_scale.unsqueeze(0).to(features)
        else:
            center = self.feature_center[int(step_index)].unsqueeze(0).to(features)
            scale = self.feature_scale[int(step_index)].unsqueeze(0).to(features)
        standardized = (features - center) / scale
        if self.standardized_feature_clip is not None:
            clip = float(self.standardized_feature_clip)
            standardized = clip * torch.tanh(standardized / clip)
        return standardized

    def _gates(
        self,
        features: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        values = features[..., self.gate_feature_index]
        if step_index is None:
            thresholds = self.gate_thresholds.unsqueeze(0).to(features)
            temperature = self.gate_temperature.unsqueeze(0).to(features)
        else:
            thresholds = self.gate_thresholds[int(step_index)].to(features)
            temperature = self.gate_temperature[int(step_index)].to(features)
        lower_switch = torch.sigmoid(
            (values - thresholds[..., 0]) / temperature
        )
        upper_switch = torch.sigmoid(
            (values - thresholds[..., 1]) / temperature
        )
        return torch.stack(
            (1.0 - lower_switch, lower_switch - upper_switch, upper_switch),
            dim=-1,
        )

    @staticmethod
    def _mix_experts(
        encoded: torch.Tensor,
        gates: torch.Tensor,
        heads: nn.ModuleList,
    ) -> torch.Tensor:
        expert_values = torch.stack([head(encoded) for head in heads], dim=-2)
        return (expert_values * gates.unsqueeze(-1)).sum(dim=-2)

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3 or int(x_path.shape[1]) < 2:
            raise ValueError("x_path must have shape (B, N + 1, d)")
        if int(x_path.shape[2]) != self.state_dim:
            raise ValueError("x_path state dimension does not match the residual")
        features = self.feature_map.features_all(x_path)
        if int(features.shape[1]) != int(self.feature_center.shape[0]):
            raise ValueError("feature sequence does not match the stored grid")
        encoded, _ = self.gru(self._standardize(features))
        gates = self._gates(features)
        return AdjointMoments(
            expected_adjoint_next=self._mix_experts(
                encoded, gates, self.expected_adjoint_next_heads
            ),
            expected_adjoint_noise_next=self._mix_experts(
                encoded, gates, self.expected_adjoint_noise_next_heads
            ),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[torch.Tensor | None, object, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, object, int]]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            gru_hidden = None
            feature_state = None
            step_index = 0
        else:
            gru_hidden, feature_state, step_index = hidden
        streaming = getattr(self.feature_map, "features_step", None)
        if callable(streaming):
            features, next_feature_state = streaming(
                x_t,
                state=feature_state,
                step_index=int(step_index),
            )
        else:
            if feature_state is None:
                prefix = x_t.unsqueeze(1)
            else:
                if not isinstance(feature_state, torch.Tensor):
                    raise TypeError("non-streaming feature state must be a prefix tensor")
                prefix = torch.cat([feature_state, x_t.unsqueeze(1)], dim=1)
            features = self.feature_map.features_at(prefix)
            next_feature_state = prefix
        output, next_gru_hidden = self.gru(
            self._standardize(features, step_index=int(step_index)).unsqueeze(1),
            gru_hidden,
        )
        encoded = output[:, -1, :]
        gates = self._gates(features, step_index=int(step_index))
        return (
            self._mix_experts(
                encoded, gates, self.expected_adjoint_next_heads
            ),
            self._mix_experts(
                encoded, gates, self.expected_adjoint_noise_next_heads
            ),
            (next_gru_hidden, next_feature_state, int(step_index) + 1),
        )


def fit_timewise_causal_regime_gates(
    feature_map: CausalFeatureMap,
    target_paths: torch.Tensor,
    *,
    feature_name: str = "rolling_realized_variance_20",
    temperature_fraction: float = 0.10,
    minimum_temperature: float = 1e-8,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Fit immutable smooth volatility-tercile gates on training paths."""

    fraction = float(temperature_fraction)
    minimum = float(minimum_temperature)
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("temperature_fraction must be positive and finite")
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_temperature must be positive and finite")
    names = tuple(str(name) for name in feature_map.feature_names)
    if str(feature_name) not in names:
        raise ValueError(f"causal feature map does not contain {feature_name!r}")
    feature_index = names.index(str(feature_name))
    with torch.no_grad():
        values = feature_map.features_all(target_paths)[..., feature_index].detach()
        thresholds = torch.quantile(
            values,
            torch.tensor(
                (1.0 / 3.0, 2.0 / 3.0),
                device=values.device,
                dtype=values.dtype,
            ),
            dim=0,
        ).transpose(0, 1)
        temperature = (
            fraction * (thresholds[:, 1] - thresholds[:, 0]).abs()
        ).clamp_min(minimum)
    return feature_index, thresholds, temperature


class _TimewiseLinearHead(nn.Module):
    """Vectorized collection of independent affine maps, one per time step."""

    def __init__(self, *, num_steps: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(
            torch.zeros(self.num_steps, self.output_dim, self.input_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_steps, self.output_dim))

    def reset_parameters(self) -> None:
        for step in range(self.num_steps):
            nn.init.kaiming_uniform_(self.weight[step], a=math.sqrt(5.0))
        bound = 1.0 / math.sqrt(float(self.input_dim))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (
            self.num_steps,
            self.input_dim,
        ):
            raise ValueError("timewise head features have the wrong shape")
        return (
            torch.einsum("bnf,nof->bno", features, self.weight)
            + self.bias.unsqueeze(0)
        )

    def forward_step(self, features: torch.Tensor, *, step_index: int) -> torch.Tensor:
        step = int(step_index)
        return torch.nn.functional.linear(
            features,
            self.weight[step],
            self.bias[step],
        )


class TimewiseLinearCausalAdjointResidualNetwork(nn.Module):
    """Timewise affine residual on a fixed observable causal feature basis.

    This is the direct representational counterpart of the timewise ridge
    conditional-expectation projection.  It retains the same neural regression
    loss and adapted control law, but has no shared recurrent encoder that can
    couple an accepted R update to a rejected representation update.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int,
        noise_adjoint_dim: int,
        feature_map: CausalFeatureMap,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        standardized_feature_clip: float | None = 4.0,
    ) -> None:
        super().__init__()
        if min(int(state_dim), int(adjoint_dim), int(noise_adjoint_dim)) < 1:
            raise ValueError("causal residual dimensions must be positive")
        if int(state_dim) != 1:
            raise ValueError("the current causal feature map is univariate")
        if feature_center.ndim != 2:
            raise ValueError("feature_center must have shape (N, f)")
        if tuple(feature_scale.shape) != tuple(feature_center.shape):
            raise ValueError("feature_scale must match feature_center")
        if int(feature_center.shape[1]) != len(feature_map.feature_names):
            raise ValueError("feature statistics do not match the feature map")
        if not bool(torch.isfinite(feature_center).all()):
            raise ValueError("feature_center must be finite")
        if not bool(torch.isfinite(feature_scale).all()) or bool(
            (feature_scale <= 0.0).any()
        ):
            raise ValueError("feature_scale must be finite and positive")
        if standardized_feature_clip is not None and float(
            standardized_feature_clip
        ) <= 0.0:
            raise ValueError("standardized_feature_clip must be positive")
        self.state_dim = int(state_dim)
        self.adjoint_dim = int(adjoint_dim)
        self.noise_adjoint_dim = int(noise_adjoint_dim)
        self.feature_map = feature_map
        self.feature_names = tuple(str(name) for name in feature_map.feature_names)
        self.standardized_feature_clip = (
            None
            if standardized_feature_clip is None
            else float(standardized_feature_clip)
        )
        self.register_buffer("feature_center", feature_center.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        num_steps, feature_dim = map(int, feature_center.shape)
        self.expected_adjoint_next_head = _TimewiseLinearHead(
            num_steps=num_steps,
            input_dim=feature_dim,
            output_dim=int(adjoint_dim),
        )
        self.expected_adjoint_noise_next_head = _TimewiseLinearHead(
            num_steps=num_steps,
            input_dim=feature_dim,
            output_dim=int(noise_adjoint_dim),
        )

    def _standardize(
        self,
        features: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        if step_index is None:
            center = self.feature_center.unsqueeze(0).to(features)
            scale = self.feature_scale.unsqueeze(0).to(features)
        else:
            center = self.feature_center[int(step_index)].unsqueeze(0).to(features)
            scale = self.feature_scale[int(step_index)].unsqueeze(0).to(features)
        standardized = (features - center) / scale
        if self.standardized_feature_clip is not None:
            clip = float(self.standardized_feature_clip)
            standardized = clip * torch.tanh(standardized / clip)
        return standardized

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3 or int(x_path.shape[2]) != self.state_dim:
            raise ValueError("x_path must have shape (B, N + 1, state_dim)")
        features = self.feature_map.features_all(x_path)
        if int(features.shape[1]) != self.expected_adjoint_next_head.num_steps:
            raise ValueError("feature sequence does not match the stored grid")
        normalized = self._standardize(features)
        return AdjointMoments(
            expected_adjoint_next=self.expected_adjoint_next_head(normalized),
            expected_adjoint_noise_next=(
                self.expected_adjoint_noise_next_head(normalized)
            ),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[object, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[object, int]]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            feature_state = None
            step_index = 0
        else:
            feature_state, step_index = hidden
        if not 0 <= int(step_index) < self.expected_adjoint_next_head.num_steps:
            raise ValueError("causal residual step exceeds the feature grid")
        streaming = getattr(self.feature_map, "features_step", None)
        if callable(streaming):
            features, next_feature_state = streaming(
                x_t,
                state=feature_state,
                step_index=int(step_index),
            )
        else:
            if feature_state is None:
                prefix = x_t.unsqueeze(1)
            else:
                if not isinstance(feature_state, torch.Tensor):
                    raise TypeError("non-streaming feature state must be a prefix tensor")
                prefix = torch.cat([feature_state, x_t.unsqueeze(1)], dim=1)
            features = self.feature_map.features_at(prefix)
            next_feature_state = prefix
        normalized = self._standardize(features, step_index=int(step_index))
        return (
            self.expected_adjoint_next_head.forward_step(
                normalized,
                step_index=int(step_index),
            ),
            self.expected_adjoint_noise_next_head.forward_step(
                normalized,
                step_index=int(step_index),
            ),
            (next_feature_state, int(step_index) + 1),
        )


class FrozenBaseCausalResidualAdjointNetwork(nn.Module):
    """Frozen source adjoints plus a trainable causal P/R residual."""

    def __init__(
        self,
        *,
        base_network: nn.Module,
        residual_network: CausalFeatureAdjointResidualNetwork,
    ) -> None:
        super().__init__()
        for attribute in ("state_dim", "adjoint_dim", "noise_adjoint_dim"):
            if not hasattr(base_network, attribute):
                raise ValueError(f"base_network must expose {attribute}")
            if int(getattr(base_network, attribute)) != int(
                getattr(residual_network, attribute)
            ):
                raise ValueError(f"base and residual {attribute} must match")
        self.base_network = base_network
        self.residual_network = residual_network
        self.state_dim = int(getattr(base_network, "state_dim"))
        self.adjoint_dim = int(getattr(base_network, "adjoint_dim"))
        self.noise_adjoint_dim = int(getattr(base_network, "noise_adjoint_dim"))
        for parameter in self.base_network.parameters():
            parameter.requires_grad_(False)

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        base = self.base_network(x_path)
        residual = self.residual_network(x_path)
        return AdjointMoments(
            expected_adjoint_next=(
                base.expected_adjoint_next + residual.expected_adjoint_next
            ),
            expected_adjoint_noise_next=(
                base.expected_adjoint_noise_next
                + residual.expected_adjoint_noise_next
            ),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[object, object] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[object, object]]:
        if hidden is None:
            base_hidden = None
            residual_hidden = None
        else:
            base_hidden, residual_hidden = hidden
        base_p, base_r, next_base_hidden = self.base_network.forward_step(
            x_t=x_t,
            hidden=base_hidden,
        )
        residual_p, residual_r, next_residual_hidden = (
            self.residual_network.forward_step(
                x_t=x_t,
                hidden=residual_hidden,
            )
        )
        return (
            base_p + residual_p,
            base_r + residual_r,
            (next_base_hidden, next_residual_hidden),
        )


def fit_timewise_causal_feature_normalization(
    feature_map: CausalFeatureMap,
    target_paths: torch.Tensor,
    *,
    minimum_scale: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit immutable train-law feature moments without future information."""

    if float(minimum_scale) <= 0.0:
        raise ValueError("minimum_scale must be positive")
    with torch.no_grad():
        features = feature_map.features_all(target_paths).detach()
        center = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False)
        fallback = torch.ones_like(scale)
        scale = torch.where(scale > float(minimum_scale), scale, fallback)
    return center, scale


__all__ = [
    "CausalFeatureAdjointResidualNetwork",
    "CausalRegimeGatedAdjointResidualNetwork",
    "FrozenBaseCausalResidualAdjointNetwork",
    "IncrementCausalFeatureMap",
    "fit_timewise_causal_regime_gates",
    "fit_timewise_causal_feature_normalization",
]
