from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import torch

from deep_mkv_gen_path_dt.ce import fit_ridge
from deep_mkv_gen_path_dt.config import _require_finite_float
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


@dataclass(frozen=True)
class LocalGaussianReferenceKernel:
    """
    Path-prefix Gaussian reference transition fitted from target paths.

    At step n, the fitted reference is

        X_{n+1} = X_n + b_ref(n, X_{0:n}) dt
                  + sigma_ref(n, X_{0:n}) sqrt(dt) eps.

    Both b_ref and log(sigma_ref^2) are ridge regressions on flattened path
    prefixes.  The model uses sigma_ref inside the specific-entropy control
    map.
    """

    grid: DiscreteTimeGrid
    drift_coefficients: tuple[torch.Tensor, ...]
    log_variance_coefficients: tuple[torch.Tensor, ...]
    feature_means: tuple[torch.Tensor, ...]
    feature_stds: tuple[torch.Tensor, ...]
    sigma_min: float = 1e-3
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        if len(self.drift_coefficients) != int(self.grid.num_steps):
            raise ValueError("drift_coefficients length must match grid.num_steps")
        if len(self.log_variance_coefficients) != int(self.grid.num_steps):
            raise ValueError("log_variance_coefficients length must match grid.num_steps")
        if len(self.feature_means) != int(self.grid.num_steps):
            raise ValueError("feature_means length must match grid.num_steps")
        if len(self.feature_stds) != int(self.grid.num_steps):
            raise ValueError("feature_stds length must match grid.num_steps")
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        sigma_max = None if self.sigma_max is None else _require_finite_float("sigma_max", self.sigma_max)
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    def _design(self, x_prefix: torch.Tensor, *, step_index: int) -> torch.Tensor:
        if x_prefix.ndim != 3:
            raise ValueError("x_prefix must have shape (B, n + 1, d)")
        step = int(step_index)
        if not (0 <= step < int(self.grid.num_steps)):
            raise ValueError("step_index is out of bounds")
        features = x_prefix.reshape(int(x_prefix.shape[0]), -1)
        expected_width = int(self.feature_means[step].shape[1])
        if int(features.shape[1]) != expected_width:
            raise ValueError("x_prefix width does not match the fitted reference step")
        mean = self.feature_means[step].to(device=x_prefix.device, dtype=x_prefix.dtype)
        std = self.feature_stds[step].to(device=x_prefix.device, dtype=x_prefix.dtype)
        normalized = (features - mean) / std
        ones = torch.ones(int(features.shape[0]), 1, device=x_prefix.device, dtype=x_prefix.dtype)
        return torch.cat([ones, normalized], dim=1)

    def evaluate(self, x_prefix: torch.Tensor, *, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        design = self._design(x_prefix, step_index=int(step_index))
        step = int(step_index)
        drift_coefficients = self.drift_coefficients[step].to(device=x_prefix.device, dtype=x_prefix.dtype)
        log_variance_coefficients = self.log_variance_coefficients[step].to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        )
        drift = design @ drift_coefficients
        log_variance = design @ log_variance_coefficients
        # Clamp in log space. This gives the same bounded volatility as
        # clamping after exp, while keeping the path derivative finite for
        # off-manifold prefixes whose fitted log variance is very large.
        log_variance = log_variance.clamp_min(
            2.0 * math.log(float(self.sigma_min))
        )
        if self.sigma_max is not None:
            log_variance = log_variance.clamp_max(
                2.0 * math.log(float(self.sigma_max))
            )
        sigma = torch.exp(0.5 * log_variance)
        return drift, sigma

    def drift_step_path_vjp(
        self,
        x_prefix: torch.Tensor,
        drift_cotangent: torch.Tensor,
        *,
        step_index: int,
    ) -> torch.Tensor:
        """Exact VJP of one fitted drift output with respect to its path prefix."""

        if x_prefix.ndim != 3:
            raise ValueError("x_prefix must have shape (B, n + 1, d)")
        if drift_cotangent.ndim != 2:
            raise ValueError("drift_cotangent must have shape (B, d)")
        if int(x_prefix.shape[0]) != int(drift_cotangent.shape[0]):
            raise ValueError("x_prefix and drift_cotangent batch sizes must match")
        if int(x_prefix.shape[2]) != int(drift_cotangent.shape[1]):
            raise ValueError("x_prefix and drift_cotangent state dimensions must match")
        step = int(step_index)
        if int(x_prefix.shape[1]) != step + 1:
            raise ValueError("x_prefix time dimension must equal step_index + 1")

        coefficients = self.drift_coefficients[step].to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        )
        std = self.feature_stds[step].to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        )
        feature_cotangent = drift_cotangent @ coefficients[1:].transpose(0, 1)
        return (feature_cotangent / std).reshape_as(x_prefix)

    def evaluate_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted prefix volatility at every date."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        return torch.stack(
            [
                self.evaluate(paths[:, : step + 1, :], step_index=step)[1]
                for step in range(int(self.grid.num_steps))
            ],
            dim=1,
        )

    def sigma_path_vjp(
        self,
        paths: torch.Tensor,
        sigma_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form VJP of all prefix-regression volatilities."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        batch, point_count, dimension = paths.shape
        expected = (batch, point_count - 1, dimension)
        if tuple(sigma_cotangent.shape) != expected:
            raise ValueError(f"sigma_cotangent must have shape {expected}")
        if int(point_count) != int(self.grid.num_steps) + 1:
            raise ValueError("paths must align with the reference grid")
        if paths.device != sigma_cotangent.device or paths.dtype != sigma_cotangent.dtype:
            raise ValueError("paths and sigma_cotangent must share device and dtype")

        with torch.no_grad():
            result = torch.zeros_like(paths)
            lower = 2.0 * math.log(float(self.sigma_min))
            upper = (
                None
                if self.sigma_max is None
                else 2.0 * math.log(float(self.sigma_max))
            )
            for step in range(int(self.grid.num_steps)):
                prefix = paths[:, : step + 1, :]
                features = prefix.reshape(batch, -1)
                mean = self.feature_means[step].to(
                    device=paths.device,
                    dtype=paths.dtype,
                )
                std = self.feature_stds[step].to(
                    device=paths.device,
                    dtype=paths.dtype,
                )
                coefficients = self.log_variance_coefficients[step].to(
                    device=paths.device,
                    dtype=paths.dtype,
                )
                normalized = (features - mean) / std
                raw_log_variance = coefficients[0].unsqueeze(0) + normalized @ coefficients[1:]
                active = raw_log_variance >= lower
                if upper is not None:
                    active = active & (raw_log_variance <= upper)
                bounded = raw_log_variance.clamp_min(lower)
                if upper is not None:
                    bounded = bounded.clamp_max(upper)
                sigma = torch.exp(0.5 * bounded)
                log_variance_bar = (
                    0.5
                    * sigma
                    * sigma_cotangent[:, step, :]
                    * active.to(dtype=paths.dtype)
                )
                normalized_bar = log_variance_bar @ coefficients[1:].transpose(0, 1)
                feature_bar = normalized_bar / std
                result[:, : step + 1, :].add_(
                    feature_bar.reshape(batch, step + 1, dimension)
                )
            return result

    def summary(self) -> dict[str, float]:
        drift_norms = [float(torch.linalg.vector_norm(coeff).item()) for coeff in self.drift_coefficients]
        log_var_norms = [float(torch.linalg.vector_norm(coeff).item()) for coeff in self.log_variance_coefficients]
        return {
            "reference_sigma_min": float(self.sigma_min),
            "reference_sigma_max": float("nan") if self.sigma_max is None else float(self.sigma_max),
            "reference_feature_dim_max": float(max(int(mean.shape[1]) for mean in self.feature_means)),
            "reference_drift_coeff_l2_mean": float(sum(drift_norms) / len(drift_norms)),
            "reference_log_variance_coeff_l2_mean": float(sum(log_var_norms) / len(log_var_norms)),
        }


@dataclass(frozen=True)
class GuyonLekeufackReferenceKernel:
    """Causal path-dependent volatility reference with two exponential scales.

    The volatility rule is the two-exponential Guyon--Lekeufack specification

        sigma = beta_0 + beta_1 R_1 + beta_2 sqrt(R_2),

    where ``R_1`` is built from past increments and ``R_2`` from past squared
    increments.  Each signal is a convex combination of a fast and a slow
    exponential filter.  The drift is supplied by a separately fitted causal
    Gaussian reference so that experiments can change the volatility reference
    without changing the drift protocol.
    """

    grid: DiscreteTimeGrid
    drift_kernel: LocalGaussianReferenceKernel
    trend_lambdas: tuple[float, float]
    activity_lambdas: tuple[float, float]
    trend_weight: float
    activity_weight: float
    beta_0: torch.Tensor
    beta_1: torch.Tensor
    beta_2: torch.Tensor
    initial_trend: torch.Tensor
    initial_activity: torch.Tensor
    calibration_mse: float
    validation_mse: float
    likelihood_scale: float = 1.0
    activity_update: str = "realized_returns"
    calibration_nll: float | None = None
    validation_nll: float | None = None
    sigma_min: float = 1e-3
    sigma_max: float | None = None
    _cache_step: int = field(default=-1, init=False, repr=False, compare=False)
    _cache_trend_states: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _cache_activity_states: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _cache_sigma: torch.Tensor | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.drift_kernel.grid != self.grid:
            raise ValueError("drift_kernel and Guyon reference must use the same grid")
        trend_lambdas = tuple(
            _require_finite_float(f"trend_lambdas[{index}]", value)
            for index, value in enumerate(self.trend_lambdas)
        )
        activity_lambdas = tuple(
            _require_finite_float(f"activity_lambdas[{index}]", value)
            for index, value in enumerate(self.activity_lambdas)
        )
        if len(trend_lambdas) != 2 or any(value <= 0.0 for value in trend_lambdas):
            raise ValueError("trend_lambdas must contain two positive rates")
        if len(activity_lambdas) != 2 or any(value <= 0.0 for value in activity_lambdas):
            raise ValueError("activity_lambdas must contain two positive rates")
        trend_weight = _require_finite_float("trend_weight", self.trend_weight)
        activity_weight = _require_finite_float("activity_weight", self.activity_weight)
        likelihood_scale = _require_finite_float("likelihood_scale", self.likelihood_scale)
        activity_update = str(self.activity_update)
        if activity_update not in {"realized_returns", "structural_variance"}:
            raise ValueError(
                "activity_update must be 'realized_returns' or 'structural_variance'"
            )
        if not (0.0 <= trend_weight <= 1.0):
            raise ValueError("trend_weight must be in [0, 1]")
        if not (0.0 <= activity_weight <= 1.0):
            raise ValueError("activity_weight must be in [0, 1]")
        if likelihood_scale <= 0.0:
            raise ValueError("likelihood_scale must be positive")
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        sigma_max = None if self.sigma_max is None else _require_finite_float("sigma_max", self.sigma_max)
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        coefficient_shapes = {
            tuple(self.beta_0.shape),
            tuple(self.beta_1.shape),
            tuple(self.beta_2.shape),
            tuple(self.initial_trend.shape),
            tuple(self.initial_activity.shape),
        }
        if len(coefficient_shapes) != 1 or self.beta_0.ndim != 2 or int(self.beta_0.shape[0]) != 1:
            raise ValueError("Guyon coefficients and initial states must all have shape (1, d)")
        if torch.any(self.beta_2 < 0.0):
            raise ValueError("beta_2 must be non-negative")
        if torch.any(self.initial_activity <= 0.0):
            raise ValueError("initial_activity must be positive")
        object.__setattr__(self, "trend_lambdas", trend_lambdas)
        object.__setattr__(self, "activity_lambdas", activity_lambdas)
        object.__setattr__(self, "trend_weight", trend_weight)
        object.__setattr__(self, "activity_weight", activity_weight)
        object.__setattr__(self, "likelihood_scale", likelihood_scale)
        object.__setattr__(self, "activity_update", activity_update)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    def _sigma_from_states(
        self,
        trend_states: list[torch.Tensor],
        activity_states: list[torch.Tensor],
    ) -> torch.Tensor:
        trend = (1.0 - float(self.trend_weight)) * trend_states[0] + float(
            self.trend_weight
        ) * trend_states[1]
        activity = (1.0 - float(self.activity_weight)) * activity_states[0] + float(
            self.activity_weight
        ) * activity_states[1]
        beta_0 = self.beta_0.to(device=trend.device, dtype=trend.dtype)
        beta_1 = self.beta_1.to(device=trend.device, dtype=trend.dtype)
        beta_2 = self.beta_2.to(device=trend.device, dtype=trend.dtype)
        sigma = beta_0 + beta_1 * trend + beta_2 * torch.sqrt(activity.clamp_min(1e-12))
        sigma = sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return sigma

    def _structural_states_for_prefix(
        self,
        x_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Advance cached structural states once, with a safe replay fallback."""
        batch_size = int(x_prefix.shape[0])
        dimension = int(x_prefix.shape[2])
        initial_trend = self.initial_trend.to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        ).expand(batch_size, dimension)
        initial_activity = self.initial_activity.to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        ).expand(batch_size, dimension)
        step = int(step_index)
        cached_trend = self._cache_trend_states
        cached_activity = self._cache_activity_states
        cached_sigma = self._cache_sigma
        cache_matches = (
            step > 0
            and int(self._cache_step) == step - 1
            and cached_trend is not None
            and cached_activity is not None
            and cached_sigma is not None
            and tuple(cached_sigma.shape) == (batch_size, dimension)
            and cached_sigma.device == x_prefix.device
            and cached_sigma.dtype == x_prefix.dtype
        )
        if step == 0:
            return [initial_trend, initial_trend], [initial_activity, initial_activity]
        if cache_matches:
            increment = x_prefix[:, -1, :] - x_prefix[:, -2, :]
            trend_states = [
                math.exp(-rate * float(self.grid.dt)) * (state + rate * increment)
                for state, rate in zip(cached_trend, self.trend_lambdas)
            ]
            activity_states = [
                math.exp(-rate * float(self.grid.dt))
                * (state + rate * cached_sigma.pow(2) * float(self.grid.dt))
                for state, rate in zip(cached_activity, self.activity_lambdas)
            ]
            return trend_states, activity_states

        trend_states = [initial_trend, initial_trend]
        activity_states = [initial_activity, initial_activity]
        for increment in (x_prefix[:, 1:, :] - x_prefix[:, :-1, :]).unbind(dim=1):
            sigma = self._sigma_from_states(trend_states, activity_states)
            trend_states = [
                math.exp(-rate * float(self.grid.dt)) * (state + rate * increment)
                for state, rate in zip(trend_states, self.trend_lambdas)
            ]
            activity_states = [
                math.exp(-rate * float(self.grid.dt))
                * (state + rate * sigma.pow(2) * float(self.grid.dt))
                for state, rate in zip(activity_states, self.activity_lambdas)
            ]
        return trend_states, activity_states

    @staticmethod
    def _filtered_state(
        increments: torch.Tensor,
        *,
        rate: float,
        dt: float,
        initial: torch.Tensor,
        squared: bool,
    ) -> torch.Tensor:
        """Return the exponential state after all increments in a prefix."""
        batch_size = int(increments.shape[0])
        dimension = int(increments.shape[2])
        initial_value = initial.to(device=increments.device, dtype=increments.dtype).expand(batch_size, dimension)
        count = int(increments.shape[1])
        if count == 0:
            return initial_value
        powers = torch.arange(
            count,
            0,
            -1,
            device=increments.device,
            dtype=increments.dtype,
        )
        decay_weights = torch.exp(-float(rate) * float(dt) * powers).view(1, count, 1)
        innovations = increments.pow(2) if squared else increments
        return (
            math.exp(-float(rate) * float(dt) * count) * initial_value
            + float(rate) * torch.sum(decay_weights * innovations, dim=1)
        )

    def _signals(self, x_prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_prefix.ndim != 3:
            raise ValueError("x_prefix must have shape (B, n + 1, d)")
        increments = x_prefix[:, 1:, :] - x_prefix[:, :-1, :]
        trend_components = tuple(
            self._filtered_state(
                increments,
                rate=rate,
                dt=float(self.grid.dt),
                initial=self.initial_trend,
                squared=False,
            )
            for rate in self.trend_lambdas
        )
        if self.activity_update == "realized_returns":
            activity_components = tuple(
                self._filtered_state(
                    increments,
                    rate=rate,
                    dt=float(self.grid.dt),
                    initial=self.initial_activity,
                    squared=True,
                )
                for rate in self.activity_lambdas
            )
        else:
            batch_size = int(x_prefix.shape[0])
            dimension = int(x_prefix.shape[2])
            trend_states = [
                self.initial_trend.to(device=x_prefix.device, dtype=x_prefix.dtype).expand(
                    batch_size,
                    dimension,
                )
                for _ in self.trend_lambdas
            ]
            activity_states = [
                self.initial_activity.to(device=x_prefix.device, dtype=x_prefix.dtype).expand(
                    batch_size,
                    dimension,
                )
                for _ in self.activity_lambdas
            ]
            for increment in increments.unbind(dim=1):
                sigma = self._sigma_from_states(trend_states, activity_states)
                trend_states = [
                    math.exp(-rate * float(self.grid.dt)) * (state + rate * increment)
                    for state, rate in zip(trend_states, self.trend_lambdas)
                ]
                activity_states = [
                    math.exp(-rate * float(self.grid.dt))
                    * (state + rate * sigma.pow(2) * float(self.grid.dt))
                    for state, rate in zip(activity_states, self.activity_lambdas)
                ]
            activity_components = tuple(activity_states)
        trend = (1.0 - float(self.trend_weight)) * trend_components[0] + float(
            self.trend_weight
        ) * trend_components[1]
        activity = (1.0 - float(self.activity_weight)) * activity_components[0] + float(
            self.activity_weight
        ) * activity_components[1]
        return trend, activity

    def evaluate(self, x_prefix: torch.Tensor, *, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        step = int(step_index)
        if not (0 <= step < int(self.grid.num_steps)):
            raise ValueError("step_index is out of bounds")
        if int(x_prefix.shape[1]) != step + 1:
            raise ValueError("x_prefix time dimension must equal step_index + 1")
        drift, _ = self.drift_kernel.evaluate(x_prefix, step_index=step)
        if self.activity_update == "structural_variance":
            trend_states, activity_states = self._structural_states_for_prefix(
                x_prefix,
                step_index=step,
            )
            sigma = self._sigma_from_states(trend_states, activity_states)
            object.__setattr__(self, "_cache_step", step)
            object.__setattr__(self, "_cache_trend_states", tuple(trend_states))
            object.__setattr__(self, "_cache_activity_states", tuple(activity_states))
            object.__setattr__(self, "_cache_sigma", sigma)
        else:
            trend, activity = self._signals(x_prefix)
            sigma = self._sigma_from_states(
                [trend, trend],
                [activity, activity],
            )
        return drift, sigma

    def drift_step_path_vjp(
        self,
        x_prefix: torch.Tensor,
        drift_cotangent: torch.Tensor,
        *,
        step_index: int,
    ) -> torch.Tensor:
        """Delegate the drift VJP to the separately fitted drift kernel."""

        return self.drift_kernel.drift_step_path_vjp(
            x_prefix,
            drift_cotangent,
            step_index=int(step_index),
        )

    def evaluate_all(self, paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        batch_size = int(paths.shape[0])
        dimension = int(paths.shape[2])
        initial_trend = self.initial_trend.to(device=paths.device, dtype=paths.dtype).expand(
            batch_size,
            dimension,
        )
        initial_activity = self.initial_activity.to(device=paths.device, dtype=paths.dtype).expand(
            batch_size,
            dimension,
        )
        trend_states = [initial_trend, initial_trend]
        activity_states = [initial_activity, initial_activity]
        drifts: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        for step in range(int(self.grid.num_steps)):
            drift, _ = self.drift_kernel.evaluate(paths[:, : step + 1, :], step_index=step)
            sigma = self._sigma_from_states(trend_states, activity_states)
            drifts.append(drift)
            sigmas.append(sigma)
            increment = paths[:, step + 1, :] - paths[:, step, :]
            trend_states = [
                math.exp(-rate * float(self.grid.dt)) * (state + rate * increment)
                for state, rate in zip(trend_states, self.trend_lambdas)
            ]
            activity_source = (
                increment.pow(2)
                if self.activity_update == "realized_returns"
                else sigma.pow(2) * float(self.grid.dt)
            )
            activity_states = [
                math.exp(-rate * float(self.grid.dt)) * (state + rate * activity_source)
                for state, rate in zip(activity_states, self.activity_lambdas)
            ]
        return torch.stack(drifts, dim=1), torch.stack(sigmas, dim=1)

    def evaluate_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Evaluate only the recurrent volatility path.

        Running-cost and Hamiltonian calculations never use the fitted drift.
        Avoiding its prefix regressions is important in the inner MP loop and
        leaves :meth:`evaluate_all` unchanged for callers that need both
        quantities.
        """

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        batch_size = int(paths.shape[0])
        dimension = int(paths.shape[2])
        trend_states = [
            self.initial_trend.to(device=paths.device, dtype=paths.dtype).expand(
                batch_size,
                dimension,
            )
            for _ in self.trend_lambdas
        ]
        activity_states = [
            self.initial_activity.to(device=paths.device, dtype=paths.dtype).expand(
                batch_size,
                dimension,
            )
            for _ in self.activity_lambdas
        ]
        sigmas: list[torch.Tensor] = []
        for step in range(int(self.grid.num_steps)):
            sigma = self._sigma_from_states(trend_states, activity_states)
            sigmas.append(sigma)
            increment = paths[:, step + 1, :] - paths[:, step, :]
            trend_states = [
                math.exp(-rate * float(self.grid.dt)) * (state + rate * increment)
                for state, rate in zip(trend_states, self.trend_lambdas)
            ]
            activity_source = (
                increment.pow(2)
                if self.activity_update == "realized_returns"
                else sigma.pow(2) * float(self.grid.dt)
            )
            activity_states = [
                math.exp(-rate * float(self.grid.dt))
                * (state + rate * activity_source)
                for state, rate in zip(activity_states, self.activity_lambdas)
            ]
        return torch.stack(sigmas, dim=1)

    def evaluate_drifts_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Evaluate only the fitted drift, without advancing volatility."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        return torch.stack(
            [
                self.drift_kernel.evaluate(
                    paths[:, : step + 1, :],
                    step_index=step,
                )[0]
                for step in range(int(self.grid.num_steps))
            ],
            dim=1,
        )

    def structural_sigma_path_vjp(
        self,
        paths: torch.Tensor,
        sigma_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        """Return an exact path VJP for the structural-volatility recursion.

        ``sigma_cotangent`` is the derivative of a scalar objective with
        respect to the reference volatility at every date.  The routine
        performs the reverse sweep of the fixed Guyon state recursion
        explicitly.  This is mathematically identical to differentiating
        :meth:`evaluate_all` with autograd, but avoids constructing thousands
        of tiny autograd nodes in every MP update.

        The method is deliberately restricted to ``structural_variance``.
        The realized-return recursion is already vectorizable and does not
        exhibit the same reverse-through-time bottleneck.
        """

        if self.activity_update != "structural_variance":
            raise ValueError(
                "structural_sigma_path_vjp requires structural_variance activity"
            )
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        batch_size, point_count, dimension = paths.shape
        expected = (int(batch_size), int(point_count) - 1, int(dimension))
        if tuple(sigma_cotangent.shape) != expected:
            raise ValueError(f"sigma_cotangent must have shape {expected}")
        if int(point_count) != int(self.grid.num_steps) + 1:
            raise ValueError("paths must align with the Guyon reference grid")
        if paths.device != sigma_cotangent.device:
            raise ValueError("paths and sigma_cotangent must share a device")
        if paths.dtype != sigma_cotangent.dtype:
            raise ValueError("paths and sigma_cotangent must share a dtype")

        # This VJP is used as a numerical first derivative.  Keeping the
        # sweep outside autograd is the source of the speedup and prevents an
        # accidental, unsupported use as a higher-order derivative.
        with torch.no_grad():
            dt = float(self.grid.dt)
            trend_rates = tuple(float(value) for value in self.trend_lambdas)
            activity_rates = tuple(float(value) for value in self.activity_lambdas)
            trend_decay = tuple(math.exp(-value * dt) for value in trend_rates)
            activity_decay = tuple(
                math.exp(-value * dt) for value in activity_rates
            )
            trend_weights = (1.0 - float(self.trend_weight), float(self.trend_weight))
            activity_weights = (
                1.0 - float(self.activity_weight),
                float(self.activity_weight),
            )
            beta_1 = self.beta_1.to(device=paths.device, dtype=paths.dtype)
            beta_2 = self.beta_2.to(device=paths.device, dtype=paths.dtype)
            beta_0 = self.beta_0.to(device=paths.device, dtype=paths.dtype)

            initial_trend = self.initial_trend.to(
                device=paths.device,
                dtype=paths.dtype,
            ).expand(int(batch_size), int(dimension))
            initial_activity = self.initial_activity.to(
                device=paths.device,
                dtype=paths.dtype,
            ).expand(int(batch_size), int(dimension))
            trend_states = [initial_trend, initial_trend]
            activity_states = [initial_activity, initial_activity]

            # Only the quantities needed by the reverse sweep are retained.
            sigma_rows: list[torch.Tensor] = []
            sqrt_activity_rows: list[torch.Tensor] = []
            raw_sigma_rows: list[torch.Tensor] = []
            for step in range(int(point_count) - 1):
                trend = (
                    trend_weights[0] * trend_states[0]
                    + trend_weights[1] * trend_states[1]
                )
                activity = (
                    activity_weights[0] * activity_states[0]
                    + activity_weights[1] * activity_states[1]
                )
                sqrt_activity = torch.sqrt(activity.clamp_min(1e-12))
                raw_sigma = beta_0 + beta_1 * trend + beta_2 * sqrt_activity
                sigma = raw_sigma.clamp_min(float(self.sigma_min))
                if self.sigma_max is not None:
                    sigma = sigma.clamp_max(float(self.sigma_max))
                sigma_rows.append(sigma)
                sqrt_activity_rows.append(sqrt_activity)
                raw_sigma_rows.append(raw_sigma)

                increment = paths[:, step + 1, :] - paths[:, step, :]
                trend_states = [
                    decay * (state + rate * increment)
                    for state, rate, decay in zip(
                        trend_states,
                        trend_rates,
                        trend_decay,
                    )
                ]
                activity_states = [
                    decay * (state + rate * sigma.pow(2) * dt)
                    for state, rate, decay in zip(
                        activity_states,
                        activity_rates,
                        activity_decay,
                    )
                ]

            path_gradient = torch.zeros_like(paths)
            trend_cotangent_next = [
                torch.zeros_like(initial_trend),
                torch.zeros_like(initial_trend),
            ]
            activity_cotangent_next = [
                torch.zeros_like(initial_activity),
                torch.zeros_like(initial_activity),
            ]

            for step in range(int(point_count) - 2, -1, -1):
                sigma = sigma_rows[step]
                sigma_bar = sigma_cotangent[:, step, :].clone()
                for next_bar, rate, decay in zip(
                    activity_cotangent_next,
                    activity_rates,
                    activity_decay,
                ):
                    sigma_bar.add_(
                        next_bar * (decay * rate * 2.0 * sigma * dt)
                    )

                raw_sigma = raw_sigma_rows[step]
                active = raw_sigma >= float(self.sigma_min)
                if self.sigma_max is not None:
                    active = active & (raw_sigma <= float(self.sigma_max))
                raw_sigma_bar = sigma_bar * active.to(dtype=paths.dtype)

                increment_bar = torch.zeros_like(initial_trend)
                trend_cotangent_current: list[torch.Tensor] = []
                for weight, rate, decay, next_bar in zip(
                    trend_weights,
                    trend_rates,
                    trend_decay,
                    trend_cotangent_next,
                ):
                    increment_bar.add_(next_bar * (decay * rate))
                    trend_cotangent_current.append(
                        decay * next_bar + weight * beta_1 * raw_sigma_bar
                    )

                sqrt_activity = sqrt_activity_rows[step]
                activity_derivative = 0.5 / sqrt_activity.clamp_min(1e-12)
                activity_cotangent_current = [
                    decay * next_bar
                    + weight * beta_2 * activity_derivative * raw_sigma_bar
                    for weight, decay, next_bar in zip(
                        activity_weights,
                        activity_decay,
                        activity_cotangent_next,
                    )
                ]

                path_gradient[:, step, :].sub_(increment_bar)
                path_gradient[:, step + 1, :].add_(increment_bar)
                trend_cotangent_next = trend_cotangent_current
                activity_cotangent_next = activity_cotangent_current

            return path_gradient

    def summary(self) -> dict[str, float]:
        dt = float(self.grid.dt)
        return self.drift_kernel.summary() | {
            "reference_kind_guyon_lekeufack": 1.0,
            "guyon_trend_half_life_fast_steps": math.log(2.0) / (self.trend_lambdas[0] * dt),
            "guyon_trend_half_life_slow_steps": math.log(2.0) / (self.trend_lambdas[1] * dt),
            "guyon_activity_half_life_fast_steps": math.log(2.0) / (self.activity_lambdas[0] * dt),
            "guyon_activity_half_life_slow_steps": math.log(2.0) / (self.activity_lambdas[1] * dt),
            "guyon_trend_weight": float(self.trend_weight),
            "guyon_activity_weight": float(self.activity_weight),
            "guyon_beta_0_mean": float(self.beta_0.mean().item()),
            "guyon_beta_1_mean": float(self.beta_1.mean().item()),
            "guyon_beta_2_mean": float(self.beta_2.mean().item()),
            "guyon_likelihood_scale": float(self.likelihood_scale),
            "guyon_activity_update_structural": float(
                self.activity_update == "structural_variance"
            ),
            "guyon_calibration_mse": float(self.calibration_mse),
            "guyon_validation_mse": float(self.validation_mse),
        } | (
            {}
            if self.calibration_nll is None
            else {
                "guyon_calibration_nll": float(self.calibration_nll),
                "guyon_validation_nll": float(self.validation_nll),
            }
        )


def _ridge_design(features: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps))
    normalized = (features - mean) / std
    ones = torch.ones(int(features.shape[0]), 1, device=features.device, dtype=features.dtype)
    return torch.cat([ones, normalized], dim=1), mean.detach(), std.detach()


def fit_local_gaussian_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    variance_target_path: torch.Tensor | None = None,
    ridge: float = 1e-3,
    variance_ridge: float | None = None,
    variance_shrinkage: float = 0.1,
    log_variance_bias_correction: float = 0.6,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    eps: float = 1e-6,
) -> LocalGaussianReferenceKernel:
    """
    Fit the local Gaussian reference kernel from target paths.

    If ``variance_target_path`` is not provided, the variance regression target
    is built from drift-regression residuals with shrinkage toward the global
    realized variance.  This is the default used by the Heston experiment.
    """
    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (B, N + 1, d)")
    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target path time dimension must align with grid")
    if int(target_paths.shape[0]) < 2:
        raise ValueError("at least two target paths are required")
    ridge = _require_finite_float("ridge", ridge)
    variance_ridge_value = ridge if variance_ridge is None else _require_finite_float("variance_ridge", variance_ridge)
    variance_shrinkage = _require_finite_float("variance_shrinkage", variance_shrinkage)
    log_variance_bias_correction = _require_finite_float(
        "log_variance_bias_correction",
        log_variance_bias_correction,
    )
    sigma_min = _require_finite_float("sigma_min", sigma_min)
    sigma_max = None if sigma_max is None else _require_finite_float("sigma_max", sigma_max)
    eps = _require_finite_float("eps", eps)
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")
    if variance_ridge_value < 0.0:
        raise ValueError("variance_ridge must be >= 0")
    if not (0.0 <= variance_shrinkage <= 1.0):
        raise ValueError("variance_shrinkage must be in [0, 1]")
    if log_variance_bias_correction < 0.0:
        raise ValueError("log_variance_bias_correction must be >= 0")
    if sigma_min <= 0.0:
        raise ValueError("sigma_min must be > 0")
    if sigma_max is not None and sigma_max <= sigma_min:
        raise ValueError("sigma_max must be > sigma_min when provided")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    if variance_target_path is not None:
        expected_shape = (int(target_paths.shape[0]), int(grid.num_steps), int(target_paths.shape[2]))
        if variance_target_path.ndim != 3 or tuple(variance_target_path.shape) != expected_shape:
            raise ValueError("variance_target_path must have shape (B, N, d)")
        variance_target_path = variance_target_path.to(device=target_paths.device, dtype=target_paths.dtype)

    drift_coefficients = []
    log_variance_coefficients = []
    feature_means = []
    feature_stds = []
    dt = float(grid.dt)
    for step in range(int(grid.num_steps)):
        features = target_paths[:, : step + 1, :].reshape(int(target_paths.shape[0]), -1)
        design, mean, std = _ridge_design(features, eps=eps)
        increments = target_paths[:, step + 1, :] - target_paths[:, step, :]
        drift_target = increments / dt
        drift_coeff = fit_ridge(design=design, target=drift_target, ridge=ridge)
        drift_pred = design @ drift_coeff
        if variance_target_path is None:
            residual = increments - drift_pred * dt
            realized_variance = residual.pow(2) / dt
            global_variance = increments.var(dim=0, unbiased=False, keepdim=True).clamp_min(
                sigma_min * sigma_min * dt
            ) / dt
            variance_target = (
                (1.0 - variance_shrinkage) * realized_variance
                + variance_shrinkage * global_variance
            ).clamp_min(sigma_min * sigma_min)
        else:
            variance_target = variance_target_path[:, step, :].clamp_min(sigma_min * sigma_min)
        log_variance_target = torch.log(variance_target) + log_variance_bias_correction
        log_variance_coeff = fit_ridge(
            design=design,
            target=log_variance_target,
            ridge=variance_ridge_value,
        )
        drift_coefficients.append(drift_coeff.detach())
        log_variance_coefficients.append(log_variance_coeff.detach())
        feature_means.append(mean.detach())
        feature_stds.append(std.detach())

    return LocalGaussianReferenceKernel(
        grid=grid,
        drift_coefficients=tuple(drift_coefficients),
        log_variance_coefficients=tuple(log_variance_coefficients),
        feature_means=tuple(feature_means),
        feature_stds=tuple(feature_stds),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def _exponential_factor_histories(
    increments: torch.Tensor,
    *,
    rates: tuple[float, ...],
    dt: float,
    initial: torch.Tensor,
    squared: bool,
) -> dict[float, torch.Tensor]:
    """Build causal factor values at the start of every transition."""
    result: dict[float, torch.Tensor] = {}
    batch_size, num_steps, dimension = map(int, increments.shape)
    innovation = increments.pow(2) if squared else increments
    for rate in rates:
        state = initial.to(device=increments.device, dtype=increments.dtype).expand(
            batch_size,
            dimension,
        )
        decay = math.exp(-float(rate) * float(dt))
        values = []
        for step in range(num_steps):
            values.append(state)
            state = decay * (state + float(rate) * innovation[:, step, :])
        result[float(rate)] = torch.stack(values, dim=1)
    return result


def _fit_standardized_guyon_coefficients(
    *,
    trend: torch.Tensor,
    activity_sqrt: torch.Tensor,
    target_sigma: torch.Tensor,
    ridge: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit beta coefficients and return them in the original feature scale."""
    features = torch.stack([trend, activity_sqrt], dim=1)
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps))
    standardized = (features - feature_mean) / feature_std
    design = torch.cat(
        [
            torch.ones(
                int(features.shape[0]),
                1,
                int(features.shape[2]),
                device=features.device,
                dtype=features.dtype,
            ),
            standardized,
        ],
        dim=1,
    )
    coefficients = []
    for dimension in range(int(features.shape[2])):
        coefficients.append(
            fit_ridge(
                design=design[:, :, dimension],
                target=target_sigma[:, dimension : dimension + 1],
                ridge=float(ridge),
            )
        )
    standardized_coefficients = torch.cat(coefficients, dim=1)
    beta_1 = standardized_coefficients[1:2, :] / feature_std[:, 0, :]
    beta_2 = standardized_coefficients[2:3, :] / feature_std[:, 1, :]
    beta_0 = (
        standardized_coefficients[0:1, :]
        - beta_1 * feature_mean[:, 0, :]
        - beta_2 * feature_mean[:, 1, :]
    )
    for dimension in range(int(features.shape[2])):
        if float(beta_2[0, dimension].item()) >= 0.0:
            continue
        reduced = fit_ridge(
            design=design[:, :2, dimension],
            target=target_sigma[:, dimension : dimension + 1],
            ridge=float(ridge),
        )
        reduced_beta_1 = reduced[1, 0] / feature_std[0, 0, dimension]
        reduced_beta_0 = reduced[0, 0] - reduced_beta_1 * feature_mean[0, 0, dimension]
        beta_0[0, dimension] = reduced_beta_0
        beta_1[0, dimension] = reduced_beta_1
        beta_2[0, dimension] = 0.0
    return beta_0, beta_1, beta_2


def fit_guyon_lekeufack_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    ridge: float = 1e-3,
    variance_ridge: float | None = None,
    variance_shrinkage: float = 0.1,
    log_variance_bias_correction: float = 0.6,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    calibration_fraction: float = 0.8,
    eps: float = 1e-6,
) -> GuyonLekeufackReferenceKernel:
    """Fit a two-exponential Guyon--Lekeufack volatility reference.

    The drift is the established datewise causal ridge reference.  Volatility
    is calibrated from observable return histories only.  A small deterministic
    grid selects fast/slow half-lives and mixture weights on a held-out portion
    of the supplied training paths; the Heston latent variance is never used.
    """
    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (B, N + 1, d)")
    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target path time dimension must align with grid")
    if int(target_paths.shape[0]) < 8:
        raise ValueError("at least eight target paths are required for Guyon calibration")
    ridge = _require_finite_float("ridge", ridge)
    variance_ridge_value = ridge if variance_ridge is None else _require_finite_float(
        "variance_ridge",
        variance_ridge,
    )
    variance_shrinkage = _require_finite_float("variance_shrinkage", variance_shrinkage)
    log_variance_bias_correction = _require_finite_float(
        "log_variance_bias_correction",
        log_variance_bias_correction,
    )
    sigma_min = _require_finite_float("sigma_min", sigma_min)
    sigma_max = None if sigma_max is None else _require_finite_float("sigma_max", sigma_max)
    calibration_fraction = _require_finite_float("calibration_fraction", calibration_fraction)
    eps = _require_finite_float("eps", eps)
    if ridge < 0.0 or variance_ridge_value < 0.0:
        raise ValueError("ridge penalties must be non-negative")
    if not (0.0 <= variance_shrinkage <= 1.0):
        raise ValueError("variance_shrinkage must be in [0, 1]")
    if log_variance_bias_correction < 0.0:
        raise ValueError("log_variance_bias_correction must be non-negative")
    if sigma_min <= 0.0:
        raise ValueError("sigma_min must be positive")
    if sigma_max is not None and sigma_max <= sigma_min:
        raise ValueError("sigma_max must exceed sigma_min")
    if not (0.5 <= calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in [0.5, 1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    drift_kernel = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=ridge,
        variance_ridge=variance_ridge_value,
        variance_shrinkage=variance_shrinkage,
        log_variance_bias_correction=log_variance_bias_correction,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        eps=eps,
    )
    dt = float(grid.dt)
    increments = target_paths[:, 1:, :] - target_paths[:, :-1, :]
    drift_predictions = torch.stack(
        [
            drift_kernel.evaluate(
                target_paths[:, : step + 1, :],
                step_index=step,
            )[0]
            for step in range(int(grid.num_steps))
        ],
        dim=1,
    ).detach()
    residuals = increments - drift_predictions * dt
    realized_variance = residuals.pow(2) / dt
    global_variance = residuals.pow(2).mean(dim=(0, 1), keepdim=True).clamp_min(
        sigma_min * sigma_min * dt
    ) / dt
    variance_target = (
        (1.0 - variance_shrinkage) * realized_variance
        + variance_shrinkage * global_variance
    ).clamp_min(sigma_min * sigma_min)
    target_sigma = torch.sqrt(variance_target) * math.exp(0.5 * log_variance_bias_correction)
    if sigma_max is not None:
        target_sigma = target_sigma.clamp_max(float(sigma_max))

    initial_activity = global_variance.reshape(1, int(target_paths.shape[2])).detach()
    initial_trend = torch.zeros_like(initial_activity)
    maximum_half_life = max(2, int(grid.num_steps) // 2)
    candidate_pairs = tuple(
        pair
        for pair in ((2, 16), (4, 32), (8, 64))
        if pair[0] < pair[1] and pair[0] <= maximum_half_life
    )
    candidate_pairs = tuple(
        (fast, min(slow, maximum_half_life))
        for fast, slow in candidate_pairs
        if fast < min(slow, maximum_half_life)
    )
    if len(candidate_pairs) == 0:
        candidate_pairs = ((1, max(2, int(grid.num_steps) // 2)),)
    unique_half_lives = tuple(sorted({value for pair in candidate_pairs for value in pair}))
    rates_by_half_life = {
        half_life: math.log(2.0) / (float(half_life) * dt)
        for half_life in unique_half_lives
    }
    rates = tuple(rates_by_half_life[value] for value in unique_half_lives)
    trend_histories = _exponential_factor_histories(
        increments,
        rates=rates,
        dt=dt,
        initial=initial_trend,
        squared=False,
    )
    activity_histories = _exponential_factor_histories(
        increments,
        rates=rates,
        dt=dt,
        initial=initial_activity,
        squared=True,
    )

    num_paths = int(target_paths.shape[0])
    split = max(4, min(num_paths - 2, int(round(calibration_fraction * num_paths))))
    train_rows = split * int(grid.num_steps)
    validation_slice = slice(split, num_paths)
    theta_values = (0.25, 0.5, 0.75)
    best: dict[str, object] | None = None
    for trend_pair in candidate_pairs:
        trend_rates = tuple(rates_by_half_life[value] for value in trend_pair)
        trend_components = tuple(trend_histories[float(rate)] for rate in trend_rates)
        for activity_pair in candidate_pairs:
            activity_rates = tuple(rates_by_half_life[value] for value in activity_pair)
            activity_components = tuple(activity_histories[float(rate)] for rate in activity_rates)
            for trend_weight in theta_values:
                trend = (
                    (1.0 - trend_weight) * trend_components[0]
                    + trend_weight * trend_components[1]
                )
                for activity_weight in theta_values:
                    activity = (
                        (1.0 - activity_weight) * activity_components[0]
                        + activity_weight * activity_components[1]
                    ).clamp_min(eps)
                    train_trend = trend[:split].reshape(train_rows, int(target_paths.shape[2]))
                    train_activity_sqrt = torch.sqrt(activity[:split]).reshape(
                        train_rows,
                        int(target_paths.shape[2]),
                    )
                    train_target = target_sigma[:split].reshape(
                        train_rows,
                        int(target_paths.shape[2]),
                    )
                    beta_0, beta_1, beta_2 = _fit_standardized_guyon_coefficients(
                        trend=train_trend,
                        activity_sqrt=train_activity_sqrt,
                        target_sigma=train_target,
                        ridge=variance_ridge_value,
                        eps=eps,
                    )
                    if torch.any(beta_2 < 0.0):
                        continue
                    prediction = (
                        beta_0.view(1, 1, -1)
                        + beta_1.view(1, 1, -1) * trend
                        + beta_2.view(1, 1, -1) * torch.sqrt(activity)
                    ).clamp_min(sigma_min)
                    if sigma_max is not None:
                        prediction = prediction.clamp_max(float(sigma_max))
                    calibration_mse = float(
                        torch.mean((prediction[:split] - target_sigma[:split]).pow(2)).item()
                    )
                    validation_mse = float(
                        torch.mean(
                            (prediction[validation_slice] - target_sigma[validation_slice]).pow(2)
                        ).item()
                    )
                    if best is None or validation_mse < float(best["validation_mse"]):
                        best = {
                            "trend_rates": trend_rates,
                            "activity_rates": activity_rates,
                            "trend_weight": trend_weight,
                            "activity_weight": activity_weight,
                            "beta_0": beta_0.detach(),
                            "beta_1": beta_1.detach(),
                            "beta_2": beta_2.detach(),
                            "calibration_mse": calibration_mse,
                            "validation_mse": validation_mse,
                        }
    if best is None:
        raise RuntimeError("Guyon calibration found no admissible positive-volatility specification")

    selected_trend_components = tuple(
        trend_histories[float(rate)] for rate in best["trend_rates"]
    )
    selected_activity_components = tuple(
        activity_histories[float(rate)] for rate in best["activity_rates"]
    )
    selected_trend = (
        (1.0 - float(best["trend_weight"])) * selected_trend_components[0]
        + float(best["trend_weight"]) * selected_trend_components[1]
    )
    selected_activity = (
        (1.0 - float(best["activity_weight"])) * selected_activity_components[0]
        + float(best["activity_weight"]) * selected_activity_components[1]
    ).clamp_min(eps)
    unscaled_prediction = (
        best["beta_0"].view(1, 1, -1)
        + best["beta_1"].view(1, 1, -1) * selected_trend
        + best["beta_2"].view(1, 1, -1) * torch.sqrt(selected_activity)
    ).clamp_min(sigma_min)
    if sigma_max is not None:
        unscaled_prediction = unscaled_prediction.clamp_max(float(sigma_max))
    likelihood_scale = float(
        torch.sqrt(
            torch.mean(
                residuals[:split].pow(2)
                / (
                    dt
                    * unscaled_prediction[:split].pow(2).clamp_min(eps * eps)
                )
            )
        ).item()
    )
    best["beta_0"] = best["beta_0"] * likelihood_scale
    best["beta_1"] = best["beta_1"] * likelihood_scale
    best["beta_2"] = best["beta_2"] * likelihood_scale
    scaled_prediction = (
        best["beta_0"].view(1, 1, -1)
        + best["beta_1"].view(1, 1, -1) * selected_trend
        + best["beta_2"].view(1, 1, -1) * torch.sqrt(selected_activity)
    ).clamp_min(sigma_min)
    if sigma_max is not None:
        scaled_prediction = scaled_prediction.clamp_max(float(sigma_max))
    best["calibration_mse"] = float(
        torch.mean((scaled_prediction[:split] - target_sigma[:split]).pow(2)).item()
    )
    best["validation_mse"] = float(
        torch.mean(
            (scaled_prediction[validation_slice] - target_sigma[validation_slice]).pow(2)
        ).item()
    )

    return GuyonLekeufackReferenceKernel(
        grid=grid,
        drift_kernel=drift_kernel,
        trend_lambdas=best["trend_rates"],
        activity_lambdas=best["activity_rates"],
        trend_weight=float(best["trend_weight"]),
        activity_weight=float(best["activity_weight"]),
        beta_0=best["beta_0"],
        beta_1=best["beta_1"],
        beta_2=best["beta_2"],
        initial_trend=initial_trend,
        initial_activity=initial_activity,
        calibration_mse=float(best["calibration_mse"]),
        validation_mse=float(best["validation_mse"]),
        likelihood_scale=likelihood_scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(torch.finfo(value.dtype).eps)
    return value + torch.log(-torch.expm1(-value))


def _guyon_likelihood_sigmas(
    *,
    increments: torch.Tensor,
    dt: float,
    trend_rates: tuple[torch.Tensor, torch.Tensor],
    activity_rates: tuple[torch.Tensor, torch.Tensor],
    trend_weight: torch.Tensor,
    activity_weight: torch.Tensor,
    beta_0: torch.Tensor,
    beta_1: torch.Tensor,
    beta_2: torch.Tensor,
    initial_trend: torch.Tensor,
    initial_activity: torch.Tensor,
    activity_update: str,
    sigma_min: float,
    sigma_max: float | None,
) -> torch.Tensor:
    """Evaluate differentiable Guyon volatilities for likelihood fitting."""
    batch_size = int(increments.shape[0])
    dimension = int(increments.shape[2])
    trend_states = [
        initial_trend.expand(batch_size, dimension),
        initial_trend.expand(batch_size, dimension),
    ]
    activity_states = [
        initial_activity.expand(batch_size, dimension),
        initial_activity.expand(batch_size, dimension),
    ]
    sigmas: list[torch.Tensor] = []
    for increment in increments.unbind(dim=1):
        trend = (1.0 - trend_weight) * trend_states[0] + trend_weight * trend_states[1]
        activity = (
            (1.0 - activity_weight) * activity_states[0]
            + activity_weight * activity_states[1]
        )
        sigma = beta_0 + beta_1 * trend + beta_2 * torch.sqrt(activity.clamp_min(1e-12))
        sigma = sigma.clamp_min(float(sigma_min))
        if sigma_max is not None:
            sigma = sigma.clamp_max(float(sigma_max))
        sigmas.append(sigma)
        trend_states = [
            torch.exp(-rate * float(dt)) * (state + rate * increment)
            for state, rate in zip(trend_states, trend_rates)
        ]
        activity_source = (
            increment.pow(2)
            if activity_update == "realized_returns"
            else sigma.pow(2) * float(dt)
        )
        activity_states = [
            torch.exp(-rate * float(dt)) * (state + rate * activity_source)
            for state, rate in zip(activity_states, activity_rates)
        ]
    return torch.stack(sigmas, dim=1)


def fit_guyon_lekeufack_likelihood_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    ridge: float = 1e-3,
    variance_ridge: float | None = None,
    variance_shrinkage: float = 0.1,
    log_variance_bias_correction: float = 0.6,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    calibration_fraction: float = 0.8,
    activity_update: str = "realized_returns",
    optimization_steps: int = 400,
    optimization_batch_size: int = 512,
    learning_rate: float = 2e-2,
    validation_every: int = 20,
    seed: int = 314159,
    eps: float = 1e-6,
) -> GuyonLekeufackReferenceKernel:
    """Fit all Guyon volatility parameters by conditional return likelihood.

    The fitted causal drift is held fixed.  Unlike the proxy-based calibration,
    every volatility parameter is selected under the same Gaussian
    quasi-likelihood used to assess the next-return distribution.
    """
    if activity_update not in {"realized_returns", "structural_variance"}:
        raise ValueError(
            "activity_update must be 'realized_returns' or 'structural_variance'"
        )
    if int(optimization_steps) < 1:
        raise ValueError("optimization_steps must be positive")
    if int(optimization_batch_size) < 1:
        raise ValueError("optimization_batch_size must be positive")
    if int(validation_every) < 1:
        raise ValueError("validation_every must be positive")
    learning_rate = _require_finite_float("learning_rate", learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    initializer = fit_guyon_lekeufack_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=ridge,
        variance_ridge=variance_ridge,
        variance_shrinkage=variance_shrinkage,
        log_variance_bias_correction=log_variance_bias_correction,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        calibration_fraction=calibration_fraction,
        eps=eps,
    )
    dt = float(grid.dt)
    increments = target_paths[:, 1:, :] - target_paths[:, :-1, :]
    drift_predictions = torch.stack(
        [
            initializer.drift_kernel.evaluate(
                target_paths[:, : step + 1, :],
                step_index=step,
            )[0]
            for step in range(int(grid.num_steps))
        ],
        dim=1,
    ).detach()
    residuals = (increments - drift_predictions * dt).detach()
    num_paths = int(target_paths.shape[0])
    split = max(4, min(num_paths - 2, int(round(float(calibration_fraction) * num_paths))))
    dimension = int(target_paths.shape[2])
    global_variance = residuals[:split].pow(2).mean(dim=(0, 1), keepdim=True) / dt
    initial_trend = torch.zeros(
        1,
        dimension,
        device=target_paths.device,
        dtype=target_paths.dtype,
    )
    initial_activity = global_variance.reshape(1, dimension).clamp_min(
        float(sigma_min) * float(sigma_min)
    )

    def tensor(value: float | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, device=target_paths.device, dtype=target_paths.dtype)

    initial_rates = tensor(
        [
            *initializer.trend_lambdas,
            *initializer.activity_lambdas,
        ]
    )
    raw_log_rates = torch.nn.Parameter(torch.log(initial_rates))
    initial_weights = tensor([initializer.trend_weight, initializer.activity_weight]).clamp(
        1e-4,
        1.0 - 1e-4,
    )
    raw_weights = torch.nn.Parameter(torch.logit(initial_weights))
    raw_beta_0 = torch.nn.Parameter(_inverse_softplus(initializer.beta_0.detach()))
    raw_beta_1 = torch.nn.Parameter(initializer.beta_1.detach().clone())
    raw_beta_2 = torch.nn.Parameter(_inverse_softplus(initializer.beta_2.detach()))
    parameters = [raw_log_rates, raw_weights, raw_beta_0, raw_beta_1, raw_beta_2]
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))

    minimum_rate = math.log(2.0) / (4.0 * float(grid.num_steps) * dt)
    maximum_rate = math.log(2.0) / dt
    minimum_log_rate = math.log(minimum_rate)
    maximum_log_rate = math.log(maximum_rate)

    def transformed() -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rates = torch.exp(raw_log_rates)
        weights = torch.sigmoid(raw_weights)
        return (
            (rates[0], rates[1]),
            (rates[2], rates[3]),
            weights[0],
            weights[1],
            torch.nn.functional.softplus(raw_beta_0),
            raw_beta_1,
            torch.nn.functional.softplus(raw_beta_2),
        )

    def nll(path_slice: slice | torch.Tensor) -> torch.Tensor:
        selected_increments = increments[path_slice]
        selected_residuals = residuals[path_slice]
        (
            trend_rates,
            activity_rates,
            trend_weight,
            activity_weight,
            beta_0,
            beta_1,
            beta_2,
        ) = transformed()
        sigmas = _guyon_likelihood_sigmas(
            increments=selected_increments,
            dt=dt,
            trend_rates=trend_rates,
            activity_rates=activity_rates,
            trend_weight=trend_weight,
            activity_weight=activity_weight,
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            initial_trend=initial_trend,
            initial_activity=initial_activity,
            activity_update=activity_update,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        standardized = selected_residuals / (sigmas * math.sqrt(dt)).clamp_min(float(eps))
        return torch.mean(torch.log(sigmas.clamp_min(float(eps))) + 0.5 * standardized.pow(2))

    validation_slice = slice(split, num_paths)
    with torch.no_grad():
        best_validation = float(nll(validation_slice).item())
        best_training = float(nll(slice(0, split)).item())
        best_state = [parameter.detach().clone() for parameter in parameters]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch_size = min(int(optimization_batch_size), split)
    patience = 0
    for step in range(1, int(optimization_steps) + 1):
        indices = torch.randint(
            0,
            split,
            (batch_size,),
            generator=generator,
            device="cpu",
        ).to(device=target_paths.device)
        optimizer.zero_grad(set_to_none=True)
        loss = nll(indices)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Guyon likelihood calibration produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            raw_log_rates.clamp_(minimum_log_rate, maximum_log_rate)
        if step % int(validation_every) == 0 or step == int(optimization_steps):
            with torch.no_grad():
                validation_value = float(nll(validation_slice).item())
                training_value = float(nll(slice(0, split)).item())
            if math.isfinite(validation_value) and validation_value < best_validation - 1e-7:
                best_validation = validation_value
                best_training = training_value
                best_state = [parameter.detach().clone() for parameter in parameters]
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break

    with torch.no_grad():
        for parameter, best_value in zip(parameters, best_state):
            parameter.copy_(best_value)
        (
            trend_rates,
            activity_rates,
            trend_weight,
            activity_weight,
            beta_0,
            beta_1,
            beta_2,
        ) = transformed()

    return GuyonLekeufackReferenceKernel(
        grid=grid,
        drift_kernel=initializer.drift_kernel,
        trend_lambdas=tuple(float(value.item()) for value in trend_rates),
        activity_lambdas=tuple(float(value.item()) for value in activity_rates),
        trend_weight=float(trend_weight.item()),
        activity_weight=float(activity_weight.item()),
        beta_0=beta_0.detach(),
        beta_1=beta_1.detach(),
        beta_2=beta_2.detach(),
        initial_trend=initial_trend.detach(),
        initial_activity=initial_activity.detach(),
        calibration_mse=float(best_training),
        validation_mse=float(best_validation),
        likelihood_scale=1.0,
        activity_update=activity_update,
        calibration_nll=float(best_training),
        validation_nll=float(best_validation),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def _forward_realized_volatility_from_prices(
    increments: torch.Tensor,
    *,
    dt: float,
    window_steps: int,
    eps: float,
) -> torch.Tensor:
    """Return forward realized volatility targets constructed only from prices.

    Entry ``(path, step, dimension)`` is the root-mean-square volatility of the
    next ``window_steps`` log-price increments, starting with the transition at
    ``step``.  Only dates with a complete forward window are returned.
    """
    if increments.ndim != 3:
        raise ValueError("increments must have shape (B, N, d)")
    if int(window_steps) < 1 or int(window_steps) > int(increments.shape[1]):
        raise ValueError("window_steps must be between 1 and the number of transitions")
    dt = _require_finite_float("dt", dt)
    eps = _require_finite_float("eps", eps)
    if dt <= 0.0 or eps <= 0.0:
        raise ValueError("dt and eps must be positive")
    squared = increments.pow(2) / float(dt)
    cumulative = torch.cat(
        [
            torch.zeros(
                int(increments.shape[0]),
                1,
                int(increments.shape[2]),
                device=increments.device,
                dtype=increments.dtype,
            ),
            torch.cumsum(squared, dim=1),
        ],
        dim=1,
    )
    count = int(window_steps)
    forward_variance = (cumulative[:, count:, :] - cumulative[:, :-count, :]) / float(
        count
    )
    return torch.sqrt(forward_variance.clamp_min(float(eps) * float(eps)))


def _guyon_observed_return_features(
    *,
    increments: torch.Tensor,
    dt: float,
    trend_rates: tuple[torch.Tensor, torch.Tensor],
    activity_rates: tuple[torch.Tensor, torch.Tensor],
    trend_weight: torch.Tensor,
    activity_weight: torch.Tensor,
    initial_trend: torch.Tensor,
    initial_activity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate differentiable Guyon factors from observed price returns."""
    batch_size, num_steps, dimension = map(int, increments.shape)
    trend_states = [
        initial_trend.to(device=increments.device, dtype=increments.dtype).expand(
            batch_size,
            dimension,
        )
        for _ in range(2)
    ]
    activity_states = [
        initial_activity.to(device=increments.device, dtype=increments.dtype).expand(
            batch_size,
            dimension,
        )
        for _ in range(2)
    ]
    trend_values = []
    activity_values = []
    for step in range(num_steps):
        trend_values.append(
            (1.0 - trend_weight) * trend_states[0] + trend_weight * trend_states[1]
        )
        activity_values.append(
            (1.0 - activity_weight) * activity_states[0]
            + activity_weight * activity_states[1]
        )
        increment = increments[:, step, :]
        trend_states = [
            torch.exp(-rate * float(dt)) * (state + rate * increment)
            for state, rate in zip(trend_states, trend_rates)
        ]
        activity_states = [
            torch.exp(-rate * float(dt)) * (state + rate * increment.pow(2))
            for state, rate in zip(activity_states, activity_rates)
        ]
    return torch.stack(trend_values, dim=1), torch.stack(activity_values, dim=1)


def fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    realized_volatility_window: int = 20,
    ridge: float = 1e-3,
    variance_ridge: float | None = None,
    variance_shrinkage: float = 0.1,
    log_variance_bias_correction: float = 0.6,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    calibration_fraction: float = 0.8,
    optimization_steps: int = 500,
    optimization_batch_size: int = 256,
    learning_rate: float = 2e-2,
    validation_every: int = 20,
    seed: int = 271828,
    eps: float = 1e-6,
) -> GuyonLekeufackReferenceKernel:
    """Fit the empirical two-exponential volatility regression from prices.

    This mirrors the empirical Guyon--Lekeufack calibration objective: a linear
    rule in a past-return trend factor and the square root of a past-squared-
    return activity factor is fitted by least squares to an observed volatility
    series.  Here that series is constructed from forward realized volatility,
    so the calibration needs log prices only.  The future window is a training
    label; generated volatility remains adapted to the simulated prefix.

    Historical factors use observed squared returns, as in the empirical fit.
    The returned simulation kernel uses structural variance for its activity
    recursion, which is the self-contained path-dependent volatility dynamics.
    """
    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (B, N + 1, d)")
    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target path time dimension must align with grid")
    if int(target_paths.shape[0]) < 8:
        raise ValueError("at least eight target paths are required for Guyon calibration")
    if int(realized_volatility_window) < 1:
        raise ValueError("realized_volatility_window must be positive")
    window_steps = min(int(realized_volatility_window), int(grid.num_steps) - 1)
    if window_steps < 1:
        raise ValueError("the grid must contain at least two transitions")
    if int(optimization_steps) < 1 or int(optimization_batch_size) < 1:
        raise ValueError("optimization_steps and optimization_batch_size must be positive")
    if int(validation_every) < 1:
        raise ValueError("validation_every must be positive")
    learning_rate = _require_finite_float("learning_rate", learning_rate)
    calibration_fraction = _require_finite_float(
        "calibration_fraction",
        calibration_fraction,
    )
    eps = _require_finite_float("eps", eps)
    if learning_rate <= 0.0 or eps <= 0.0:
        raise ValueError("learning_rate and eps must be positive")
    if not (0.5 <= calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in [0.5, 1)")

    initializer = fit_guyon_lekeufack_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=ridge,
        variance_ridge=variance_ridge,
        variance_shrinkage=variance_shrinkage,
        log_variance_bias_correction=log_variance_bias_correction,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        calibration_fraction=calibration_fraction,
        eps=eps,
    )
    dt = float(grid.dt)
    increments = (target_paths[:, 1:, :] - target_paths[:, :-1, :]).detach()
    target_sigma = _forward_realized_volatility_from_prices(
        increments,
        dt=dt,
        window_steps=window_steps,
        eps=eps,
    ).detach()
    valid_steps = int(target_sigma.shape[1])
    num_paths = int(target_paths.shape[0])
    dimension = int(target_paths.shape[2])
    split = max(4, min(num_paths - 2, int(round(calibration_fraction * num_paths))))
    initial_activity = (
        increments[:split].pow(2).mean(dim=(0, 1), keepdim=True) / dt
    ).reshape(1, dimension).clamp_min(float(sigma_min) * float(sigma_min))
    initial_trend = torch.zeros_like(initial_activity)

    def tensor(value: float | torch.Tensor | list[float]) -> torch.Tensor:
        return torch.as_tensor(value, device=target_paths.device, dtype=target_paths.dtype)

    raw_log_rates = torch.nn.Parameter(
        torch.log(
            tensor(
                [
                    *initializer.trend_lambdas,
                    *initializer.activity_lambdas,
                ]
            )
        )
    )
    initial_weights = tensor(
        [initializer.trend_weight, initializer.activity_weight]
    ).clamp(1e-4, 1.0 - 1e-4)
    raw_weights = torch.nn.Parameter(torch.logit(initial_weights))
    beta_0 = torch.nn.Parameter(initializer.beta_0.detach().clone())
    beta_1 = torch.nn.Parameter(initializer.beta_1.detach().clone())
    raw_beta_2 = torch.nn.Parameter(_inverse_softplus(initializer.beta_2.detach()))
    parameters = [raw_log_rates, raw_weights, beta_0, beta_1, raw_beta_2]
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))

    minimum_rate = math.log(2.0) / (4.0 * float(grid.num_steps) * dt)
    maximum_rate = math.log(2.0) / dt
    minimum_log_rate = math.log(minimum_rate)
    maximum_log_rate = math.log(maximum_rate)

    def transformed() -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rates = torch.exp(raw_log_rates)
        weights = torch.sigmoid(raw_weights)
        return (
            (rates[0], rates[1]),
            (rates[2], rates[3]),
            weights[0],
            weights[1],
            torch.nn.functional.softplus(raw_beta_2),
        )

    def mse(path_slice: slice | torch.Tensor) -> torch.Tensor:
        selected_increments = increments[path_slice]
        selected_target = target_sigma[path_slice]
        trend_rates, activity_rates, trend_weight, activity_weight, beta_2 = transformed()
        trend, activity = _guyon_observed_return_features(
            increments=selected_increments,
            dt=dt,
            trend_rates=trend_rates,
            activity_rates=activity_rates,
            trend_weight=trend_weight,
            activity_weight=activity_weight,
            initial_trend=initial_trend,
            initial_activity=initial_activity,
        )
        prediction = beta_0 + beta_1 * trend[:, :valid_steps, :] + beta_2 * torch.sqrt(
            activity[:, :valid_steps, :].clamp_min(float(eps) * float(eps))
        )
        prediction = prediction.clamp_min(float(sigma_min))
        if sigma_max is not None:
            prediction = prediction.clamp_max(float(sigma_max))
        return torch.mean((prediction - selected_target).pow(2))

    validation_slice = slice(split, num_paths)
    with torch.no_grad():
        best_validation = float(mse(validation_slice).item())
        best_training = float(mse(slice(0, split)).item())
        best_state = [parameter.detach().clone() for parameter in parameters]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch_size = min(int(optimization_batch_size), split)
    patience = 0
    for step in range(1, int(optimization_steps) + 1):
        indices = torch.randint(0, split, (batch_size,), generator=generator, device="cpu")
        indices = indices.to(device=target_paths.device)
        optimizer.zero_grad(set_to_none=True)
        loss = mse(indices)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("price-only Guyon calibration produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            raw_log_rates.clamp_(minimum_log_rate, maximum_log_rate)
        if step % int(validation_every) == 0 or step == int(optimization_steps):
            with torch.no_grad():
                validation_value = float(mse(validation_slice).item())
                training_value = float(mse(slice(0, split)).item())
            if math.isfinite(validation_value) and validation_value < best_validation - 1e-10:
                best_validation = validation_value
                best_training = training_value
                best_state = [parameter.detach().clone() for parameter in parameters]
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break

    with torch.no_grad():
        for parameter, best_value in zip(parameters, best_state):
            parameter.copy_(best_value)
        trend_rates, activity_rates, trend_weight, activity_weight, fitted_beta_2 = transformed()

    return GuyonLekeufackReferenceKernel(
        grid=grid,
        drift_kernel=initializer.drift_kernel,
        trend_lambdas=tuple(float(value.item()) for value in trend_rates),
        activity_lambdas=tuple(float(value.item()) for value in activity_rates),
        trend_weight=float(trend_weight.item()),
        activity_weight=float(activity_weight.item()),
        beta_0=beta_0.detach(),
        beta_1=beta_1.detach(),
        beta_2=fitted_beta_2.detach(),
        initial_trend=initial_trend.detach(),
        initial_activity=initial_activity.detach(),
        calibration_mse=float(best_training),
        validation_mse=float(best_validation),
        likelihood_scale=1.0,
        activity_update="structural_variance",
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def guyon_lekeufack_reference_kernel_from_parameters(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    trend_lambdas: tuple[float, float],
    activity_lambdas: tuple[float, float],
    trend_weight: float,
    activity_weight: float,
    beta_0: float | torch.Tensor,
    beta_1: float | torch.Tensor,
    beta_2: float | torch.Tensor,
    ridge: float = 1e-3,
    variance_ridge: float | None = None,
    variance_shrinkage: float = 0.1,
    log_variance_bias_correction: float = 0.6,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    calibration_fraction: float = 0.8,
    initial_trend: torch.Tensor | None = None,
    initial_activity: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> GuyonLekeufackReferenceKernel:
    """Build the reference from externally calibrated Guyon--Lekeufack parameters.

    This is the adapter used for parameters produced by the authors' public
    two-exponential calibration code.  The volatility parameters are not
    refitted here.  Only the matched causal drift and, when omitted, the
    pre-history states are estimated from ``target_paths``.
    """
    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (B, N + 1, d)")
    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target path time dimension must align with grid")
    if int(target_paths.shape[0]) < 8:
        raise ValueError("at least eight target paths are required")
    calibration_fraction = _require_finite_float(
        "calibration_fraction",
        calibration_fraction,
    )
    if not (0.5 <= calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in [0.5, 1)")
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    drift_kernel = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=ridge,
        variance_ridge=variance_ridge,
        variance_shrinkage=variance_shrinkage,
        log_variance_bias_correction=log_variance_bias_correction,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        eps=eps,
    )
    dimension = int(target_paths.shape[2])
    device = target_paths.device
    dtype = target_paths.dtype
    increments = target_paths[:, 1:, :] - target_paths[:, :-1, :]
    drift_predictions = torch.stack(
        [
            drift_kernel.evaluate(
                target_paths[:, : step + 1, :],
                step_index=step,
            )[0]
            for step in range(int(grid.num_steps))
        ],
        dim=1,
    ).detach()
    residuals = increments - drift_predictions * float(grid.dt)
    global_variance = residuals.pow(2).mean(dim=(0, 1), keepdim=True).clamp_min(
        float(sigma_min) * float(sigma_min) * float(grid.dt)
    ) / float(grid.dt)
    if initial_trend is None:
        initial_trend = torch.zeros(1, dimension, device=device, dtype=dtype)
    else:
        initial_trend = initial_trend.to(device=device, dtype=dtype)
    if initial_activity is None:
        initial_activity = global_variance.reshape(1, dimension).detach()
    else:
        initial_activity = initial_activity.to(device=device, dtype=dtype)

    def coefficient(value: float | torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.numel() == 1:
            tensor = tensor.reshape(1, 1).expand(1, dimension).clone()
        if tuple(tensor.shape) != (1, dimension):
            raise ValueError(f"{name} must be scalar or have shape (1, d)")
        return tensor.detach()

    reference = GuyonLekeufackReferenceKernel(
        grid=grid,
        drift_kernel=drift_kernel,
        trend_lambdas=trend_lambdas,
        activity_lambdas=activity_lambdas,
        trend_weight=trend_weight,
        activity_weight=activity_weight,
        beta_0=coefficient(beta_0, name="beta_0"),
        beta_1=coefficient(beta_1, name="beta_1"),
        beta_2=coefficient(beta_2, name="beta_2"),
        initial_trend=initial_trend,
        initial_activity=initial_activity,
        calibration_mse=0.0,
        validation_mse=0.0,
        likelihood_scale=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    _, predicted_sigma = reference.evaluate_all(target_paths)
    realized_variance = residuals.pow(2) / float(grid.dt)
    variance_target = (
        (1.0 - float(variance_shrinkage)) * realized_variance
        + float(variance_shrinkage) * global_variance
    ).clamp_min(float(sigma_min) * float(sigma_min))
    target_sigma = torch.sqrt(variance_target) * math.exp(
        0.5 * float(log_variance_bias_correction)
    )
    if sigma_max is not None:
        target_sigma = target_sigma.clamp_max(float(sigma_max))
    split = max(
        4,
        min(
            int(target_paths.shape[0]) - 2,
            int(round(calibration_fraction * int(target_paths.shape[0]))),
        ),
    )
    calibration_mse = float(
        torch.mean((predicted_sigma[:split] - target_sigma[:split]).pow(2)).item()
    )
    validation_mse = float(
        torch.mean((predicted_sigma[split:] - target_sigma[split:]).pow(2)).item()
    )
    return replace(
        reference,
        calibration_mse=calibration_mse,
        validation_mse=validation_mse,
    )
