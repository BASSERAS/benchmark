from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Sequence

import torch


def _normalize_observed_indices(num_steps: int, observed_indices: Sequence[int] | None) -> tuple[int, ...]:
    if observed_indices is None:
        return tuple(range(int(num_steps) + 1))
    values = tuple(_require_int("observed_indices", index) for index in observed_indices)
    if len(values) == 0:
        raise ValueError("observed_indices must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError("observed_indices must be unique")
    if values != tuple(sorted(values)):
        raise ValueError("observed_indices must be sorted")
    if values[0] != 0:
        raise ValueError("observed_indices must include 0")
    if values[-1] != int(num_steps):
        raise ValueError("observed_indices must include num_steps")
    if any(index < 0 or index > int(num_steps) for index in values):
        raise ValueError("observed_indices must lie in [0, num_steps]")
    return values


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _is_uniform(values: Sequence[float], *, T: float, num_steps: int) -> bool:
    dt = float(T) / float(num_steps)
    tolerance = max(1e-12, 1e-6 * max(1.0, abs(float(T))))
    return all(abs(float(value) - dt * index) <= tolerance for index, value in enumerate(values))


def _normalize_point_times(
    T: float,
    num_steps: int,
    point_times: Sequence[float] | None,
    *,
    allow_nonuniform: bool,
) -> tuple[float, ...] | None:
    if point_times is None:
        return None
    values = tuple(_require_finite_float("point_times", value) for value in point_times)
    if len(values) != int(num_steps) + 1:
        raise ValueError("point_times must have length num_steps + 1")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("point_times must be finite")
    if abs(values[0]) > 1e-12:
        raise ValueError("point_times must start at 0")
    if abs(values[-1] - float(T)) > 1e-12:
        raise ValueError("point_times must end at T")
    if any(right <= left for left, right in zip(values[:-1], values[1:])):
        raise ValueError("point_times must be strictly increasing")
    if not allow_nonuniform and not _is_uniform(values, T=float(T), num_steps=int(num_steps)):
        raise ValueError("nonuniform point_times require allow_nonuniform=True")
    return values


@dataclass(frozen=True)
class DiscreteTimeGrid:
    T: float
    num_steps: int
    observed_indices: Sequence[int] | None = None
    point_times: Sequence[float] | None = None
    allow_nonuniform: bool = False

    def __post_init__(self) -> None:
        T = _require_finite_float("T", self.T)
        num_steps = _require_int("num_steps", self.num_steps)
        allow_nonuniform = _require_bool("allow_nonuniform", self.allow_nonuniform)
        if T <= 0.0:
            raise ValueError("T must be > 0")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "allow_nonuniform", allow_nonuniform)
        object.__setattr__(self, "observed_indices", _normalize_observed_indices(num_steps, self.observed_indices))
        object.__setattr__(
            self,
            "point_times",
            _normalize_point_times(T, num_steps, self.point_times, allow_nonuniform=allow_nonuniform),
        )

    @property
    def dt(self) -> float:
        if self.point_times is not None and not self.is_uniform:
            raise ValueError("dt is undefined for nonuniform point_times")
        return float(self.T) / float(self.num_steps)

    @property
    def is_uniform(self) -> bool:
        if self.point_times is None:
            return True
        return _is_uniform(tuple(self.point_times), T=float(self.T), num_steps=int(self.num_steps))

    @property
    def time_points_count(self) -> int:
        return int(self.num_steps) + 1

    @property
    def observed_count(self) -> int:
        return len(tuple(self.observed_indices))

    def validate_point_index(self, point_index: int) -> None:
        index = _require_int("point_index", point_index)
        if not (0 <= index <= int(self.num_steps)):
            raise ValueError(f"point_index must be in [0, {self.num_steps}]")

    def validate_step_index(self, step_index: int) -> None:
        index = _require_int("step_index", step_index)
        if not (0 <= index < int(self.num_steps)):
            raise ValueError(f"step_index must be in [0, {self.num_steps - 1}]")

    def time_points(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.point_times is not None:
            return torch.tensor(
                tuple(self.point_times),
                device=device,
                dtype=dtype if dtype is not None else torch.float32,
            )
        return torch.linspace(
            0.0,
            float(self.T),
            int(self.num_steps) + 1,
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )

    def point_indices(self, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.arange(0, int(self.num_steps) + 1, device=device, dtype=torch.long)

    def observed_point_indices(self, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(tuple(self.observed_indices), device=device, dtype=torch.long)

    def observed_mask(self, *, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(int(self.num_steps) + 1, device=device, dtype=torch.bool)
        mask[self.observed_point_indices(device=device)] = True
        return mask

    def observation_position_for_point(self, point_index: int) -> int:
        point_index = _require_int("point_index", point_index)
        self.validate_point_index(point_index)
        for position, observed_index in enumerate(tuple(self.observed_indices)):
            if int(observed_index) == int(point_index):
                return int(position)
        raise ValueError("point_index is not an observed date")

    def eta_observation_position(self, step_index: int) -> int:
        step_index = _require_int("step_index", step_index)
        self.validate_step_index(step_index)
        position = 0
        for candidate_position, observed_index in enumerate(tuple(self.observed_indices)):
            if int(observed_index) <= int(step_index):
                position = int(candidate_position)
            else:
                break
        return position

    def eta_point_index(self, step_index: int) -> int:
        return int(tuple(self.observed_indices)[self.eta_observation_position(_require_int("step_index", step_index))])
