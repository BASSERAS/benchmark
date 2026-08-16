from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


@dataclass(frozen=True)
class HamiltonianControlInputs:
    step_index: int
    x_prefix: torch.Tensor
    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor
    parameters: object | None = None


class HamiltonianControlMap(Protocol):
    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class RunningCostInputs:
    """Inputs used to evaluate the running part of the control objective.

    ``controls`` are interpreted as fixed realized controls by default when
    the cost is differentiated with respect to ``paths``.  This is the
    partial state/path derivative appearing in the maximum-principle adjoint
    equation, rather than a derivative through the policy that produced the
    controls.  A matched direct-objective diagnostic can explicitly set
    ``differentiate_controls=True`` to recover the total objective gradient.

    A running-cost callable must return the empirical mean of the actual
    discrete stage costs.  If a displayed Hamiltonian integrand has been
    normalized by the time step, the callable is responsible for restoring
    that time-step factor.
    """

    paths: torch.Tensor
    controls: torch.Tensor
    grid: DiscreteTimeGrid
    parameters: object | None = None
    differentiate_controls: bool = False


@dataclass(frozen=True)
class RunningCostResult:
    value: torch.Tensor
    metrics: dict[str, float]


class PathDependentRunningCost(Protocol):
    def __call__(self, inputs: RunningCostInputs) -> RunningCostResult:
        ...


@dataclass(frozen=True)
class DynamicsStepInputs:
    step_index: int
    x_prefix: torch.Tensor
    control: torch.Tensor
    noise_next: torch.Tensor
    parameters: object | None = None


class DynamicsStepMap(Protocol):
    def __call__(self, inputs: DynamicsStepInputs) -> torch.Tensor:
        ...
