from __future__ import annotations

from dataclasses import dataclass

import torch

from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import PolicyStepResult


@dataclass(frozen=True)
class VolatilityOnlyPDiagnosticReferencePolicy:
    """Reference-volatility law with a regressed passive P diagnostic."""

    model: DiscreteMPModel
    p_network: torch.nn.Module

    def initial_state(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> object:
        del batch_size, device, dtype
        return None

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult:
        normalized_p, _, next_state = self.p_network.forward_step(
            x_t=x_prefix[:, -1, :], hidden=state
        )
        p = self.model._denormalize_adjoint_step(
            normalized_p, step_index=int(step_index), noise_adjoint=False
        )
        r = torch.zeros_like(p)
        control = self.model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p,
                expected_adjoint_noise_next=r,
                parameters=self.model.parameters,
            )
        )
        return PolicyStepResult(
            control=control,
            state=next_state,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


@dataclass(frozen=True)
class VolatilityOnlyBlockwiseRPolicy:
    """Deploy blockwise fractions of one causal ``R`` policy with passive ``P``.

    The candidate policy supplies both costates, but only ``R`` is interpolated
    from the exact zero-adjoint reference.  The volatility-only control map
    ignores ``P`` when generating the forward law.
    """

    model: DiscreteMPModel
    candidate_policy: object
    theta_r: torch.Tensor
    blocks: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if tuple(self.theta_r.shape) != (len(self.blocks),):
            raise ValueError("theta_r must contain one coordinate per time block")
        if not self.blocks or self.blocks[0][0] != 0 or self.blocks[-1][1] != int(
            self.model.grid.num_steps
        ):
            raise ValueError("blocks must cover the complete decision grid")
        if any(right <= left for left, right in self.blocks):
            raise ValueError("time blocks must be nonempty")

    def initial_state(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> object:
        return self.candidate_policy.initial_state(
            batch_size=batch_size, device=device, dtype=dtype
        )

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult:
        candidate = self.candidate_policy.control_step(
            x_prefix=x_prefix,
            step_index=int(step_index),
            state=state,
        )
        if (
            candidate.expected_adjoint_next is None
            or candidate.expected_adjoint_noise_next is None
        ):
            raise TypeError("candidate policy must expose P and R")
        block = next(
            index
            for index, (left, right) in enumerate(self.blocks)
            if left <= int(step_index) < right
        )
        fraction = self.theta_r[block].to(candidate.expected_adjoint_noise_next)
        r = fraction * candidate.expected_adjoint_noise_next
        p = candidate.expected_adjoint_next
        control = self.model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p,
                expected_adjoint_noise_next=r,
                parameters=self.model.parameters,
            )
        )
        return PolicyStepResult(
            control=control,
            state=candidate.state,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


__all__ = [
    "VolatilityOnlyBlockwiseRPolicy",
    "VolatilityOnlyPDiagnosticReferencePolicy",
]
