from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


@dataclass(frozen=True)
class PathFunctionalDiscrepancyResult:
    value: torch.Tensor
    metrics: dict[str, float]


class PathFunctionalDiscrepancy(Protocol):
    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        ...

