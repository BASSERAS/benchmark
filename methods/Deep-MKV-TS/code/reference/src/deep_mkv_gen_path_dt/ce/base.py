from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ConditionalExpectationResult:
    prediction: torch.Tensor
    metrics: dict[str, float]


class ConditionalExpectationEstimator(Protocol):
    def fit_predict(
        self,
        *,
        prefixes_train: torch.Tensor,
        targets_train: torch.Tensor,
        prefixes_eval: torch.Tensor | None = None,
    ) -> ConditionalExpectationResult:
        """
        Estimate E[target | prefix].

        `prefixes_train` has shape `(B, k + 1, d)`, `targets_train` has shape
        `(B, m)`, and `prefixes_eval` defaults to `prefixes_train`.
        """

        ...

