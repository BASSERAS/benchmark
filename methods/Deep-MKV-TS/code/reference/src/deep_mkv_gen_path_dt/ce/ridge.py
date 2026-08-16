from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from deep_mkv_gen_path_dt.ce.base import ConditionalExpectationResult


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def fit_ridge(
    *,
    design: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if design.ndim != 2:
        raise ValueError("design must have shape (B, p)")
    if target.ndim != 2:
        raise ValueError("target must have shape (B, m)")
    if int(design.shape[0]) != int(target.shape[0]):
        raise ValueError("design and target must have the same batch size")
    ridge = _require_finite_float("ridge", ridge)
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")
    if ridge == 0.0:
        return torch.linalg.lstsq(design, target).solution

    eye = torch.eye(
        int(design.shape[1]),
        device=design.device,
        dtype=design.dtype,
    )
    lhs = design.transpose(0, 1) @ design + ridge * eye
    rhs = design.transpose(0, 1) @ target
    cholesky, info = torch.linalg.cholesky_ex(lhs)
    if int(torch.max(info).item()) == 0:
        return torch.cholesky_solve(rhs, cholesky)

    penalty = math.sqrt(ridge) * eye
    augmented_design = torch.cat([design, penalty], dim=0)
    augmented_target = torch.cat(
        [
            target,
            torch.zeros(
                int(penalty.shape[0]),
                int(target.shape[1]),
                device=target.device,
                dtype=target.dtype,
            ),
        ],
        dim=0,
    )
    return torch.linalg.lstsq(augmented_design, augmented_target).solution


@dataclass(frozen=True)
class RidgeConditionalExpectation:
    ridge: float = 1e-3
    eps: float = 1e-6

    def __post_init__(self) -> None:
        ridge = _require_finite_float("ridge", self.ridge)
        eps = _require_finite_float("eps", self.eps)
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "eps", eps)

    def fit_predict(
        self,
        *,
        prefixes_train: torch.Tensor,
        targets_train: torch.Tensor,
        prefixes_eval: torch.Tensor | None = None,
    ) -> ConditionalExpectationResult:
        if prefixes_train.ndim != 3:
            raise ValueError("prefixes_train must have shape (B, k + 1, d)")
        if targets_train.ndim != 2:
            raise ValueError("targets_train must have shape (B, m)")
        if int(prefixes_train.shape[0]) != int(targets_train.shape[0]):
            raise ValueError("prefixes_train and targets_train must have the same batch size")
        train_features = prefixes_train.reshape(int(prefixes_train.shape[0]), -1)
        mean = train_features.mean(dim=0, keepdim=True)
        std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(self.eps))
        normalized_train = (train_features - mean) / std
        ones_train = torch.ones(
            int(normalized_train.shape[0]),
            1,
            device=normalized_train.device,
            dtype=normalized_train.dtype,
        )
        design_train = torch.cat([ones_train, normalized_train], dim=1)
        coefficients = fit_ridge(design=design_train, target=targets_train, ridge=float(self.ridge))
        train_prediction = design_train @ coefficients
        if prefixes_eval is None:
            prediction = train_prediction
        else:
            if prefixes_eval.ndim != 3:
                raise ValueError("prefixes_eval must have shape (B_eval, k + 1, d)")
            if tuple(prefixes_eval.shape[1:]) != tuple(prefixes_train.shape[1:]):
                raise ValueError("prefix train/eval shapes must match after the batch dimension")
            eval_features = prefixes_eval.reshape(int(prefixes_eval.shape[0]), -1)
            normalized_eval = (eval_features - mean) / std
            ones_eval = torch.ones(
                int(normalized_eval.shape[0]),
                1,
                device=normalized_eval.device,
                dtype=normalized_eval.dtype,
            )
            design_eval = torch.cat([ones_eval, normalized_eval], dim=1)
            prediction = design_eval @ coefficients
        fit_rms = float(torch.sqrt(torch.mean((train_prediction - targets_train).pow(2))).item())
        return ConditionalExpectationResult(
            prediction=prediction,
            metrics={
                "ce_feature_dim": float(train_features.shape[1]),
                "ce_ridge": float(self.ridge),
                "ce_fit_rms": fit_rms,
            },
        )
