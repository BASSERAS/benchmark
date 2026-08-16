from __future__ import annotations

from dataclasses import dataclass, field
import math
import operator
from typing import Sequence

import torch

from path_dt_experiments.shadowing import (
    ShadowingFeatureConfig,
    select_shadow_paths,
)


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return result


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class DrawdownRiskBudgetConfig:
    """Locked decision rule for a futures drawdown-risk overlay.

    A path is expressed in log-price units.  The portfolio holds a constant
    futures notional from the observed prefix endpoint to the end of the
    forecast horizon.  Its unlevered P&L is the simple price return from that
    endpoint.  Peak-to-trough P&L drawdown is therefore linear in leverage.
    """

    prefix_length: int = 65
    future_horizon: int = 32
    top_k: int = 256
    drawdown_quantile: float = 0.90
    risk_budget: float = 0.01
    safety_factor: float = 1.0
    max_leverage: float = 5.0
    min_leverage: float = 0.0
    feature_config: ShadowingFeatureConfig = field(
        default_factory=ShadowingFeatureConfig
    )
    search_backend: str = "auto"
    exact_batch_size: int = 64


@dataclass(frozen=True)
class DrawdownRiskBudgetResult:
    model_name: str
    config: DrawdownRiskBudgetConfig
    selected_indices: torch.Tensor
    prefix_distances: torch.Tensor
    path_metrics: dict[str, torch.Tensor]
    summary: dict[str, float | str]


@dataclass(frozen=True)
class SafetyFactorCalibration:
    safety_factor: float
    coverage_level: float
    conformal_rank: int
    num_sessions: int
    empirical_validation_violation_rate: float


def _validated_config(config: DrawdownRiskBudgetConfig) -> DrawdownRiskBudgetConfig:
    prefix_length = _integer("prefix_length", config.prefix_length)
    future_horizon = _integer("future_horizon", config.future_horizon)
    top_k = _integer("top_k", config.top_k)
    exact_batch_size = _integer("exact_batch_size", config.exact_batch_size)
    if prefix_length < 2:
        raise ValueError("prefix_length must be at least 2")
    if future_horizon < 1:
        raise ValueError("future_horizon must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if exact_batch_size < 1:
        raise ValueError("exact_batch_size must be positive")
    drawdown_quantile = float(config.drawdown_quantile)
    if not math.isfinite(drawdown_quantile) or not 0.5 < drawdown_quantile < 1.0:
        raise ValueError("drawdown_quantile must lie strictly between 0.5 and 1")
    risk_budget = _positive_float("risk_budget", config.risk_budget)
    safety_factor = _positive_float("safety_factor", config.safety_factor)
    max_leverage = _positive_float("max_leverage", config.max_leverage)
    min_leverage = float(config.min_leverage)
    if not math.isfinite(min_leverage) or min_leverage < 0.0:
        raise ValueError("min_leverage must be nonnegative and finite")
    if min_leverage > max_leverage:
        raise ValueError("min_leverage cannot exceed max_leverage")
    search_backend = str(config.search_backend).lower()
    if search_backend not in {"auto", "exact", "faiss"}:
        raise ValueError("search_backend must be auto, exact, or faiss")
    return DrawdownRiskBudgetConfig(
        prefix_length=prefix_length,
        future_horizon=future_horizon,
        top_k=top_k,
        drawdown_quantile=drawdown_quantile,
        risk_budget=risk_budget,
        safety_factor=safety_factor,
        max_leverage=max_leverage,
        min_leverage=min_leverage,
        feature_config=config.feature_config,
        search_backend=search_backend,
        exact_batch_size=exact_batch_size,
    )


def _validate_log_paths(paths: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(paths, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if paths.ndim != 3 or int(paths.shape[2]) != 1:
        raise ValueError(f"{name} must have shape (paths, points, 1)")
    if int(paths.shape[0]) < 1 or int(paths.shape[1]) < 2:
        raise ValueError(f"{name} must contain nonempty paths")
    values = paths.to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")
    return values


def futures_pnl_max_drawdown(
    log_paths: torch.Tensor,
    *,
    start_index: int,
    future_horizon: int,
) -> torch.Tensor:
    """Peak-to-trough futures P&L drawdown over a continuation window.

    The result is an unlevered fraction of capital.  The initial P&L is zero,
    so a path that falls one percent from its decision-time price has a P&L
    drawdown of at least 0.01.  Extra leading dimensions, such as a scenario
    axis, are supported; time must be the penultimate dimension.
    """

    if not isinstance(log_paths, torch.Tensor) or log_paths.ndim < 2:
        raise ValueError("log_paths must have at least a time and asset dimension")
    if int(log_paths.shape[-1]) != 1:
        raise ValueError("the risk-budgeting diagnostic currently supports one asset")
    values = log_paths.to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("log_paths must be finite")
    start = _integer("start_index", start_index)
    horizon = _integer("future_horizon", future_horizon)
    if start < 0 or horizon < 1:
        raise ValueError("start_index must be nonnegative and future_horizon positive")
    stop = start + horizon + 1
    if stop > int(values.shape[-2]):
        raise ValueError("continuation window exceeds the path length")
    window = values[..., start:stop, 0]
    relative_pnl = torch.exp(window - window[..., :1]) - 1.0
    running_peak = torch.cummax(relative_pnl, dim=-1).values
    return (running_peak - relative_pnl).amax(dim=-1)


def _terminal_simple_return(
    log_paths: torch.Tensor,
    *,
    start_index: int,
    future_horizon: int,
) -> torch.Tensor:
    end = int(start_index) + int(future_horizon)
    return torch.exp(log_paths[..., end, 0] - log_paths[..., int(start_index), 0]) - 1.0


def drawdown_policy_metrics(
    *,
    predicted_drawdown_quantile: torch.Tensor,
    realized_unlevered_drawdown: torch.Tensor,
    realized_unlevered_terminal_return: torch.Tensor,
    risk_budget: float,
    max_leverage: float,
    min_leverage: float = 0.0,
    safety_factor: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Apply the risk rule after scenario matching has been completed."""

    budget = _positive_float("risk_budget", risk_budget)
    conservatism = _positive_float("safety_factor", safety_factor)
    maximum = _positive_float("max_leverage", max_leverage)
    minimum = float(min_leverage)
    if not math.isfinite(minimum) or minimum < 0.0 or minimum > maximum:
        raise ValueError("min_leverage must lie in [0, max_leverage]")
    predicted = predicted_drawdown_quantile.to(dtype=torch.float64).reshape(-1)
    realized = realized_unlevered_drawdown.to(dtype=torch.float64).reshape(-1)
    terminal = realized_unlevered_terminal_return.to(dtype=torch.float64).reshape(-1)
    if not (int(predicted.numel()) == int(realized.numel()) == int(terminal.numel())):
        raise ValueError("policy inputs must have matching path counts")
    if int(predicted.numel()) < 1:
        raise ValueError("policy inputs must be nonempty")
    if not bool(torch.isfinite(torch.cat((predicted, realized, terminal))).all()):
        raise ValueError("policy inputs must be finite")
    if bool((predicted < 0.0).any()) or bool((realized < 0.0).any()):
        raise ValueError("drawdowns must be nonnegative")
    unconstrained = budget / (conservatism * predicted.clamp_min(float(eps)))
    leverage = unconstrained.clamp(min=minimum, max=maximum)
    predicted_leveraged = leverage * predicted
    realized_leveraged = leverage * realized
    violation = (realized_leveraged > budget).to(dtype=torch.float64)
    return {
        "predicted_unlevered_drawdown_quantile": predicted,
        "realized_unlevered_drawdown": realized,
        "realized_unlevered_terminal_return": terminal,
        "leverage": leverage,
        "leverage_at_upper_cap": torch.isclose(
            leverage,
            torch.full_like(leverage, maximum),
            rtol=0.0,
            atol=1e-12,
        ).to(dtype=torch.float64),
        "predicted_leveraged_drawdown": predicted_leveraged,
        "realized_leveraged_drawdown": realized_leveraged,
        "risk_utilization": realized_leveraged / budget,
        "violation": violation,
        "violation_excess": torch.clamp(realized_leveraged - budget, min=0.0),
        "realized_terminal_pnl": leverage * terminal,
    }


def calibrate_drawdown_safety_factor(
    *,
    predicted_drawdown_quantile: torch.Tensor,
    realized_unlevered_drawdown: torch.Tensor,
    coverage_level: float = 0.90,
    eps: float = 1e-12,
) -> SafetyFactorCalibration:
    """Split-conformal calibration of one multiplicative risk factor.

    The nonconformity score is realized drawdown divided by the scenario-law
    drawdown quantile.  The finite-sample conformal order statistic is used,
    then frozen before test evaluation.
    """

    coverage = float(coverage_level)
    if not math.isfinite(coverage) or not 0.5 < coverage < 1.0:
        raise ValueError("coverage_level must lie strictly between 0.5 and 1")
    predicted = predicted_drawdown_quantile.detach().cpu().to(dtype=torch.float64).reshape(-1)
    realized = realized_unlevered_drawdown.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if int(predicted.numel()) != int(realized.numel()) or int(predicted.numel()) < 1:
        raise ValueError("calibration inputs must have the same positive length")
    if not bool(torch.isfinite(torch.cat((predicted, realized))).all()):
        raise ValueError("calibration inputs must be finite")
    if bool((predicted < 0.0).any()) or bool((realized < 0.0).any()):
        raise ValueError("calibration drawdowns must be nonnegative")
    ratios = torch.where(
        predicted > float(eps),
        realized / predicted.clamp_min(float(eps)),
        torch.where(
            realized <= float(eps),
            torch.zeros_like(realized),
            torch.full_like(realized, torch.inf),
        ),
    )
    count = int(ratios.numel())
    rank = min(count, int(math.ceil((count + 1) * coverage)))
    factor = float(torch.kthvalue(ratios, rank).values.item())
    if not math.isfinite(factor) or factor <= 0.0:
        raise RuntimeError("validation cannot produce a finite positive safety factor")
    violations = realized > factor * predicted
    return SafetyFactorCalibration(
        safety_factor=factor,
        coverage_level=coverage,
        conformal_rank=rank,
        num_sessions=count,
        empirical_validation_violation_rate=float(
            violations.to(dtype=torch.float64).mean().item()
        ),
    )


def summarize_drawdown_policy(
    path_metrics: dict[str, torch.Tensor],
    *,
    drawdown_quantile: float,
    risk_budget: float,
    max_leverage: float,
    safety_factor: float = 1.0,
) -> dict[str, float]:
    required = {
        "leverage",
        "leverage_at_upper_cap",
        "predicted_leveraged_drawdown",
        "realized_leveraged_drawdown",
        "risk_utilization",
        "violation",
        "violation_excess",
        "realized_terminal_pnl",
    }
    missing = required - set(path_metrics)
    if missing:
        raise ValueError(f"path_metrics is missing {sorted(missing)}")
    values = {
        name: tensor.detach().cpu().to(dtype=torch.float64).reshape(-1)
        for name, tensor in path_metrics.items()
    }
    counts = {int(tensor.numel()) for tensor in values.values()}
    if len(counts) != 1 or next(iter(counts)) < 1:
        raise ValueError("all path metrics must have the same positive length")
    violation = values["violation"]
    excess = values["violation_excess"]
    violating_excess = excess[violation > 0.5]
    target_rate = 1.0 - float(drawdown_quantile)
    observed_rate = float(violation.mean().item())
    return {
        "num_sessions": float(next(iter(counts))),
        "drawdown_quantile": float(drawdown_quantile),
        "target_violation_rate": target_rate,
        "risk_budget": float(risk_budget),
        "safety_factor": float(safety_factor),
        "max_leverage": float(max_leverage),
        "observed_violation_rate": observed_rate,
        "violation_rate_gap": observed_rate - target_rate,
        "average_leverage": float(values["leverage"].mean().item()),
        "median_leverage": float(torch.quantile(values["leverage"], 0.5).item()),
        "leverage_q10": float(torch.quantile(values["leverage"], 0.1).item()),
        "leverage_q90": float(torch.quantile(values["leverage"], 0.9).item()),
        "upper_leverage_cap_fraction": float(
            values["leverage_at_upper_cap"].mean().item()
        ),
        "average_predicted_leveraged_drawdown": float(
            values["predicted_leveraged_drawdown"].mean().item()
        ),
        "average_realized_leveraged_drawdown": float(
            values["realized_leveraged_drawdown"].mean().item()
        ),
        "realized_leveraged_drawdown_q90": float(
            torch.quantile(values["realized_leveraged_drawdown"], 0.9).item()
        ),
        "average_risk_utilization": float(values["risk_utilization"].mean().item()),
        "mean_violation_excess_unconditional": float(excess.mean().item()),
        "mean_violation_excess_conditional": (
            0.0
            if int(violating_excess.numel()) == 0
            else float(violating_excess.mean().item())
        ),
        "average_realized_terminal_pnl": float(
            values["realized_terminal_pnl"].mean().item()
        ),
        "realized_terminal_pnl_q10": float(
            torch.quantile(values["realized_terminal_pnl"], 0.1).item()
        ),
    }


def evaluate_drawdown_risk_budget(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    model_name: str,
    config: DrawdownRiskBudgetConfig | None = None,
) -> DrawdownRiskBudgetResult:
    """Evaluate one unconditional scenario bank as a conditional risk engine."""

    cfg = _validated_config(config or DrawdownRiskBudgetConfig())
    real = _validate_log_paths(real_paths, name="real_paths")
    generated = _validate_log_paths(generated_paths, name="generated_paths").to(
        device=real.device
    )
    if tuple(real.shape[1:]) != tuple(generated.shape[1:]):
        raise ValueError("real and generated paths must have identical point dimensions")
    if cfg.prefix_length + cfg.future_horizon > int(real.shape[1]):
        raise ValueError("prefix plus future horizon exceeds the path length")
    if cfg.top_k > int(generated.shape[0]):
        raise ValueError("top_k cannot exceed the generated bank size")

    selected_indices, prefix_distances = select_shadow_paths(
        real_paths=real,
        generated_paths=generated,
        prefix_length=cfg.prefix_length,
        top_k=cfg.top_k,
        feature_config=cfg.feature_config,
        search_backend=cfg.search_backend,
        exact_batch_size=cfg.exact_batch_size,
    )
    selected = generated[selected_indices.to(device=generated.device)]
    start = cfg.prefix_length - 1
    scenario_drawdowns = futures_pnl_max_drawdown(
        selected,
        start_index=start,
        future_horizon=cfg.future_horizon,
    )
    predicted_quantile = torch.quantile(
        scenario_drawdowns,
        cfg.drawdown_quantile,
        dim=1,
    )
    realized_drawdown = futures_pnl_max_drawdown(
        real,
        start_index=start,
        future_horizon=cfg.future_horizon,
    )
    realized_terminal = _terminal_simple_return(
        real,
        start_index=start,
        future_horizon=cfg.future_horizon,
    )
    path_metrics = drawdown_policy_metrics(
        predicted_drawdown_quantile=predicted_quantile,
        realized_unlevered_drawdown=realized_drawdown,
        realized_unlevered_terminal_return=realized_terminal,
        risk_budget=cfg.risk_budget,
        max_leverage=cfg.max_leverage,
        min_leverage=cfg.min_leverage,
        safety_factor=cfg.safety_factor,
    )
    path_metrics["prefix_distance_mean"] = prefix_distances.to(
        dtype=torch.float64
    ).mean(dim=1)
    summary: dict[str, float | str] = {
        "model_name": str(model_name),
        "prefix_distance_mean": float(prefix_distances.mean().item()),
        "prefix_distance_q95": float(
            torch.quantile(prefix_distances.reshape(-1), 0.95).item()
        ),
    }
    summary.update(
        summarize_drawdown_policy(
            path_metrics,
            drawdown_quantile=cfg.drawdown_quantile,
            risk_budget=cfg.risk_budget,
            max_leverage=cfg.max_leverage,
            safety_factor=cfg.safety_factor,
        )
    )
    return DrawdownRiskBudgetResult(
        model_name=str(model_name),
        config=cfg,
        selected_indices=selected_indices.detach().cpu(),
        prefix_distances=prefix_distances.detach().cpu(),
        path_metrics={
            name: values.detach().cpu() for name, values in path_metrics.items()
        },
        summary=summary,
    )


def bootstrap_drawdown_risk_budget(
    result: DrawdownRiskBudgetResult,
    *,
    num_replicates: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    block_length_sessions: int = 1,
) -> dict[str, dict[str, float]]:
    """Bootstrap mean path-level policy metrics over held-out sessions."""

    replicates = _integer("num_replicates", num_replicates)
    if replicates < 1:
        raise ValueError("num_replicates must be positive")
    level = float(confidence_level)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    count = int(next(iter(result.path_metrics.values())).numel())
    indices = _session_bootstrap_indices(
        count=count,
        num_replicates=replicates,
        block_length_sessions=block_length_sessions,
        seed=seed,
    )
    alpha = 0.5 * (1.0 - level)
    output = {}
    for name, raw in sorted(result.path_metrics.items()):
        values = raw.detach().cpu().to(dtype=torch.float64).reshape(-1)
        means = values[indices].mean(dim=1)
        output[name] = {
            "estimate": float(values.mean().item()),
            "standard_error": float(means.std(unbiased=False).item()),
            "ci_lower": float(torch.quantile(means, alpha).item()),
            "ci_upper": float(torch.quantile(means, 1.0 - alpha).item()),
            "confidence_level": level,
            "num_sessions": float(count),
            "num_replicates": float(replicates),
            "block_length_sessions": float(
                min(_integer("block_length_sessions", block_length_sessions), count)
            ),
        }
    return output


def paired_bootstrap_drawdown_difference(
    primary: DrawdownRiskBudgetResult,
    comparison: DrawdownRiskBudgetResult,
    *,
    metric_names: Sequence[str] = (
        "violation",
        "violation_excess",
        "leverage",
        "realized_leveraged_drawdown",
        "realized_terminal_pnl",
    ),
    num_replicates: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    block_length_sessions: int = 1,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of primary-minus-comparison economic outcomes."""

    names = tuple(str(name) for name in metric_names)
    if not names:
        raise ValueError("metric_names must be nonempty")
    missing = [
        name
        for name in names
        if name not in primary.path_metrics or name not in comparison.path_metrics
    ]
    if missing:
        raise ValueError(f"paired results are missing {sorted(set(missing))}")
    counts = {
        int(primary.path_metrics[name].numel()) for name in names
    } | {int(comparison.path_metrics[name].numel()) for name in names}
    if len(counts) != 1:
        raise ValueError("paired results must contain the same held-out sessions")
    count = next(iter(counts))
    replicates = _integer("num_replicates", num_replicates)
    if replicates < 1:
        raise ValueError("num_replicates must be positive")
    level = float(confidence_level)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    indices = _session_bootstrap_indices(
        count=count,
        num_replicates=replicates,
        block_length_sessions=block_length_sessions,
        seed=seed,
    )
    alpha = 0.5 * (1.0 - level)
    output = {}
    for name in names:
        difference = (
            primary.path_metrics[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
            - comparison.path_metrics[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
        )
        means = difference[indices].mean(dim=1)
        output[name] = {
            "estimate": float(difference.mean().item()),
            "standard_error": float(means.std(unbiased=False).item()),
            "ci_lower": float(torch.quantile(means, alpha).item()),
            "ci_upper": float(torch.quantile(means, 1.0 - alpha).item()),
            "confidence_level": level,
            "num_sessions": float(count),
            "num_replicates": float(replicates),
            "block_length_sessions": float(
                min(_integer("block_length_sessions", block_length_sessions), count)
            ),
            "orientation": "primary_minus_comparison",
        }
    return output


def _session_bootstrap_indices(
    *,
    count: int,
    num_replicates: int,
    block_length_sessions: int,
    seed: int,
) -> torch.Tensor:
    """Circular moving-block bootstrap indices for chronological sessions."""

    count = _integer("count", count)
    replicates = _integer("num_replicates", num_replicates)
    block = _integer("block_length_sessions", block_length_sessions)
    if count < 1 or replicates < 1 or block < 1:
        raise ValueError("bootstrap dimensions must be positive")
    block = min(block, count)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_integer("seed", seed))
    if block == 1:
        return torch.randint(
            0,
            count,
            (replicates, count),
            generator=generator,
        )
    blocks_per_replicate = int(math.ceil(count / block))
    starts = torch.randint(
        0,
        count,
        (replicates, blocks_per_replicate),
        generator=generator,
    )
    offsets = torch.arange(block, dtype=torch.long)
    indices = (starts[:, :, None] + offsets[None, None, :]) % count
    return indices.reshape(replicates, -1)[:, :count]
