from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch
from torch import nn

from path_dt_experiments.metrics import path_returns


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_positive_int(name: str, value: object) -> int:
    result = _require_int(name, value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_paths(paths: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(paths, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if paths.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N + 1, d)")
    if int(paths.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one path")
    if int(paths.shape[1]) < 2:
        raise ValueError(f"{name} must contain at least two time points")
    if int(paths.shape[2]) < 1:
        raise ValueError(f"{name} must contain at least one state dimension")
    if not torch.is_floating_point(paths):
        paths = paths.float()
    return paths


def _check_same_path_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if int(left.shape[1]) != int(right.shape[1]):
        raise ValueError("path time dimensions must match")
    if int(left.shape[2]) != int(right.shape[2]):
        raise ValueError("path state dimensions must match")


def _seeded_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(_require_int("seed", seed))
    return generator


def _device_seed(device: torch.device, seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(_require_int("seed", seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_require_int("seed", seed))


def _sample_indices(count: int, batch_size: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    indices = torch.randint(count, (batch_size,), generator=generator, device="cpu")
    return indices.to(device=device)


def _shuffle_indices(count: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(count, generator=generator, device="cpu")


def _train_eval_indices(count: int, *, train_fraction: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 1:
        indices = torch.zeros(1, dtype=torch.long)
        return indices, indices
    order = _shuffle_indices(count, generator)
    train_count = int(round(float(count) * float(train_fraction)))
    train_count = min(max(1, train_count), count - 1)
    return order[:train_count], order[train_count:]


def _normalizer(values: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=(0, 1), keepdim=True)
    std = values.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(eps)
    return mean, std


class _PathClassifier(nn.Module):
    def __init__(self, *, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(paths)
        return self.head(hidden[-1]).squeeze(-1)


class _NextReturnPredictor(nn.Module):
    def __init__(self, *, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(self, prefixes: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(prefixes)
        return self.head(hidden)


@dataclass(frozen=True)
class BenchmarkScoreConfig:
    """Training budget for learned experiment scores."""

    train_steps: int = 300
    batch_size: int = 128
    hidden_dim: int = 32
    lr: float = 1e-3
    train_fraction: float = 0.7
    seed: int = 1234

    def validate(self) -> "BenchmarkScoreConfig":
        train_steps = _require_positive_int("train_steps", self.train_steps)
        batch_size = _require_positive_int("batch_size", self.batch_size)
        hidden_dim = _require_positive_int("hidden_dim", self.hidden_dim)
        lr = _require_finite_float("lr", self.lr)
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        train_fraction = _require_finite_float("train_fraction", self.train_fraction)
        if train_fraction <= 0.0 or train_fraction >= 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        seed = _require_int("seed", self.seed)
        return BenchmarkScoreConfig(
            train_steps=train_steps,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            lr=lr,
            train_fraction=train_fraction,
            seed=seed,
        )


@torch.no_grad()
def empirical_crps(
    forecast_samples: torch.Tensor,
    observations: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Empirical CRPS for an unconditional ensemble forecast.

    ``forecast_samples`` and ``observations`` must have shape ``(B, N, d)``.
    The score is averaged over time and assets.  Lower is better.
    """
    forecast_samples = _validate_paths(forecast_samples, name="forecast_samples")
    observations = _validate_paths(observations, name="observations").to(
        device=forecast_samples.device,
        dtype=forecast_samples.dtype,
    )
    _check_same_path_shape(forecast_samples, observations)
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    forecast = forecast_samples.reshape(int(forecast_samples.shape[0]), -1)
    observed = observations.reshape(int(observations.shape[0]), -1)
    sample_count, feature_count = int(forecast.shape[0]), int(forecast.shape[1])
    sorted_forecast, _ = torch.sort(forecast, dim=0)
    prefix = torch.cat(
        [torch.zeros(1, feature_count, device=forecast.device, dtype=forecast.dtype), torch.cumsum(sorted_forecast, dim=0)],
        dim=0,
    )
    weights = (
        2.0 * torch.arange(1, sample_count + 1, device=forecast.device, dtype=forecast.dtype)
        - float(sample_count)
        - 1.0
    ).reshape(sample_count, 1)
    pairwise_abs_mean = 2.0 * torch.sum(weights * sorted_forecast, dim=0) / float(sample_count * sample_count)

    feature_scores = []
    for feature_index in range(feature_count):
        values = sorted_forecast[:, feature_index].contiguous()
        obs = observed[:, feature_index].contiguous()
        left_count = torch.searchsorted(values, obs, right=False)
        left_sum = prefix[left_count, feature_index]
        total = prefix[-1, feature_index]
        right_count = sample_count - left_count
        right_sum = total - left_sum
        mean_abs = (
            obs * left_count.to(obs.dtype)
            - left_sum
            + right_sum
            - obs * right_count.to(obs.dtype)
        ) / float(sample_count)
        feature_scores.append(mean_abs.mean() - 0.5 * pairwise_abs_mean[feature_index])
    crps_by_feature = torch.stack(feature_scores)

    scale = observed.std(dim=0, unbiased=False).clamp_min(eps)
    normalized = crps_by_feature / scale
    return {
        "crps": float(crps_by_feature.mean().item()),
        "crps_normalized": float(normalized.mean().item()),
    }


def discriminative_score(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    config: BenchmarkScoreConfig | None = None,
) -> dict[str, float]:
    """
    Train a classifier to distinguish generated from target paths.

    ``discriminative_score = abs(accuracy - 0.5)``.  Lower is better; zero
    means the classifier is no better than random guessing.
    """
    cfg = (config or BenchmarkScoreConfig()).validate()
    generated_paths = _validate_paths(generated_paths, name="generated_paths")
    target_paths = _validate_paths(target_paths, name="target_paths").to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    _check_same_path_shape(generated_paths, target_paths)
    count = min(int(generated_paths.shape[0]), int(target_paths.shape[0]))
    generated_paths = generated_paths[:count]
    target_paths = target_paths[:count]

    generator = _seeded_generator(cfg.seed)
    gen_train, gen_eval = _train_eval_indices(count, train_fraction=cfg.train_fraction, generator=generator)
    tgt_train, tgt_eval = _train_eval_indices(count, train_fraction=cfg.train_fraction, generator=generator)
    gen_train = gen_train.to(generated_paths.device)
    gen_eval = gen_eval.to(generated_paths.device)
    tgt_train = tgt_train.to(generated_paths.device)
    tgt_eval = tgt_eval.to(generated_paths.device)

    train_combined = torch.cat([generated_paths[gen_train], target_paths[tgt_train]], dim=0)
    mean, std = _normalizer(train_combined)
    generated_norm = (generated_paths - mean) / std
    target_norm = (target_paths - mean) / std

    _device_seed(generated_paths.device, cfg.seed + 101)
    model = _PathClassifier(state_dim=int(generated_paths.shape[2]), hidden_dim=cfg.hidden_dim).to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_generator = _seeded_generator(cfg.seed + 102)
    batch_half = max(1, cfg.batch_size // 2)

    for _ in range(cfg.train_steps):
        generated_batch = generated_norm[gen_train[_sample_indices(int(gen_train.numel()), batch_half, train_generator, generated_paths.device)]]
        target_batch = target_norm[tgt_train[_sample_indices(int(tgt_train.numel()), batch_half, train_generator, generated_paths.device)]]
        batch = torch.cat([generated_batch, target_batch], dim=0)
        labels = torch.cat(
            [
                torch.zeros(int(generated_batch.shape[0]), device=generated_paths.device, dtype=generated_paths.dtype),
                torch.ones(int(target_batch.shape[0]), device=generated_paths.device, dtype=generated_paths.dtype),
            ],
            dim=0,
        )
        logits = model(batch)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_batch = torch.cat([generated_norm[gen_eval], target_norm[tgt_eval]], dim=0)
        eval_labels = torch.cat(
            [
                torch.zeros(int(gen_eval.numel()), device=generated_paths.device, dtype=generated_paths.dtype),
                torch.ones(int(tgt_eval.numel()), device=generated_paths.device, dtype=generated_paths.dtype),
            ],
            dim=0,
        )
        eval_logits = model(eval_batch)
        eval_loss = loss_fn(eval_logits, eval_labels)
        predictions = (torch.sigmoid(eval_logits) >= 0.5).to(eval_labels.dtype)
        accuracy = torch.mean((predictions == eval_labels).to(eval_labels.dtype))
    accuracy_value = float(accuracy.item())
    return {
        "discriminative_accuracy": accuracy_value,
        "discriminative_score": abs(accuracy_value - 0.5),
        "discriminative_loss": float(eval_loss.item()),
        "discriminative_eval_paths": float(eval_batch.shape[0]),
    }


def predictive_score(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    config: BenchmarkScoreConfig | None = None,
) -> dict[str, float]:
    """
    Train a one-step return predictor on generated paths and evaluate on target paths.

    Lower MAE/RMSE means generated scenarios contain more useful predictive
    structure for the real held-out paths.
    """
    cfg = (config or BenchmarkScoreConfig()).validate()
    generated_paths = _validate_paths(generated_paths, name="generated_paths")
    target_paths = _validate_paths(target_paths, name="target_paths").to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    _check_same_path_shape(generated_paths, target_paths)

    generated_prefixes = generated_paths[:, :-1, :]
    generated_returns = path_returns(generated_paths)
    target_prefixes = target_paths[:, :-1, :]
    target_returns = path_returns(target_paths)
    prefix_mean, prefix_std = _normalizer(generated_prefixes)
    return_mean, return_std = _normalizer(generated_returns)
    generated_prefixes = (generated_prefixes - prefix_mean) / prefix_std
    generated_returns_norm = (generated_returns - return_mean) / return_std
    target_prefixes = (target_prefixes - prefix_mean) / prefix_std
    target_returns_norm = (target_returns - return_mean) / return_std

    _device_seed(generated_paths.device, cfg.seed + 201)
    model = _NextReturnPredictor(state_dim=int(generated_paths.shape[2]), hidden_dim=cfg.hidden_dim).to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_generator = _seeded_generator(cfg.seed + 202)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.train_steps):
        indices = _sample_indices(int(generated_prefixes.shape[0]), cfg.batch_size, train_generator, generated_paths.device)
        prediction = model(generated_prefixes[indices])
        loss = loss_fn(prediction, generated_returns_norm[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction_norm = model(target_prefixes)
        error_norm = prediction_norm - target_returns_norm
        prediction = prediction_norm * return_std + return_mean
        error = prediction - target_returns
        mae = torch.mean(error.abs())
        rmse = torch.sqrt(torch.mean(error.pow(2)))
        normalized_mae = torch.mean(error_norm.abs())
        normalized_rmse = torch.sqrt(torch.mean(error_norm.pow(2)))
    return {
        "predictive_mae": float(mae.item()),
        "predictive_rmse": float(rmse.item()),
        "predictive_normalized_mae": float(normalized_mae.item()),
        "predictive_normalized_rmse": float(normalized_rmse.item()),
    }


def probabilistic_forecasting_scores(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
) -> dict[str, float]:
    """CRPS scores for path levels and returns."""
    generated_paths = _validate_paths(generated_paths, name="generated_paths")
    target_paths = _validate_paths(target_paths, name="target_paths").to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    _check_same_path_shape(generated_paths, target_paths)
    level = empirical_crps(generated_paths, target_paths)
    returns = empirical_crps(path_returns(generated_paths), path_returns(target_paths))
    return {
        "crps_level": level["crps"],
        "crps_level_normalized": level["crps_normalized"],
        "crps_return": returns["crps"],
        "crps_return_normalized": returns["crps_normalized"],
    }


def benchmark_scores(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    config: BenchmarkScoreConfig | None = None,
) -> dict[str, float]:
    """Compute discriminative, predictive, and CRPS scores."""
    cfg = (config or BenchmarkScoreConfig()).validate()
    generated_paths = _validate_paths(generated_paths, name="generated_paths")
    target_paths = _validate_paths(target_paths, name="target_paths").to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    _check_same_path_shape(generated_paths, target_paths)
    metrics: dict[str, float] = {}
    metrics.update(discriminative_score(generated_paths, target_paths, config=cfg))
    metrics.update(predictive_score(generated_paths, target_paths, config=cfg))
    metrics.update(probabilistic_forecasting_scores(generated_paths, target_paths))
    metrics.update(
        {
            "score_train_steps": float(cfg.train_steps),
            "score_batch_size": float(cfg.batch_size),
            "score_hidden_dim": float(cfg.hidden_dim),
        }
    )
    return metrics
