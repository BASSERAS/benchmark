from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unconditional and drawdown-memory fidelity."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--generated-data", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_prices(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return value


def pooled_acf(values: np.ndarray, lags: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    variance = np.mean(centered * centered)
    return np.asarray(
        [np.mean(centered[:, :-lag] * centered[:, lag:]) / variance for lag in lags]
    )


def excess_kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    variance = np.mean(centered**2)
    return float(np.mean(centered**4) / (variance * variance) - 3.0)


def drawdown_features(
    prices: np.ndarray,
    *,
    threshold: float,
    split: int,
    history_cutoff: int,
    recent_window: int,
    horizon: int,
    annualization: float,
) -> dict[str, np.ndarray]:
    log_paths = np.log(prices / prices[:, :1])
    returns = np.diff(log_paths, axis=1)
    running_max = np.maximum.accumulate(log_paths, axis=1)
    drawdown = running_max - log_paths
    early_hit = drawdown[:, : history_cutoff + 1].max(axis=1) >= threshold
    recent_rv = np.sqrt(
        annualization * np.mean(returns[:, split - recent_window : split] ** 2, axis=1)
    )
    future_rv = np.sqrt(
        annualization * np.mean(returns[:, split : split + horizon] ** 2, axis=1)
    )
    return {
        "log_paths": log_paths,
        "returns": returns,
        "early_hit": early_hit,
        "recent_rv": recent_rv,
        "future_rv": future_rv,
        "current_drawdown": drawdown[:, split],
        "current_log_return": log_paths[:, split],
    }


def nested_signal(features: dict[str, np.ndarray]) -> dict[str, float]:
    y = features["future_rv"]
    y_std = max(float(y.std()), 1e-12)
    y_scaled = (y - y.mean()) / y_std
    columns = []
    for name in ("recent_rv", "current_drawdown", "current_log_return"):
        value = features[name]
        columns.append((value - value.mean()) / max(float(value.std()), 1e-12))
    baseline = np.column_stack([np.ones(len(y)), *columns])
    augmented = np.column_stack([baseline, features["early_hit"].astype(np.float64)])
    beta_base = np.linalg.lstsq(baseline, y_scaled, rcond=None)[0]
    beta_aug = np.linalg.lstsq(augmented, y_scaled, rcond=None)[0]
    residual_base = y_scaled - baseline @ beta_base
    residual_aug = y_scaled - augmented @ beta_aug
    r2_base = 1.0 - float(np.mean(residual_base**2))
    r2_aug = 1.0 - float(np.mean(residual_aug**2))
    return {
        "early_hit_standardized_coefficient": float(beta_aug[-1]),
        "baseline_r2": r2_base,
        "augmented_r2": r2_aug,
        "early_history_incremental_r2": r2_aug - r2_base,
    }


def summarize_memory(features: dict[str, np.ndarray]) -> dict[str, float]:
    hit = features["early_hit"]
    future = features["future_rv"]
    if hit.all() or (~hit).all():
        raise RuntimeError("drawdown groups must both be non-empty")
    signal = nested_signal(features)
    no_hit_mean = float(future[~hit].mean())
    hit_mean = float(future[hit].mean())
    return {
        "early_hit_rate": float(hit.mean()),
        "future_rv_no_hit_mean": no_hit_mean,
        "future_rv_hit_mean": hit_mean,
        "future_rv_hit_gap": hit_mean - no_hit_mean,
        "early_hit_future_rv_correlation": float(
            np.corrcoef(hit.astype(np.float64), future)[0, 1]
        ),
        **signal,
    }


def nearest_train_diagnostic(train_returns: np.ndarray, query_returns: np.ndarray) -> dict[str, float]:
    import faiss

    scale = max(float(train_returns.std()), 1e-12)
    train = np.ascontiguousarray((train_returns / scale).astype(np.float32))
    query = np.ascontiguousarray((query_returns / scale).astype(np.float32))
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    distance, neighbors = index.search(query, 1)
    rmse = np.sqrt(distance[:, 0] / train.shape[1])
    return {
        "median_standardized_nearest_train_path_rmse": float(np.median(rmse)),
        "mean_standardized_nearest_train_path_rmse": float(np.mean(rmse)),
        "distinct_nearest_training_paths": int(np.unique(neighbors[:, 0]).size),
    }


def main() -> None:
    args = parse_args()
    train = load_prices(args.train_data)
    test = load_prices(args.test_data)
    generated = load_prices(args.generated_data)
    if test.shape[1] != generated.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("all path banks must share the same sequence length")
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    kwargs = {
        "threshold": float(config["drawdown_threshold"]),
        "split": int(config["split_index"]),
        "history_cutoff": int(config["history_cutoff"]),
        "recent_window": int(config["recent_window"]),
        "horizon": int(config["future_horizon"]),
        "annualization": 1.0 / float(config["dt"]),
    }
    test_features = drawdown_features(test, **kwargs)
    generated_features = drawdown_features(generated, **kwargs)
    target_memory = summarize_memory(test_features)
    generated_memory = summarize_memory(generated_features)
    lags = np.arange(1, 51)
    target_returns = test_features["returns"]
    generated_returns = generated_features["returns"]
    target_abs_acf = pooled_acf(np.abs(target_returns), lags)
    generated_abs_acf = pooled_acf(np.abs(generated_returns), lags)
    target_sq_acf = pooled_acf(target_returns**2, lags)
    generated_sq_acf = pooled_acf(generated_returns**2, lags)
    target_return_scale = max(float(target_returns.std()), 1e-12)
    target_timewise_std = target_returns.std(axis=0)
    mean_target_timewise_std = max(float(target_timewise_std.mean()), 1e-12)

    metrics = {
        "return_mean_error": abs(
            float(generated_returns.mean() - target_returns.mean())
        ),
        "return_std_error": abs(float(generated_returns.std() - target_returns.std())),
        "timewise_return_mean_rmse_normalized": float(
            np.sqrt(
                np.mean(
                    (generated_returns.mean(axis=0) - target_returns.mean(axis=0))
                    ** 2
                )
            )
            / target_return_scale
        ),
        "timewise_return_std_rmse_normalized": float(
            np.sqrt(
                np.mean(
                    (generated_returns.std(axis=0) - target_timewise_std) ** 2
                )
            )
            / mean_target_timewise_std
        ),
        "excess_kurtosis_error": abs(
            excess_kurtosis(generated_returns) - excess_kurtosis(target_returns)
        ),
        "abs_return_acf_rmse_lags_1_50": float(
            np.sqrt(np.mean((generated_abs_acf - target_abs_acf) ** 2))
        ),
        "squared_return_acf_rmse_lags_1_50": float(
            np.sqrt(np.mean((generated_sq_acf - target_sq_acf) ** 2))
        ),
        "terminal_log_price_ks": float(
            ks_2samp(
                generated_features["log_paths"][:, -1],
                test_features["log_paths"][:, -1],
            ).statistic
        ),
        "future_rv_wasserstein": float(
            wasserstein_distance(
                generated_features["future_rv"],
                test_features["future_rv"],
            )
        ),
        "early_hit_rate_error": abs(
            generated_memory["early_hit_rate"] - target_memory["early_hit_rate"]
        ),
        "future_rv_hit_gap_error": abs(
            generated_memory["future_rv_hit_gap"] - target_memory["future_rv_hit_gap"]
        ),
        "early_history_incremental_r2_error": abs(
            generated_memory["early_history_incremental_r2"]
            - target_memory["early_history_incremental_r2"]
        ),
    }
    train_returns = np.diff(np.log(train), axis=1)
    novelty = nearest_train_diagnostic(train_returns, generated_returns)
    payload = {
        "configuration": kwargs,
        "target_memory": target_memory,
        "generated_memory": generated_memory,
        "errors": metrics,
        "novelty": novelty,
        "sources": {
            "train": str(args.train_data.resolve()),
            "test": str(args.test_data.resolve()),
            "generated": str(args.generated_data.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
