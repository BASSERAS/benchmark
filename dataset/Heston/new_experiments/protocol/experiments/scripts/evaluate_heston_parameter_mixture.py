from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.heston_mixture import (  # noqa: E402
    DT,
    extract_price_path_features,
    parameter_values_by_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Heston-mixture and observable path-law fidelity.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--generated-data", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-gate-report", type=Path, required=True)
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


def leverage_curve(returns: np.ndarray, lags: np.ndarray) -> np.ndarray:
    result = []
    for lag in lags:
        left = returns[:, : returns.shape[1] - lag] if lag else returns
        right = returns[:, lag:] ** 2 if lag else returns**2
        left = left - left.mean()
        right = right - right.mean()
        result.append(
            np.mean(left * right)
            / np.sqrt(np.mean(left * left) * np.mean(right * right) + 1e-18)
        )
    return np.asarray(result)


def excess_kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    variance = np.mean(centered**2)
    return float(np.mean(centered**4) / (variance * variance) - 3.0)


def novelty(train_returns: np.ndarray, query_returns: np.ndarray) -> dict[str, float]:
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


def oracle_summary(classifier, prices: np.ndarray) -> dict[str, object]:
    features = extract_price_path_features(prices)
    probabilities = classifier.predict_proba(features)
    prediction = np.argmax(probabilities, axis=1)
    proportions = np.bincount(prediction, minlength=8).astype(np.float64) / len(prediction)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    parameter_samples = {
        name: probabilities @ parameter_values_by_label(name)
        for name in ("theta", "xi", "rho")
    }
    return {
        "probabilities": probabilities,
        "prediction": prediction,
        "regime_proportions": proportions,
        "mean_posterior_entropy": float(entropy.mean()),
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "low_confidence_fraction": float(np.mean(probabilities.max(axis=1) < 0.6)),
        "parameter_samples": parameter_samples,
    }


def main() -> None:
    args = parse_args()
    train = load_prices(args.train_data)
    test = load_prices(args.test_data)
    generated = load_prices(args.generated_data)
    classifier = joblib.load(args.oracle)
    target_oracle = oracle_summary(classifier, test)
    generated_oracle = oracle_summary(classifier, generated)
    target_returns = np.diff(np.log(test), axis=1)
    generated_returns = np.diff(np.log(generated), axis=1)
    train_returns = np.diff(np.log(train), axis=1)
    acf_lags = np.arange(1, 51)
    leverage_lags = np.arange(0, 21)
    target_rv = np.sqrt(np.mean(target_returns**2, axis=1) / DT)
    generated_rv = np.sqrt(np.mean(generated_returns**2, axis=1) / DT)
    target_proportions = target_oracle["regime_proportions"]
    generated_proportions = generated_oracle["regime_proportions"]

    parameter_metrics = {}
    for name in ("theta", "xi", "rho"):
        target_parameter = target_oracle["parameter_samples"][name]
        generated_parameter = generated_oracle["parameter_samples"][name]
        parameter_support = parameter_values_by_label(name)
        support_width = float(parameter_support.max() - parameter_support.min())
        wasserstein = float(
            wasserstein_distance(target_parameter, generated_parameter)
        )
        parameter_metrics[name] = {
            "wasserstein": wasserstein,
            "support_normalized_wasserstein": wasserstein / support_width,
            "mean_error": abs(
                float(generated_parameter.mean() - target_parameter.mean())
            ),
            "std_ratio": float(
                generated_parameter.std() / max(float(target_parameter.std()), 1e-12)
            ),
            "q05_error": abs(
                float(
                    np.quantile(generated_parameter, 0.05)
                    - np.quantile(target_parameter, 0.05)
                )
            ),
            "q95_error": abs(
                float(
                    np.quantile(generated_parameter, 0.95)
                    - np.quantile(target_parameter, 0.95)
                )
            ),
        }
    payload = {
        "mixture_fidelity": {
            "regime_proportion_tvd": float(
                0.5 * np.sum(np.abs(generated_proportions - target_proportions))
            ),
            "target_regime_proportions": target_proportions.tolist(),
            "generated_regime_proportions": generated_proportions.tolist(),
            "target_mean_posterior_entropy": target_oracle["mean_posterior_entropy"],
            "generated_mean_posterior_entropy": generated_oracle[
                "mean_posterior_entropy"
            ],
            "target_mean_max_probability": target_oracle["mean_max_probability"],
            "generated_mean_max_probability": generated_oracle[
                "mean_max_probability"
            ],
            "target_low_confidence_fraction": target_oracle[
                "low_confidence_fraction"
            ],
            "generated_low_confidence_fraction": generated_oracle[
                "low_confidence_fraction"
            ],
            "parameters": parameter_metrics,
        },
        "observable_fidelity": {
            "return_std_error": abs(
                float(generated_returns.std() - target_returns.std())
            ),
            "excess_kurtosis_error": abs(
                excess_kurtosis(generated_returns)
                - excess_kurtosis(target_returns)
            ),
            "abs_return_acf_rmse_lags_1_50": float(
                np.sqrt(
                    np.mean(
                        (
                            pooled_acf(np.abs(generated_returns), acf_lags)
                            - pooled_acf(np.abs(target_returns), acf_lags)
                        )
                        ** 2
                    )
                )
            ),
            "squared_return_acf_rmse_lags_1_50": float(
                np.sqrt(
                    np.mean(
                        (
                            pooled_acf(generated_returns**2, acf_lags)
                            - pooled_acf(target_returns**2, acf_lags)
                        )
                        ** 2
                    )
                )
            ),
            "leverage_curve_rmse_lags_0_20": float(
                np.sqrt(
                    np.mean(
                        (
                            leverage_curve(generated_returns, leverage_lags)
                            - leverage_curve(target_returns, leverage_lags)
                        )
                        ** 2
                    )
                )
            ),
            "realized_volatility_wasserstein": float(
                wasserstein_distance(target_rv, generated_rv)
            ),
            "terminal_log_price_ks": float(
                ks_2samp(
                    np.log(test[:, -1] / test[:, 0]),
                    np.log(generated[:, -1] / generated[:, 0]),
                ).statistic
            ),
        },
        "novelty": novelty(train_returns, generated_returns),
        "oracle_gate": json.loads(
            args.oracle_gate_report.read_text(encoding="utf-8")
        ),
        "sources": {
            "train": str(args.train_data.resolve()),
            "test": str(args.test_data.resolve()),
            "generated": str(args.generated_data.resolve()),
            "oracle": str(args.oracle.resolve()),
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
