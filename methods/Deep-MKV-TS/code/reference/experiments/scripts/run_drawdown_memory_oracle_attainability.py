from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.drawdown_oracle import (  # noqa: E402
    control_prediction_metrics,
    delayed_state,
    exact_observable_oracle,
    fit_return_only_oracle,
    fit_sigma_supervised_oracle,
    observable_drawdown_memory,
    rollout_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the delayed drawdown-memory law is attainable by the "
            "existing Euler dynamics when supplied with increasingly strong "
            "oracle controls."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "drawdown_memory_seed0_comparison_20260730"
            / "data"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--num-paths", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=730_001)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"drawdown_memory_oracle_attainability_{timestamp}"


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_bank(
    *,
    dataset_root: Path,
    bank_path: Path,
    output_path: Path,
) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "evaluate_drawdown_memory.py"),
            "--train-data",
            str(dataset_root / "train.npy"),
            "--test-data",
            str(dataset_root / "disc.npy"),
            "--generated-data",
            str(bank_path),
            "--dataset-manifest",
            str(dataset_root / "manifest.json"),
            "--output",
            str(output_path),
        ],
        check=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    run_dir = Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    train_prices = load_prices(dataset_root / "train.npy")
    validation_prices = load_prices(dataset_root / "disc.npy")
    train_sigma = np.asarray(
        np.load(dataset_root / "train_sigma.npy"),
        dtype=np.float64,
    )
    validation_sigma = np.asarray(
        np.load(dataset_root / "disc_sigma.npy"),
        dtype=np.float64,
    )
    train_log = np.log(train_prices / train_prices[:, :1])
    validation_log = np.log(validation_prices / validation_prices[:, :1])
    threshold = float(config["drawdown_threshold"])
    half_life = float(config["memory_half_life"])
    delay = int(config["response_delay"])
    dt = float(config["dt"])

    train_current = observable_drawdown_memory(
        train_log,
        threshold=threshold,
        half_life=half_life,
    )
    validation_current = observable_drawdown_memory(
        validation_log,
        threshold=threshold,
        half_life=half_life,
    )
    train_delayed = delayed_state(train_current, delay=delay)
    validation_delayed = delayed_state(validation_current, delay=delay)

    arms = {
        "return_only_current_state": fit_return_only_oracle(
            log_paths=train_log,
            state=train_current,
            dt=dt,
            state_source="current",
            ridge=float(args.ridge),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        ),
        "return_only_delayed_state": fit_return_only_oracle(
            log_paths=train_log,
            state=train_delayed,
            dt=dt,
            state_source="delayed",
            ridge=float(args.ridge),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        ),
        "sigma_supervised_delayed_state": fit_sigma_supervised_oracle(
            state=train_delayed,
            sigma_history=train_sigma,
            state_source="delayed",
            ridge=max(float(args.ridge) * 0.01, 1e-12),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        ),
        "exact_observable_recursion": exact_observable_oracle(
            sigma_low=float(config["sigma_low"]),
            sigma_high=float(config["sigma_high"]),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        ),
    }

    results: dict[str, object] = {}
    for arm_index, (name, control) in enumerate(arms.items()):
        print(f"running {name}", flush=True)
        prediction = control_prediction_metrics(
            control=control,
            current_memory=validation_current,
            delayed_memory=validation_delayed,
            target_sigma=validation_sigma,
        )
        prices, generated_sigma = rollout_oracle(
            control=control,
            num_paths=int(args.num_paths),
            sequence_length=int(config["sequence_length"]),
            seed=int(args.seed) + arm_index,
            dt=dt,
            s0=float(config["s0"]),
            drawdown_threshold=threshold,
            memory_half_life=half_life,
            response_delay=delay,
        )
        arm_dir = run_dir / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        bank_path = arm_dir / (
            f"generated_paths_{int(args.num_paths)}x"
            f"{int(config['sequence_length'])}.npy"
        )
        np.save(bank_path, prices.astype(np.float32))
        np.save(arm_dir / "generated_sigma.npy", generated_sigma.astype(np.float32))
        metrics = evaluate_bank(
            dataset_root=dataset_root,
            bank_path=bank_path,
            output_path=arm_dir / "validation_metrics.json",
        )
        result = {
            "control": control.summary(),
            "heldout_control_prediction": prediction,
            "validation": metrics,
        }
        write_json(arm_dir / "diagnostic.json", result)
        results[name] = result

    target = next(iter(results.values()))["validation"]["target_memory"]
    summary = {
        "design": {
            "dataset_root": str(dataset_root.resolve()),
            "fit_split": "train.npy",
            "validation_split": "disc.npy",
            "test_split_used": False,
            "same_forward_dynamics": (
                "X[t+1] = X[t] + alpha[t] * dt "
                "+ sigma[t] * sqrt(dt) * epsilon[t+1]"
            ),
            "observable_memory_is_deterministic": True,
            "response_delay": delay,
            "return_only_uses_sigma_labels": False,
            "sigma_supervised_is_privileged_upper_bound": True,
        },
        "target_memory": target,
        "arms": results,
    }
    write_json(run_dir / "summary.json", summary)
    compact = {
        name: {
            **result["heldout_control_prediction"],
            **{
                key: result["validation"]["generated_memory"][key]
                for key in (
                    "early_history_incremental_r2",
                    "early_hit_future_rv_correlation",
                    "early_hit_standardized_coefficient",
                    "future_rv_hit_gap",
                    "early_hit_rate",
                )
            },
            **{
                f"error_{key}": result["validation"]["errors"][key]
                for key in (
                    "future_rv_wasserstein",
                    "abs_return_acf_rmse_lags_1_50",
                    "squared_return_acf_rmse_lags_1_50",
                    "terminal_log_price_ks",
                )
            },
        }
        for name, result in results.items()
    }
    write_json(run_dir / "compact_summary.json", compact)
    print(json.dumps({"run_dir": str(run_dir.resolve()), "results": compact}, indent=2))


if __name__ == "__main__":
    main()
