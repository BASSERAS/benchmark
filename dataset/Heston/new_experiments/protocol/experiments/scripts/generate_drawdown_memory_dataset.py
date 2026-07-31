from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a delayed drawdown-memory stochastic-volatility benchmark."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-paths", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dt", type=float, default=1.0 / 250.0)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--sigma-low", type=float, default=0.12)
    parser.add_argument("--sigma-high", type=float, default=0.45)
    parser.add_argument("--drawdown-threshold", type=float, default=0.04)
    parser.add_argument("--memory-half-life", type=float, default=60.0)
    parser.add_argument("--response-delay", type=int, default=32)
    return parser.parse_args()


def simulate(
    *,
    seed: int,
    num_paths: int,
    sequence_length: int,
    dt: float,
    s0: float,
    sigma_low: float,
    sigma_high: float,
    drawdown_threshold: float,
    memory_half_life: float,
    response_delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    log_paths = np.zeros((num_paths, sequence_length), dtype=np.float64)
    memory = np.zeros(num_paths, dtype=np.float64)
    memory_history = np.zeros_like(log_paths)
    running_max = np.zeros(num_paths, dtype=np.float64)
    sigma_history = np.empty((num_paths, sequence_length - 1), dtype=np.float64)
    decay = math.exp(-math.log(2.0) / float(memory_half_life))

    for step in range(sequence_length - 1):
        drawdown = running_max - log_paths[:, step]
        memory = np.maximum(
            decay * memory,
            (drawdown >= float(drawdown_threshold)).astype(np.float64),
        )
        memory_history[:, step] = memory
        delayed_memory = (
            memory_history[:, step - response_delay]
            if step >= response_delay
            else np.zeros(num_paths, dtype=np.float64)
        )
        sigma = sigma_low + (sigma_high - sigma_low) * delayed_memory
        sigma_history[:, step] = sigma
        innovation = rng.standard_normal(num_paths)
        log_paths[:, step + 1] = (
            log_paths[:, step]
            - 0.5 * sigma * sigma * dt
            + sigma * math.sqrt(dt) * innovation
        )
        running_max = np.maximum(running_max, log_paths[:, step + 1])

    prices = float(s0) * np.exp(log_paths)
    return prices, sigma_history


def main() -> None:
    args = parse_args()
    if args.num_paths < 1 or args.sequence_length < 3:
        raise ValueError("num_paths must be positive and sequence_length must be >= 3")
    if args.response_delay < 1 or args.response_delay >= args.sequence_length - 1:
        raise ValueError("response_delay must be in [1, sequence_length - 2]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "name": "delayed_drawdown_memory_volatility",
        "num_paths": int(args.num_paths),
        "sequence_length": int(args.sequence_length),
        "dt": float(args.dt),
        "s0": float(args.s0),
        "sigma_low": float(args.sigma_low),
        "sigma_high": float(args.sigma_high),
        "drawdown_threshold": float(args.drawdown_threshold),
        "memory_half_life": float(args.memory_half_life),
        "response_delay": int(args.response_delay),
        "split_index": 64,
        "history_cutoff": 40,
        "recent_window": 20,
        "future_horizon": 32,
        "seeds": {"train": 0, "test": 1, "disc": 2},
    }
    diagnostics: dict[str, object] = {}
    for name, seed in configuration["seeds"].items():
        prices, sigma = simulate(
            seed=int(seed),
            num_paths=int(args.num_paths),
            sequence_length=int(args.sequence_length),
            dt=float(args.dt),
            s0=float(args.s0),
            sigma_low=float(args.sigma_low),
            sigma_high=float(args.sigma_high),
            drawdown_threshold=float(args.drawdown_threshold),
            memory_half_life=float(args.memory_half_life),
            response_delay=int(args.response_delay),
        )
        np.save(args.output_dir / f"{name}.npy", prices)
        np.save(args.output_dir / f"{name}_sigma.npy", sigma.astype(np.float32))
        returns = np.diff(np.log(prices), axis=1)
        diagnostics[name] = {
            "shape": list(prices.shape),
            "log_return_mean": float(returns.mean()),
            "log_return_std": float(returns.std()),
            "sigma_mean": float(sigma.mean()),
            "sigma_std": float(sigma.std()),
            "price_min": float(prices.min()),
            "price_max": float(prices.max()),
        }

    payload = {"configuration": configuration, "diagnostics": diagnostics}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
