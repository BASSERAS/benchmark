#!/usr/bin/env python3
"""Matched final comparison for the persistent Heston-mixture experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from run_matched_control_synthetic_validation import _evaluate_common  # noqa: E402


SEEDS = (0, 1, 3, 4)
MIXTURE_ROOT = REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
FINAL_ROOT = (
    REPO_ROOT
    / "runs"
    / "selected_deep_mkv_online_completion_20260805"
    / "mixture_eta2_lr1p5e3_gate"
)
SBTS_ROOT = (
    REPO_ROOT
    / "runs"
    / "paper_synthetic_4seed_20260801"
    / "heston_mixture"
    / "sbts"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "final_mixture_baseline_comparison_20260806"

COMMON_GROUPS = {
    "Distribution": (
        "path_swd_normalized",
        "terminal_w1_normalized",
        "increment_w1_normalized",
        "realized_volatility_w1_normalized",
        "maximum_drawdown_w1_normalized",
        "return_qq_rmse_normalized",
    ),
    "Dependence": (
        "return_acf_rmse",
        "absolute_return_acf_rmse",
        "squared_return_acf_rmse",
    ),
}
MIXTURE_KEYS = (
    "regime_proportion_tvd",
    "theta_support_normalized_wasserstein",
    "xi_support_normalized_wasserstein",
    "rho_support_normalized_wasserstein",
)
CONDITIONAL_KEYS = (
    "prefix_future_rv_correlation_error",
    "prefix_high_low_future_rv_gap_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deep-root", type=Path, default=FINAL_ROOT)
    parser.add_argument(
        "--deep-bank-relative",
        type=Path,
        default=Path("volatility_only_online_mp/validation_bank.npy"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def paths_for_seed(
    seed: int,
    *,
    deep_root: Path,
    deep_bank_relative: Path,
) -> dict[str, Path]:
    final_seed = deep_root / f"seed_{seed}"
    return {
        "Fitted reference": final_seed / "reference" / "validation_bank.npy",
        "Deep MKV": final_seed / deep_bank_relative,
        "SBTS": SBTS_ROOT / f"seed_{seed}" / "generated_paths_8192x128.npy",
    }


def conditional_errors(metrics: dict[str, object]) -> dict[str, float]:
    memory = metrics["conditional_path_memory"]
    target = memory["target"]
    generated = memory["generated"]
    return {
        "prefix_future_rv_correlation_error": abs(
            float(target["prefix_future_rv_correlation"])
            - float(generated["prefix_future_rv_correlation"])
        ),
        "prefix_high_low_future_rv_gap_error": abs(
            float(target["prefix_high_low_future_rv_gap"])
            - float(generated["prefix_high_low_future_rv_gap"])
        ),
    }


def flattened_mixture(metrics: dict[str, object]) -> dict[str, float]:
    fidelity = metrics["mixture_fidelity"]
    parameters = fidelity["parameters"]
    return {
        "regime_proportion_tvd": float(fidelity["regime_proportion_tvd"]),
        **{
            f"{name}_support_normalized_wasserstein": float(
                parameters[name]["support_normalized_wasserstein"]
            )
            for name in ("theta", "xi", "rho")
        },
    }


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "sample_sd": float(stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    deep_root = args.deep_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = MIXTURE_ROOT / "data" / "train.npy"
    target_path = MIXTURE_ROOT / "data" / "test.npy"
    oracle_path = MIXTURE_ROOT / "oracle" / "oracle.joblib"
    gate_path = MIXTURE_ROOT / "oracle" / "gate_report.json"
    target = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.float64)
    device = torch.device(args.device)

    result: dict[str, object] = {
        "protocol": {
            "purpose": "final_evaluation_only",
            "selection_scope": "no_model_or_checkpoint_selection",
            "deep_root": str(deep_root),
            "deep_bank_relative": str(args.deep_bank_relative),
            "seeds": list(SEEDS),
            "target": str(target_path),
            "target_sha256": sha256(target_path),
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "common_evaluator": (
                "run_matched_control_synthetic_validation._evaluate_common"
            ),
            "mixture_evaluator": "evaluate_heston_parameter_mixture.py",
        },
        "per_seed": {},
    }

    for seed in SEEDS:
        seed_result: dict[str, object] = {}
        for label, bank_path in paths_for_seed(
            seed,
            deep_root=deep_root,
            deep_bank_relative=args.deep_bank_relative,
        ).items():
            if not bank_path.is_file():
                raise FileNotFoundError(bank_path)
            bank = np.asarray(np.load(bank_path, allow_pickle=False), dtype=np.float64)
            if bank.shape != target.shape:
                raise ValueError(
                    f"{label} seed {seed}: {bank.shape} != target {target.shape}"
                )
            if not np.isfinite(bank).all() or np.any(bank <= 0.0):
                raise ValueError(f"invalid bank: {bank_path}")

            method_slug = label.lower().replace(" ", "_")
            method_dir = output / f"seed_{seed}" / method_slug
            common_path = method_dir / "common_metrics.json"
            mixture_path = method_dir / "mixture_metrics.json"
            if common_path.is_file():
                common = read_json(common_path)
            else:
                common = _evaluate_common(
                    target_prices=target,
                    generated=bank,
                    device=device,
                    seed=seed,
                )
                write_json(common_path, common)
            if not mixture_path.is_file():
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            REPO_ROOT
                            / "experiments"
                            / "scripts"
                            / "evaluate_heston_parameter_mixture.py"
                        ),
                        "--train-data",
                        str(train_path),
                        "--test-data",
                        str(target_path),
                        "--generated-data",
                        str(bank_path),
                        "--oracle",
                        str(oracle_path),
                        "--oracle-gate-report",
                        str(gate_path),
                        "--output",
                        str(mixture_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            mixture = read_json(mixture_path)
            seed_result[label] = {
                "bank": str(bank_path),
                "bank_sha256": sha256(bank_path),
                "common_metrics": str(common_path),
                "mixture_metrics": str(mixture_path),
                "path_law": common["path_law"],
                "conditional": conditional_errors(common),
                "mixture_fidelity": flattened_mixture(mixture),
                "observable_fidelity": mixture["observable_fidelity"],
                "novelty": mixture["novelty"],
            }
            print(f"completed seed={seed} method={label}", flush=True)
        result["per_seed"][str(seed)] = seed_result

    methods = ("Fitted reference", "Deep MKV", "SBTS")
    aggregate: dict[str, object] = {}
    for method in methods:
        aggregate[method] = {
            "path_law": {
                key: summary(
                    [
                        float(result["per_seed"][str(seed)][method]["path_law"][key])
                        for seed in SEEDS
                    ]
                )
                for keys in COMMON_GROUPS.values()
                for key in keys
            },
            "conditional": {
                key: summary(
                    [
                        float(result["per_seed"][str(seed)][method]["conditional"][key])
                        for seed in SEEDS
                    ]
                )
                for key in CONDITIONAL_KEYS
            },
            "mixture_fidelity": {
                key: summary(
                    [
                        float(
                            result["per_seed"][str(seed)][method]["mixture_fidelity"][key]
                        )
                        for seed in SEEDS
                    ]
                )
                for key in MIXTURE_KEYS
            },
        }
    result["aggregate"] = aggregate

    comparison: dict[str, object] = {}
    categories = {
        **COMMON_GROUPS,
        "Conditional memory": CONDITIONAL_KEYS,
        "Hidden regimes": MIXTURE_KEYS,
    }
    for category, keys in categories.items():
        source = (
            "path_law"
            if category in COMMON_GROUPS
            else "conditional"
            if category == "Conditional memory"
            else "mixture_fidelity"
        )
        paired = []
        metric_median_wins = 0
        for key in keys:
            deep_median = float(aggregate["Deep MKV"][source][key]["median"])
            sbts_median = float(aggregate["SBTS"][source][key]["median"])
            if deep_median < sbts_median:
                metric_median_wins += 1
            for seed in SEEDS:
                deep = float(result["per_seed"][str(seed)]["Deep MKV"][source][key])
                sbts = float(result["per_seed"][str(seed)]["SBTS"][source][key])
                paired.append((sbts - deep) / max(abs(sbts), 1e-12))
        comparison[category] = {
            "metrics_with_lower_deep_median": metric_median_wins,
            "number_of_metrics": len(keys),
            "paired_seed_metric_wins": sum(value > 0.0 for value in paired),
            "paired_seed_metric_cases": len(paired),
            "median_relative_error_reduction_vs_sbts": float(median(paired)),
        }
    result["deep_mkv_vs_sbts"] = comparison
    write_json(output / "SUMMARY.json", result)

    def med(method: str, source: str, key: str) -> float:
        return float(aggregate[method][source][key]["median"])

    lines = [
        "# Final matched persistent-mixture comparison",
        "",
        "All entries use four frozen banks (seeds 0, 1, 3, and 4) and the same",
        "independent test target. No model, checkpoint, or hyperparameter was selected",
        "from these scores. Lower values are better.",
        "",
        "## Path and dependence metrics (median over four seeds)",
        "",
        "| Method | Path SWD | RV W1 | Drawdown W1 | |r| ACF | r^2 ACF | Early/future corr. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method} | {med(method, 'path_law', 'path_swd_normalized'):.4f} "
            f"| {med(method, 'path_law', 'realized_volatility_w1_normalized'):.4f} "
            f"| {med(method, 'path_law', 'maximum_drawdown_w1_normalized'):.4f} "
            f"| {med(method, 'path_law', 'absolute_return_acf_rmse'):.4f} "
            f"| {med(method, 'path_law', 'squared_return_acf_rmse'):.4f} "
            f"| {med(method, 'conditional', 'prefix_future_rv_correlation_error'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Hidden-regime metrics (median over four seeds)",
            "",
            "| Method | Regime TVD | Theta W1 | Xi W1 | Rho W1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        lines.append(
            f"| {method} | {med(method, 'mixture_fidelity', 'regime_proportion_tvd'):.4f} "
            f"| {med(method, 'mixture_fidelity', 'theta_support_normalized_wasserstein'):.4f} "
            f"| {med(method, 'mixture_fidelity', 'xi_support_normalized_wasserstein'):.4f} "
            f"| {med(method, 'mixture_fidelity', 'rho_support_normalized_wasserstein'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Deep MKV versus SBTS",
            "",
            "| Metric family | Lower Deep median | Paired seed-metric wins | Median relative error reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, row in comparison.items():
        lines.append(
            f"| {category} | {row['metrics_with_lower_deep_median']}/{row['number_of_metrics']} "
            f"| {row['paired_seed_metric_wins']}/{row['paired_seed_metric_cases']} "
            f"| {100.0 * row['median_relative_error_reduction_vs_sbts']:.1f}% |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
