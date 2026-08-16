from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.locked_checkpoint import (  # noqa: E402
    coverage_focused_checkpoint_rule,
    summarize_metric_rows,
)
from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingResult,
    bootstrap_shadowing_metrics,
    evaluate_path_shadowing,
    paired_bootstrap_shadowing_difference,
)
from run_shadowing_sweep import _simulate_heston_bank  # noqa: E402
from run_specific_entropy_joint_shadow import (  # noqa: E402
    _build_models,
    _metric_subset,
    _sample_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen step-1600 coverage candidate against the "
            "selected joint specific-entropy checkpoint and a matched Heston "
            "oracle on multiple newly simulated test laws."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument(
        "--coverage-checkpoint",
        type=Path,
        default=Path(
            "runs/heston_dt_specific_entropy_complete_mp_dt_coverage_frontier_"
            "seed1234_20260721/validation_checkpoints/step_1600_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path(
            "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_"
            "10x200_seed1234_20260722/selected_model_checkpoint.pt"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=(917001, 917002, 917003, 917004, 917005),
    )
    parser.add_argument("--model-bank-seed-base", type=int, default=918001)
    parser.add_argument("--oracle-bank-seed-base", type=int, default=919001)
    parser.add_argument("--reference-paths", type=int, default=4096)
    parser.add_argument("--bank-paths", type=int, default=4096)
    parser.add_argument("--real-paths", type=int, default=512)
    parser.add_argument("--bank-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="exact")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001329)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument(
        "--save-banks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    test_seeds = tuple(int(value) for value in args.test_seeds)
    if not test_seeds or len(set(test_seeds)) != len(test_seeds):
        raise ValueError("test-seeds must be non-empty and unique")
    for name in (
        "reference_paths",
        "bank_paths",
        "real_paths",
        "bank_batch_size",
        "top_k",
        "shadow_exact_batch_size",
        "bootstrap_replicates",
        "swd_projections",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.reference_paths) < int(args.real_paths):
        raise ValueError("reference-paths must be at least real-paths")
    if int(args.bank_paths) < int(args.top_k):
        raise ValueError("bank-paths must be at least top-k")
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("bootstrap-replicates must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0, 1)")
    return test_seeds


def _load_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a checkpoint dictionary")
    return checkpoint


def _shadow(
    *,
    real_paths: torch.Tensor,
    bank: torch.Tensor,
    args: argparse.Namespace,
) -> ShadowingResult:
    return evaluate_path_shadowing(
        real_paths=real_paths,
        generated_paths=bank,
        prefix_length=int(args.split_step) + 1,
        top_k=int(args.top_k),
        future_horizon=int(args.future_steps),
        search_backend=str(args.shadow_search),
        exact_batch_size=int(args.shadow_exact_batch_size),
    )


def _stylized(
    bank: torch.Tensor,
    reference: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> dict[str, float]:
    lags = tuple(
        value for value in (1, 2, 5, 10, 20) if value < int(reference.shape[1]) - 1
    )
    return stylized_fact_metrics(
        bank,
        reference,
        lags=lags,
        tail_quantiles=(0.90, 0.95, 0.99),
        include_swd=True,
        swd_projections=int(args.swd_projections),
    )


def _bootstrap(
    result: ShadowingResult,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, dict[str, float]]:
    return bootstrap_shadowing_metrics(
        result,
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(seed),
    )


def _paired(
    primary: ShadowingResult,
    comparison: ShadowingResult,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, dict[str, float]]:
    return paired_bootstrap_shadowing_difference(
        primary,
        comparison,
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(seed),
    )


def _pooled_result(results: list[ShadowingResult]) -> ShadowingResult:
    if not results:
        raise ValueError("results must be non-empty")
    keys = set(results[0].path_metrics)
    if not keys or any(set(result.path_metrics) != keys for result in results[1:]):
        raise ValueError("all pooled results must expose identical path metrics")
    path_metrics = {
        name: torch.cat(
            [result.path_metrics[name].detach().cpu().reshape(-1) for result in results]
        )
        for name in sorted(keys)
    }
    metrics = {
        name: float(values.to(dtype=torch.float64).mean().item())
        for name, values in path_metrics.items()
    }
    empty = torch.empty(0)
    return ShadowingResult(
        selected_indices=empty.to(dtype=torch.long),
        prefix_distances=empty,
        shadow_mean=empty,
        lower_50=empty,
        upper_50=empty,
        lower_90=empty,
        upper_90=empty,
        metrics=metrics,
        path_metrics=path_metrics,
    )


def main() -> None:
    args = parse_args()
    test_seeds = _validate_args(args)
    run_dir = ensure_run_dir(args.run_dir)
    (
        _source_model,
        template_model,
        _training,
        _stored_config,
        heston,
        _target_paths,
        _heldout_paths,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    current_model = copy.deepcopy(template_model)
    coverage_model = copy.deepcopy(template_model)
    current_model.load_checkpoint_state(
        _load_checkpoint(Path(args.selected_checkpoint))
    )
    coverage_model.load_checkpoint_state(
        _load_checkpoint(Path(args.coverage_checkpoint))
    )

    laws = ("selected_specific_entropy", "step_1600", "heston_oracle")
    shadow_results: dict[str, list[ShadowingResult]] = {name: [] for name in laws}
    shadow_rows: dict[str, list[dict[str, float | str]]] = {
        name: [] for name in laws
    }
    stylized_rows: dict[str, list[dict[str, float | str]]] = {
        name: [] for name in laws
    }
    per_seed: dict[str, object] = {}
    per_path: dict[str, object] = {}
    device = torch.device(args.device)
    for index, reference_seed in enumerate(test_seeds):
        model_seed = int(args.model_bank_seed_base) + index
        oracle_seed = int(args.oracle_bank_seed_base) + index
        print(
            f"seed {reference_seed}: simulating fresh Heston reference and three banks",
            flush=True,
        )
        reference = _simulate_heston_bank(
            num_paths=int(args.reference_paths),
            batch_size=int(args.bank_batch_size),
            config=heston,
            seed=int(reference_seed),
            device=device,
            dtype=template_model.dtype,
        )
        current_bank = _sample_bank(
            current_model,
            num_paths=int(args.bank_paths),
            seed=model_seed,
            x0=float(heston.x0),
            batch_size=int(args.bank_batch_size),
        )
        coverage_bank = _sample_bank(
            coverage_model,
            num_paths=int(args.bank_paths),
            seed=model_seed,
            x0=float(heston.x0),
            batch_size=int(args.bank_batch_size),
        )
        oracle_bank = _simulate_heston_bank(
            num_paths=int(args.bank_paths),
            batch_size=int(args.bank_batch_size),
            config=heston,
            seed=oracle_seed,
            device=device,
            dtype=template_model.dtype,
        )
        banks = {
            "selected_specific_entropy": current_bank,
            "step_1600": coverage_bank,
            "heston_oracle": oracle_bank,
        }
        current_results = {
            name: _shadow(
                real_paths=reference[: int(args.real_paths)],
                bank=bank,
                args=args,
            )
            for name, bank in banks.items()
        }
        current_shadow_rows = {
            name: _metric_subset(result)
            for name, result in current_results.items()
        }
        current_stylized_rows = {
            name: _stylized(bank, reference, args=args)
            for name, bank in banks.items()
        }
        for name in laws:
            shadow_results[name].append(current_results[name])
            shadow_rows[name].append(current_shadow_rows[name])
            stylized_rows[name].append(current_stylized_rows[name])
        decision = coverage_focused_checkpoint_rule(
            candidate=current_shadow_rows["step_1600"],
            baseline=current_shadow_rows["selected_specific_entropy"],
        )
        seed_key = str(reference_seed)
        bootstrap_seed = int(args.bootstrap_seed) + 1000 * index
        per_seed[seed_key] = {
            "reference_seed": int(reference_seed),
            "model_bank_seed": model_seed,
            "oracle_bank_seed": oracle_seed,
            "shadowing": current_shadow_rows,
            "stylized": current_stylized_rows,
            "coverage_focused_rule": decision,
            "bootstrap": {
                name: _bootstrap(
                    result,
                    args=args,
                    seed=bootstrap_seed + law_index,
                )
                for law_index, (name, result) in enumerate(current_results.items())
            },
            "paired_bootstrap": {
                "step_1600_minus_selected_specific_entropy": _paired(
                    current_results["step_1600"],
                    current_results["selected_specific_entropy"],
                    args=args,
                    seed=bootstrap_seed + 100,
                ),
                "step_1600_minus_heston_oracle": _paired(
                    current_results["step_1600"],
                    current_results["heston_oracle"],
                    args=args,
                    seed=bootstrap_seed + 200,
                ),
            },
        }
        per_path[seed_key] = {
            name: {
                metric: values.detach().cpu()
                for metric, values in result.path_metrics.items()
            }
            for name, result in current_results.items()
        }
        if bool(args.save_banks):
            seed_dir = ensure_run_dir(run_dir / f"seed_{reference_seed}")
            torch.save(reference, seed_dir / "reference_paths.pt")
            for name, bank in banks.items():
                torch.save(bank, seed_dir / f"{name}_bank.pt")
        print(
            "  coverage selected={selected:.4f} step1600={candidate:.4f} "
            "oracle={oracle:.4f}; rv_crps {selected_crps:.6f}->{candidate_crps:.6f}; "
            "rule={passed}".format(
                selected=float(
                    current_shadow_rows["selected_specific_entropy"][
                        "realized_volatility_coverage_90"
                    ]
                ),
                candidate=float(
                    current_shadow_rows["step_1600"][
                        "realized_volatility_coverage_90"
                    ]
                ),
                oracle=float(
                    current_shadow_rows["heston_oracle"][
                        "realized_volatility_coverage_90"
                    ]
                ),
                selected_crps=float(
                    current_shadow_rows["selected_specific_entropy"][
                        "realized_volatility_crps"
                    ]
                ),
                candidate_crps=float(
                    current_shadow_rows["step_1600"][
                        "realized_volatility_crps"
                    ]
                ),
                passed=bool(decision["pass"]),
            ),
            flush=True,
        )

    pooled = {
        name: _pooled_result(results) for name, results in shadow_results.items()
    }
    pooled_rule = coverage_focused_checkpoint_rule(
        candidate=pooled["step_1600"].metrics,
        baseline=pooled["selected_specific_entropy"].metrics,
    )
    per_seed_passes = [
        bool(dict(per_seed[str(seed)])["coverage_focused_rule"]["pass"])
        for seed in test_seeds
    ]
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "protocol": {
            "candidate_frozen_before_tests": True,
            "checkpoint_selection_uses_fresh_tests": False,
            "fresh_test_seeds_predeclared": True,
            "coverage_focused_rule": (
                "step 1600 must move 90% RV coverage strictly closer to 90%, "
                "not worsen RV CRPS, and not worsen cumulative-return CRPS"
            ),
            "stylized_facts_are_diagnostic_only": True,
            "common_random_numbers_between_mp_checkpoints": True,
            "heston_oracle_bank_is_independent": True,
        },
        "config": {
            "source_run_dir": str(args.source_run_dir),
            "coverage_checkpoint": str(args.coverage_checkpoint),
            "selected_checkpoint": str(args.selected_checkpoint),
            "test_seeds": list(test_seeds),
            "model_bank_seed_base": int(args.model_bank_seed_base),
            "oracle_bank_seed_base": int(args.oracle_bank_seed_base),
            "reference_paths": int(args.reference_paths),
            "bank_paths": int(args.bank_paths),
            "real_paths": int(args.real_paths),
            "prefix_length": int(args.split_step) + 1,
            "future_steps": int(args.future_steps),
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "swd_projections": int(args.swd_projections),
        },
        "per_seed": per_seed,
        "aggregate": {
            "shadowing": {
                name: summarize_metric_rows(rows)
                for name, rows in shadow_rows.items()
            },
            "stylized": {
                name: summarize_metric_rows(rows)
                for name, rows in stylized_rows.items()
            },
            "pooled_shadowing_metrics": {
                name: dict(result.metrics) for name, result in pooled.items()
            },
            "pooled_coverage_focused_rule": pooled_rule,
            "per_seed_pass_count": int(sum(per_seed_passes)),
            "per_seed_total": len(per_seed_passes),
            "all_seed_point_rule_pass": bool(all(per_seed_passes)),
            "pooled_bootstrap": {
                name: _bootstrap(
                    result,
                    args=args,
                    seed=int(args.bootstrap_seed) + 100_000 + index,
                )
                for index, (name, result) in enumerate(pooled.items())
            },
            "pooled_paired_bootstrap": {
                "step_1600_minus_selected_specific_entropy": _paired(
                    pooled["step_1600"],
                    pooled["selected_specific_entropy"],
                    args=args,
                    seed=int(args.bootstrap_seed) + 110_000,
                ),
                "step_1600_minus_heston_oracle": _paired(
                    pooled["step_1600"],
                    pooled["heston_oracle"],
                    args=args,
                    seed=int(args.bootstrap_seed) + 120_000,
                ),
            },
        },
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(per_path, run_dir / "shadowing_per_path_metrics.pt")
    print(
        "pooled coverage selected={selected:.4f} step1600={candidate:.4f} "
        "oracle={oracle:.4f}; seed passes={passes}/{total}; pooled rule={pooled_pass}".format(
            selected=float(
                pooled["selected_specific_entropy"].metrics[
                    "realized_volatility_coverage_90"
                ]
            ),
            candidate=float(
                pooled["step_1600"].metrics["realized_volatility_coverage_90"]
            ),
            oracle=float(
                pooled["heston_oracle"].metrics["realized_volatility_coverage_90"]
            ),
            passes=sum(per_seed_passes),
            total=len(per_seed_passes),
            pooled_pass=bool(pooled_rule["pass"]),
        ),
        flush=True,
    )
    print(f"wrote locked comparison to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
