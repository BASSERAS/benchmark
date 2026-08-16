from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.plots import plot_volatility_diagnostic_summary  # noqa: E402
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.volatility_diagnostics import (  # noqa: E402
    build_volatility_diagnostic_report,
    volatility_path_statistics,
    volatility_statistics_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose unconditional and prefix-conditioned volatility in named path banks."
    )
    parser.add_argument("--reference-paths", type=Path, required=True)
    parser.add_argument("--reference-label", type=str, default="reference")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named candidate tensor; repeat for multiple generated banks.",
    )
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 20, 32))
    parser.add_argument("--lags", type=int, nargs="+", default=(1, 2, 5, 10, 20))
    parser.add_argument("--asset", type=int, default=0)
    parser.add_argument("--max-reference-paths", type=int, default=None)
    parser.add_argument("--max-candidate-paths", type=int, default=100_000)
    parser.add_argument("--subsample-seed", type=int, default=24001306)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"candidate must have the form LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", maxsplit=1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"candidate must have the form LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def _load_path_subset(path: Path, *, max_paths: int | None, seed: int) -> tuple[torch.Tensor, int]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{path} must contain a tensor with shape (B, N + 1, d)")
    original_count = int(value.shape[0])
    if max_paths is not None:
        if int(max_paths) < 1:
            raise ValueError("path subsample limits must be >= 1")
        if original_count > int(max_paths):
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            indices = torch.randperm(original_count, generator=generator)[: int(max_paths)]
            value = value.index_select(0, indices)
    if not torch.is_floating_point(value):
        value = value.float()
    return value, original_count


def _format_ratio(value: object) -> str:
    return "undefined" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    candidates = [_parse_candidate(value) for value in args.candidate]
    labels = [str(args.reference_label), *(label for label, _ in candidates)]
    if len(set(labels)) != len(labels):
        raise ValueError("reference and candidate labels must be unique")

    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="volatility_diagnostics"))
    reference_paths, reference_original_count = _load_path_subset(
        args.reference_paths,
        max_paths=args.max_reference_paths,
        seed=int(args.subsample_seed),
    )
    common_shape = tuple(reference_paths.shape[1:])
    statistics_by_label = {
        str(args.reference_label): volatility_path_statistics(
            reference_paths,
            prefix_length=int(args.prefix_length),
            prefix_window=int(args.prefix_window),
            horizons=args.horizons,
            lags=args.lags,
            asset=int(args.asset),
        )
    }
    sources: dict[str, object] = {
        str(args.reference_label): {
            "path": str(args.reference_paths),
            "original_num_paths": reference_original_count,
            "evaluated_num_paths": int(reference_paths.shape[0]),
        }
    }
    del reference_paths

    for candidate_number, (label, path) in enumerate(candidates, start=1):
        paths, original_count = _load_path_subset(
            path,
            max_paths=int(args.max_candidate_paths),
            seed=int(args.subsample_seed) + candidate_number,
        )
        if tuple(paths.shape[1:]) != common_shape:
            raise ValueError(f"{label} path shape {tuple(paths.shape[1:])} does not match {common_shape}")
        statistics_by_label[label] = volatility_path_statistics(
            paths,
            prefix_length=int(args.prefix_length),
            prefix_window=int(args.prefix_window),
            horizons=args.horizons,
            lags=args.lags,
            asset=int(args.asset),
        )
        sources[label] = {
            "path": str(path),
            "original_num_paths": original_count,
            "evaluated_num_paths": int(paths.shape[0]),
        }
        del paths

    report = build_volatility_diagnostic_report(
        statistics_by_label,
        reference_label=str(args.reference_label),
    )
    report["sources"] = sources
    report["subsample_seed"] = int(args.subsample_seed)
    report["run_dir"] = str(run_dir)
    write_json(run_dir / "volatility_diagnostics.json", report)
    torch.save(
        volatility_statistics_artifact(statistics_by_label),
        run_dir / "volatility_path_statistics.pt",
    )
    if not bool(args.no_plot):
        figure = plot_volatility_diagnostic_summary(
            statistics_by_label=statistics_by_label,
            reference_label=str(args.reference_label),
            output_path=run_dir / "volatility_diagnostics.png",
        )
        import matplotlib.pyplot as plt

        plt.close(figure)

    largest_horizon = str(max(int(value) for value in args.horizons))
    print(f"wrote volatility diagnostics to {run_dir}")
    for label, comparison in report["comparisons_to_reference"].items():
        unconditional = comparison["future_realized_volatility"][largest_horizon]
        dataset = report["datasets"][label]
        persistence = dataset["persistence"][largest_horizon]
        print(
            f"{label}: h={largest_horizon} std_ratio="
            f"{_format_ratio(unconditional['candidate_to_reference_std_ratio'])} "
            f"iqr_ratio={_format_ratio(unconditional['candidate_to_reference_iqr_ratio'])} "
            f"abs_acf_rmse={comparison['absolute_return_acf_error_rms']:.4f} "
            f"prefix_future_corr={persistence['prefix_future_correlation']:.4f}"
        )


if __name__ == "__main__":
    main()
