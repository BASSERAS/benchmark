#!/usr/bin/env python3
"""Aggregate the locked three-dimensional volatility-only R operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.runners import write_json  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    jacobian_spectrum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", nargs="+", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        json.loads((path.resolve() / "report.json").read_text(encoding="utf-8"))
        for path in args.replicates
    ]
    secondary_report = json.loads(
        (args.secondary.resolve() / "report.json").read_text(encoding="utf-8")
    )
    if len(reports) != 3:
        raise ValueError("the locked protocol requires exactly three primary matrices")
    matrices = torch.tensor([row["matrix"] for row in reports], dtype=torch.float64)
    secondary = torch.tensor(secondary_report["matrix"], dtype=torch.float64)
    mean = matrices.mean(dim=0)
    sd = matrices.std(dim=0, unbiased=True)
    radii = torch.tensor(
        [row["spectrum"]["spectral_radius_T"] for row in reports],
        dtype=torch.float64,
    )
    eps = torch.finfo(torch.float64).eps
    matrix_noise = float(
        torch.linalg.vector_norm(sd)
        / torch.linalg.vector_norm(mean).clamp_min(eps)
    )
    radius_cv = float(radii.std(unbiased=True) / radii.mean().abs().clamp_min(eps))
    step_change = float(
        torch.linalg.vector_norm(matrices[0] - secondary)
        / torch.linalg.vector_norm(matrices[0]).clamp_min(eps)
    )
    contractive = bool(torch.all(radii < 1.0).item())
    primary_reproducible = matrix_noise <= 0.10 and radius_cv <= 0.10
    route = (
        "ordinary_mp_update"
        if contractive and primary_reproducible
        else "one_residual_newton_cycle"
        if (not contractive) and primary_reproducible
        else "stop"
    )
    report = {
        "protocol": {
            "operator": "three-dimensional early/middle/late R-to-R",
            "alpha_identically_zero": True,
            "primary_reproducibility_threshold": 0.10,
            "half_step_is_diagnostic_not_used_for_newton_matrix": True,
            "test_paths_used": False,
        },
        "replicate_paths": [str(path.resolve()) for path in args.replicates],
        "secondary_path": str(args.secondary.resolve()),
        "matrices": matrices.tolist(),
        "mean_matrix": mean.tolist(),
        "elementwise_sd": sd.tolist(),
        "replicate_spectral_radius": {
            "values": radii.tolist(),
            "mean": float(radii.mean().item()),
            "sd": float(radii.std(unbiased=True).item()),
            "coefficient_of_variation": radius_cv,
        },
        "mean_spectrum": jacobian_spectrum(mean),
        "replication_noise_frobenius_ratio": matrix_noise,
        "secondary_matrix": secondary.tolist(),
        "secondary_spectrum": secondary_report["spectrum"],
        "finite_difference_relative_change": step_change,
        "gate": {
            "contractive": contractive,
            "primary_reproducible": primary_reproducible,
            "passed_to_terminal_cycle": route != "stop",
            "next_phase": route,
        },
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "report.json", report)
    torch.save(
        {"matrices": matrices, "secondary_matrix": secondary},
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Volatility-only three-block R spectrum\n\n"
        f"- Replicate radii: `{radii.tolist()}`\n"
        f"- Mean radius: `{radii.mean().item():.6g}`\n"
        f"- Primary matrix-noise ratio: `{matrix_noise:.6g}`\n"
        f"- Radius coefficient of variation: `{radius_cv:.6g}`\n"
        f"- Half-step matrix change: `{step_change:.6g}`\n"
        f"- Contractive: `{contractive}`\n"
        f"- Next phase: `{route}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
