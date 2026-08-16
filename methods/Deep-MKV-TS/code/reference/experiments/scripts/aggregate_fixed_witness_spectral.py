#!/usr/bin/env python3
"""Aggregate the locked fixed-witness spectrum and apply its no-go gate."""

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


DEFAULT_DYNAMIC = REPO_ROOT / "runs" / "direct_ridge_r_spectral_20260803"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", nargs="+", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--dynamic-run-dir", type=Path, default=DEFAULT_DYNAMIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        json.loads((path.resolve() / "report.json").read_text(encoding="utf-8"))
        for path in args.replicates
    ]
    if len(reports) < 3:
        raise ValueError("the locked audit requires three primary replicates")
    secondary_report = json.loads(
        (args.secondary.resolve() / "report.json").read_text(encoding="utf-8")
    )
    dynamic = json.loads(
        (args.dynamic_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    matrices = torch.tensor(
        [report["matrix"] for report in reports], dtype=torch.float64
    )
    secondary = torch.tensor(secondary_report["matrix"], dtype=torch.float64)
    mean = matrices.mean(dim=0)
    sd = matrices.std(dim=0, unbiased=True)
    radii = torch.tensor(
        [report["spectrum"]["spectral_radius_T"] for report in reports],
        dtype=torch.float64,
    )
    mean_spectrum = jacobian_spectrum(mean)
    eps = torch.finfo(torch.float64).eps
    replication_noise = float(
        torch.linalg.vector_norm(sd)
        / torch.linalg.vector_norm(mean).clamp_min(eps)
    )
    step_change = float(
        torch.linalg.vector_norm(matrices[0] - secondary)
        / torch.linalg.vector_norm(matrices[0]).clamp_min(eps)
    )
    dynamic_mean = float(dynamic["replicate_spectral_radius"]["mean"])
    dynamic_minimum = float(dynamic["replicate_spectral_radius"]["minimum"])
    dynamic_mean_matrix = float(dynamic["mean_spectrum"]["spectral_radius_T"])
    dynamic_matrix = torch.tensor(dynamic["mean_matrix"], dtype=torch.float64)
    relative_matrix_change = float(
        torch.linalg.vector_norm(mean - dynamic_matrix)
        / torch.linalg.vector_norm(dynamic_matrix).clamp_min(eps)
    )
    column_change = [
        float(
            torch.linalg.vector_norm(mean[:, index] - dynamic_matrix[:, index])
            / torch.linalg.vector_norm(dynamic_matrix[:, index]).clamp_min(eps)
        )
        for index in range(int(mean.shape[1]))
    ]
    radius_reduction = 1.0 - float(radii.mean().item()) / dynamic_mean
    mean_matrix_reduction = (
        1.0 - float(mean_spectrum["spectral_radius_T"]) / dynamic_mean_matrix
    )
    coefficient_of_variation = float(
        radii.std(unbiased=True) / radii.mean().abs().clamp_min(eps)
    )

    # These thresholds are locked before reading the fixed-witness matrices.
    # The witness must remove at least half of the stiffness, improve every
    # independent replicate, and leave a numerically reproducible derivative.
    criteria = {
        "mean_radius_reduced_by_at_least_50_percent": radius_reduction >= 0.50,
        "mean_matrix_radius_reduced_by_at_least_50_percent": mean_matrix_reduction
        >= 0.50,
        "every_fixed_witness_radius_below_dynamic_minimum": bool(
            torch.all(radii < dynamic_minimum).item()
        ),
        "replicate_radius_cv_at_most_5_percent": coefficient_of_variation <= 0.05,
        "replication_matrix_noise_at_most_10_percent": replication_noise <= 0.10,
        "step_halving_matrix_change_at_most_25_percent": step_change <= 0.25,
    }
    passed = all(criteria.values())
    eigenvalues, eigenvectors = torch.linalg.eig(mean)
    dominant_index = int(torch.argmax(torch.abs(eigenvalues)).item())
    energy = eigenvectors[:, dominant_index].abs().pow(2)
    energy = energy / energy.sum().clamp_min(eps)
    report = {
        "protocol": {
            "replicate_count": len(reports),
            "common_primary_step": reports[0]["protocol"]["finite_difference_step"],
            "secondary_step": secondary_report["protocol"]["finite_difference_step"],
            "residual_population_law_source_regeneration_disabled": True,
            "all_other_mp_sources_refreshed": True,
            "direct_ridge_R_refitted_under_each_candidate_law": True,
            "thresholds_locked_before_fixed_witness_results": True,
            "test_paths_used": False,
        },
        "replicate_paths": [str(path.resolve()) for path in args.replicates],
        "secondary_path": str(args.secondary.resolve()),
        "dynamic_comparator": str(args.dynamic_run_dir.resolve()),
        "replicate_spectra": [report["spectrum"] for report in reports],
        "replicate_spectral_radius": {
            "values": radii.tolist(),
            "mean": float(radii.mean().item()),
            "sd": float(radii.std(unbiased=True).item()),
            "minimum": float(radii.min().item()),
            "maximum": float(radii.max().item()),
            "coefficient_of_variation": coefficient_of_variation,
        },
        "mean_matrix": mean.tolist(),
        "elementwise_sd": sd.tolist(),
        "mean_spectrum": mean_spectrum,
        "dominant_mode": {
            "eigenvalue": {
                "real": float(eigenvalues[dominant_index].real.item()),
                "imag": float(eigenvalues[dominant_index].imag.item()),
            },
            "coordinate_energy": energy.tolist(),
            "p_block_energy": float(energy[:3].sum().item()),
            "r_block_energy": float(energy[3:].sum().item()),
        },
        "replication_noise_frobenius_ratio": replication_noise,
        "secondary_matrix": secondary.tolist(),
        "secondary_spectrum": secondary_report["spectrum"],
        "finite_difference_relative_change": step_change,
        "comparison": {
            "dynamic_mean_radius": dynamic_mean,
            "fixed_witness_mean_radius": float(radii.mean().item()),
            "mean_radius_reduction_fraction": radius_reduction,
            "dynamic_mean_matrix_radius": dynamic_mean_matrix,
            "fixed_witness_mean_matrix_radius": float(
                mean_spectrum["spectral_radius_T"]
            ),
            "mean_matrix_radius_reduction_fraction": mean_matrix_reduction,
            "relative_mean_matrix_change": relative_matrix_change,
            "relative_column_changes_P_then_R": column_change,
        },
        "gate": {
            "criteria": criteria,
            "passed": passed,
            "next_phase": (
                "alternating_primal_dual_witness_probe"
                if passed
                else "stop_dual_witness_solver_branch"
            ),
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
        "# Fixed dual-witness reduced spectral audit\n\n"
        f"- Gate passed: `{passed}`\n"
        f"- Dynamic / fixed-witness mean radius: `{dynamic_mean:.6g}` / "
        f"`{radii.mean().item():.6g}`\n"
        f"- Mean-radius reduction: `{radius_reduction:.2%}`\n"
        f"- Fixed-witness replicate radii: `{radii.tolist()}`\n"
        f"- Dominant-mode P/R energy: `{energy[:3].sum().item():.3%}` / "
        f"`{energy[3:].sum().item():.3%}`\n"
        f"- Replication-noise ratio: `{replication_noise:.6g}`\n"
        f"- Step-halving change: `{step_change:.6g}`\n"
        f"- Relative changes in P columns: "
        f"`{[round(value, 4) for value in column_change[:3]]}`\n"
        f"- Relative changes in R columns: "
        f"`{[round(value, 4) for value in column_change[3:]]}`\n"
        f"- Next phase: `{report['gate']['next_phase']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
