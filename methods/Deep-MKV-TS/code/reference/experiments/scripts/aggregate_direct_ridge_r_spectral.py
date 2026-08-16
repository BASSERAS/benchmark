#!/usr/bin/env python3
"""Aggregate replicated direct-ridge R reduced-Jacobian audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.terminal_fixed_point_diagnostic import jacobian_spectrum  # noqa: E402
from path_dt_experiments.runners import write_json  # noqa: E402


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
    eps = torch.finfo(torch.float64).eps
    eigenvalues, eigenvectors = torch.linalg.eig(mean)
    dominant_index = int(torch.argmax(torch.abs(eigenvalues)).item())
    dominant_vector = eigenvectors[:, dominant_index]
    dominant_energy = dominant_vector.abs().pow(2)
    dominant_energy = dominant_energy / dominant_energy.sum().clamp_min(eps)
    report = {
        "protocol": {
            "replicate_count": len(reports),
            "common_primary_step": reports[0]["protocol"]["finite_difference_step"],
            "secondary_step": secondary_report["protocol"]["finite_difference_step"],
            "direct_ridge_R_refitted_under_each_candidate_law": True,
            "test_paths_used": False,
        },
        "replicate_paths": [str(path.resolve()) for path in args.replicates],
        "secondary_path": str(args.secondary.resolve()),
        "replicate_spectra": [report["spectrum"] for report in reports],
        "replicate_spectral_radius": {
            "values": radii.tolist(),
            "mean": float(radii.mean().item()),
            "sd": float(radii.std(unbiased=True).item()),
            "minimum": float(radii.min().item()),
            "maximum": float(radii.max().item()),
        },
        "mean_matrix": mean.tolist(),
        "elementwise_sd": sd.tolist(),
        "mean_spectrum": jacobian_spectrum(mean),
        "dominant_mode": {
            "eigenvalue": {
                "real": float(eigenvalues[dominant_index].real.item()),
                "imag": float(eigenvalues[dominant_index].imag.item()),
            },
            "coordinate_energy": dominant_energy.tolist(),
            "p_block_energy": float(dominant_energy[:3].sum().item()),
            "r_block_energy": float(dominant_energy[3:].sum().item()),
        },
        "replication_noise_frobenius_ratio": float(
            torch.linalg.vector_norm(sd)
            / torch.linalg.vector_norm(mean).clamp_min(eps)
        ),
        "secondary_matrix": secondary.tolist(),
        "secondary_spectrum": secondary_report["spectrum"],
        "finite_difference_relative_change": float(
            torch.linalg.vector_norm(matrices[0] - secondary)
            / torch.linalg.vector_norm(matrices[0]).clamp_min(eps)
        ),
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "report.json", report)
    torch.save(
        {"matrices": matrices, "secondary_matrix": secondary},
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Direct-ridge R reduced spectral audit\n\n"
        f"- Replicate radii: `{radii.tolist()}`\n"
        f"- Mean radius: `{radii.mean().item():.6g}`\n"
        f"- Mean-matrix radius: `{report['mean_spectrum']['spectral_radius_T']:.6g}`\n"
        f"- Dominant-mode P/R energy: `{report['dominant_mode']['p_block_energy']:.3%}` / "
        f"`{report['dominant_mode']['r_block_energy']:.3%}`\n"
        f"- Replication-noise ratio: `{report['replication_noise_frobenius_ratio']:.6g}`\n"
        f"- Step-halving change: `{report['finite_difference_relative_change']:.6g}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["replicate_spectral_radius"], indent=2))


if __name__ == "__main__":
    main()
