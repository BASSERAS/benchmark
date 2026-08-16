#!/usr/bin/env python3
"""Aggregate independently executed first-operator objective variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.residual_volatility_gate import (  # noqa: E402
    residual_phase_one_gate,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


VARIANTS = (
    "current_objective_eta1",
    "residual_conditional_eta1",
    "residual_conditional_eta4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partial-run-dirs", nargs=3, type=Path, required=True)
    return parser.parse_args()


def _summary(report: dict[str, object]) -> str:
    lines = [
        "# Residual-volatility first-operator audit",
        "",
        f"- Gate passed: `{report['phase_one_gate']['passed']}`",
        f"- Passing variants: `{', '.join(report['phase_one_gate']['passing_variants']) or 'none'}`",
        f"- Next phase: `{report['phase_one_gate']['next_phase']}`",
        "",
        "| Bank | Variant | P CE rel. | R CE rel. | R explained | log-vol move | upper bound | target-control log-vol |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in ("fit", "confirmation"):
        for name in VARIANTS:
            row = report["audit"][name][bank]
            lines.append(
                f"| {bank} | {name} | {row['p_ce_relative_residual']:.3f} | "
                f"{row['r_ce_relative_residual']:.3f} | "
                f"{row['r_ce_explained_energy']:.3f} | "
                f"{row['log_volatility_move_rms']:.4g} | "
                f"{row['induced_sigma_upper_activity']:.2%} | "
                f"{row['fitted_target_log_volatility_consensus_rms']:.4g} |"
            )
    lines.extend(["", "## Locked checks", ""])
    for name, details in report["phase_one_gate"]["variants"].items():
        lines.append(f"### {name}: `{details['passed']}`")
        lines.append("")
        for check, passed in details["checks"].items():
            lines.append(f"- {check}: `{passed}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output = ensure_run_dir(args.run_dir.resolve())
    partials = []
    for directory in args.partial_run_dirs:
        path = directory.resolve() / "report.json"
        partials.append((directory.resolve(), json.loads(path.read_text())))
    found = {}
    for directory, report in partials:
        audit = report["audit"]
        if len(audit) != 1:
            raise ValueError(f"partial run must contain one variant: {directory}")
        name = next(iter(audit))
        if name in found:
            raise ValueError(f"duplicate partial variant: {name}")
        found[name] = (directory, report)
    if set(found) != set(VARIANTS):
        raise ValueError(f"expected variants {VARIANTS}, found {tuple(found)}")
    baseline = found[VARIANTS[0]][1]

    def common_settings(report: dict[str, object]) -> dict[str, object]:
        settings = json.loads(json.dumps(report["effective_settings"]))
        cli = settings["cli"]
        for name in ("device", "run_dir", "variants"):
            cli.pop(name, None)
        return settings

    for name in VARIANTS[1:]:
        report = found[name][1]
        for field in (
            "protocol",
            "common_initial_network_sha256",
            "scenario_banks",
            "dataset_fingerprint",
        ):
            if report[field] != baseline[field]:
                raise RuntimeError(f"partial runs disagree on {field}")
        if common_settings(report) != common_settings(baseline):
            raise RuntimeError("partial runs disagree on effective settings")
    audit = {name: found[name][1]["audit"][name] for name in VARIANTS}
    combined = {
        "protocol": baseline["protocol"],
        "objective_design": {
            name: found[name][1]["objective_design"][name] for name in VARIANTS
        },
        "common_initial_network_sha256": baseline[
            "common_initial_network_sha256"
        ],
        "effective_settings": baseline["effective_settings"],
        "scenario_banks": baseline["scenario_banks"],
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "law_fingerprints": {
            name: found[name][1]["law_fingerprints"][name] for name in VARIANTS
        },
        "partial_run_directories": {
            name: str(found[name][0]) for name in VARIANTS
        },
        "audit": audit,
        "phase_one_gate": residual_phase_one_gate(audit),
    }
    if len(
        {
            json.dumps(value, sort_keys=True)
            for value in combined["law_fingerprints"].values()
        }
    ) != 1:
        raise RuntimeError("partial runs did not use the same exact reference law")
    for bank in ("fit", "confirmation"):
        fingerprints = {
            combined["audit"][name][bank]["bank_fingerprint"]
            for name in VARIANTS
        }
        datasets = {
            combined["audit"][name][bank]["dataset_fingerprint"]
            for name in VARIANTS
        }
        if len(fingerprints) != 1 or len(datasets) != 1:
            raise RuntimeError(
                f"partial runs disagree on evaluated {bank} bank or dataset"
            )
    write_json(output / "report.json", combined)
    (output / "SUMMARY.md").write_text(_summary(combined), encoding="utf-8")
    print(json.dumps(combined["phase_one_gate"], indent=2))


if __name__ == "__main__":
    main()
