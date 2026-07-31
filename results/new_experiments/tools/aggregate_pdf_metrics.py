"""Aggregate the protocol evaluator's per-seed JSONs into the README's PDF-metrics table.

Method-neutral: works for Experiment A (``evaluate_drawdown_memory.py``) and Experiment B
(``evaluate_heston_parameter_mixture.py``) because it flattens every numeric leaf to a dotted
path and aggregates whatever it finds. No per-experiment key list to keep in sync.

Aggregation follows the protocol: mean +/- sample std (ddof=1) over 5 seeds, 95% CI with
t_{0.975,4} = 2.776. Seeds whose file is missing are reported, never silently dropped.

Usage:
  python aggregate_pdf_metrics.py \
      --model-dir ../experiment_A/LS4 --floor-dir ../experiment_A/perfect_floor \
      --pattern '*_drawdown_memory.json'
"""
import os
import json
import glob
import argparse
import numpy as np

T_CRIT_4DOF = 2.776


def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix.rstrip(".")] = float(node)
    return out


def collect(directory, pattern):
    paths = sorted(glob.glob(os.path.join(directory, "pdf_metrics", pattern)))
    if not paths:
        raise SystemExit(f"no evaluator JSON matching {pattern} under {directory}/pdf_metrics")
    series, seeds = {}, []
    for p in paths:
        seeds.append(os.path.basename(p))
        for k, v in flatten(json.load(open(p))).items():
            series.setdefault(k, []).append(v)
    return series, seeds


def stats(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    half = T_CRIT_4DOF * std / np.sqrt(a.size) if a.size > 1 else 0.0
    return mean, std, half


def fmt(mean, std):
    return f"{mean:.6g} ± {std:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="method dir containing pdf_metrics/")
    ap.add_argument("--floor-dir", default=None, help="perfect_floor dir containing pdf_metrics/")
    ap.add_argument("--pattern", default="*_drawdown_memory.json")
    ap.add_argument("--label", default="Model")
    ap.add_argument("--exclude-prefix", nargs="*", default=[],
                    help="dotted-path prefixes to omit, e.g. configuration oracle_gate")
    a = ap.parse_args()

    model, model_files = collect(a.model_dir, a.pattern)
    floor, floor_files = (collect(a.floor_dir, a.pattern) if a.floor_dir else ({}, []))

    print(f"<!-- model seeds: {len(model_files)} | floor seeds: {len(floor_files)} -->")
    for name, series in (("model", model), ("floor", floor)):
        short = {k: len(v) for k, v in series.items() if len(v) != 5}
        if short:
            print(f"<!-- WARNING {name}: keys with != 5 seeds: {short} -->")

    header = f"| Metric | {a.label} (mean ± std) | 95% CI half-width |"
    align = "|---|---|---|"
    if floor:
        header += " Perfect floor (mean ± std) |"
        align += "---|"
    print(header)
    print(align)

    for key in sorted(set(model) | set(floor)):
        if any(key.startswith(p + ".") or key == p for p in a.exclude_prefix):
            continue
        row = f"| `{key}` |"
        if key in model:
            m, s, h = stats(model[key])
            row += f" {fmt(m, s)} | {h:.3g} |"
        else:
            row += " n/a | n/a |"
        if floor:
            if key in floor:
                fm, fs, _ = stats(floor[key])
                row += f" {fmt(fm, fs)} |"
            else:
                row += " n/a |"
        print(row)


if __name__ == "__main__":
    main()
