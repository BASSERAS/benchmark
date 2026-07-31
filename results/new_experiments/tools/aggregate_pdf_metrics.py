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
import re
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


def collect(directory, pattern, subdir="pdf_metrics"):
    paths = sorted(glob.glob(os.path.join(directory, subdir, pattern)))
    if not paths:
        raise SystemExit(f"no evaluator JSON matching {pattern} under {directory}/{subdir}")
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


def true_seeds(directory, pattern, subdir="pdf_metrics"):
    """The TRUE generating seeds, read from each JSON's `sources.generated` path.

    Two things this deliberately does not do:

    * It does not use the filename. `perfect_floor/pdf_metrics/` names its files
      seed_0..seed_4 while the banks behind them are floor_seed1000..floor_seed1004, so a
      filename-based check calls that pair "aligned" and emits a paired interval over two
      unrelated seed sets. Measured: that is exactly what the first version of this guard did.
    * It does not go through `flatten()`, which keeps only numeric leaves and therefore drops
      `sources.generated` -- a string -- entirely.
    """
    seeds = []
    for p in sorted(glob.glob(os.path.join(directory, subdir, pattern))):
        gen = str(json.load(open(p)).get("sources", {}).get("generated", ""))
        hits = re.findall(r"seed_?(\d+)", gen)
        seeds.append(int(hits[-1]) if hits else None)
    return seeds


def paired(a):
    """PDF §5 paired comparison: five seedwise differences + the paired interval.

    The pairing is by SEED, so seed q of one method is differenced against seed q of the
    other. Averaging each method separately and subtracting the means throws away exactly the
    seed-to-seed covariance the paired interval exists to exploit -- the two means agree, but
    the interval does not, and it is usually much too wide. That is the whole point of the
    requirement.
    """
    left, lf = collect(a.model_dir, a.pattern, a.subdir)
    right, rf = collect(a.paired_dir, a.pattern, a.subdir)

    # Alignment is a precondition, not a nicety -- and FILENAMES ARE NOT THE TEST.
    # The perfect_floor writes seed_0..seed_4_<stem>.json for DGP seeds 1000..1004, so
    # comparing basenames says "aligned" for a comparison that is nothing of the kind.
    # The true seed is in sources.generated: ".../seed_0/..." vs ".../floor_seed1000.npy".
    lseeds = true_seeds(a.model_dir, a.pattern, a.subdir)
    rseeds = true_seeds(a.paired_dir, a.pattern, a.subdir)
    if lseeds != rseeds or None in lseeds:
        raise SystemExit(
            "REFUSING to pair: seeds are not aligned.\n"
            f"  {a.model_dir}: seeds {lseeds}  (files {lf})\n"
            f"  {a.paired_dir}: seeds {rseeds}  (files {rf})\n"
            "PDF 5 requires the paired interval only 'for comparisons between models run "
            "with aligned seeds'. Model-vs-perfect-floor is NOT one of those -- the floor is "
            "drawn at seeds 1000-1004 -- even though its files are named seed_0..seed_4. "
            "Report the unpaired table there, and say in prose why it is unpaired."
        )

    n = len(lf)
    print(f"<!-- PDF 5 paired comparison | {a.label} - {a.paired_label} | "
          f"{n} aligned seeds: {lseeds} -->")
    print(f"| Metric | {a.label} | {a.paired_label} | Seedwise differences "
          f"({a.label} - {a.paired_label}) | Mean diff ± std | Paired 95% CI |")
    print("|---|---|---|---|---|---|")

    for key in sorted(set(left) & set(right)):
        if any(key.startswith(p + ".") or key == p for p in a.exclude_prefix):
            continue
        lv, rv = left[key], right[key]
        if len(lv) != n or len(rv) != n:
            print(f"| `{key}` | INCOMPLETE | INCOMPLETE | — | — | — |")
            continue
        try:
            d = np.asarray(lv, dtype=float) - np.asarray(rv, dtype=float)
        except (TypeError, ValueError):
            continue  # non-numeric leaf (a path, a name); nothing to difference
        dm, ds, dh = stats(d)
        lm, ls, _ = stats(lv)
        rm, rs, _ = stats(rv)
        diffs = ", ".join(f"{x:+.4g}" for x in d)
        print(f"| `{key}` | {fmt(lm, ls)} | {fmt(rm, rs)} | {diffs} | "
              f"{fmt(dm, ds)} | ±{dh:.3g} |")

    print(f"\n<!-- Paired half-width = t(0.975,4)={T_CRIT_4DOF} * s_d / sqrt({n}). "
          "A CI excluding 0 is a per-seed-consistent difference; one containing 0 is not, "
          "however far apart the two means look. -->")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="method dir containing pdf_metrics/")
    ap.add_argument("--floor-dir", default=None, help="perfect_floor dir containing pdf_metrics/")
    ap.add_argument("--pattern", default="*_drawdown_memory.json")
    ap.add_argument("--label", default="Model")
    ap.add_argument("--exclude-prefix", nargs="*", default=[],
                    help="dotted-path prefixes to omit, e.g. configuration oracle_gate")
    # PDF §7 checklist item 2 requires evidence that disc.npy was used for validation and
    # test.npy only after the config freeze. That evidence is a second scoring pass whose
    # JSONs live in pdf_metrics_validation/, so the table has to be renderable from either
    # directory by the same code -- otherwise the validation column gets hand-copied.
    ap.add_argument("--subdir", default="pdf_metrics",
                    help="pdf_metrics (test side) or pdf_metrics_validation (disc side)")
    # PDF §5: "For comparisons between models run with aligned seeds, also report the five
    # seedwise differences and their corresponding paired interval." That sentence binds only
    # when the seeds are ALIGNED. Model-vs-perfect-floor is NOT such a comparison -- the floor
    # is drawn at seeds 1000-1004 against a model's 0-4 -- so an unpaired table is correct
    # there, and this mode refuses to run rather than manufacturing a bogus pairing.
    # Model-vs-model (e.g. CSDI seeds 0-4 vs LS4 seeds 0-4) IS aligned, and then the paired
    # interval is mandatory, not optional.
    ap.add_argument("--paired-dir", default=None,
                    help="second METHOD dir with the same seeds; emits the PDF 5 paired table")
    ap.add_argument("--paired-label", default="Baseline")
    a = ap.parse_args()

    if a.paired_dir:
        return paired(a)

    model, model_files = collect(a.model_dir, a.pattern, a.subdir)
    floor, floor_files = (collect(a.floor_dir, a.pattern, a.subdir) if a.floor_dir else ({}, []))

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
