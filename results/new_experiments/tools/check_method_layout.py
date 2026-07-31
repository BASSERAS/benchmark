#!/usr/bin/env python
"""Verify that a method directory matches the layout contract in
`results/new_experiments/guideline_new_experiment.md` §6.2 / §6.4 / §6.5.

"Same structure of files" is a claim, and claims get checked. Run this BEFORE
writing the README: it is far cheaper to find a missing seed here than inside a
regenerated table.

    python results/new_experiments/tools/check_method_layout.py \
        --root results/new_experiments/experiment_A/LS4 --experiment A

Exit codes
----------
0   every check passed
1   at least one violation (all violations are printed, not just the first)

The checker verifies presence AND shape. A file that exists but holds a
(8192, 100) array, or a bank sitting at the wrong price scale, is the porting
bug this catches -- it loads fine and scores catastrophically.

Method-neutral by construction: nothing here knows what LS4 is. Run it against
LS4 first; if it fails on the reference method, the checker is wrong, not your
new method.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

SEEDS = range(5)
N_PATHS, N_STEPS = 8192, 128
S0 = 100.0

# PDF evaluator output name, per experiment (§2.4 / §3.4).
EVAL_STEM = {"A": "drawdown_memory", "B": "heston_mixture"}

SUMMARY_HEADER = ["metric", "mean", "std", "seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]


class Report:
    """Collects failures so one run surfaces every problem, not just the first."""

    def __init__(self) -> None:
        self.fails: list[str] = []
        self.n_checks = 0

    def check(self, ok: bool, msg: str) -> bool:
        self.n_checks += 1
        if not ok:
            self.fails.append(msg)
        return ok

    def fail(self, msg: str) -> None:
        self.n_checks += 1
        self.fails.append(msg)


def _exists(rep: Report, root: str, rel: str) -> bool:
    return rep.check(os.path.isfile(os.path.join(root, rel)), f"MISSING FILE   {rel}")


def _load_json(rep: Report, root: str, rel: str):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        rep.fail(f"MISSING FILE   {rel}")
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
        rep.fail(f"UNPARSEABLE    {rel}: {exc}")
        return None


def check_banks(rep: Report, root: str) -> None:
    """§6.5 required output 1 + PDF §1.4 output contract, re-measured from disk."""
    for s in SEEDS:
        rel = f"generated_paths/seed_{s}/generated_paths_{N_PATHS}x{N_STEPS}.npy"
        path = os.path.join(root, rel)
        if not _exists(rep, root, rel):
            continue
        a = np.load(path)
        rep.check(a.shape == (N_PATHS, N_STEPS), f"BAD SHAPE      {rel}: {a.shape} != ({N_PATHS}, {N_STEPS})")
        rep.check(a.dtype in (np.float32, np.float64), f"BAD DTYPE      {rel}: {a.dtype} not float32/float64")
        rep.check(bool(np.isfinite(a).all()), f"NON-FINITE     {rel}: {int((~np.isfinite(a)).sum())} bad entries")
        if np.isfinite(a).all():
            rep.check(bool((a > 0).all()), f"NON-POSITIVE   {rel}: min={a.min()}")
            # PDF §1.4 / §7 item 5. Exact equality is the protocol's wording.
            rep.check(bool(np.all(a[:, 0] == S0)),
                      f"S0 != {S0}      {rel}: max deviation {np.abs(a[:, 0] - S0).max():.3e}")


def check_metadata(rep: Report, root: str, experiment: str) -> None:
    """§6.4 schema. first_nan_epoch/gen_has_nan are how a diverged seed is caught."""
    required = ["method", "experiment", "seed", "shape", "min_val", "max_val",
                "generated_mean", "generated_std", "real_mean", "real_std",
                "gpu", "date", "params", "epochs_run", "epochs_max",
                "min_total_loss", "first_nan_epoch", "gen_has_nan"]
    for s in SEEDS:
        rel = f"generated_paths/seed_{s}/metadata.json"
        d = _load_json(rep, root, rel)
        if d is None:
            continue
        for k in required:
            rep.check(k in d, f"MISSING KEY    {rel}: '{k}'")
        if "seed" in d:
            rep.check(d["seed"] == s, f"WRONG SEED     {rel}: records {d['seed']}, lives in seed_{s}/")
        if "experiment" in d:
            rep.check(str(d["experiment"]).upper() == experiment,
                      f"WRONG EXP      {rel}: records {d['experiment']!r}, expected {experiment!r}")
        if "gen_has_nan" in d:
            rep.check(d["gen_has_nan"] is False, f"NaN IN BANK    {rel}: gen_has_nan={d['gen_has_nan']}")
        if "first_nan_epoch" in d:
            rep.check(d["first_nan_epoch"] is None,
                      f"DIVERGED       {rel}: first_nan_epoch={d['first_nan_epoch']}")


def check_manifests(rep: Report, root: str) -> None:
    """PDF §1.4 mandatory per-seed submission artefact (§11.3 item 6)."""
    for s in SEEDS:
        rel = f"generated_paths/seed_{s}/generation_manifest.json"
        _load_json(rep, root, rel)


def check_weights_and_losses(rep: Report, root: str, experiment: str) -> None:
    """§6.5 required outputs 3 and 4, plus the independence evidence of §8 item 15."""
    digests, mus = set(), set()
    for s in SEEDS:
        _exists(rep, root, f"weights/seed_{s}_model.pt")

        rel = f"weights/seed_{s}_config.json"
        cfg = _load_json(rep, root, rel)
        if cfg is not None:
            for k in ("scaler_mu", "scaler_sigma", "data", "experiment"):
                rep.check(k in cfg, f"MISSING KEY    {rel}: '{k}'")
            if "experiment" in cfg:
                rep.check(str(cfg["experiment"]).upper() == experiment,
                          f"WRONG EXP      {rel}: records {cfg['experiment']!r}, expected {experiment!r}")
            if "data" in cfg:
                # The firewall is only auditable if the path is recorded and points
                # at THIS experiment's directory.
                rep.check(f"experiment_{experiment}" in str(cfg["data"]),
                          f"WRONG DATA     {rel}: {cfg['data']!r} is not experiment_{experiment}")
            if "scaler_mu" in cfg:
                mus.add(round(float(cfg["scaler_mu"]), 12))

        rel = f"losses/seed_{s}_losses.csv"
        path = os.path.join(root, rel)
        if not _exists(rep, root, rel):
            continue
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            rep.fail(f"EMPTY          {rel}")
            continue
        header = [c.strip() for c in rows[0]]
        rep.check(header[0] == "epoch", f"BAD HEADER     {rel}: first column {header[0]!r} != 'epoch'")
        rep.check(header[-1] == "lr", f"BAD HEADER     {rel}: last column {header[-1]!r} != 'lr'")
        rep.check(len(rows) > 1, f"NO EPOCHS      {rel}")
        digests.add(tuple(rows[-1]))

    # Five identical final rows means the seeds were not independent -- or, more
    # often, that one training was copied five times.
    rep.check(len(digests) == len(list(SEEDS)) or not digests,
              f"NOT INDEPENDENT losses/: only {len(digests)} distinct final rows across 5 seeds")
    rep.check(len(mus) <= 1,
              f"SCALER DRIFT   weights/: {len(mus)} distinct scaler_mu across seeds of one experiment")


def check_metrics(rep: Report, root: str, experiment: str) -> None:
    """§6.2 metrics side: A/B suite, plus the PDF evaluator on BOTH sides (§11.4)."""
    stem = EVAL_STEM[experiment]
    for s in SEEDS:
        _load_json(rep, root, f"seed_{s}_metrics.json")
        _load_json(rep, root, f"pdf_metrics/seed_{s}_{stem}.json")
        _load_json(rep, root, f"pdf_metrics_validation/seed_{s}_{stem}.json")

    rel = "metrics_summary.csv"
    path = os.path.join(root, rel)
    if _exists(rep, root, rel):
        with open(path, newline="") as fh:
            header = next(csv.reader(fh))
        rep.check([c.strip() for c in header] == SUMMARY_HEADER,
                  f"BAD HEADER     {rel}: {header} != {SUMMARY_HEADER}")


def check_code_and_plots(rep: Report, root: str, experiment: str) -> None:
    """§6.5 required output 5, and the figures README §3/§4 must be able to link."""
    for s in SEEDS:
        _exists(rep, root, f"code/logs/exp{experiment}_seed{s}.log")
    for rel in ("losses/loss_convergence.png", "plots/heston_diagnostics.png"):
        _exists(rep, root, rel)
    _exists(rep, root, "README.md")

    code_dir = os.path.join(root, "code")
    trainers = os.listdir(code_dir) if os.path.isdir(code_dir) else []
    rep.check(any(f.startswith("train_") and f.endswith("_experiment.py") for f in trainers),
              "MISSING FILE   code/train_<method>_experiment.py")
    rep.check("compute_metrics_experiment.py" in trainers,
              "MISSING FILE   code/compute_metrics_experiment.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/new_experiments/experiment_<X>/<Method>")
    ap.add_argument("--experiment", required=True, choices=["A", "B"])
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        print(f"FATAL: --root is not a directory: {root}", file=sys.stderr)
        return 1

    rep = Report()
    check_banks(rep, root)
    check_metadata(rep, root, a.experiment)
    check_manifests(rep, root)
    check_weights_and_losses(rep, root, a.experiment)
    check_metrics(rep, root, a.experiment)
    check_code_and_plots(rep, root, a.experiment)

    label = f"{os.path.basename(root)} (experiment {a.experiment})"
    if rep.fails:
        print(f"FAIL  {label}: {len(rep.fails)} of {rep.n_checks} checks failed\n")
        for f in rep.fails:
            print(f"  {f}")
        return 1
    print(f"PASS  {label}: all {rep.n_checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
