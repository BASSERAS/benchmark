#!/usr/bin/env python3
"""
collect_artifacts.py
--------------------
Derive the tracked `losses/` bookkeeping files from the untracked banks.

The banks themselves are gitignored (6144x128x8 float64 = 50 MB each), but the
`metadata.json` written beside each one IS tracked, and it already carries every
number the README needs: `(h, K, N_pi)`, worker count and wall-clock. This script
turns those JSONs into the two artefacts `render_readme.py` reads, so the timing
table in the README can never drift away from the runs that produced it.

The Heston SBTS tree gets the same two files from `run_all_multiasset.py`, which
generates and books in one process. TrueDataset generation went through
`generate_bank_true.py` (one bank per invocation, driven by a shell loop), so the
bookkeeping has to be a separate pass.

Reads
  results/trueexperiment/SBTS/generated_paths/seed_*/metadata.json

Writes
  results/trueexperiment/SBTS/losses/generation_time.csv
  results/trueexperiment/SBTS/losses/seed_<S>_bandwidth.json

Usage
    python results/trueexperiment/SBTS/code/collect_artifacts.py
"""

import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SBTS = os.path.dirname(HERE)

FIELDS = ["seed", "n_workers", "elapsed_min", "h", "K", "N_pi"]

# Read the physical timestep from the variant's own stats file rather than
# hardcoding it: a rebuilt variant with a different bar size would otherwise
# silently keep publishing the old number.
_STATS = os.path.join(SBTS, "losses", "dataset_stats.json")
if os.path.exists(_STATS):
    with open(_STATS) as _fh:
        PHYSICAL_DT = json.load(_fh)["dt"]
else:                                   # copied in by hand? fail loudly, not silently.
    raise SystemExit(f"missing {_STATS} -- copy it from the variant directory first; "
                     "it is the only in-tree record of the physical dt.")


def _seed_of(path):
    """Seed id from the directory name, not from the JSON.

    The directory is what every other tool keys on (`compute_all_multiasset.py`
    globs `generated_paths/seed_<S>/`), so a metadata file whose `seed` field
    disagreed with its own folder would be a silent mislabel. Cross-check both.
    """
    m = re.search(r"seed_(\d+)", path)
    if not m:
        raise ValueError(f"cannot parse a seed id out of {path}")
    return int(m.group(1))


def main():
    metas = sorted(glob.glob(os.path.join(SBTS, "generated_paths", "seed_*",
                                          "metadata.json")),
                   key=_seed_of)
    if not metas:
        raise SystemExit(f"no metadata.json under {SBTS}/generated_paths/seed_*/")

    losses = os.path.join(SBTS, "losses")
    os.makedirs(losses, exist_ok=True)

    rows = []
    for path in metas:
        seed = _seed_of(path)
        with open(path) as fh:
            m = json.load(fh)
        if int(m["seed"]) != seed:
            raise SystemExit(f"{path}: metadata seed {m['seed']} != folder seed {seed}")

        rows.append({"seed": seed,
                     "n_workers": m["n_workers"],
                     "elapsed_min": round(m["elapsed_sec"] / 60.0, 2),
                     "h": m["h"], "K": m["K"], "N_pi": m["N_pi"]})

        # Per-seed hyperparameter record. SBTS has no loss curve, so this file is
        # the `losses/` directory's only per-seed content -- it is the complete
        # description of the "model" for that seed, alongside the training array.
        # The `dt` in metadata.json is SBTS's internal gauge DT (1/252), NOT the
        # dataset's physical timestep (9.512937595129376e-07 for 30 s bars).
        # Publishing it under the bare name "dt" would hand a reader the exact
        # number the guideline's §3.2 warns against -- the two differ by 4173x,
        # which is 64.6x in annualised vol. So it is named for what it is, and
        # the physical timestep is carried alongside it.
        with open(os.path.join(losses, f"seed_{seed}_bandwidth.json"), "w") as fh:
            json.dump({"seed": seed, "h": m["h"], "K": m["K"], "N_pi": m["N_pi"],
                       "dt_gauge": m["dt"], "dt_physical": PHYSICAL_DT,
                       "dt_note": "dt_gauge is SBTS's internal DT: h is measured "
                                  "in units of sqrt(DT), and build_training_tensor "
                                  "always rescales to std(X) = sqrt(DT), so DT "
                                  "cancels and is a pure gauge. dt_physical is the "
                                  "real bar spacing and is what metrics/ must be "
                                  "passed via --dt. They are NOT interchangeable.",
                       "n_workers": m["n_workers"],
                       "elapsed_sec": m["elapsed_sec"],
                       "note": "SBTS is non-parametric: no weights, no loss. "
                               "These four numbers plus the training array ARE the model."},
                      fh, indent=2)

    csv_path = os.path.join(losses, "generation_time.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    total = sum(r["elapsed_min"] for r in rows)
    print(f"wrote {csv_path}  ({len(rows)} seeds, {total:.1f} min total)")
    print(f"wrote {len(rows)} x seed_<S>_bandwidth.json in {losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
