#!/usr/bin/env python3
"""DIAGNOSTIC ONLY -- stage the best-`fit` checkpoint per seed into weights/.

Why this file exists
--------------------
``select_checkpoint_true.py`` ABORTED on all five seeds: no checkpoint is
admissible under the real-vs-real envelope (vol_err <= 24.59%, corr_err <=
0.1843).  That abort is the honest result and this file does NOT overturn it.

But "no admissible checkpoint" does not answer the separate question "does the
learned control improve on the untrained reference SDE at all?".  To answer
that we still need a bank, and a bank needs a checkpoint.  This file picks one
by the ONLY rule that adds no new freedom:

    the checkpoint that select_checkpoint_true.py's OWN objective (`fit`)
    ranks best, with the admissibility gate lifted.

`fit` is not invented here -- it is the scalar the selection code already
minimises and already printed for every one of the 30 checkpoints.  Lifting the
gate is the single, explicit, documented deviation.  Picking on `fit` is also
the reading MOST FAVOURABLE to Deep-MKV-TS: if the method loses to the
reference even on its own best checkpoint, no choice of checkpoint saves it.

The `fit` values come from the selection logs, which are the artefacts the
selection run actually produced; they are re-read here rather than retyped.

What this does NOT claim
------------------------
`selected_step` is written, but `selection_rule` says in words that the gate
was lifted and that the bank is a diagnostic.  Every metadata.json downstream
inherits that string, so no artefact produced from these weights can be
mistaken for a selection-approved one.

The provisional bare state_dicts the trainer left in weights/ are MOVED to
weights/_provisional_from_trainer/ rather than deleted: they are the evidence
that selection never completed.

Usage
-----
    /home/tbasseras/.cc-venv/bin/python stage_bestfit_checkpoints.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
LOGS = METHOD_ROOT / "logs"
WEIGHTS = METHOD_ROOT / "weights"
RUNS = HERE / "runs"
SEEDS = [0, 1, 2, 3, 4]

RULE = (
    "DIAGNOSTIC, NOT A SELECTION: select_checkpoint_true.py aborted on this "
    "seed because no checkpoint is admissible under the real-vs-real envelope "
    "(vol_err <= 24.59%, corr_err <= 0.1843). This checkpoint is the one the "
    "selection objective `fit` ranks best WITH THE ADMISSIBILITY GATE LIFTED. "
    "Banks generated from it measure whether the learned control beats the "
    "untrained reference SDE; they are not eligible for publication as "
    "Deep-MKV-TS results."
)

# "  seed=0 step=  500 vol=41.23% ... fit=0.471961 REJECTED..."
ROW = re.compile(r"seed=(\d+)\s+step=\s*(\d+).*?fit=([0-9.eE+-]+)")


def fits_for_seed(seed: int) -> dict[int, float]:
    log = LOGS / f"select_seed_{seed}.log"
    if not log.is_file():
        raise SystemExit(f"ABORT: {log} missing -- selection never ran for this seed")
    out: dict[int, float] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        m = ROW.search(line)
        if m and int(m.group(1)) == seed:
            out[int(m.group(2))] = float(m.group(3))
    if not out:
        raise SystemExit(f"ABORT: no scored checkpoint rows parsed from {log}")
    return out


def main() -> None:
    stash = WEIGHTS / "_provisional_from_trainer"
    stash.mkdir(parents=True, exist_ok=True)

    chosen: dict[int, int] = {}
    for seed in SEEDS:
        fits = fits_for_seed(seed)
        step = min(fits, key=lambda s: fits[s])
        src = RUNS / f"seed_{seed}" / "training_checkpoints" / f"step_{step:04d}.pt"
        if not src.is_file():
            raise SystemExit(f"ABORT: {src} missing")

        dst = WEIGHTS / f"seed_{seed}_model.pt"
        if dst.is_file() and not (stash / dst.name).is_file():
            shutil.move(str(dst), str(stash / dst.name))
        shutil.copy2(src, dst)

        cfg_path = WEIGHTS / f"seed_{seed}_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        if cfg_path.is_file() and not (stash / cfg_path.name).is_file():
            shutil.copy2(cfg_path, stash / cfg_path.name)
        cfg["selected_step"] = int(step)
        cfg["selection_rule"] = RULE
        cfg["selection_fit"] = fits[step]
        cfg["selection_fit_all_steps"] = {str(k): v for k, v in sorted(fits.items())}
        cfg["selection_admissible"] = False
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        chosen[seed] = step
        ranked = "  ".join(f"{s}:{fits[s]:.4f}" for s in sorted(fits))
        print(f"seed {seed}: step {step}  fit {fits[step]:.6f}   [{ranked}]", flush=True)

    print("\nchosen steps:", {k: v for k, v in sorted(chosen.items())}, flush=True)
    print(f"provisional trainer weights preserved in {stash}", flush=True)


if __name__ == "__main__":
    main()
