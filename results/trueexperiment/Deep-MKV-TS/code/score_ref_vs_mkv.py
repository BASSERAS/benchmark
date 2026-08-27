#!/usr/bin/env python3
"""Score the untrained reference SDE and the trained Deep-MKV-TS banks on the
SAME real-vs-real envelope, so the two are directly comparable.

The question
------------
``select_checkpoint_true.py`` aborted on all five seeds -- no checkpoint clears
the envelope.  That says Deep-MKV-TS fails an absolute bar.  It does NOT say
whether the learned control is better or worse than the untrained reference SDE
it starts from, because the reference has never been put through this gate:
``losses/envelope_screen.json`` has ``"seeds": {}``.

That comparison is the whole point of a control-variate method.  If the learned
control cannot beat Theta = eta*(sigma^ref)^-1 with Zhat == 0, the method adds
nothing on this dataset regardless of where the ceiling sits.

Both sides are scored by ``selection_true.score_candidate`` against the SAME
``val`` split with the SAME ``median_nn_heldout`` denominator, so the numbers
sit on the same axis as the published ceilings (vol_err <= 24.59%,
corr_err <= 0.1843) and as every row in ``logs/select_seed_*.log``.

Pure numpy, no torch, no GPU.

Usage
-----
    OMP_NUM_THREADS=16 taskset -c 16-31 \\
    /home/tbasseras/.cc-venv/bin/python score_ref_vs_mkv.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import selection_true as sel  # noqa: E402

TRUE_ROOT = HERE.parent.parent  # results/trueexperiment
SEQ_TAG = "6144x128x8"
SEEDS = [0, 1, 2, 3, 4]

ARMS = {
    "reference (untrained, Zhat=0)": TRUE_ROOT / "reference" / "generated_paths",
    "Deep-MKV-TS (best-fit ckpt)": (
        TRUE_ROOT / "Deep-MKV-TS" / "diagnostic_bestfit" / "generated_paths"
    ),
}

COLS = ["vol_err_pct", "corr_err", "kurt_err_pct", "nn_ratio", "abs_log_nn_ratio"]


def main() -> None:
    env = sel.envelope()
    train, val = sel.load_splits()
    module = sel.load_criterion(sel.DEFAULT_DATA_DIR)
    med_nn = env["median_nn_heldout"]

    print(
        f"envelope: vol_err <= {env['vol_err_pct_max']:.2f}%   "
        f"corr_err <= {env['corr_err_max']:.4f}   "
        f"median_nn_heldout = {med_nn:.6f}",
        flush=True,
    )

    out: dict[str, dict] = {}
    for arm, root in ARMS.items():
        per_seed = []
        for seed in SEEDS:
            npy = root / f"seed_{seed}" / f"generated_paths_{SEQ_TAG}.npy"
            if not npy.is_file():
                raise SystemExit(f"ABORT: {npy} missing")
            gen = np.load(npy)
            rec = sel.score_candidate(
                gen, train=train, val=val, module=module, median_nn_heldout=med_nn
            )
            rec["seed"] = seed
            rec["reasons"] = sel.admissibility(rec, env)
            rec["admissible"] = not rec["reasons"]
            per_seed.append(rec)
            verdict = "ADMISSIBLE" if rec["admissible"] else "; ".join(rec["reasons"])
            print(
                f"  {arm:32s} seed {seed}: "
                f"vol {rec['vol_err_pct']:7.2f}%  corr {rec['corr_err']:.4f}  "
                f"kurt {rec['kurt_err_pct']:8.2f}%  NN {rec['nn_ratio']:.4f}  "
                f"{verdict}",
                flush=True,
            )
        agg: dict = {}
        for col in COLS:
            vals = [r[col] for r in per_seed if np.isfinite(r[col])]
            agg[col] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)) if vals else float("nan"),
            }
        agg["n_admissible"] = sum(r["admissible"] for r in per_seed)
        out[arm] = {"per_seed": per_seed, "aggregate": agg}

    print("\n" + "=" * 96)
    print(
        f"{'arm':34s}{'vol_err %':>18s}{'corr_err':>18s}"
        f"{'kurt_err %':>18s}{'admissible':>12s}"
    )
    print("-" * 96)
    for arm, blk in out.items():
        a = blk["aggregate"]
        print(
            f"{arm:34s}"
            f"{a['vol_err_pct']['mean']:11.2f} +-{a['vol_err_pct']['std']:5.2f}"
            f"{a['corr_err']['mean']:12.4f} +-{a['corr_err']['std']:6.4f}"
            f"{a['kurt_err_pct']['mean']:11.2f} +-{a['kurt_err_pct']['std']:5.2f}"
            f"{a['n_admissible']:8d} / 5"
        )
    print("-" * 96)
    print(
        f"{'CEILING (real-vs-real)':34s}"
        f"{env['vol_err_pct_max']:18.2f}{env['corr_err_max']:18.4f}"
        f"{'(not gated)':>18s}"
    )
    print("=" * 96)

    dest = TRUE_ROOT / "Deep-MKV-TS" / "diagnostic_bestfit" / "ref_vs_mkv_envelope.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"envelope": env, "arms": out}, indent=2, sort_keys=True, default=float
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {dest}", flush=True)


if __name__ == "__main__":
    main()
