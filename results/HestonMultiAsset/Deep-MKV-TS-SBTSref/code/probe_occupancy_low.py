#!/usr/bin/env python
"""Kernel occupancy at the LOW bandwidths, K = 20.

WHY THIS EXISTS SEPARATELY.  ``probe_jacobian_tail.py`` hard-codes its
small-h endpoint list as ``(0.05, 0.10, 0.20, 0.31, 0.36)`` at line 168 and
reports a single ``step = 120``, and it recomputes the whole Jacobian-tail
study on the way there.  The low-h re-sweep tests ``h in {0.28, 0.31, 0.33}``,
so two of those three bandwidths have never been measured at all.

WHY THE MEASUREMENT IS REQUIRED BEFORE PROMOTING.  At K = 20 the kernel has a
death point: below some bandwidth no bank path lies inside the joint support of
all 20 lags, every weight underflows, and ``b^ref`` is identically zero.  A run
there is not "a good bandwidth that happens to score well" -- it is a model
trained against a reference drift of exactly 0, and its validation discrepancy
measures something else entirely.  Since a low-h arm is the likely winner of
the extended sweep, "is this arm alive?" has to be answered with a number
rather than assumed from the fact that it produced a score.

Bank and queries are DISJOINT, exactly as in ``probe_jacobian_tail.py``: the
bank is paths [0, 4096) and the queries are [4096, 4352).  A shared array makes
every query its own nearest neighbour at distance 0, which survives any
bandwidth and reports a spurious n_eff = 1.  Training also cannot self-match,
because there the prefixes are model rollouts.

Three steps, not one.  Occupancy is not constant along the path: early steps
have fewer than K lags available and late steps have drifted further from the
bank.  A bandwidth that is alive at step 120 but dead at step 250 is still
unusable, and one number cannot show that.

Reads ``heston_ma_S_8192x252x8.npy`` -- a bare float64 array of shape
(path, time, asset) = (8192, 252, 8) holding strictly positive PRICE levels
that start at 100.0 for every asset.  No field names, no dates.  The script
takes the log itself.  It writes nothing to disk; output is stdout only.

Run:
    /home/tbasseras/gpu-venv/bin/python probe_occupancy_low.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sbts_reference as sr

DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
TRAIN = DATASET / "heston_ma_S_8192x252x8.npy"
DT = 1.0 / 252.0
K = 20
BATCH = 256
BANK_PATHS = 4096
# The FULL re-sweep grid, low arms and original arms together, so that one file
# is the single source of truth for the occupancy column in tabulate_hfix.py.
# That table previously carried the numbers as a hand-typed dict whose own
# comment said they were "transcribed" -- and a transcription of 12.66 for a
# true 16.7 has already happened once in this project.  h = 0.36 through 2.00
# are therefore recomputed here rather than copied, and if they disagree with
# the published occupancy grid the disagreement is itself the finding.
H_LIST = (0.28, 0.31, 0.33, 0.36, 0.50, 0.70, 1.00, 1.50, 2.00)
STEPS = (25, 120, 250)
# Step 120 is the one the summary table quotes: it is mid-path, so every lag of
# the K = 20 window exists (unlike step 25 at the start) without the extra
# drift away from the bank that step 250 accumulates.
REPORT_STEP = 120
OUT = Path(__file__).resolve().parent / "sweep" / "occupancy_k20.json"


class _Grid:
    def __init__(self, dt: float, num_steps: int) -> None:
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.T = float(dt) * int(num_steps)


def main() -> None:
    all_prices = np.load(TRAIN)
    grid = _Grid(DT, int(all_prices.shape[1]) - 1)

    prices = np.ascontiguousarray(all_prices[:BANK_PATHS])
    queries = np.ascontiguousarray(all_prices[BANK_PATHS : BANK_PATHS + BATCH])
    x_full = torch.from_numpy(np.log(queries))

    print(f"[data] {TRAIN.name} bank={prices.shape} queries={queries.shape}")
    print(f"[note] K={K} float64 CPU, bank and queries disjoint")
    print()
    print("=" * 78)
    print("low-bandwidth occupancy, K = 20")
    print("=" * 78)
    print(f"{'h':>6} {'step':>6} {'alive':>10} {'med n_eff':>11} {'%one-hot':>10}  verdict")

    record: dict[str, dict] = {}
    for h in H_LIST:
        kern = sr.build_sbts_reference_kernel(
            train_prices=torch.from_numpy(prices),
            grid=grid,
            h=h,
            markov_order=K,
            npi=1,
            weight_grad_mode="analytic",
            device="cpu",
            dtype=torch.float64,
            jacobian_lags=-1,
        )
        for step in STEPS:
            lags = min(step, K)
            returns = kern._scaled_returns_of_prefix(x_full[:, : step + 1, :], lags=lags)
            p, alive = kern._normalised_weights(
                kern._log_weights(returns, step_index=step, lags=lags)
            )
            n_alive = int(alive.sum())
            total = int(alive.numel())
            if n_alive == 0:
                print(f"{h:>6.2f} {step:>6d} {f'{n_alive}/{total}':>10} "
                      f"{'--':>11} {'--':>10}  DEAD: b^ref := 0 on every row")
                if step == REPORT_STEP:
                    record[f"{h:.2f}"] = {
                        "h": h, "step": step, "alive": 0, "total": total,
                        "alive_pct": 0.0, "n_eff_median": None,
                        "onehot_pct": None, "verdict": "DEAD",
                    }
                continue
            n_eff = (1.0 / p.pow(2).sum(dim=1).clamp(min=1e-300)).numpy()[alive.numpy()]
            med = float(np.median(n_eff))
            onehot = float((n_eff < 1.1).mean()) * 100.0
            verdict = "1-NN lookup" if med < 1.1 else (
                "near-degenerate" if med < 10.0 else "averaging"
            )
            dead = total - n_alive
            extra = f" ({dead} rows zero drift)" if dead else ""
            print(f"{h:>6.2f} {step:>6d} {f'{n_alive}/{total}':>10} "
                  f"{med:>11.2f} {onehot:>9.1f}%  {verdict}{extra}")
            if step == REPORT_STEP:
                record[f"{h:.2f}"] = {
                    "h": h, "step": step, "alive": n_alive, "total": total,
                    "alive_pct": 100.0 * n_alive / total,
                    "n_eff_median": med, "onehot_pct": onehot,
                    "verdict": verdict,
                }
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": Path(__file__).name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dataset": TRAIN.name,
        "bank_paths": BANK_PATHS,
        "queries": BATCH,
        "markov_order": K,
        "report_step": REPORT_STEP,
        "note": "bank rows [0,4096), query rows [4096,4352) -- disjoint",
        "by_h": record,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {OUT}")

    print("reading: 'alive' = rows where at least one bank weight did not")
    print("underflow.  A dead row trains against b^ref = 0.  'med n_eff' =")
    print("median 1/sum_m p_m^2, the effective number of contributing bank")
    print("paths.  n_eff near 1 means the 'conditional expectation' is a")
    print("single training path, i.e. a memorisation channel, not an average.")


if __name__ == "__main__":
    main()
