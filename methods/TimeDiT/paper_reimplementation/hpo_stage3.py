"""TimeDiT Sine-disc HPO — STAGE 3 (honest, consistent-budget sweep).

WHY THIS EXISTS
---------------
Stage-2 was run at 3000 steps while the stage-1 leader was measured at 4000
steps.  The SAME recipe (znorm.linear.ls=F.ddpm_fixed.lr=3e-4.no-EMA) scored
0.0115 at 4000 steps but 0.1130 at 3000 steps -- a fidelity threshold sits
between those two step counts.  So stage-2's "EMA wins" ranking was EMA merely
compensating for my own under-training; it does not transfer.

Stage-3 fixes the methodology:
  * The proven no-EMA recipe is the BASE (locked knobs below).
  * `steps` is a deliberate, RECORDED axis -- every trial's operating point is
    explicit, so rankings are comparable and the step->disc trend is visible.
  * EMA is tested head-to-head at the SAME step budget as its no-EMA twin, and
    is only worth keeping if it actually beats no-EMA there.

Locked (all stage-1 winners): znorm . linear . learn_sigma=False . ddpm_fixed.
Grid  : lr=3e-4 (proven) x steps{6000,10000,14000} x ema{0.0, 0.999} = 6 trials.
Goal  : find the smallest recipe whose disc mean+-std overlaps the paper 0.0086.

Usage (one GPU):
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 16-23 \
      python hpo_stage3.py
"""
import argparse
import itertools
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from hpo_stage2 import run_trial   # identical trained pipeline (no-EMA + EMA+warmup)

# Locked axes (stage-1 winners / proven base)
NORM = "znorm"
SCHEDULE = "linear"
LEARN_SIGMA = False
SAMPLER = "ddpm_fixed"
LR = 3e-4


def build_grid():
    """steps is an explicit axis; ema is the head-to-head no-EMA-vs-EMA test."""
    step_grid = [6000, 10000, 14000]
    emas = [0.0, 0.999]
    grid = []
    for steps, ema in itertools.product(step_grid, emas):
        grid.append({"norm": NORM, "schedule": SCHEDULE, "learn_sigma": LEARN_SIGMA,
                     "sampler": SAMPLER, "ddim_steps": None, "lr": LR,
                     "ema": ema, "steps": steps})
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--subset", type=int, default=4000)
    p.add_argument("--n_gen", type=int, default=1000)
    p.add_argument("--disc_seeds", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    grid = build_grid()
    out = "hpo_stage3_shard0.jsonl"
    print(f"[hpo3] {len(grid)} trials -> {out}  locked: norm={NORM} sched={SCHEDULE} "
          f"ls={LEARN_SIGMA} samp={SAMPLER} lr={LR:.0e}", flush=True)

    data01 = load_dataset("sine", seq_len=args.seq_len, seed=0)
    if args.subset and len(data01) > args.subset:
        data01 = data01[:args.subset]
    print(f"[hpo3] sine data {data01.shape}", flush=True)

    for k, cfg in enumerate(grid):
        args.steps = cfg["steps"]            # per-trial operating point
        t0 = time.time()
        res = run_trial(cfg, data01, device, args)
        row = {**cfg, **res, "wall_s": time.time() - t0}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[hpo3] {k + 1}/{len(grid)} steps={cfg['steps']} ema={cfg['ema']} "
              f"-> disc={res['disc_mean']:.4f}+-{res.get('disc_std', 0):.4f} "
              f"pred={res['pred_mean']:.4f} ({row['wall_s']:.0f}s)", flush=True)
    print("[hpo3] DONE", flush=True)
    with open("hpo_status.txt", "a") as f:
        f.write("STAGE3_DONE\n")


if __name__ == "__main__":
    main()
