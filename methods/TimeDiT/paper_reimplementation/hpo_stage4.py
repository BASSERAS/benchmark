"""TimeDiT Sine-disc HPO — STAGE 4 (broad, honest hyperparameter search).

WHY THIS EXISTS
---------------
Stages 2-3 confounded the search: stage-2 was under-trained (3000 steps) and
stage-3 degenerated into a pure step-count sweep with lr LOCKED.  Neither
actually searched hyperparameters.  The user's directive is explicit:

    "no ema worked better so u keep this config and try optimize parameters ...
     if ur method will allow to beat the no ema u keep it but else u do a no ema
     ... until reaching paper results"
    "broad one pls and u can use a second GPU for it ... 100% of both GPU"

So stage-4 fixes the training budget at ONE adequate value (steps=6000 -- past
the fidelity threshold that sat between 3000 and 4000) and sweeps the REAL
optimizer/regularization knobs at that fixed budget:

  * lr           {2e-4, 3e-4, 5e-4}   -- the learning-rate knife-edge
  * ema          {0.0, 0.999+warmup}  -- head-to-head at the SAME budget; EMA
                                          is kept ONLY if it truly beats no-EMA
  * weight_decay {0.0, 1e-4}          -- L2 regularization
  * batch        {128, 256}           -- SGD noise scale

Grid = 3 x 2 x 2 x 2 = 24 trials.  Everything else LOCKED to stage-1 winners:
znorm . linear . learn_sigma=False . ddpm_fixed.  Steps is NOT an axis here --
it is a fixed, adequate budget, so rankings reflect hyperparameters, not luck.

Sharding: --n_shards N --shard_idx i runs trials where (k % N == i), so two
GPUs each take 12 trials and both stay at ~100% utilization.

Usage (two GPUs, per the user's 2-GPU authorization):
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 16-23 \
      python hpo_stage4.py --n_shards 2 --shard_idx 0 &
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 24-31 \
      python hpo_stage4.py --n_shards 2 --shard_idx 1 &
  wait
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
FIXED_STEPS = 6000


def build_grid():
    """Broad HP grid at a FIXED training budget. Deterministic order so shard
    assignment (k % n_shards) is stable across processes."""
    lrs = [2e-4, 3e-4, 5e-4]
    emas = [0.0, 0.999]
    wds = [0.0, 1e-4]
    batches = [128, 256]
    grid = []
    for lr, ema, wd, batch in itertools.product(lrs, emas, wds, batches):
        grid.append({"norm": NORM, "schedule": SCHEDULE, "learn_sigma": LEARN_SIGMA,
                     "sampler": SAMPLER, "ddim_steps": None, "lr": lr,
                     "ema": ema, "weight_decay": wd, "batch": batch,
                     "steps": FIXED_STEPS})
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)   # overridden per-trial
    p.add_argument("--steps", type=int, default=FIXED_STEPS)
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
    mine = [(k, cfg) for k, cfg in enumerate(grid)
            if k % args.n_shards == args.shard_idx]
    out = f"hpo_stage4_shard{args.shard_idx}.jsonl"
    print(f"[hpo4] shard {args.shard_idx}/{args.n_shards}: {len(mine)}/{len(grid)} "
          f"trials -> {out}  FIXED steps={FIXED_STEPS}  locked: norm={NORM} "
          f"sched={SCHEDULE} ls={LEARN_SIGMA} samp={SAMPLER}", flush=True)

    data01 = load_dataset("sine", seq_len=args.seq_len, seed=0)
    if args.subset and len(data01) > args.subset:
        data01 = data01[:args.subset]
    print(f"[hpo4] sine data {data01.shape}", flush=True)

    for n, (k, cfg) in enumerate(mine):
        args.batch = cfg["batch"]            # per-trial SGD noise scale
        args.steps = cfg["steps"]            # fixed budget, but explicit
        t0 = time.time()
        res = run_trial(cfg, data01, device, args)
        row = {**cfg, **res, "trial_idx": k, "wall_s": time.time() - t0}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[hpo4] {n + 1}/{len(mine)} (global {k}) lr={cfg['lr']:.0e} "
              f"ema={cfg['ema']} wd={cfg['weight_decay']:.0e} b={cfg['batch']} "
              f"-> disc={res['disc_mean']:.4f}+-{res.get('disc_std', 0):.4f} "
              f"pred={res['pred_mean']:.4f} ({row['wall_s']:.0f}s)", flush=True)
    print(f"[hpo4] shard {args.shard_idx} DONE", flush=True)
    with open("hpo_status.txt", "a") as f:
        f.write(f"STAGE4_SHARD{args.shard_idx}_DONE\n")


if __name__ == "__main__":
    main()
