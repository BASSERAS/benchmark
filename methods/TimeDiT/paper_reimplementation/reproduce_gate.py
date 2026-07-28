"""TimeDiT paper-reproduction GATE harness (Sine + Stock, seq_len=24).

Takes the winning HPO recipe (from hpo_sine.py stage-1 / hpo_stage2.py stage-2)
and runs it at FULL data + FULL training length, over several training seeds, to
produce the paper-vs-ours discriminative/predictive table needed for the Phase-1
GATE.  It reuses the exact trained pipeline (`run_trial` in hpo_stage2), so the
only differences from an HPO trial are: full dataset (no --subset), longer
--steps, and averaging over N training seeds.

Defaults are the current stage-1 leader
    znorm · linear · learn_sigma=False · ddpm_fixed · lr=3e-4 · no-EMA
Override any knob from the command line once the stage-2 winner is known, e.g.
    python reproduce_gate.py --dataset sine  --ema 0.9999 --sampler ddim_eta0 --ddim_steps 250
    python reproduce_gate.py --dataset stock --stock_csv <path>

Paper targets (Table 6): Sine disc 0.0086 / pred 0.093 ; Stock disc 0.0087.
"""
import argparse
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from hpo_stage2 import run_trial

DEFAULT_STOCK_CSV = ("/home/tbasseras/benchmark/methods/DiffusionTS/"
                     "code/reference/Data/datasets/stock_data.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sine", "stock"], default="sine")
    p.add_argument("--stock_csv", default=DEFAULT_STOCK_CSV)
    p.add_argument("--seq_len", type=int, default=24)
    # winning recipe (stage-1 leader defaults; override for stage-2 winner)
    p.add_argument("--norm", default="znorm")
    p.add_argument("--schedule", default="linear")
    p.add_argument("--learn_sigma", action="store_true")
    p.add_argument("--sampler", default="ddpm_fixed")
    p.add_argument("--ddim_steps", type=int, default=None)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    # full-length training
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n_gen", type=int, default=3000)
    p.add_argument("--disc_seeds", type=int, default=5)
    p.add_argument("--train_seeds", type=int, default=3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = {"norm": args.norm, "schedule": args.schedule,
           "learn_sigma": args.learn_sigma, "sampler": args.sampler,
           "ddim_steps": args.ddim_steps, "ema": args.ema, "lr": args.lr}
    # full dataset (no subset); run_trial reads args.subset -> set 0 to disable
    args.subset = 0

    data01 = load_dataset(args.dataset, seq_len=args.seq_len,
                          stock_csv=args.stock_csv, seed=0)
    print(f"[gate] {args.dataset} data {data01.shape}  cfg={cfg}  "
          f"steps={args.steps} train_seeds={args.train_seeds}", flush=True)

    per_seed = []
    for ts in range(args.train_seeds):
        t0 = time.time()
        res = run_trial(cfg, data01, device, args, train_seed=ts)
        res["train_seed"] = ts
        res["wall_s"] = time.time() - t0
        per_seed.append(res)
        print(f"[gate] train_seed={ts} disc={res['disc_mean']:.4f} "
              f"pred={res['pred_mean']:.4f} ({res['wall_s']:.0f}s)", flush=True)

    discs = [r["disc_mean"] for r in per_seed]
    preds = [r["pred_mean"] for r in per_seed]
    summary = {
        "dataset": args.dataset, "config": cfg, "seeds": args.train_seeds,
        "disc_mean": float(np.mean(discs)), "disc_std": float(np.std(discs)),
        "pred_mean": float(np.mean(preds)), "pred_std": float(np.std(preds)),
        "per_seed": per_seed,
    }
    out = f"gate_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[gate] {args.dataset}: disc={summary['disc_mean']:.4f}"
          f"±{summary['disc_std']:.4f}  pred={summary['pred_mean']:.4f}"
          f"±{summary['pred_std']:.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
