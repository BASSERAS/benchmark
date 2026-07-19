#!/usr/bin/env python
"""Reproduce the SBTS paper (arXiv 2503.02943, Alouadi et al., ICAIF'25) Stocks
result using the *official* SBTS code, verbatim from the demo notebook.

Paper Table 1 (Stocks), SBTS row:
    Discriminative score = 0.010 +/- 0.008
    Predictive   score  = 0.017 +/- 0.000

Pipeline (identical to reference/experiments_demo.ipynb "Stock Dataset" cells):
  1. Load metrics/fbm_stock_metrics/data/X_stock.pt   -> (1002, 10, 5) log-returns
  2. Generate synthetic log-returns with simulateSB_multi(N_pi=100, h=0.2)
  3. Score with get_scores(itt=2000, n_temp=10, min_max=True) -- the exact
     discriminative/predictive protocol described in the paper (GRU disc:
     2 layers, hidden=4; GRU pred: 1 layer, hidden=max(d/2,1); 2000 epochs,
     batch 128, |acc-0.5|, 10 runs, TSTR).

Run:  CUDA_VISIBLE_DEVICES=3 python reproduce_stock.py
"""
import os
import sys
import json
import time

import numpy as np
import torch

# --- wire up the official reference package exactly like the notebook ---------
REF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference")
)
sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(REF, "metrics"))

from models.sbts_multi import simulateSB_multi          # noqa: E402
from metrics.eval_functions import get_scores, get_stats  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def main():
    seed = 987
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_ids = [0]  # single visible GPU -> logical index 0
    print(f"device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    # 1) real Stock log-returns (official preprocessed tensor) -----------------
    xpath = os.path.join(REF, "metrics", "fbm_stock_metrics", "data", "X_stock.pt")
    X_stock = torch.load(xpath, map_location="cpu").to(torch.float32)  # (1002,10,5)
    M, N, d = X_stock.shape
    print(f"X_stock shape = {tuple(X_stock.shape)}", flush=True)

    # notebook prepends a zero row so log_returns has N+1 steps
    log_returns = np.zeros((M, N + 1, d))
    log_returns[:, 1:] = X_stock.numpy()

    # 2) SBTS generation (kernel, numba) -- verbatim notebook settings ---------
    deltati = 1 / 252
    h = 0.2
    t0 = time.perf_counter()
    X_stock_sbts = simulateSB_multi(
        N, M, d, X=log_returns, N_pi=100, h=h, deltati=deltati, M_simu=1000
    )
    gen_sec = time.perf_counter() - t0
    print(f"generated SBTS {X_stock_sbts.shape} in {gen_sec:.1f}s", flush=True)

    # persist generated synthetic paths (the "weights"/output of a kernel model)
    np.save(os.path.join(DATA, "X_stock_real_logret.npy"), X_stock.numpy())
    np.save(os.path.join(DATA, "X_stock_sbts_logret.npy"), X_stock_sbts)

    # distribution sanity table
    stats = get_stats(X_stock.numpy(), X_stock_sbts)
    stats.to_csv(os.path.join(OUT, "stock_stats.csv"))

    # 3) paper-protocol discriminative + predictive scores ---------------------
    # real has 1002 samples, fake 1000 -> get_scores handles the sub-sampling
    disc, pred = get_scores(
        X_stock.numpy(), X_stock_sbts,
        itt=2000, n_temp=10, min_max=True, device=device, device_ids=dev_ids,
    )

    result = {
        "dataset": "Stocks",
        "source": "official SBTS code (reference/), demo-notebook Stock pipeline",
        "generation": {"model": "simulateSB_multi", "N_pi": 100, "h": h,
                        "deltati": deltati, "M_simu": 1000, "gen_sec": gen_sec},
        "protocol": {"itt": 2000, "n_temp": 10, "min_max": True,
                     "disc": "GRU 2-layer hidden=4, |acc-0.5|",
                     "pred": "GRU 1-layer hidden=max(d/2,1), MAE TSTR"},
        "discriminative_score": {"mean": float(disc.mean()), "std": float(disc.std()),
                                  "runs": disc.tolist()},
        "predictive_score": {"mean": float(pred.mean()), "std": float(pred.std()),
                             "runs": pred.tolist()},
        "paper_reference": {"discriminative": "0.010 +/- 0.008",
                            "predictive": "0.017 +/- 0.000"},
        "seed": seed,
    }
    with open(os.path.join(OUT, "sbts_stock_scores.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== SBTS Stocks reproduction ====")
    print(f"  disc  ours = {disc.mean():.3f} +/- {disc.std():.3f}   paper = 0.010 +/- 0.008")
    print(f"  pred  ours = {pred.mean():.3f} +/- {pred.std():.3f}   paper = 0.017 +/- 0.000")
    print("saved -> results/sbts_stock_scores.json")


if __name__ == "__main__":
    main()
