#!/usr/bin/env python
"""Reproduce the TimeGAN paper (Yoon et al., NeurIPS 2019) Stocks result.

Paper Table 1 (Stocks), TimeGAN row (as reported in the paper and re-quoted in
the SBTS benchmark, arXiv 2503.02943):
    Discriminative score = 0.102 +/- 0.031
    Predictive   score  = 0.038 +/- 0.001

IMPORTANT — reproduction caveat
-------------------------------
The *official* TimeGAN code (reference/main_timegan.py, timegan.py,
metrics/discriminative_metrics.py) is TensorFlow 1.x and cannot run on this
CUDA-13 machine (TF1 has no CUDA-13 build; it silently falls back to CPU for
50k*3 iterations = many hours). This is the documented reason the benchmark
ships a PyTorch port. We therefore reproduce with:
  * generator   = our faithful PyTorch TimeGAN port (code/timegan_torch.py),
                  trained with the paper hyperparameters below;
  * metric      = the SBTS PyTorch port of the *same* Yoon-et-al. disc/pred
                  protocol (reference discriminative_score.py / predictive_score.py),
                  so both TimeGAN and SBTS "our-run" columns share one metric.

Paper hyperparameters (reference/README.md example command):
    --data_name stock --seq_len 24 --module gru --hidden_dim 24
    --num_layer 3 --iteration 50000 --batch_size 128 --metric_iteration 10

Run:  CUDA_VISIBLE_DEVICES=1 python reproduce_stock.py
"""
import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
TG_CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
SBTS_REF = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "SBTS", "code", "reference")
)
sys.path.insert(0, TG_CODE)                              # timegan_torch
sys.path.insert(0, os.path.join(SBTS_REF, "metrics"))    # discriminative/predictive
sys.path.insert(0, SBTS_REF)

from timegan_torch import TimeGAN                              # noqa: E402
from metrics.eval_functions import get_scores                 # noqa: E402

STOCK_CSV = os.path.abspath(
    os.path.join(TG_CODE, "reference", "data", "stock_data.csv")
)
OUT = os.path.abspath(os.path.join(HERE, "..", "results"))
DATA = os.path.abspath(os.path.join(HERE, "..", "dataset"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

SEQ_LEN = 24


def min_max_scaler(data):
    num = data - np.min(data, 0)
    den = np.max(data, 0) - np.min(data, 0)
    return num / (den + 1e-7)


def real_data_loading(seq_len):
    """Verbatim port of reference/data_loading.py real_data_loading('stock')."""
    ori = np.loadtxt(STOCK_CSV, delimiter=",", skiprows=1)
    ori = ori[::-1]                       # chronological
    ori = min_max_scaler(ori)             # [0,1] per feature
    windows = [ori[i:i + seq_len] for i in range(len(ori) - seq_len)]
    idx = np.random.permutation(len(windows))
    return np.asarray([windows[i] for i in idx])  # (N, seq_len, d)


def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    real = real_data_loading(SEQ_LEN)            # (N, 24, 6), already [0,1]
    N, T, d = real.shape
    print(f"stock windows = {real.shape}", flush=True)

    # ---- train PyTorch TimeGAN with paper hyperparameters -------------------
    iters = 50000
    model = TimeGAN(
        n_features=d, hidden_dim=24, num_layers=3, batch_size=128,
        device="cuda", embedding_steps=iters, supervised_steps=iters,
        joint_steps=iters, log_every=1000,
    )
    t0 = time.perf_counter()
    model.fit(real)                              # fit re-min-max internally (no-op on [0,1])
    train_sec = time.perf_counter() - t0
    print(f"trained in {train_sec/60:.1f} min", flush=True)

    fake = model.sample(N)                       # (N, 24, 6), back to [0,1] scale
    if fake.ndim == 2:
        fake = fake[:, :, None]
    print(f"generated = {fake.shape}", flush=True)

    # persist generated data + weights
    np.save(os.path.join(DATA, "stock_real.npy"), real)
    np.save(os.path.join(DATA, "stock_timegan.npy"), fake)
    torch.save(model.state_dict(), os.path.join(OUT, "timegan_stock_model.pt"))
    with open(os.path.join(OUT, "timegan_stock_config.json"), "w") as f:
        json.dump(model.config(), f, indent=2)

    # ---- disc + pred scores (same protocol as SBTS reproduction) -----------
    disc, pred = get_scores(
        real, fake, itt=2000, n_temp=10, min_max=True,
        device=device, device_ids=[0],
    )

    result = {
        "dataset": "Stocks",
        "generator": "PyTorch TimeGAN port (code/timegan_torch.py)",
        "note": "official TimeGAN is TF1; cannot run on CUDA-13. Metric shared with SBTS.",
        "hyperparameters": {"seq_len": SEQ_LEN, "module": "gru", "hidden_dim": 24,
                            "num_layers": 3, "iteration_per_phase": iters,
                            "batch_size": 128, "train_min": train_sec / 60},
        "protocol": {"itt": 2000, "n_temp": 10, "min_max": True},
        "discriminative_score": {"mean": float(disc.mean()), "std": float(disc.std()),
                                  "runs": disc.tolist()},
        "predictive_score": {"mean": float(pred.mean()), "std": float(pred.std()),
                             "runs": pred.tolist()},
        "paper_reference": {"discriminative": "0.102 +/- 0.031",
                            "predictive": "0.038 +/- 0.001"},
        "seed": seed,
    }
    with open(os.path.join(OUT, "timegan_stock_scores.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== TimeGAN Stocks reproduction ====")
    print(f"  disc  ours = {disc.mean():.3f} +/- {disc.std():.3f}   paper = 0.102 +/- 0.031")
    print(f"  pred  ours = {pred.mean():.3f} +/- {pred.std():.3f}   paper = 0.038 +/- 0.001")
    print("saved -> results/timegan_stock_scores.json")


if __name__ == "__main__":
    main()
