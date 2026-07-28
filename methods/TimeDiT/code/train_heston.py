"""TimeDiT (arXiv:2409.02322) — training + generation on the Heston target.

Trains one from-scratch TimeDiT (DiT-S) diffusion model per seed on the 8192x128
Heston price paths in ../../../dataset/Heston/heston_S_8192x128.npy, then samples
8192 length-128 paths and writes them back in price scale.

RECIPE — locked to the paper-reproduction GATE winner (Sine + Stock, seq_len=24).
We replicate the paper's own synthetic-generation setting; we do NOT tune on Heston
(GUIDELINE Q4). The winning operating point, validated in paper_reimplementation/
(gate_sine.json / gate_stock.json), is:

    model      : DiT-S  (hidden=384, depth=12, heads=6)   [paper Table 9 "S"]
    variance   : learn_sigma=False  (L_simple objective)
    schedule   : linear, T=1000                            [DDPM]
    sampler    : ddpm_fixed  (ancestral, fixed posterior variance)
    optimiser  : Adam lr=3e-4, weight_decay=0, grad-clip 1.0, NO EMA
    batch      : 256
    steps      : 15000  (full-length training)

Normalisation chain (Heston price -> model space -> price):
  S(price) --minmax--> [0,1] --znorm(mu,sd over train)--> model space
  sample --znorm^-1 (clip[0,1])--> [0,1] --minmax invert--> price  (clip >= 1e-6)

The model + diffusion are the verbatim from-scratch modules used for the paper
reproduction (code/timedit_model.py, code/gaussian_diffusion.py). Only the Heston
data loader and the price<->[0,1] wrapper live here.

Usage:
  python train_heston.py --seed 0 --gpu 0
  python train_heston.py --seed 0 --gpu 0 --steps 500 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from timedit_model import build_timedit           # noqa: E402
from gaussian_diffusion import GaussianDiffusion   # noqa: E402

METHOD = os.path.join(HERE, "..")
DEFAULT_DATA = os.path.join(METHOD, "..", "..", "dataset", "Heston",
                            "heston_S_8192x128.npy")

# ---- locked recipe (GATE winner) ----
RECIPE = dict(model_size="S", learn_sigma=False, schedule="linear", T=1000,
              sampler="ddpm_fixed", lr=3e-4, weight_decay=0.0, batch=256,
              steps=15000, grad_clip=1.0)


def znorm_fit(X01):
    """Fit z-norm on [0,1] data over (paths,time); return (mu, sd)."""
    mu = X01.mean(axis=(0, 1), keepdims=True)
    sd = X01.std(axis=(0, 1), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=RECIPE["steps"])
    ap.add_argument("--batch", type=int, default=RECIPE["batch"])
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths (smoke only)")
    ap.add_argument("--tag", default="", help="run tag (e.g. 'smoke'); prefixes outputs")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    tagp = (a.tag + "_") if a.tag else ""

    # ---- data: price -> minmax[0,1] -> znorm ----
    S = np.load(os.path.abspath(a.data)).astype(np.float64)   # (8192, 128) price
    if a.frac < 1.0:
        S = S[:int(round(S.shape[0] * a.frac))]
    N, L = S.shape
    C = 1
    lo, hi = float(S.min()), float(S.max())
    X01 = (S - lo) / (hi - lo)                                # [0,1]
    mu, sd = znorm_fit(X01)                                   # fit on ALL train paths
    Xn = ((X01 - mu) / sd).astype(np.float32)[:, :, None]     # (N, L, 1) model space
    train = torch.tensor(Xn, device=device)

    # ---- model + diffusion (verbatim from-scratch modules) ----
    model = build_timedit(L, C, model_size=RECIPE["model_size"],
                          learn_sigma=RECIPE["learn_sigma"]).to(device)
    diff = GaussianDiffusion(T=RECIPE["T"], schedule=RECIPE["schedule"],
                             loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"],
                           weight_decay=RECIPE["weight_decay"])
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[timedit seed {a.seed}] params={nparam} steps={a.steps} "
          f"N={N} L={L} batch={a.batch} device={device}", flush=True)

    # ---- output dirs ----
    weights_dir = os.path.join(METHOD, "weights")
    losses_dir = os.path.join(METHOD, "losses")
    gen_dir = os.path.join(METHOD, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # ---- training ----
    records = []          # (step, phase, loss_total)
    first_nan = None
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, N, (a.batch,), device=device)
        x0 = train[idx]
        loss, parts = diff.training_loss(model, x0,
                                         learn_sigma=RECIPE["learn_sigma"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
        opt.step()
        lv = float(loss.item())
        if not np.isfinite(lv) and first_nan is None:
            first_nan = step
        if step % 100 == 0 or step == a.steps - 1:
            records.append((step, "diffusion", lv))
    train_time = time.time() - t0

    # ---- loss csv (GUIDELINE §4.4: step, phase, loss_total) ----
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    with open(os.path.join(losses_dir, loss_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "phase", "loss_total"])
        w.writerows(records)
    min_loss = min((l for _, _, l in records), default=float("nan"))

    # ---- generate paths (ddpm_fixed, batched) ----
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    model.eval()
    g0 = time.time()
    fakes = []
    remaining = gen_n
    with torch.no_grad():
        while remaining > 0:
            b = min(1000, remaining)
            x = diff.p_sample_loop(model, (b, L, C), device,
                                   learn_sigma=RECIPE["learn_sigma"],
                                   sample_var="fixed")
            fakes.append(x.cpu().numpy())
            remaining -= b
    fz = np.concatenate(fakes, 0)[:gen_n]                     # (gen_n, L, 1) model space
    gen_time = time.time() - g0

    # ---- denorm: znorm^-1 -> [0,1] -> price ----
    X01_g = np.clip(fz[:, :, 0] * sd + mu, 0.0, 1.0)          # mu,sd are (1,1) -> broadcast
    Xg = X01_g * (hi - lo) + lo                               # (gen_n, L) price
    Xg = np.clip(Xg, 1e-6, None).astype(np.float64)
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_name = f"{tagp}generated_paths_8192x128.npy" if a.tag else "generated_paths_8192x128.npy"
    np.save(os.path.join(gen_dir, out_name), Xg)

    # ---- weights + config + metadata (canonical runs only for weights/config) ----
    if not a.tag:
        torch.save(model.state_dict(),
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "TimeDiT", "seed": a.seed, "model_size": RECIPE["model_size"],
               "hidden_size": 384, "depth": 12, "num_heads": 6,
               "learn_sigma": RECIPE["learn_sigma"], "schedule": RECIPE["schedule"],
               "T": RECIPE["T"], "sampler": RECIPE["sampler"], "lr": RECIPE["lr"],
               "weight_decay": RECIPE["weight_decay"], "ema": 0.0,
               "batch_size": a.batch, "n_steps": a.steps, "n_train": N,
               "seq_len": L, "feature_size": C, "dataset": "Heston",
               "paper_hyperparams": True, "minmax": [lo, hi]}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    meta = {"method": "TimeDiT", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "gen_time_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "scale_min": lo, "scale_max": hi, "params": int(nparam),
            "steps": a.steps, "min_loss": float(min_loss),
            "first_nan_step": first_nan, "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
