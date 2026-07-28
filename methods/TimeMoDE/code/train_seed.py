"""TimeMoDE (arXiv:2606.15172) -- single-seed training + generation on Heston.

Trains ONE TimeMoDE model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then samples 8192 length-128 paths and writes them back in price scale.

EXACT SAME ARCHITECTURE AS THE SLC "FROM SCRATCH" SEED-0 REPRODUCTION.
This file imports `build_timemode` and `Diffusion` directly from
  ../paper_reimplementation/{timemode_model.py, diffusion.py}
so there is literally no way for the Heston model to differ from the model that
passed the reproduction gate (README.md sec 4/sec 7). Only the input geometry
changes: SLC used (L=24, C=1); Heston uses (L=128, C=1). Nothing else -- same
hidden 256 / depth 6 / heads 4 / K=8 top-2 / expansion 4 / T=250 config, same
optimiser (AdamW lr 1e-4 wd 1e-5), same batch 2048, 1000 epochs, EMA 0.9999
(decay 0 for first 100 steps), grad-clip 1.0, w_proto=w_aux=0.01.

Normalisation (mirrors the SLC pipeline exactly -- model trains in [0,1], NOT
[-1,1]; see run_paper.py which trains on [0,1] windows and clips fakes to [0,1]):
  S(price) --global minmax--> [0,1]   (model trains here)
  sample --clip[0,1]--> [0,1] --minmax invert--> price  (clip min 1e-6)

Usage (single seed on a chosen GPU):
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 0

Smoke (few epochs, sanity only -- does NOT overwrite canonical outputs):
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 0 --smoke
"""
import os
import sys
import csv
import json
import copy
import time
import argparse

import numpy as np
import torch

METHOD = os.path.dirname(os.path.abspath(__file__))          # .../TimeMoDE/code
PAPER = os.path.abspath(os.path.join(METHOD, "..", "paper_reimplementation"))
sys.path.insert(0, PAPER)

# The EXACT seed-0 model + diffusion -- imported, not reimplemented.
from timemode_model import build_timemode        # noqa: E402
from diffusion import Diffusion                  # noqa: E402

DEFAULT_DATA = os.path.join(METHOD, "..", "..", "..", "dataset", "Heston",
                            "heston_S_8192x128.npy")


def ema_update(ema_model, model, decay):
    """Identical to run_paper.py: EMA over params, hard-copy buffers."""
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def dp_batch(pool_t, n):
    """Domain-Prompt exemplars: a random batch of clean training windows
    (single-domain setting). Identical to run_paper.py."""
    idx = torch.randint(0, len(pool_t), (n,), device=pool_t.device)
    return pool_t[idx]


def minmax_fit(X):
    return float(X.min()), float(X.max())


def parse_args():
    p = argparse.ArgumentParser(description="TimeMoDE single-seed Heston training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=DEFAULT_DATA)
    # --- ALL defaults below are the SLC seed-0 values (paper Table 7/8) ---
    p.add_argument("--T", type=int, default=250)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--micro", type=int, default=256,
                   help="micro-batch for gradient accumulation; effective batch "
                        "stays --batch (the SLC seed-0 value). MoDE evaluates all "
                        "8 experts densely so memory scales with L; at L=128 the "
                        "full 2048 OOMs on 80GB, so we accumulate 8x256 instead.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--w_aux", type=float, default=0.01)
    p.add_argument("--w_proto", type=float, default=0.01)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--gen_num", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device if torch.cuda.is_available() else "cpu"

    # --- data: price -> global min-max [0,1] (model space, same as SLC) ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)   # (8192,128) price
    lo, hi = minmax_fit(S)
    Xn = ((S - lo) / (hi - lo)).astype(np.float32)            # [0,1]
    train = Xn[:, :, None]                                     # (N, L, 1)
    N, L, C = train.shape

    epochs = a.epochs
    if a.smoke:
        train = train[: max(int(len(train) * 0.05), 64)]
        N = len(train)
        epochs = 30
    batch = min(a.batch, N)
    micro = min(a.micro, batch)
    steps_per_epoch = max(N // batch, 1)
    print(f"[data] train {train.shape} price=[{S.min():.3f},{S.max():.3f}] "
          f"norm=[{train.min():.3f},{train.max():.3f}] batch={batch} "
          f"micro={micro} epochs={epochs}", flush=True)

    train_t = torch.tensor(train, device=device)

    # --- model + diffusion: EXACT seed-0 build (only L/C differ) ---
    model = build_timemode(L, C, learn_sigma=False).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(T=a.T, schedule=a.schedule)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TimeMoDE params={n_params/1e6:.2f}M seq_len={L} C={C}", flush=True)

    # --- train (identical loop to run_paper.py) ---
    records = []          # (step, loss_total, simple, proto, aux)
    loss_log = []
    model.train()
    t0 = time.time()
    gstep = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            # One optimiser step over an EFFECTIVE batch of `batch` samples,
            # accumulated over micro-batches of `micro`. Each micro-loss is
            # weighted by (mbs/batch) before backward so the summed gradient
            # equals the mean over the full effective batch -- mathematically
            # identical to a single batch=`batch` step (LayerNorm/AdaLN model,
            # no BatchNorm, so per-sample stats are unaffected by the split).
            opt.zero_grad()
            parts = {"simple": 0.0, "proto": 0.0, "aux": 0.0}
            loss_total = 0.0
            done = 0
            while done < batch:
                mbs = min(micro, batch - done)
                w = mbs / batch
                idx = np.random.randint(0, N, mbs)
                x0 = train_t[idx]
                dpx = dp_batch(train_t, mbs)
                mloss, mparts = diff.training_loss(model, x0, dpx,
                                                   w_proto=a.w_proto, w_aux=a.w_aux)
                (mloss * w).backward()
                loss_total += float(mloss.item()) * w
                for k in parts:
                    parts[k] += mparts[k] * w
                done += mbs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            decay = 0.0 if gstep < 100 else a.ema_decay
            ema_update(ema, model, decay)
            if not np.isfinite(loss_total):
                raise RuntimeError(f"NaN/Inf loss at step {gstep}: {parts}")
            records.append((gstep, loss_total,
                            parts["simple"], parts["proto"], parts["aux"]))
            gstep += 1
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            loss_log.append({"epoch": epoch, "step": gstep, "loss": loss_total, **parts})
            print(f"[ep {epoch:4d} step {gstep:6d}] loss={loss_total:.4f} "
                  f"simple={parts['simple']:.4f} proto={parts['proto']:.2f} "
                  f"aux={parts['aux']:.2f} ({time.time()-t0:.1f}s)", flush=True)
    train_time = time.time() - t0

    # --- generate 8192 paths (EMA model, same sampler as SLC) ---
    ema.eval()
    fakes = []
    g0 = time.time()
    with torch.no_grad():
        remaining = a.gen_num
        while remaining > 0:
            b = min(1024, remaining)
            dpx = dp_batch(train_t, b)
            x = diff.p_sample_loop(ema, (b, L, C), device, dpx)
            fakes.append(x.cpu().numpy())
            remaining -= b
    fake = np.concatenate(fakes, 0)[: a.gen_num]
    fake = np.clip(fake, 0.0, 1.0).astype(np.float64)        # [0,1], same as SLC
    gen_time = time.time() - g0

    # --- invert to price scale (clip min 1e-6 -> strictly positive prices) ---
    Xg = fake[:, :, 0] * (hi - lo) + lo                      # (N, L) price
    Xg = np.clip(Xg, 1e-6, None).astype(np.float64)
    gen_has_nan = bool(not np.isfinite(Xg).all())
    print(f"[gen] fake {Xg.shape} price=[{Xg.min():.3f},{Xg.max():.3f}]", flush=True)

    # --- output dirs ---
    weights_dir = os.path.join(METHOD, "..", "weights")
    losses_dir = os.path.join(METHOD, "..", "losses")
    gen_dir = os.path.join(METHOD, "..", "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    tagp = "smoke_" if a.smoke else ""

    # --- loss curve (GUIDELINE naming: seed_{i}_losses.csv for canonical runs) ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv"
    with open(os.path.join(losses_dir, loss_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss_total", "simple", "proto", "aux"])
        w.writerows(records)
    first_nan = next((s for s, lt, *_ in records if not np.isfinite(lt)), None)
    min_loss = min((lt for _, lt, *_ in records), default=float("nan"))

    # --- generated paths ---
    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg)

    # --- weights + config + metadata (GUIDELINE sec 4.3; canonical runs only) ---
    if not a.smoke:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "arch": "timemode", "seed": a.seed, "minmax": [lo, hi]},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"arch": "timemode", "hidden": 256, "depth": 6, "heads": 4,
               "experts": 8, "top_k": 2, "shared_expert": 1, "expansion": 4,
               "T": a.T, "schedule": a.schedule, "patch_size": 1,
               "learn_sigma": False, "feature_size": C, "seq_length": L,
               "epochs": epochs, "batch": batch, "lr": a.lr, "wd": a.wd,
               "w_proto": a.w_proto, "w_aux": a.w_aux, "ema_decay": a.ema_decay,
               "seed": a.seed, "params": int(n_params)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    meta = {"method": "TimeMoDE", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "scale_min": lo, "scale_max": hi, "params": int(n_params),
            "steps": gstep, "epochs": epochs,
            "min_loss": float(min_loss), "first_nan_step": first_nan,
            "gen_has_nan": gen_has_nan, "arch": "timemode",
            "loss_log": loss_log}
    meta_name = f"{tagp}metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "loss_log"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
