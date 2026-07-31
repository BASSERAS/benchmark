"""TimeDiT on Heston with SBTS log-return preprocessing (4096 paths).

Same from-scratch TimeDiT (DiT-S) architecture and locked recipe as
``methods/TimeDiT/code/train_heston.py`` (imports the verbatim reference modules
``timedit_model.build_timedit`` and ``gaussian_diffusion.GaussianDiffusion``).
The ONLY differences vs the original are (a) the SBTS log-return front-end in place
of the raw-price input, (b) 4096 train/gen paths instead of 8192.

Normalisation chain (Theo-approved: keep TimeDiT's native minmax->znorm, but apply
it to the SBTS log-return panel X_sbts instead of raw price):

  original :  price --minmax[0,1]--> --znorm(mu,sd)--> model
              sample --znorm^-1--> clip[0,1] --minmax^-1--> price
  this file:  price --SBTS(log-ret /sigma *sqrt(dt), dummy-0 col)--> X_sbts
              --minmax[0,1]--> --znorm(mu,sd)--> model
              sample --znorm^-1--> clip[0,1] --minmax^-1--> X_sbts space
              --drop dummy col, *sigma/sqrt(dt), cumsum-exp anchored at 100--> price

SBTS preprocessing (verbatim from ``methods/SBTS/code/sbts_generate.py``):
  R        = log(S[:,1:]/S[:,:-1])                    # (M,127) raw log-returns
  sigma    = float(R.std())                           # pooled over M x 127, TRAIN split
  R_tilde  = R * sqrt(dt) / sigma                      # (M,127), std == sqrt(dt)
  X_sbts   = hstack([zeros(M,1), R_tilde])            # (M,128) dummy-0 first column

Trains on ../../../../dataset/Heston/preprocessing_with_log_returns/heston_S_4096x128.npy
and writes the methods-style folders (weights/ losses/ generated_paths/) under
results/Heston/preprocessing_with_log_returns/TimeDiT/.

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_timedit_logret.py --seed 0
  CUDA_VISIBLE_DEVICES=1 python train_timedit_logret.py --seed 0 --steps 300 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))                        # .../TimeDiT/code
TIMEDIT_DIR = os.path.dirname(HERE)                                      # .../TimeDiT
# results/Heston/preprocessing_with_log_returns/TimeDiT -> benchmark/
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(TIMEDIT_DIR))))

# import the UNTOUCHED from-scratch TimeDiT modules used for the main-benchmark run
REFERENCE = os.path.join(BENCH_ROOT, "methods", "TimeDiT", "code")
sys.path.insert(0, REFERENCE)
from timedit_model import build_timedit            # noqa: E402
from gaussian_diffusion import GaussianDiffusion    # noqa: E402

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
DEFAULT_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")

DT = 1.0 / 250.0
S0 = 100.0

# ---- locked recipe (GATE winner) -- verbatim from methods/TimeDiT/code/train_heston.py ----
RECIPE = dict(model_size="S", learn_sigma=False, schedule="linear", T=1000,
              sampler="ddpm_fixed", lr=3e-4, weight_decay=0.0, batch=256,
              steps=15000, grad_clip=1.0)


def scale_log_returns(S):
    """SBTS: price paths -> scaled log-returns + pooled sigma. Verbatim from SBTS."""
    R = np.log(S[:, 1:] / S[:, :-1])            # (M, T-1)
    sigma = float(R.std())
    R_tilde = (R * np.sqrt(DT) / sigma).astype(np.float64)
    return R_tilde, sigma


def znorm_fit(X01):
    """Fit z-norm on [0,1] data over (paths,time); return (mu, sd). Verbatim from train_heston."""
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
    ap.add_argument("--gen_num", type=int, default=4096)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths (smoke only)")
    ap.add_argument("--tag", default="", help="run tag (e.g. 'smoke'); prefixes outputs, skips canonical weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    tagp = (a.tag + "_") if a.tag else ""

    # ---- data: price -> SBTS log-return panel -> minmax[0,1] -> znorm ----
    S = np.load(os.path.abspath(a.data)).astype(np.float64)   # (4096, 128) price
    if a.frac < 1.0:
        S = S[:int(round(S.shape[0] * a.frac))]
    N, L = S.shape
    C = 1
    R_tilde, sigma = scale_log_returns(S)                     # (N,127), pooled sigma
    X_sbts = np.hstack([np.zeros((N, 1), dtype=np.float64), R_tilde])   # (N,128) dummy-0 col
    # TimeDiT's OWN native normalization, now applied to the log-return panel (not price):
    lo, hi = float(X_sbts.min()), float(X_sbts.max())
    X01 = (X_sbts - lo) / (hi - lo)                            # [0,1]
    mu, sd = znorm_fit(X01)                                    # fit on ALL train paths
    mu, sd = float(np.asarray(mu).reshape(-1)[0]), float(np.asarray(sd).reshape(-1)[0])
    Xn = ((X01 - mu) / sd).astype(np.float32)[:, :, None]      # (N, L, 1) model space
    train = torch.tensor(Xn, device=device)
    print(f"=== TimeDiT-logret Heston  seed={a.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={device} ===",
          flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"SBTS sigma={sigma:.8f}  R~[min={R_tilde.min():.4f},max={R_tilde.max():.4f}] "
          f"R~.std={R_tilde.std():.6f} (sqrt(dt)={np.sqrt(DT):.6f})  "
          f"X_sbts minmax=[{lo:.6f},{hi:.6f}]  znorm mu={float(mu):.6g} sd={float(sd):.6f}  "
          f"steps={a.steps}", flush=True)

    # ---- model + diffusion (verbatim from-scratch modules) ----
    model = build_timedit(L, C, model_size=RECIPE["model_size"],
                          learn_sigma=RECIPE["learn_sigma"]).to(device)
    diff = GaussianDiffusion(T=RECIPE["T"], schedule=RECIPE["schedule"],
                             loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"],
                           weight_decay=RECIPE["weight_decay"])
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[timedit seed {a.seed}] params={nparam} steps={a.steps} "
          f"N={N} L={L} batch={a.batch}", flush=True)

    # ---- output dirs ----
    weights_dir = os.path.join(TIMEDIT_DIR, "weights")
    losses_dir = os.path.join(TIMEDIT_DIR, "losses")
    gen_dir = os.path.join(TIMEDIT_DIR, "generated_paths", f"seed_{a.seed}")
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
        loss, parts = diff.training_loss(model, x0, learn_sigma=RECIPE["learn_sigma"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
        opt.step()
        lv = float(loss.item())
        if not np.isfinite(lv) and first_nan is None:
            first_nan = step
        if step % 100 == 0 or step == a.steps - 1:
            records.append((step, "diffusion", lv))
            print(f"[step {step:5d}] loss={lv:.6f}", flush=True)
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

    # ---- inverse: znorm^-1 -> clip[0,1] -> minmax^-1 -> X_sbts space -> SBTS inverse -> price ----
    X01_g = np.clip(fz[:, :, 0] * sd + mu, 0.0, 1.0)          # mu,sd (1,1) broadcast
    X_sbts_gen = X01_g * (hi - lo) + lo                       # (gen_n, 128) back to R~ panel
    R_tilde_gen = X_sbts_gen[:, 1:]                           # (gen_n, 127) drop dummy col
    R_gen = R_tilde_gen * sigma / np.sqrt(DT)                 # raw log-returns
    Xg = np.empty((gen_n, L), dtype=np.float64)
    Xg[:, 0] = S0
    Xg[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))         # price anchored at 100
    Xg = np.clip(Xg, 1e-6, None)
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_name = f"{tagp}generated_paths_{gen_n}x{L}.npy" if a.tag else f"generated_paths_{gen_n}x{L}.npy"
    np.save(os.path.join(gen_dir, out_name), Xg.astype(np.float64))

    # ---- weights + config + metadata (canonical runs only for weights/config) ----
    if not a.tag:
        save = {"model": model.state_dict(), "sbts_sigma": sigma, "dt": DT, "s0": S0,
                "minmax_lo": lo, "minmax_hi": hi, "znorm_mu": float(mu), "znorm_sd": float(sd),
                "seed": a.seed}
        torch.save(save, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "TimeDiT",
               "variant": "TimeDiT (DiT-S, paper recipe) + SBTS log-return preproc",
               "seed": a.seed, "model_size": RECIPE["model_size"],
               "hidden_size": 384, "depth": 12, "num_heads": 6,
               "learn_sigma": RECIPE["learn_sigma"], "schedule": RECIPE["schedule"],
               "T": RECIPE["T"], "sampler": RECIPE["sampler"], "lr": RECIPE["lr"],
               "weight_decay": RECIPE["weight_decay"], "ema": 0.0,
               "batch_size": a.batch, "n_steps": a.steps, "n_train": N,
               "seq_len": L, "feature_size": C, "dataset": "Heston",
               "paper_hyperparams": True,
               "scaler": "sbts_log_return+minmax+znorm",
               "sbts_sigma": sigma, "dt": DT, "s0": S0,
               "minmax": [lo, hi], "znorm_mu": float(mu), "znorm_sd": float(sd),
               "params": int(nparam)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    meta = {"method": "TimeDiT", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "sbts_sigma": float(sigma), "minmax_lo": lo, "minmax_hi": hi,
            "znorm_mu": float(mu), "znorm_sd": float(sd),
            "preproc": "sbts_log_return+minmax+znorm",
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "steps": a.steps, "min_loss": float(min_loss),
            "first_nan_step": first_nan, "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} steps={a.steps} min_loss={min_loss:.6f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
