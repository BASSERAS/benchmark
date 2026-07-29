"""
CSDI on Heston with NO preprocessing -- the original global-standardize pipeline (4096 paths).

This is the control run for the SBTS log-return preprocessing study. It uses the EXACT
released CSDI architecture and training loop as ``methods/CSDI/code/train_heston.py`` (imports
the reference ``CSDI_Heston`` subclass + ``BASE_CONFIG`` verbatim -- the paper's is_unconditional
diffusion backbone). The ONLY difference vs the with-preprocessing run
(``../../code/train_csdi_logret.py``) is the scaler:

  this file (baseline) :  price --(X-mu)/sigma--> N(0,1) space --train--> sample --*sigma+mu--> price
  logret variant       :  price --SBTS log-return scaling + unit-var wrapper--> R~ space --train-->
                          sample --inverse-scale + cumulative-exp anchored at 100--> price

So this run isolates the effect of the preprocessing: same data (heston_S_4096x128.npy),
same 4096 training paths, same 200 epochs, same seed, same model, same optimizer/scheduler --
just the plain global standardization the original CSDI uses on price paths.

Trains on ../../../../../dataset/Heston/preprocessing_with_log_returns/heston_S_4096x128.npy
and writes the methods-style folders (weights/ losses/ generated_paths/) under
results/Heston/preprocessing_with_log_returns/CSDI/baseline_no_preproc/.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_csdi_raw.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_csdi_raw.py --seed 0 --frac 0.02 --epochs 2 --tag smoke
"""
import os
import sys
import csv
import json
import time
import random
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))                        # .../baseline_no_preproc/code
BASE_DIR = os.path.dirname(HERE)                                         # .../baseline_no_preproc
CSDI_DIR = os.path.dirname(BASE_DIR)                                     # .../CSDI
# results/Heston/preprocessing_with_log_returns/CSDI -> benchmark/
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CSDI_DIR))))

# import the UNTOUCHED reference CSDI trainer (subclass + config)
REF_TRAIN = os.path.join(BENCH_ROOT, "methods", "CSDI", "code")
sys.path.insert(0, REF_TRAIN)
from train_heston import CSDI_Heston, BASE_CONFIG                        # noqa: E402

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
DEFAULT_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=200,
                    help="canonical CSDI = 200 (GUIDELINE M1); MUST match the with-preproc runs")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gen_num", type=int, default=4096)
    ap.add_argument("--gen_batch", type=int, default=512)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths to use (smoke test)")
    ap.add_argument("--tag", default="",
                    help="run tag (e.g. 'smoke'); prefixes output names, skips canonical weights")
    ap.add_argument("--out", default=BASE_DIR, help="output base dir (weights/losses/generated_paths)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (a.tag + "_") if a.tag else ""

    cfg = json.loads(json.dumps(BASE_CONFIG))  # deep copy of the paper's config
    lr = cfg["train"]["lr"]

    # --- data: ORIGINAL global standardization (no preprocessing) ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)      # (4096, 128) price
    if a.frac < 1.0:
        n = int(round(S.shape[0] * a.frac))
        S = S[:n]
    N, seq_len = S.shape
    mu = float(S.mean())
    sigma = float(S.std())
    X_train = (S - mu) / sigma                                    # (N, 128) std == 1
    Xt = X_train.astype(np.float32)[:, :, None]                   # (N, L, K=1)
    print(f"=== CSDI-raw (no-preproc) Heston  seed={a.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===",
          flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"mu={mu:.6f} sigma={sigma:.6f} -> X_train.std={X_train.std():.4f}  "
          f"epochs={a.epochs}", flush=True)

    ds = TensorDataset(torch.from_numpy(Xt))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    # --- model (verbatim CSDI, unconditional) ---
    model = CSDI_Heston(cfg, device, target_dim=1).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam} epochs={a.epochs} batch={a.batch_size} "
          f"N={N} T={seq_len} num_steps={cfg['diffusion']['num_steps']}", flush=True)

    # --- output dirs (methods-style) ---
    weights_dir = os.path.join(a.out, "weights")
    losses_dir = os.path.join(a.out, "losses")
    gen_dir = os.path.join(a.out, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- training loop (CSDI utils.train recipe: Adam wd=1e-6, MultiStepLR @0.75/0.9 x0.1) ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1.0e-6)
    p1, p2 = int(0.75 * a.epochs), int(0.9 * a.epochs)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    records = []          # (step, loss)
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        for (xb,) in dl:
            optimizer.zero_grad()
            loss = model({"observed_data": xb})
            loss.backward()
            optimizer.step()
            records.append((step, float(loss.item())))
            step += 1
        lr_sched.step()
        if epoch % max(1, a.epochs // 20) == 0 or epoch == a.epochs - 1:
            recent = np.mean([l for _, l in records[-len(dl):]]) if len(dl) else float("nan")
            print(f"  epoch {epoch:4d}/{a.epochs}  avg_loss={recent:.5f}  "
                  f"lr={lr_sched.get_last_lr()[0]:.2e}", flush=True)
    train_time = time.time() - t0
    first_nan = next((s for s, l in records if not np.isfinite(l)), None)
    min_loss = min((l for _, l in records), default=float("nan"))

    # --- loss curve (step,loss schema -- CSDI native) ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"])
        w.writerows(records)

    # --- generate gen_n paths (unconditional), standardized scale ---
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    Xg_std = model.generate(gen_n, seq_len, gen_batch=a.gen_batch)   # (gen_n, L) unit-var scale
    gen_time = time.time() - g0

    # --- inverse: destandardize directly to price (original CSDI decode) ---
    Xg = (Xg_std * sigma + mu).astype(np.float64)                   # (gen_n, 128) price
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_{gen_n}x{seq_len}.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- canonical weights/config (GUIDELINE schema) for real runs only ---
    if not a.tag:
        torch.save({"model": model.state_dict(), "seed": a.seed, "mu": mu, "sigma": sigma},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        out_cfg = {"method": "CSDI",
                   "variant": "CSDI (released is_unconditional) + NO preprocessing (global standardize)",
                   "is_unconditional": 1, "target_dim": 1, "seq_length": seq_len,
                   "epochs": a.epochs, "batch_size": a.batch_size, "lr": lr,
                   **cfg["diffusion"], "timeemb": cfg["model"]["timeemb"],
                   "featureemb": cfg["model"]["featureemb"], "seed": a.seed, "n_train": N,
                   "scaler": "global_standardize", "mu": mu, "sigma": sigma,
                   "params": int(nparam)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(out_cfg, f, indent=2)

    # --- metadata (GUIDELINE 4.3 schema) ---
    meta = {"method": "CSDI", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "mu": mu, "sigma": sigma,
            "preproc": "none_global_standardize",
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "epochs": a.epochs,
            "batch_size": a.batch_size, "num_steps": cfg["diffusion"]["num_steps"],
            "min_loss": float(min_loss), "first_nan_step": first_nan,
            "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} epochs={a.epochs} min_loss={min_loss:.5f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
