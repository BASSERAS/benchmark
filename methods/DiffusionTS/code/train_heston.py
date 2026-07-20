"""
Diffusion-TS (Yuan & Qiao, ICLR 2024) -- training + generation on the Heston target.

Trains one Diffusion-TS model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then samples 8192 length-128 paths and writes them back in price scale.

Reuses the ORIGINAL repo code verbatim: engine.solver.Trainer runs the exact
DDPM training loop (Adam betas [0.9,0.96], grad-clip 1.0, EMA,
ReduceLROnPlateauWithWarmup); Models...Diffusion_TS is the exact model; sampling
uses ema.ema_model.generate_mts via Trainer.sample. Only the data-loading and the
[0,1]<->price min-max wrapper live here (Heston paths are already windowed, so the
CSV-based Utils.Data_utils dataset is bypassed -- same pattern as methods/FourierFlow).

Normalisation chain (mirrors the repo's own auto_norm pipeline + a price wrapper):
  S(price) --minmax--> [0,1] --(*2-1)--> [-1,1]  (model trains here)
  sample --unnormalize_to_zero_to_one--> [0,1] --minmax invert--> price

Architectures (--arch): the three closest paper configs. They differ ONLY in
encoder/decoder depth and step count; everything else is identical across the paper's
own configs (d_model=64, timesteps=500, cosine, L1, n_heads=4, mlp_hidden_times=4).
  stocks : enc=2 dec=2  max_epochs=10000  (validated Stocks reproduction config)
  mujoco : enc=3 dec=3  max_epochs=12000  (paper's seq_len=100 config -- closest length)
  etth   : enc=3 dec=2  max_epochs=18000  (paper's Table-3 long-term-generation config)

Usage:
  PYTHONPATH=code/reference python train_heston.py --arch etth --seed 0
  PYTHONPATH=code/reference python train_heston.py --arch etth --seed 0 --steps 3000 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

METHOD = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(METHOD, "reference")
sys.path.insert(0, REF)

from engine.solver import Trainer
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

DEFAULT_DATA = os.path.join(METHOD, "..", "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")

# --- the three candidate architectures (model depth + training length) ---
ARCH = {
    "stocks": {"n_layer_enc": 2, "n_layer_dec": 2, "max_epochs": 10000, "patience": 2000},
    "mujoco": {"n_layer_enc": 3, "n_layer_dec": 3, "max_epochs": 12000, "patience": 3000},
    "etth":   {"n_layer_enc": 3, "n_layer_dec": 2, "max_epochs": 18000, "patience": 4000},
}
# shared model params (identical across the paper's stocks/mujoco/etth configs)
SHARED_MODEL = dict(d_model=64, timesteps=500, sampling_timesteps=500, loss_type="l1",
                    beta_schedule="cosine", n_heads=4, mlp_hidden_times=4,
                    attn_pd=0.0, resid_pd=0.0, kernel_size=1, padding_size=0)


class CaptureLogger:
    """Minimal logger implementing the interface engine.solver.Trainer expects,
    so the verbatim training loop runs unchanged while we record the loss curve."""
    def __init__(self):
        self.records = []  # (step, loss)

    def log_info(self, msg, **kwargs):
        print(msg, flush=True)

    def add_scalar(self, tag=None, scalar_value=None, global_step=None, **kwargs):
        if tag == "train/loss":
            self.records.append((int(global_step), float(scalar_value)))


class Args:
    def __init__(self, name):
        self.name = name


def minmax_fit(X):
    return float(X.min()), float(X.max())


def minmax_apply(X, lo, hi):
    return (X - lo) / (hi - lo)


def minmax_invert(Xn, lo, hi):
    return Xn * (hi - lo) + lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="etth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=0, help="override max_epochs (0 = use arch default)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training paths to use")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="", help="run tag (e.g. 'smoke'); prefixes output dirs")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.cuda.set_device(a.gpu)
    device = torch.device(f"cuda:{a.gpu}")

    arch = ARCH[a.arch]
    steps = a.steps if a.steps > 0 else arch["max_epochs"]
    tagp = (a.tag + "_") if a.tag else ""

    # --- data ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)  # (8192, 128) price
    if a.frac < 1.0:
        n = int(round(S.shape[0] * a.frac))
        S = S[:n]
    seq_len = S.shape[1]
    lo, hi = minmax_fit(S)
    Xn = minmax_apply(S, lo, hi)                 # [0,1]
    Xt = (Xn * 2.0 - 1.0).astype(np.float32)     # [-1,1], model space
    Xt = Xt[:, :, None]                          # (N, T, 1)
    ds = TensorDataset(torch.from_numpy(Xt))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    class _Unwrap:
        """Trainer does next(dl).to(device); yield the bare tensor, not a 1-tuple."""
        def __init__(self, dl): self.dl = dl
        def __iter__(self):
            for (x,) in self.dl:
                yield x
        def __len__(self): return len(self.dl)
    dl_u = _Unwrap(dl)

    # --- model (verbatim Diffusion_TS) ---
    model = Diffusion_TS(seq_length=seq_len, feature_size=1,
                         n_layer_enc=arch["n_layer_enc"], n_layer_dec=arch["n_layer_dec"],
                         **SHARED_MODEL).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{a.arch} seed {a.seed}] params={nparam} steps={steps} N={S.shape[0]} T={seq_len}", flush=True)

    # --- output dirs ---
    weights_dir = os.path.join(METHOD, "..", "weights")
    losses_dir = os.path.join(METHOD, "..", "losses")
    gen_dir = os.path.join(METHOD, "..", "generated_paths", f"seed_{a.seed}")
    ckpt_dir = os.path.join(METHOD, "..", "weights", f"{tagp}ckpt_{a.arch}_seed{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- config for the verbatim Trainer ---
    save_cycle = max(1, steps // 10)
    config = {"solver": {
        "max_epochs": steps,
        "gradient_accumulate_every": 2,
        "save_cycle": save_cycle,
        "results_folder": ckpt_dir,      # Trainer appends f"_{seq_len}"
        "base_lr": 1.0e-5,
        "ema": {"decay": 0.995, "update_interval": 10},
        "scheduler": {
            "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
            "params": {"factor": 0.5, "patience": arch["patience"], "min_lr": 1.0e-5,
                       "threshold": 1.0e-1, "threshold_mode": "rel",
                       "warmup_lr": 8.0e-4, "warmup": 500, "verbose": False},
        },
    }}

    logger = CaptureLogger()
    trainer = Trainer(config=config, args=Args(f"heston_{a.arch}_s{a.seed}"),
                      model=model, dataloader={"dataloader": dl_u}, logger=logger)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # --- loss curve ---
    # Canonical runs use the GUIDELINE naming (seed_{i}_losses.csv); tagged
    # (smoke) runs keep a distinct prefix so they never collide with real runs.
    loss_name = f"{tagp}{a.arch}_seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"])
        w.writerows(logger.records)
    first_nan = next((s for s, l in logger.records if not np.isfinite(l)), None)
    min_loss = min((l for _, l in logger.records), default=float("nan"))

    # --- generate 8192 paths (timed) ---
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    samples = trainer.sample(num=gen_n, size_every=2001, shape=[seq_len, 1])  # (>=N,T,1), [-1,1]
    samples = samples[:gen_n]                                                # exact count (sample() overshoots)
    samples = unnormalize_to_zero_to_one(samples)                            # [0,1]
    Xg = minmax_invert(samples[:, :, 0], lo, hi)                             # (N,T) price
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- final weights + config + metadata (GUIDELINE §4.3 schema for canonical runs) ---
    if not a.tag:  # only persist canonical weights/config for real runs
        torch.save({"model": model.state_dict(), "ema": trainer.ema.state_dict(),
                    "arch": a.arch, "seed": a.seed, "minmax": [lo, hi]},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"arch": a.arch, "n_layer_enc": arch["n_layer_enc"], "n_layer_dec": arch["n_layer_dec"],
               "feature_size": 1, "seq_length": seq_len, "max_epochs": steps,
               "batch_size": a.batch_size, "patience": arch["patience"], "seed": a.seed,
               **SHARED_MODEL, "base_lr": 1.0e-5, "ema_decay": 0.995,
               "gradient_accumulate_every": 2}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    meta = {"method": "DiffusionTS", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "scale_min": lo, "scale_max": hi, "params": int(nparam), "steps": steps,
            "min_loss": float(min_loss), "first_nan_step": first_nan,
            "gen_has_nan": gen_has_nan, "arch": a.arch}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
