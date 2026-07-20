"""
TimeVAE (Desai, Freeman, Beaver, Wang -- arXiv:2111.08095v3) -- training +
generation on the Heston target.

Trains one TimeVAE-Base model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then prior-generates 8192 length-128 paths and writes them back in price scale.

Reuses our faithful PyTorch port ``timevae_torch.py`` (architecture + VAE loss
matched line-for-line to the released TF code ``code/reference/src/vae/*``) with the
released hyperparameters from ``code/reference/src/config/hyperparameters.yaml``
(``timeVAE`` preset), the SAME preset that reproduced the paper's sine table:

    latent_dim=8, hidden_layer_sizes=[50,100,200], reconstruction_wt=3.0,
    batch_size=16, use_residual_conn=True, trend_poly=0, custom_seas=None
    Adam(lr=1e-3), max_epochs=1000 + EarlyStopping(patience=50, min_delta=1e-2) +
    ReduceLROnPlateau(factor=0.5, patience=30)    [vae_base.fit_on_data]

Normalisation chain (mirrors the released ``vae_pipeline.run_vae_pipeline``):
  S(price) --MinMaxScaler(per-(t,feature))--> ~[0,1]  (model trains here)
  prior sample --inverse_transform--> price
The per-(t,feature) MinMaxScaler is the port of the reference ``data_utils.MinMaxScaler``
and is exactly the scaler used in the paper reproduction.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0 --frac 0.05 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch

METHOD_CODE = os.path.dirname(os.path.abspath(__file__))          # methods/TimeVAE/code
METHOD_DIR = os.path.dirname(METHOD_CODE)                          # methods/TimeVAE
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))          # benchmark/
sys.path.insert(0, METHOD_CODE)
from timevae_torch import TimeVAE, train_timevae, MinMaxScaler     # noqa: E402

DEFAULT_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_8192x128.npy")

# released ``timeVAE`` preset (config/hyperparameters.yaml) -- same as paper repro
PRESET = dict(latent_dim=8, hidden_layer_sizes=(50, 100, 200), reconstruction_wt=3.0,
              trend_poly=0, custom_seas=None, use_residual_conn=True)
TRAIN = dict(max_epochs=1000, batch_size=16, lr=1e-3,
             es_patience=50, es_min_delta=1e-2, rlr_patience=30, rlr_factor=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=0,
                    help="override max_epochs (0 = use preset default 1000)")
    ap.add_argument("--batch_size", type=int, default=TRAIN["batch_size"])
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths to use (smoke test)")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--tag", default="",
                    help="run tag (e.g. 'smoke'); prefixes output names, skips canonical weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    epochs = a.epochs if a.epochs > 0 else TRAIN["max_epochs"]
    tagp = (a.tag + "_") if a.tag else ""

    # --- data ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)      # (8192, 128) price
    if a.frac < 1.0:
        n = int(round(S.shape[0] * a.frac))
        S = S[:n]
    N, seq_len = S.shape
    X = S[:, :, None].astype(np.float32)                         # (N, T, 1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)                                 # ~[0,1], per-(t,feature)
    print(f"=== TimeVAE Heston  seed={a.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===",
          flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"scaled[min={Xs.min():.4f},max={Xs.max():.4f}]  epochs={epochs}", flush=True)

    # --- model (verbatim TimeVAE-Base port) ---
    model = TimeVAE(seq_len=seq_len, feat_dim=1, **PRESET)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam}", flush=True)

    # --- train ---
    t0 = time.time()
    hist = train_timevae(
        model, Xs, max_epochs=epochs, batch_size=a.batch_size, lr=TRAIN["lr"],
        device=device, es_patience=TRAIN["es_patience"], es_min_delta=TRAIN["es_min_delta"],
        rlr_patience=TRAIN["rlr_patience"], rlr_factor=TRAIN["rlr_factor"],
        verbose=1, log_every=a.log_every, seed=a.seed,
    )
    train_time = time.time() - t0
    totals = [h["total_loss"] for h in hist]
    min_total = float(min(totals))
    first_nan = next((h["epoch"] for h in hist if not np.isfinite(h["total_loss"])), None)

    # --- generate 8192 paths (prior samples), invert to price scale ---
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    gen_s = model.generate(gen_n, device=device).astype(np.float32)   # (N,T,1), ~[0,1]
    Xg = scaler.inverse_transform(gen_s)[:, :, 0]                     # (N,T) price
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    # --- output dirs ---
    weights_dir = os.path.join(METHOD_DIR, "weights")
    losses_dir = os.path.join(METHOD_DIR, "losses")
    gen_dir = os.path.join(METHOD_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- generated paths ---
    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- loss curve (canonical name for real runs, tagged prefix for smoke) ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "reconstruction_loss", "kl_loss", "lr"])
        w.writeheader()
        w.writerows(hist)

    # --- weights + config (only for canonical runs) ---
    if not a.tag:
        torch.save({"model": model.state_dict(),
                    "scaler_mini": scaler.mini, "scaler_range": scaler.range,
                    "seed": a.seed}, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "TimeVAE", "variant": "TimeVAE-Base", "seed": a.seed,
               "feat_dim": 1, "seq_len": seq_len, "epochs_max": epochs,
               "epochs_run": len(hist), **{k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in PRESET.items()},
               "batch_size": a.batch_size, "lr": TRAIN["lr"],
               "es_patience": TRAIN["es_patience"], "es_min_delta": TRAIN["es_min_delta"],
               "rlr_patience": TRAIN["rlr_patience"], "rlr_factor": TRAIN["rlr_factor"],
               "params": int(nparam)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # --- metadata (GUIDELINE §4.3 schema) ---
    meta = {"method": "TimeVAE", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "epochs_run": len(hist), "epochs_max": epochs,
            "min_total_loss": min_total, "first_nan_epoch": first_nan,
            "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} epochs={len(hist)} first_total={totals[0]:.2f} "
          f"last_total={totals[-1]:.2f} min_total={min_total:.2f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
