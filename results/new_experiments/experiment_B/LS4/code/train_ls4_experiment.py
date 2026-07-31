"""LS4 on a blinded-protocol experiment (Experiment A or B).

Identical model, optimiser, scaler and generation path as
``methods/LS4/code/train_heston.py`` -- the only differences are (a) the target
dataset is the protocol's ``train.npy`` instead of the original Heston bank and
(b) artefacts land under ``results/new_experiments/experiment_<X>/LS4``.

Released ``solar_weekly`` preset, verbatim from
``methods/LS4/code/reference/configs/monash/vae_solarweekly_released.yaml``:
z_dim=5, in_channels=1, sigma=0.1; decoder/prior/posterior each d_state=64,
d_model=64, n_layers=4, backbone=autoreg, s4_type=s4, latent_type=split.
Optim: AdamW(lr=1e-3, wd=0) + ReduceLROnPlateau(patience=20, factor=0.5) +
EMA(lamb=0.99, start_step=200).

Information firewall: this script reads ONLY ``train.npy``. It never touches
``test.npy``, ``disc.npy``, ``*_sigma.npy``, ``*_labels.npy`` or the oracle.

Usage:
  CUDA_VISIBLE_DEVICES=2 python train_ls4_experiment.py --seed 0 \
      --data ../../../../../dataset/Heston/new_experiments/experiment_A/train.npy
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
import torch.optim as optim

CODE_DIR = os.path.dirname(os.path.abspath(__file__))              # .../LS4/code
OUT_DIR = os.path.dirname(CODE_DIR)                                 # .../LS4
BENCH_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", "..", ".."))
REFERENCE = os.path.join(BENCH_ROOT, "methods", "LS4", "code", "reference")
sys.path.insert(0, REFERENCE)

from omegaconf import OmegaConf                                     # noqa: E402
from models.ls4 import VAE                                          # noqa: E402

CONFIG_YAML = os.path.join(REFERENCE, "configs", "monash", "vae_solarweekly_released.yaml")


def setup_optimizer(model, lr, weight_decay):
    """Verbatim from the released ``train_monash.setup_optimizer``."""
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    return optimizer, scheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True, help="protocol train.npy")
    ap.add_argument("--experiment", default="A", choices=("A", "B"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=25)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    data_path = os.path.abspath(a.data)
    if os.path.basename(data_path) != "train.npy":
        raise SystemExit(f"firewall: generators may only read train.npy, got {data_path}")
    S = np.load(data_path).astype(np.float64)
    N, seq_len = S.shape
    mu = float(S.mean())
    sigma = float(S.std())
    if sigma == 0:
        sigma = 1.0
    Xs = ((S - mu) / sigma).astype(np.float32)[:, :, None]
    print(f"=== LS4 experiment_{a.experiment}  seed={a.seed}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] {data_path}", flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"mu={mu:.4f} sigma={sigma:.4f}  scaled[min={Xs.min():.3f},max={Xs.max():.3f}]  "
          f"epochs={a.epochs}", flush=True)

    X = torch.tensor(Xs, dtype=torch.float32)
    M = torch.ones_like(X, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X, M)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         drop_last=False)

    config = OmegaConf.load(CONFIG_YAML)
    config.model.n_labels = 1
    config.model.in_channels = 1
    model = VAE(config.model).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam}  z_dim={config.model.z_dim} "
          f"n_layers={config.model.decoder.prior.n_layers} "
          f"s4_type={config.model.decoder.prior.s4_type}", flush=True)

    lamb = float(config.optim.get("lamb", 0.99))
    start_step = int(config.optim.get("start_step", 200))
    use_ema = bool(config.optim.get("use_ema", True))
    ema_avg = lambda avg_p, p, n: lamb * avg_p + (1 - lamb) * p      # noqa: E731
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg) if use_ema else None

    optimizer, scheduler = setup_optimizer(model, lr=float(config.optim.lr),
                                           weight_decay=float(config.optim.weight_decay))

    t0 = time.time()
    hist = []
    step = 0
    for epoch in range(a.epochs):
        model.train()
        tot = kld = nll = mse = 0.0
        nb = 0
        for data, masks in loader:
            step += 1
            data = data.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss, log_info = model(data, None, masks, plot=False, sum=False)
            loss.backward()
            optimizer.step()
            tot += loss.item()
            kld += log_info["kld_loss"]
            nll += log_info["nll_loss"]
            mse += log_info["mse_loss"]
            nb += 1
            if ema_model is not None and step > start_step:
                ema_model.update_parameters(model)
        tot /= nb; kld /= nb; nll /= nb; mse /= nb
        scheduler.step(tot)
        lr_now = optimizer.param_groups[0]["lr"]
        hist.append({"epoch": epoch, "total_loss": tot, "kld_loss": kld,
                     "nll_loss": nll, "mse_loss": mse, "lr": lr_now})
        if epoch % a.log_every == 0 or epoch == a.epochs - 1:
            print(f"[ep {epoch:4d}] total={tot:.4f} kld={kld:.4f} nll={nll:.4f} "
                  f"mse={mse:.6f} lr={lr_now:.2e}", flush=True)
    train_time = time.time() - t0
    totals = [h["total_loss"] for h in hist]
    min_total = float(min(totals))
    first_nan = next((h["epoch"] for h in hist if not np.isfinite(h["total_loss"])), None)

    gen_model = ema_model.module if (ema_model is not None and step > start_step) else model
    gen_model.eval()
    gen_model.setup_rnn()

    g0 = time.time()
    chunks = []
    with torch.no_grad():
        done = 0
        while done < a.gen_num:
            b = min(a.gen_batch, a.gen_num - done)
            chunks.append(gen_model.generate(b, seq_len, device=device).detach().cpu().numpy())
            done += b
    gen_s = np.concatenate(chunks, axis=0).astype(np.float32)
    Xg = gen_s[:, :, 0] * sigma + mu
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    weights_dir = os.path.join(OUT_DIR, "weights")
    losses_dir = os.path.join(OUT_DIR, "losses")
    gen_dir = os.path.join(OUT_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    np.save(os.path.join(gen_dir, "generated_paths_8192x128.npy"), Xg.astype(np.float64))

    with open(os.path.join(losses_dir, f"seed_{a.seed}_losses.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "kld_loss",
                                          "nll_loss", "mse_loss", "lr"])
        w.writeheader()
        w.writerows(hist)

    save = {"model": model.state_dict(), "scaler_mu": mu, "scaler_sigma": sigma, "seed": a.seed}
    if ema_model is not None:
        save["ema_model"] = ema_model.state_dict()
    torch.save(save, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
    cfg = {"method": "LS4", "variant": "LS4 (released solar_weekly preset)",
           "experiment": a.experiment, "seed": a.seed, "data": data_path,
           "feat_dim": 1, "seq_len": seq_len, "epochs": a.epochs,
           "z_dim": int(config.model.z_dim),
           "d_state": int(config.model.decoder.prior.d_state),
           "d_model": int(config.model.decoder.prior.d_model),
           "n_layers": int(config.model.decoder.prior.n_layers),
           "s4_type": str(config.model.decoder.prior.s4_type),
           "latent_type": str(config.model.decoder.prior.latent_type),
           "batch_size": a.batch_size, "lr": float(config.optim.lr),
           "weight_decay": float(config.optim.weight_decay),
           "ema_lamb": lamb, "ema_start_step": start_step,
           "scaler": "global_standardize", "scaler_mu": mu, "scaler_sigma": sigma,
           "params": int(nparam)}
    with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    meta = {"method": "LS4", "experiment": a.experiment, "seed": a.seed,
            "shape": list(Xg.shape), "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "epochs_run": len(hist), "epochs_max": a.epochs,
            "min_total_loss": min_total, "first_nan_epoch": first_nan,
            "gen_has_nan": gen_has_nan}
    with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} epochs={len(hist)} first_total={totals[0]:.2f} "
          f"last_total={totals[-1]:.2f} min_total={min_total:.2f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
