"""
LS4 on Heston with SBTS log-return preprocessing (4096 paths).

Same released LS4 architecture as ``methods/LS4/code/train_heston.py`` (imports the
reference ``models.ls4.VAE`` verbatim, released solar_weekly preset). The ONLY
differences vs the original are the front-end scaler and the back-end inverse:

  original :  price --(X-mu)/sigma--> N(0,1) space --train--> prior sample --*sigma+mu--> price
  this file:  price --SBTS log-return scaling--> R~ space --train--> prior sample
              --inverse-scale + cumulative-exp anchored at 100--> price

SBTS preprocessing (verbatim from ``methods/SBTS/code/sbts_generate.py``):
  R        = log(S[:,1:]/S[:,:-1])                    # (M,127) raw log-returns
  sigma    = float(R.std())                           # pooled over M x 127
  R_tilde  = R * sqrt(dt) / sigma                      # (M,127), std == sqrt(dt)
  X_train  = hstack([zeros(M,1), R_tilde])            # (M,128) dummy-0 first column
  --- LS4 trains on X_train (128-length sequences, col 0 == 0) ---
  R~_gen   = generate(...)[:, 1:]                      # (G,127) drop dummy column
  R_gen    = R~_gen * sigma / sqrt(dt)                 # inverse scale
  S_gen[:,0]=100 ;  S_gen[:,1:] = 100*exp(cumsum(R_gen))

Trains on ../../../../dataset/Heston/preprocessing_with_log_returns/heston_S_4096x128.npy
and writes the methods-style folders (weights/ losses/ generated_paths/) under
results/Heston/preprocessing_with_log_returns/LS4/.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_ls4_logret.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_ls4_logret.py --seed 0 --frac 0.05 --epochs 20 --tag smoke
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

HERE = os.path.dirname(os.path.abspath(__file__))                        # .../LS4/code
LS4_DIR = os.path.dirname(HERE)                                          # .../LS4
# results/Heston/preprocessing_with_log_returns/LS4 -> benchmark/
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(LS4_DIR))))

# import the UNTOUCHED reference LS4 model + the patched S4 kernel
REFERENCE = os.path.join(BENCH_ROOT, "methods", "LS4", "code", "reference")
sys.path.insert(0, REFERENCE)
from omegaconf import OmegaConf                                          # noqa: E402
from models.ls4 import VAE                                              # noqa: E402

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
DEFAULT_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")
CONFIG_YAML = os.path.join(REFERENCE, "configs", "monash", "vae_solarweekly_released.yaml")

DT = 1.0 / 250.0
S0 = 100.0


# --- released S4 optimizer setup (verbatim from train_monash.setup_optimizer) ---
def setup_optimizer(model, lr, weight_decay):
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


def scale_log_returns(S):
    """SBTS: price paths -> scaled log-returns + pooled sigma. Verbatim from SBTS."""
    R = np.log(S[:, 1:] / S[:, :-1])            # (M, T-1)
    sigma = float(R.std())
    R_tilde = (R * np.sqrt(DT) / sigma).astype(np.float64)
    return R_tilde, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gen_num", type=int, default=4096)
    ap.add_argument("--gen_batch", type=int, default=1024,
                    help="chunk size for step-mode generation")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths to use (smoke test)")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--tag", default="",
                    help="run tag (e.g. 'smoke'); prefixes output names, skips canonical weights")
    ap.add_argument("--out", default=LS4_DIR, help="output base dir (weights/losses/generated_paths)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (a.tag + "_") if a.tag else ""

    # --- data: SBTS log-return preprocessing ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)      # (4096, 128) price
    if a.frac < 1.0:
        n = int(round(S.shape[0] * a.frac))
        S = S[:n]
    N, seq_len = S.shape
    R_tilde, sigma = scale_log_returns(S)                        # (N, 127), pooled sigma
    X_sbts = np.hstack([np.zeros((N, 1), dtype=np.float64), R_tilde])    # (N, 128) dummy-0 col
    # LS4's own input standardization (identical to the price pipeline: global mean/std -> unit var).
    # SBTS-scaled returns have std = sqrt(dt) = 0.063, which sits BELOW LS4's fixed decoder noise
    # floor (sigma=0.1) and collapses the VAE; standardizing to unit variance restores the SNR the
    # released preset was tuned for. sigma (the SBTS scale) is untouched and still reported.
    x_mu = float(X_sbts.mean())
    x_sd = float(X_sbts.std())
    X_train = (X_sbts - x_mu) / x_sd                             # (N, 128) std == 1
    Xs = X_train.astype(np.float32)[:, :, None]                  # (N, T, 1)
    print(f"=== LS4-logret Heston  seed={a.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===",
          flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"SBTS sigma={sigma:.8f}  R~[min={R_tilde.min():.4f},max={R_tilde.max():.4f}] "
          f"R~.std={R_tilde.std():.6f} (sqrt(dt)={np.sqrt(DT):.6f})  "
          f"x_mu={x_mu:.6g} x_sd={x_sd:.6f} -> X_train.std={X_train.std():.4f}  "
          f"epochs={a.epochs}", flush=True)

    X = torch.tensor(Xs, dtype=torch.float32)                    # (N, T, 1)
    M = torch.ones_like(X, dtype=torch.float32)                  # masks all-ones
    ds = torch.utils.data.TensorDataset(X, M)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         drop_last=False)

    # --- model (released solar_weekly preset, verbatim) ---
    config = OmegaConf.load(CONFIG_YAML)
    config.model.n_labels = 1
    config.model.in_channels = 1
    model = VAE(config.model).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam}  z_dim={config.model.z_dim} "
          f"n_layers={config.model.decoder.prior.n_layers} s4_type={config.model.decoder.prior.s4_type}",
          flush=True)

    # --- EMA (released: lamb=0.99, start_step=200) ---
    lamb = float(config.optim.get("lamb", 0.99))
    start_step = int(config.optim.get("start_step", 200))
    use_ema = bool(config.optim.get("use_ema", True))
    ema_avg = lambda avg_p, p, n: lamb * avg_p + (1 - lamb) * p
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg) if use_ema else None

    optimizer, scheduler = setup_optimizer(model, lr=float(config.optim.lr),
                                           weight_decay=float(config.optim.weight_decay))

    # --- train (ELBO = kld + nll, EMA update after start_step) ---
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

    # --- pick generation model (EMA if warmed up) + set STEP-mode ---
    if ema_model is not None and step > start_step:
        gen_model = ema_model.module
    else:
        gen_model = model
    gen_model.eval()
    gen_model.setup_rnn()

    # --- generate gen_n paths (prior samples), chunked ---
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    chunks = []
    with torch.no_grad():
        done = 0
        while done < gen_n:
            b = min(a.gen_batch, gen_n - done)
            g = gen_model.generate(b, seq_len, device=device)       # (b, T, C) in R~ space
            chunks.append(g.detach().cpu().numpy())
            done += b
    gen_s = np.concatenate(chunks, axis=0).astype(np.float64)        # (gen_n, T, 1)

    # --- inverse: destandardize (undo LS4 unit-var), drop dummy col, inverse SBTS scale, cum-exp ---
    X_sbts_gen = gen_s[:, :, 0] * x_sd + x_mu                        # (gen_n, 128) back to R~ space
    R_tilde_gen = X_sbts_gen[:, 1:]                                  # (gen_n, 127) drop col 0
    R_gen = R_tilde_gen * sigma / np.sqrt(DT)                        # raw log-returns
    Xg = np.empty((gen_n, seq_len), dtype=np.float64)
    Xg[:, 0] = S0
    Xg[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))                # price anchored at 100
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    # --- output dirs (methods-style) ---
    weights_dir = os.path.join(a.out, "weights")
    losses_dir = os.path.join(a.out, "losses")
    gen_dir = os.path.join(a.out, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- generated paths ---
    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_{gen_n}x{seq_len}.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- loss curve ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "kld_loss", "nll_loss", "mse_loss", "lr"])
        w.writeheader()
        w.writerows(hist)

    # --- weights + config (only for canonical runs) ---
    if not a.tag:
        save = {"model": model.state_dict(), "sbts_sigma": sigma, "dt": DT, "s0": S0,
                "x_mu": x_mu, "x_sd": x_sd, "seed": a.seed}
        if ema_model is not None:
            save["ema_model"] = ema_model.state_dict()
        torch.save(save, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "LS4", "variant": "LS4 (released solar_weekly preset) + SBTS log-return preproc",
               "seed": a.seed, "feat_dim": 1, "seq_len": seq_len, "epochs": a.epochs,
               "n_train": N,
               "z_dim": int(config.model.z_dim), "d_state": int(config.model.decoder.prior.d_state),
               "d_model": int(config.model.decoder.prior.d_model),
               "n_layers": int(config.model.decoder.prior.n_layers),
               "s4_type": str(config.model.decoder.prior.s4_type),
               "latent_type": str(config.model.decoder.prior.latent_type),
               "batch_size": a.batch_size, "lr": float(config.optim.lr),
               "weight_decay": float(config.optim.weight_decay),
               "ema_lamb": lamb, "ema_start_step": start_step,
               "scaler": "sbts_log_return+unit_var", "sbts_sigma": sigma, "dt": DT, "s0": S0,
               "x_mu": x_mu, "x_sd": x_sd,
               "params": int(nparam)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # --- metadata (GUIDELINE 4.3 schema) ---
    meta = {"method": "LS4", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "sbts_sigma": float(sigma), "x_mu": x_mu, "x_sd": x_sd,
            "preproc": "sbts_log_return+unit_var",
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "epochs_run": len(hist), "epochs_max": a.epochs,
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
