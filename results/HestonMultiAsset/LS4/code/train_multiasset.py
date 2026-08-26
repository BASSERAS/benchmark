"""
LS4 on the multi-asset Heston target (d = 8 correlated assets).

Reference
---------
Zhou, Kang, Molina-Salgado, Wu, Ermon, Grover -- "Deep Latent State Space Models
for Time-Series Generation", ICML 2023.  Architecture is the released ``VAE``
module (``reference_models/ls4.py``) with the released ``solar_weekly`` preset
(``reference_configs/vae_solarweekly_released.yaml``).

Provenance (d = 1 validation, run before this port)
---------------------------------------------------
``methods/LS4/code/train_heston.py --seed 0 --epochs 5`` was re-run and
reproduced the committed ``methods/LS4/losses/seed_0_losses.csv`` BIT-EXACTLY
(epoch 0: total 24.563837595283985 / kld 0.4643734924029559 / nll
24.09946396201849 / mse 0.5096622176934034; epoch 4: 0.03708194966020528 /
0.35022108210250735 / -0.3131391337956302 / 0.02141012719948776), together with
params = 2146857, scaler_mu = 101.3255, scaler_sigma = 9.9717.  The vendored
model code under ``reference_models/`` is therefore the same LS4 that produced
the committed d = 1 results.

IMPORTANT -- pure-PyTorch cauchy fix
------------------------------------
``reference_models/s4.py`` line 795 sums the naive Cauchy kernel over conjugate
pole PAIRS (``cauchy_naive(_conj(v), z, _conj(w))``), matching the keops/CUDA
path.  REQUIRED here because ``model.generate`` rolls the prior via
``latent.step`` (STEP-mode), where the unpatched kernel disagrees with conv-mode.

Deviations from the d = 1 run (recorded as ``retuned_for_d8``)
--------------------------------------------------------------
1. ``in_channels``  1 -> 8.  The released YAML ties ``model.in_channels``,
   ``model.decoder.decoder.d_output`` and ``model.encoder.posterior.d_input`` to a
   single ``&channel`` anchor; ALL THREE are set explicitly here.  Likewise
   ``z_dim`` is anchored into ``decoder.prior.d_input/d_output``,
   ``decoder.decoder.d_input`` and ``encoder.posterior.d_output`` -- all set
   explicitly.  Setting only ``config.model.in_channels`` (as the d = 1 trainer
   does, which is correct there because 1 == 1) would silently keep a 1-channel
   decoder head.
2. ``z_dim``  5 -> selected on the validation split.  A 5-dim latent for 8
   correlated assets is not defensible by default.
3. ``scaler``  global -> per-channel ``(mu_j, sigma_j)``, j = 1..8.  The eight
   marginals have different price scales (std 11.1 .. 18.6).  A per-channel
   AFFINE map leaves the cross-asset correlation matrix EXACTLY unchanged, so
   the target coupling Sigma^s survives standardisation and its inverse.

S0 contract
-----------
GUIDELINE section 4 mandates ``S[:, 0, :] == 100.0`` exactly.  LS4 prior samples
do not satisfy this (d = 1 gave S0 in [99.30, 100.48], std 0.055), so each
generated path is rescaled per asset:  ``S <- S * 100 / S[:, 0, :]``.  A per-path
per-asset CONSTANT multiplier is exactly a shift of the log-price level, hence
every log-return is bit-identical and metrics A1-A25 / A27-A34 (all log-return
based) are unaffected.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_multiasset.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_multiasset.py --seed 0 --epochs 3 --tag probe --no_generate
"""
import os
import sys
import csv
import json
import time
import random
import hashlib
import argparse
import numpy as np
import torch
import torch.optim as optim

CODE_DIR = os.path.dirname(os.path.abspath(__file__))          # LS4/code
LS4_DIR = os.path.dirname(CODE_DIR)                            # LS4
MA_DIR = os.path.dirname(LS4_DIR)                              # results/HestonMultiAsset
BENCH_ROOT = os.path.dirname(os.path.dirname(MA_DIR))          # benchmark/
sys.path.insert(0, CODE_DIR)

from omegaconf import OmegaConf                                # noqa: E402
from reference_models.ls4 import VAE                           # noqa: E402

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "HestonMultiAsset")
TRAIN_NPY = os.path.join(DATA_DIR, "heston_ma_S_8192x252x8.npy")
VAL_NPY = os.path.join(DATA_DIR, "heston_ma_S_val_8192x252x8.npy")
PARAMS_JSON = os.path.join(DATA_DIR, "parameters.json")
CONFIG_YAML = os.path.join(CODE_DIR, "reference_configs", "vae_solarweekly_released.yaml")

EXPECTED_DIGEST = "231da80bdedf22e9"
D = 8
SEQ_LEN = 252
N_TRAIN = 8192


def _digest(obj):
    """Per-asset parameter digest -- SAME formula as
    metrics/gen_perfect_recovery_multiasset.py (no ``separators`` argument)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def check_dataset_digest():
    """Abort if the dataset was regenerated under a different numpy.

    ``np.random.Generator`` is NOT version-stable; a different digest means the
    paths are not the ones SBTS was trained and scored on, so nothing would be
    comparable.  Do not bypass this guard.
    """
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    got = _digest(params["per_asset"])
    if got != EXPECTED_DIGEST:
        raise SystemExit(
            f"[FATAL] per-asset digest mismatch: got {got}, expected {EXPECTED_DIGEST}.\n"
            f"        The dataset under {DATA_DIR} is not the one SBTS was scored on.\n"
            f"        Regenerate it with the frozen generator, or stop. "
            f"Do not bypass this guard."
        )
    return got


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


def per_channel_scaler(S):
    """(mu_j, sigma_j) per asset.  Affine per channel => correlation preserved."""
    mu = S.mean(axis=(0, 1))                        # (d,)
    sigma = S.std(axis=(0, 1))                      # (d,)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return mu, sigma


def build_config(z_dim, d):
    """Load the released preset and override EVERY field the YAML anchors tie
    to ``&channel`` / ``&z_dim``.  OmegaConf resolves anchors at load time, so
    setting ``model.in_channels`` alone does NOT propagate."""
    config = OmegaConf.load(CONFIG_YAML)
    config.model.n_labels = 1
    # --- &channel anchor: three call sites ---
    config.model.in_channels = d
    config.model.decoder.decoder.d_output = d
    config.model.encoder.posterior.d_input = d
    # --- &z_dim anchor: four call sites ---
    config.model.z_dim = z_dim
    config.model.decoder.prior.d_input = z_dim
    config.model.decoder.prior.d_output = z_dim
    config.model.decoder.decoder.d_input = z_dim
    config.model.encoder.posterior.d_output = z_dim
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=TRAIN_NPY)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=512,
                    help="chunk size for step-mode generation")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths to use (smoke/probe)")
    ap.add_argument("--val", action="store_true",
                    help="also evaluate the ELBO on the held-out validation split each epoch")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--tag", default="",
                    help="run tag (e.g. 'probe'); prefixes outputs, skips canonical weights")
    ap.add_argument("--no_generate", action="store_true",
                    help="train only (timing probe / z_dim sweep)")
    a = ap.parse_args()

    digest = check_dataset_digest()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (a.tag + "_") if a.tag else ""

    # --- data ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)         # (N, T, d) price
    if S.ndim != 3 or S.shape[2] != D:
        raise SystemExit(f"[FATAL] expected (N, T, {D}) prices, got {S.shape}")
    if a.frac < 1.0:
        S = S[:int(round(S.shape[0] * a.frac))]
    N, seq_len, d = S.shape
    mu, sigma = per_channel_scaler(S)
    Xs = ((S - mu) / sigma).astype(np.float32)                      # (N, T, d) ~N(0,1) per channel

    print(f"=== LS4 multi-asset Heston  seed={a.seed}  d={d}  z_dim={a.z_dim}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] S{S.shape} digest={digest} price[min={S.min():.2f},max={S.max():.2f}]",
          flush=True)
    print(f"[scaler] per_channel  mu={np.round(mu, 3).tolist()}", flush=True)
    print(f"[scaler] per_channel  sigma={np.round(sigma, 3).tolist()}", flush=True)
    print(f"[data] scaled[min={Xs.min():.3f},max={Xs.max():.3f}]  epochs={a.epochs}", flush=True)

    X = torch.tensor(Xs, dtype=torch.float32)
    M = torch.ones_like(X, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, M),
        batch_size=a.batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if a.val:
        Sv = np.load(VAL_NPY).astype(np.float64)
        if a.frac < 1.0:
            Sv = Sv[:int(round(Sv.shape[0] * a.frac))]
        Xv = torch.tensor(((Sv - mu) / sigma).astype(np.float32), dtype=torch.float32)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xv, torch.ones_like(Xv)),
            batch_size=a.batch_size, shuffle=False, drop_last=False)
        print(f"[val] Sv{Sv.shape} (scaled with the TRAIN scaler)", flush=True)

    # --- model ---
    config = build_config(a.z_dim, d)
    model = VAE(config.model).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam}  in_channels={config.model.in_channels} "
          f"z_dim={config.model.z_dim} n_layers={config.model.decoder.prior.n_layers} "
          f"s4_type={config.model.decoder.prior.s4_type}", flush=True)

    # --- EMA (released: lamb=0.99, start_step=200) ---
    lamb = float(config.optim.get("lamb", 0.99))
    start_step = int(config.optim.get("start_step", 200))
    use_ema = bool(config.optim.get("use_ema", True))
    ema_avg = lambda avg_p, p, n: lamb * avg_p + (1 - lamb) * p     # noqa: E731
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg) if use_ema else None

    optimizer, scheduler = setup_optimizer(model, lr=float(config.optim.lr),
                                           weight_decay=float(config.optim.weight_decay))

    # --- train (ELBO = kld + nll) ---
    t0 = time.time()
    hist = []
    step = 0
    epoch_secs = []
    for epoch in range(a.epochs):
        te = time.time()
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
        epoch_secs.append(time.time() - te)
        hist.append({"step": step, "phase": "train", "loss_total": tot, "epoch": epoch,
                     "kld_loss": kld, "nll_loss": nll, "mse_loss": mse, "lr": lr_now})

        vtot = None
        if val_loader is not None:
            model.eval()
            vt = 0.0; vb = 0
            with torch.no_grad():
                for data, masks in val_loader:
                    vloss, _ = model(data.to(device), None, masks.to(device),
                                     plot=False, sum=False)
                    vt += vloss.item(); vb += 1
            vtot = vt / vb
            hist.append({"step": step, "phase": "val", "loss_total": vtot, "epoch": epoch,
                         "kld_loss": "", "nll_loss": "", "mse_loss": "", "lr": lr_now})

        if epoch % a.log_every == 0 or epoch == a.epochs - 1:
            vstr = f" val={vtot:.4f}" if vtot is not None else ""
            print(f"[ep {epoch:4d}] total={tot:.4f}{vstr} kld={kld:.4f} nll={nll:.4f} "
                  f"mse={mse:.6f} lr={lr_now:.2e} ({epoch_secs[-1]:.1f}s)", flush=True)
    train_time = time.time() - t0

    tr = [h for h in hist if h["phase"] == "train"]
    totals = [h["loss_total"] for h in tr]
    min_total = float(min(totals))
    first_nan = next((h["epoch"] for h in tr if not np.isfinite(h["loss_total"])), None)
    val_totals = [h["loss_total"] for h in hist if h["phase"] == "val"]
    min_val = float(min(val_totals)) if val_totals else None

    # --- output dirs ---
    weights_dir = os.path.join(LS4_DIR, "weights")
    losses_dir = os.path.join(LS4_DIR, "losses")
    gen_dir = os.path.join(LS4_DIR, "generated_paths", f"seed_{a.seed}")
    for dd in (weights_dir, losses_dir, gen_dir):
        os.makedirs(dd, exist_ok=True)

    # --- loss curve (GUIDELINE section 3.2: step, phase, loss_total minimum) ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "phase", "loss_total", "epoch",
                                          "kld_loss", "nll_loss", "mse_loss", "lr"])
        w.writeheader()
        w.writerows(hist)

    gen_time = 0.0
    Xg = None
    if not a.no_generate:
        # --- pick generation model (EMA if warmed up) + STEP-mode ---
        gen_model = ema_model.module if (ema_model is not None and step > start_step) else model
        gen_model.eval()
        gen_model.setup_rnn()

        gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
        g0 = time.time()
        chunks = []
        with torch.no_grad():
            done = 0
            while done < gen_n:
                b = min(a.gen_batch, gen_n - done)
                g = gen_model.generate(b, seq_len, device=device)        # (b, T, d) standardized
                chunks.append(g.detach().cpu().numpy())
                done += b
        gen_s = np.concatenate(chunks, axis=0).astype(np.float64)        # (gen_n, T, d)
        gen_time = time.time() - g0

        # --- invert the per-channel scaler, then enforce S0 == 100 exactly ---
        Xg = gen_s * sigma + mu
        Xg = np.clip(Xg, 1e-6, None)
        Xg = Xg * (100.0 / Xg[:, 0:1, :])          # constant per (path, asset) => log-returns unchanged
        Xg[:, 0, :] = 100.0
        out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_{gen_n}x{seq_len}x{d}.npy")
        np.save(out_npy, Xg.astype(np.float64))

    # --- weights + config (canonical runs only) ---
    if not a.tag:
        save = {"model": model.state_dict(), "scaler_mu": mu.tolist(),
                "scaler_sigma": sigma.tolist(), "seed": a.seed, "z_dim": a.z_dim}
        if ema_model is not None:
            save["ema_model"] = ema_model.state_dict()
        torch.save(save, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "LS4", "variant": "LS4 (released solar_weekly preset, retuned for d=8)",
               "seed": a.seed, "feat_dim": d, "seq_len": seq_len,
               "joint_or_per_asset": "joint",
               "paper_hyperparams": False,
               "retuned_for_d8": ["in_channels", "z_dim", "scaler"],
               "epochs": a.epochs, "z_dim": int(config.model.z_dim),
               "d_state": int(config.model.decoder.prior.d_state),
               "d_model": int(config.model.decoder.prior.d_model),
               "n_layers": int(config.model.decoder.prior.n_layers),
               "s4_type": str(config.model.decoder.prior.s4_type),
               "latent_type": str(config.model.decoder.prior.latent_type),
               "batch_size": a.batch_size, "lr": float(config.optim.lr),
               "weight_decay": float(config.optim.weight_decay),
               "ema_lamb": lamb, "ema_start_step": start_step,
               "scaler": "per_channel_standardize",
               "scaler_mu": mu.tolist(), "scaler_sigma": sigma.tolist(),
               "params": int(nparam), "per_asset_digest": digest,
               "train_time_sec": round(train_time, 1)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # --- metadata (GUIDELINE section 4.3 schema) ---
    meta = {"method": "LS4", "seed": a.seed,
            "gen_time_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "sec_per_epoch": round(float(np.mean(epoch_secs)), 2),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "z_dim": a.z_dim,
            "epochs_run": len(tr), "epochs_max": a.epochs,
            "min_total_loss": min_total, "min_val_loss": min_val,
            "first_nan_epoch": first_nan,
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "per_asset_digest": digest, "s0_rescaled": True}
    if Xg is not None:
        meta.update({"shape": list(Xg.shape), "dtype": "float64", "d": d, "S0": 100.0,
                     "S_min": float(Xg.min()), "S_max": float(Xg.max()),
                     "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
                     "S0_exact": bool(np.all(Xg[:, 0, :] == 100.0)),
                     "all_finite": bool(np.isfinite(Xg).all()),
                     "all_positive": bool((Xg > 0).all())})
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} epochs={len(tr)} first_total={totals[0]:.2f} "
          f"last_total={totals[-1]:.2f} min_total={min_total:.2f} "
          f"train={train_time:.1f}s ({np.mean(epoch_secs):.1f}s/epoch) gen={gen_time:.1f}s",
          flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
