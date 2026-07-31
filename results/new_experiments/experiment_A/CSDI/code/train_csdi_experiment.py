"""CSDI on a blinded-protocol experiment (Experiment A or B).

Identical model, optimiser, scaler and generation path as
``methods/CSDI/code/train_heston.py`` -- the only differences are (a) the target
dataset is the protocol's ``train.npy`` instead of the original Heston bank and
(b) artefacts land under ``results/new_experiments/experiment_<X>/CSDI``.

The model class and the hyperparameter block are IMPORTED from
``methods/CSDI/code/train_heston.py`` rather than re-declared, so the Stage-0
gate result (bitwise reproduction of the canonical 8192 Heston run) transfers:
there is exactly one definition of CSDI_Heston and of BASE_CONFIG in the repo.

Released ``config/base.yaml`` preset, verbatim: epochs=200, batch=16, lr=1e-3;
diffusion layers=4, channels=64, nheads=8, diffusion_embedding_dim=128,
beta_start=1e-4, beta_end=0.5, num_steps=50, schedule=quad;
model is_unconditional=1, timeemb=128, featureemb=16.
Optim: Adam(lr=1e-3, weight_decay=1e-6) + MultiStepLR([0.75E, 0.9E], gamma=0.1).

Information firewall: this script reads ONLY ``train.npy``. It never touches
``test.npy``, ``disc.npy``, ``*_sigma.npy``, ``*_labels.npy`` or the oracle.

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_csdi_experiment.py --seed 0 \
      --data ../../../../../dataset/Heston/new_experiments/experiment_A/train.npy
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

CODE_DIR = os.path.dirname(os.path.abspath(__file__))              # .../CSDI/code
OUT_DIR = os.path.dirname(CODE_DIR)                                 # .../CSDI
BENCH_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", "..", ".."))
CANONICAL = os.path.join(BENCH_ROOT, "methods", "CSDI", "code")
sys.path.insert(0, CANONICAL)

from train_heston import CSDI_Heston, BASE_CONFIG   # noqa: E402  (path set above)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True, help="protocol train.npy")
    ap.add_argument("--experiment", default="A", choices=("A", "B"))
    ap.add_argument("--epochs", type=int, default=0, help="0 = base.yaml default 200")
    ap.add_argument("--batch_size", type=int, default=0, help="0 = base.yaml default 16")
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=512)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    cfg = json.loads(json.dumps(BASE_CONFIG))   # deep copy
    epochs = a.epochs if a.epochs > 0 else cfg["train"]["epochs"]
    batch_size = a.batch_size if a.batch_size > 0 else cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]

    data_path = os.path.abspath(a.data)
    if os.path.basename(data_path) != "train.npy":
        raise SystemExit(f"firewall: generators may only read train.npy, got {data_path}")

    # --- data: z-score standardize (CSDI PhysioNet convention) ---
    S = np.load(data_path).astype(np.float64)                 # (N, L) price
    N, seq_len = S.shape
    mean = float(S.mean())
    std = float(S.std())
    if std == 0:
        std = 1.0
    Xn = ((S - mean) / std).astype(np.float32)
    Xt = Xn[:, :, None]                                       # (N, L, K=1)
    ds = TensorDataset(torch.from_numpy(Xt))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    print(f"=== CSDI experiment_{a.experiment}  seed={a.seed}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] {data_path}", flush=True)
    print(f"[data] S{S.shape} price[min={S.min():.2f},max={S.max():.2f}]  "
          f"mean={mean:.4f} std={std:.4f}  scaled[min={Xn.min():.3f},max={Xn.max():.3f}]  "
          f"epochs={epochs} batch={batch_size}", flush=True)

    # --- model (verbatim CSDI, unconditional) ---
    model = CSDI_Heston(cfg, device, target_dim=1).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam} num_steps={cfg['diffusion']['num_steps']} "
          f"channels={cfg['diffusion']['channels']} layers={cfg['diffusion']['layers']}",
          flush=True)

    weights_dir = os.path.join(OUT_DIR, "weights")
    losses_dir = os.path.join(OUT_DIR, "losses")
    gen_dir = os.path.join(OUT_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- training loop (CSDI's utils.train recipe: Adam wd=1e-6, MultiStepLR @0.75/0.9 x0.1) ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1.0e-6)
    p1, p2 = int(0.75 * epochs), int(0.9 * epochs)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    records = []          # (step, loss)     -- CSDI's native per-step trace
    epoch_hist = []       # per-epoch rows   -- the layout contract's schema
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for (xb,) in dl:
            optimizer.zero_grad()
            loss = model({"observed_data": xb})
            loss.backward()
            optimizer.step()
            records.append((step, float(loss.item())))
            step += 1
        lr_now = float(lr_sched.get_last_lr()[0])
        lr_sched.step()
        ep_losses = [l for _, l in records[-len(dl):]]
        epoch_hist.append({"epoch": epoch, "avg_loss": float(np.mean(ep_losses)),
                           "min_loss": float(np.min(ep_losses)), "lr": lr_now})
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  avg_loss={epoch_hist[-1]['avg_loss']:.5f}  "
                  f"lr={lr_now:.2e}", flush=True)
    train_time = time.time() - t0

    # --- loss curves ---
    # seed_N_losses.csv follows the layout contract (first column 'epoch', last 'lr');
    # the per-STEP trace CSDI natively logs -- the artefact the Stage-0 reproduction gate
    # compared bitwise -- is kept alongside it, not thrown away.
    with open(os.path.join(losses_dir, f"seed_{a.seed}_losses.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "avg_loss", "min_loss", "lr"])
        w.writeheader(); w.writerows(epoch_hist)
    with open(os.path.join(losses_dir, f"seed_{a.seed}_losses_steps.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"])
        w.writerows(records)
    first_nan_step = next((s for s, l in records if not np.isfinite(l)), None)
    first_nan_epoch = next((h["epoch"] for h in epoch_hist
                            if not np.isfinite(h["avg_loss"])), None)
    min_loss = min((l for _, l in records), default=float("nan"))
    min_total_loss = float(min(h["avg_loss"] for h in epoch_hist))

    # --- generate paths (timed) ---
    g0 = time.time()
    Xg_std = model.generate(a.gen_num, seq_len, gen_batch=a.gen_batch)   # (N, L) standardized
    Xg = Xg_std * std + mean                                             # price scale
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    # The bank is written RAW (un-anchored). The declared PDF §1.3 repair
    # ``S <- 100 * S / S[:, :1]`` is applied afterwards by the committed tool
    # ``tools/apply_s0_repair.py --apply`` -- never inline here, so that exactly one
    # implementation of the transformation exists and ``--raw-dir`` can audit it.
    # A pristine copy is kept under raw_banks/ for precisely that strong audit.
    raw_dir = os.path.join(OUT_DIR, "raw_banks", "generated_paths", f"seed_{a.seed}")
    os.makedirs(raw_dir, exist_ok=True)
    np.save(os.path.join(gen_dir, "generated_paths_8192x128.npy"), Xg.astype(np.float64))
    np.save(os.path.join(raw_dir, "generated_paths_8192x128.npy"), Xg.astype(np.float64))
    s0_dev = float(np.abs(Xg[:, 0] - 100.0).max())

    torch.save({"model": model.state_dict(), "seed": a.seed, "zscore": [mean, std]},
               os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
    out_cfg = {"method": "CSDI", "variant": "CSDI (released base.yaml, unconditional)",
               "experiment": a.experiment, "seed": a.seed, "data": data_path,
               "is_unconditional": 1, "target_dim": 1, "seq_length": seq_len,
               "epochs": epochs, "batch_size": batch_size, "lr": lr,
               **cfg["diffusion"], "timeemb": cfg["model"]["timeemb"],
               "featureemb": cfg["model"]["featureemb"],
               "weight_decay": 1.0e-6, "lr_milestones": [p1, p2], "lr_gamma": 0.1,
               "scaler": "global_standardize", "zscore_mean": mean, "zscore_std": std,
               "scaler_mu": mean, "scaler_sigma": std,
               "params": int(nparam)}
    with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    meta = {"method": "CSDI", "experiment": a.experiment, "seed": a.seed,
            "shape": list(Xg.shape), "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "zscore_mean": mean, "zscore_std": std, "params": int(nparam),
            "epochs": epochs, "epochs_run": len(epoch_hist), "epochs_max": epochs,
            "batch_size": batch_size,
            "num_steps": cfg["diffusion"]["num_steps"],
            "min_total_loss": min_total_loss, "first_nan_epoch": first_nan_epoch,
            "min_loss": float(min_loss), "first_nan_step": first_nan_step,
            "s0_max_deviation_raw": s0_dev,
            "gen_has_nan": gen_has_nan}
    with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} steps={step} min_loss={min_loss:.5f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"S0_dev_raw={s0_dev:.3e} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
