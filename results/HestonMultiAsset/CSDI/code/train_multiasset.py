"""CSDI (Tashiro et al., NeurIPS 2021) -- unconditional training + generation, d = 8.

This is `methods/CSDI/code/train_heston.py` lifted to the multi-asset target. The
model is untouched: `main_model.CSDI_base` from `code/reference/` is the authors'
verbatim 2-D (time x feature) Transformer diffusion backbone, and the quad beta
schedule / DDPM forward+reverse are their `calc_loss` and `impute`. CSDI is
feature-agnostic -- `target_dim` is a constructor argument that flows into the
feature embedding and the feature-axis Transformer -- so d = 8 is `target_dim=8`
and a **joint** model, not 8 univariate ones. See MULTIASSET_GUIDELINE.md section 0,
question 1, and code/README.md.

Four things differ from the d = 1 script, all of them consequences of K = 1 -> K = 8
or of the guideline rather than of the method:

1. `target_dim=8`, `seq_len=252`, and the data tensor is loaded as (N, T, 8)
   directly instead of being given a trailing singleton axis.
2. The z-score is **per channel**. The d = 1 script used `S.mean()` / `S.std()`
   over the whole array, which for K = 1 *is* the per-feature statistic; CSDI's
   own PhysioNet convention is per-feature. Per-asset sigma here ranges 11.14-18.55,
   so a single scalar would hand the model eight differently-scaled channels. A
   per-channel affine map leaves every correlation invariant, so A20 -- the row
   that tests whether Sigma^s survived -- is unaffected by the choice.
3. `generate()` returns all K channels. The d = 1 version indexed `samples[:, 0, 0, :]`,
   which silently keeps asset 0 only.
4. Losses are logged in the guideline section 3.2 schema (`step,phase,loss_total`,
   every 100 steps) with a validation pass per epoch, instead of the d = 1
   `step,loss` every step.

Nothing is written to a shared path: each seed writes only under its own
`generated_paths/seed_{i}/` plus its own `weights/seed_{i}_*` and
`losses/seed_{i}_losses.csv`. `losses/generation_time.csv` is rebuilt afterwards by
`collect_artifacts.py` -- see guideline section 4, "the concurrent-seed version of
that trap".

Usage:
  CUDA_VISIBLE_DEVICES=2 python train_multiasset.py --seed 0
  CUDA_VISIBLE_DEVICES=2 python train_multiasset.py --seed 0 --epochs 2 --tag probe
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
CSDI_DIR = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(CSDI_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "reference"))

from main_model import CSDI_base  # noqa: E402  (path set above)

DATA_DIR = os.path.join(REPO, "dataset", "HestonMultiAsset")
TRAIN_NPY = os.path.join(DATA_DIR, "heston_ma_S_8192x252x8.npy")
VAL_NPY = os.path.join(DATA_DIR, "heston_ma_S_val_8192x252x8.npy")

D = 8
SEQ_LEN = 252
S0 = 100.0
GPU_NAME = "A100-SXM4-80GB"

# Verbatim from the released config/base.yaml, i.e. byte-identical to the d = 1 run
# in methods/CSDI. `epochs` and `batch_size` are overridable on the CLI; every
# override lands in weights/seed_{i}_config.json under "retuned_for_d8".
BASE_CONFIG = {
    "train": {"epochs": 200, "batch_size": 16, "lr": 1.0e-3},
    "diffusion": {
        "layers": 4, "channels": 64, "nheads": 8, "diffusion_embedding_dim": 128,
        "beta_start": 0.0001, "beta_end": 0.5, "num_steps": 50,
        "schedule": "quad", "is_linear": False,
    },
    "model": {
        "is_unconditional": 1,        # unconditional generation variant (paper Sec 4.1)
        "timeemb": 128, "featureemb": 16,
        "target_strategy": "random",  # unused: forward() forces cond_mask = 0
    },
}


class CSDI_MultiAsset(CSDI_base):
    """Unconditional CSDI over K = 8 jointly-modelled assets, L = 252.

    `process_data` yields fully-observed sequences (observed_mask == 1) and
    `forward` forces cond_mask == 0, so `target_mask = observed_mask - cond_mask`
    is 1 everywhere and `calc_loss` is the plain DDPM objective. `get_side_info`,
    `calc_loss`, `calc_loss_valid` and `impute` are the parent's verbatim code.
    """

    def __init__(self, config, device, target_dim=D):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()   # (B, L, K)
        B, L, K = observed_data.shape
        observed_mask = torch.ones(B, L, K, device=self.device)
        gt_mask = torch.zeros(B, L, K, device=self.device)
        observed_tp = torch.arange(L, device=self.device).float().unsqueeze(0).expand(B, -1)

        observed_data = observed_data.permute(0, 2, 1)                   # (B, K, L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(B, device=self.device).long()
        return observed_data, observed_mask, observed_tp, gt_mask, observed_mask, cut_length

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, _gt, _fpm, _cl = self.process_data(batch)
        cond_mask = torch.zeros_like(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    @torch.no_grad()
    def generate(self, n_paths, seq_len, gen_batch=128):
        """Draw `n_paths` unconditional samples -> (n_paths, seq_len, K), standardized."""
        self.eval()
        out, done = [], 0
        while done < n_paths:
            B = min(gen_batch, n_paths - done)
            cond_mask = torch.zeros(B, self.target_dim, seq_len, device=self.device)
            observed_tp = torch.arange(seq_len, device=self.device).float().unsqueeze(0).expand(B, -1)
            side_info = self.get_side_info(observed_tp, cond_mask)
            dummy = torch.zeros(B, self.target_dim, seq_len, device=self.device)  # shapes the sampler only
            samples = self.impute(dummy, cond_mask, side_info, n_samples=1)       # (B, 1, K, L)
            out.append(samples[:, 0].permute(0, 2, 1).cpu().numpy())              # (B, L, K)
            done += B
        return np.concatenate(out, axis=0)[:n_paths]


def rescale_to_s0(Xg):
    """Put every path on S[:, 0, :] == 100.0 exactly, per guideline section 4.

    A per-(path, asset) constant multiplier is exactly a shift of the log-price
    level, so every log-return is bit-identical before and after and A1-A25 /
    A27-A34 are unaffected by construction. Overwriting the first row alone would
    dump the whole correction into the first increment. Returns the array and the
    count of non-positive entries clipped to 1e-6 before the rescale. Clipping *is*
    a real distortion -- unlike the multiplier it does not preserve log-returns --
    so it is counted and reported rather than hidden. `n_s0` is broken out
    separately because a non-positive *first* price is far worse than a
    non-positive interior one: the multiplier is 100/S[:,0,:], so a first price
    clipped to 1e-6 scales that whole path by 1e8.
    """
    n_s0 = int((Xg[:, 0:1, :] <= 0).sum())
    n_total = int((Xg <= 0).sum())
    Xg = np.where(Xg <= 0.0, 1e-6, Xg)
    Xg = Xg * (S0 / Xg[:, 0:1, :])
    Xg[:, 0, :] = S0
    return Xg, n_s0, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="0 = base.yaml default 200")
    ap.add_argument("--batch_size", type=int, default=0, help="0 = base.yaml default 16")
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=128)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training paths (probes only)")
    ap.add_argument("--val_n", type=int, default=256,
                    help="validation paths per pass. CSDI's calc_loss_valid averages the loss "
                         "over ALL 50 diffusion steps, so a full 8192-path pass costs ~50x a "
                         "training epoch and would dominate the run. 256 paths on a 20-point "
                         "cadence keeps the overhead near 2%% and still draws a usable curve.")
    ap.add_argument("--tag", default="", help="run tag; prefixes losses/metadata and skips canonical weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(json.dumps(BASE_CONFIG))
    epochs = a.epochs if a.epochs > 0 else cfg["train"]["epochs"]
    batch_size = a.batch_size if a.batch_size > 0 else cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    tagp = (a.tag + "_") if a.tag else ""

    # --- data: per-channel z-score (CSDI PhysioNet convention) ---
    S = np.load(TRAIN_NPY).astype(np.float64)                 # (8192, 252, 8) price
    Sv = np.load(VAL_NPY).astype(np.float64)
    if a.frac < 1.0:
        S = S[: int(round(S.shape[0] * a.frac))]
    Sv = Sv[: a.val_n]
    assert S.shape[1:] == (SEQ_LEN, D), f"unexpected train shape {S.shape}"
    mean = S.mean(axis=(0, 1))                                # (8,)
    std = S.std(axis=(0, 1))                                  # (8,)
    Xt = ((S - mean) / std).astype(np.float32)
    Xv = ((Sv - mean) / std).astype(np.float32)               # val uses TRAIN statistics
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt)), batch_size=batch_size,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # drop_last=False: with drop_last=True a --val_n smaller than --batch_size yields an
    # empty loader and a silent nan validation curve instead of an error.
    dl_val = DataLoader(TensorDataset(torch.from_numpy(Xv)), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = CSDI_MultiAsset(cfg, device, target_dim=D).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[CSDI d=8 seed {a.seed}] params={nparam} epochs={epochs} batch={batch_size} "
          f"N={S.shape[0]} T={SEQ_LEN} K={D} steps/epoch={len(dl)}", flush=True)

    weights_dir = os.path.join(CSDI_DIR, "weights")
    losses_dir = os.path.join(CSDI_DIR, "losses")
    gen_dir = os.path.join(CSDI_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- CSDI's utils.train recipe: Adam wd=1e-6, MultiStepLR @0.75/0.9 epochs x0.1 ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1.0e-6)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.75 * epochs), int(0.9 * epochs)], gamma=0.1)

    records = []      # (step, phase, loss_total)
    step, epoch_secs = 0, []
    val_every = max(1, epochs // 20)
    t0 = time.time()
    for epoch in range(epochs):
        te = time.time()
        model.train()
        run, nb = 0.0, 0
        for (xb,) in dl:
            optimizer.zero_grad()
            loss = model({"observed_data": xb})
            loss.backward()
            optimizer.step()
            run += float(loss.item()); nb += 1
            if step % 100 == 0:
                records.append((step, "train", float(loss.item())))
            step += 1
        lr_sched.step()

        val_loss = float("nan")
        if epoch % val_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                vs = [float(model({"observed_data": xb}, is_train=0).item()) for (xb,) in dl_val]
            val_loss = float(np.mean(vs)) if vs else float("nan")
            records.append((step, "val", val_loss))
        epoch_secs.append(time.time() - te)
        print(f"  epoch {epoch:4d}/{epochs}  train={run / max(nb, 1):.5f}  val={val_loss:.5f}  "
              f"lr={lr_sched.get_last_lr()[0]:.2e}  {epoch_secs[-1]:.1f}s", flush=True)
    train_time = time.time() - t0

    loss_csv = os.path.join(losses_dir, f"{tagp}seed_{a.seed}_losses.csv")
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "phase", "loss_total"]); w.writerows(records)
    tr = [l for _, p, l in records if p == "train"]
    va = [l for _, p, l in records if p == "val"]
    first_nan = next((s for s, _, l in records if not np.isfinite(l)), None)

    # --- generate (timed) ---
    g0 = time.time()
    Xg_std = model.generate(a.gen_num, SEQ_LEN, gen_batch=a.gen_batch)   # (N, T, K) standardized
    Xg = Xg_std.astype(np.float64) * std + mean                          # price scale, per channel
    raw_min, raw_max = float(Xg.min()), float(Xg.max())
    Xg, n_nonpos_s0, n_nonpos_all = rescale_to_s0(Xg)
    gen_time = time.time() - g0

    s0_exact = bool(np.all(Xg[:, 0, :] == S0))
    finite_pos = bool(np.isfinite(Xg).all() and (Xg > 0).all())

    if not a.tag:
        torch.save({"model": model.state_dict(), "seed": a.seed,
                    "zscore_mean": mean.tolist(), "zscore_std": std.tolist()},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        retuned = []
        if epochs != BASE_CONFIG["train"]["epochs"]:
            retuned.append("epochs")
        if batch_size != BASE_CONFIG["train"]["batch_size"]:
            retuned.append("batch_size")
        out_cfg = {
            "method": "CSDI", "seed": a.seed, "dataset": "HestonMultiAsset", "d": D,
            "seq_len": SEQ_LEN, "n_train": int(S.shape[0]), "joint_or_per_asset": "joint",
            "is_unconditional": 1, "target_dim": D,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            **cfg["diffusion"], "timeemb": cfg["model"]["timeemb"],
            "featureemb": cfg["model"]["featureemb"],
            "params": int(nparam), "paper_hyperparams": not retuned,
            "retuned_for_d8": retuned,
            "zscore": "per-channel", "zscore_mean": mean.tolist(), "zscore_std": std.tolist(),
            "train_time_sec": round(train_time, 1), "gpu": GPU_NAME,
        }
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(out_cfg, f, indent=2)

    np.save(os.path.join(gen_dir, f"{tagp}generated_paths_8192x252x8.npy"), Xg)
    meta = {
        "method": "CSDI", "seed": a.seed, "shape": list(Xg.shape), "dtype": str(Xg.dtype),
        "S0": S0, "S0_exact": s0_exact, "S_min": float(Xg.min()), "S_max": float(Xg.max()),
        "all_finite_positive": finite_pos,
        "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
        "gen_time_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
        "sec_per_epoch": round(float(np.mean(epoch_secs)), 1) if epoch_secs else None,
        "epochs_run": epochs, "batch_size": batch_size, "params": int(nparam),
        "num_steps": cfg["diffusion"]["num_steps"],
        "min_total_loss": float(min(tr)) if tr else None,
        "min_val_loss": float(min(va)) if va else None,
        "first_nan_step": first_nan,
        "raw_price_min": raw_min, "raw_price_max": raw_max,
        "n_nonpositive_s0_before_rescale": n_nonpos_s0,
        "n_nonpositive_total_before_rescale": n_nonpos_all,
        "n_entries": int(Xg.size),
        "s0_rescaled": True, "gpu": GPU_NAME, "date": time.strftime("%Y-%m-%d"),
    }
    with open(os.path.join(gen_dir, f"{tagp}metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
