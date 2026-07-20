"""
CSDI (Tashiro et al., NeurIPS 2021) -- unconditional training + generation on the Heston target.

Trains one CSDI model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then samples 8192 length-128 paths and writes them back in price scale.

Reuses the ORIGINAL repo code verbatim: main_model.CSDI_base is the exact model
(2-D time+feature Transformer diffusion backbone from diff_models.py), the quad
beta schedule / DDPM forward+reverse are the authors' `calc_loss` and `impute`.
Only the data-loading, the z-score wrapper, and a thin CSDI_Heston subclass live
here (Heston paths are already windowed, so the physio/pm25 Dataset classes are
bypassed -- same pattern as methods/DiffusionTS and methods/FourierFlow).

--- Why unconditional generation is `is_unconditional=1` + `cond_mask == 0` ---
The paper (Sec 4.1 / Appendix C) states the `is_unconditional=1` variant "can also
be used for data generation". In that mode `CSDI_base.set_input_to_diffmodel` feeds
the network ONLY the noisy sequence -- `cond_mask` NEVER gates the network input, it
only selects which points enter the loss (`target_mask = observed_mask - cond_mask`).
We therefore set `observed_mask = 1` and `cond_mask = 0` everywhere:
  * training  -> target_mask = 1 everywhere -> every timestep is a denoising target
                 (the standard unconditional DDPM objective);
  * sampling  -> `impute` with cond_mask=0 collapses line 167 to pure ancestral
                 sampling `current_sample` (no conditioning term).
Training and generation thus see the identical input distribution (full noisy
sequence); the architecture, diffusion process and hyperparameters are the paper's
is_unconditional variant, unchanged. See ../code/README.md for the full justification.

Normalisation chain (CSDI's own PhysioNet convention is per-feature z-score; diffusion
needs zero-mean/unit-var data, NOT min-max):
  S(price) --(-mean)/std--> standardized  (model trains + samples here)
  sample   --*std + mean--> price

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0 --frac 0.05 --epochs 20 --tag smoke
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

METHOD = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(METHOD, "reference")
sys.path.insert(0, REF)

from main_model import CSDI_base  # noqa: E402  (path set above)

DEFAULT_DATA = os.path.join(METHOD, "..", "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")

# CSDI hyperparameters -- verbatim from the released config/base.yaml (the same config
# that reproduced Table 2). Kept identical so the Heston run is the paper's model.
BASE_CONFIG = {
    "train": {"epochs": 200, "batch_size": 16, "lr": 1.0e-3, "itr_per_epoch": 1.0e8},
    "diffusion": {
        "layers": 4, "channels": 64, "nheads": 8, "diffusion_embedding_dim": 128,
        "beta_start": 0.0001, "beta_end": 0.5, "num_steps": 50,
        "schedule": "quad", "is_linear": False,
    },
    "model": {
        "is_unconditional": 1,          # <-- unconditional generation variant (paper Sec 4.1)
        "timeemb": 128, "featureemb": 16,
        "target_strategy": "random",     # unused: forward() below forces cond_mask=0
    },
}


class CSDI_Heston(CSDI_base):
    """Unconditional CSDI for single-feature Heston paths (target_dim K=1, L=128).

    process_data yields fully-observed sequences (observed_mask == 1); forward()
    forces cond_mask == 0 so the whole sequence is a denoising target (generation
    regime). Everything else -- get_side_info, calc_loss, impute -- is the parent's
    verbatim code.
    """

    def __init__(self, config, device, target_dim=1):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        # batch["observed_data"]: (B, L, K) -> parent convention, permute to (B, K, L)
        observed_data = batch["observed_data"].to(self.device).float()
        B, L, K = observed_data.shape
        observed_mask = torch.ones(B, L, K, device=self.device)
        gt_mask = torch.zeros(B, L, K, device=self.device)          # cond_mask=0 at eval -> pure generation
        observed_tp = torch.arange(L, device=self.device).float().unsqueeze(0).expand(B, -1)

        observed_data = observed_data.permute(0, 2, 1)              # (B, K, L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(B, device=self.device).long()
        for_pattern_mask = observed_mask
        return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, _gt, _fpm, _cl = self.process_data(batch)
        cond_mask = torch.zeros_like(observed_mask)                 # unconditional: every point is a target
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    @torch.no_grad()
    def generate(self, n_paths, seq_len, gen_batch=512):
        """Draw n_paths unconditional samples, shape (n_paths, seq_len), standardized scale."""
        self.eval()
        out = []
        done = 0
        while done < n_paths:
            B = min(gen_batch, n_paths - done)
            cond_mask = torch.zeros(B, self.target_dim, seq_len, device=self.device)
            observed_tp = torch.arange(seq_len, device=self.device).float().unsqueeze(0).expand(B, -1)
            side_info = self.get_side_info(observed_tp, cond_mask)
            dummy = torch.zeros(B, self.target_dim, seq_len, device=self.device)  # only shapes the sampler
            samples = self.impute(dummy, cond_mask, side_info, n_samples=1)       # (B,1,K,L)
            out.append(samples[:, 0, 0, :].cpu().numpy())                          # (B, L), K=1
            done += B
        return np.concatenate(out, axis=0)[:n_paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=0, help="override epochs (0 = base.yaml default 200)")
    ap.add_argument("--batch_size", type=int, default=0, help="override batch (0 = base.yaml default 16)")
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=512)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training paths to use")
    ap.add_argument("--tag", default="", help="run tag (e.g. 'smoke'); prefixes output dirs, skips canonical weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(json.dumps(BASE_CONFIG))  # deep copy
    epochs = a.epochs if a.epochs > 0 else cfg["train"]["epochs"]
    batch_size = a.batch_size if a.batch_size > 0 else cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    tagp = (a.tag + "_") if a.tag else ""

    # --- data: z-score standardize (CSDI PhysioNet convention) ---
    S = np.load(os.path.abspath(a.data)).astype(np.float64)   # (8192, 128) price
    if a.frac < 1.0:
        S = S[: int(round(S.shape[0] * a.frac))]
    seq_len = S.shape[1]
    mean = float(S.mean())
    std = float(S.std())
    Xn = ((S - mean) / std).astype(np.float32)                # standardized
    Xt = Xn[:, :, None]                                       # (N, L, K=1)
    ds = TensorDataset(torch.from_numpy(Xt))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    # --- model (verbatim CSDI, unconditional) ---
    model = CSDI_Heston(cfg, device, target_dim=1).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[CSDI seed {a.seed}] params={nparam} epochs={epochs} batch={batch_size} "
          f"N={S.shape[0]} T={seq_len}", flush=True)

    # --- output dirs ---
    weights_dir = os.path.join(METHOD, "..", "weights")
    losses_dir = os.path.join(METHOD, "..", "losses")
    gen_dir = os.path.join(METHOD, "..", "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- training loop (CSDI's utils.train recipe: Adam wd=1e-6, MultiStepLR @0.75/0.9 x0.1) ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1.0e-6)
    p1, p2 = int(0.75 * epochs), int(0.9 * epochs)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    records = []          # (step, loss)
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
        lr_sched.step()
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            recent = np.mean([l for _, l in records[-len(dl):]])
            print(f"  epoch {epoch:4d}/{epochs}  avg_loss={recent:.5f}  lr={lr_sched.get_last_lr()[0]:.2e}",
                  flush=True)
    train_time = time.time() - t0

    # --- loss curve ---
    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"])
        w.writerows(records)
    first_nan = next((s for s, l in records if not np.isfinite(l)), None)
    min_loss = min((l for _, l in records), default=float("nan"))

    # --- generate paths (timed) ---
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    Xg_std = model.generate(gen_n, seq_len, gen_batch=a.gen_batch)   # (N, L) standardized
    Xg = Xg_std * std + mean                                          # price scale
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- canonical weights/config (GUIDELINE schema) for real runs only ---
    if not a.tag:
        torch.save({"model": model.state_dict(), "seed": a.seed, "zscore": [mean, std]},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        out_cfg = {"is_unconditional": 1, "target_dim": 1, "seq_length": seq_len,
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   **cfg["diffusion"], "timeemb": cfg["model"]["timeemb"],
                   "featureemb": cfg["model"]["featureemb"], "seed": a.seed,
                   "zscore_mean": mean, "zscore_std": std}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(out_cfg, f, indent=2)

    meta = {"method": "CSDI", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "zscore_mean": mean, "zscore_std": std, "params": int(nparam),
            "epochs": epochs, "batch_size": batch_size, "num_steps": cfg["diffusion"]["num_steps"],
            "min_loss": float(min_loss), "first_nan_step": first_nan,
            "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
