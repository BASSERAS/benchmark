#!/usr/bin/env python
"""Re-score TimeGAN (Stocks) with a 1-LAYER GRU discriminator, across 5 seeds.

WHY THIS SCRIPT EXISTS
----------------------
The main reproduction (reproduce_stock.py) scores with the SBTS harness, whose
discriminative judge is a **2-layer** GRU. The *official* TimeGAN discriminative
metric (reference/metrics/discriminative_metrics.py) uses a **1-layer** GRU
(hidden = int(dim/2)). A deeper judge is a stronger adversary and mechanically
yields a HIGHER (worse) discriminative score. This script isolates that effect:
it re-implements the official 1-layer judge in PyTorch and runs it across 5
independent training seeds, so we can report the disc score at the *paper's own
judge depth* with proper seed statistics.

  * generator      = our faithful PyTorch TimeGAN port (code/timegan_torch.py),
                     trained fresh per seed with the paper hyperparameters;
  * disc judge     = LOCAL 1-layer GRU here (matches official depth exactly);
  * pred judge     = SBTS predictive_score_metrics (unchanged; already 1-layer
                     in both harnesses, which is why predictive already matched).

Preprocessing mirrors SBTS eval_functions.get_scores verbatim:
  * disc scored on RAW [0,1] data, real sub-sampled to len(fake) via permutation;
  * pred scored on feature-wise min-max scaled data;
  * itt=2000, n_temp=10 test runs.

Run (one seed):  CUDA_VISIBLE_DEVICES=1 python rescore_1layer.py --seed 0
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
TG_CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
SBTS_REF = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "SBTS", "code", "reference")
)
sys.path.insert(0, TG_CODE)                              # timegan_torch
sys.path.insert(0, os.path.join(SBTS_REF, "metrics"))    # predictive_score
sys.path.insert(0, SBTS_REF)

from timegan_torch import TimeGAN                                     # noqa: E402
from metrics.predictive_score import predictive_score_metrics        # noqa: E402

STOCK_CSV = os.path.abspath(
    os.path.join(TG_CODE, "reference", "data", "stock_data.csv")
)
OUT = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)

SEQ_LEN = 24


def min_max_scaler(data):
    num = data - np.min(data, 0)
    den = np.max(data, 0) - np.min(data, 0)
    return num / (den + 1e-7)


def real_data_loading(seq_len):
    """Verbatim port of reference/data_loading.py real_data_loading('stock')."""
    ori = np.loadtxt(STOCK_CSV, delimiter=",", skiprows=1)
    ori = ori[::-1]                       # chronological
    ori = min_max_scaler(ori)             # [0,1] per feature
    windows = [ori[i:i + seq_len] for i in range(len(ori) - seq_len)]
    idx = np.random.permutation(len(windows))
    return np.asarray([windows[i] for i in idx])  # (N, seq_len, d)


def min_max_data(data):
    """Feature-wise min-max over (sample, time), as SBTS eval_functions."""
    m = data.min((0, 1))
    M = data.max((0, 1))
    return (data - m) / (M - m + 1e-12)


# ----------------------------------------------------------------------------
# 1-LAYER GRU discriminator — matches official discriminative_metrics.py depth
# (single GRUCell there; here nn.GRU with num_layers=1, hidden = int(dim/2)).
# ----------------------------------------------------------------------------
class Discriminator1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)          # h: (1, B, H)
        return self.fc(h[-1])       # (B, 1) logit


def disc_score_1layer(ori, gen, iterations, device):
    """Post-hoc 1-layer GRU discriminator. Returns |0.5 - accuracy|.

    Mirrors the official TimeGAN discriminative_metrics.py protocol:
    hidden = int(dim/2), 80/20 train/test split, batch 128, BCE on logits.
    """
    N, T, d = ori.shape
    hidden = max(int(d / 2), 1)
    model = Discriminator1(d, hidden).to(device)
    opt = torch.optim.Adam(model.parameters())
    bce = nn.BCEWithLogitsLoss()

    # 80/20 split for both classes
    def split(x):
        n = len(x)
        idx = np.random.permutation(n)
        cut = int(n * 0.8)
        return x[idx[:cut]], x[idx[cut:]]

    tr_r, te_r = split(ori)
    tr_f, te_f = split(gen)
    tr_r = torch.tensor(tr_r, dtype=torch.float32, device=device)
    tr_f = torch.tensor(tr_f, dtype=torch.float32, device=device)
    te_r = torch.tensor(te_r, dtype=torch.float32, device=device)
    te_f = torch.tensor(te_f, dtype=torch.float32, device=device)

    bs = 128
    model.train()
    for _ in range(iterations):
        ir = np.random.randint(0, len(tr_r), bs)
        iff = np.random.randint(0, len(tr_f), bs)
        xr, xf = tr_r[ir], tr_f[iff]
        opt.zero_grad()
        lr = model(xr)
        lf = model(xf)
        loss = bce(lr, torch.ones_like(lr)) + bce(lf, torch.zeros_like(lf))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pr = torch.sigmoid(model(te_r)).cpu().numpy() > 0.5   # want 1
        pf = torch.sigmoid(model(te_f)).cpu().numpy() > 0.5   # want 0
    correct = int((pr == 1).sum() + (pf == 0).sum())
    total = len(pr) + len(pf)
    acc = correct / total
    return float(np.abs(0.5 - acc))


def get_scores_1layer(X_data, X_sbts, itt, n_temp, device):
    """Mirror SBTS eval_functions.get_scores (min_max=True branch) but swap in
    the LOCAL 1-layer disc judge. Same per-temp permutation shared by disc+pred,
    same feature-wise min-max scaling for the predictive TSTR judge."""
    disc, pred = np.zeros(n_temp), np.zeros(n_temp)
    Xr_scaled = min_max_data(X_data)
    Xf_scaled = min_max_data(X_sbts)
    col = X_data.shape[-1] - 1                    # predict last feature
    for i in range(n_temp):
        # len(real) >= len(fake): sub-sample real to len(fake) (same idx reused)
        idx = np.random.permutation(len(X_data))
        sub = idx[:len(X_sbts)]
        disc[i] = disc_score_1layer(X_data[sub], X_sbts, itt, device)
        pred[i] = predictive_score_metrics(
            Xr_scaled[sub], Xf_scaled, col, itt, device=device
        )
    return disc, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iters", type=int, default=50000)
    args = ap.parse_args()
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[seed {seed}] device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    real = real_data_loading(SEQ_LEN)            # (N, 24, 6), [0,1]
    N, T, d = real.shape
    print(f"[seed {seed}] stock windows = {real.shape}", flush=True)

    model = TimeGAN(
        n_features=d, hidden_dim=24, num_layers=3, batch_size=128,
        device="cuda", embedding_steps=args.iters, supervised_steps=args.iters,
        joint_steps=args.iters, log_every=2000,
    )
    t0 = time.perf_counter()
    model.fit(real)
    train_sec = time.perf_counter() - t0
    print(f"[seed {seed}] trained in {train_sec/60:.1f} min", flush=True)

    fake = model.sample(N)
    if fake.ndim == 2:
        fake = fake[:, :, None]
    print(f"[seed {seed}] generated = {fake.shape}", flush=True)

    disc, pred = get_scores_1layer(real, fake, itt=2000, n_temp=10, device=device)

    result = {
        "dataset": "Stocks",
        "seed": seed,
        "judge": "1-layer GRU discriminator (official TimeGAN depth, hidden=int(d/2))",
        "generator": "PyTorch TimeGAN port, trained fresh this seed",
        "hyperparameters": {"seq_len": SEQ_LEN, "hidden_dim": 24, "num_layers": 3,
                            "iteration_per_phase": args.iters, "batch_size": 128,
                            "train_min": train_sec / 60},
        "protocol": {"itt": 2000, "n_temp": 10},
        "discriminative_score": {"mean": float(disc.mean()), "std": float(disc.std()),
                                  "runs": disc.tolist()},
        "predictive_score": {"mean": float(pred.mean()), "std": float(pred.std()),
                             "runs": pred.tolist()},
        "paper_reference": {"discriminative": "0.102 +/- 0.031",
                            "predictive": "0.038 +/- 0.001"},
    }
    out_path = os.path.join(OUT, f"rescore_1layer_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[seed {seed}] disc(1L) = {disc.mean():.3f} +/- {disc.std():.3f}   "
          f"pred = {pred.mean():.3f} +/- {pred.std():.3f}   paper disc = 0.102")
    print(f"[seed {seed}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
