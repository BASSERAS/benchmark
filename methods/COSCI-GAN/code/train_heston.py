"""
COSCI-GAN (Seyfi, Rajotte, Ng -- NeurIPS 2022) -- training + generation on the
Heston target, in the single-channel (C = 1) configuration.

Trains one COSCI-GAN model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then samples 8192 length-128 paths and writes them back in price scale to
  ../generated_paths/seed_{i}/generated_paths_8192x128.npy   (8192, 128) float64

--------------------------------------------------------------------------------
Why C = 1, and what the Central Discriminator degenerates to
--------------------------------------------------------------------------------
COSCI-GAN is a *multivariate* generator: C per-channel "Channel GANs"
(LSTMGenerator + LSTMDiscriminator each) plus one MLP Central Discriminator (CD)
that sees the concatenation of all C channels and enforces inter-channel
correlation. The Heston target in this benchmark is a univariate price series
(C = 1), so:

  * There is exactly ONE Channel GAN (one LSTM generator + one LSTM discriminator).
  * The shared-noise mechanism (all generators share one z) is trivial with a
    single generator.
  * The Central Discriminator, fed the concatenation of "all" channels, receives
    the SAME 128-dim vector as the single channel discriminator -- it becomes a
    redundant second (MLP) critic on the one channel. There is no cross-channel
    correlation for it to preserve (the notion is undefined for C = 1).

We keep the full three-player machinery ON and honest (with_CD = True), exactly
as the paper's code runs it, and document the degeneracy rather than silently
dropping the CD. In this regime COSCI-GAN reduces to a single LSTM-GAN regularised
by an auxiliary MLP critic. It still produces valid univariate paths, which is
what the benchmark evaluates.

--------------------------------------------------------------------------------
Faithfulness
--------------------------------------------------------------------------------
The nn.Module classes (LSTMGenerator, LSTMDiscriminator, Discriminator) are
imported verbatim from the copied upstream reference
(code/reference/Main_modules.py). The training loop below is transcribed from
Main_modules.COSCIGAN (upstream lines 263-381); only I/O differs (single-GPU,
in-memory generator, MinMax price<->[0,1] scaling, loss logging, npy export).

Paper / demo hyper-parameters (kept verbatim; justified in code/README.md):
  noise_len = 32       batch_size = 32        gamma = 5.0
  generator_lr = 1e-3  discriminator_lr = 1e-3  central_disc_lr = 1e-4
  criterion = BCE      LSTM_G = LSTM_D = True   CD_type = MLP
  Adam betas = (0.5, 0.9)   hidden_dim = 256 (LSTM), num_layers = 1

Only two knobs are Heston-specific and justified in the README:
  * n_samples = 128  (the Heston sequence length, vs 100 for EEG windows)
  * num_epochs       (chosen so total generator updates are comparable to the
                      paper's EEG run; see README. Default 120.)

Normalisation chain (analog of the upstream "ZeroOne" EEG preprocessing, which
min-max scales each channel to [0,1]; here one channel -> one scalar min/max):
  S(price) --(X-min)/(max-min)--> ~[0,1]  (model trains here)
  generated --* (max-min) + min--> price

Usage:
  CUDA_VISIBLE_DEVICES=3 python train_heston.py --seed 0
  CUDA_VISIBLE_DEVICES=3 python train_heston.py --seed 0 --frac 0.05 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
from torch import nn

METHOD_CODE = os.path.dirname(os.path.abspath(__file__))          # methods/COSCI-GAN/code
METHOD_DIR = os.path.dirname(METHOD_CODE)                          # methods/COSCI-GAN
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))          # benchmark/
REF = os.path.join(METHOD_CODE, "reference")
sys.path.insert(0, REF)
from Main_modules import LSTMGenerator, LSTMDiscriminator, Discriminator  # noqa: E402

DEFAULT_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_8192x128.npy")

# paper / demo preset (Main_modules.COSCIGAN defaults + demo notebook)
PRESET = dict(noise_len=32, n_samples=128, gamma=5.0,
              glr=1e-3, dlr=1e-3, cdlr=1e-4, batch_size=32,
              betas=(0.5, 0.9), alpha=0.1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n_groups", type=int, default=1, help="channels (Heston = 1)")
    ap.add_argument("--n_samples", type=int, default=PRESET["n_samples"])
    ap.add_argument("--noise_len", type=int, default=PRESET["noise_len"])
    ap.add_argument("--num_epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=PRESET["batch_size"])
    ap.add_argument("--gamma", type=float, default=PRESET["gamma"])
    ap.add_argument("--glr", type=float, default=PRESET["glr"])
    ap.add_argument("--dlr", type=float, default=PRESET["dlr"])
    ap.add_argument("--cdlr", type=float, default=PRESET["cdlr"])
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of paths (smoke test)")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--tag", default="", help="run tag ('smoke'); prefixes outputs, skips canonical weights")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (a.tag + "_") if a.tag else ""
    ng, ns, nl = a.n_groups, a.n_samples, a.noise_len

    # ---- data: Heston prices -> [0,1] via scalar MinMax (ZeroOne analog) ----
    S = np.load(os.path.abspath(a.data)).astype(np.float64)       # (8192, 128) price
    if a.frac < 1.0:
        S = S[:int(round(S.shape[0] * a.frac))]
    N, seq_len = S.shape
    assert seq_len == ns * ng, (seq_len, ns * ng)
    smin, smax = float(S.min()), float(S.max())
    Xs = ((S - smin) / (smax - smin)).astype(np.float32)          # ~[0,1]
    train = torch.tensor(Xs)
    ds = torch.utils.data.TensorDataset(train)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size,
                                         shuffle=True, pin_memory=True)

    print(f"=== COSCI-GAN Heston (C={ng})  seed={a.seed}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] S{S.shape} price[min={smin:.2f},max={smax:.2f}]  "
          f"scaled[{Xs.min():.4f},{Xs.max():.4f}]  epochs={a.num_epochs}", flush=True)

    # ---- models: 1 LSTM Channel GAN + 1 MLP Central Discriminator ----
    G = {i: LSTMGenerator(latent_dim=nl, ts_dim=ns).to(device) for i in range(ng)}
    D = {i: LSTMDiscriminator(ts_dim=ns).to(device) for i in range(ng)}
    CD = Discriminator(n_samples=ng * ns, alpha=PRESET["alpha"]).apply(init_weights).to(device)
    nparam = (sum(p.numel() for i in range(ng) for p in G[i].parameters())
              + sum(p.numel() for i in range(ng) for p in D[i].parameters())
              + sum(p.numel() for p in CD.parameters()))
    print(f"[model] params={nparam}  (G+D per channel x{ng} + MLP CD)", flush=True)

    loss_fn = nn.BCELoss()
    optD = {i: torch.optim.Adam(D[i].parameters(), lr=a.dlr, betas=list(PRESET["betas"])) for i in range(ng)}
    optG = {i: torch.optim.Adam(G[i].parameters(), lr=a.glr, betas=list(PRESET["betas"])) for i in range(ng)}
    optCD = torch.optim.Adam(CD.parameters(), lr=a.cdlr, betas=list(PRESET["betas"]))

    # ---- output dirs ----
    weights_dir = os.path.join(METHOD_DIR, "weights")
    losses_dir = os.path.join(METHOD_DIR, "losses")
    gen_dir = os.path.join(METHOD_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    loss_name = f"{tagp}seed_{a.seed}_losses.csv" if a.tag else f"seed_{a.seed}_losses.csv"
    loss_csv = os.path.join(losses_dir, loss_name)

    hist = []
    t0 = time.time()
    first_nan = None
    with open(loss_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch"] + [f"loss_D_{i}" for i in range(ng)]
                   + [f"loss_G_{i}" for i in range(ng)] + ["loss_CD"])

        for epoch in range(a.num_epochs):
            for (signals,) in loader:
                signals = signals.to(device)
                nb = len(signals)
                grp = {i: signals[:, i * ns:(i + 1) * ns] for i in range(ng)}
                z = torch.randn((nb, nl), device=device).float()   # shared noise
                gen = {i: G[i](z).float() for i in range(ng)}

                lbl0 = torch.zeros((nb, 1), device=device)
                lbl1 = torch.ones((nb, 1), device=device)
                lbl_all = torch.cat((lbl1, lbl0))

                # channel discriminators
                loss_D = {}
                for i in range(ng):
                    optD[i].zero_grad()
                    out = D[i](torch.cat((grp[i], gen[i])).float())
                    loss_D[i] = loss_fn(out, lbl_all)
                    loss_D[i].backward(retain_graph=True)
                    optD[i].step()

                # central discriminator (C=1: sees the single channel, redundant critic)
                g_cat = gen[0]
                r_cat = grp[0]
                for i in range(1, ng):
                    g_cat = torch.hstack((g_cat, gen[i]))
                    r_cat = torch.hstack((r_cat, grp[i]))
                all_cd = torch.cat((g_cat, r_cat))
                lbl_cd = torch.cat((lbl0, lbl1))
                optCD.zero_grad()
                out_cd = CD(all_cd.float())
                loss_CD = loss_fn(out_cd, lbl_cd)
                loss_CD.backward(retain_graph=True)
                optCD.step()

                # generators: local adversarial - gamma * central
                loss_G = {}
                for i in range(ng):
                    optG[i].zero_grad()
                    local = loss_fn(D[i](gen[i]), lbl1)
                    gen_new = {}
                    for j in range(ng):
                        gj = G[j](z)
                        gen_new[j] = gj.float() if i == j else gj.detach().float()
                    gcat = gen_new[0]
                    for j in range(1, ng):
                        gcat = torch.hstack((gcat, gen_new[j]))
                    cd_new = CD(torch.cat((gcat, r_cat)).float())
                    loss_cd_new = loss_fn(cd_new, lbl_cd)
                    loss_G[i] = local - a.gamma * loss_cd_new
                    loss_G[i].backward(retain_graph=True)
                    optG[i].step()

            row = {"epoch": epoch,
                   **{f"loss_D_{i}": float(loss_D[i].item()) for i in range(ng)},
                   **{f"loss_G_{i}": float(loss_G[i].item()) for i in range(ng)},
                   "loss_CD": float(loss_CD.item())}
            hist.append(row)
            w.writerow([epoch] + [f"{loss_D[i].item():.5f}" for i in range(ng)]
                       + [f"{loss_G[i].item():.5f}" for i in range(ng)]
                       + [f"{loss_CD.item():.5f}"])
            if first_nan is None and not np.isfinite(loss_CD.item()):
                first_nan = epoch
            if epoch % a.log_every == 0 or epoch == a.num_epochs - 1:
                print(f"epoch {epoch:3d}  loss_CD={loss_CD.item():.4f}  "
                      f"loss_D0={loss_D[0].item():.4f}  loss_G0={loss_G[0].item():.4f}",
                      flush=True)
    train_time = time.time() - t0

    # ---- generate gen_num paths, invert to price scale ----
    for i in range(ng):
        G[i].eval()
    gen_n = int(round(a.gen_num * a.frac)) if a.frac < 1.0 else a.gen_num
    g0 = time.time()
    with torch.no_grad():
        znew = torch.randn((gen_n, nl), device=device).float()
        cols = [G[i](znew).cpu().numpy() for i in range(ng)]       # each (gen_n, ns)
    block = np.hstack(cols)                                        # (gen_n, ns*ng)
    Xg = (block * (smax - smin) + smin).astype(np.float64)         # price scale, (gen_n, seq_len)
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg)

    # ---- weights + config (canonical runs only) ----
    if not a.tag:
        torch.save({"G": {i: G[i].state_dict() for i in range(ng)},
                    "D": {i: D[i].state_dict() for i in range(ng)},
                    "CD": CD.state_dict(),
                    "scaler_min": smin, "scaler_max": smax, "seed": a.seed},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "COSCI-GAN", "variant": f"C={ng} (univariate Heston)",
               "seed": a.seed, "n_groups": ng, "n_samples": ns, "noise_len": nl,
               "num_epochs": a.num_epochs, "batch_size": a.batch_size,
               "gamma": a.gamma, "glr": a.glr, "dlr": a.dlr, "cdlr": a.cdlr,
               "criterion": "BCE", "LSTM_G": True, "LSTM_D": True, "CD_type": "MLP",
               "betas": list(PRESET["betas"]), "hidden_dim": 256, "num_layers": 1,
               "params": int(nparam), "epochs_run": len(hist)}
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # ---- metadata (GUIDELINE §4.3 schema) ----
    meta = {"method": "COSCI-GAN", "seed": a.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(nparam), "epochs_run": len(hist), "epochs_max": a.num_epochs,
            "first_nan_epoch": first_nan, "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if a.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} epochs={len(hist)} "
          f"loss_CD_last={hist[-1]['loss_CD']:.4f} gen={Xg.shape} "
          f"price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
