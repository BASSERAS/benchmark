"""
COSCI-GAN paper reproduction — train on EEG eye-state (5 best channels).

Faithful re-run of the upstream three-player COSCI-GAN training loop
(Main_modules.COSCIGAN, Code/COSCI-GAN_demo.ipynb) with the paper/demo
hyper-parameters:

  n_groups = 5   (5 EEG channels)          gamma            = 5.0
  n_samples= 100 (window length)           generator_lr     = 1e-3
  noise_len= 32                            discriminator_lr = 1e-3
  batch_size = 32                          central_disc_lr  = 1e-4
  num_epochs = 200 (demo notebook)         criterion        = BCE
  LSTM_G = LSTM_D = True, CD_type = MLP     betas            = (0.5, 0.9)

The nn.Module classes are imported verbatim from the copied upstream
reference (code/reference/Main_modules.py) so the architecture is identical.
The loop below is transcribed from Main_modules.COSCIGAN (upstream lines
263-381); only I/O differs (single-GPU, in-memory generators, loss logging,
npy export in the exact upstream evaluation convention).

Output:
  results/ours_eeg_generated_label{L}.npy   shape (1024, 100, 5)
  results/train_losses_label{L}.csv         per-epoch loss_D_i / loss_G_i / loss_CD
"""
import argparse
import os
import sys
import csv

import numpy as np
import pandas as pd
import torch
from torch import nn

REF = os.path.join(os.path.dirname(__file__), "..", "code", "reference")
sys.path.insert(0, os.path.abspath(REF))
from Main_modules import LSTMGenerator, LSTMDiscriminator, Discriminator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="EEG 5best block-layout CSV")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--n_groups", type=int, default=5)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--noise_len", type=int, default=32)
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--glr", type=float, default=1e-3)
    ap.add_argument("--dlr", type=float, default=1e-3)
    ap.add_argument("--cdlr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ng, ns, nl = args.n_groups, args.n_samples, args.noise_len

    # ---- data (block channel-major CSV) ----
    df = pd.read_csv(args.real)
    train = torch.tensor(df.values.astype(np.float32))
    assert train.shape[1] == ng * ns, (train.shape, ng * ns)
    ds = torch.utils.data.TensorDataset(train)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    def init_w(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # ---- models (LSTM G/D per channel + MLP central discriminator) ----
    G = {i: LSTMGenerator(latent_dim=nl, ts_dim=ns).to(device) for i in range(ng)}
    D = {i: LSTMDiscriminator(ts_dim=ns).to(device) for i in range(ng)}
    CD = Discriminator(n_samples=ng * ns, alpha=0.1).apply(init_w).to(device)

    loss_fn = nn.BCELoss()
    optD = {i: torch.optim.Adam(D[i].parameters(), lr=args.dlr, betas=[0.5, 0.9]) for i in range(ng)}
    optG = {i: torch.optim.Adam(G[i].parameters(), lr=args.glr, betas=[0.5, 0.9]) for i in range(ng)}
    optCD = torch.optim.Adam(CD.parameters(), lr=args.cdlr, betas=[0.5, 0.9])

    log_path = os.path.join(args.outdir, f"train_losses_label{args.label}.csv")
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch"] + [f"loss_D_{i}" for i in range(ng)]
                   + [f"loss_G_{i}" for i in range(ng)] + ["loss_CD"])

        for epoch in range(args.num_epochs):
            for (signals,) in loader:
                signals = signals.to(device)
                nb = len(signals)
                grp = {i: signals[:, i * ns:(i + 1) * ns] for i in range(ng)}
                z = torch.randn((nb, nl), device=device).float()  # shared noise
                gen = {i: G[i](z).float() for i in range(ng)}

                lbl0 = torch.zeros((nb, 1), device=device)
                lbl1 = torch.ones((nb, 1), device=device)
                lbl_all = torch.cat((lbl1, lbl0))

                # discriminators
                loss_D = {}
                for i in range(ng):
                    optD[i].zero_grad()
                    out = D[i](torch.cat((grp[i], gen[i])).float())
                    loss_D[i] = loss_fn(out, lbl_all)
                    loss_D[i].backward(retain_graph=True)
                    optD[i].step()

                # central discriminator
                g_cat = gen[0]
                r_cat = grp[0]
                for i in range(1, ng):
                    g_cat = torch.hstack((g_cat, gen[i]))
                    r_cat = torch.hstack((r_cat, grp[i]))
                all_cd = torch.cat((g_cat, r_cat))
                lbl_cd = torch.cat((torch.zeros((nb, 1), device=device),
                                    torch.ones((nb, 1), device=device)))
                optCD.zero_grad()
                out_cd = CD(all_cd.float())
                loss_CD = loss_fn(out_cd, lbl_cd)
                loss_CD.backward(retain_graph=True)
                optCD.step()

                # generators (local adversarial - gamma * central)
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
                    loss_G[i] = local - args.gamma * loss_cd_new
                    loss_G[i].backward(retain_graph=True)
                    optG[i].step()

            w.writerow([epoch] + [f"{loss_D[i].item():.5f}" for i in range(ng)]
                       + [f"{loss_G[i].item():.5f}" for i in range(ng)]
                       + [f"{loss_CD.item():.5f}"])
            if epoch % 20 == 0 or epoch == args.num_epochs - 1:
                print(f"epoch {epoch:3d}  loss_CD={loss_CD.item():.4f}  "
                      f"loss_D0={loss_D[0].item():.4f}  loss_G0={loss_G[0].item():.4f}",
                      flush=True)

    # ---- generate 1024 samples, upstream export convention ----
    for i in range(ng):
        G[i].eval()
    with torch.no_grad():
        znew = torch.randn((1024, nl), device=device).float()
        cols = [G[i](znew).cpu().numpy() for i in range(ng)]  # each (1024, 100)
    block = np.hstack(cols)                    # (1024, 500) block channel-major
    gen_arr = block.reshape(block.shape[0], -1, ng)  # (1024, 100, 5), upstream reshape
    out_npy = os.path.join(args.outdir, f"ours_eeg_generated_label{args.label}.npy")
    np.save(out_npy, gen_arr)
    print("saved", out_npy, gen_arr.shape)
    print("losses", log_path)


if __name__ == "__main__":
    main()
