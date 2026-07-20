"""
Diagnostic for the TimeVAE sine discriminative-score gap (ours ~0.11 vs paper 0.021).

Trains one TimeVAE on the paper's sine archive and dumps BOTH:
  - prior samples      z ~ N(0, I)      -> decoder   (what the paper scores)
  - reconstructions    z = z_mean(enc(X)) -> decoder (decoder capacity upper bound)

Comparing disc(real, recon) vs disc(real, prior) isolates the cause:
  recon good but prior bad  -> aggregate-posterior != prior (KL / prior-hole)
  even recon bad            -> decoder cannot represent sine (need capacity/inductive bias)

Configs are chosen by CLI so we can sweep reconstruction_wt (paper tuned it in
[0.5, 3.5], §4.2) and the interpretable seasonal decoder (the paper's headline design).

Usage (torch env):
  CUDA_VISIBLE_DEVICES=0 python diag_sine.py --tag base_wt3 --recon-wt 3
  CUDA_VISIBLE_DEVICES=3 python diag_sine.py --tag base_wt8 --recon-wt 8
  CUDA_VISIBLE_DEVICES=0 python diag_sine.py --tag seas_wt3 --recon-wt 3 --custom-seas 8 3
"""
import argparse, ast, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
from timevae_torch import TimeVAE, train_timevae, MinMaxScaler  # noqa: E402

REF_DATA = os.path.join(CODE, "reference", "data")


def main(a):
    t0 = time.time()
    npz = os.path.join(REF_DATA, f"sine_subsampled_train_perc_{a.perc}.npz")
    data = np.load(npz)["data"].astype(np.float32)
    N, T, D = data.shape
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    os.makedirs(a.outdir, exist_ok=True)

    custom_seas = None
    if a.custom_seas:
        # pairs: [num_seasons, len_per_season, ...]
        cs = a.custom_seas
        custom_seas = [(cs[i], cs[i + 1]) for i in range(0, len(cs), 2)]

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = TimeVAE(
        seq_len=T, feat_dim=D, latent_dim=a.latent_dim,
        hidden_layer_sizes=tuple(a.hidden), reconstruction_wt=a.recon_wt,
        trend_poly=a.trend_poly, custom_seas=custom_seas, use_residual_conn=True,
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hist = train_timevae(
        model, scaled, max_epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        device=dev, verbose=1, log_every=a.log_every, seed=a.seed,
        es_patience=a.patience,
    )

    prior = model.generate(N, device=dev).astype(np.float32)
    with torch.no_grad():
        Xt = torch.as_tensor(scaled, dtype=torch.float32, device=dev)
        recon = model.reconstruct(Xt).cpu().numpy().astype(np.float32)

    np.save(os.path.join(a.outdir, f"sine_diag_{a.tag}_prior.npy"), prior)
    np.save(os.path.join(a.outdir, f"sine_diag_{a.tag}_recon.npy"), recon)
    print(f"[done {a.tag}] epochs={len(hist)} recon_wt={a.recon_wt} seas={custom_seas} "
          f"trend={a.trend_poly} | real_std={scaled.std():.4f} "
          f"prior_std={prior.std():.4f} recon_std={recon.std():.4f} "
          f"prior_range=[{prior.min():.3f},{prior.max():.3f}] ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--perc", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=8, dest="latent_dim")
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--recon-wt", type=float, default=3.0, dest="recon_wt")
    p.add_argument("--trend-poly", type=int, default=0, dest="trend_poly")
    p.add_argument("--custom-seas", type=int, nargs="+", default=None, dest="custom_seas")
    p.add_argument("--log-every", type=int, default=50, dest="log_every")
    p.add_argument("--outdir", default=os.path.join(HERE, "..", "results", "diag"))
    main(p.parse_args())
