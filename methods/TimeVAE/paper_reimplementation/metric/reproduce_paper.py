"""
TimeVAE — paper reproduction (train + generate half).  Runs in the torch env.

Reproduces the TimeVAE headline result (Desai et al., 2021, Table 1: discriminative
score) on one of the paper's own datasets using our faithful PyTorch port
(``../../code/timevae_torch.py``) with the released hyperparameters, verbatim from
``../../code/reference/src/config/hyperparameters.yaml`` (``timeVAE`` preset):

    latent_dim=8, hidden_layer_sizes=[50,100,200], reconstruction_wt=3.0,
    batch_size=16, use_residual_conn=True, trend_poly=0, custom_seas=None
    Adam(lr=1e-3), max_epochs=1000 + EarlyStopping(patience=50, min_delta=1e-2) +
    ReduceLROnPlateau(factor=0.5, patience=30)   [vae_base.fit_on_data]

Protocol (matches the released ``vae_pipeline.run_vae_pipeline``):
  load .npz -> MinMax scale to [0,1] -> train -> prior-generate N=len(train) samples.
The generated + real arrays are written in the **[0,1] scaled** space (the space the
TimeGAN discriminative/predictive judges score in).  Scoring is done separately by
``score_paper.py`` in the legacy-keras TF env (the original TimeGAN metric code).

Each seed re-seeds torch/numpy so model init + prior draws vary, producing the
distribution reported as mean +/- std across seeds.

Usage:
  CUDA_VISIBLE_DEVICES=0 python reproduce_paper.py --dataset sine --perc 100 \
      --seed 0 --outdir ../results/artifacts
"""
import argparse, csv, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
from timevae_torch import TimeVAE, train_timevae, MinMaxScaler  # noqa: E402

REF_DATA = os.path.join(CODE, "reference", "data")


def main(a):
    t0 = time.time()
    npz = os.path.join(REF_DATA, f"{a.dataset}_subsampled_train_perc_{a.perc}.npz")
    data = np.load(npz)["data"].astype(np.float32)
    N, T, D = data.shape
    print(f"[data] {a.dataset} perc={a.perc} shape={data.shape} "
          f"min={data.min():.4f} max={data.max():.4f}", flush=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)                 # [0,1]
    os.makedirs(a.outdir, exist_ok=True)

    # real scaled saved once (seed-independent) so the scorer always finds it
    real_path = os.path.join(a.outdir, f"{a.dataset}_real_scaled.npy")
    if not os.path.exists(real_path):
        np.save(real_path, scaled.astype(np.float32))

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = TimeVAE(
        seq_len=T, feat_dim=D,
        latent_dim=a.latent_dim,
        hidden_layer_sizes=tuple(a.hidden),
        reconstruction_wt=a.reconstruction_wt,
        trend_poly=0, custom_seas=None, use_residual_conn=True,
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hist = train_timevae(
        model, scaled, max_epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, device=dev, verbose=1, log_every=a.log_every, seed=a.seed,
    )

    gen = model.generate(N, device=dev).astype(np.float32)  # prior samples, [0,1]-ish
    gen_path = os.path.join(a.outdir, f"{a.dataset}_gen_seed{a.seed}.npy")
    np.save(gen_path, gen)

    # per-seed training history
    hist_path = os.path.join(a.outdir, f"{a.dataset}_hist_seed{a.seed}.csv")
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "reconstruction_loss", "kl_loss", "lr"])
        w.writeheader()
        w.writerows(hist)

    print(f"[done] seed={a.seed} epochs={len(hist)} "
          f"first_total={hist[0]['total_loss']:.2f} last_total={hist[-1]['total_loss']:.2f} "
          f"gen={gen.shape} nan={bool(np.isnan(gen).any())} "
          f"gen_range=[{gen.min():.3f},{gen.max():.3f}] ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sine", choices=["sine", "stockv", "energy", "air"])
    p.add_argument("--perc", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=8, dest="latent_dim")
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--reconstruction-wt", type=float, default=3.0, dest="reconstruction_wt")
    p.add_argument("--log-every", type=int, default=25, dest="log_every")
    p.add_argument("--outdir", default=os.path.join(HERE, "..", "results", "artifacts"))
    main(p.parse_args())
