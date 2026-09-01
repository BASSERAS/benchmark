"""
SBBTS on the benchmark Heston dataset — one seed, one GPU.

Pipeline (Alexandre Alouadi, 2026-09-01: "same approach as SBTS, transform prices
into log-returns, generate that, and invert to recover prices"). The log-return
convention is copied from ``methods/SBTS/code/sbts_generate.py`` so the two
Schrodinger-bridge methods are transformed identically:

  1. R = diff(log S)                                   (M, 127)
  2. X[:, 0] = 0; X[:, 1:] = R                         (M, 128, 1)
  3. scale = X.std(over path and time) / sqrt(T)       scalar per channel
     X /= scale                                        (run_heston.py convention)
  4. Train ScoreNN with Algorithm 1 (large-beta branch, beta = 100, K = 5).
  5. Generate, then exp(cumsum(sample * scale))        (M_simu, 127, 1)
  6. Anchor at S0: S[:, 0] = S0; S[:, 1:] = S0 * ratio (M_simu, 128)

d = 1 (price only). The paper's own experiment is bivariate because its metric is
a (price, variance) MLE, but the benchmark scores price paths only and every other
method here sees price only -- feeding SBBTS the variance channel would be
privileged information. See ../paper_reimplementation/ for the d = 2 run.

Run (per benchmark GUIDELINE section 4.1/4.2):

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 0 --gpu 0
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
BENCH = CODE.parent.parent.parent
sys.path.insert(0, str(CODE))

from sbbts_torch import ScoreNN, generate_dsbm, training_sbbts_dsbm  # noqa: E402

# Benchmark dataset conventions (identical to methods/SBTS/code/sbts_generate.py).
S0 = 100.0
SEQ_LEN = 128

# Canonical SBBTS hyperparameters: authors' run_heston.py.
CFG = dict(
    T=1, beta=100, K=5, safe_t=1e-2,
    batch_size=128, n_epochs=1000, lr=1e-3, patience=15, delta=1e-3,
    d_model=128, hidden_dim=64, nhead=32, n_layers=2,
    N_pi=60, M_simu=8192, N_batch=4,
)

DATA_PATH = BENCH / "dataset" / "Heston" / "heston_S_8192x128.npy"
METHOD_ROOT = BENCH / "methods" / "SBBTS"


def to_scaled_log_returns(S, T, device):
    """Prices (M, 128) -> scaled log-returns (M, 128, 1) plus the scale tensor."""
    R = np.diff(np.log(S), axis=1)                        # (M, 127)
    X = np.zeros((S.shape[0], S.shape[1], 1), dtype=np.float64)
    X[:, 1:, 0] = R
    X = torch.tensor(X, dtype=torch.float32, device=device)
    scale = X.std(dim=(0, 1)) / np.sqrt(T)                # (1,)
    return X / scale, scale


def to_prices(ratio, s0=S0):
    """Generated price ratios (M, 127, 1) -> price paths (M, 128) anchored at s0."""
    ratio = np.asarray(ratio, dtype=np.float64)
    if ratio.ndim == 3:
        ratio = ratio[:, :, 0]
    S = np.empty((ratio.shape[0], ratio.shape[1] + 1), dtype=np.float64)
    S[:, 0] = s0
    S[:, 1:] = s0 * ratio
    return np.clip(S, 1e-6, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0,
                    help="informational only; select the device with CUDA_VISIBLE_DEVICES")
    ap.add_argument("--n-epochs", type=int, default=CFG["n_epochs"])
    ap.add_argument("--K", type=int, default=CFG["K"])
    ap.add_argument("--beta", type=float, default=CFG["beta"])
    ap.add_argument("--N-pi", type=int, default=CFG["N_pi"])
    ap.add_argument("--d-model", type=int, default=CFG["d_model"])
    ap.add_argument("--nhead", type=int, default=CFG["nhead"])
    ap.add_argument("--n-layers", type=int, default=CFG["n_layers"])
    ap.add_argument("--hidden-dim", type=int, default=CFG["hidden_dim"])
    ap.add_argument("--M-simu", type=int, default=CFG["M_simu"])
    ap.add_argument("--data", type=str, default=str(DATA_PATH))
    ap.add_argument("--out-root", type=str, default=str(METHOD_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    gen_dir = out_root / "generated_paths" / f"seed_{args.seed}"
    w_dir = out_root / "weights"
    l_dir = out_root / "losses"
    for p in (gen_dir, w_dir, l_dir):
        p.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"SBBTS | Heston | seed {args.seed} | {gpu_name} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    print("=" * 70, flush=True)

    S_train = np.load(args.data)
    assert S_train.shape[1] == SEQ_LEN, f"expected length {SEQ_LEN}, got {S_train.shape}"
    print(f"train data {S_train.shape}  prices [{S_train.min():.2f}, {S_train.max():.2f}]",
          flush=True)

    X, scale = to_scaled_log_returns(S_train, CFG["T"], device)
    N = X.shape[1] - 1                                     # 127 bridge intervals
    d = X.shape[-1]
    print(f"scaled log-returns {tuple(X.shape)}  scale={scale.item():.6f}  N={N}", flush=True)

    model = ScoreNN(d, args.d_model, args.hidden_dim, args.nhead,
                    args.n_layers, N + 1, device=device).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"ScoreNN d={d} params={n_par:,}", flush=True)

    t0 = time.perf_counter()
    model, y_0, history = training_sbbts_dsbm(
        X, model, CFG["T"], args.beta, args.K,
        n_epochs=args.n_epochs, batch_size=CFG["batch_size"],
        patience=CFG["patience"], delta=CFG["delta"],
        safe_t=CFG["safe_t"], lr=CFG["lr"],
    )
    train_sec = time.perf_counter() - t0
    print(f"training done in {train_sec / 60:.1f} min", flush=True)

    torch.save(model.state_dict(), w_dir / f"seed_{args.seed}_model.pt")
    torch.save({"y_0": y_0.cpu(), "scale": scale.cpu()},
               w_dir / f"seed_{args.seed}_transport.pt")

    with open(l_dir / f"seed_{args.seed}_losses.csv", "w") as f:
        f.write("step,phase,loss_total,val_loss\n")
        for step, h in enumerate(history, start=1):
            f.write(f"{step},k{h['k']},{h['train_loss']:.8f},{h['val_loss']:.8f}\n")

    model.eval()
    t0 = time.perf_counter()
    ratio = generate_dsbm(N, X, model, y_0, N_pi=args.N_pi, T=CFG["T"],
                          beta=args.beta, M_simu=args.M_simu, N_batch=CFG["N_batch"],
                          scale=scale, exp=True, safe_t=CFG["safe_t"])
    gen_sec = time.perf_counter() - t0

    S_gen = to_prices(ratio)
    n_bad = int((~np.isfinite(S_gen)).sum())
    print(f"generated {S_gen.shape} in {gen_sec / 60:.1f} min  "
          f"prices [{S_gen.min():.2f}, {S_gen.max():.2f}]  non-finite={n_bad}", flush=True)
    np.save(gen_dir / f"generated_paths_{S_gen.shape[0]}x{S_gen.shape[1]}.npy", S_gen)

    meta = dict(
        method="SBBTS", seed=args.seed, shape=list(S_gen.shape),
        min_val=float(S_gen.min()), max_val=float(S_gen.max()),
        gen_time_sec=round(gen_sec, 1), train_time_sec=round(train_sec, 1),
        gpu=gpu_name, date=date.today().isoformat(),
    )
    (gen_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    cfg = dict(
        method="SBBTS", seed=args.seed, dataset="Heston",
        d=d, seq_len=SEQ_LEN, N_intervals=N, S0=S0,
        beta=args.beta, K=args.K, T=CFG["T"], safe_t=CFG["safe_t"],
        d_model=args.d_model, hidden_dim=args.hidden_dim,
        nhead=args.nhead, n_layers=args.n_layers,
        batch_size=CFG["batch_size"], n_epochs=args.n_epochs, lr=CFG["lr"],
        patience=CFG["patience"], delta=CFG["delta"],
        N_pi=args.N_pi, M_simu=args.M_simu, N_batch=CFG["N_batch"],
        n_train=int(S_train.shape[0]), n_params=n_par, scale=float(scale.item()),
        paper_hyperparams=(args.beta == CFG["beta"] and args.K == CFG["K"]
                           and args.N_pi == CFG["N_pi"] and args.d_model == CFG["d_model"]
                           and args.nhead == CFG["nhead"] and args.n_layers == CFG["n_layers"]),
    )
    (w_dir / f"seed_{args.seed}_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
