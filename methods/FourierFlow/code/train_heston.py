"""
Fourier Flow — training + generation on the Heston target dataset.

Trains one Fourier Flow (Alaa et al., ICLR 2021) per seed on the 8192x128 Heston
price paths in ../../../dataset/Heston/heston_S_8192x128.npy, then samples 8192
synthetic paths of length 128 back on the original price scale.

The model code is the *released* implementation, imported verbatim from
./reference (SequentialFlows.FourierFlow). Only the data plumbing (Heston load,
min-max scaling, saving of weights / losses / generated paths / metadata) lives
here.

Per-seed artifacts written (mirroring the other benchmark methods):
  ../weights/seed_{s}_model.pt          full state_dict
  ../weights/seed_{s}_config.json       hyper-parameters + scaling constants
  ../losses/seed_{s}_losses.csv         per-epoch negative-log-likelihood loss
  ../generated_paths/seed_{s}/generated_paths_8192x128.npy   (8192,128) price scale
  ../generated_paths/seed_{s}/metadata.json

Hyper-parameters default to the paper's *Stocks* configuration (Table 2 /
run_experiment_2.py), which is the natural template for Heston: both are
univariate price series. See code/README.md for the full justification and for
how to sweep them.

Usage (single seed):
    cd methods/FourierFlow/code/reference
    PYTHONPATH=$PWD python ../train_heston.py --seed 0

Usage (all 5 seeds in parallel) -> see train_all.sh
"""
from __future__ import absolute_import, division, print_function
import argparse, json, os, time
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)  # match torch 1.3 default

from SequentialFlows import FourierFlow

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = os.path.dirname(HERE)  # methods/FourierFlow
DEFAULT_DATA = os.path.join(METHOD, "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")


class FourierFlowClamp(FourierFlow):
    """Released FourierFlow with TWO numerical guards; flow math otherwise VERBATIM.

    Both guards are needed to train the paper's ``normalize=True`` Stocks regime on
    the Heston target. Neither touches the flow's math (DFT, coupling, base density,
    loss) -- one rescales a degenerate normalisation constant, the other rescales
    gradients, exactly like the ExponentialLR scheduler already in the loop.

    Guard 1 -- zero-variance spectral bin clamp ('Fix A')
    -----------------------------------------------------
    ``normalize=True`` standardises each spectral bin by its across-batch std. For an
    even original length T (here 128, anchored to odd fft_size=T+1=129) exactly ONE
    bin is structurally degenerate -- the imaginary-DC component, identically 0 for
    every real signal -> std == 0 -> ``(x-mean)/std`` is 0/0 -> NaN. The paper's
    Stocks run (odd length 101) never hit this: odd-length FFT roundoff gave its
    imaginary-DC bin a tiny NONZERO std. We clamp any exactly-zero std to 1.0; that
    bin then passes through as a constant 0 (it carries no information).

    Guard 2 -- gradient clipping (max-norm = ``grad_clip``, default 1.0)
    -------------------------------------------------------------------
    Root cause (verified against the UNMODIFIED reference flow, no clip, 1000 epochs):
        - Stocks : first_nan=-1  (stable)   min_loss=23.3  max_gradnorm=2142
        - Heston : first_nan=499 (NaN)      min_loss=50.0  max_gradnorm=8304
    Same code, same hyper-parameters -- ONLY the data differs. Every Heston path
    starts at the identical deterministic S0=100, so Var(value at t=1) across samples
    is ~0 (3.6e-15 vs Stocks 6.1e-2) and the spectral covariance is near-singular
    (92/130 bins with std<1e-3 vs Stocks 41/102). A max-likelihood flow matching a
    near-singular distribution drives its affine-coupling log-scales toward -inf along
    the degenerate directions; around epoch 499 that overflows -> grad-norm explodes
    -> NaN. Stocks avoids it because sliding-window slicing gives every window a
    different start value, so no spectral direction is degenerate. This is a DATA
    property, not a port bug -- the reference has the same latent vulnerability.

    Clipping the gradient norm to ``grad_clip`` caps the log-scale blowup without
    changing any flow math (pure gradient rescale). Verified: no NaN, loss plateau
    ~107 at 1000 epochs.
    """

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100,
            grad_clip=1.0):
        X_train = torch.from_numpy(np.array(X)).float()

        # --- reference fit(), verbatim ---
        X_train_spectral = self.FourierTransform(X_train)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)

        # --- Guard 1: clamp the structurally-zero-variance bin(s) ---
        n_clamped = int((self.fft_std == 0).sum().item())
        if n_clamped:
            self.fft_std = torch.where(
                self.fft_std == 0, torch.ones_like(self.fft_std), self.fft_std)
        self._n_clamped = n_clamped
        self._grad_clip = grad_clip

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        max_gradnorm = 0.0
        for step in range(epochs):
            optim.zero_grad()
            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()
            losses.append(loss.detach().numpy())
            loss.backward()
            # --- Guard 2: clip gradient norm (caps the near-singular log-scale blowup) ---
            if grad_clip is not None and grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                max_gradnorm = max(max_gradnorm, float(gn))
            optim.step()
            scheduler.step()
            if ((step % display_step) == 0) or (step == epochs - 1):
                print(f"step: {step} \t/ {epochs} \tloss: {float(loss):.3f}", flush=True)
        self._max_gradnorm = max_gradnorm
        print("Finished training!", flush=True)
        return losses


def minmax_fit(X):
    lo, hi = float(X.min()), float(X.max())
    return lo, hi


def minmax_apply(X, lo, hi):
    return (X - lo) / (hi - lo)


def minmax_invert(Xn, lo, hi):
    return Xn * (hi - lo) + lo


def main(a):
    t0 = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- data ----
    S = np.load(os.path.abspath(a.dataset)).astype(np.float64)  # (8192, 128) price scale
    N, T = S.shape
    print(f"[seed {a.seed}] target {S.shape} price[min={S.min():.3f} max={S.max():.3f}]", flush=True)
    lo, hi = minmax_fit(S)
    Xn = minmax_apply(S, lo, hi).astype(np.float32)             # [0,1], (N, T)

    # Prepend a 0 anchor -> length T+1 (odd for T=128). This is exactly the
    # released Stocks pipeline (run_experiment_2.real_data_loading: hstack((0, .))).
    # Two reasons it is REQUIRED, not cosmetic:
    #   1. FourierFlow.forward reshapes the DFT output (2*crop_size = N+2 values for
    #      even N) to size N+1 -> only ODD fft_size is consistent. T+1=129 is odd.
    #   2. It matches the released Stocks pipeline exactly.
    # The anchor column is stripped back off after sampling (see below).
    # NOTE: even T leaves exactly ONE structurally-zero-variance spectral bin (the
    # imaginary-DC term). FourierFlowClamp handles it (normalize=True + std clamp);
    # see its docstring. This is why we can keep the paper's normalize=True regime.
    fft_len = T + 1
    Xa = np.hstack([np.zeros((N, 1), dtype=np.float32), Xn])    # (N, T+1)

    # ---- model (released FourierFlow + zero-variance-bin clamp) ----
    model = FourierFlowClamp(hidden=a.hidden, n_flows=a.n_flows,
                             normalize=bool(a.normalize), fft_size=fft_len)

    # report (do NOT abort) how many degenerate spectral bins will be clamped
    if a.normalize:
        spec = model.FourierTransform(torch.from_numpy(Xa).float())[0]
        nzero = int((torch.std(spec, dim=0) == 0).sum().item())
        print(f"[seed {a.seed}] normalize=True; {nzero} zero-variance spectral "
              f"bin(s) will be clamped (std 0->1).", flush=True)

    # ---- train (full-batch, exact max-likelihood) ----
    losses = model.fit(Xa, epochs=a.epochs, batch_size=a.batch_size,
                       learning_rate=a.lr, display_step=a.display_step)
    train_sec = time.time() - t0

    # ---- save weights + config ----
    wdir = os.path.join(METHOD, "weights"); os.makedirs(wdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(wdir, f"seed_{a.seed}_model.pt"))
    cfg = {
        "seed": a.seed, "framework": "pytorch", "model": "FourierFlow",
        "hidden": a.hidden, "n_flows": a.n_flows, "normalize": bool(a.normalize),
        "n_clamped_bins": int(getattr(model, "_n_clamped", 0)),
        "fft_size": fft_len, "prepend_anchor": True, "epochs": a.epochs,
        "batch_size": a.batch_size,
        "learning_rate": a.lr, "N_train": N, "seq_len": T,
        "scale_min": lo, "scale_max": hi, "torch": torch.__version__,
        "train_sec": round(train_sec, 1),
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
    }
    with open(os.path.join(wdir, f"seed_{a.seed}_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- save losses ----
    ldir = os.path.join(METHOD, "losses"); os.makedirs(ldir, exist_ok=True)
    with open(os.path.join(ldir, f"seed_{a.seed}_losses.csv"), "w") as f:
        f.write("epoch,loss\n")
        for i, l in enumerate(losses):
            f.write(f"{i},{float(l):.6f}\n")

    # ---- generate ----
    tg = time.time()
    Xg_a = np.asarray(model.sample(a.n_gen))                    # (n_gen, T+1) in [0,1], incl. anchor
    Xg_n = Xg_a[:, 1:]                                          # strip anchor -> (n_gen, T)
    Xg = minmax_invert(Xg_n, lo, hi)                            # back to price scale
    gen_sec = time.time() - tg

    gdir = os.path.join(METHOD, "generated_paths", f"seed_{a.seed}")
    os.makedirs(gdir, exist_ok=True)
    np.save(os.path.join(gdir, "generated_paths_8192x128.npy"), Xg.astype(np.float64))
    meta = {
        "seed": a.seed, "method": "FourierFlow", "shape": list(Xg.shape),
        "seq_len": T, "n_gen": int(a.n_gen),
        "scale_min": lo, "scale_max": hi,
        "gen_min_val": float(Xg.min()), "gen_max_val": float(Xg.max()),
        "train_sec": round(train_sec, 1), "gen_sec": round(gen_sec, 1),
        "hidden": a.hidden, "n_flows": a.n_flows, "normalize": bool(a.normalize),
        "epochs": a.epochs, "learning_rate": a.lr,
    }
    with open(os.path.join(gdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[seed {a.seed}] done train={train_sec:.1f}s gen={gen_sec:.1f}s "
          f"loss {losses[0]:.2f}->{losses[-1]:.2f} gen[min={Xg.min():.2f} max={Xg.max():.2f}]",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fourier Flow — Heston training/generation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default=DEFAULT_DATA)
    # paper Stocks config (see README for justification)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--n-flows", type=int, default=3)
    p.add_argument("--normalize", type=int, default=1)   # 1=True (paper), 0=False
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=500)  # ignored by fit (full-batch)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--display-step", type=int, default=100)
    p.add_argument("--n-gen", type=int, default=8192)
    main(p.parse_args())
