"""
End-to-end reproduction of arXiv:2604.07159 section 5.1 (Heston / Figure 2).

Pipeline, mirroring the authors' ``run_heston.py`` step for step:

  1. Simulate M=5000 heterogeneous Heston paths, N=252, d=2 (price, variance).
  2. log_returns[:, 0] = 0; log_returns[:, 1:] = diff(log(X), axis=1).
     Truncate the level array to X[:, 1:] so real and generated align.
  3. scale = std(log_returns over (path, time)) / sqrt(T); divide through.
  4. Train ScoreNN with Algorithm 1 (large-beta branch, beta = 100, K = 5).
  5. Generate M_simu paths, rescale, exp(cumsum) back to levels.
     Both channels are levels because the paper sets S0 = v0 = 1.
  6. Per-path Heston MLE on real and generated; compare the five estimated
     parameter distributions (the paper's Figure 2).

Run (single GPU, 8 cores, per benchmark GUIDELINE section 4.1):

    cd methods/SBBTS/paper_reimplementation/metric
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      taskset -c 0-7 /home/tbasseras/gpu-venv/bin/python reproduce_heston.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # paper_reimplementation/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT.parent / "code"))

from generate_paper_heston import generate as generate_paper_heston  # noqa: E402
from heston_mle import get_params_estimation, plot_figure2, summarize  # noqa: E402
from sbbts_torch import ScoreNN, generate_dsbm, training_sbbts_dsbm  # noqa: E402

# Canonical hyperparameters: authors' run_heston.py (Alexandre Alouadi, 2026-09-01).
# Divergences from the paper's Table 2 are recorded in ../README.md section 2.
CFG = dict(
    M=5000, N=252, T=1, beta=100, K=5, safe_t=1e-2,
    batch_size=128, n_epochs=1000, lr=1e-3, patience=15, delta=1e-3,
    d_model=128, hidden_dim=64, nhead=32, n_layers=2,
    N_pi=60, M_simu=4000, N_batch=2,
)


def build_training_tensor(X_levels, device):
    """Levels (M, N+1, 2) -> scaled log-returns (M, N+1, 2), plus scale and aligned levels."""
    log_returns = np.zeros_like(X_levels)
    log_returns[:, 1:] = np.diff(np.log(X_levels), axis=1)
    levels_aligned = X_levels[:, 1:]                      # (M, N, 2)

    X = torch.tensor(log_returns, dtype=torch.float32, device=device)
    scale = X.std(dim=(0, 1)) / np.sqrt(CFG["T"])         # (2,)
    X = X / scale
    return X, scale, levels_aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "results"))
    ap.add_argument("--max-epochs", type=int, default=CFG["n_epochs"],
                    help="override n_epochs (smoke tests only)")
    ap.add_argument("--K", type=int, default=CFG["K"])
    ap.add_argument("--M", type=int, default=CFG["M"])
    ap.add_argument("--M-simu", type=int, default=CFG["M_simu"])
    ap.add_argument("--patience", type=int, default=CFG["patience"],
                    help="early-stopping patience; the default is run_heston.py's 15")
    ap.add_argument("--delta", type=float, default=CFG["delta"],
                    help="early-stopping improvement threshold; default run_heston.py's 1e-3")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a CFG entry that has no dedicated flag, e.g. "
                         "--set beta=200 --set N_pi=120 --set safe_t=1e-3")
    ap.add_argument("--mle-jobs", type=int, default=8)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    # Applied after parsing, so it can only reach keys read from CFG at call time.
    # The keys below are captured into `args` when the parser is built, which is
    # before any --set runs; overriding them here would silently do nothing.
    reserved = {"K", "M", "M_simu", "patience", "delta", "n_epochs"}
    for kv in args.set:
        key, sep, raw = kv.partition("=")
        if not sep or key not in CFG:
            sys.exit(f"--set {kv}: unknown key; valid keys are "
                     f"{sorted(set(CFG) - reserved)}")
        if key in reserved:
            sys.exit(f"--set {kv}: use the dedicated flag instead "
                     f"(--{key.lower().replace('_', '-')} / --max-epochs for n_epochs)")
        CFG[key] = type(CFG[key])(raw)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device={device}  seed={args.seed}", flush=True)

    # -- 1. Dataset --
    ds_dir = ROOT / "dataset"
    x_path = ds_dir / f"X_heston_paper_{args.M}x{CFG['N'] + 1}x2.npy"
    if x_path.exists():
        print(f"[1/6] loading {x_path.name}", flush=True)
        X_levels = np.load(x_path)
    else:
        print(f"[1/6] simulating {args.M} Heston paths (N={CFG['N']}, d=2)", flush=True)
        t0 = time.perf_counter()
        X_levels, params_true = generate_paper_heston(args.M, CFG["N"], args.seed)
        print(f"      done in {time.perf_counter() - t0:.1f}s", flush=True)
        np.save(x_path, X_levels)
        np.save(ds_dir / f"params_true_{args.M}x5.npy", params_true)

    print(f"      X_levels {X_levels.shape}  "
          f"price[{X_levels[:, :, 0].min():.3g}, {X_levels[:, :, 0].max():.3g}]  "
          f"var[{X_levels[:, :, 1].min():.3g}, {X_levels[:, :, 1].max():.3g}]", flush=True)

    # -- 2-3. Transform --
    X, scale, levels_aligned = build_training_tensor(X_levels, device)
    d = X.shape[-1]
    print(f"[2/6] scaled log-returns {tuple(X.shape)}  scale={scale.tolist()}", flush=True)

    # -- 4. Train --
    model = ScoreNN(d, CFG["d_model"], CFG["hidden_dim"], CFG["nhead"],
                    CFG["n_layers"], CFG["N"], device=device).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[3/6] ScoreNN d={d} params={n_par:,}", flush=True)

    t0 = time.perf_counter()
    model, y_0, history = training_sbbts_dsbm(
        X, model, CFG["T"], CFG["beta"], args.K,
        n_epochs=args.max_epochs, batch_size=CFG["batch_size"],
        patience=args.patience, delta=args.delta,
        safe_t=CFG["safe_t"], lr=CFG["lr"],
    )
    train_sec = time.perf_counter() - t0
    print(f"[4/6] training done in {train_sec / 60:.1f} min", flush=True)

    torch.save({"state_dict": model.state_dict(), "y_0": y_0.cpu(),
                "scale": scale.cpu(), "cfg": CFG}, out / f"model{tag}.pt")
    with open(out / f"losses{tag}.csv", "w") as f:
        f.write("k,epoch,train_loss,val_loss\n")
        for h in history:
            f.write(f"{h['k']},{h['epoch']},{h['train_loss']:.8f},{h['val_loss']:.8f}\n")

    # -- 5. Generate --
    model.eval()
    t0 = time.perf_counter()
    X_sbb = generate_dsbm(CFG["N"], X, model, y_0, N_pi=CFG["N_pi"], T=CFG["T"],
                          beta=CFG["beta"], M_simu=args.M_simu, N_batch=CFG["N_batch"],
                          scale=scale, exp=True, safe_t=CFG["safe_t"])
    gen_sec = time.perf_counter() - t0
    print(f"[5/6] generated {X_sbb.shape} in {gen_sec / 60:.1f} min  "
          f"price[{X_sbb[:, :, 0].min():.3g}, {X_sbb[:, :, 0].max():.3g}]  "
          f"var[{X_sbb[:, :, 1].min():.3g}, {X_sbb[:, :, 1].max():.3g}]", flush=True)
    np.save(out / f"X_sbbts_{X_sbb.shape[0]}x{X_sbb.shape[1]}x2{tag}.npy", X_sbb)

    # The MLE needs strictly positive S and v (it takes log S and divides by v).
    finite = np.isfinite(X_sbb).all(axis=(1, 2))
    positive = (X_sbb > 0).all(axis=(1, 2))
    valid = finite & positive
    n_dropped = int((~valid).sum())
    X_sbb_mle = X_sbb[valid].astype(np.float64)
    print(f"      MLE-eligible paths: {len(X_sbb_mle)}/{len(X_sbb)} "
          f"({n_dropped} dropped: non-finite or non-positive)", flush=True)

    # -- 6. Paper metric --
    print(f"[6/6] per-path Heston MLE ({args.mle_jobs} workers)", flush=True)
    t0 = time.perf_counter()
    # The data-side fit depends only on the cached dataset, never on the model,
    # so it is shared across every run at this M.
    data_cache = ds_dir / f"params_data_{args.M}x5.npy"
    if data_cache.exists():
        params_data = np.load(data_cache)
        print(f"      data MLE: cache hit {data_cache.name}", flush=True)
    else:
        params_data = get_params_estimation(np.ascontiguousarray(levels_aligned),
                                            dt=1 / 252, n_jobs=args.mle_jobs)
        np.save(data_cache, params_data)
    params_sbbts = get_params_estimation(np.ascontiguousarray(X_sbb_mle),
                                         dt=1 / 252, n_jobs=args.mle_jobs)
    mle_sec = time.perf_counter() - t0
    np.save(out / f"params_data{tag}.npy", params_data)
    np.save(out / f"params_sbbts{tag}.npy", params_sbbts)

    rows = summarize(params_data, params_sbbts)
    fig = plot_figure2(params_data, params_sbbts, out / f"figure2_params_kde{tag}.png")

    print(f"\n{'param':>6} | {'data mean+-std':>22} | {'SBBTS mean+-std':>22} | "
          f"{'std ratio':>9} | {'W1':>8}")
    print("-" * 82)
    for r in rows:
        print(f"{r['param']:>6} | {r['data_mean']:>10.4f} +- {r['data_std']:<8.4f} | "
              f"{r['gen_mean']:>10.4f} +- {r['gen_std']:<8.4f} | "
              f"{r['std_ratio']:>9.3f} | {r['w1']:>8.4f}")

    scores = dict(
        config={**CFG, "K": args.K, "n_epochs": args.max_epochs,
                "patience": args.patience, "delta": args.delta,
                "M": args.M, "M_simu": args.M_simu, "seed": args.seed,
                "n_params": n_par, "scale": scale.tolist()},
        timing=dict(train_min=round(train_sec / 60, 2), gen_min=round(gen_sec / 60, 2),
                    mle_min=round(mle_sec / 60, 2)),
        generated=dict(shape=list(X_sbb.shape), n_dropped_for_mle=n_dropped,
                       n_mle_paths=int(len(X_sbb_mle))),
        summary=rows,
        paper_reference=(
            "Figure 2 is qualitative. Claims: (a) SBBTS spans the full real range of all "
            "five parameters; (b) SBTS collapses xi and rho onto a narrow spike at the "
            "centre of the range while still recovering kappa, theta, r. The falsifiable "
            "statistic here is std_ratio ~ 1 for xi and rho."
        ),
        figure=str(Path(fig).name),
    )
    (out / f"sbbts_heston_scores{tag}.json").write_text(json.dumps(scores, indent=2))
    print(f"\nwrote {out / f'sbbts_heston_scores{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
