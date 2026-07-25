#!/usr/bin/env python
"""Fine-tune TimesFM-1.0-200m on Heston prices for the forecaster-reference task.

Role in the benchmark
---------------------
TimesFM is used here as a *forecaster reference* (like Chronos-2), NOT an
unconditional generator.  The eval task is: given the 64-step real prefix of a
held-out Heston test path, forecast the next 64 steps.  This script produces the
`finetune` row of that reference — 5 seeds, full fine-tune of the released
timesfm-1.0-200m checkpoint on the Heston TRAIN split.

Training recipe (mirrors the official TimesFM finetuner, code/reference/
v1_finetuning/finetuning_torch.py::_process_batch)
--------------------------------------------------
  * Input  : x_context = path[:64]   (2 input patches of length 32)
  * Target : x_future  = path[64:128] (64 steps)
  * Forward: preds = model(x_context, x_padding.float(), freq)
             mean  = preds[..., 0]            # (B, n_patches, output_patch_len)
             last  = mean[:, -1, :64]         # last input patch predicts steps 64..
  * Loss   : MSE(mean_head) + sum_q pinball(quantile_head_q)  -- ALL masked to the
             first 64 output steps because TimesFM's output_patch_len is 128 but a
             Heston path only has 64 future steps.  The pinball term over the 9
             quantile heads (levels 0.1..0.9, exactly what create_quantiles()
             returns) is ESSENTIAL: the forecaster eval builds its K=77 ensemble
             from the quantile heads, so supervising only the mean head (as the
             finetuner does by default) decalibrates the quantiles and makes the
             fine-tuned CRPS far worse than zero-shot.  This mirrors the official
             example's use_quantile_loss=True setting.
  * Optim  : Adam lr=1e-4, weight_decay=0.01  (finetuner defaults)
  * full fine-tune (all decoder parameters), ft_steps=1000, ft_batch=256.

Prices are fed in raw scale (TimesFM applies reversible instance-norm on the
context internally), matching how the eval `forecast()` is called.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/timesfm-v1-venv/bin/python finetune_heston.py --seed 0
  # strategic smoke test on 5% of the data:
  ... finetune_heston.py --seed 0 --frac 0.05 --steps 100 --smoke
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

PS_DIR = os.path.dirname(os.path.abspath(__file__))          # methods/TimesFM/path_shadowing
METHOD_DIR = os.path.dirname(PS_DIR)                          # methods/TimesFM
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))     # benchmark/
WEIGHTS_DIR = os.path.join(METHOD_DIR, "weights")
TRAIN_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_8192x128.npy")

MODEL_ID = "google/timesfm-1.0-200m-pytorch"
PREFIX_LEN = 64
HORIZON = 64
CONTEXT_LEN = 64          # 2 input patches of length 32
NUM_LAYERS = 20           # 1.0-200m
USE_POS_EMB = True        # 1.0-200m
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # create_quantiles()


def pinball(pred, actual, q):
    """Official TimesFM pinball loss (finetuning_torch._quantile_loss)."""
    dev = actual - pred
    return 2.0 * torch.where(dev * q >= 0, dev * q, -dev * (1.0 - q))


def build_model(device):
    """Load released timesfm-1.0-200m and return its trainable torch decoder."""
    import timesfm
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if device == "cuda" else "cpu",
            per_core_batch_size=32,
            horizon_len=128,           # native output patch length
            num_layers=NUM_LAYERS,
            context_len=512,           # wrapper cap; we feed 64 directly
            use_positional_embedding=USE_POS_EMB,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=MODEL_ID),
    )
    model = tfm._model.to(device)      # PatchedTimeSeriesDecoder, weights loaded
    return model


def make_samples(series, context_len, horizon_len):
    """series (N, T) -> (context (N, context_len), future (N, horizon_len))."""
    ctx = series[:, :context_len]
    fut = series[:, context_len:context_len + horizon_len]
    return ctx.astype(np.float32), fut.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of train paths")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--no-quantile-loss", dest="quantile_loss",
                    action="store_false",
                    help="disable pinball loss on the quantile heads (NOT recommended)")
    ap.set_defaults(quantile_loss=True)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke mode: don't save weights, just report loss/NaN")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X = np.load(TRAIN_DATA).astype(np.float64)                # (8192, 128)
    if args.frac < 1.0:
        n = max(args.batch, int(len(X) * args.frac))
        rng = np.random.default_rng(args.seed)
        X = X[rng.choice(len(X), n, replace=False)]
    ctx_np, fut_np = make_samples(X, CONTEXT_LEN, HORIZON)
    N = ctx_np.shape[0]
    print(f"=== TimesFM-1.0-200m Heston fine-tune  seed={args.seed}  device={dev_name} ===",
          flush=True)
    print(f"[data] train paths={N}  ctx={CONTEXT_LEN}  horizon={HORIZON}  "
          f"frac={args.frac}  steps={args.steps}  batch={args.batch}", flush=True)

    ctx = torch.tensor(ctx_np, device=device)
    fut = torch.tensor(fut_np, device=device)
    pad = torch.zeros_like(ctx)                               # (N, ctx) no padding
    freq = torch.zeros((args.batch, 1), dtype=torch.long, device=device)

    model = build_model(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    rng = np.random.default_rng(args.seed)
    loss_log = []
    nan_hit = False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        bidx = rng.integers(0, N, size=args.batch)
        xb = ctx[bidx]                                        # (B, ctx)
        yb = fut[bidx]                                        # (B, horizon)
        pb = pad[bidx]
        preds = model(xb, pb.float(), freq)                  # (B, n_patch, opl, 1+Q)
        mean = preds[..., 0]                                 # (B, n_patch, opl)
        last = mean[:, -1, :HORIZON]                         # (B, horizon)
        loss = torch.mean((last - yb) ** 2)
        if args.quantile_loss:
            for qi, q in enumerate(QUANTILES):
                qp = preds[:, -1, :HORIZON, qi + 1]          # (B, horizon) quantile head
                loss = loss + torch.mean(pinball(qp, yb, q))

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu())
        loss_log.append((step, lv))
        if not np.isfinite(lv):
            nan_hit = True
            print(f"  [step {step}] LOSS NON-FINITE ({lv}) -- stopping", flush=True)
            break
        if step % args.log_every == 0 or step == 1:
            print(f"  [step {step:4d}/{args.steps}] mse={lv:.6f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    dt = time.time() - t0
    first = loss_log[0][1] if loss_log else float("nan")
    last_l = loss_log[-1][1] if loss_log else float("nan")
    print(f"[done] seed={args.seed}  first_mse={first:.6f}  last_mse={last_l:.6f}  "
          f"decreased={last_l < first}  nan={nan_hit}  ({dt:.1f}s)", flush=True)

    if args.smoke:
        print("[smoke] weights NOT saved.", flush=True)
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    ckpt = os.path.join(WEIGHTS_DIR, f"seed_{args.seed}_model.pt")
    torch.save(model.state_dict(), ckpt)
    cfg = {
        "method": "TimesFM", "variant": "finetune", "seed": args.seed,
        "model_id": MODEL_ID, "num_layers": NUM_LAYERS,
        "use_positional_embedding": USE_POS_EMB,
        "feat_dim": 1, "seq_len": 128, "context_len": CONTEXT_LEN,
        "horizon": HORIZON, "ft_pred_len": HORIZON, "ft_steps": args.steps,
        "ft_lr": args.lr, "ft_batch": args.batch, "ft_weight_decay": args.wd,
        "finetune_mode": "full", "use_quantile_loss": args.quantile_loss,
        "quantiles": QUANTILES,
        "loss": "masked_mse_mean + masked_pinball_9q (first64)",
        "n_train": N, "first_mse": first, "last_mse": last_l,
        "train_time_sec": round(dt, 1),
    }
    with open(os.path.join(WEIGHTS_DIR, f"seed_{args.seed}_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(WEIGHTS_DIR, f"seed_{args.seed}_losses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mse"])
        w.writerows(loss_log)
    print(f"[saved] {ckpt}", flush=True)
    print(f"[saved] seed_{args.seed}_config.json + seed_{args.seed}_losses.csv", flush=True)


if __name__ == "__main__":
    main()
