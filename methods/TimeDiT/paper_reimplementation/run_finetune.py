"""TimeDiT finetuning + synthetic-generation eval (paper §5.6, faithful reproduction).

Loads the SSL-pretrained DiT-S checkpoint (run_pretrain.py, K_MAX=40 channels) and
finetunes it on a single synthetic-generation target (Sine or Stocks) with the
reconstruction mask M^Rec = 0 -> unconditional whole-window generation.  To match
the pretraining distribution, the target is z-normalised per channel (Standardize
Pipeline); generated samples are mapped back to [0,1] for the frozen Yoon metrics so
numbers are directly comparable to run_paper.py and to the paper's Table 6.

--init pretrain : load pretrain_dit_s.pt then finetune          (the paper's recipe)
--init scratch  : random init, same everything else             (controlled baseline)

Usage
-----
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 python run_finetune.py --dataset sine \
    --ckpt pretrain_dit_s.pt --steps 15000 --out results_sine_finetune.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from data_pretrain import K_MAX
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit
from yoon_metrics import discriminative_score, predictive_score


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def parse_args():
    p = argparse.ArgumentParser(description="TimeDiT finetune + SG eval")
    p.add_argument("--dataset", choices=["sine", "stock"], default="sine")
    p.add_argument("--stock_csv", default=None)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--model_size", choices=["S", "B", "L"], default="S")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--ckpt", default="pretrain_dit_s.pt")
    p.add_argument("--init", choices=["pretrain", "scratch"], default="pretrain")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--n_gen", type=int, default=0)
    p.add_argument("--eval_seeds", type=int, default=3)
    p.add_argument("--sample_var", choices=["learned", "fixed", "both"], default="both")
    p.add_argument("--out", default="results_finetune.json")
    p.add_argument("--save_model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- data in [0,1], z-normalise per channel (pretrain distribution) ----
    data01 = load_dataset(args.dataset, seq_len=args.seq_len,
                          stock_csv=args.stock_csv, seed=args.seed)
    steps = args.steps
    if args.smoke:
        data01 = data01[:max(int(len(data01) * 0.05), args.batch)]
        steps = 300
    N, L, C = data01.shape
    cut = int(N * 0.8)
    mu = data01[:cut].mean(axis=(0, 1), keepdims=True)
    sd = data01[:cut].std(axis=(0, 1), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    dataz = ((data01 - mu) / sd).astype(np.float32)          # (N,L,C) z-normed
    print(f"[data] {args.dataset}: {data01.shape} C={C} znorm mu={mu.ravel()} "
          f"sd={sd.ravel()} steps={steps} init={args.init}")

    # pad channels to K_MAX
    trainz_np = np.concatenate(
        [dataz[:cut], np.zeros((cut, L, K_MAX - C), dtype=np.float32)], axis=-1)
    trainz = torch.tensor(trainz_np, device=device)
    valid = torch.zeros(1, 1, K_MAX, device=device); valid[..., :C] = 1.0

    # ---- model (K_MAX channels) + optional pretrained init ----
    model = build_timedit(L, K_MAX, model_size=args.model_size, learn_sigma=True).to(device)
    if args.init == "pretrain":
        sd_ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd_ckpt, strict=False)
        print(f"[init] loaded {args.ckpt} (missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print("[init] scratch (no checkpoint)")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = GaussianDiffusion(T=args.T, schedule=args.schedule, loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] DiT-{args.model_size} params={n_params/1e6:.2f}M")

    # ---- finetune (M^Rec = 0 -> unconditional) ----
    zero_cond = torch.zeros(args.batch, L, K_MAX, device=device)
    valid_b = valid.expand(args.batch, 1, K_MAX)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = np.random.randint(0, len(trainz), args.batch)
        x0 = trainz[idx]
        loss, parts = diff.masked_training_loss(model, x0, zero_cond, valid_b, learn_sigma=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = 0.0 if step < 100 else args.ema_decay
        ema_update(ema, model, decay)
        if not np.isfinite(loss.item()):
            raise RuntimeError(f"NaN/Inf at step {step}: {parts}")
        if step % 200 == 0 or step == steps - 1:
            print(f"[{step:6d}] loss={loss.item():.4f} simple={parts['simple']:.4f} "
                  f"vlb={parts['vlb']:.4f} ({time.time()-t0:.1f}s)")
    train_time = time.time() - t0

    # ---- sample + score per variance mode ----
    n_gen = args.n_gen if args.n_gen > 0 else N
    var_modes = ["learned", "fixed"] if args.sample_var == "both" else [args.sample_var]
    ema.eval()

    def sample(vm):
        fakes = []
        with torch.no_grad():
            remaining = n_gen
            while remaining > 0:
                b = min(1024, remaining)
                x = diff.p_sample_loop(ema, (b, L, K_MAX), device,
                                       obs=torch.zeros(b, L, K_MAX, device=device),
                                       mask=torch.zeros(b, L, K_MAX, device=device),
                                       learn_sigma=True, sample_var=vm)
                fakes.append(x[..., :C].cpu().numpy())
                remaining -= b
        fz = np.concatenate(fakes, 0)[:n_gen]
        f01 = fz * sd + mu                                  # denorm to [0,1] scale
        return np.clip(f01, 0.0, 1.0).astype(np.float32)

    by_var = {}
    saved_fake = None
    for vm in var_modes:
        fake = sample(vm)
        print(f"[gen:{vm}] fake {fake.shape} range=[{fake.min():.3f},{fake.max():.3f}]")
        disc, pred = [], []
        for s in range(args.eval_seeds):
            disc.append(discriminative_score(data01, fake, device=device, seed=s))
            pred.append(predictive_score(data01, fake, device=device, seed=s))
            print(f"[eval:{vm} seed {s}] disc={disc[-1]:.4f} pred={pred[-1]:.4f}")
        by_var[vm] = {"discriminative_mean": float(np.mean(disc)),
                      "discriminative_std": float(np.std(disc)),
                      "predictive_mean": float(np.mean(pred)),
                      "predictive_std": float(np.std(pred)),
                      "discriminative_per_seed": [float(x) for x in disc],
                      "predictive_per_seed": [float(x) for x in pred]}
        if saved_fake is None:
            saved_fake = fake

    primary = "learned" if "learned" in by_var else var_modes[0]
    pv = by_var[primary]
    result = {"dataset": args.dataset, "variant": "pretrain_finetune", "init": args.init,
              "seq_len": L, "channels": C, "k_max": K_MAX, "n_train": int(cut),
              "n_gen": int(n_gen), "steps": steps, "smoke": args.smoke,
              "model_size": args.model_size, "schedule": args.schedule,
              "ckpt": args.ckpt if args.init == "pretrain" else None,
              "params_M": n_params / 1e6, "train_time_sec": train_time,
              "primary_sample_var": primary, "by_sample_var": by_var,
              "discriminative_mean": pv["discriminative_mean"],
              "discriminative_std": pv["discriminative_std"],
              "predictive_mean": pv["predictive_mean"],
              "predictive_std": pv["predictive_std"],
              "discriminative_per_seed": pv["discriminative_per_seed"],
              "predictive_per_seed": pv["predictive_per_seed"]}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_model:
        torch.save(ema.state_dict(), args.save_model)
        np.save(args.save_model.rsplit(".", 1)[0] + "_fake.npy", saved_fake)

    print(f"\n[done] {args.dataset} ({args.init}) -> {args.out}")
    for vm in var_modes:
        b = by_var[vm]
        print(f"   [{vm:7s}] disc={b['discriminative_mean']:.4f}±{b['discriminative_std']:.4f} "
              f"pred={b['predictive_mean']:.4f}±{b['predictive_std']:.4f}")


if __name__ == "__main__":
    main()
