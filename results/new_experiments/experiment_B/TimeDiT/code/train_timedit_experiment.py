"""TimeDiT on a blinded-protocol experiment (Experiment A or B).

Identical model, diffusion, optimiser, scaler and sampling path as
``methods/TimeDiT/code/train_heston.py`` -- the only differences are (a) the target dataset is
the protocol's ``train.npy`` instead of the original 8192x128 Heston bank and (b) artefacts land
under ``results/new_experiments/experiment_<X>/TimeDiT``.

``RECIPE`` and ``znorm_fit`` are IMPORTED from the canonical script rather than re-declared, and
the network and diffusion come from the same ``timedit_model.py`` / ``gaussian_diffusion.py``
modules. That is deliberate and load-bearing: it means the Stage-0 gate result -- bitwise
reproduction of the committed Heston run -- transfers to this file, because there is exactly one
definition of the hyperparameters and one definition of the model in the repository. A copied
hyperparameter block would silently decouple the moment either side was edited.

The locked recipe (paper-reproduction GATE winner, arXiv:2409.02322; **not** tuned on Heston,
``train.npy`` or ``disc.npy``): DiT-S hidden=384 depth=12 heads=6, learn_sigma=False, linear
schedule T=1000, ancestral ``ddpm_fixed`` sampler, Adam lr=3e-4 weight_decay=0, grad-clip 1.0,
NO EMA, batch=256, 15000 steps.

Normalisation chain, unchanged from the canonical script:
    S(price) --minmax--> [0,1] --znorm(mu,sd)--> model space
and inverted on the way out. **Every constant is fitted on this experiment's ``train.npy`` and
nothing else** -- ``lo``/``hi`` from its min/max, ``mu``/``sd`` from its mean/std. The fitted
values therefore differ between Experiment A and Experiment B, which §8 item 15 of the guideline
uses as the one piece of evidence that does not depend on the config file being honest.

**Information firewall.** This script reads ONLY ``train.npy``, enforced below by basename. It
never touches ``test.npy``, ``disc.npy``, ``*_sigma.npy``, ``*_labels.npy``, ``oracle_*`` or
``oracle.joblib``.

**Loss files.** TimeDiT trains by optimiser step with a constant learning rate and samples
minibatches with replacement, so it has no epochs. Inventing them would be a lie in a column
header, and dropping the native trace would break comparability with the Stage-0 gate, so both
files are written:

* ``seed_N_losses_steps.csv`` -- ``step,phase,loss_total``, one row per 100 steps. Byte-for-byte
  the schema the canonical script writes and the gate compared.
* ``seed_N_losses.csv`` -- ``epoch,avg_loss,min_loss,lr``, the layout contract
  (``check_method_layout.py`` requires first column ``epoch``, last ``lr``). One row per
  **100-step block**; ``epoch`` is the block index, not a pass over the data. Disclosed here and
  in the README rather than papered over.

**The bank is written RAW (un-anchored).** The declared PDF §1.3 repair
``S <- 100 * S / S[:, :1]`` is applied afterwards by the committed
``tools/apply_s0_repair.py --apply``, never inline, so exactly one implementation of the
transformation exists and ``--raw-dir`` can audit it against a pristine copy.

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_timedit_experiment.py --seed 0 --experiment A \
      --data ../../../../../dataset/Heston/new_experiments/experiment_A/train.npy
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))               # .../TimeDiT/code
OUT_DIR = os.path.dirname(CODE_DIR)                                 # .../TimeDiT
BENCH_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", "..", ".."))
CANONICAL = os.path.join(BENCH_ROOT, "methods", "TimeDiT", "code")
sys.path.insert(0, CANONICAL)

from train_heston import RECIPE, znorm_fit          # noqa: E402  (path set above)
from timedit_model import build_timedit             # noqa: E402
from gaussian_diffusion import GaussianDiffusion    # noqa: E402

LOG_EVERY = 100        # the canonical script's logging cadence; also the contract block size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True, help="protocol train.npy")
    ap.add_argument("--experiment", default="A", choices=("A", "B"))
    ap.add_argument("--steps", type=int, default=0, help="0 = locked recipe 15000")
    ap.add_argument("--batch", type=int, default=0, help="0 = locked recipe 256")
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--gen_batch", type=int, default=1000)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    steps = a.steps if a.steps > 0 else RECIPE["steps"]
    batch = a.batch if a.batch > 0 else RECIPE["batch"]

    data_path = os.path.abspath(a.data)
    if os.path.basename(data_path) != "train.npy":
        raise SystemExit(f"firewall: generators may only read train.npy, got {data_path}")

    # --- data: price -> minmax[0,1] -> znorm, all constants fitted on train.npy alone ---
    S = np.load(data_path).astype(np.float64)                  # (N, L) price
    N, L = S.shape
    C = 1
    lo, hi = float(S.min()), float(S.max())
    X01 = (S - lo) / (hi - lo)
    mu, sd = znorm_fit(X01)
    Xn = ((X01 - mu) / sd).astype(np.float32)[:, :, None]      # (N, L, 1) model space
    train = torch.tensor(Xn, device=device)

    print(f"=== TimeDiT experiment_{a.experiment}  seed={a.seed}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] {data_path}", flush=True)
    print(f"[data] S{S.shape} price[min={lo:.4f},max={hi:.4f}]  "
          f"znorm mu={float(mu.ravel()[0]):.6f} sd={float(sd.ravel()[0]):.6f}  "
          f"steps={steps} batch={batch}", flush=True)

    # --- model + diffusion (verbatim canonical modules, locked recipe) ---
    model = build_timedit(L, C, model_size=RECIPE["model_size"],
                          learn_sigma=RECIPE["learn_sigma"]).to(device)
    diff = GaussianDiffusion(T=RECIPE["T"], schedule=RECIPE["schedule"], loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"],
                           weight_decay=RECIPE["weight_decay"])
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparam} size={RECIPE['model_size']} T={RECIPE['T']} "
          f"schedule={RECIPE['schedule']} sampler={RECIPE['sampler']} lr={RECIPE['lr']} "
          f"ema=0", flush=True)

    weights_dir = os.path.join(OUT_DIR, "weights")
    losses_dir = os.path.join(OUT_DIR, "losses")
    gen_dir = os.path.join(OUT_DIR, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- training (canonical loop: sample-with-replacement minibatches, constant lr) ---
    every_step = []       # (step, loss) for ALL steps -- feeds the contract's block statistics
    records = []          # (step, phase, loss) every LOG_EVERY -- the canonical artefact
    first_nan_step = None
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        x0 = train[idx]
        loss, _parts = diff.training_loss(model, x0, learn_sigma=RECIPE["learn_sigma"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
        opt.step()
        lv = float(loss.item())
        every_step.append((step, lv))
        if not np.isfinite(lv) and first_nan_step is None:
            first_nan_step = step
        if step % LOG_EVERY == 0 or step == steps - 1:
            records.append((step, "diffusion", lv))
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            print(f"  step {step:6d}/{steps}  loss={lv:.5f}", flush=True)
    train_time = time.time() - t0

    # --- loss curves: native trace + layout contract ---
    with open(os.path.join(losses_dir, f"seed_{a.seed}_losses_steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "phase", "loss_total"])
        w.writerows(records)

    blocks = []
    for b0 in range(0, steps, LOG_EVERY):
        chunk = [l for s, l in every_step[b0:b0 + LOG_EVERY]]
        blocks.append({"epoch": b0 // LOG_EVERY, "avg_loss": float(np.mean(chunk)),
                       "min_loss": float(np.min(chunk)), "lr": float(RECIPE["lr"])})
    with open(os.path.join(losses_dir, f"seed_{a.seed}_losses.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "avg_loss", "min_loss", "lr"])
        w.writeheader()
        w.writerows(blocks)

    first_nan_epoch = next((b["epoch"] for b in blocks if not np.isfinite(b["avg_loss"])), None)
    min_loss = min((l for _, _, l in records), default=float("nan"))
    min_total_loss = float(min(b["avg_loss"] for b in blocks))

    # --- generation (ancestral ddpm_fixed, batched -- canonical path) ---
    model.eval()
    g0 = time.time()
    fakes, remaining = [], a.gen_num
    with torch.no_grad():
        while remaining > 0:
            b = min(a.gen_batch, remaining)
            x = diff.p_sample_loop(model, (b, L, C), device,
                                   learn_sigma=RECIPE["learn_sigma"], sample_var="fixed")
            fakes.append(x.cpu().numpy())
            remaining -= b
    fz = np.concatenate(fakes, 0)[:a.gen_num]
    gen_time = time.time() - g0

    # --- denorm: znorm^-1 -> [0,1] -> price ---
    X01_g = np.clip(fz[:, :, 0] * sd + mu, 0.0, 1.0)
    Xg = (X01_g * (hi - lo) + lo)
    Xg = np.clip(Xg, 1e-6, None).astype(np.float64)
    gen_has_nan = bool(not np.isfinite(Xg).all())

    raw_dir = os.path.join(OUT_DIR, "raw_banks", "generated_paths", f"seed_{a.seed}")
    os.makedirs(raw_dir, exist_ok=True)
    np.save(os.path.join(gen_dir, "generated_paths_8192x128.npy"), Xg)
    np.save(os.path.join(raw_dir, "generated_paths_8192x128.npy"), Xg)
    s0_dev = float(np.abs(Xg[:, 0] - 100.0).max())

    torch.save({"model": model.state_dict(), "seed": a.seed,
                "minmax": [lo, hi], "znorm": [float(mu.ravel()[0]), float(sd.ravel()[0])]},
               os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
    out_cfg = {"method": "TimeDiT", "variant": "TimeDiT DiT-S (paper reimplementation)",
               "experiment": a.experiment, "seed": a.seed, "data": data_path,
               "model_size": RECIPE["model_size"], "hidden_size": 384, "depth": 12,
               "num_heads": 6, "learn_sigma": RECIPE["learn_sigma"],
               "schedule": RECIPE["schedule"], "T": RECIPE["T"], "sampler": RECIPE["sampler"],
               "lr": RECIPE["lr"], "weight_decay": RECIPE["weight_decay"], "ema": 0.0,
               "grad_clip": RECIPE["grad_clip"], "batch_size": batch, "n_steps": steps,
               "n_train": N, "seq_len": L, "feature_size": C,
               "scaler": "minmax_then_znorm", "minmax": [lo, hi],
               "scaler_mu": float(mu.ravel()[0]), "scaler_sigma": float(sd.ravel()[0]),
               "paper_hyperparams": True, "params": int(nparam)}
    with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    meta = {"method": "TimeDiT", "experiment": a.experiment, "seed": a.seed,
            "shape": list(Xg.shape), "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "scale_min": lo, "scale_max": hi,
            "znorm_mean": float(mu.ravel()[0]), "znorm_std": float(sd.ravel()[0]),
            "params": int(nparam), "steps": steps, "batch_size": batch,
            "epochs": len(blocks), "epochs_run": len(blocks), "epochs_max": len(blocks),
            "min_total_loss": min_total_loss, "first_nan_epoch": first_nan_epoch,
            "min_loss": float(min_loss), "first_nan_step": first_nan_step,
            "s0_max_deviation_raw": s0_dev,
            "gen_has_nan": gen_has_nan}
    with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={a.seed} steps={steps} min_loss={min_loss:.5f} "
          f"gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"S0_dev_raw={s0_dev:.3e} "
          f"train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
