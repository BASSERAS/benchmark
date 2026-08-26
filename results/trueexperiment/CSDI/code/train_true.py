#!/usr/bin/env python3
"""CSDI on TrueDataset (real crypto, d = 8) -- train one seed and write its A/B bank.

Port of `results/HestonMultiAsset/CSDI/code/train_multiasset.py` to the real-data
build. The model, the diffusion process and every hyperparameter are unchanged;
see `csdi_true.py` for what is the authors' and what is ours.

WHAT DIFFERS FROM THE HESTON d = 8 SIBLING, AND WHY
----------------------------------------------------
Every difference below is forced by the dataset or by truedatasetguideline.md.
None of them is a change to the method.

1. `--data-dir` and `--seq-tag` are REQUIRED, not constants. Guideline section 13.1:
   the splits live under `dataset/TrueDataset/variants/<build>/`, not in a fixed
   directory, and the filename tag is `6144x128x8` for the locked build. A
   hardcoded path or tag resolves silently against the wrong build and nothing
   errors.
2. `--results-dir` is separate from the code directory. Guideline section 5:
   artefacts publish under `results/trueexperiment/<Method>/`, which the
   `results/<dataset>/<method>` convention cannot name.
3. T = 128 and N = 6144 are read from the array rather than asserted against a
   constant, so a build with a different tag fails loudly at the contract check
   instead of quietly training on the wrong thing.
4. **6 144 paths, not a round number.** Guideline section 4: every real-vs-real
   threshold this bank is judged against was measured between real splits of
   exactly that size, and the volatility estimator's sampling noise falls like
   1/sqrt(m). A bank of a different size compares a differently-noisy estimate
   against those thresholds. The ONE exception is the conditional-CRPS pool at
   8 192, which is produced by `generate_bank_true.py`, not by this script.
5. `metadata.json` carries `data_dir` and `seq_tag` (guideline section 4). It is the
   only tracked audit record for a 50 MB array that is gitignored.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not touch `test` or `disc`. `val` is read for the validation loss curve
only, never for early stopping or model selection -- there is no early stopping
here at all, the run is a fixed 200 epochs. `val` is the same era as `train`
(guideline section 3.3) so reading it is permitted by section 2 question 5; `test`
and `disc` are the future era and touching either voids the run.

It writes nothing to a shared path. Each invocation touches only
`generated_paths/seed_{i}/`, `weights/seed_{i}_*` and `losses/seed_{i}_losses.csv`,
so concurrent seeds cannot race. `losses/generation_time.csv` is rebuilt
afterwards by `collect_artifacts.py`.

Usage
-----
  V=dataset/TrueDataset/variants/om_2022-07_N6144
  CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python train_true.py \
      --seed 0 --data-dir $V --seq-tag 6144x128x8

  # smoke test before committing hours of compute (guideline section 13.2):
  ... --seed 0 --epochs 2 --frac 0.05 --gen-num 64 --tag probe
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from csdi_true import (BASE_CONFIG, GPU_NAME, S0, CSDI_TrueData, load_split,
                       rescale_to_s0, zscore_stats)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dataset/TrueDataset/variants/<build>")
    ap.add_argument("--seq-tag", required=True, help="e.g. 6144x128x8")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="0 = base.yaml default 200")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = base.yaml default 16")
    ap.add_argument("--gen-num", type=int, default=0,
                    help="0 = size of the train split (6144). Guideline section 4 "
                         "forbids a round number here.")
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of training paths -- SMOKE TESTS ONLY, never a run")
    ap.add_argument("--val-n", type=int, default=256,
                    help="validation paths per pass. CSDI's calc_loss_valid averages "
                         "the loss over ALL 50 diffusion steps, so a full 6144-path "
                         "pass costs ~50x a training epoch and would dominate the run. "
                         "256 paths on a 20-point cadence keeps the overhead near 2%% "
                         "and still draws a usable curve.")
    ap.add_argument("--tag", default="",
                    help="run tag; prefixes losses/metadata and SKIPS canonical weights, "
                         "so a probe can never overwrite a real seed")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(json.dumps(BASE_CONFIG))
    epochs = a.epochs if a.epochs > 0 else cfg["train"]["epochs"]
    batch_size = a.batch_size if a.batch_size > 0 else cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    tagp = (a.tag + "_") if a.tag else ""

    # --- data ---------------------------------------------------------------
    S_full = load_split(a.data_dir, a.seq_tag, "")     # train -- the only fittable split
    Sv = load_split(a.data_dir, a.seq_tag, "val")      # same era, curve only
    n_train_full, seq_len, D = S_full.shape
    S = S_full[: int(round(n_train_full * a.frac))] if a.frac < 1.0 else S_full
    Sv = Sv[: a.val_n]

    # z-score fitted on train only, and on the FULL train split even under --frac,
    # so a probe and a real run share the same input scale.
    mean, std = zscore_stats(S_full)
    Xt = ((S - mean) / std).astype(np.float32)
    Xv = ((Sv - mean) / std).astype(np.float32)

    dl = DataLoader(TensorDataset(torch.from_numpy(Xt)), batch_size=batch_size,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # drop_last=False on validation: with drop_last=True a --val-n smaller than
    # --batch-size yields an empty loader and a silent nan curve instead of an error.
    dl_val = DataLoader(TensorDataset(torch.from_numpy(Xv)), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = CSDI_TrueData(cfg, device, target_dim=D).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[CSDI TrueDataset seed {a.seed}] params={nparam} epochs={epochs} "
          f"batch={batch_size} N={S.shape[0]} T={seq_len} K={D} "
          f"steps/epoch={len(dl)} device={device}", flush=True)
    print(f"  zscore mean {np.round(mean, 4)}", flush=True)
    print(f"  zscore std  {np.round(std, 4)}", flush=True)

    R = a.results_dir
    weights_dir = os.path.join(R, "weights")
    losses_dir = os.path.join(R, "losses")
    gen_dir = os.path.join(R, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- CSDI's own utils.train recipe: Adam wd=1e-6, MultiStepLR @0.75/0.9 x0.1 ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1.0e-6)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.75 * epochs), int(0.9 * epochs)], gamma=0.1)

    records = []                      # (step, phase, loss_total) -- guideline schema
    step, epoch_secs = 0, []
    val_every = max(1, epochs // 20)
    t0 = time.time()
    for epoch in range(epochs):
        te = time.time()
        model.train()
        run, nb = 0.0, 0
        for (xb,) in dl:
            optimizer.zero_grad()
            loss = model({"observed_data": xb})
            loss.backward()
            optimizer.step()
            run += float(loss.item())
            nb += 1
            if step % 100 == 0:
                records.append((step, "train", float(loss.item())))
            step += 1
        lr_sched.step()

        val_loss = float("nan")
        if epoch % val_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                vs = [float(model({"observed_data": xb}, is_train=0).item()) for (xb,) in dl_val]
            val_loss = float(np.mean(vs)) if vs else float("nan")
            records.append((step, "val", val_loss))
        epoch_secs.append(time.time() - te)
        print(f"  epoch {epoch:4d}/{epochs}  train={run / max(nb, 1):.5f}  "
              f"val={val_loss:.5f}  lr={lr_sched.get_last_lr()[0]:.2e}  "
              f"{epoch_secs[-1]:.1f}s", flush=True)
    train_time = time.time() - t0

    loss_csv = os.path.join(losses_dir, f"{tagp}seed_{a.seed}_losses.csv")
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "phase", "loss_total"])
        w.writerows(records)
    tr = [l for _, p, l in records if p == "train"]
    va = [l for _, p, l in records if p == "val"]
    first_nan = next((s for s, _, l in records if not np.isfinite(l)), None)

    # --- generate the A/B bank (timed) --------------------------------------
    gen_num = a.gen_num if a.gen_num > 0 else n_train_full
    g0 = time.time()
    Xg_std = model.generate(gen_num, seq_len, gen_batch=a.gen_batch)   # (N, T, K) standardized
    Xg = Xg_std.astype(np.float64) * std + mean                        # price scale, per channel
    raw_min, raw_max = float(Xg.min()), float(Xg.max())
    Xg, n_nonpos_s0, n_nonpos_all = rescale_to_s0(Xg)
    gen_time = time.time() - g0

    s0_exact = bool(np.all(Xg[:, 0, :] == S0))
    finite_pos = bool(np.isfinite(Xg).all() and (Xg > 0).all())
    if not (s0_exact and finite_pos):
        raise SystemExit(f"ABORT: generated bank violates the section 4 contract "
                         f"(S0_exact={s0_exact}, finite_positive={finite_pos}) "
                         f"-- refusing to write")

    if not a.tag:
        torch.save({"model": model.state_dict(), "seed": a.seed, "target_dim": D,
                    "seq_len": seq_len, "config": cfg,
                    "zscore_mean": mean.tolist(), "zscore_std": std.tolist()},
                   os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        retuned = []
        if epochs != BASE_CONFIG["train"]["epochs"]:
            retuned.append("epochs")
        if batch_size != BASE_CONFIG["train"]["batch_size"]:
            retuned.append("batch_size")
        out_cfg = {
            "method": "CSDI", "seed": a.seed, "dataset": "TrueDataset",
            "data_dir": a.data_dir, "seq_tag": a.seq_tag,
            "d": D, "seq_len": seq_len, "n_train": int(S.shape[0]),
            "joint_or_per_asset": "joint", "is_unconditional": 1, "target_dim": D,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "steps_per_epoch": len(dl), "total_steps": step,
            **cfg["diffusion"],
            "timeemb": cfg["model"]["timeemb"], "featureemb": cfg["model"]["featureemb"],
            "params": int(nparam),
            "paper_hyperparams": not retuned,
            "retuned_for_truedata": retuned,
            "zscore": "per-channel, fitted on train split only",
            "zscore_mean": mean.tolist(), "zscore_std": std.tolist(),
            "splits_read": ["train", "val"],
            "train_time_sec": round(train_time, 1), "gpu": GPU_NAME,
        }
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(out_cfg, f, indent=2)

    n, t, d = Xg.shape
    npy = os.path.join(gen_dir, f"{tagp}generated_paths_{n}x{t}x{d}.npy")
    np.save(npy, Xg)
    meta = {
        "method": "CSDI", "seed": a.seed, "dataset": "TrueDataset",
        "data_dir": a.data_dir, "seq_tag": a.seq_tag,
        "shape": list(Xg.shape), "dtype": str(Xg.dtype),
        "S0": S0, "S0_exact": s0_exact, "s0_rescaled": True,
        "S_min": float(Xg.min()), "S_max": float(Xg.max()),
        "all_finite_positive": finite_pos,
        "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
        "gen_time_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
        "sec_per_epoch": round(float(np.mean(epoch_secs)), 1) if epoch_secs else None,
        "epochs_run": epochs, "batch_size": batch_size, "params": int(nparam),
        "num_steps": cfg["diffusion"]["num_steps"],
        "min_total_loss": float(min(tr)) if tr else None,
        "min_val_loss": float(min(va)) if va else None,
        "final_val_loss": float(va[-1]) if va else None,
        "first_nan_step": first_nan,
        "raw_price_min": raw_min, "raw_price_max": raw_max,
        "n_nonpositive_s0_before_rescale": n_nonpos_s0,
        "n_nonpositive_total_before_rescale": n_nonpos_all,
        "n_entries": int(Xg.size),
        "gpu": GPU_NAME, "date": time.strftime("%Y-%m-%d"),
    }
    with open(os.path.join(gen_dir, f"{tagp}metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {npy} ({os.path.getsize(npy) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
