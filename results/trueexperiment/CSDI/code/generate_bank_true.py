#!/usr/bin/env python3
"""Draw an arbitrary-sized CSDI bank from a trained checkpoint.

`train_true.py` writes the 6 144-path A/B bank as part of the training run.
This script exists for the OTHER bank: the 8 192-path conditional-CRPS pool of
guideline section 8, which is pinned to the paper's size and must not be 6 144.

Why a separate script rather than a `--gen-num 8192` on the trainer: generating
the CRPS pool from the SAME checkpoint the A/B bank came from is the point. If
the pool were produced by a second training run, the two tables would describe
two different models and the A-table would stop being a sanity gate on Table C.
So this reloads `weights/seed_{i}_model.pt` and samples again -- same weights,
same z-score statistics (they are stored in the checkpoint, not recomputed),
different draw.

THE STANDARDISATION STATISTICS COME FROM THE CHECKPOINT, NOT FROM THE DATA
--------------------------------------------------------------------------
`train_true.py` saves `zscore_mean` / `zscore_std` alongside the weights. This
script uses those and never recomputes them from `--data-dir`. Recomputing would
usually give the same answer and would silently give a different one the moment
anybody points this at a different build -- and the failure would be a plausible
bank, not an error. The `--data-dir` argument here is used ONLY to record
provenance in `metadata.json` and to cross-check the shape against the weights.

Usage
-----
  V=dataset/TrueDataset/variants/om_2022-07_N6144
  R=results/trueexperiment/CSDI
  for S in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python generate_bank_true.py \
        --seed $S --m-simu 8192 --data-dir $V --seq-tag 6144x128x8 \
        --results-dir $R --out-root $R/crps_banks
  done
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from csdi_true import GPU_NAME, S0, CSDI_TrueData, load_split, rescale_to_s0

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--m-simu", type=int, required=True,
                    help="8192 for the conditional-CRPS pool (guideline section 8). "
                         "Do NOT use this script for the A/B bank -- that is 6144 "
                         "and train_true.py writes it.")
    ap.add_argument("--data-dir", required=True, help="provenance + shape cross-check only")
    ap.add_argument("--seq-tag", required=True)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS,
                    help="where weights/seed_{i}_model.pt lives")
    ap.add_argument("--out-root", default=None, help="default: <results-dir>/crps_banks")
    ap.add_argument("--gen-batch", type=int, default=256)
    a = ap.parse_args()

    out_root = a.out_root or os.path.join(a.results_dir, "crps_banks")
    ckpt_path = os.path.join(a.results_dir, "weights", f"seed_{a.seed}_model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"ABORT: no checkpoint at {ckpt_path} -- train this seed first")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # weights_only=True: the checkpoint train_true.py writes holds only tensors,
    # dicts and lists of primitives, so the safe loader is sufficient. Leaving the
    # torch default (False) would unpickle arbitrary objects for no benefit.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    D = int(ckpt["target_dim"])
    seq_len = int(ckpt["seq_len"])
    mean = np.asarray(ckpt["zscore_mean"], dtype=np.float64)
    std = np.asarray(ckpt["zscore_std"], dtype=np.float64)

    # Cross-check the checkpoint against the build we claim to be sampling for.
    S_train = load_split(a.data_dir, a.seq_tag, "")
    if S_train.shape[1:] != (seq_len, D):
        raise SystemExit(f"ABORT: checkpoint is (T={seq_len}, d={D}) but "
                         f"{a.data_dir} @ {a.seq_tag} is {S_train.shape[1:]} -- "
                         f"wrong build for these weights")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = CSDI_TrueData(ckpt["config"], device, target_dim=D).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[CSDI bank] seed={a.seed} m_simu={a.m_simu} T={seq_len} d={D} "
          f"from {ckpt_path}", flush=True)

    g0 = time.time()
    Xg_std = model.generate(a.m_simu, seq_len, gen_batch=a.gen_batch)
    Xg = Xg_std.astype(np.float64) * std + mean
    raw_min, raw_max = float(Xg.min()), float(Xg.max())
    Xg, n_nonpos_s0, n_nonpos_all = rescale_to_s0(Xg)
    gen_time = time.time() - g0

    s0_exact = bool(np.all(Xg[:, 0, :] == S0))
    finite_pos = bool(np.isfinite(Xg).all() and (Xg > 0).all())
    if not (s0_exact and finite_pos):
        raise SystemExit(f"ABORT: bank violates the section 4 contract "
                         f"(S0_exact={s0_exact}, finite_positive={finite_pos})")

    d_out = os.path.join(out_root, "generated_paths", f"seed_{a.seed}")
    os.makedirs(d_out, exist_ok=True)
    n, t, k = Xg.shape
    npy = os.path.join(d_out, f"generated_paths_{n}x{t}x{k}.npy")
    np.save(npy, Xg)

    meta = {
        "method": "CSDI", "seed": a.seed, "dataset": "TrueDataset",
        "role": "conditional-CRPS pool (guideline section 8)",
        "data_dir": a.data_dir, "seq_tag": a.seq_tag,
        "checkpoint": os.path.relpath(ckpt_path, DEFAULT_RESULTS),
        "shape": list(Xg.shape), "dtype": str(Xg.dtype),
        "S0": S0, "S0_exact": s0_exact, "s0_rescaled": True,
        "S_min": float(Xg.min()), "S_max": float(Xg.max()),
        "all_finite_positive": finite_pos,
        "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
        "raw_price_min": raw_min, "raw_price_max": raw_max,
        "n_nonpositive_s0_before_rescale": n_nonpos_s0,
        "n_nonpositive_total_before_rescale": n_nonpos_all,
        "zscore_source": "checkpoint (not recomputed from --data-dir)",
        "zscore_mean": mean.tolist(), "zscore_std": std.tolist(),
        "gen_time_sec": round(gen_time, 1), "gpu": GPU_NAME,
        "date": time.strftime("%Y-%m-%d"),
    }
    with open(os.path.join(d_out, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {npy} ({os.path.getsize(npy) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
