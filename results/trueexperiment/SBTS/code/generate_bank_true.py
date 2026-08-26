#!/usr/bin/env python3
"""
Generate the final SBTS bank on a TrueDataset build, at the calibrated (h, K).

`sbts_generate_true.py` is a library: the calibrator imports `generate_paths`
and drives it over a grid. This is the driver for the ONE run that produces the
bank every downstream metric is scored on, so it writes the artefact in the
layout `compute_all_multiasset.py` expects:

    <out-root>/generated_paths/seed_<S>/generated_paths_<N>x<T>x<d>.npy
    <out-root>/generated_paths/seed_<S>/metadata.json

M_simu defaults to the size of the training split rather than a round number.
Every real-vs-real threshold this bank is judged against was measured between
real splits of exactly that size, and the vol estimator's sampling noise falls
like 1/sqrt(m) (1.67 pp at 512, 0.42 pp at 8192). Generating a different count
would compare a differently-noisy estimate against those thresholds.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_generator():
    """Load the generator module and REGISTER it in sys.modules.

    Registration is not optional: generate_paths fans out over multiprocessing,
    and the default fork+pickle path pickles `_worker` by qualified name. An
    unregistered module makes that name resolve to a different object in the
    child, which fails with `it's not the same object as sbts_generate_true._worker`.
    """
    spec = importlib.util.spec_from_file_location(
        "sbts_generate_true", os.path.join(HERE, "sbts_generate_true.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sbts_generate_true"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seq-tag", required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N-pi", type=int, default=50)
    ap.add_argument("--m-simu", type=int, default=None,
                    help="paths to generate. Default: size of the train split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-root", default=os.path.join(HERE, ".."))
    args = ap.parse_args()

    gen = _load_generator()
    S_train = np.load(os.path.join(args.data_dir, f"true_S_{args.seq_tag}.npy"))
    m_simu = args.m_simu or len(S_train)

    print(f"train {S_train.shape}  ->  bank {m_simu} paths  "
          f"h={args.h} K={args.K} N_pi={args.N_pi} seed={args.seed}", flush=True)

    gen.warmup_jit(d=S_train.shape[2])
    S_gen, meta = gen.generate_paths(S_train, m_simu, args.h, K=args.K,
                                     N_pi=args.N_pi, n_workers=args.workers,
                                     seed=args.seed)

    if not np.isfinite(S_gen).all():
        raise SystemExit("generated bank contains non-finite values -- refusing to write")

    d = os.path.join(args.out_root, "generated_paths", f"seed_{args.seed}")
    os.makedirs(d, exist_ok=True)
    n, t, a = S_gen.shape
    path = os.path.join(d, f"generated_paths_{n}x{t}x{a}.npy")
    np.save(path, S_gen)

    meta.update(data_dir=args.data_dir, seq_tag=args.seq_tag,
                train_shape=list(S_train.shape))
    with open(os.path.join(d, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
