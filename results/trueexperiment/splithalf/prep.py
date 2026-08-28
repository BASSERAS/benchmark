#!/usr/bin/env python3
"""Build the split-half evaluation trees. Reads only; writes only under /tmp/splithalf.

Half A = test/disc windows [0:3072]   (earlier part of the late era)
Half B = test/disc windows [3072:6144] (later part)

Every bank (4 generators + the real-vs-real floor) is cut to its FIRST 3072
paths and that same cut is reused for both halves, so the only thing that
differs between run A and run B is the real evaluation data.
"""
import json
import os

import numpy as np

DATA = "/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144"
RES = "/home/tbasseras/benchmark/results/trueexperiment"
OUT = "/tmp/splithalf"
TAG_IN, TAG_OUT = "6144x128x8", "3072x128x8"
H = 3072
BANKS = ["Deep-MKV-TS", "reference", "SBTS", "CSDI", "real_floor"]

halves = {"A": slice(0, H), "B": slice(H, 2 * H)}

# ---- real splits -----------------------------------------------------------
for split in ("test", "disc"):
    src = os.path.join(DATA, f"true_S_{split}_{TAG_IN}.npy")
    a = np.load(src)
    assert a.shape == (2 * H, 128, 8), f"{src}: {a.shape}"
    for name, sl in halves.items():
        d = os.path.join(OUT, f"data{name}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"true_S_{split}_{TAG_OUT}.npy"),
                np.ascontiguousarray(a[sl]))
    print(f"{split}: {a.shape} -> A{a[halves['A']].shape} B{a[halves['B']].shape}")

# ---- banks (same cut for both halves) --------------------------------------
for m in BANKS:
    for seed in range(5):
        src = os.path.join(RES, m, "generated_paths", f"seed_{seed}",
                           f"generated_paths_{TAG_IN}.npy")
        if not os.path.exists(src):
            print(f"  !! MISSING {src}")
            continue
        a = np.load(src)
        assert a.shape[0] >= H, f"{src}: only {a.shape[0]} paths"
        d = os.path.join(OUT, "gen", m, "generated_paths", f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"generated_paths_{TAG_OUT}.npy"),
                np.ascontiguousarray(a[:H]))
    print(f"bank {m}: cut to {H}")

# ---- provenance ------------------------------------------------------------
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump({
        "purpose": "split-half robustness of tables A and B on the test era",
        "half_A": "test/disc windows [0:3072] -- earlier part of the late era",
        "half_B": "test/disc windows [3072:6144] -- later part",
        "bank_cut": "first 3072 paths, IDENTICAL for both halves",
        "why": ("holding the bank fixed means the only difference between run A "
                "and run B is the real evaluation data, which isolates split "
                "sensitivity from generator sampling noise"),
        "source_data": DATA,
        "source_results": RES,
        "banks": BANKS,
    }, fh, indent=2)
print("\nwrote", os.path.join(OUT, "manifest.json"))
