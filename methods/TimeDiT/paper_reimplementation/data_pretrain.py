"""Pretraining corpus assembly for the TimeDiT paper reproduction (§5.6).

The paper pretrains DiT-S on a mixture of datasets, then finetunes on the
synthetic-generation targets (Sine, Stocks).  This module assembles the
pretraining corpus.

Corpus (per user decision "ETT x4 now, add elec/weather if you provide them"):
  * always: etth1, etth2, ettm1, ettm2  (downloaded from the official ETDataset
    GitHub mirror, fully reproducible).
  * optionally: electricity.csv, weather.csv  -- auto-included if the user drops
    them into `pretrain_data/`.  Documented as a 2-dataset deviation until then.

Standardize-Pipeline (paper §4.1): per-channel z-normalisation, window to a fixed
seq_len, pad the channel axis to K_MAX with a validity vector so padded channels
can be excluded from the diffusion loss and masked at sampling.  Datasets with more
than K_MAX channels are split into K_MAX-sized channel blocks.

Data format (ETT CSVs): first column `date` (YYYY-MM-DD HH:MM:SS) is dropped; the
remaining 7 columns (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) are numeric.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_DIR = os.path.join(_HERE, "pretrain_data")
K_MAX = 40

_ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
ETT_URLS = {
    "etth1": f"{_ETT_BASE}/ETTh1.csv",
    "etth2": f"{_ETT_BASE}/ETTh2.csv",
    "ettm1": f"{_ETT_BASE}/ETTm1.csv",
    "ettm2": f"{_ETT_BASE}/ETTm2.csv",
}
OPTIONAL_CSVS = ["electricity.csv", "weather.csv"]


def _download_ett(force=False):
    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    for name, url in ETT_URLS.items():
        dst = os.path.join(PRETRAIN_DIR, f"{name}.csv")
        if force or not os.path.exists(dst):
            print(f"[pretrain-data] downloading {name} <- {url}")
            urllib.request.urlretrieve(url, dst)
        else:
            print(f"[pretrain-data] cached {name} ({os.path.getsize(dst)} bytes)")


def _load_csv_numeric(path):
    """Read a CSV, drop the leading date/timestamp column, return (T, C) float32."""
    df = pd.read_csv(path)
    # drop any non-numeric column (the date/timestamp)
    num = df.select_dtypes(include=[np.number])
    return num.to_numpy(dtype=np.float32)


def _znorm(arr):
    """Per-channel z-normalisation on (T, C)."""
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (arr - mu) / sd


def _windows(arr, seq_len, stride):
    """Sliding windows over time: (T, C) -> (N, seq_len, C)."""
    T = arr.shape[0]
    if T < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)
    idx = range(0, T - seq_len + 1, stride)
    out = np.stack([arr[i:i + seq_len] for i in idx], axis=0)
    return out.astype(np.float32)


def _channel_blocks(arr, k_max):
    """Split channel axis into <= k_max blocks: (T, C) -> list of (T, c_i)."""
    C = arr.shape[1]
    if C <= k_max:
        return [arr]
    return [arr[:, i:i + k_max] for i in range(0, C, k_max)]


def load_pretrain_corpus(seq_len=24, stride=1, max_windows_per_ds=20000, seed=0):
    """Assemble the pretraining corpus.

    Returns a list of dicts, one per (dataset, channel-block):
        {"name": str, "windows": (N, seq_len, C), "valid": (K_MAX,) float32, "C": int}
    where C <= K_MAX is the true channel count and `valid[:C]==1`, rest 0.
    """
    rng = np.random.RandomState(seed)
    _download_ett()

    # gather available csv paths
    paths = {name: os.path.join(PRETRAIN_DIR, f"{name}.csv") for name in ETT_URLS}
    for opt in OPTIONAL_CSVS:
        p = os.path.join(PRETRAIN_DIR, opt)
        if os.path.exists(p):
            paths[opt.rsplit(".", 1)[0]] = p
            print(f"[pretrain-data] optional dataset present: {opt}")

    corpus = []
    for name, path in paths.items():
        arr = _znorm(_load_csv_numeric(path))          # (T, C)
        for bi, block in enumerate(_channel_blocks(arr, K_MAX)):
            w = _windows(block, seq_len, stride)        # (N, L, c)
            if len(w) == 0:
                continue
            if len(w) > max_windows_per_ds:
                sel = rng.choice(len(w), max_windows_per_ds, replace=False)
                w = w[sel]
            c = w.shape[-1]
            valid = np.zeros(K_MAX, dtype=np.float32)
            valid[:c] = 1.0
            tag = name if bi == 0 else f"{name}_b{bi}"
            corpus.append({"name": tag, "windows": w, "valid": valid, "C": c})
            print(f"[pretrain-data] {tag}: windows={w.shape} channels={c}")
    total = sum(len(d["windows"]) for d in corpus)
    print(f"[pretrain-data] corpus: {len(corpus)} blocks, {total} windows total")
    return corpus


def pad_to_kmax(x, C):
    """Pad channel axis of (B, L, C) up to (B, L, K_MAX); return (padded, valid).

    valid is (K_MAX,) float32 with valid[:C]=1.  Padded channels are zeros.
    """
    B, L, c = x.shape
    assert c == C <= K_MAX, f"expected C={C}<=K_MAX={K_MAX}, got {c}"
    if C < K_MAX:
        pad = np.zeros((B, L, K_MAX - C), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=-1)
    valid = np.zeros(K_MAX, dtype=np.float32)
    valid[:C] = 1.0
    return x, valid


if __name__ == "__main__":
    corpus = load_pretrain_corpus(seq_len=24)
    for d in corpus:
        print(d["name"], d["windows"].shape, "C=", d["C"], "valid_sum=", d["valid"].sum())
