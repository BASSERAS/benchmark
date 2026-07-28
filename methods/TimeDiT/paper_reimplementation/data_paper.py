"""Paper-reproduction datasets — Yoon et al. (2019) synthetic-generation benchmark.

Two datasets used by TimeDiT Table 6 (synthetic generation, seq_len=24):
  - Sine   : 5 independent sinusoids per sample, random freq/phase (Yoon 2019).
  - Stocks : Google daily OHLCV+AdjClose (6 features), sliding windows (Yoon 2019).

Both are min-max normalised to [0, 1] per the paper (Appendix E: "Minmax scaler is
used for all models") and windowed to length 24.  Returned array: (N, L, C) float32.
"""
import numpy as np


def sine_data_generation(no=10000, seq_len=24, dim=5, seed=0):
    """Yoon 2019 sine data: each of `dim` channels is sin(freq·t + phase),
    freq,phase ~ U(0, 0.1), then mapped from [-1,1] to [0,1]."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(no):
        sample = np.zeros((seq_len, dim))
        for k in range(dim):
            freq = rng.uniform(0, 0.1)
            phase = rng.uniform(0, 0.1)
            sig = np.sin(freq * np.arange(seq_len) + phase)
            sample[:, k] = (sig + 1) * 0.5
        data.append(sample)
    return np.asarray(data, dtype=np.float32)


def _minmax_windows(arr, seq_len):
    """Global per-feature min-max to [0,1] then sliding windows of length seq_len."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min(0, keepdims=True)
    mx = arr.max(0, keepdims=True)
    norm = (arr - mn) / (mx - mn + 1e-7)
    windows = [norm[i:i + seq_len] for i in range(len(norm) - seq_len + 1)]
    return np.asarray(windows, dtype=np.float32)


def stock_data_generation(csv_path, seq_len=24):
    """Yoon 2019 Stocks loader: read CSV, reverse to chronological, min-max, window."""
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    raw = raw[::-1]  # Yoon reverses so time runs forward
    return _minmax_windows(raw, seq_len)


def load_dataset(name, seq_len=24, stock_csv=None, seed=0):
    if name == "sine":
        return sine_data_generation(no=10000, seq_len=seq_len, dim=5, seed=seed)
    if name == "stock":
        assert stock_csv is not None, "stock_csv path required"
        return stock_data_generation(stock_csv, seq_len=seq_len)
    raise ValueError(name)
