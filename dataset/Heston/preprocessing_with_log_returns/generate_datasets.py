"""
Datasets for the log-return-preprocessing LS4 experiment.

Reuses the *unmodified* Heston generator from the parent folder
(`generate_heston.generate_heston`) so the SDE, parameters and RNG stream are
byte-identical to the main benchmark; only the sample counts differ.

Splits produced (all float64, price + variance):
  train  seed 0  n=4096   generator sees only this split
  test   seed 1  n=4096   held-out real reference for the A/B metrics
  disc   seed 2  n=4096   A18/A19 discriminative-classifier real data
  ps     seed 3  n=512    fresh, strictly-independent real queries for
                          path-shadowing (never overlaps the 1M generated bank)

The seeds match the main benchmark's train/test/disc convention (0/1/2); the
path-shadowing query split uses a *new* seed (3) so it is independent of every
other split and of the generated bank.
"""
import os, sys
import numpy as np

# import the untouched parent generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_heston import generate_heston, DT

OUT = os.path.dirname(os.path.abspath(__file__))

SPLITS = [
    ("",      0, 4096),   # train
    ("_test", 1, 4096),   # test
    ("_disc", 2, 4096),   # disc
    ("_ps",   3,  512),   # path-shadowing real queries
]


def sbts_sigma(S):
    """SBTS pooled sigma = std of raw log-returns over all paths x timesteps."""
    R = np.log(S[:, 1:] / S[:, :-1])
    return float(R.std())


if __name__ == "__main__":
    for name, seed, n in SPLITS:
        S, v = generate_heston(n_samples=n, seq_len=128, seed=seed)
        np.save(os.path.join(OUT, f"heston_S{name}_{n}x128.npy"), S)
        np.save(os.path.join(OUT, f"heston_v{name}_{n}x128.npy"), v)
        tag = name.lstrip("_") or "train"
        print(f"{tag:6s} seed={seed} n={n:5d}  "
              f"S[:,0].mean={S[:,0].mean():.2f}  S[:,0].std={S[:,0].std():.2e}")

    # report the SBTS-style sigma on the TRAIN split (seed 0, n=4096)
    S_train, _ = generate_heston(n_samples=4096, seq_len=128, seed=0)
    sig = sbts_sigma(S_train)
    R = np.log(S_train[:, 1:] / S_train[:, :-1])
    print("\n=== SBTS sigma (train split, 4096x128) ===")
    print(f"DT            = {DT:.6f}  (1/250)")
    print(f"sqrt(DT)      = {np.sqrt(DT):.6f}")
    print(f"sigma=std(R)  = {sig:.8f}   # pooled over 4096 x 127 raw log-returns")
    print(f"R.mean        = {R.mean():.3e}")
    print(f"scaled R std  = {(R*np.sqrt(DT)/sig).std():.6f}  (should be sqrt(DT)={np.sqrt(DT):.6f})")
