"""Seed-only reproducibility shim for the official SBTS multiprocessing worker.

The official worker seeds NumPy's Python RNG, while its Brownian draws occur
inside a Numba-jitted kernel with an independent RNG state.  This module seeds
that Numba state inside each worker before calling the unchanged official
simulation kernel.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numba as nb
import numpy as np


OFFICIAL_ADAPTER = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/methods/SBTS/code/sbts_generate.py"
)
sys.path.insert(0, str(OFFICIAL_ADAPTER.parent))
import sbts_generate as official_sbts  # noqa: E402


@nb.jit(nopython=True, cache=True)
def _seed_numba_rng(seed: int) -> None:
    np.random.seed(seed)


def seeded_worker(args):
    """Official ``_worker`` with one additional Numba-RNG seed operation."""

    chunk_size, x_train, n_steps, n_train, order, n_pi, bandwidth, dt, seed = args
    np.random.seed(seed)
    _seed_numba_rng(int(seed))
    data = np.zeros((chunk_size, x_train.shape[1]), dtype=np.float64)
    for path_index in range(chunk_size):
        data[path_index] = official_sbts._simulate_one(
            n_steps,
            n_train,
            order,
            x_train,
            n_pi,
            bandwidth,
            dt,
        )
    return data[:, 1:]


def enable_seeded_worker(adapter) -> None:
    """Replace only the official process worker's RNG initialization."""

    adapter._worker = seeded_worker


__all__ = ["enable_seeded_worker", "seeded_worker"]
