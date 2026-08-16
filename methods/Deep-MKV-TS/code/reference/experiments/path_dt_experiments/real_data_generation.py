from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .real_data_protocol import (
    protocol_by_identifier,
    select_date_window,
    training_indices,
)


@dataclass(frozen=True)
class FrozenRealDataFit:
    requested_protocol: str
    scenario_bank_protocol: str
    index: str
    train_dates: np.ndarray
    validation_dates: np.ndarray
    train_prices: np.ndarray
    validation_prices: np.ndarray

    @property
    def train_sha256(self) -> str:
        contiguous = np.ascontiguousarray(self.train_prices)
        return sha256(contiguous.view(np.uint8)).hexdigest()


def load_frozen_real_data_fit(
    canonical_root: Path,
    *,
    protocol_identifier: str,
    index: str,
) -> FrozenRealDataFit:
    root = Path(canonical_root)
    index_name = str(index).upper()
    if index_name not in {"ES", "NQ", "RTY", "YM"}:
        raise ValueError(f"unsupported index: {index}")
    requested = protocol_by_identifier(protocol_identifier)
    bank_protocol = protocol_by_identifier(requested.scenario_bank_identifier)
    index_root = root / index_name
    # Fitting and validation must not read the canonical all-period arrays,
    # because those files also contain the untouched test interval.  The
    # preprocessing pipeline materializes explicit split files for this use.
    train_dates = np.load(
        index_root / "dates_train.npy", allow_pickle=False
    ).astype(str)
    validation_dates = np.load(
        index_root / "dates_validation.npy", allow_pickle=False
    ).astype(str)
    train_prices = np.asarray(
        np.load(index_root / "prices_train.npy", allow_pickle=False),
        dtype=np.float32,
    )
    validation_prices = np.asarray(
        np.load(index_root / "prices_validation.npy", allow_pickle=False),
        dtype=np.float32,
    )
    if train_prices.ndim != 2 or train_prices.shape[0] != train_dates.shape[0]:
        raise ValueError("canonical training dates and prices have inconsistent shapes")
    if (
        validation_prices.ndim != 2
        or validation_prices.shape[0] != validation_dates.shape[0]
    ):
        raise ValueError("canonical validation dates and prices have inconsistent shapes")
    if (
        not np.isfinite(train_prices).all()
        or not np.isfinite(validation_prices).all()
        or np.any(train_prices <= 0.0)
        or np.any(validation_prices <= 0.0)
    ):
        raise ValueError("canonical train/validation prices must be finite and positive")
    if train_prices.shape[0] < 2 or validation_prices.shape[0] < 1:
        raise ValueError("the frozen train and validation slices must be nonempty")
    if not np.allclose(train_prices[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise ValueError("canonical training prices must start at 100")
    return FrozenRealDataFit(
        requested_protocol=requested.identifier,
        scenario_bank_protocol=bank_protocol.identifier,
        index=index_name,
        train_dates=np.asarray(train_dates),
        validation_dates=np.asarray(validation_dates),
        train_prices=train_prices,
        validation_prices=validation_prices,
    )


def materialize_deep_mkv_adapter_dataset(
    fit: FrozenRealDataFit,
    output_root: Path,
) -> dict[str, Path]:
    """
    Materialize the legacy adapter's three-file interface without test leakage.

    The backend reads only ``train.npy`` during fitting and reconstruction.
    Its legacy ``test`` and ``disc`` filenames are populated with the frozen
    validation split, never with a test or event window.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": root / "train.npy",
        "test": root / "test.npy",
        "disc": root / "disc.npy",
    }
    np.save(paths["train"], fit.train_prices)
    np.save(paths["test"], fit.validation_prices)
    np.save(paths["disc"], fit.validation_prices)
    np.save(root / "train_dates.npy", fit.train_dates)
    np.save(root / "validation_dates.npy", fit.validation_dates)
    return paths
