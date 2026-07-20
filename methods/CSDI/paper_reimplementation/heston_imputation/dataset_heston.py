"""Heston imputation dataset for CSDI — mirrors dataset_physio.py.

Purpose: feed the authors' *conditional* CSDI (``is_unconditional=0``,
``target_strategy='random'``) an imputation task on the benchmark's own 8192×128
Heston price paths, so that CSDI's **own** metric — quantile CRPS
(``utils.calc_quantile_CRPS``) — can be applied to Heston. This is the exact
paper metric transferred to Heston for the results-README §6 comparison column.

This is a *different* task from ``methods/CSDI/code/train_heston.py`` (the
unconditional generator, which sets ``gt_mask≡0`` and never masks points): here
a random ``missing_ratio`` fraction of each path is held out as imputation
targets, exactly as ``dataset_physio.parse_id`` does for PhysioNet.

Batch format is byte-identical to ``Physio_Dataset.__getitem__`` so the
authors' ``CSDI_Physio`` / ``train`` / ``evaluate`` run unchanged:
  observed_data (L,K)=(128,1)  z-scored price
  observed_mask (L,K)          all ones (every price is observed)
  gt_mask       (L,K)          1 except a random missing_ratio fraction → 0
  timepoints    (L,)           arange(128)

Source paths : ../../../../dataset/Heston/heston_S_8192x128.npy   (8192,128) float64
Normalisation: global z-score (single feature) — mean/std of the canonical
               unconditional run (101.32547381502401 / 9.971659995159825),
               matching CSDI's PhysioNet convention (evaluate scaler=1, mean=0).
"""
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA_NPY = os.path.join(REPO, "dataset", "Heston", "heston_S_8192x128.npy")
CACHE_DIR = os.path.join(HERE, "cache")

# z-score of the canonical unconditional CSDI Heston run (weights/seed_0_config.json)
ZMEAN = 101.32547381502401
ZSTD = 9.971659995159825


class Heston_Dataset(Dataset):
    """One Heston price path = one sample; random missing_ratio held out as targets."""

    def __init__(self, eval_length=128, use_index_list=None, missing_ratio=0.1, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground-truth (imputation-target) choice

        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"heston_missing{missing_ratio}_seed{seed}.pk")

        if not os.path.isfile(path):
            prices = np.load(DATA_NPY).astype("float32")          # (N, L)
            n, L = prices.shape
            assert L == eval_length, f"expected L={eval_length}, got {L}"

            observed_values = ((prices - ZMEAN) / ZSTD)[:, :, None]  # (N, L, 1) z-scored
            observed_masks = np.ones_like(observed_values)           # every price observed

            gt_masks = observed_masks.copy()
            for i in range(n):
                # hold out a random missing_ratio fraction of the L observed points
                obs_idx = np.arange(L)
                miss = np.random.choice(
                    obs_idx, int(L * missing_ratio), replace=False
                )
                gt_masks[i, miss, 0] = 0.0

            self.observed_values = observed_values
            self.observed_masks = observed_masks.astype("float32")
            self.gt_masks = gt_masks.astype("float32")
            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)
        else:
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=0, batch_size=16, missing_ratio=0.1):
    """70/10/20 train/valid/test split — same scheme as dataset_physio.get_dataloader."""
    dataset = Heston_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = int(nfold * 0.2 * len(dataset))
    end = int((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_loader = DataLoader(
        Heston_Dataset(use_index_list=train_index, missing_ratio=missing_ratio, seed=seed),
        batch_size=batch_size, shuffle=1,
    )
    valid_loader = DataLoader(
        Heston_Dataset(use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed),
        batch_size=batch_size, shuffle=0,
    )
    test_loader = DataLoader(
        Heston_Dataset(use_index_list=test_index, missing_ratio=missing_ratio, seed=seed),
        batch_size=batch_size, shuffle=0,
    )
    return train_loader, valid_loader, test_loader
