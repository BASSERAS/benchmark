"""
A13 Discriminative Score — PyTorch implementation.

Two post-hoc classifiers trained to distinguish real from synthetic sequences.
Returns |accuracy - 0.5| (lower = harder to tell apart = better generation).

Classifiers:
  GRUDiscriminator : 2-layer GRU -> last hidden -> Linear(1)
  MLPDiscriminator : flatten(T*d) -> Dense(128,ReLU) -> Dense(64,ReLU) -> Dense(1)

Fixes the hidden_dim=int(1/2)=0 crash from the original TF1 implementation
for 1D time series (Heston price: d=1).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def _to_tensor(x, device):
    return torch.FloatTensor(x).to(device)


class GRUDiscriminator(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)          # h: (2, batch, hidden)
        return self.fc(h[-1])       # (batch, 1)


class MLPDiscriminator(nn.Module):
    def __init__(self, seq_len: int, n_features: int):
        super().__init__()
        in_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)          # (batch, 1)


def _train_and_eval(
    model: nn.Module,
    real: np.ndarray,
    fake: np.ndarray,
    n_steps: int = 2000,
    batch_size: int = 128,
    device_str: str = "cpu",
) -> float:
    """Train classifier, return |acc - 0.5|."""
    device = torch.device(device_str)
    model  = model.to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Labels: 1 = real, 0 = fake
    N = min(len(real), len(fake))
    X = np.concatenate([real[:N], fake[:N]], axis=0)
    y = np.concatenate([np.ones(N, np.float32), np.zeros(N, np.float32)])
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    X_tr_t = _to_tensor(X_tr, device)
    y_tr_t = _to_tensor(y_tr, device).unsqueeze(1)

    model.train()
    for _ in range(n_steps):
        idx_b = np.random.randint(0, len(X_tr), batch_size)
        logits = model(X_tr_t[idx_b])
        loss   = loss_fn(logits, y_tr_t[idx_b])
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(_to_tensor(X_te, device)).squeeze(1)
        preds = (torch.sigmoid(logits_te) > 0.5).cpu().numpy()
    acc = float((preds == y_te).mean())
    return abs(acc - 0.5)


def compute_discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    n_steps: int = 2000,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute A13 discriminative score using GRU and MLP classifiers.

    Parameters
    ----------
    real, fake : ndarray shape (N, T) or (N, T, d)
    n_steps    : training iterations per classifier

    Returns
    -------
    dict with keys disc_score_gru, disc_score_mlp  (lower is better)
    """
    if real.ndim == 2:
        real = real[:, :, None]
    if fake.ndim == 2:
        fake = fake[:, :, None]

    N      = min(len(real), len(fake))
    T, d   = real.shape[1], real.shape[2]
    # Min-max normalize both to [0,1] using real stats
    mn = real.reshape(-1, d).min(axis=0)
    mx = real.reshape(-1, d).max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    real_n = (real - mn) / denom
    fake_n = (fake - mn) / denom

    hidden = max(8, d * 8)

    score_gru = _train_and_eval(
        GRUDiscriminator(d, hidden),
        real_n[:N], fake_n[:N], n_steps, device_str=device,
    )
    score_mlp = _train_and_eval(
        MLPDiscriminator(T, d),
        real_n[:N], fake_n[:N], n_steps, device_str=device,
    )
    return {"disc_score_gru": round(score_gru, 6),
            "disc_score_mlp": round(score_mlp, 6)}
