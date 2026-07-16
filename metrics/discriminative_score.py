"""
A13 Discriminative Score — PyTorch implementation.

Two post-hoc classifiers trained to distinguish real from synthetic sequences.
Returns |accuracy - 0.5| (lower = harder to tell apart = better generation).
Also returns BCE loss history per training step for each classifier.

Convention (matches paper & reference code):
  score = |accuracy - 0.5|
  score = 0   → classifier at random chance → cannot distinguish → PERFECT generator
  score = 0.5 → classifier perfectly separates real from fake   → BAD generator

Paper (Yoon et al. NeurIPS 2019) uses a 2-layer LSTM; we use GRU + MLP.

Classifiers:
  GRUDiscriminator : 2-layer GRU -> last hidden -> Linear(1)
  MLPDiscriminator : flatten(T*d) -> Dense(128,ReLU) -> Dense(64,ReLU) -> Linear(1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List


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
    log_every: int = 50,
) -> Tuple[float, List[Dict]]:
    """Train classifier. Returns (|acc - 0.5|, loss_history).

    loss_history: list of {step, train_loss} dicts logged every log_every steps.
    """
    device = torch.device(device_str)
    model  = model.to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Labels: 1 = real, 0 = fake; 80/20 train/test split
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

    loss_history: List[Dict] = []

    model.train()
    for step in range(n_steps):
        idx_b  = np.random.randint(0, len(X_tr), batch_size)
        logits = model(X_tr_t[idx_b])
        loss   = loss_fn(logits, y_tr_t[idx_b])
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % log_every == 0:
            loss_history.append({"step": step + 1, "train_bce": round(loss.item(), 6)})

    model.eval()
    with torch.no_grad():
        logits_te = model(_to_tensor(X_te, device)).squeeze(1)
        preds = (torch.sigmoid(logits_te) > 0.5).cpu().numpy()
    acc = float((preds == y_te).mean())
    return abs(acc - 0.5), loss_history


def compute_discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    n_steps: int = 2000,
    device: str = "cpu",
    log_every: int = 50,
) -> Dict:
    """
    Compute A13 discriminative score using GRU and MLP classifiers.

    Parameters
    ----------
    real, fake : ndarray shape (N, T) or (N, T, d)
    n_steps    : training iterations per classifier
    log_every  : log BCE loss every this many steps

    Returns
    -------
    dict with keys:
      disc_score_gru     : float  (|acc-0.5|, lower is better)
      disc_score_mlp     : float
      loss_history_gru   : list of {step, train_bce}
      loss_history_mlp   : list of {step, train_bce}
    """
    if real.ndim == 2:
        real = real[:, :, None]
    if fake.ndim == 2:
        fake = fake[:, :, None]

    N    = min(len(real), len(fake))
    T, d = real.shape[1], real.shape[2]

    # Min-max normalize using real stats (keeps both in same [0,1] scale)
    mn    = real.reshape(-1, d).min(axis=0)
    mx    = real.reshape(-1, d).max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    real_n = (real - mn) / denom
    fake_n = (fake - mn) / denom

    hidden = max(8, d * 8)

    score_gru, hist_gru = _train_and_eval(
        GRUDiscriminator(d, hidden),
        real_n[:N], fake_n[:N], n_steps, log_every=log_every, device_str=device,
    )
    score_mlp, hist_mlp = _train_and_eval(
        MLPDiscriminator(T, d),
        real_n[:N], fake_n[:N], n_steps, log_every=log_every, device_str=device,
    )

    return {
        "disc_score_gru":   round(score_gru, 6),
        "disc_score_mlp":   round(score_mlp, 6),
        "loss_history_gru": hist_gru,
        "loss_history_mlp": hist_mlp,
    }
