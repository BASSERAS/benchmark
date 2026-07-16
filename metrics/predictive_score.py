"""
A14 Predictive Score — PyTorch implementation (TSTR protocol).

Train predictor on SYNTHETIC data, evaluate one-step-ahead MAE on REAL data.
Lower MAE = better temporal fidelity of generated sequences.

Predictors:
  GRUPredictor : 2-layer GRU -> Dense(1)   [sequence-to-sequence next-step]
  MLPPredictor : sliding window of 8 steps -> Dense(64,ReLU)->Dense(32,ReLU)->Dense(1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

WINDOW = 8


def _to_tensor(x, device):
    return torch.FloatTensor(x).to(device)


class GRUPredictor(nn.Module):
    """Predicts next step from full sequence prefix."""
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)   # (batch, T, 1)


class MLPPredictor(nn.Module):
    """Predicts next step from a fixed-size context window."""
    def __init__(self, window: int, n_features: int):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window, d) -> (batch, 1)
        return self.net(x)


def _train_gru(model, data_norm, n_steps, batch_size, device_str, log_every=100):
    """TSTR: train GRU on synthetic data (next-step prediction)."""
    device  = torch.device(device_str)
    model   = model.to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    N, T, d = data_norm.shape
    data_t  = _to_tensor(data_norm, device)

    model.train()
    history = []
    for step in range(1, n_steps + 1):
        idx    = np.random.randint(0, N, batch_size)
        X_b    = data_t[idx, :-1, :]      # (batch, T-1, d)
        Y_b    = data_t[idx, 1:,  :1]     # (batch, T-1, 1) — predict first feature
        loss   = loss_fn(model(X_b), Y_b)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            history.append({"step": step, "train_mae": round(loss.item(), 6)})
    return model, history


def _eval_gru(model, real_norm, device_str):
    device = torch.device(device_str)
    model.eval()
    with torch.no_grad():
        X = _to_tensor(real_norm[:, :-1, :], device)
        preds = model(X).squeeze(-1).cpu().numpy()   # (N, T-1)
    targets = real_norm[:, 1:, 0]                   # (N, T-1)
    return float(np.mean(np.abs(preds - targets)))


def _train_mlp(model, data_norm, n_steps, batch_size, device_str, window, log_every=100):
    device  = torch.device(device_str)
    model   = model.to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    N, T, d = data_norm.shape
    data_t  = _to_tensor(data_norm, device)

    model.train()
    history = []
    for step in range(1, n_steps + 1):
        idx_n  = np.random.randint(0, N, batch_size)
        idx_t  = np.random.randint(window, T, batch_size)
        X_b    = torch.stack([data_t[n, t-window:t, :] for n, t in zip(idx_n, idx_t)])
        Y_b    = data_t[torch.tensor(idx_n), torch.tensor(idx_t), :1]
        loss   = loss_fn(model(X_b), Y_b)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            history.append({"step": step, "train_mae": round(loss.item(), 6)})
    return model, history


def _eval_mlp(model, real_norm, device_str, window):
    device = torch.device(device_str)
    model.eval()
    N, T, d = real_norm.shape
    real_t  = _to_tensor(real_norm, device)
    preds, targets = [], []
    with torch.no_grad():
        for t in range(window, T):
            X = real_t[:, t-window:t, :]
            preds.append(model(X).squeeze(-1).cpu().numpy())
            targets.append(real_norm[:, t, 0])
    return float(np.mean(np.abs(np.stack(preds,1) - np.stack(targets,1))))


def compute_predictive_score(
    real: np.ndarray,
    fake: np.ndarray,
    n_steps: int = 5000,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute A14 predictive score using GRU and MLP predictors (TSTR).

    Parameters
    ----------
    real, fake : ndarray shape (N, T) or (N, T, d)
    n_steps    : training iterations per predictor

    Returns
    -------
    dict with keys pred_score_gru, pred_score_mlp (lower is better),
    loss_history_gru, loss_history_mlp (list of {step, train_mae})
    """
    if real.ndim == 2:
        real = real[:, :, None]
    if fake.ndim == 2:
        fake = fake[:, :, None]

    d = real.shape[2]
    # Normalize using real statistics
    mn = real.reshape(-1, d).min(axis=0)
    mx = real.reshape(-1, d).max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    real_n = ((real - mn) / denom).astype(np.float32)
    fake_n = ((fake - mn) / denom).astype(np.float32)

    hidden = max(8, d * 8)

    gru_m, hist_gru = _train_gru(GRUPredictor(d, hidden), fake_n, n_steps, 128, device)
    score_gru = _eval_gru(gru_m, real_n, device)

    mlp_m, hist_mlp = _train_mlp(MLPPredictor(WINDOW, d), fake_n, n_steps, 128, device, WINDOW)
    score_mlp = _eval_mlp(mlp_m, real_n, device, WINDOW)

    return {"pred_score_gru": round(score_gru, 6),
            "pred_score_mlp": round(score_mlp, 6),
            "loss_history_gru": hist_gru,
            "loss_history_mlp": hist_mlp}
