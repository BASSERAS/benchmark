"""Yoon et al. (2019) discriminative & predictive scores — faithful torch port.

These are the two metrics TimeDiT reports for the synthetic-generation task
(Table 6), described in the paper (Appendix D): a post-hoc 2-layer recurrent
classifier (discriminative) and a post-hoc 2-layer recurrent one-step predictor
trained on synthetic / evaluated on real (predictive, TSTR).

Discriminative score = |accuracy − 0.5|      (0 = indistinguishable, ideal)
Predictive score     = MAE of TSTR predictor (lower = better)

Protocol matches the released TimeGAN metric code (hidden = int(dim/2) clamped to
≥1, 2000 / 5000 training iters, batch 128, Adam).  Real and generated arrays are
(N, L, C) in [0, 1].
"""
import numpy as np
import torch
import torch.nn as nn


def _train_test_split(x, frac=0.8):
    n = len(x)
    idx = np.random.permutation(n)
    cut = int(n * frac)
    return x[idx[:cut]], x[idx[cut:]]


class _LSTMClassifier(nn.Module):
    """Paper Appendix D: post-hoc 2-layer LSTM classifier."""
    def __init__(self, dim, hidden, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM returns (out, (h, c)); unpack the state tuple
        return self.fc(out[:, -1, :])  # last step logit


class _LSTMPredictor(nn.Module):
    """Paper Appendix D: post-hoc 2-layer LSTM one-step predictor."""
    def __init__(self, in_dim, hidden, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM returns (out, (h, c)); unpack the state tuple
        return self.fc(out)  # per-step prediction


def discriminative_score(real, fake, device="cuda", iters=2000, batch=128, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim = real.shape[-1]
    hidden = max(int(dim / 2), 1)
    r_tr, r_te = _train_test_split(real)
    f_tr, f_te = _train_test_split(fake)
    model = _LSTMClassifier(dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters())
    bce = nn.BCEWithLogitsLoss()
    r_tr_t = torch.tensor(r_tr, dtype=torch.float32, device=device)
    f_tr_t = torch.tensor(f_tr, dtype=torch.float32, device=device)
    model.train()
    for _ in range(iters):
        ri = np.random.randint(0, len(r_tr_t), batch)
        fi = np.random.randint(0, len(f_tr_t), batch)
        xr, xf = r_tr_t[ri], f_tr_t[fi]
        logits = model(torch.cat([xr, xf], 0))
        y = torch.cat([torch.ones(len(xr), 1, device=device),
                       torch.zeros(len(xf), 1, device=device)], 0)
        loss = bce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        xr = torch.tensor(r_te, dtype=torch.float32, device=device)
        xf = torch.tensor(f_te, dtype=torch.float32, device=device)
        pr = torch.sigmoid(model(xr)).cpu().numpy() > 0.5
        pf = torch.sigmoid(model(xf)).cpu().numpy() > 0.5
    acc = (pr.sum() + (~pf).sum()) / (len(pr) + len(pf))
    return abs(acc - 0.5)


def predictive_score(real, fake, device="cuda", iters=5000, batch=128, seed=0):
    """TSTR: train one-step predictor on `fake`, evaluate MAE on `real`.
    Predict the last feature's next step from the other features (Yoon 2019)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim = real.shape[-1]
    hidden = max(int(dim / 2), 1)
    in_dim = max(dim - 1, 1)
    model = _LSTMPredictor(in_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters())
    l1 = nn.L1Loss()
    f_t = torch.tensor(fake, dtype=torch.float32, device=device)

    def make_xy(t):
        if dim == 1:
            x = t[:, :-1, :]
            y = t[:, 1:, :]
        else:
            x = t[:, :-1, :-1]
            y = t[:, 1:, -1:]
        return x, y

    model.train()
    for _ in range(iters):
        bi = np.random.randint(0, len(f_t), batch)
        x, y = make_xy(f_t[bi])
        pred = model(x)
        loss = l1(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        r_t = torch.tensor(real, dtype=torch.float32, device=device)
        x, y = make_xy(r_t)
        mae = l1(model(x), y).item()
    return mae
