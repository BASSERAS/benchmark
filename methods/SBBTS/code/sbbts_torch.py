"""
SBBTS — Schrodinger Bass Bridge for Time Series.

Self-contained port of the official reference implementation
(github.com/alexouadi/SBBTS, mirrored verbatim under ``reference/``).

The reference is split across ``models/``, ``training/`` and ``diffusion_dsbm.py``
and relies on bare imports (``from encoder_only import EncoderOnly``,
``from early_stopping import EarlyStopping``) that only resolve when each
sub-package directory is itself on ``sys.path``. Merging into one module removes
that fragility; the maths is unchanged. See ``README.md`` §"Deviations".

Paper: Schrodinger Bass Bridge for Time Series, arXiv:2604.07159.
"""

import copy
import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ── Model ────────────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps, embedding_dim=128):
    """Sinusoidal time embedding, (..., 1) -> (..., embedding_dim)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, [0, 1])
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:x.size(1)]


class EncoderOnly(nn.Module):
    """Causal encoder-only transformer producing the past-trajectory context c_i."""

    def __init__(self, input_dim, d_model, nhead, n_layers, N, device):
        super().__init__()
        self.mask = nn.Transformer(batch_first=True).generate_square_subsequent_mask(N).to(device)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.past_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, y_past, training=False):
        y_proj = self.input_proj(y_past)          # (B, L, d_model)
        y_emb = y_proj + self.pe(y_proj)          # (B, L, d_model)

        if training:
            # Causal mask: position i may not attend to future steps.
            L = y_emb.size(1)
            return self.past_encoder(y_emb, mask=self.mask[:L, :L])
        return self.past_encoder(y_emb)[:, -1:]   # (B, 1, d_model)


class MLP(nn.Module):
    """Drift head s_theta(t, Y_t, c_i)."""

    def __init__(self, input_dim, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model

        self.t_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.cond_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, t, y, h):
        t_embed = self.t_encoder(get_timestep_embedding(t, self.d_model))
        y_embed = self.y_encoder(y)
        y_emb = torch.cat([t_embed, y_embed, h], dim=-1)
        return self.cond_fusion(y_emb)


class ScoreNN(nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, nhead, n_layers, L, device):
        super().__init__()
        self.tf_encoder = EncoderOnly(input_dim, d_model, nhead, n_layers, L, device)
        self.get_drift = MLP(input_dim, d_model, hidden_dim)

    def forward(self, t, y, y_past):
        h = self.tf_encoder(y_past)
        return self.get_drift(t, y, h)


class InverseMLP(nn.Module):
    """Inverse transport map, only used for the small-beta branch (beta < 100)."""

    def __init__(self, input_dim, d_model, t_model):
        super().__init__()
        self.d_model = d_model

        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_model),
            nn.LayerNorm(t_model),
            nn.GELU(),
            nn.Linear(t_model, t_model),
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model + t_model, min(d_model, t_model)),
            nn.SiLU(),
            nn.Linear(min(d_model, t_model), input_dim),
        )

    def forward(self, t, y):
        t_embed = self.t_encoder(t)
        y_embed = self.y_encoder(y)
        return self.decoder(torch.cat([t_embed, y_embed], dim=-1))


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch += 1
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = self.current_epoch
        elif score + self.delta >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = self.current_epoch
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def clean_memory(device):
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()


# ── Training ─────────────────────────────────────────────────────────────────

def get_loss(model, y_0, y_T, T, eps=None, t=None, safe_t=1e-2):
    """Score-matching loss (4.1) on Brownian-bridge samples (4.2)."""
    B, L, d = y_T.shape
    h_n = model.tf_encoder(y_0, training=True)
    if eps is None:
        eps = torch.randn_like(y_T, device=y_0.device)
        t = torch.FloatTensor(B, L, 1).uniform_(0, T - safe_t).to(y_0.device)

    sigma = torch.sqrt(t * (1 - t / T))
    mu_t = (1 - t / T) * y_0 + t / T * y_T
    y_t = mu_t + sigma * eps

    score_pred = model.get_drift(t, y_t, h_n)
    score_target = (y_T - y_t) / (T - t)
    return ((score_pred - score_target) ** 2).sum(dim=-1).mean()


def training_sbbts_dsbm(X, model, T, beta, K, n_epochs=100, batch_size=32, patience=10,
                        delta=1e-3, safe_t=1e-2, lr=1e-3, verbose=True, log_every=10):
    """Algorithm 1: iterative transport-map refinement (large-beta branch).

    Returns
    -------
    model    : trained ScoreNN (best-val weights per outer iteration)
    y_0      : (d,) initial condition for generation
    history  : list of dicts {k, epoch, train_loss, val_loss}
    """
    device = X.device
    optimizer = optim.Adam(model.parameters(), lr=lr)

    M, L, d = X.shape
    x_0 = X[:, :-1]   # (M, L-1, d)
    x_T = X[:, 1:]    # (M, L-1, d)

    size = int(M * 0.8)
    train_loader = DataLoader(TensorDataset(x_0[:size], x_T[:size]),
                              batch_size=batch_size, shuffle=True)

    history = []

    for k in range(K):
        if verbose:
            print(f"\nTraining s^{k + 1}:", flush=True)

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        curr_epoch = min(n_epochs, max(1000, int(n_epochs * np.exp(-0.2 * k))))

        # Frozen validation noise so the val loss is comparable across epochs.
        eps = torch.randn_like(x_T[size:]).to(device)
        t = torch.FloatTensor(len(x_T[size:]), L - 1, 1).uniform_(0, T - safe_t).to(device)
        val_loader = DataLoader(TensorDataset(x_0[size:], x_T[size:], eps, t),
                                batch_size=batch_size, shuffle=True)

        for epoch in range(curr_epoch):
            total_loss = 0.0
            model.train()

            for x_0_, x_T_ in train_loader:
                if k == 0:
                    y_0_, y_T_ = x_0_.clone(), x_T_.clone()
                else:
                    h_n = model.tf_encoder(x_0_, training=True)
                    t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                    y_0_ = x_0_ - 1 / beta * model.get_drift(t_0, x_0_, h_n)
                    t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                    y_T_ = x_T_ - 1 / beta * model.get_drift(t_N, x_T_, h_n)

                loss = get_loss(model, y_0_, y_T_, T, safe_t=safe_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for x_0_, x_T_, eps_, t_ in val_loader:
                    if k == 0:
                        y_0_, y_T_ = x_0_.clone(), x_T_.clone()
                    else:
                        h_n = model.tf_encoder(x_0_, training=True)
                        t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                        y_0_ = x_0_ - 1 / beta * model.get_drift(t_0, x_0_, h_n)
                        t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                        y_T_ = x_T_ - 1 / beta * model.get_drift(t_N, x_T_, h_n)

                    total_loss += get_loss(model, y_0_, y_T_, T, eps_, t_, safe_t=safe_t).item()
                val_loss = total_loss / len(val_loader)

            history.append(dict(k=k, epoch=epoch, train_loss=train_loss, val_loss=val_loss))

            if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
                print(f"Epoch [{epoch + 1}/{curr_epoch}] - "
                      f"Training Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                early_stopping.load_best_model(model)
                if verbose:
                    print(f"Early stopping at epoch {early_stopping.current_epoch}, "
                          f"best epoch = {early_stopping.best_epoch}, "
                          f"best val loss = {np.round(early_stopping.best_score, 4)}", flush=True)
                break

    with torch.no_grad():
        h_n = model.tf_encoder(x_0[:1, :1])
        t_0 = torch.zeros(1, 1, 1, device=device)
        y_0 = x_0[:1, :1] - 1 / beta * model.get_drift(t_0, x_0[:1, :1], h_n)

    return model, y_0.detach()[0, 0], history


# ── Generation ───────────────────────────────────────────────────────────────

def generate_dsbm_batch(N, X, model, y_0, N_pi, T, beta, M_simu, safe_t=1e-2):
    """Euler-Maruyama simulation of (2.4) + inverse transport, one batch.

    Returns a tensor of shape (M_simu, N, d) in the *scaled* space.
    """
    device = X.device
    d = X.shape[-1]

    time_step_Euler = T / N_pi
    v_time_step_Euler = torch.arange(0, T + 1e-9, time_step_Euler).to(device)

    num_brownian = (N * (len(v_time_step_Euler) - 1), M_simu, d)
    Brownian = torch.normal(0, 1, num_brownian).to(device)

    sbbts_sample = torch.zeros((M_simu, N + 1, d)).to(device)
    index_ = 0

    with torch.no_grad():
        T_tensor = torch.ones((M_simu, 1, 1), device=device) * (T - safe_t)
        Y = torch.ones((M_simu, d), device=device) * y_0.clone()
        past = torch.zeros((M_simu, N + 1, d)).to(device)
        past[:, 0] = y_0.clone()

        for i in range(N):
            Y_past = past[:, :i + 1].clone()
            h_n = model.tf_encoder(Y_past)          # (M_simu, 1, d_model)

            for k in range(len(v_time_step_Euler) - 1):
                timeprev = v_time_step_Euler[k]
                timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
                drift = model.get_drift(timeprev.repeat(M_simu, 1, 1), Y.unsqueeze(1), h_n)[:, 0]
                Y += drift * timestep + Brownian[index_] * torch.sqrt(timestep).to(device)
                index_ += 1

            X_ = Y + 1 / beta * model.get_drift(T_tensor, Y.unsqueeze(1), h_n)[:, 0]
            sbbts_sample[:, i + 1] = X_
            past[:, i + 1] = Y

            if i % 20 == 0:
                del X_, h_n, drift
                clean_memory(device)

    return sbbts_sample[:, 1:]


def generate_dsbm(N, X, model, y_0, N_pi, T, beta, M_simu, N_batch,
                  scale=1., safe_t=1e-2, exp=False, verbose=True):
    """Batched generation. Returns a NumPy array of shape (M_simu, N, d)."""
    base, extra = divmod(M_simu, N_batch)
    samples = []

    for j in range(N_batch):
        M_batch = base + (1 if j < extra else 0)
        if verbose:
            print(f"  inference batch {j + 1}/{N_batch} ({M_batch} paths)", flush=True)
        sample = generate_dsbm_batch(N, X, model, y_0, N_pi, T, beta, M_batch, safe_t=safe_t)
        sample = (sample * scale).cpu().detach().numpy()
        if exp:
            sample = np.exp(sample.cumsum(axis=1))
        samples.append(sample)

    return np.concatenate(samples, axis=0)
