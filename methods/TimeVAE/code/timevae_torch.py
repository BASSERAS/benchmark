"""
Faithful PyTorch port of TimeVAE (Desai, Freeman, Beaver, Wang — ICLR 2022 submission,
arXiv:2111.08095v3).  Official TensorFlow/Keras code: github.com/abudesai/timeVAE
(mirrored verbatim under ``code/reference/``).

Why a PyTorch port?  The released code is ``tensorflow==2.16.1``; TensorFlow is
GPU-broken on this machine (CUDA-13 driver -> cuDNN INTERNAL_ERROR, GPUs=[]), so it
would run CPU-only.  This module reimplements the *exact* architecture and loss of
``code/reference/src/vae/{vae_base.py, timevae.py}`` in PyTorch so training runs on
the A100s.  Every layer, activation, and the custom VAE loss are matched line-for-line
against the reference (see the class docstrings for the source anchors).

Default variant is **TimeVAE-Base**: level model + residual connection, with the
interpretable ``trend_poly`` / ``custom_seas`` blocks disabled — exactly the
``config/hyperparameters.yaml`` ``timeVAE`` preset (latent_dim=8, hidden=[50,100,200],
reconstruction_wt=3.0, batch_size=16, use_residual_conn=true, trend_poly=0,
custom_seas=null).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# TF-style "same" padding helpers (PyTorch Conv1d has no 'same' with stride > 1)
# ----------------------------------------------------------------------------------
def _tf_same_pad_total(length: int, kernel: int, stride: int) -> int:
    """Total padding TensorFlow adds for padding='same' (output = ceil(L/stride))."""
    if length % stride == 0:
        return max(kernel - stride, 0)
    return max(kernel - (length % stride), 0)


def _same_out_len(length: int, stride: int) -> int:
    return math.ceil(length / stride)


class SameConv1d(nn.Module):
    """Conv1d matching Keras Conv1D(padding='same').  Input/-output are (N, C, L).

    TF puts the extra pad on the right: pad_left = total//2, pad_right = total - total//2.
    Mirrors ``timevae.py`` encoder ``Conv1D(kernel_size=3, strides=2, padding='same')``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 2):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = _tf_same_pad_total(x.shape[-1], self.kernel, self.stride)
        x = F.pad(x, (total // 2, total - total // 2))
        return self.conv(x)


class SameConvTranspose1d(nn.Module):
    """ConvTranspose1d matching Keras Conv1DTranspose(padding='same', strides=2).

    For kernel=3, stride=2 the TF 'same' output length is exactly 2*L_in; the PyTorch
    equivalent is padding=1, output_padding=1.  Mirrors ``timevae.py`` decoder
    ``Conv1DTranspose(kernel_size=3, strides=2, padding='same')``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 2):
        super().__init__()
        assert kernel == 3 and stride == 2, "port matches the reference k=3,s=2 only"
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


# ----------------------------------------------------------------------------------
# Interpretable decoder blocks (TimeVAE-full).  Off by default (TimeVAE-Base).
# ----------------------------------------------------------------------------------
class TrendLayer(nn.Module):
    """Port of ``timevae.py::TrendLayer`` — polynomial trend basis."""

    def __init__(self, feat_dim: int, trend_poly: int, seq_len: int):
        super().__init__()
        self.feat_dim, self.trend_poly, self.seq_len = feat_dim, trend_poly, seq_len
        self.dense1 = nn.LazyLinear(feat_dim * trend_poly)
        self.dense2 = nn.Linear(feat_dim * trend_poly, feat_dim * trend_poly)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = F.relu(self.dense1(z))
        p = self.dense2(p)
        p = p.view(-1, self.feat_dim, self.trend_poly)  # (N, D, P)
        lin = torch.arange(0, self.seq_len, device=z.device, dtype=torch.float32) / self.seq_len
        poly = torch.stack([lin ** float(i + 1) for i in range(self.trend_poly)], dim=0)  # (P, T)
        trend = torch.matmul(p, poly)          # (N, D, T)
        return trend.transpose(1, 2)           # (N, T, D)


class SeasonalLayer(nn.Module):
    """Port of ``timevae.py::SeasonalLayer`` — custom seasonal basis."""

    def __init__(self, feat_dim: int, seq_len: int, custom_seas: Sequence[Tuple[int, int]]):
        super().__init__()
        self.feat_dim, self.seq_len = feat_dim, seq_len
        self.custom_seas = list(custom_seas)
        self.dense = nn.ModuleList(
            [nn.LazyLinear(feat_dim * num_seasons) for num_seasons, _ in self.custom_seas]
        )

    def _season_indexes_over_seq(self, num_seasons: int, len_per_season: int, device) -> torch.Tensor:
        idx = torch.arange(num_seasons, device=device).view(-1, 1) + torch.zeros(
            num_seasons, len_per_season, dtype=torch.long, device=device
        )
        idx = idx.reshape(-1)
        reps = self.seq_len // len_per_season + 1
        return idx.repeat(reps)[: self.seq_len]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        all_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            params = self.dense[i](z).view(N, self.feat_dim, num_seasons)  # (N, D, S)
            season_idx = self._season_indexes_over_seq(num_seasons, len_per_season, z.device)  # (T,)
            idx = season_idx.view(1, 1, -1).expand(N, self.feat_dim, self.seq_len)  # (N, D, T)
            vals = torch.gather(params, 2, idx)  # (N, D, T)
            all_vals.append(vals)
        out = torch.stack(all_vals, dim=-1).sum(dim=-1)  # (N, D, T)
        return out.transpose(1, 2)                       # (N, T, D)


# ----------------------------------------------------------------------------------
# Encoder / Decoder
# ----------------------------------------------------------------------------------
class Encoder(nn.Module):
    """Port of ``timevae.py::_get_encoder``.

    Conv1D stack (filters=hidden, k=3, s=2, ReLU, 'same') -> Flatten -> z_mean, z_log_var.
    """

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = feat_dim
        for h in hidden:
            layers.append(SameConv1d(in_ch, h, 3, 2))
            layers.append(nn.ReLU())
            in_ch = h
        self.convs = nn.Sequential(*layers)

        L = seq_len
        for _ in hidden:
            L = _same_out_len(L, 2)
        self.enc_last_L = L
        self.flat_dim = L * hidden[-1]          # matches TF Flatten of (N, L, C)
        self.hidden_last = hidden[-1]
        self.fc_mean = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)                   # (N, D, T)
        h = self.convs(h)                       # (N, C, L)
        h = h.transpose(1, 2).reshape(h.shape[0], -1)  # flatten as TF (N, L*C)
        return self.fc_mean(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Port of ``timevae.py::_get_decoder`` (level + optional trend/seasonal + residual)."""

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        hidden: Sequence[int],
        enc_last_L: int,
        flat_dim: int,
        use_residual_conn: bool = True,
        trend_poly: int = 0,
        custom_seas: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.seq_len, self.feat_dim, self.latent_dim = seq_len, feat_dim, latent_dim
        self.hidden = list(hidden)
        self.enc_last_L = enc_last_L
        self.use_residual_conn = use_residual_conn
        self.trend_poly = trend_poly
        self.custom_seas = list(custom_seas) if custom_seas else None

        # level_model
        self.level_fc1 = nn.Linear(latent_dim, feat_dim)
        self.level_fc2 = nn.Linear(feat_dim, feat_dim)

        if trend_poly and trend_poly > 0:
            self.trend = TrendLayer(feat_dim, trend_poly, seq_len)
        if self.custom_seas:
            self.season = SeasonalLayer(feat_dim, seq_len, self.custom_seas)

        if use_residual_conn:
            self.res_fc = nn.Linear(latent_dim, flat_dim)
            deconvs: List[nn.Module] = []
            in_ch = hidden[-1]
            for h in reversed(hidden[:-1]):
                deconvs.append(SameConvTranspose1d(in_ch, h, 3, 2))
                deconvs.append(nn.ReLU())
                in_ch = h
            # final de-conv to feat_dim (ReLU, as in reference)
            deconvs.append(SameConvTranspose1d(in_ch, feat_dim, 3, 2))
            deconvs.append(nn.ReLU())
            self.deconvs = nn.Sequential(*deconvs)
            # length after len(hidden) transpose-doublings
            self.res_L = enc_last_L * (2 ** len(hidden))
            self.res_out = nn.Linear(feat_dim * self.res_L, seq_len * feat_dim)

    def _level(self, z: torch.Tensor) -> torch.Tensor:
        p = F.relu(self.level_fc1(z))
        p = self.level_fc2(p)                       # (N, D)
        p = p.view(-1, 1, self.feat_dim)            # (N, 1, D)
        ones = torch.ones(1, self.seq_len, 1, device=z.device)
        return p * ones                             # (N, T, D)

    def _residual(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.res_fc(z))                          # (N, flat_dim)
        x = x.view(-1, self.enc_last_L, self.hidden[-1])    # (N, L, C) as TF reshape
        x = x.transpose(1, 2)                               # (N, C, L)
        x = self.deconvs(x)                                 # (N, feat_dim, res_L)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)       # flatten as TF (N, res_L*feat_dim)
        x = self.res_out(x)                                 # (N, T*D)
        return x.view(-1, self.seq_len, self.feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self._level(z)
        if self.trend_poly and self.trend_poly > 0:
            out = out + self.trend(z)
        if self.custom_seas:
            out = out + self.season(z)
        if self.use_residual_conn:
            out = out + self._residual(z)
        return out


# ----------------------------------------------------------------------------------
# TimeVAE model
# ----------------------------------------------------------------------------------
class TimeVAE(nn.Module):
    """PyTorch TimeVAE.  Architecture + loss ported from the reference TF code.

    Args match ``config/hyperparameters.yaml``'s ``timeVAE`` preset.
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int = 8,
        hidden_layer_sizes: Sequence[int] = (50, 100, 200),
        reconstruction_wt: float = 3.0,
        trend_poly: int = 0,
        custom_seas: Optional[Sequence[Tuple[int, int]]] = None,
        use_residual_conn: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.reconstruction_wt = reconstruction_wt
        self.trend_poly = trend_poly
        self.custom_seas = list(custom_seas) if custom_seas else None
        self.use_residual_conn = use_residual_conn

        self.encoder = Encoder(seq_len, feat_dim, latent_dim, self.hidden_layer_sizes)
        self.decoder = Decoder(
            seq_len,
            feat_dim,
            latent_dim,
            self.hidden_layer_sizes,
            self.encoder.enc_last_L,
            self.encoder.flat_dim,
            use_residual_conn=use_residual_conn,
            trend_poly=trend_poly,
            custom_seas=self.custom_seas,
        )

    # -- reparameterization (Sampling layer) --
    @staticmethod
    def _sample(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps

    def forward(self, x: torch.Tensor):
        z_mean, z_log_var = self.encoder(x)
        z = self._sample(z_mean, z_log_var)
        x_recons = self.decoder(z)
        return x_recons, z_mean, z_log_var, z

    # -- loss: exact port of vae_base.py train_step --
    def loss(self, x: torch.Tensor):
        x_recons, z_mean, z_log_var, _ = self.forward(x)
        # reconstruction: overall SSE + SSE of feature-axis means (axis=2 in TF)
        recon = torch.sum((x - x_recons) ** 2)
        recon = recon + torch.sum((x.mean(dim=2) - x_recons.mean(dim=2)) ** 2)
        # KL, summed over latent then over batch
        kl = -0.5 * (1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var))
        kl = torch.sum(torch.sum(kl, dim=1))
        total = self.reconstruction_wt * recon + kl
        return total, recon, kl

    # -- prior generation: decoder(randn(N, latent)) (vae_base.get_prior_samples) --
    @torch.no_grad()
    def generate(self, num_samples: int, device=None, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        device = device or next(self.parameters()).device
        outs = []
        remaining = num_samples
        while remaining > 0:
            n = min(batch_size, remaining)
            z = torch.randn(n, self.latent_dim, device=device)
            outs.append(self.decoder(z).cpu().numpy())
            remaining -= n
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior reconstruction using z_mean (vae_base.call)."""
        self.eval()
        z_mean, _ = self.encoder(x)
        return self.decoder(z_mean)


# ----------------------------------------------------------------------------------
# MinMax scaler (exact port of data_utils.MinMaxScaler)
# ----------------------------------------------------------------------------------
class MinMaxScaler:
    """Per-(t,feature) min-max to ~[0,1]; identical to reference data_utils.MinMaxScaler."""

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.mini = data.min(axis=0)
        self.range = data.max(axis=0) - self.mini
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mini) / (self.range + 1e-7)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data.copy()
        data *= self.range
        data += self.mini
        return data


# ----------------------------------------------------------------------------------
# Training loop (ports vae_base.fit_on_data: Adam, EarlyStopping, ReduceLROnPlateau)
# ----------------------------------------------------------------------------------
def train_timevae(
    model: TimeVAE,
    train_data: np.ndarray,
    max_epochs: int = 1000,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda",
    es_patience: int = 50,
    es_min_delta: float = 1e-2,
    rlr_patience: int = 30,
    rlr_factor: float = 0.5,
    verbose: int = 1,
    log_every: int = 1,
    seed: int = 0,
):
    """Train with the reference schedule.

    EarlyStopping(monitor=total_loss, min_delta=1e-2, patience=50, mode=min) and
    ReduceLROnPlateau(factor=0.5, patience=30) are matched to vae_base.fit_on_data.
    Monitored quantity is the epoch-mean total loss (Keras' Mean tracker).
    Returns a per-epoch history list of dicts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=rlr_factor, patience=rlr_patience
    )

    X = torch.as_tensor(train_data, dtype=torch.float32)
    n = X.shape[0]
    history = []
    best = math.inf
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        tot = rec = klt = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(device)
            opt.zero_grad()
            total, recon, kl = model.loss(xb)
            total.backward()
            opt.step()
            tot += float(total.item())
            rec += float(recon.item())
            klt += float(kl.item())
            nb += 1
        # epoch-mean (Keras Mean metric averages the per-batch scalars)
        tot_m, rec_m, kl_m = tot / nb, rec / nb, klt / nb
        sched.step(tot_m)
        cur_lr = opt.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "total_loss": tot_m, "reconstruction_loss": rec_m,
             "kl_loss": kl_m, "lr": cur_lr}
        )
        if verbose and (epoch % log_every == 0):
            print(f"[epoch {epoch:4d}] total={tot_m:.4f} recon={rec_m:.4f} "
                  f"kl={kl_m:.4f} lr={cur_lr:.2e}", flush=True)

        # EarlyStopping (min mode, min_delta): improvement must exceed min_delta
        if best - tot_m > es_min_delta:
            best = tot_m
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= es_patience:
                if verbose:
                    print(f"[early stop] epoch {epoch}, best total={best:.4f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
