"""TimeDiT backbone — a Diffusion Transformer (DiT) adapted for time series.

From-scratch faithful reimplementation of the model in

    Cao, Ye, Zhang, Liu. "TimeDiT: General-purpose Diffusion Transformers for
    Time Series Foundation Model." arXiv:2409.02322v1 (Sept 2024).

There is **no official TimeDiT code release**. The paper states (Appendix C) that
the codebase is "modified from https://github.com/facebookresearch/DiT" (Peebles &
Xie, 2022). This file therefore reimplements the DiT-S architecture (adaLN-Zero
conditioning, sinusoidal timestep embedding, transformer blocks) with the two
time-series–specific changes TimeDiT makes:

  1. Tokenisation over time: each of the L time steps becomes one token; the K
     channels are the per-token feature dimension (linearly projected to d_model).
  2. A Time Series Mask Unit that concatenates the (partial) observation and its
     binary mask to the noised target, so one model serves generation / imputation
     / forecasting.  For the *synthetic-generation* task (Table 6) the mask is the
     reconstruction mask M^Rec = 0 → the whole window is generated unconditionally,
     conditioned only on the diffusion timestep.

Architecture (DiT-S, paper Table 9 "S"): d_model=384, depth=12, heads=6.
"""
import math
import numpy as np
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(dim, length):
    """Standard 1D sin-cos positional embedding, shape (length, dim)."""
    pos = np.arange(length, dtype=np.float64)[:, None]
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos * omega[None, :]
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps into a d-dim vector (sinusoidal + MLP)."""

    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class DiTBlock(nn.Module):
    """A DiT transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class TimeDiT(nn.Module):
    """DiT-S for time series with a Time Series Mask Unit.

    Input to forward():
      x    : (B, L, C) noised target at diffusion step t
      t    : (B,)      diffusion timesteps
      obs  : (B, L, C) observed values (0 where unobserved)  — optional
      mask : (B, L, C) observation mask (1 = observed) — optional; None → M^Rec=0

    Output: (B, L, C) if learn_sigma=False, else (B, L, 2C) = [eps, v].
    """

    def __init__(self, seq_len, in_channels, hidden_size=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, learn_sigma=True):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        # Mask Unit: token features = [noised target | observation | mask]
        self.x_embedder = nn.Linear(in_channels * 3, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.register_buffer(
            "pos_embed", get_1d_sincos_pos_embed(hidden_size, seq_len).unsqueeze(0)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # zero-out adaLN modulation (adaLN-Zero) and final layer
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, obs=None, mask=None):
        if obs is None:
            obs = torch.zeros_like(x)
        if mask is None:
            mask = torch.zeros_like(x)
        tokens = torch.cat([x, obs, mask], dim=-1)          # (B, L, 3C)
        h = self.x_embedder(tokens) + self.pos_embed        # (B, L, d)
        c = self.t_embedder(t)                              # (B, d)
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)                       # (B, L, out_channels)


def build_timedit(seq_len, in_channels, model_size="S", learn_sigma=True):
    cfg = {
        "S": dict(hidden_size=384, depth=12, num_heads=6),
        "B": dict(hidden_size=768, depth=12, num_heads=12),
        "L": dict(hidden_size=1024, depth=24, num_heads=16),
    }[model_size]
    return TimeDiT(seq_len, in_channels, learn_sigma=learn_sigma, **cfg)
