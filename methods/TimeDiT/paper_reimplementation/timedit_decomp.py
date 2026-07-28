"""TimeDiT-Decomp — DiT backbone + Diffusion-TS-style trend/seasonality x0 head.

This is an *improvement variant* of the from-scratch TimeDiT reproduction, built to
close the discriminative-score gap that the plain eps-prediction model leaves open.

Diagnosis (empirical, from the plain TimeDiT run): shape / lag-1 statistics of the
generated Sine samples matched the real data almost exactly, but the samples were
~9x too *jagged* (mean abs 2nd-difference 0.0102 vs 0.0011 real).  A DDIM ablation
proved the jaggedness lives in the trained eps-field, not the sampler.  The fix is a
model-side inductive bias toward smooth signals.

Following Diffusion-TS (Yuan & Qiao, ICLR 2024) we keep the DiT-S transformer
backbone but replace the output head: instead of a free per-token linear readout,
the model predicts x0 as the sum of

  * a **polynomial trend**   : global low-degree (deg P) polynomial in normalised
                               time, coefficients regressed from the pooled hidden
                               state  ->  smooth, captures drift / level.
  * a **Fourier seasonality**: per-token readout passed through a low-pass filter
                               (keep only the lowest `n_freq` rFFT bins) -> smooth,
                               band-limited, captures the (low-frequency) oscillation.

Because both components are band-limited by construction, the head *cannot* emit the
high-frequency jitter that inflated the 2nd-difference.  This is the whole point.

Backbone (DiT-S, paper Table 9 "S"): d_model=384, depth=12, heads=6.  Unconditional
synthetic generation => Time Series Mask Unit reduces to M^Rec=0, so we drop the
[x|obs|mask] concatenation and embed x directly.

The model predicts **x0** (not eps); no learned variance.  Sampling uses fixed
(posterior) variance.  See run_decomp.py for the diffusion process.
"""
import numpy as np
import torch
import torch.nn as nn

from timedit_model import (
    DiTBlock,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
    modulate,
)


class DecompHead(nn.Module):
    """Predict x0 = polynomial trend + low-pass Fourier seasonality.

    h : (B, L, d) transformer output ; c : (B, d) conditioning (timestep embed).
    Returns x0_hat : (B, L, C).
    """

    def __init__(self, hidden_size, seq_len, out_channels, n_poly=4, n_freq=8):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.n_poly = n_poly
        self.n_freq = n_freq
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # trend: pooled hidden -> (P+1) polynomial coefficients per channel
        self.trend_proj = nn.Linear(hidden_size, out_channels * (n_poly + 1), bias=True)
        # seasonality: per-token hidden -> raw sequence (then low-pass in forward)
        self.season_proj = nn.Linear(hidden_size, out_channels, bias=True)

        # Vandermonde matrix V[l, p] = (l/(L-1))**p  on normalised time in [0, 1]
        t = np.linspace(0.0, 1.0, seq_len, dtype=np.float64)
        V = np.stack([t ** p for p in range(n_poly + 1)], axis=-1)  # (L, P+1)
        self.register_buffer("vander", torch.tensor(V, dtype=torch.float32))

        # low-pass mask over rFFT bins: keep bins 1..n_freq (drop DC -> trend owns it)
        n_bins = seq_len // 2 + 1
        lp = np.zeros(n_bins, dtype=np.float32)
        hi = min(n_freq + 1, n_bins)
        lp[1:hi] = 1.0
        self.register_buffer("lowpass", torch.tensor(lp).view(1, n_bins, 1))

    def forward(self, h, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        h = modulate(self.norm(h), shift, scale)          # (B, L, d)

        # --- trend (smooth, global polynomial) ---
        pooled = h.mean(dim=1)                             # (B, d)
        coeffs = self.trend_proj(pooled).view(
            -1, self.out_channels, self.n_poly + 1
        )                                                  # (B, C, P+1)
        trend = torch.einsum("lp,bcp->blc", self.vander, coeffs)  # (B, L, C)

        # --- seasonality (band-limited, low-pass Fourier) ---
        s_raw = self.season_proj(h)                        # (B, L, C)
        S = torch.fft.rfft(s_raw, dim=1)                   # (B, n_bins, C) complex
        S = S * self.lowpass                               # keep low frequencies only
        season = torch.fft.irfft(S, n=self.seq_len, dim=1)  # (B, L, C)

        return trend + season


class TimeDiTDecomp(nn.Module):
    """DiT-S backbone with a trend/seasonality x0 head (predicts x0, no learn_sigma).

    forward(x, t) -> x0_hat (B, L, C).  `obs`/`mask` accepted but ignored
    (unconditional generation, M^Rec = 0) so the call signature stays compatible
    with the samplers / metric code.
    """

    def __init__(self, seq_len, in_channels, hidden_size=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, n_poly=4, n_freq=8):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.register_buffer(
            "pos_embed", get_1d_sincos_pos_embed(hidden_size, seq_len).unsqueeze(0)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.head = DecompHead(hidden_size, seq_len, in_channels,
                               n_poly=n_poly, n_freq=n_freq)
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
        # adaLN-Zero on every block
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # zero the head's adaLN + projections so x0_hat starts near 0
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.trend_proj.weight, 0)
        nn.init.constant_(self.head.trend_proj.bias, 0)
        nn.init.constant_(self.head.season_proj.weight, 0)
        nn.init.constant_(self.head.season_proj.bias, 0)

    def forward(self, x, t, obs=None, mask=None):
        h = self.x_embedder(x) + self.pos_embed            # (B, L, d)
        c = self.t_embedder(t)                             # (B, d)
        for block in self.blocks:
            h = block(h, c)
        return self.head(h, c)                             # (B, L, C) = x0_hat


def build_timedit_decomp(seq_len, in_channels, model_size="S", n_poly=4, n_freq=8):
    cfg = {
        "S": dict(hidden_size=384, depth=12, num_heads=6),
        "B": dict(hidden_size=768, depth=12, num_heads=12),
        "L": dict(hidden_size=1024, depth=24, num_heads=16),
    }[model_size]
    return TimeDiTDecomp(seq_len, in_channels, n_poly=n_poly, n_freq=n_freq, **cfg)
