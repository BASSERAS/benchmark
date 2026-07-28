"""TimeMoDE backbone — DiT + Mixture of Domain Experts (MoDE).

From-scratch faithful reimplementation of the model in

    Yao, Zheng, Zuo, Zhang. "Towards a Unified Generative Model for Scarce Time
    Series with Domain Experts." ICML 2026 (PMLR 306). arXiv:2606.15172v1.

There is **no official TimeMoDE code release**. The paper (Appendix B / Fig. 2)
states the architecture is a Diffusion Transformer (DiT, Peebles & Xie 2022) whose
MLP sub-blocks are replaced by a Mixture of Domain Experts. This file reimplements
the DiT-MoDE stack with the components the paper specifies:

  - DiT-MoDE block (Eq 5): z = z + gate1 * MHSA(AdaLN(z, c));
                           z = z + gate2 * MoDE(AdaLN(z, c))   (adaLN-Zero).
  - Domain Prompt (Eq 6): DP = Avg(Linear(Conv(x)) + PE) from a domain exemplar.
  - Time-series prototypes P in R^{K x h x d} (Eq 7/9 orthonormality losses).
  - Router (Eq 10): s_i ∝ ||Proto_i · DP||^2 + (W z)_i + tau, normalised over experts.
  - Gating (Eq 12): G(s) = Softmax(Top-2(s)); shared expert E_0 always on (Eq 13).
  - GLU expert (Eq 11): stage-aware AdaLN -> W_down(SiLU(W_gate z) ⊙ W_up z) + z.
  - Load-balancing aux loss (Eq 14, Switch-Transformer form).

Config (paper Table 8, "From Scratch" uses the same architecture, no pretraining):
  hidden 256, heads 4, depth 6, experts K=8, activated top-2, expansion n=4,
  diffusion steps 250. Prototype basis h and temperature tau are set below.

For the univariate StarLightCurves generation benchmark the model is trained
UNCONDITIONALLY (c = timestep embedding only); class conditioning is optional.
Reproduction choices that the paper leaves unspecified are marked "REPRO:".
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(dim, length):
    pos = np.arange(length, dtype=np.float64)[:, None]
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos * omega[None, :]
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)


class TimestepEmbedder(nn.Module):
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
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class DomainPrompt(nn.Module):
    """DP = Avg(Linear(Conv(x)) + PE)  (Eq 6).

    Conv1d over time captures (multivariate) correlations; temporal average
    pooling compresses a domain exemplar into a compact d-dim domain identity.
    Input x: (B, L, C) clean exemplar series -> DP: (B, d).
    """

    def __init__(self, in_channels, hidden_size, seq_len, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, hidden_size, kernel_size,
                              padding=kernel_size // 2)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.register_buffer("pe", get_1d_sincos_pos_embed(hidden_size, seq_len).unsqueeze(0))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, L, d)
        h = self.linear(h) + self.pe                        # (B, L, d)
        return h.mean(dim=1)                                # (B, d)


class GLUExpert(nn.Module):
    """Stage-aware AdaLN -> SwiGLU with residual (Eq 11).

    z_norm = AdaLN(z, c);  z' = W_down(SiLU(W_gate z_norm) ⊙ W_up z_norm) + z.
    """

    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        inner = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.w_gate = nn.Linear(hidden_size, inner, bias=False)
        self.w_up = nn.Linear(hidden_size, inner, bias=False)
        self.w_down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, z, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        h = modulate(self.norm(z), shift, scale)
        h = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return h + z


class MoDEBlock(nn.Module):
    """Mixture of Domain Experts (Eq 10/12/13) with prototype router.

    K routed experts (top-2) + one always-on shared expert E_0.
    Returns (output tokens, aux_load_balance_loss).
    """

    def __init__(self, hidden_size, num_experts=8, top_k=2, expansion=4,
                 n_basis=4, tau=1.0):
        super().__init__()
        self.K = num_experts
        self.top_k = top_k
        self.tau = tau
        self.experts = nn.ModuleList(
            [GLUExpert(hidden_size, expansion) for _ in range(num_experts)]
        )
        self.shared = GLUExpert(hidden_size, expansion)
        # prototypes P in R^{K x h x d}; W in R^{K x d} calibrates on the token.
        self.prototypes = nn.Parameter(torch.randn(num_experts, n_basis, hidden_size))
        self.w_route = nn.Linear(hidden_size, num_experts)

    def normalized_prototypes(self):
        return F.normalize(self.prototypes, dim=-1)  # unit-norm basis (||P_i^j||=1)

    def proto_loss(self):
        """L_proto = ||P Pᵀ − I||²_F over all K·h basis (Eq 7 diag + Eq 9 off-diag)."""
        P = self.normalized_prototypes().reshape(self.K * self.prototypes.shape[1], -1)
        gram = P @ P.t()
        eye = torch.eye(gram.shape[0], device=gram.device)
        return ((gram - eye) ** 2).sum()

    def forward(self, z, dp, c):
        B, L, d = z.shape
        proto = self.normalized_prototypes()                       # (K, h, d)
        # matching degree ||Proto_i · DP||^2 per expert, per batch item (Eq 10 numerator)
        match = torch.einsum("khd,bd->bkh", proto, dp)             # (B, K, h)
        match = (match ** 2).sum(-1)                               # (B, K)
        logits = match.unsqueeze(1) + self.w_route(z) + self.tau   # (B, L, K)
        scores = logits / logits.sum(-1, keepdim=True).clamp_min(1e-9)

        top_val, top_idx = scores.topk(self.top_k, dim=-1)         # (B, L, k)
        gate = F.softmax(top_val, dim=-1)                          # (B, L, k)

        out = self.shared(z, c)                                    # E_0 always on (Eq 13)
        # dense compute of every expert, mask to top-k (K small -> simplest & correct)
        expert_out = torch.stack([e(z, c) for e in self.experts], dim=2)  # (B,L,K,d)
        full_gate = torch.zeros(B, L, self.K, device=z.device)
        full_gate.scatter_(-1, top_idx, gate)
        out = out + (full_gate.unsqueeze(-1) * expert_out).sum(dim=2)

        # load-balancing aux loss (Eq 14, Switch form): K * sum_i f_i * P_i
        with torch.no_grad():
            sel = torch.zeros(B, L, self.K, device=z.device)
            sel.scatter_(-1, top_idx, 1.0)
            f_i = sel.reshape(-1, self.K).mean(0)                  # fraction routed
        p_i = scores.reshape(-1, self.K).mean(0)                   # mean prob
        aux = self.K * (f_i * p_i).sum()
        return out, aux


class DiTMoDEBlock(nn.Module):
    """DiT block with adaLN-Zero; MLP replaced by MoDE (Eq 5)."""

    def __init__(self, hidden_size, num_heads, num_experts, top_k, expansion,
                 n_basis, tau):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mode = MoDEBlock(hidden_size, num_experts, top_k, expansion, n_basis, tau)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, dp):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        mode_out, aux = self.mode(modulate(self.norm2(x), shift2, scale2), dp, c)
        x = x + gate2.unsqueeze(1) * mode_out
        return x, aux


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class TimeMoDE(nn.Module):
    """DiT-MoDE for time series.

    forward(x, t, dp_exemplar, y=None) -> (eps_pred, aux_loss, proto_loss)
      x : (B, L, C) noised target;  t : (B,) diffusion steps
      dp_exemplar : (B, L, C) clean domain exemplars (build the Domain Prompt)
      y : (B,) optional class labels for conditioning
    """

    def __init__(self, seq_len, in_channels, hidden_size=256, depth=6,
                 num_heads=4, num_experts=8, top_k=2, expansion=4, n_basis=4,
                 tau=1.0, patch_size=1, num_classes=0, learn_sigma=False):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.n_tokens = seq_len // patch_size
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * (2 if learn_sigma else 1)

        self.x_embedder = nn.Linear(patch_size * in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_classes = num_classes
        if num_classes > 0:
            self.y_embedder = nn.Embedding(num_classes, hidden_size)
        self.dp = DomainPrompt(in_channels, hidden_size, seq_len)
        self.register_buffer(
            "pos_embed", get_1d_sincos_pos_embed(hidden_size, self.n_tokens).unsqueeze(0)
        )
        self.blocks = nn.ModuleList([
            DiTMoDEBlock(hidden_size, num_heads, num_experts, top_k, expansion,
                        n_basis, tau) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x):
        B, L, C = x.shape
        return x.reshape(B, self.n_tokens, self.patch_size * C)

    def unpatchify(self, x):
        B = x.shape[0]
        return x.reshape(B, self.seq_len, self.out_channels)

    def forward(self, x, t, dp_exemplar=None, y=None):
        if dp_exemplar is None:
            dp_exemplar = x
        dp = self.dp(dp_exemplar)                            # (B, d)
        h = self.x_embedder(self.patchify(x)) + self.pos_embed
        c = self.t_embedder(t)
        if self.num_classes > 0 and y is not None:
            c = c + self.y_embedder(y)
        aux_total = 0.0
        for block in self.blocks:
            h, aux = block(h, c, dp)
            aux_total = aux_total + aux
        out = self.unpatchify(self.final_layer(h, c))
        proto = sum(b.mode.proto_loss() for b in self.blocks)
        return out, aux_total, proto


def build_timemode(seq_len, in_channels, learn_sigma=False, num_classes=0,
                   patch_size=1, **overrides):
    cfg = dict(hidden_size=256, depth=6, num_heads=4, num_experts=8, top_k=2,
               expansion=4, n_basis=4, tau=1.0)
    cfg.update(overrides)
    return TimeMoDE(seq_len, in_channels, learn_sigma=learn_sigma,
                    num_classes=num_classes, patch_size=patch_size, **cfg)
