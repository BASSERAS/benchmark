from __future__ import annotations

import torch
import torch.nn as nn

from deep_mkv_gen_path_dt.networks.adjoint import AdjointMoments


def _zero_init_final_linear(module: nn.Module) -> None:
    final = module[-1] if isinstance(module, nn.Sequential) else None
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class CausalTransformerAdjointNetwork(nn.Module):
    """Causal prefix Transformer with the same contract as the GRU adjoint net.

    The network sees only the generated prefix available at a decision date.
    ``forward_step`` stores that prefix and recomputes the short causal sequence;
    this simple implementation is intended for the exact-tree diagnostic, where
    the horizon is small.  It deliberately does not use ridge targets or exact
    tree conditional expectations during training.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int | None = None,
        noise_adjoint_dim: int | None = None,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        max_steps: int = 512,
        dropout: float = 0.0,
        input_mode: str = "level",
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be positive")
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive")
        input_mode = str(input_mode).strip().lower()
        if input_mode not in {"level", "increment"}:
            raise ValueError("input_mode must be level or increment")
        self.state_dim = int(state_dim)
        self.adjoint_dim = int(state_dim if adjoint_dim is None else adjoint_dim)
        self.noise_adjoint_dim = int(
            self.adjoint_dim if noise_adjoint_dim is None else noise_adjoint_dim
        )
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.input_mode = input_mode
        self.input_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(self.max_steps, self.hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(
                2 * self.hidden_dim if feedforward_dim is None else feedforward_dim
            ),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.expected_adjoint_next_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.adjoint_dim),
        )
        self.expected_adjoint_noise_next_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.noise_adjoint_dim),
        )
        _zero_init_final_linear(self.expected_adjoint_next_head)
        _zero_init_final_linear(self.expected_adjoint_noise_next_head)

    def _tokens_from_levels(self, levels: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "level":
            return levels
        increments = levels[:, 1:, :] - levels[:, :-1, :]
        return torch.cat((torch.zeros_like(levels[:, :1, :]), increments), dim=1)

    def _encode_levels(self, levels: torch.Tensor) -> torch.Tensor:
        if levels.ndim != 3 or int(levels.shape[-1]) != self.state_dim:
            raise ValueError("levels must have shape (B, L, state_dim)")
        length = int(levels.shape[1])
        if not 1 <= length <= self.max_steps:
            raise ValueError("prefix length exceeds max_steps")
        tokens = self._tokens_from_levels(levels)
        encoded = self.input_projection(tokens)
        encoded = encoded + self.position_embedding[:length].unsqueeze(0).to(encoded)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=encoded.device),
            diagonal=1,
        )
        return self.final_norm(self.encoder(encoded, mask=causal_mask))

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3 or int(x_path.shape[1]) < 2:
            raise ValueError("x_path must have shape (B, N + 1, state_dim)")
        features = self._encode_levels(x_path[:, :-1, :])
        return AdjointMoments(
            expected_adjoint_next=self.expected_adjoint_next_head(features),
            expected_adjoint_noise_next=self.expected_adjoint_noise_next_head(features),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            levels = x_t.unsqueeze(1)
        else:
            if hidden.ndim != 3 or tuple(hidden.shape[::2]) != (
                int(x_t.shape[0]),
                self.state_dim,
            ):
                raise ValueError("hidden must contain the previous path prefix")
            levels = torch.cat((hidden, x_t.unsqueeze(1)), dim=1)
        features = self._encode_levels(levels)[:, -1, :]
        return (
            self.expected_adjoint_next_head(features),
            self.expected_adjoint_noise_next_head(features),
            levels,
        )


__all__ = ["CausalTransformerAdjointNetwork"]


class _CachedCausalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, int(feedforward_dim)),
            nn.GELU(),
            nn.Linear(int(feedforward_dim), self.hidden_dim),
        )

    def _qkv(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = value.shape
        qkv = self.qkv(value).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def _merge(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(1, 2).contiguous().reshape(
            int(value.shape[0]), int(value.shape[2]), self.hidden_dim
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        q, k, v = self._qkv(normalized)
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        value = value + self.out(self._merge(attended))
        return value + self.feedforward(self.norm2(value))

    def forward_step(
        self,
        value: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if int(value.shape[1]) != 1:
            raise ValueError("cached Transformer step requires one token")
        normalized = self.norm1(value)
        q, k, v = self._qkv(normalized)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        value = value + self.out(self._merge(attended))
        value = value + self.feedforward(self.norm2(value))
        return value, (k, v)


class CachedCausalTransformerAdjointNetwork(nn.Module):
    """Causal Transformer with incremental key/value caches for long rollouts."""

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int | None = None,
        noise_adjoint_dim: int | None = None,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        max_steps: int = 512,
        input_mode: str = "level",
    ) -> None:
        super().__init__()
        input_mode = str(input_mode).strip().lower()
        if input_mode not in {"level", "increment"}:
            raise ValueError("input_mode must be level or increment")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive")
        self.state_dim = int(state_dim)
        self.adjoint_dim = int(state_dim if adjoint_dim is None else adjoint_dim)
        self.noise_adjoint_dim = int(
            self.adjoint_dim if noise_adjoint_dim is None else noise_adjoint_dim
        )
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.input_mode = input_mode
        self.input_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(self.max_steps, self.hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        ff_dim = int(
            2 * self.hidden_dim if feedforward_dim is None else feedforward_dim
        )
        self.blocks = nn.ModuleList(
            [
                _CachedCausalTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    feedforward_dim=ff_dim,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.expected_adjoint_next_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.adjoint_dim),
        )
        self.expected_adjoint_noise_next_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.noise_adjoint_dim),
        )
        _zero_init_final_linear(self.expected_adjoint_next_head)
        _zero_init_final_linear(self.expected_adjoint_noise_next_head)

    def _tokens(self, levels: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "level":
            return levels
        increments = levels[:, 1:, :] - levels[:, :-1, :]
        return torch.cat((torch.zeros_like(levels[:, :1, :]), increments), dim=1)

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3 or int(x_path.shape[-1]) != self.state_dim:
            raise ValueError("x_path must have shape (B, N + 1, state_dim)")
        levels = x_path[:, :-1, :]
        steps = int(levels.shape[1])
        if not 1 <= steps <= self.max_steps:
            raise ValueError("path length exceeds max_steps")
        value = self.input_projection(self._tokens(levels))
        value = value + self.position_embedding[:steps].unsqueeze(0).to(value)
        for block in self.blocks:
            value = block(value)
        value = self.final_norm(value)
        return AdjointMoments(
            expected_adjoint_next=self.expected_adjoint_next_head(value),
            expected_adjoint_noise_next=self.expected_adjoint_noise_next_head(value),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[
            int,
            torch.Tensor | None,
            tuple[tuple[torch.Tensor, torch.Tensor] | None, ...],
        ]
        | None,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            step = 0
            previous_x = None
            caches: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] = tuple(
                None for _ in self.blocks
            )
        else:
            step, previous_x, caches = hidden
        if int(step) >= self.max_steps or len(caches) != len(self.blocks):
            raise ValueError("invalid or exhausted Transformer cache")
        if self.input_mode == "increment":
            token = torch.zeros_like(x_t) if previous_x is None else x_t - previous_x
        else:
            token = x_t
        value = self.input_projection(token).unsqueeze(1)
        value = value + self.position_embedding[int(step)].reshape(1, 1, -1).to(value)
        next_caches = []
        for block, cache in zip(self.blocks, caches, strict=True):
            value, next_cache = block.forward_step(value, cache)
            next_caches.append(next_cache)
        features = self.final_norm(value[:, 0, :])
        next_hidden = (
            int(step) + 1,
            x_t if self.input_mode == "increment" else None,
            tuple(next_caches),
        )
        return (
            self.expected_adjoint_next_head(features),
            self.expected_adjoint_noise_next_head(features),
            next_hidden,
        )


__all__.append("CachedCausalTransformerAdjointNetwork")
