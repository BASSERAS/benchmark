from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdjointMoments:
    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor


def _zero_init_final_linear(module: nn.Module) -> None:
    final = module[-1] if isinstance(module, nn.Sequential) else None
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class AdjointMomentNetwork(nn.Module):
    """
    Prefix network producing the conditional adjoint moments needed by the
    Hamiltonian control map.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        adjoint_dim: int | None = None,
        hidden_dim: int = 96,
        num_layers: int = 1,
        noise_adjoint_dim: int | None = None,
        input_mode: str = "level",
    ) -> None:
        super().__init__()
        if int(state_dim) < 1:
            raise ValueError("state_dim must be >= 1")
        output_dim = int(state_dim if adjoint_dim is None else adjoint_dim)
        if output_dim < 1:
            raise ValueError("adjoint_dim must be >= 1")
        noise_output_dim = int(output_dim if noise_adjoint_dim is None else noise_adjoint_dim)
        if noise_output_dim < 1:
            raise ValueError("noise_adjoint_dim must be >= 1")
        if not isinstance(input_mode, str):
            raise ValueError("input_mode must be level or increment")
        input_mode = input_mode.strip().lower()
        if input_mode not in {"level", "increment"}:
            raise ValueError("input_mode must be level or increment")
        self.state_dim = int(state_dim)
        self.adjoint_dim = int(output_dim)
        self.noise_adjoint_dim = int(noise_output_dim)
        self.input_mode = input_mode
        self.gru = nn.GRU(
            input_size=int(state_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.expected_adjoint_next_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )
        self.expected_adjoint_noise_next_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(noise_output_dim)),
        )
        _zero_init_final_linear(self.expected_adjoint_next_head)
        _zero_init_final_linear(self.expected_adjoint_noise_next_head)

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        if x_path.ndim != 3:
            raise ValueError("x_path must have shape (B, N + 1, d)")
        if int(x_path.shape[1]) < 2:
            raise ValueError("x_path must contain at least two time points")
        if int(x_path.shape[2]) != int(self.state_dim):
            raise ValueError("x_path state dimension does not match network state_dim")
        network_inputs = x_path[:, :-1, :]
        if self.input_mode == "increment":
            previous_increments = x_path[:, 1:-1, :] - x_path[:, :-2, :]
            network_inputs = torch.cat(
                [torch.zeros_like(x_path[:, :1, :]), previous_increments],
                dim=1,
            )
        output, _ = self.gru(network_inputs)
        return AdjointMoments(
            expected_adjoint_next=self.expected_adjoint_next_head(output),
            expected_adjoint_noise_next=self.expected_adjoint_noise_next_head(output),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ]:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (B, d)")
        if int(x_t.shape[1]) != int(self.state_dim):
            raise ValueError("x_t state dimension does not match network state_dim")
        if self.input_mode == "level":
            if isinstance(hidden, tuple):
                raise TypeError("level-input hidden state must be a tensor or None")
            network_input = x_t
            gru_hidden = hidden
        elif hidden is None:
            network_input = torch.zeros_like(x_t)
            gru_hidden = None
        else:
            if not isinstance(hidden, tuple) or len(hidden) != 2:
                raise TypeError("increment-input hidden state must contain GRU state and previous level")
            gru_hidden, previous_x = hidden
            network_input = x_t - previous_x
        output, next_hidden = self.gru(network_input.unsqueeze(1), gru_hidden)
        features = output[:, -1, :]
        recurrent_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if self.input_mode == "increment":
            recurrent_state = (next_hidden, x_t)
        else:
            recurrent_state = next_hidden
        return (
            self.expected_adjoint_next_head(features),
            self.expected_adjoint_noise_next_head(features),
            recurrent_state,
        )
