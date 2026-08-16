from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SetTransformerCEConfig:
    hidden_dim: int = 64
    num_heads: int = 4
    num_inducing_points: int = 16
    steps: int = 1_500
    minimum_steps: int = 200
    batch_size: int = 64
    eval_every: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    seed: int = 0


@dataclass(frozen=True)
class SetTransformerCEFit:
    network: nn.Module
    scale: torch.Tensor
    selected_step: int
    history: tuple[dict[str, float], ...]


class SetTransformerConditionalMean(nn.Module):
    """Query-conditioned, permutation-invariant estimator of conditional R.

    Each query is a generated prefix.  The unordered set contains independent
    branch-level observations of the pathwise R target.  Inducing attention
    compresses that set before a query token reads it.  The ordinary sample
    mean is an explicit residual connection, so a learned correction must earn
    its improvement over Monte Carlo averaging.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        output_dim: int,
        max_steps: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_inducing_points: int = 16,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.state_dim = int(state_dim)
        self.output_dim = int(output_dim)
        self.max_steps = int(max_steps)
        self.query_projection = nn.Sequential(
            nn.Linear(
                self.max_steps * self.state_dim + self.max_steps,
                int(hidden_dim),
            ),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.element_projection = nn.Sequential(
            nn.Linear(self.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.inducing_points = nn.Parameter(
            torch.empty(int(num_inducing_points), int(hidden_dim))
        )
        nn.init.normal_(self.inducing_points, mean=0.0, std=0.02)
        self.inducing_attention = nn.MultiheadAttention(
            int(hidden_dim), int(num_heads), batch_first=True
        )
        self.query_attention = nn.MultiheadAttention(
            int(hidden_dim), int(num_heads), batch_first=True
        )
        self.inducing_norm = nn.LayerNorm(int(hidden_dim))
        self.query_norm = nn.LayerNorm(int(hidden_dim))
        self.correction = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        final = self.correction[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _queries(self, paths: torch.Tensor) -> torch.Tensor:
        batch = int(paths.shape[0])
        steps = int(paths.shape[1]) - 1
        if steps != self.max_steps or int(paths.shape[2]) != self.state_dim:
            raise ValueError("paths do not match the Set Transformer configuration")
        observed = paths[:, :-1, :]
        time = torch.arange(steps, device=paths.device)
        causal = time[None, :] <= time[:, None]
        prefixes = (
            observed[:, None, :, :]
            .expand(batch, steps, steps, self.state_dim)
            * causal[None, :, :, None]
        )
        masks = causal.to(paths.dtype)[None, :, :].expand(batch, -1, -1)
        query_input = torch.cat(
            (prefixes.reshape(batch, steps, -1), masks), dim=-1
        )
        return self.query_projection(query_input)

    def forward(
        self,
        *,
        paths: torch.Tensor,
        branch_values: torch.Tensor,
    ) -> torch.Tensor:
        if branch_values.ndim != 4:
            raise ValueError("branch_values must have shape (B, N, M, q)")
        batch, steps, branches, output_dim = branch_values.shape
        if (
            int(steps) != self.max_steps
            or int(output_dim) != self.output_dim
            or int(paths.shape[0]) != int(batch)
        ):
            raise ValueError("branch values do not match paths or configuration")
        query = self._queries(paths).reshape(batch * steps, 1, -1)
        elements = self.element_projection(
            branch_values.reshape(batch * steps, branches, output_dim)
        )
        inducing = self.inducing_points.unsqueeze(0).expand(
            batch * steps, -1, -1
        )
        induced, _ = self.inducing_attention(
            inducing, elements, elements, need_weights=False
        )
        induced = self.inducing_norm(inducing + induced)
        pooled, _ = self.query_attention(
            query, induced, induced, need_weights=False
        )
        pooled = self.query_norm(query + pooled)[:, 0, :]
        correction = self.correction(pooled).reshape(batch, steps, output_dim)
        return branch_values.mean(dim=2) + correction


def _batched_prediction(
    network: SetTransformerConditionalMean,
    *,
    paths: torch.Tensor,
    branches: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, int(paths.shape[0]), int(batch_size)):
        stop = min(int(paths.shape[0]), start + int(batch_size))
        rows.append(
            network(paths=paths[start:stop], branch_values=branches[start:stop])
        )
    return torch.cat(rows, dim=0)


def fit_set_transformer_ce(
    *,
    fit_paths: torch.Tensor,
    fit_branches: torch.Tensor,
    fit_independent_target: torch.Tensor,
    validation_paths: torch.Tensor,
    validation_branches: torch.Tensor,
    validation_independent_target: torch.Tensor,
    config: SetTransformerCEConfig,
) -> SetTransformerCEFit:
    if int(config.steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("steps and batch_size must be positive")
    device = fit_paths.device
    dtype = fit_paths.dtype
    torch.manual_seed(int(config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))
    scale = fit_independent_target.double().pow(2).mean(dim=0).sqrt().clamp_min(
        1e-3
    ).to(device=device, dtype=dtype)
    fit_values = fit_branches / scale.unsqueeze(0).unsqueeze(2)
    fit_target = fit_independent_target / scale.unsqueeze(0)
    validation_values = validation_branches / scale.unsqueeze(0).unsqueeze(2)
    validation_target = validation_independent_target / scale.unsqueeze(0)
    network = SetTransformerConditionalMean(
        state_dim=int(fit_paths.shape[-1]),
        output_dim=int(fit_branches.shape[-1]),
        max_steps=int(fit_branches.shape[1]),
        hidden_dim=int(config.hidden_dim),
        num_heads=int(config.num_heads),
        num_inducing_points=int(config.num_inducing_points),
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 1)
    best_loss = math.inf
    best_step = 0
    best_state = copy.deepcopy(network.state_dict())
    history: list[dict[str, float]] = []

    def evaluate(step: int) -> float:
        network.eval()
        with torch.no_grad():
            prediction = _batched_prediction(
                network,
                paths=validation_paths,
                branches=validation_values,
                batch_size=int(config.batch_size),
            )
            loss = torch.mean((prediction - validation_target).pow(2))
            raw_mean = validation_values.mean(dim=2)
            raw_loss = torch.mean((raw_mean - validation_target).pow(2))
        history.append(
            {
                "step": float(step),
                "validation_loss": float(loss.item()),
                "validation_sample_mean_loss": float(raw_loss.item()),
            }
        )
        network.train()
        return float(loss.item())

    initial = evaluate(0)
    best_loss = initial
    for step in range(1, int(config.steps) + 1):
        count = min(int(config.batch_size), int(fit_paths.shape[0]))
        indices = torch.randint(
            int(fit_paths.shape[0]),
            (count,),
            generator=generator,
            device="cpu",
        ).to(device)
        prediction = network(
            paths=fit_paths.index_select(0, indices),
            branch_values=fit_values.index_select(0, indices),
        )
        loss = torch.mean(
            (prediction - fit_target.index_select(0, indices)).pow(2)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            network.parameters(), float(config.grad_clip_norm)
        )
        optimizer.step()
        if step % int(config.eval_every) == 0 or step == int(config.steps):
            validation_loss = evaluate(step)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_state)
    network.eval()
    return SetTransformerCEFit(
        network=network,
        scale=scale.detach(),
        selected_step=int(best_step),
        history=tuple(history),
    )


def predict_set_transformer_ce(
    fit: SetTransformerCEFit,
    *,
    paths: torch.Tensor,
    branches: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    normalized = branches / fit.scale.unsqueeze(0).unsqueeze(2)
    fit.network.eval()
    with torch.no_grad():
        prediction = _batched_prediction(
            fit.network,
            paths=paths,
            branches=normalized,
            batch_size=int(batch_size),
        )
    return prediction * fit.scale.unsqueeze(0)


__all__ = [
    "SetTransformerCEConfig",
    "SetTransformerCEFit",
    "SetTransformerConditionalMean",
    "fit_set_transformer_ce",
    "predict_set_transformer_ce",
]
