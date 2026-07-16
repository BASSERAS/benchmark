"""
PyTorch TimeGAN — faithful reimplementation of Yoon et al. NeurIPS 2019.
GPU-accelerated, tested on NVIDIA A100.

Architecture:
  Embedder     : 3-layer GRU -> Linear -> Sigmoid  -> (batch, T, hidden_dim)
  Recovery     : 3-layer GRU -> Linear             -> (batch, T, n_features)
  Generator    : 3-layer GRU -> Linear -> Sigmoid  -> (batch, T, hidden_dim)
  Supervisor   : 2-layer GRU -> Linear -> Sigmoid  -> (batch, T, hidden_dim)
  Discriminator: 3-layer GRU -> Linear             -> (batch, T, 1)  [logits]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional


class Embedder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class Recovery(nn.Module):
    def __init__(self, hidden_dim: int, n_features: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, n_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r, _ = self.rnn(h)
        return self.fc(r)


class Generator(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.rnn(z)
        return torch.sigmoid(self.fc(e))


class Supervisor(nn.Module):
    """Uses num_layers-1 GRU layers as in the original paper."""
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, max(1, num_layers - 1), batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s, _ = self.rnn(h)
        return torch.sigmoid(self.fc(s))


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d, _ = self.rnn(h)
        return self.fc(d)   # logits (batch, T, 1)


class TimeGAN:
    """TimeGAN trainer and sampler."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 24,
        num_layers: int = 3,
        batch_size: int = 128,
        device: str = "cuda",
        embedding_steps: int = 5000,
        supervised_steps: int = 5000,
        joint_steps: int = 10000,
        log_every: int = 100,
        gamma: float = 1.0,
    ):
        self.n_features      = n_features
        self.hidden_dim      = hidden_dim
        self.num_layers      = num_layers
        self.batch_size      = batch_size
        self.device          = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_steps = embedding_steps
        self.supervised_steps= supervised_steps
        self.joint_steps     = joint_steps
        self.log_every       = log_every
        self.gamma           = gamma

        # Networks
        self.embedder      = Embedder(n_features, hidden_dim, num_layers).to(self.device)
        self.recovery      = Recovery(hidden_dim, n_features, num_layers).to(self.device)
        self.generator     = Generator(n_features, hidden_dim, num_layers).to(self.device)
        self.supervisor    = Supervisor(hidden_dim, num_layers).to(self.device)
        self.discriminator = Discriminator(hidden_dim, num_layers).to(self.device)

        # Optimizers (one per component)
        self.opt_er   = optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=1e-3)
        self.opt_sup  = optim.Adam(self.supervisor.parameters(), lr=1e-3)
        self.opt_gen  = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=1e-3)

        # State
        self.loss_history: List[Dict[str, Any]] = []
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self._seq_len: int = 128

    # ── Utilities ─────────────────────────────────────────────────────────

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(x).to(self.device)

    def _batch(self, data: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(0, data.shape[0], (self.batch_size,))
        return data[idx]

    def _noise(self, n: int, T: int) -> torch.Tensor:
        return torch.rand(n, T, self.n_features, device=self.device)

    def _log(self, step: int, phase: str, **kw):
        self.loss_history.append({"step": step, "phase": phase, **kw})

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        flat = data.reshape(-1, data.shape[-1])
        self.min_val = flat.min(axis=0)
        self.max_val = flat.max(axis=0)
        denom = np.where(self.max_val - self.min_val == 0, 1.0, self.max_val - self.min_val)
        return (data - self.min_val) / (denom + 1e-7)

    def _denormalize(self, norm: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_val - self.min_val == 0, 1.0, self.max_val - self.min_val)
        return norm * (denom + 1e-7) + self.min_val

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def _phase1(self, data: torch.Tensor):
        T = data.shape[1]
        print(f"[Phase 1] Embedding pre-train  {self.embedding_steps} steps  T={T}")
        mse = nn.functional.mse_loss
        for step in range(self.embedding_steps):
            X = self._batch(data)
            self.opt_er.zero_grad()
            H = self.embedder(X)
            e_loss = mse(self.recovery(H), X)
            e_loss.backward()
            self.opt_er.step()
            if (step + 1) % self.log_every == 0:
                self._log(step + 1, "embedding", e_loss=round(e_loss.item(), 6),
                          s_loss=None, g_loss=None, d_loss=None)
                if (step + 1) % (self.log_every * 10) == 0:
                    print(f"  step {step+1:5d}  e_loss={e_loss.item():.6f}")

    # ── Phase 2 ───────────────────────────────────────────────────────────

    def _phase2(self, data: torch.Tensor):
        print(f"[Phase 2] Supervisor pre-train  {self.supervised_steps} steps")
        mse = nn.functional.mse_loss
        for step in range(self.supervised_steps):
            X = self._batch(data)
            self.opt_sup.zero_grad()
            with torch.no_grad():
                H = self.embedder(X)
            s_loss = mse(self.supervisor(H[:, :-1, :]), H[:, 1:, :])
            s_loss.backward()
            self.opt_sup.step()
            if (step + 1) % self.log_every == 0:
                self._log(step + 1, "supervised", e_loss=None,
                          s_loss=round(s_loss.item(), 6), g_loss=None, d_loss=None)
                if (step + 1) % (self.log_every * 10) == 0:
                    print(f"  step {step+1:5d}  s_loss={s_loss.item():.6f}")

    # ── Phase 3 ───────────────────────────────────────────────────────────

    def _phase3(self, data: torch.Tensor):
        T = data.shape[1]
        print(f"[Phase 3] Joint adversarial  {self.joint_steps} steps")
        mse = nn.functional.mse_loss
        bce = nn.functional.binary_cross_entropy_with_logits

        for step in range(self.joint_steps):
            # ── 2x generator update ──────────────────────────────────
            for _ in range(2):
                X = self._batch(data)
                Z = self._noise(self.batch_size, T)

                with torch.no_grad():
                    H_real = self.embedder(X)

                E_hat = self.generator(Z)
                H_hat = self.supervisor(E_hat)
                H_sup = self.supervisor(H_real[:, :-1, :])

                Y_fake   = self.discriminator(H_hat)
                Y_fake_e = self.discriminator(E_hat)

                ones = torch.ones_like(Y_fake)
                g_loss_u   = bce(Y_fake,   ones)
                g_loss_u_e = bce(Y_fake_e, ones)
                g_loss_s   = mse(H_sup, H_real[:, 1:, :])

                # Moment matching on recovered output
                with torch.no_grad():
                    X_hat_mm = self.recovery(H_hat)
                g_loss_v = (torch.mean(torch.abs(X_hat_mm.mean(0) - X.mean(0))) +
                            torch.mean(torch.abs(X_hat_mm.std(0)  - X.std(0))))

                g_loss = (g_loss_u + self.gamma * g_loss_u_e
                          + 10.0 * torch.sqrt(g_loss_s + 1e-8)
                          + 100.0 * g_loss_v)

                self.opt_gen.zero_grad()
                self.opt_sup.zero_grad()
                g_loss.backward()
                self.opt_gen.step()
                self.opt_sup.step()

            # ── discriminator (conditional) ───────────────────────
            X = self._batch(data)
            Z = self._noise(self.batch_size, T)
            with torch.no_grad():
                H_real = self.embedder(X)
                E_hat  = self.generator(Z)
                H_hat  = self.supervisor(E_hat)

            Y_real   = self.discriminator(H_real)
            Y_fake   = self.discriminator(H_hat)
            Y_fake_e = self.discriminator(E_hat)

            ones  = torch.ones_like(Y_real)
            zeros = torch.zeros_like(Y_fake)
            d_loss = (bce(Y_real, ones) + bce(Y_fake, zeros)
                      + self.gamma * bce(Y_fake_e, zeros))

            if d_loss.item() > 0.15:
                self.opt_disc.zero_grad()
                d_loss.backward()
                self.opt_disc.step()

            # ── embedder update ──────────────────────────────────
            X = self._batch(data)
            self.opt_er.zero_grad()
            H = self.embedder(X)
            e_loss = (mse(self.recovery(H), X)
                      + 10.0 * torch.sqrt(mse(self.supervisor(H[:, :-1, :]),
                                              H[:, 1:, :]) + 1e-8))
            e_loss.backward()
            self.opt_er.step()

            if (step + 1) % self.log_every == 0:
                self._log(step + 1, "joint",
                          e_loss=round(e_loss.item(), 6),
                          s_loss=round(g_loss_s.item(), 6),
                          g_loss=round(g_loss.item(), 6),
                          d_loss=round(d_loss.item(), 6))
                if (step + 1) % (self.log_every * 10) == 0:
                    print(f"  step {step+1:5d}  "
                          f"e={e_loss.item():.4f}  s={g_loss_s.item():.4f}  "
                          f"g={g_loss.item():.4f}  d={d_loss.item():.4f}")

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray):
        """Train on data shape (N, T) or (N, T, d)."""
        if data.ndim == 2:
            data = data[:, :, None]
        self._seq_len = data.shape[1]
        norm = self._normalize(data)
        td   = self._to_tensor(norm)
        self._phase1(td)
        self._phase2(td)
        self._phase3(td)

    def sample(self, n: int) -> np.ndarray:
        """Return n generated paths, shape (n, T), original scale."""
        T = self._seq_len
        for m in [self.embedder, self.recovery, self.generator,
                  self.supervisor, self.discriminator]:
            m.eval()
        with torch.no_grad():
            Z     = self._noise(n, T)
            E_hat = self.generator(Z)
            H_hat = self.supervisor(E_hat)
            X_hat = self.recovery(H_hat).cpu().numpy()   # (n, T, d)
        for m in [self.embedder, self.recovery, self.generator,
                  self.supervisor, self.discriminator]:
            m.train()
        out = self._denormalize(X_hat)          # (n, T, d)
        return out[:, :, 0] if out.shape[-1] == 1 else out

    def state_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k).state_dict()
                for k in ["embedder","recovery","generator","supervisor","discriminator"]}

    def load_state_dict(self, sd: Dict):
        for k in ["embedder","recovery","generator","supervisor","discriminator"]:
            getattr(self, k).load_state_dict(sd[k])

    def config(self) -> Dict[str, Any]:
        return {
            "n_features":       self.n_features,
            "hidden_dim":       self.hidden_dim,
            "num_layers":       self.num_layers,
            "batch_size":       self.batch_size,
            "embedding_steps":  self.embedding_steps,
            "supervised_steps": self.supervised_steps,
            "joint_steps":      self.joint_steps,
            "seq_len":          self._seq_len,
        }
