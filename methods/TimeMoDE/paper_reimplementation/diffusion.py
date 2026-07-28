"""DDPM diffusion for TimeMoDE (fixed-variance, eps-prediction).

Standard Ho et al. (2020) DDPM forward/reverse process. TimeMoDE's training
objective (Eq 3 / Eq 15) is the *simple* eps-prediction loss L_DDPM with NO
learned variance term, so sampling uses the fixed posterior variance.

The total training loss is L = L_DDPM + w_proto * L_proto + w_aux * L_aux
(Eq 15), where L_proto (prototype orthonormality) and L_aux (expert load
balancing) come from the model's forward pass.

Model signature: model(x, t, dp_exemplar, y) -> (eps_pred, aux_loss, proto_loss).
Shapes are (B, L, C). Diffusion steps default T=250 (paper Table 8).
"""
import numpy as np
import torch
import torch.nn.functional as F


def make_beta_schedule(T, kind="linear"):
    if kind == "linear":
        scale = 1000.0 / T
        return np.linspace(scale * 1e-4, scale * 2e-2, T, dtype=np.float64)
    if kind == "cosine":
        steps = T + 1
        s = 0.008
        x = np.linspace(0, T, steps, dtype=np.float64)
        ac = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return np.clip(betas, 0, 0.999)
    raise ValueError(kind)


def _extract(arr, t, shape):
    out = torch.as_tensor(arr, device=t.device, dtype=torch.float32)[t]
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class Diffusion:
    def __init__(self, T=250, schedule="linear"):
        self.T = T
        betas = make_beta_schedule(T, schedule)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_acp = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_var_clipped = np.log(
            np.append(self.posterior_var[1], self.posterior_var[1:])
        )
        self.posterior_mean_c1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_c2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        return _extract(self.sqrt_acp, t, x0.shape) * x0 + \
            _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise

    def predict_x0_from_eps(self, xt, t, eps):
        return (xt - _extract(self.sqrt_one_minus_acp, t, xt.shape) * eps) / \
            _extract(self.sqrt_acp, t, xt.shape)

    def q_posterior_mean(self, x0, xt, t):
        return _extract(self.posterior_mean_c1, t, xt.shape) * x0 + \
            _extract(self.posterior_mean_c2, t, xt.shape) * xt

    def training_loss(self, model, x0, dp_exemplar, y=None,
                      w_proto=0.0, w_aux=0.0):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps, aux, proto = model(xt, t, dp_exemplar, y)
        loss_simple = F.mse_loss(eps, noise)
        loss = loss_simple + w_proto * proto + w_aux * aux
        parts = {"simple": loss_simple.item(),
                 "proto": float(proto) if torch.is_tensor(proto) else float(proto),
                 "aux": float(aux) if torch.is_tensor(aux) else float(aux)}
        return loss, parts

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, dp_exemplar, y=None):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps, _, _ = model(x, t, dp_exemplar, y)
            x0 = self.predict_x0_from_eps(x, t, eps).clamp(-3, 3)
            mean = self.q_posterior_mean(x0, x, t)
            if i > 0:
                log_var = _extract(self.posterior_log_var_clipped, t, x.shape)
                mean = mean + torch.exp(0.5 * log_var) * torch.randn_like(x)
            x = mean
        return x
