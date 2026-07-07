"""CausalGAN: GAN with structural causal equations (Kocaoglu et al., 2018)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["CausalGAN"]


class _CausalGANModule(nn.Module):
    """Structural generators X <- noise_x; T <- (X, noise_t); Y <- (X, T, noise_y)."""

    def __init__(self, x_dim: int, hidden: int = 128, noise_dim: int = 8):
        super().__init__()
        self.x_dim = x_dim
        self.noise_dim = noise_dim
        self.gen_x = MLP(noise_dim, x_dim, hidden=(hidden, hidden), activation=nn.ELU)
        self.gen_t = MLP(x_dim + noise_dim, 1, hidden=(hidden,), activation=nn.ELU)
        self.gen_y = MLP(x_dim + 1 + noise_dim, 1, hidden=(hidden, hidden), activation=nn.ELU)
        self.disc = MLP(x_dim + 2, 1, hidden=(hidden, hidden), activation=nn.ELU)
        self.lab_x = MLP(x_dim, 1, hidden=(hidden // 2,), activation=nn.ELU)
        self.lab_t = MLP(1, 1, hidden=(hidden // 2,), activation=nn.ELU)
        self.lab_y = MLP(1, 1, hidden=(hidden // 2,), activation=nn.ELU)

    def generate(self, batch: int, device: torch.device, do_t: float | None = None):
        nz = torch.randn(batch, self.noise_dim, device=device)
        x_hat = self.gen_x(nz)
        nt = torch.randn(batch, self.noise_dim, device=device)
        t_logit = self.gen_t(torch.cat([x_hat, nt], dim=-1))
        if do_t is not None:
            t_hat = torch.full((batch, 1), do_t, device=device)
        else:
            t_hat = torch.sigmoid(t_logit)
        ny = torch.randn(batch, self.noise_dim, device=device)
        y_hat = self.gen_y(torch.cat([x_hat, t_hat, ny], dim=-1))
        return x_hat, t_hat, y_hat

    def forward_real(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.disc(torch.cat([x, t, y], dim=-1)).squeeze(-1)


class CausalGAN(BaseDeepEstimator):
    """CausalGAN for interventional and counterfactual queries under X -> T -> Y."""

    def __init__(self, hidden: int = 128, noise_dim: int = 8, lambda_lab: float = 0.5, **kwargs):
        kwargs.setdefault("epochs", 150)
        super().__init__(**kwargs)
        self.hidden = hidden
        self.noise_dim = noise_dim
        self.lambda_lab = lambda_lab

    def fit(self, X, treatment, outcome) -> CausalGAN:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment).reshape(-1, 1)
        y = to_numpy(outcome).astype(np.float64).ravel().reshape(-1, 1)
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        self.module_ = module_to_device(_CausalGANModule(x.shape[1], self.hidden, self.noise_dim), device)
        opt = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                batch = xb.shape[0]
                opt.zero_grad()
                real_logit = self.module_.forward_real(xb, tb, yb)
                xf, tf, yf = self.module_.generate(batch, device)
                fake_logit = self.module_.forward_real(xf, tf, yf)
                d_loss = F.binary_cross_entropy_with_logits(
                    real_logit, torch.ones_like(real_logit)
                ) + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
                g_adv = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
                g_lab = (
                    F.binary_cross_entropy_with_logits(
                        self.module_.lab_x(xf), torch.ones(batch, 1, device=device)
                    )
                    + F.binary_cross_entropy_with_logits(
                        self.module_.lab_t(tf), torch.ones(batch, 1, device=device)
                    )
                    + F.binary_cross_entropy_with_logits(
                        self.module_.lab_y(yf), torch.ones(batch, 1, device=device)
                    )
                )
                g_loss = g_adv + self.lambda_lab * g_lab
                (d_loss + g_loss).backward()
                opt.step()
                losses.append((d_loss.item() + g_loss.item()) / 2)

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_cate(self, X, n_samples: int = 200) -> np.ndarray:
        self._check_fitted()
        device = self._device
        n = len(check_array(X, "X"))
        y0_acc = np.zeros(n)
        y1_acc = np.zeros(n)
        self.module_.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                _, _, y0 = self.module_.generate(n, device, do_t=0.0)
                _, _, y1 = self.module_.generate(n, device, do_t=1.0)
                y0_acc += to_numpy(y0).ravel()
                y1_acc += to_numpy(y1).ravel()
        return (y1_acc - y0_acc) / n_samples

    def predict_ate(self, X) -> float:
        return float(self.predict_cate(X).mean())
