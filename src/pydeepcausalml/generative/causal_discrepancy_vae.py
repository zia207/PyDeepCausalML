"""CausalDiscrepancyVAE: VAE with MMD balancing in latent space."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator, mmd_rbf
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["CausalDiscrepancyVAE"]


class _CDVAEModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.encoder = MLP(input_dim + 1, latent_dim * 2, hidden=(hidden, hidden), activation=nn.ELU)
        self.decoder_x = MLP(latent_dim, input_dim, hidden=(hidden, hidden), activation=nn.ELU)
        self.head_y0 = MLP(latent_dim, 1, hidden=(hidden // 2,), activation=nn.ELU)
        self.head_y1 = MLP(latent_dim, 1, hidden=(hidden // 2,), activation=nn.ELU)

    def encode(self, x: torch.Tensor, t: torch.Tensor):
        h = self.encoder(torch.cat([x, t.unsqueeze(-1)], dim=-1))
        mu, logvar = h.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + eps * (0.5 * logvar).exp()
        return z, mu, logvar

    def potential_outcomes(self, z: torch.Tensor):
        return self.head_y0(z).squeeze(-1), self.head_y1(z).squeeze(-1)


class CausalDiscrepancyVAE(BaseDeepEstimator):
    """VAE with treatment/outcome heads and MMD balancing between treated/control latents."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden: int = 128,
        beta_kl: float = 1.0,
        beta_mmd: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta_kl = beta_kl
        self.beta_mmd = beta_mmd

    def fit(self, X, treatment, outcome) -> CausalDiscrepancyVAE:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        self.module_ = module_to_device(_CDVAEModule(x.shape[1], self.latent_dim, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                z, mu, logvar = self.module_.encode(xb, tb)
                x_hat = self.module_.decoder_x(z)
                y0, y1 = self.module_.potential_outcomes(z)
                y_hat = torch.where(tb.bool(), y1, y0)
                recon = F.mse_loss(x_hat, xb) + F.mse_loss(y_hat, yb)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                z0, z1 = z[~tb.bool()], z[tb.bool()]
                mmd = mmd_rbf(z0, z1) if z0.shape[0] > 1 and z1.shape[0] > 1 else torch.tensor(0.0, device=device)
                loss = recon + self.beta_kl * kl + self.beta_mmd * mmd
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_cate(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        t = torch.zeros(len(x), device=device)
        self.module_.eval()
        with torch.no_grad():
            z, _, _ = self.module_.encode(to_tensor(x, device=device), t)
            y0, y1 = self.module_.potential_outcomes(z)
        return to_numpy(y1 - y0)

    def predict_ate(self, X) -> float:
        return float(self.predict_cate(X).mean())
