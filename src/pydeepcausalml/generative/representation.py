"""iVAE and CausalVAE: identifiable and causal representation learning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator
from ..utils import check_array, module_to_device, notears_acyclicity_h, to_numpy, to_tensor

__all__ = ["IVAE", "CausalVAE"]


class _IVAEModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, aux_dim: int, hidden: int):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim * 2, hidden=(hidden, hidden), activation=nn.ELU)
        self.decoder = MLP(latent_dim, input_dim, hidden=(hidden, hidden), activation=nn.ELU)
        self.prior_net = MLP(aux_dim, latent_dim * 2, hidden=(hidden,), activation=nn.ELU)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, mu, logvar

    def prior_params(self, u: torch.Tensor):
        h = self.prior_net(u)
        return h.chunk(2, dim=-1)


class IVAE(BaseDeepEstimator):
    """Identifiable VAE with auxiliary-variable conditioned prior (iVAE, Khemakhem et al., 2020)."""

    def __init__(self, latent_dim: int = 8, hidden: int = 128, beta_kl: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta_kl = beta_kl

    def fit(self, X, aux) -> IVAE:
        device = self._setup()
        x = check_array(X, "X")
        u = check_array(aux, "aux")
        if len(x) != len(u):
            raise ValueError("X and aux must have the same length.")

        self.module_ = module_to_device(_IVAEModule(x.shape[1], self.latent_dim, u.shape[1], self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(u)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, ub in loader:
                xb, ub = xb.to(device), ub.to(device)
                optimizer.zero_grad()
                z, mu, logvar = self.module_.encode(xb)
                x_hat = self.module_.decoder(z)
                p_mu, p_logvar = self.module_.prior_params(ub)
                kl = 0.5 * (
                    p_logvar - logvar + (logvar.exp() + (mu - p_mu).pow(2)) / p_logvar.exp() - 1
                ).sum(-1).mean()
                loss = F.mse_loss(x_hat, xb) + self.beta_kl * kl
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        self.module_.eval()
        with torch.no_grad():
            z, _, _ = self.module_.encode(to_tensor(x, device=device))
        return to_numpy(z)


class _CausalVAEModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, latent_dim * 2, hidden=(hidden, hidden), activation=nn.ELU)
        self.decoder = MLP(latent_dim, input_dim, hidden=(hidden, hidden), activation=nn.ELU)
        self.adj = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, mu, logvar

    def dag_penalty(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.latent_dim, device=self.adj.device))
        return notears_acyclicity_h(w).pow(2) + w.abs().sum()


class CausalVAE(BaseDeepEstimator):
    """CausalVAE with learned DAG over latent variables (Yang et al., 2021)."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: int = 128,
        beta_kl: float = 1.0,
        lambda_dag: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta_kl = beta_kl
        self.lambda_dag = lambda_dag

    def fit(self, X) -> CausalVAE:
        device = self._setup()
        x = check_array(X, "X")
        self.module_ = module_to_device(_CausalVAEModule(x.shape[1], self.latent_dim, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(to_tensor(x)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for (xb,) in loader:
                xb = xb.to(device)
                optimizer.zero_grad()
                z, mu, logvar = self.module_.encode(xb)
                x_hat = self.module_.decoder(z)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                loss = F.mse_loss(x_hat, xb) + self.beta_kl * kl + self.lambda_dag * self.module_.dag_penalty()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        with torch.no_grad():
            w = torch.sigmoid(self.module_.adj).cpu().numpy()
        np.fill_diagonal(w, 0)
        return w

    def transform(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        self.module_.eval()
        with torch.no_grad():
            z, _, _ = self.module_.encode(to_tensor(x, device=device))
        return to_numpy(z)
