"""DAG-GNN and nonlinear NOTEARS for causal structure learning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseDeepEstimator, MLP
from ..utils import check_array, module_to_device, notears_acyclicity_h, to_numpy, to_tensor

__all__ = ["DAGGNN", "NOTEARSNonlinearMLP", "NOTEARSNonlinearSobolev"]


class _DAGGNNModule(nn.Module):
    def __init__(self, d: int, hidden: int, latent: int):
        super().__init__()
        self.d = d
        self.encoder = MLP(d, latent, hidden=(hidden,), activation=nn.ELU)
        self.decoder = MLP(latent, d, hidden=(hidden,), activation=nn.ELU)
        self.adj = nn.Parameter(torch.zeros(d, d))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def dag_penalty(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.d, device=self.adj.device))
        return notears_acyclicity_h(w).pow(2) + w.abs().sum()


class DAGGNN(BaseDeepEstimator):
    """DAG-GNN: VAE-style graph learning with augmented-Lagrangian DAG penalty."""

    def __init__(self, hidden: int = 64, latent: int = 16, lambda_dag: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.latent = latent
        self.lambda_dag = lambda_dag

    def fit(self, X) -> DAGGNN:
        device = self._setup()
        x = check_array(X, "X")
        d = x.shape[1]
        self.module_ = module_to_device(_DAGGNNModule(d, self.hidden, self.latent), device)
        optimizer = self._make_optimizer(self.module_)
        x_t = to_tensor(x, device=device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x_hat, _ = self.module_(x_t)
            recon = F.mse_loss(x_hat, x_t)
            dag = self.module_.dag_penalty()
            loss = recon + self.lambda_dag * dag
            loss.backward()
            optimizer.step()
            self._record(loss=loss.item(), dag=dag.item())
            self._log_epoch(epoch, loss=loss.item())

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        with torch.no_grad():
            w = torch.sigmoid(self.module_.adj).cpu().numpy()
        np.fill_diagonal(w, 0)
        return w


class _NotearsMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.d = d
        self.adj = nn.Parameter(torch.zeros(d, d))
        self.mlps = nn.ModuleList([MLP(d, 1, hidden=(hidden, hidden), activation=nn.ELU) for _ in range(d)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([mlp(x) for mlp in self.mlps], dim=-1)

    def h(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.d, device=self.adj.device))
        return notears_acyclicity_h(w)


class NOTEARSNonlinearMLP(BaseDeepEstimator):
    """Nonlinear NOTEARS with per-node MLPs (Zheng et al., 2020 extension)."""

    def __init__(self, hidden: int = 32, lambda1: float = 0.01, rho: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.lambda1 = lambda1
        self.rho = rho

    def fit(self, X) -> NOTEARSNonlinearMLP:
        device = self._setup()
        x = check_array(X, "X")
        d = x.shape[1]
        self.module_ = module_to_device(_NotearsMLP(d, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        x_t = to_tensor(x, device=device)
        rho = self.rho

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x_hat = self.module_(x_t)
            recon = F.mse_loss(x_hat, x_t)
            h = self.module_.h()
            w = torch.sigmoid(self.module_.adj)
            loss = recon + 0.5 * rho * h.pow(2) + self.lambda1 * w.abs().sum()
            loss.backward()
            optimizer.step()
            if h.item() < 1e-8:
                rho *= 10
            self._record(loss=loss.item(), h=h.item())
            self._log_epoch(epoch, loss=loss.item())

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        with torch.no_grad():
            w = torch.sigmoid(self.module_.adj).cpu().numpy()
        np.fill_diagonal(w, 0)
        return w


class _NotearsSobolev(nn.Module):
    """Simplified Sobolev-basis NOTEARS: polynomial features + linear mixing."""

    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        self.d = d
        self.degree = degree
        n_basis = d * degree
        self.adj = nn.Parameter(torch.zeros(d, d))
        self.coeff = nn.Parameter(torch.randn(d, n_basis) * 0.01)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for _ in range(1, self.degree):
            parts.append(parts[-1] * x)
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._basis(x)
        return basis @ self.coeff.T

    def h(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.d, device=self.adj.device))
        return notears_acyclicity_h(w)


class NOTEARSNonlinearSobolev(BaseDeepEstimator):
    """Sobolev-basis nonlinear NOTEARS."""

    def __init__(self, degree: int = 2, lambda1: float = 0.01, rho: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.lambda1 = lambda1
        self.rho = rho

    def fit(self, X) -> NOTEARSNonlinearSobolev:
        device = self._setup()
        x = check_array(X, "X")
        d = x.shape[1]
        self.module_ = module_to_device(_NotearsSobolev(d, self.degree), device)
        optimizer = self._make_optimizer(self.module_)
        x_t = to_tensor(x, device=device)
        rho = self.rho

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x_hat = self.module_(x_t)
            recon = F.mse_loss(x_hat, x_t)
            h = self.module_.h()
            w = torch.sigmoid(self.module_.adj)
            loss = recon + 0.5 * rho * h.pow(2) + self.lambda1 * w.abs().sum()
            loss.backward()
            optimizer.step()
            if h.item() < 1e-8:
                rho *= 10
            self._record(loss=loss.item(), h=h.item())
            self._log_epoch(epoch, loss=loss.item())

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        with torch.no_grad():
            w = torch.sigmoid(self.module_.adj).cpu().numpy()
        np.fill_diagonal(w, 0)
        return w
