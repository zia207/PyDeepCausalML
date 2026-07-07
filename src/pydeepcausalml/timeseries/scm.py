"""Structural causal models for time series: DeepSCM, DECI."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator, MLP
from ..timeseries.granger import make_lagged_sequences
from ..utils import check_array, module_to_device, notears_acyclicity_h, to_numpy, to_tensor

__all__ = ["DeepSCM", "DECI"]


class _DeepSCMModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.equations = nn.ModuleList(
            [MLP(n_vars, 1, hidden=(hidden,), activation=nn.ELU) for _ in range(n_vars)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([eq(x) for eq in self.equations], dim=-1)


class DeepSCM(BaseDeepEstimator):
    """Deep variational SCM with fixed-graph structural equations."""

    def __init__(self, lag: int = 1, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X) -> DeepSCM:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_DeepSCMModule(n_vars, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin[:, -1], xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(self.module_(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def intervene(self, X, var_idx: int, value: float) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        x_t = to_tensor(x, device=device)
        x_t[:, var_idx] = value
        self.module_.eval()
        with torch.no_grad():
            return to_numpy(self.module_(x_t))


class _DECIModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.adj = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.equations = nn.ModuleList(
            [MLP(n_vars, 1, hidden=(hidden, hidden), activation=nn.ELU) for _ in range(n_vars)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.n_vars, device=x.device))
        x_masked = x @ w.T
        return torch.cat([eq(x_masked) for eq in self.equations], dim=-1)

    def dag_penalty(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.n_vars, device=self.adj.device))
        return notears_acyclicity_h(w).pow(2) + w.abs().sum()


class DECI(BaseDeepEstimator):
    """DECI: joint graph and structural equation learning (Geffner et al., 2022)."""

    def __init__(self, lag: int = 1, hidden: int = 32, lambda_dag: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.lambda_dag = lambda_dag

    def fit(self, X) -> DECI:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_DECIModule(n_vars, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin[:, -1], xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                loss = F.mse_loss(pred, yb) + self.lambda_dag * self.module_.dag_penalty()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        w = torch.sigmoid(self.module_.adj).detach().cpu().numpy()
        np.fill_diagonal(w, 0)
        return w

    def predict_ate(self, X, intervention_var: int = 0, n_samples: int = 100) -> float:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        x_t = to_tensor(x[:n_samples], device=device)
        self.module_.eval()
        with torch.no_grad():
            x0 = x_t.clone()
            x1 = x_t.clone()
            x0[:, intervention_var] = 0.0
            x1[:, intervention_var] = 1.0
            y0 = self.module_(x0)[:, -1].mean()
            y1 = self.module_(x1)[:, -1].mean()
        return float((y1 - y0).cpu())
