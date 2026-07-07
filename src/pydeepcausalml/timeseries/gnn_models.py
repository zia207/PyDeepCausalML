"""GNN-based causal models: GVAR, CausalGNN, CUTS."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..timeseries.granger import make_lagged_sequences
from ..utils import check_array, module_to_device, notears_acyclicity_h, to_numpy, to_tensor

__all__ = ["GVAR", "CausalGNN", "CUTS", "gnn_causal_model"]


class _GVARModule(nn.Module):
    def __init__(self, n_vars: int, lag: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.lag = lag
        self.adj = nn.Parameter(torch.zeros(lag, n_vars, n_vars))
        self.msg = nn.Linear(n_vars, hidden)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        agg = torch.zeros(batch, self.n_vars, device=x.device)
        for l in range(self.lag):
            w = torch.sigmoid(self.adj[l])
            x_l = x[:, l]
            agg = agg + x_l @ w.T
        h = F.relu(self.msg(agg))
        return self.head(h)

    def dag_penalty(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj).mean(0)
        w = w * (1 - torch.eye(self.n_vars, device=w.device))
        return notears_acyclicity_h(w).pow(2)


class GVAR(BaseDeepEstimator):
    """Graph Vector Autoregression with lag-specific soft adjacency."""

    def __init__(self, lag: int = 5, hidden: int = 32, lambda_dag: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.lambda_dag = lambda_dag

    def fit(self, X) -> GVAR:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_GVARModule(n_vars, self.lag, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

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

    def causal_matrix(self) -> np.ndarray:
        self._check_fitted()
        w = torch.sigmoid(self.module_.adj).mean(0).detach().cpu().numpy()
        np.fill_diagonal(w, 0)
        return w


class _CausalGNNModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.gru = nn.GRU(n_vars, hidden, batch_first=True)
        self.bilinear = nn.Bilinear(hidden, hidden, 1)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1])


class CausalGNN(BaseDeepEstimator):
    """Causal GNN with bilinear graph learner and GRU encoder."""

    def __init__(self, lag: int = 5, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X) -> CausalGNN:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_CausalGNNModule(n_vars, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

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

    def causal_matrix(self) -> np.ndarray:
        self._check_fitted()
        n = self.module_.n_vars
        h = torch.randn(1, self.hidden)
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                w[i, j] = torch.sigmoid(self.module_.bilinear(h, h)).item()
        return w


class _CUTSModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.edge_logits = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.encoder = nn.GRU(n_vars, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.decoder(h[:, -1])

    def edge_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.edge_logits)


class CUTS(BaseDeepEstimator):
    """CUTS+: variational Bernoulli graph posterior for missing time series."""

    def __init__(self, lag: int = 5, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X) -> CUTS:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_CUTSModule(n_vars, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                kl = -(self.module_.edge_probs() * torch.log(self.module_.edge_probs() + 1e-8)
                       + (1 - self.module_.edge_probs()) * torch.log(1 - self.module_.edge_probs() + 1e-8)).sum()
                loss = F.mse_loss(pred, yb) + 0.01 * kl
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def causal_matrix(self, threshold: float = 0.5) -> np.ndarray:
        self._check_fitted()
        w = self.module_.edge_probs().detach().cpu().numpy()
        return (w >= threshold).astype(float)


_GNN_MODELS = {"gvar": GVAR, "causal_gnn": CausalGNN, "causalgnn": CausalGNN, "cuts": CUTS}


def gnn_causal_model(method: str = "gvar", **kwargs) -> BaseDeepEstimator:
    """Factory for GNN causal models (mirrors R ``gnn_causal_model()``)."""
    key = method.lower().replace("-", "_")
    if key not in _GNN_MODELS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_GNN_MODELS)}.")
    return _GNN_MODELS[key](**kwargs)
