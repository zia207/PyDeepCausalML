"""Extended neural Granger models: cLSTM, EconomySRU, NRI."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..utils import check_array, module_to_device, to_numpy, to_tensor
from .granger import NeuralGrangerCMLP, make_lagged_sequences

__all__ = [
    "NeuralGrangerCLSTM",
    "NeuralGrangerEconomySRU",
    "NeuralRelationalInference",
    "neural_granger_model",
]


class _CLSTMModule(nn.Module):
    def __init__(self, n_vars: int, lag: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.lag = lag
        self.lstms = nn.ModuleList(
            [nn.LSTM(n_vars, hidden, batch_first=True) for _ in range(n_vars)]
        )
        self.sparse_w = nn.Parameter(torch.ones(n_vars, n_vars) * 0.5)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        mask = torch.sigmoid(self.sparse_w)
        for i in range(self.n_vars):
            x_masked = x * mask[i].view(1, 1, -1)
            h, _ = self.lstms[i](x_masked)
            preds.append(self.heads[i](h[:, -1]).squeeze(-1))
        return torch.stack(preds, dim=1)


class NeuralGrangerCLSTM(BaseDeepEstimator):
    """Component-wise LSTM neural Granger with sparse input mask."""

    def __init__(self, lag: int = 5, hidden: int = 32, lambda_sparse: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.lambda_sparse = lambda_sparse

    def fit(self, X) -> NeuralGrangerCLSTM:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_CLSTMModule(n_vars, self.lag, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                loss = F.mse_loss(pred, yb) + self.lambda_sparse * torch.sigmoid(self.module_.sparse_w).abs().sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def adjacency_matrix(self, threshold: float = 0.5) -> np.ndarray:
        self._check_fitted()
        w = torch.sigmoid(self.module_.sparse_w).detach().cpu().numpy()
        return (w >= threshold).astype(float)


class _EconomySRUModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.mask = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        self.gru = nn.GRU(n_vars, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (torch.sigmoid(self.mask) > 0.5).float() * torch.sigmoid(self.mask)
        x_m = x * m.sum(0).view(1, 1, -1)
        h, _ = self.gru(x_m)
        return self.head(h[:, -1])


class NeuralGrangerEconomySRU(BaseDeepEstimator):
    """EconomySRU: structured recurrent unit with learnable causal mask."""

    def __init__(self, lag: int = 5, hidden: int = 32, lambda_sparse: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.lambda_sparse = lambda_sparse

    def fit(self, X) -> NeuralGrangerEconomySRU:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_EconomySRUModule(n_vars, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                loss = F.mse_loss(pred, yb) + self.lambda_sparse * torch.sigmoid(self.module_.mask).abs().sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def adjacency_matrix(self, threshold: float = 0.5) -> np.ndarray:
        self._check_fitted()
        w = torch.sigmoid(self.module_.mask).detach().cpu().numpy()
        return (w >= threshold).astype(float)


class _NRIModule(nn.Module):
    """Simplified Neural Relational Inference encoder-decoder."""

    def __init__(self, n_vars: int, hidden: int, n_edge_types: int = 2):
        super().__init__()
        self.n_vars = n_vars
        self.edge_logits = nn.Parameter(torch.zeros(n_vars, n_vars, n_edge_types))
        self.encoder = nn.GRU(n_vars, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.decoder(h[:, -1])

    def edge_probs(self) -> torch.Tensor:
        return F.softmax(self.edge_logits, dim=-1)


class NeuralRelationalInference(BaseDeepEstimator):
    """Neural Relational Inference for latent graph discovery."""

    def __init__(self, lag: int = 5, hidden: int = 32, n_edge_types: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.n_edge_types = n_edge_types

    def fit(self, X) -> NeuralRelationalInference:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_NRIModule(n_vars, self.hidden, self.n_edge_types), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                kl = -(self.module_.edge_probs() * (self.module_.edge_probs() + 1e-8).log()).sum()
                loss = F.mse_loss(pred, yb) + 0.01 * kl
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        probs = self.module_.edge_probs().detach().cpu().numpy()
        return (probs[..., 1] > 0.5).astype(float)


_NEURAL_GRANGER_MODELS = {
    "cmlp": NeuralGrangerCMLP,
    "clstm": NeuralGrangerCLSTM,
    "economysru": NeuralGrangerEconomySRU,
    "nri": NeuralRelationalInference,
}


def neural_granger_model(method: str = "cmlp", **kwargs) -> BaseDeepEstimator:
    """Factory for neural Granger models (mirrors R ``neural_granger_ml()``)."""
    key = method.lower().replace("-", "_")
    if key not in _NEURAL_GRANGER_MODELS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_NEURAL_GRANGER_MODELS)}.")
    return _NEURAL_GRANGER_MODELS[key](**kwargs)
