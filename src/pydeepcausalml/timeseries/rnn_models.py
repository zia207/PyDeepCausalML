"""RNN-based causal models: RETAIN, InterventionAwareRNN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..timeseries.forecasting import CausalLSTMForecaster
from ..timeseries.granger import make_lagged_sequences
from ..utils import check_array, module_to_device, to_numpy, to_tensor

__all__ = ["CausalLSTM", "RETAIN", "InterventionAwareRNN", "rnn_causal_model"]


class _CausalLSTMModule(nn.Module):
    """Stacked LSTM with learnable soft causal-adjacency mask."""

    def __init__(self, n_vars: int, hidden: int, n_layers: int = 2):
        super().__init__()
        self.n_vars = n_vars
        self.adj = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.lstm = nn.LSTM(n_vars, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.adj)
        x_m = x * mask.sum(0).view(1, 1, -1)
        h, _ = self.lstm(x_m)
        return self.head(h[:, -1])


class CausalLSTM(BaseDeepEstimator):
    """CausalLSTM with learnable soft adjacency mask and L1 sparsity."""

    def __init__(self, lag: int = 10, hidden: int = 32, n_layers: int = 2, lambda_sparse: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.n_layers = n_layers
        self.lambda_sparse = lambda_sparse

    def fit(self, X) -> CausalLSTM:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_CausalLSTMModule(n_vars, self.hidden, self.n_layers), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                loss = F.mse_loss(pred, yb) + self.lambda_sparse * torch.sigmoid(self.module_.adj).abs().sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        self.module_.eval()
        with torch.no_grad():
            return to_numpy(self.module_(xin.to(device)))

    def causal_matrix(self, threshold: float = 0.5) -> np.ndarray:
        self._check_fitted()
        w = torch.sigmoid(self.module_.adj).detach().cpu().numpy()
        return (w >= threshold).astype(float)


class _RETAINModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.n_vars = n_vars
        self.gru = nn.GRU(n_vars, hidden, batch_first=True)
        self.alpha = nn.Linear(hidden, 1)
        self.beta = nn.Linear(hidden, n_vars)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x.flip(1))
        h = h.flip(1)
        a = torch.softmax(self.alpha(h).squeeze(-1), dim=1)
        context = (a.unsqueeze(-1) * h).sum(1)
        return self.head(context)


class RETAIN(BaseDeepEstimator):
    """RETAIN: reverse-time GRU with temporal and variable attention (Choi et al., 2016)."""

    def __init__(self, lag: int = 10, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X) -> RETAIN:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_RETAINModule(n_vars, self.hidden), device)
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

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        self.module_.eval()
        with torch.no_grad():
            return to_numpy(self.module_(xin.to(device)))

    def causal_matrix(self) -> np.ndarray:
        self._check_fitted()
        w = self.module_.beta.weight.detach().cpu().numpy()
        return np.abs(w).mean(0, keepdims=True).T @ np.ones((1, self.module_.n_vars))


class _InterventionRNNModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int, n_regimes: int = 2):
        super().__init__()
        self.n_vars = n_vars
        self.lstm = nn.LSTM(n_vars + 1, hidden, batch_first=True)
        self.regime = nn.Linear(hidden, n_regimes)
        self.adj = nn.Parameter(torch.zeros(n_regimes, n_vars, n_vars))
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor, intervention: torch.Tensor | None = None) -> torch.Tensor:
        if intervention is None:
            intervention = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)
        h, _ = self.lstm(torch.cat([x, intervention], dim=-1))
        return self.head(h[:, -1])


class InterventionAwareRNN(BaseDeepEstimator):
    """LSTM with soft regime detector and intervention channel."""

    def __init__(self, lag: int = 10, hidden: int = 32, n_regimes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.n_regimes = n_regimes

    def fit(self, X, intervention=None) -> InterventionAwareRNN:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_InterventionRNNModule(n_vars, self.hidden, self.n_regimes), device)
        optimizer = self._make_optimizer(self.module_)
        if intervention is not None:
            inter = to_tensor(intervention).reshape(-1, self.lag, 1)
        else:
            inter = torch.zeros(xin.shape[0], self.lag, 1)
        loader = DataLoader(TensorDataset(xin, inter, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, ib, yb in loader:
                xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(self.module_(xb, ib), yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict(self, X, intervention=None) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        if intervention is not None:
            inter = to_tensor(intervention, device=device).reshape(-1, self.lag, 1)
        else:
            inter = torch.zeros(xin.shape[0], self.lag, 1, device=device)
        self.module_.eval()
        with torch.no_grad():
            return to_numpy(self.module_(xin.to(device), inter))

    def causal_matrix(self, regime: int = 0) -> np.ndarray:
        self._check_fitted()
        return torch.sigmoid(self.module_.adj[regime]).detach().cpu().numpy()


_RNN_MODELS = {
    "causal_lstm": CausalLSTM,
    "retain": RETAIN,
    "intervention_rnn": InterventionAwareRNN,
    "causal_lstm_forecaster": CausalLSTMForecaster,
}


def rnn_causal_model(method: str = "causal_lstm", **kwargs) -> BaseDeepEstimator:
    """Factory for RNN-based causal models (mirrors R ``rnn_causal_model()``)."""
    key = method.lower().replace("-", "_")
    if key not in _RNN_MODELS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_RNN_MODELS)}.")
    return _RNN_MODELS[key](**kwargs)
