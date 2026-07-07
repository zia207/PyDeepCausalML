"""Neural Granger causality models.

Two complementary approaches from the tutorial series:

* :class:`NeuralGrangerCMLP` — component-wise MLPs with a group-LASSO penalty
  on per-(target, source) lag groups; zeroed groups indicate absent edges
  (Tank et al., 2021).
* :class:`GrangerLSTM` — a full-versus-reduced ablation test: mask one source
  channel at a time and measure the increase in held-out prediction error.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..utils import check_array, to_numpy, to_tensor

__all__ = ["NeuralGrangerCMLP", "GrangerLSTM", "make_lagged_sequences"]


def make_lagged_sequences(
    x: np.ndarray, lag: int, standardize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Build (X[t-lag:t], X[t]) supervised pairs from a (T, p) series.

    Returns input tensor (N, lag, p), target tensor (N, p), and the
    mean/std used for standardization.
    """
    x = check_array(x, "X")
    mean = x.mean(axis=0) if standardize else np.zeros(x.shape[1])
    std = x.std(axis=0) + 1e-8 if standardize else np.ones(x.shape[1])
    xs = (x - mean) / std
    t_len = xs.shape[0]
    if t_len <= lag:
        raise ValueError("Series is too short for the requested lag.")
    xin = np.stack([xs[t - lag : t] for t in range(lag, t_len)])
    xout = xs[lag:]
    return to_tensor(xin), to_tensor(xout), mean, std


class _CMLPModule(nn.Module):
    """Component-wise MLP with group-sparse first-layer weights."""

    def __init__(self, n_vars: int, lag: int, hidden_dim: int):
        super().__init__()
        self.n_vars = n_vars
        self.lag = lag
        self.hidden = hidden_dim
        self.w1 = nn.Parameter(torch.randn(n_vars, n_vars, lag) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_vars, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_vars, hidden_dim, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_vars))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lag, n_vars)
        batch = x.shape[0]
        x_flat = x.permute(0, 2, 1).reshape(batch, -1)  # (batch, n_vars*lag)
        preds = []
        for i in range(self.n_vars):
            proj = self.w1[i].reshape(-1, 1).expand(-1, self.hidden)
            h = torch.relu(x_flat @ proj + self.b1[i])
            preds.append((h @ self.w2[i]).squeeze(-1) + self.b2[i])
        return torch.stack(preds, dim=1)

    def group_norms(self) -> torch.Tensor:
        return self.w1.pow(2).sum(dim=2).sqrt()


class NeuralGrangerCMLP(BaseDeepEstimator):
    """Component-wise MLP neural Granger causality with group LASSO.

    Parameters
    ----------
    lag : int
        Autoregressive window length.
    hidden_dim : int
        Hidden width per component MLP.
    lambda_group : float
        Group-LASSO penalty weight on (target, source) lag groups.
    """

    def __init__(self, lag: int = 5, hidden_dim: int = 32, lambda_group: float = 0.01, **kwargs):
        kwargs.setdefault("epochs", 200)
        kwargs.setdefault("lr", 5e-4)
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.lambda_group = lambda_group

    def fit(self, X) -> NeuralGrangerCMLP:
        """Fit on a (T, p) multivariate time series."""
        device = self._setup()
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(X, self.lag)
        self.n_vars_ = xin.shape[2]

        self.module_ = _CMLPModule(self.n_vars_, self.lag, self.hidden_dim).to(device)
        optimizer = self._make_optimizer(self.module_)
        mse = nn.MSELoss()
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        self.module_.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = mse(self.module_(xb), yb) + self.lambda_group * self.module_.group_norms().sum()
                loss.backward()
                optimizer.step()
                total += loss.item()
            self._record(loss=total / len(loader))
            self._log_epoch(epoch, loss=total / len(loader))

        with torch.no_grad():
            self.scores_ = to_numpy(self.module_.group_norms())
        self._fitted = True
        return self

    def get_scores(self) -> np.ndarray:
        """Group-norm causal-score matrix (rows = targets, columns = sources)."""
        self._check_fitted()
        return self.scores_.copy()

    def get_adjacency(self, threshold: float = 0.05) -> np.ndarray:
        """Binary adjacency (``A[i, j] = 1`` means :math:`X_j \\to X_i`)."""
        self._check_fitted()
        a = (self.scores_ > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a

    def adjacency_matrix(self, threshold: float = 0.05) -> np.ndarray:
        """Alias for :meth:`get_adjacency` (consistent with other estimators)."""
        return self.get_adjacency(threshold)


class _LSTMForecaster(nn.Module):
    def __init__(self, n_vars: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_vars, hidden_dim, n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, n_vars)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask.float().unsqueeze(0).unsqueeze(0)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GrangerLSTM(BaseDeepEstimator):
    """Ablation-based nonlinear Granger causality with an LSTM forecaster.

    Trains a full model on all channels, then per-source reduced models with
    the source masked; the score ``F[i, j]`` is the increase in test MSE for
    target *i* when source *j* is unavailable.

    Parameters
    ----------
    lag : int
        Input window length.
    hidden_dim, n_layers : int
        LSTM capacity.
    test_fraction : float
        Fraction of sequences held out to measure the error increase.
    """

    def __init__(
        self,
        lag: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 2,
        test_fraction: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 80)
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.test_fraction = test_fraction

    def _train_one(
        self, xin: torch.Tensor, xout: torch.Tensor, device: torch.device
    ) -> nn.Module:
        model = _LSTMForecaster(self.n_vars_, self.hidden_dim, self.n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        return model

    def fit(self, X) -> GrangerLSTM:
        """Fit full and per-source reduced models on a (T, p) series."""
        device = self._setup()
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(X, self.lag)
        self.n_vars_ = xin.shape[2]

        n_train = int((1 - self.test_fraction) * len(xin))
        if n_train < 1 or n_train >= len(xin):
            raise ValueError("test_fraction leaves no data for training or evaluation.")
        xin_tr, xin_te = xin[:n_train], xin[n_train:].to(device)
        xout_tr, xout_te = xout[:n_train], xout[n_train:].to(device)

        criterion = nn.MSELoss(reduction="none")
        full_model = self._train_one(xin_tr, xout_tr, device)
        with torch.no_grad():
            full_mse = to_numpy(criterion(full_model(xin_te), xout_te).mean(dim=0))

        scores = np.zeros((self.n_vars_, self.n_vars_))
        for j in range(self.n_vars_):
            mask = torch.ones(self.n_vars_, dtype=torch.bool, device=device)
            mask[j] = False
            reduced = self._train_one(xin_tr * mask.cpu().float(), xout_tr, device)
            with torch.no_grad():
                red_mse = to_numpy(criterion(reduced(xin_te, mask=mask), xout_te).mean(dim=0))
            scores[:, j] = red_mse - full_mse
            self._record(masked_source=float(j))

        self.scores_ = scores
        self._fitted = True
        return self

    def get_scores(self) -> np.ndarray:
        """Error-increase score matrix ``F`` (rows = targets, columns = sources)."""
        self._check_fitted()
        return self.scores_.copy()

    def get_adjacency(self, quantile: float = 0.6) -> np.ndarray:
        """Threshold positive score increases at the given quantile."""
        self._check_fitted()
        positive = self.scores_[self.scores_ > 0]
        if positive.size == 0:
            return np.zeros_like(self.scores_, dtype=int)
        threshold = np.quantile(positive, quantile)
        a = (self.scores_ > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a

    def adjacency_matrix(self, quantile: float = 0.6) -> np.ndarray:
        """Alias for :meth:`get_adjacency` (consistent with other estimators)."""
        return self.get_adjacency(quantile)
