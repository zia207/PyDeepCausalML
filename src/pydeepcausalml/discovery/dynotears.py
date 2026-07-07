"""DYNOTEARS-style lagged causal discovery with a DAG constraint.

Learns one weight matrix per lag; the summed absolute lag matrices are
constrained toward acyclicity with the NOTEARS penalty inside an augmented
Lagrangian, as in Pamfil et al. (2020) and the accompanying tutorial series.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..utils import check_array, to_numpy, to_tensor
from .notears import notears_acyclicity

__all__ = ["DynoTEARS"]


class _DynoModule(nn.Module):
    def __init__(self, n_vars: int, lag: int):
        super().__init__()
        self.n_vars = n_vars
        self.lag = lag
        self.w_lags = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_vars, n_vars)) for _ in range(lag)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lag, n_vars); w_lags[k][i, j] = effect of X_j(t-k-1) on X_i(t)
        pred = torch.zeros(x.shape[0], self.n_vars, device=x.device)
        for k, w in enumerate(self.w_lags):
            pred = pred + x[:, self.lag - 1 - k, :] @ w.T
        return pred

    def aggregate(self) -> torch.Tensor:
        return sum(w.abs() for w in self.w_lags)


class DynoTEARS(BaseDeepEstimator):
    """Lag-resolved linear causal discovery with acyclicity constraint.

    Parameters
    ----------
    lag : int
        Number of autoregressive lags.
    lambda_l1 : float
        L1 sparsity weight across all lag matrices.
    rho_init : float
        Initial augmented-Lagrangian penalty coefficient.
    rho_update_every : int
        Epoch interval at which ``rho`` doubles (capped at ``1e4``).
    """

    def __init__(
        self,
        lag: int = 5,
        lambda_l1: float = 0.02,
        rho_init: float = 1.0,
        rho_update_every: int = 50,
        standardize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 300)
        kwargs.setdefault("lr", 3e-3)
        kwargs.setdefault("batch_size", 256)
        super().__init__(**kwargs)
        self.lag = lag
        self.lambda_l1 = lambda_l1
        self.rho_init = rho_init
        self.rho_update_every = rho_update_every
        self.standardize = standardize

    def _sequences(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        t_len = x.shape[0]
        xin = np.stack([x[t - self.lag : t] for t in range(self.lag, t_len)])
        xout = x[self.lag :]
        return to_tensor(xin), to_tensor(xout)

    def fit(self, X) -> DynoTEARS:
        """Learn lagged causal structure from a (T, p) multivariate series."""
        device = self._setup()
        x = check_array(X, "X")
        if x.shape[0] <= self.lag + 1:
            raise ValueError("Series is too short for the requested lag.")
        if self.standardize:
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        self.n_vars_ = x.shape[1]

        xin, xout = self._sequences(x)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        self.module_ = _DynoModule(self.n_vars_, self.lag).to(device)
        optimizer = self._make_optimizer(self.module_)
        criterion = nn.MSELoss()
        rho, alpha = self.rho_init, 0.0

        self.module_.train()
        for epoch in range(self.epochs):
            mse_total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                h = notears_acyclicity(self.module_.aggregate())
                mse = criterion(self.module_(xb), yb)
                loss = (
                    mse
                    + self.lambda_l1 * sum(w.abs().sum() for w in self.module_.w_lags)
                    + 0.5 * rho * h**2
                    + alpha * h
                )
                loss.backward()
                optimizer.step()
                mse_total += mse.item()

            with torch.no_grad():
                h_epoch = float(notears_acyclicity(self.module_.aggregate()))
            alpha += rho * h_epoch
            if epoch > 0 and epoch % self.rho_update_every == 0:
                rho = min(rho * 2.0, 1e4)

            self._record(mse=mse_total / len(loader), h=abs(h_epoch))
            self._log_epoch(epoch, mse=mse_total / len(loader), h=abs(h_epoch))

        with torch.no_grad():
            self.weights_ = to_numpy(self.module_.aggregate())
            self.lag_weights_ = [to_numpy(w.detach()) for w in self.module_.w_lags]
        self._fitted = True
        return self

    def get_adjacency(self, threshold: float = 0.1) -> np.ndarray:
        """Binary adjacency aggregated over lags (``A[i, j]``: :math:`X_j \\to X_i`)."""
        self._check_fitted()
        a = (self.weights_ > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a

    def adjacency_matrix(self, threshold: float = 0.1) -> np.ndarray:
        """Alias for :meth:`get_adjacency`."""
        return self.get_adjacency(threshold)

    def get_scores(self) -> np.ndarray:
        """Aggregated |weight| causal-score matrix over all lags."""
        self._check_fitted()
        return self.weights_.copy()
