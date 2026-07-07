"""DAGMA: DAG learning via M-matrix log-det acyclicity (Bello et al., 2022)."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseDeepEstimator, MLP
from ..utils import check_array, module_to_device, to_numpy, to_tensor

__all__ = ["DagmaLinear", "DagmaNonlinearMLP"]


class DagmaLinear(BaseDeepEstimator):
    """Linear DAGMA structure learning (numpy/scipy optimization)."""

    def __init__(
        self,
        loss_type: str = "l2",
        lambda1: float = 0.02,
        max_iter: int = 1000,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 1)
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.lambda1 = lambda1
        self.max_iter = max_iter

    def _h(self, w: np.ndarray, s: float = 1.0) -> tuple[float, np.ndarray]:
        d = w.shape[0]
        eye = np.eye(d)
        m = s * eye - w * w
        h = -np.linalg.slogdet(m)[1] + d * np.log(s)
        g_h = 2 * w * sla.inv(m).T
        return h, g_h

    def _score(self, w: np.ndarray, cov: np.ndarray, n: int, x: np.ndarray | None) -> tuple[float, np.ndarray]:
        d = w.shape[0]
        eye = np.eye(d)
        if self.loss_type == "l2":
            dif = eye - w
            rhs = cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            g_loss = -rhs
        else:
            r = x @ w
            loss = (np.logaddexp(0, r) - x * r).sum() / n
            g_loss = (x.T @ (1 / (1 + np.exp(-r)))) / n - cov
        return loss, g_loss

    def fit(self, X) -> DagmaLinear:
        self._setup()
        x = check_array(X, "X")
        n, d = x.shape
        cov = (x.T @ x) / n
        w = np.zeros((d, d))
        mu = 10.0
        tau = 0.0
        lr = 0.0003
        h_prev = np.inf

        for _ in range(self.max_iter):
            loss, g_loss = self._score(w, cov, n, x if self.loss_type == "logistic" else None)
            h, g_h = self._h(w)
            g_obj = g_loss + (mu * h + tau) * g_h + self.lambda1 * np.sign(w)
            w -= lr * g_obj
            if h < 1e-8:
                break
            if h > 0.25 * h_prev:
                mu *= 10
            tau = mu * h
            h_prev = h

        self.adjacency_ = w
        self._fitted = True
        return self

    def adjacency_matrix(self) -> np.ndarray:
        self._check_fitted()
        return self.adjacency_.copy()


class _DagmaMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.d = d
        self.adj = nn.Parameter(torch.zeros(d, d))
        self.mlps = nn.ModuleList([MLP(d, 1, hidden=(hidden, hidden), activation=nn.ELU) for _ in range(d)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for j in range(self.d):
            outs.append(self.mlps[j](x))
        return torch.cat(outs, dim=-1)

    def h(self) -> torch.Tensor:
        w = torch.sigmoid(self.adj) * (1 - torch.eye(self.d, device=self.adj.device))
        m = torch.eye(self.d, device=w.device) - w * w
        return -torch.linalg.slogdet(m)[1]


class DagmaNonlinearMLP(BaseDeepEstimator):
    """Nonlinear DAGMA with per-node MLP structural equations."""

    def __init__(self, hidden: int = 32, lambda1: float = 0.02, mu_init: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.lambda1 = lambda1
        self.mu_init = mu_init

    def fit(self, X) -> DagmaNonlinearMLP:
        device = self._setup()
        x = check_array(X, "X")
        n, d = x.shape
        self.module_ = module_to_device(_DagmaMLP(d, self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        x_t = to_tensor(x, device=device)
        mu = self.mu_init

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            x_hat = self.module_(x_t)
            recon = F.mse_loss(x_hat, x_t)
            h = self.module_.h()
            loss = recon + 0.5 * mu * h.pow(2) + self.lambda1 * torch.sigmoid(self.module_.adj).abs().sum()
            loss.backward()
            optimizer.step()
            if h.item() < 1e-6:
                mu *= 2
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
