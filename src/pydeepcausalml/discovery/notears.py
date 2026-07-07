"""NOTEARS: DAG structure learning via a smooth acyclicity constraint.

Implements the linear NOTEARS model (Zheng et al., 2018) with the continuous
acyclicity characterization :math:`h(W) = \\mathrm{tr}(e^{W \\odot W}) - d = 0`,
optimized by an augmented Lagrangian with Adam inner steps.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..base import BaseDeepEstimator
from ..utils import check_array, to_numpy, to_tensor

__all__ = ["NOTEARSLinear", "notears_acyclicity"]


def notears_acyclicity(w: torch.Tensor) -> torch.Tensor:
    """Smooth acyclicity penalty ``tr(exp(W*W)) - d`` (zero iff W encodes a DAG).

    Evaluated in float64 on CPU for numerical stability of ``matrix_exp``;
    gradients still flow to ``w``.
    """
    d = w.shape[0]
    m = (w * w).to(device=torch.device("cpu"), dtype=torch.float64)
    h = torch.trace(torch.linalg.matrix_exp(m)) - float(d)
    return h.to(device=w.device, dtype=w.dtype)


class NOTEARSLinear(BaseDeepEstimator):
    """Linear structural equation DAG learner.

    Model: :math:`X = X W + \\varepsilon`, with ``W[j, i]`` the weight of the
    edge :math:`X_j \\to X_i`. The recovered adjacency is reported in the
    package convention ``A[i, j] = 1`` meaning :math:`X_j \\to X_i`.

    Parameters
    ----------
    lambda_l1 : float
        L1 sparsity weight on ``W``.
    rho_init, rho_max : float
        Initial and maximum penalty coefficient of the augmented Lagrangian.
    h_tol : float
        Target tolerance on the acyclicity constraint.
    n_outer : int
        Number of augmented-Lagrangian outer iterations.
    """

    def __init__(
        self,
        lambda_l1: float = 0.05,
        rho_init: float = 1.0,
        rho_max: float = 1e8,
        h_tol: float = 1e-6,
        n_outer: int = 12,
        standardize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 200)
        kwargs.setdefault("lr", 1e-2)
        super().__init__(**kwargs)
        self.lambda_l1 = lambda_l1
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.h_tol = h_tol
        self.n_outer = n_outer
        self.standardize = standardize

    def fit(self, X) -> NOTEARSLinear:
        """Learn a weighted DAG from an (n, d) data matrix."""
        device = self._setup()
        x = check_array(X, "X")
        if self.standardize:
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        n, d = x.shape
        x_t = to_tensor(x, device=device)

        w = nn.Parameter(torch.zeros(d, d, device=device))
        rho, alpha = self.rho_init, 0.0
        h_val = float("inf")

        for outer in range(self.n_outer):
            optimizer = torch.optim.Adam([w], lr=self.lr)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                resid = x_t - x_t @ w
                loss_fit = 0.5 * (resid**2).sum() / n
                h = notears_acyclicity(w)
                loss = loss_fit + self.lambda_l1 * w.abs().sum() + 0.5 * rho * h**2 + alpha * h
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    w.fill_diagonal_(0.0)

            with torch.no_grad():
                h_val = float(notears_acyclicity(w))
            self._record(h=abs(h_val), rho=rho)
            self._log_epoch(outer, h=abs(h_val), rho=rho)

            if abs(h_val) <= self.h_tol:
                break
            alpha += rho * h_val
            rho = min(rho * 10.0, self.rho_max)

        # Package convention: A[i, j] means X_j -> X_i, so transpose W (source, target).
        self.weights_ = to_numpy(w.detach()).T
        self._fitted = True
        return self

    def get_adjacency(self, threshold: float = 0.3) -> np.ndarray:
        """Binary adjacency (``A[i, j] = 1`` means :math:`X_j \\to X_i`)."""
        self._check_fitted()
        a = (np.abs(self.weights_) > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a

    def adjacency_matrix(self, threshold: float = 0.3) -> np.ndarray:
        """Alias for :meth:`get_adjacency`."""
        return self.get_adjacency(threshold)
