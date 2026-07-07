"""Neural Double Machine Learning (partially linear model) for the ATE.

Two-stage residualization with cross-fitting (Chernozhukov et al., 2018):

1. Estimate nuisances :math:`\\hat m(x) = E[Y \\mid X]` and
   :math:`\\hat e(x) = E[T \\mid X]` on held-out folds.
2. Regress the outcome residual on the treatment residual to recover the
   treatment coefficient (the ATE under the partially linear model).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, to_numpy, to_tensor

__all__ = ["NeuralDML"]


class NeuralDML(BaseDeepEstimator):
    """Cross-fitted double machine learning with MLP nuisance learners.

    Parameters
    ----------
    n_splits : int
        Number of cross-fitting folds (>= 2).
    hidden : tuple of int
        Hidden layer sizes for both nuisance networks.

    Attributes
    ----------
    ate_ : float
        Estimated average treatment effect after :meth:`fit`.
    ate_stderr_ : float
        Heteroskedasticity-robust standard error of ``ate_``.
    """

    def __init__(self, n_splits: int = 2, hidden: tuple = (64, 64), standardize: bool = True, **kwargs):
        super().__init__(**kwargs)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 for cross-fitting.")
        self.n_splits = n_splits
        self.hidden = hidden
        self.standardize = standardize

    # ------------------------------------------------------------------ #
    def _fit_nuisance(
        self, x: torch.Tensor, target: torch.Tensor, binary: bool, device: torch.device
    ) -> nn.Module:
        model = MLP(x.shape[1], 1, hidden=self.hidden).to(device)
        loss_fn = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = DataLoader(TensorDataset(x, target), batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss_fn(model(xb).squeeze(-1), yb).backward()
                optimizer.step()
        model.eval()
        return model

    def fit(self, X, treatment, outcome) -> NeuralDML:
        """Estimate the ATE via cross-fitted residual-on-residual regression."""
        device = self._setup()
        x = check_array(X, "X")
        if self.standardize:
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        t = check_binary_treatment(treatment)
        y_raw = to_numpy(outcome).astype(np.float64).ravel()
        # Standardize the outcome for stable nuisance fitting; the final
        # coefficient is rescaled back to the original outcome units.
        y_scale = y_raw.std() + 1e-8
        y = (y_raw - y_raw.mean()) / y_scale

        x_t = to_tensor(x)
        t_t = to_tensor(t)
        y_t = to_tensor(y)

        y_resid = np.zeros(len(y))
        t_resid = np.zeros(len(t))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(x)):
            m_hat = self._fit_nuisance(x_t[tr_idx], y_t[tr_idx], binary=False, device=device)
            e_hat = self._fit_nuisance(x_t[tr_idx], t_t[tr_idx], binary=True, device=device)
            with torch.no_grad():
                xv = x_t[val_idx].to(device)
                y_resid[val_idx] = y[val_idx] - to_numpy(m_hat(xv).squeeze(-1))
                t_resid[val_idx] = t[val_idx] - to_numpy(torch.sigmoid(e_hat(xv).squeeze(-1)))
            self._record(fold=float(fold))

        denom = float(np.dot(t_resid, t_resid))
        if denom == 0.0:
            raise RuntimeError("Treatment residuals are degenerate; cannot identify the ATE.")
        self.ate_ = float(np.dot(t_resid, y_resid) / denom) * y_scale

        # Heteroskedasticity-robust (sandwich) standard error.
        psi = t_resid * (y_resid * y_scale - self.ate_ * t_resid)
        self.ate_stderr_ = float(np.sqrt(np.mean(psi**2)) / (denom / len(psi)) / np.sqrt(len(psi)))

        self._fitted = True
        return self

    def predict_ate(self, X: Optional[object] = None) -> float:
        """Return the cross-fitted ATE estimate (``X`` is accepted for API symmetry)."""
        self._check_fitted()
        return self.ate_

    def confidence_interval(self, level: float = 0.95) -> tuple:
        """Normal-approximation confidence interval for the ATE."""
        self._check_fitted()
        from scipy.stats import norm  # local import; scipy is a sklearn dependency

        z = norm.ppf(0.5 + level / 2)
        return (self.ate_ - z * self.ate_stderr_, self.ate_ + z * self.ate_stderr_)
