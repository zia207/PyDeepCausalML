"""TARNet and CFRNet: representation-based treatment-effect estimators.

TARNet (Shalit, Johansson & Sontag, 2017) learns a shared covariate
representation with two treatment-specific outcome heads. CFRNet augments the
TARNet loss with an integral-probability-metric penalty (here, RBF-kernel MMD)
that aligns the treated and control representation distributions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator, EarlyStopping, mmd_rbf
from ..utils import check_array, check_binary_treatment, to_numpy, to_tensor

__all__ = ["TARNet", "CFRNet"]


class _TARNetModule(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int, head_dim: int):
        super().__init__()
        self.encoder = MLP(input_dim, repr_dim, hidden=(128,), activation=nn.ELU)
        self.head0 = MLP(repr_dim, 1, hidden=(head_dim,), activation=nn.ELU)
        self.head1 = MLP(repr_dim, 1, hidden=(head_dim,), activation=nn.ELU)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return torch.where(t.bool(), y1, y0)

    def potential_outcomes(self, x: torch.Tensor) -> tuple:
        phi = self.encoder(x)
        return self.head0(phi).squeeze(-1), self.head1(phi).squeeze(-1)


class TARNet(BaseDeepEstimator):
    """Treatment-Agnostic Representation Network for CATE estimation.

    Parameters
    ----------
    repr_dim : int
        Width of the shared representation.
    head_dim : int
        Hidden width of each outcome head.
    **kwargs
        Training options forwarded to :class:`~pydeepcausalml.base.BaseDeepEstimator`
        (``epochs``, ``lr``, ``batch_size``, ``device``, ``random_state``, ...).

    Examples
    --------
    >>> from pydeepcausalml.datasets import make_confounded_data
    >>> from pydeepcausalml.effect import TARNet
    >>> df = make_confounded_data(n=2000, random_state=0)
    >>> X = df[["age", "education", "prior_income"]].values
    >>> est = TARNet(epochs=50, random_state=0).fit(X, df["treatment"], df["outcome"])
    >>> cate = est.predict_cate(X)
    """

    def __init__(
        self,
        repr_dim: int = 64,
        head_dim: int = 32,
        standardize: bool = True,
        standardize_outcome: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repr_dim = repr_dim
        self.head_dim = head_dim
        self.standardize = standardize
        self.standardize_outcome = standardize_outcome

    # ------------------------------------------------------------------ #
    def _scale_fit(self, x: np.ndarray) -> np.ndarray:
        if self.standardize:
            self.x_mean_ = x.mean(axis=0)
            self.x_std_ = x.std(axis=0) + 1e-8
            return (x - self.x_mean_) / self.x_std_
        self.x_mean_, self.x_std_ = 0.0, 1.0
        return x

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean_) / self.x_std_ if self.standardize else x

    def _factual_loss(
        self, module: nn.Module, xb: torch.Tensor, tb: torch.Tensor, yb: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(module(xb, tb), yb)

    def fit(self, X, treatment, outcome) -> TARNet:
        """Fit on covariates ``X``, binary ``treatment``, and continuous ``outcome``."""
        device = self._setup()
        x = self._scale_fit(check_array(X, "X"))
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        if self.standardize_outcome:
            self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-8
            y = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_, self.y_std_ = 0.0, 1.0
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        self.module_ = _TARNetModule(x.shape[1], self.repr_dim, self.head_dim).to(device)
        optimizer = self._make_optimizer(self.module_)
        stopper = (
            EarlyStopping(self.early_stopping_patience)
            if self.early_stopping_patience
            else None
        )

        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.module_.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = self._factual_loss(self.module_, xb, tb, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            epoch_loss = total / len(loader)
            self._record(loss=epoch_loss)
            self._log_epoch(epoch, loss=epoch_loss)
            if stopper is not None and stopper.step(epoch_loss):
                break

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def predict_potential_outcomes(self, X) -> tuple:
        """Return ``(y0_hat, y1_hat)`` predictions for each row of ``X``."""
        self._check_fitted()
        x = to_tensor(self._scale(check_array(X, "X")), device=self._device)
        self.module_.eval()
        with torch.no_grad():
            y0, y1 = self.module_.potential_outcomes(x)
        y0 = to_numpy(y0) * self.y_std_ + self.y_mean_
        y1 = to_numpy(y1) * self.y_std_ + self.y_mean_
        return y0, y1

    def predict_cate(self, X) -> np.ndarray:
        """Predict the conditional average treatment effect for each row of ``X``."""
        y0, y1 = self.predict_potential_outcomes(X)
        return y1 - y0

    def predict_ate(self, X) -> float:
        """Predict the average treatment effect over ``X``."""
        return float(self.predict_cate(X).mean())


class CFRNet(TARNet):
    """CounterFactual Regression network: TARNet plus MMD representation balancing.

    Parameters
    ----------
    alpha : float
        Weight of the MMD penalty between treated and control representations.
    mmd_bandwidth : float
        RBF kernel bandwidth used by the MMD term.
    """

    def __init__(self, alpha: float = 0.5, mmd_bandwidth: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.mmd_bandwidth = mmd_bandwidth

    def _factual_loss(
        self, module: nn.Module, xb: torch.Tensor, tb: torch.Tensor, yb: torch.Tensor
    ) -> torch.Tensor:
        phi = module.encoder(xb)
        y0 = module.head0(phi).squeeze(-1)
        y1 = module.head1(phi).squeeze(-1)
        pred = torch.where(tb.bool(), y1, y0)
        loss = nn.functional.mse_loss(pred, yb)

        phi_t, phi_c = phi[tb.bool()], phi[~tb.bool()]
        if len(phi_t) > 1 and len(phi_c) > 1:
            loss = loss + self.alpha * mmd_rbf(phi_t, phi_c, self.mmd_bandwidth)
        return loss
