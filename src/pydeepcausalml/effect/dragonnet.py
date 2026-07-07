"""DragonNet: propensity-integrated treatment-effect estimation.

DragonNet (Shi, Blei & Veitch, 2019) shares a covariate representation across
two outcome heads and a propensity head, with a targeted-regularization term
that pushes the estimator toward double robustness.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator, EarlyStopping
from ..utils import check_array, check_binary_treatment, to_numpy, to_tensor

__all__ = ["DragonNet"]


class _DragonNetModule(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int, head_dim: int):
        super().__init__()
        self.encoder = MLP(input_dim, repr_dim, hidden=(128,), activation=nn.ELU)
        self.head0 = MLP(repr_dim, 1, hidden=(head_dim,), activation=nn.ELU)
        self.head1 = MLP(repr_dim, 1, hidden=(head_dim,), activation=nn.ELU)
        self.prop_head = MLP(
            repr_dim, 1, hidden=(head_dim,), activation=nn.ELU, final_activation=nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple:
        phi = self.encoder(x)
        return (
            self.head0(phi).squeeze(-1),
            self.head1(phi).squeeze(-1),
            self.prop_head(phi).squeeze(-1),
        )


class DragonNet(BaseDeepEstimator):
    """DragonNet estimator of heterogeneous treatment effects.

    Parameters
    ----------
    repr_dim, head_dim : int
        Widths of the shared representation and per-head hidden layers.
    alpha : float
        Weight of the propensity (BCE) loss.
    beta : float
        Weight of the targeted-regularization term.
    standardize : bool
        Standardize covariates before training.
    """

    def __init__(
        self,
        repr_dim: int = 64,
        head_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 0.1,
        standardize: bool = True,
        standardize_outcome: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repr_dim = repr_dim
        self.head_dim = head_dim
        self.alpha = alpha
        self.beta = beta
        self.standardize = standardize
        self.standardize_outcome = standardize_outcome

    def fit(self, X, treatment, outcome) -> DragonNet:
        """Fit on covariates ``X``, binary ``treatment``, and continuous ``outcome``."""
        device = self._setup()
        x = check_array(X, "X")
        if self.standardize:
            self.x_mean_ = x.mean(axis=0)
            self.x_std_ = x.std(axis=0) + 1e-8
            x = (x - self.x_mean_) / self.x_std_
        else:
            self.x_mean_, self.x_std_ = 0.0, 1.0
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        if self.standardize_outcome:
            self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-8
            y = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_, self.y_std_ = 0.0, 1.0

        self.module_ = _DragonNetModule(x.shape[1], self.repr_dim, self.head_dim).to(device)
        optimizer = self._make_optimizer(self.module_)
        stopper = (
            EarlyStopping(self.early_stopping_patience)
            if self.early_stopping_patience
            else None
        )
        mse, bce = nn.MSELoss(), nn.BCELoss()
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
                y0, y1, e = self.module_(xb)
                y_hat = torch.where(tb.bool(), y1, y0)

                loss_out = mse(y_hat, yb)
                loss_prop = bce(e.clamp(1e-6, 1 - 1e-6), tb)

                eps = 1e-7
                h1 = tb / (e + eps)
                h0 = (1 - tb) / (1 - e + eps)
                loss_targ = ((h1 * (yb - y1)) ** 2 + (h0 * (yb - y0)) ** 2).mean()

                loss = loss_out + self.alpha * loss_prop + self.beta * loss_targ
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

    def _transform(self, X) -> torch.Tensor:
        x = check_array(X, "X")
        x = (x - self.x_mean_) / self.x_std_ if self.standardize else x
        return to_tensor(x, device=self._device)

    def predict_potential_outcomes(self, X) -> tuple:
        """Return ``(y0_hat, y1_hat)`` predictions."""
        self._check_fitted()
        self.module_.eval()
        with torch.no_grad():
            y0, y1, _ = self.module_(self._transform(X))
        y0 = to_numpy(y0) * self.y_std_ + self.y_mean_
        y1 = to_numpy(y1) * self.y_std_ + self.y_mean_
        return y0, y1

    def predict_cate(self, X) -> np.ndarray:
        """Predict the conditional average treatment effect per row."""
        y0, y1 = self.predict_potential_outcomes(X)
        return y1 - y0

    def predict_ate(self, X) -> float:
        """Predict the average treatment effect over ``X``."""
        return float(self.predict_cate(X).mean())

    def predict_propensity(self, X) -> np.ndarray:
        """Predict treatment propensity scores :math:`e(x) = P(T=1 \\mid X=x)`."""
        self._check_fitted()
        self.module_.eval()
        with torch.no_grad():
            _, _, e = self.module_(self._transform(X))
        return to_numpy(e)
