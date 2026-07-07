"""DSCM: Deep Structural Causal Model with fixed DAG."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["DSCM"]


class _DSCMModule(nn.Module):
    """Fixed DAG: X -> T -> Y with per-variable structural equations."""

    def __init__(self, x_dim: int, hidden: int):
        super().__init__()
        self.treatment_net = MLP(x_dim, 1, hidden=(hidden,), activation=nn.ELU)
        self.outcome_net = MLP(x_dim + 1, 1, hidden=(hidden, hidden), activation=nn.ELU)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        t_logit = self.treatment_net(x).squeeze(-1)
        if t is None:
            t = torch.sigmoid(t_logit)
        y = self.outcome_net(torch.cat([x, t.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return t_logit, y

    def counterfactual_y(self, x: torch.Tensor, do_t: float) -> torch.Tensor:
        t = torch.full((x.shape[0], 1), do_t, device=x.device, dtype=x.dtype)
        return self.outcome_net(torch.cat([x, t], dim=-1)).squeeze(-1)


class DSCM(BaseDeepEstimator):
    """Deep Structural Causal Model for counterfactual outcome prediction."""

    def __init__(self, hidden: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden

    def fit(self, X, treatment, outcome) -> DSCM:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        self.module_ = module_to_device(_DSCMModule(x.shape[1], self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                t_logit, y_hat = self.module_(xb, tb)
                loss = F.binary_cross_entropy_with_logits(t_logit, tb) + F.mse_loss(y_hat, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_potential_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        self.module_.eval()
        with torch.no_grad():
            y0 = self.module_.counterfactual_y(to_tensor(x, device=device), 0.0)
            y1 = self.module_.counterfactual_y(to_tensor(x, device=device), 1.0)
        return to_numpy(y0), to_numpy(y1)

    def predict_cate(self, X) -> np.ndarray:
        y0, y1 = self.predict_potential_outcomes(X)
        return y1 - y0

    def predict_ate(self, X) -> float:
        return float(self.predict_cate(X).mean())
