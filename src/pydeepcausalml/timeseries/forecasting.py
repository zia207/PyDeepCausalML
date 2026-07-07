"""Causal forecasting: counterfactual multi-step prediction under interventions.

:class:`CausalLSTMForecaster` uses a dual-stream LSTM — one encoder for the
outcome history and one for the treatment/intervention history — merged into
a multi-step decoder. Feeding an alternative treatment sequence yields a
counterfactual forecast, and the factual/counterfactual gap estimates the
intervention effect over the forecast horizon.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator, EarlyStopping
from ..utils import to_numpy, to_tensor

__all__ = ["CausalLSTMForecaster"]


class _CausalLSTMModule(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int):
        super().__init__()
        self.outcome_lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.treatment_lstm = nn.LSTM(1, hidden_dim // 2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len),
        )

    def forward(self, x_hist: torch.Tensor, t_hist: torch.Tensor) -> torch.Tensor:
        _, (h_out, _) = self.outcome_lstm(x_hist)
        _, (h_trt, _) = self.treatment_lstm(t_hist)
        combined = torch.cat([h_out[-1], h_trt[-1]], dim=-1)
        return self.decoder(combined)


class CausalLSTMForecaster(BaseDeepEstimator):
    """Dual-stream LSTM for factual and counterfactual multi-step forecasting.

    Parameters
    ----------
    pred_len : int
        Forecast horizon (number of future steps).
    hidden_dim : int
        Outcome-encoder width (treatment encoder uses ``hidden_dim // 2``).

    Notes
    -----
    Inputs to :meth:`fit` are 3-D arrays shaped ``(n_units, seq_len, 1)``
    for both the outcome history and the treatment history, and a 2-D
    target array ``(n_units, pred_len)``. 2-D histories are auto-expanded.
    """

    def __init__(self, pred_len: int = 12, hidden_dim: int = 64, **kwargs):
        kwargs.setdefault("epochs", 100)
        kwargs.setdefault("batch_size", 32)
        super().__init__(**kwargs)
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

    @staticmethod
    def _as_3d(x) -> torch.Tensor:
        t = to_tensor(x)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        if t.dim() != 3:
            raise ValueError(f"Expected 2-D or 3-D history, got shape {tuple(t.shape)}.")
        return t

    def fit(self, outcome_history, treatment_history, future_outcomes) -> CausalLSTMForecaster:
        """Train on unit-level histories and observed future outcomes."""
        device = self._setup()
        x = self._as_3d(outcome_history)
        t = self._as_3d(treatment_history)
        y = to_tensor(future_outcomes)
        if y.dim() != 2 or y.shape[1] != self.pred_len:
            raise ValueError(
                f"future_outcomes must be (n_units, pred_len={self.pred_len}); got {tuple(y.shape)}."
            )
        if not (len(x) == len(t) == len(y)):
            raise ValueError("All inputs must contain the same number of units.")

        self.module_ = _CausalLSTMModule(self.hidden_dim, self.pred_len).to(device)
        optimizer = self._make_optimizer(self.module_)
        criterion = nn.MSELoss()
        stopper = (
            EarlyStopping(self.early_stopping_patience)
            if self.early_stopping_patience
            else None
        )
        loader = DataLoader(TensorDataset(x, t, y), batch_size=self.batch_size, shuffle=True)

        self.module_.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.module_(xb, tb), yb)
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

    def forecast(self, outcome_history, treatment_history) -> np.ndarray:
        """Multi-step forecast under the supplied treatment history."""
        self._check_fitted()
        x = self._as_3d(outcome_history).to(self._device)
        t = self._as_3d(treatment_history).to(self._device)
        self.module_.eval()
        with torch.no_grad():
            return to_numpy(self.module_(x, t))

    def forecast_counterfactual(self, outcome_history, counterfactual_treatment) -> np.ndarray:
        """Forecast under a hypothetical (do-)treatment sequence."""
        return self.forecast(outcome_history, counterfactual_treatment)

    def estimate_effect(
        self, outcome_history, factual_treatment, counterfactual_treatment
    ) -> np.ndarray:
        """Per-unit, per-step effect: counterfactual minus factual forecast."""
        factual = self.forecast(outcome_history, factual_treatment)
        counterfactual = self.forecast(outcome_history, counterfactual_treatment)
        return counterfactual - factual
