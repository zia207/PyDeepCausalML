"""Attention-based causal models: CausalTransformer, TFTNet."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..timeseries.granger import make_lagged_sequences
from ..utils import check_array, module_to_device, to_numpy, to_tensor

__all__ = ["CausalTransformer", "TFTNet", "attn_causal_model"]


class _CausalTransformerModule(nn.Module):
    def __init__(self, n_vars: int, d_model: int, nhead: int, n_layers: int):
        super().__init__()
        self.n_vars = n_vars
        self.input_proj = nn.Linear(n_vars, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_vars)
        self.attn_weights = nn.Parameter(torch.zeros(n_vars, n_vars))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], device=x.device) * float("-inf"), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.head(h[:, -1])


class CausalTransformer(BaseDeepEstimator):
    """Transformer with causal masking; attention weights form causal graph."""

    def __init__(self, lag: int = 10, d_model: int = 32, nhead: int = 4, n_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers

    def fit(self, X) -> CausalTransformer:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(
            _CausalTransformerModule(n_vars, self.d_model, self.nhead, self.n_layers), device
        )
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, xout), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.module_(xb)
                loss = F.mse_loss(pred, yb)
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
        return torch.sigmoid(self.module_.attn_weights).detach().cpu().numpy()


class _TFTModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.var_select = nn.Linear(n_vars, n_vars)
        self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.var_select(x.mean(1)), dim=-1)
        x_sel = x * w.unsqueeze(1)
        h, _ = self.lstm(x_sel)
        attn_out, _ = self.attn(h, h, h)
        return self.head(attn_out[:, -1])


class TFTNet(BaseDeepEstimator):
    """Temporal Fusion Transformer for interpretable time-series forecasting."""

    def __init__(self, lag: int = 10, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X) -> TFTNet:
        device = self._setup()
        x = check_array(X, "X")
        xin, xout, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        n_vars = x.shape[1]
        self.module_ = module_to_device(_TFTModule(n_vars, self.hidden), device)
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
        w = torch.softmax(self.module_.var_select.weight, dim=0).detach().cpu().numpy()
        return np.abs(w)


_ATTN_MODELS = {"tcdf": None, "causal_transformer": CausalTransformer, "tft": TFTNet}


def attn_causal_model(method: str = "causal_transformer", **kwargs) -> BaseDeepEstimator:
    """Factory for attention-based causal models (mirrors R ``attn_causal_model()``)."""
    from .tcdf import TCDF

    models = {**_ATTN_MODELS, "tcdf": TCDF}
    key = method.lower().replace("-", "_")
    if key not in models:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_ATTN_MODELS)}.")
    return models[key](**kwargs)
