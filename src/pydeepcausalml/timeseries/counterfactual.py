"""Counterfactual time-series models: DeepSynth, CRN, GNet."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator, MLP
from ..timeseries.granger import make_lagged_sequences
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["DeepSynth", "CRN", "GNet", "counterfactual_model"]


class _DeepSynthModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.encoder = nn.GRU(n_vars, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.factual = nn.Linear(hidden, 1)
        self.counterfactual = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.encoder(x)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        ctx = (a.unsqueeze(-1) * h).sum(1)
        return self.factual(ctx).squeeze(-1), self.counterfactual(ctx).squeeze(-1)


class DeepSynth(BaseDeepEstimator):
    """Neural synthetic control with donor attention."""

    def __init__(self, lag: int = 10, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X, outcome) -> DeepSynth:
        device = self._setup()
        x = check_array(X, "X")
        y = to_numpy(outcome).astype(np.float64).ravel()
        xin, _, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        if len(xin) != len(y) - self.lag:
            y = y[self.lag:]
        self.module_ = module_to_device(_DeepSynthModule(x.shape[1], self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(TensorDataset(xin, to_tensor(y)), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                factual, _ = self.module_(xb)
                loss = F.mse_loss(factual, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_counterfactual(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        self.module_.eval()
        with torch.no_grad():
            _, cf = self.module_(xin.to(device))
        return to_numpy(cf)


class _CRNModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.encoder = nn.GRU(n_vars + 1, hidden, batch_first=True)
        self.disc = MLP(hidden, 1, hidden=(hidden // 2,), activation=nn.ELU)
        self.y0_head = nn.Linear(hidden, 1)
        self.y1_head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_seq = t.unsqueeze(-1).unsqueeze(1).expand(-1, x.shape[1], -1)
        h, _ = self.encoder(torch.cat([x, t_seq], dim=-1))
        return h[:, -1]


class CRN(BaseDeepEstimator):
    """Counterfactual Recurrent Network with adversarial treatment balancing."""

    def __init__(self, lag: int = 10, hidden: int = 32, lambda_adv: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden
        self.lambda_adv = lambda_adv

    def fit(self, X, treatment, outcome) -> CRN:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        xin, _, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        t_seq = t[self.lag:]
        y_seq = y[self.lag:]
        self.module_ = module_to_device(_CRNModule(x.shape[1], self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(xin, to_tensor(t_seq), to_tensor(y_seq)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                rep = self.module_.encode(xb, tb)
                y0 = self.module_.y0_head(rep).squeeze(-1)
                y1 = self.module_.y1_head(rep).squeeze(-1)
                y_hat = torch.where(tb.bool(), y1, y0)
                outcome_loss = F.mse_loss(y_hat, yb)
                t_logit = self.module_.disc(rep).squeeze(-1)
                adv_loss = F.binary_cross_entropy_with_logits(t_logit, tb)
                loss = outcome_loss + self.lambda_adv * adv_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_ite(self, X, treatment) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        t_seq = t[self.lag:]
        self.module_.eval()
        with torch.no_grad():
            rep = self.module_.encode(xin.to(device), to_tensor(t_seq, device=device))
            y0 = self.module_.y0_head(rep).squeeze(-1)
            y1 = self.module_.y1_head(rep).squeeze(-1)
        return to_numpy(y1 - y0)


class _GNetModule(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.backbone = nn.GRU(n_vars, hidden, batch_first=True)
        self.transition = nn.Linear(hidden + 1, n_vars)
        self.outcome = nn.Linear(hidden + n_vars, 1)

    def forward(self, x: torch.Tensor, do_t: torch.Tensor) -> torch.Tensor:
        h, _ = self.backbone(x)
        h_last = h[:, -1]
        x_next = self.transition(torch.cat([h_last, do_t.unsqueeze(-1)], dim=-1))
        return self.outcome(torch.cat([h_last, x_next], dim=-1)).squeeze(-1)


class GNet(BaseDeepEstimator):
    """Deep G-computation with covariate transition and outcome heads."""

    def __init__(self, lag: int = 10, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.hidden = hidden

    def fit(self, X, treatment, outcome) -> GNet:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        xin, _, self.mean_, self.std_ = make_lagged_sequences(x, self.lag)
        t_seq = t[self.lag:]
        y_seq = y[self.lag:]
        self.module_ = module_to_device(_GNetModule(x.shape[1], self.hidden), device)
        optimizer = self._make_optimizer(self.module_)
        loader = DataLoader(
            TensorDataset(xin, to_tensor(t_seq), to_tensor(y_seq)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            losses = []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(self.module_(xb, tb), yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self._record(loss=np.mean(losses))
            self._log_epoch(epoch, loss=losses[-1])

        self._fitted = True
        return self

    def predict_ite(self, X) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        xin, _, _, _ = make_lagged_sequences(x, self.lag)
        n = xin.shape[0]
        self.module_.eval()
        with torch.no_grad():
            y0 = self.module_(xin.to(device), torch.zeros(n, device=device))
            y1 = self.module_(xin.to(device), torch.ones(n, device=device))
        return to_numpy(y1 - y0)


_CF_MODELS = {"deepsynth": DeepSynth, "crn": CRN, "gnet": GNet}


def counterfactual_model(method: str = "crn", **kwargs) -> BaseDeepEstimator:
    """Factory for counterfactual models (mirrors R ``counterfactual_model()``)."""
    key = method.lower().replace("-", "_")
    if key not in _CF_MODELS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_CF_MODELS)}.")
    return _CF_MODELS[key](**kwargs)
