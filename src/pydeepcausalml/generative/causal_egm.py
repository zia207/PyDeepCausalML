"""CausalEGM: causal encoding generative modeling for ITE/ATE estimation.

An encoder splits covariates into three latent blocks — confounding (``z_c``),
treatment-only (``z_t``), and outcome-only (``z_y``) — with a decoder
reconstructing X, a propensity head on ``(z_c, z_t)``, an outcome head on
``(z_c, z_y, t)``, and an adversarial discriminator encouraging independence
between ``z_t`` and ``z_y`` (Liu et al., 2022; tutorial adaptation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MLP, BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, to_numpy, to_tensor

__all__ = ["CausalEGM"]


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, dim_c: int, dim_t: int, dim_y: int, hidden: int):
        super().__init__()
        self.shared = MLP(input_dim, hidden, hidden=(hidden,), activation=nn.ELU)
        self.act = nn.ELU()
        self.head_c = nn.Linear(hidden, dim_c)
        self.head_t = nn.Linear(hidden, dim_t)
        self.head_y = nn.Linear(hidden, dim_y)

    def forward(self, x: torch.Tensor):
        h = self.act(self.shared(x))
        return self.head_c(h), self.head_t(h), self.head_y(h)


class _EGMModule(nn.Module):
    def __init__(self, input_dim: int, dim_c: int, dim_t: int, dim_y: int, hidden: int):
        super().__init__()
        self.encoder = _Encoder(input_dim, dim_c, dim_t, dim_y, hidden)
        self.decoder = MLP(dim_c + dim_t + dim_y, input_dim, hidden=(hidden, hidden), activation=nn.ELU)
        self.treatment_head = MLP(dim_c + dim_t, 1, hidden=(hidden // 2,), activation=nn.ELU)
        self.outcome_head = MLP(dim_c + dim_y + 1, 1, hidden=(hidden // 2, hidden // 2), activation=nn.ELU)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        zc, zt, zy = self.encoder(x)
        x_hat = self.decoder(torch.cat([zc, zt, zy], dim=-1))
        t_logit = self.treatment_head(torch.cat([zc, zt], dim=-1)).squeeze(-1)
        y_hat = self.outcome_head(torch.cat([zc, zy, t.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return x_hat, t_logit, y_hat, (zc, zt, zy)


class CausalEGM(BaseDeepEstimator):
    """Causal Encoding Generative Model for individual treatment effects.

    Parameters
    ----------
    dim_c, dim_t, dim_y : int
        Sizes of the confounder, treatment-specific, and outcome-specific
        latent blocks.
    hidden : int
        Hidden width of the encoder/decoder networks.
    lambda_recon, lambda_treat, lambda_outcome, lambda_disent : float
        Loss weights for reconstruction, propensity, outcome, and the
        adversarial disentanglement term.
    """

    def __init__(
        self,
        dim_c: int = 8,
        dim_t: int = 4,
        dim_y: int = 4,
        hidden: int = 128,
        lambda_recon: float = 1.0,
        lambda_treat: float = 2.0,
        lambda_outcome: float = 2.0,
        lambda_disent: float = 0.5,
        lr_disc: float = 5e-4,
        standardize_outcome: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 100)
        super().__init__(**kwargs)
        self.dim_c = dim_c
        self.dim_t = dim_t
        self.dim_y = dim_y
        self.hidden = hidden
        self.lambda_recon = lambda_recon
        self.lambda_treat = lambda_treat
        self.lambda_outcome = lambda_outcome
        self.lambda_disent = lambda_disent
        self.lr_disc = lr_disc
        self.standardize_outcome = standardize_outcome

    def fit(self, X, treatment, outcome) -> CausalEGM:
        """Fit on covariates ``X``, binary ``treatment``, and continuous ``outcome``."""
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()

        self.x_mean_, self.x_std_ = x.mean(axis=0), x.std(axis=0) + 1e-8
        x = (x - self.x_mean_) / self.x_std_
        if self.standardize_outcome:
            self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-8
            y = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_, self.y_std_ = 0.0, 1.0

        self.module_ = _EGMModule(x.shape[1], self.dim_c, self.dim_t, self.dim_y, self.hidden).to(device)
        self.disc_ = MLP(self.dim_t + self.dim_y, 1, hidden=(self.hidden // 2,), activation=nn.ELU).to(device)

        opt_model = self._make_optimizer(self.module_)
        opt_disc = torch.optim.Adam(self.disc_.parameters(), lr=self.lr_disc)
        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            self.module_.train()
            self.disc_.train()
            totals = {"loss": 0.0, "recon": 0.0, "treat": 0.0, "outcome": 0.0, "disent": 0.0}

            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)

                # --- discriminator step: real (zt, zy) vs shuffled zy ---
                with torch.no_grad():
                    _, zt_d, zy_d = self.module_.encoder(xb)
                perm = torch.randperm(len(zy_d), device=device)
                real = self.disc_(torch.cat([zt_d, zy_d], dim=-1)).squeeze(-1)
                fake = self.disc_(torch.cat([zt_d, zy_d[perm]], dim=-1)).squeeze(-1)
                loss_disc = 0.5 * (
                    F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                    + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
                )
                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                # --- generator/estimator step ---
                x_hat, t_logit, y_hat, (zc, zt, zy) = self.module_(xb, tb)
                l_recon = F.mse_loss(x_hat, xb)
                l_treat = F.binary_cross_entropy_with_logits(t_logit, tb)
                l_outcome = F.mse_loss(y_hat, yb)
                # Fool the discriminator: make joint (zt, zy) look independent.
                joint = self.disc_(torch.cat([zt, zy], dim=-1)).squeeze(-1)
                l_disent = F.binary_cross_entropy_with_logits(joint, torch.zeros_like(joint))

                loss = (
                    self.lambda_recon * l_recon
                    + self.lambda_treat * l_treat
                    + self.lambda_outcome * l_outcome
                    + self.lambda_disent * l_disent
                )
                opt_model.zero_grad()
                loss.backward()
                opt_model.step()

                totals["loss"] += loss.item()
                totals["recon"] += l_recon.item()
                totals["treat"] += l_treat.item()
                totals["outcome"] += l_outcome.item()
                totals["disent"] += l_disent.item()

            averaged = {k: v / len(loader) for k, v in totals.items()}
            self._record(**averaged)
            self._log_epoch(epoch, **averaged)

        self._fitted = True
        return self

    def _encode(self, X) -> tuple:
        x = (check_array(X, "X") - self.x_mean_) / self.x_std_
        return self.module_.encoder(to_tensor(x, device=self._device))

    def predict_ite(self, X) -> np.ndarray:
        """Individual treatment effects :math:`\\hat y_1(x) - \\hat y_0(x)`."""
        self._check_fitted()
        self.module_.eval()
        with torch.no_grad():
            zc, _, zy = self._encode(X)
            ones = torch.ones(len(zc), 1, device=self._device)
            zeros = torch.zeros(len(zc), 1, device=self._device)
            y1 = self.module_.outcome_head(torch.cat([zc, zy, ones], dim=-1)).squeeze(-1)
            y0 = self.module_.outcome_head(torch.cat([zc, zy, zeros], dim=-1)).squeeze(-1)
        return to_numpy(y1 - y0) * self.y_std_

    def predict_cate(self, X) -> np.ndarray:
        """Alias of :meth:`predict_ite` for API consistency with other estimators."""
        return self.predict_ite(X)

    def predict_ate(self, X) -> float:
        """Average treatment effect over ``X``."""
        return float(self.predict_ite(X).mean())

    def predict_propensity(self, X) -> np.ndarray:
        """Propensity scores from the treatment head."""
        self._check_fitted()
        self.module_.eval()
        with torch.no_grad():
            zc, zt, _ = self._encode(X)
            logits = self.module_.treatment_head(torch.cat([zc, zt], dim=-1)).squeeze(-1)
        return to_numpy(torch.sigmoid(logits))
