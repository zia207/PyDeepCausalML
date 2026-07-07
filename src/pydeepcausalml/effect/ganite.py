"""GANITE: GAN-based individual treatment effect estimation (Yoon et al., 2018)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["GANITE"]


def _xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class _GANITEGenerator(nn.Module):
    def __init__(self, input_dim: int, h_dim: int, dropout: bool = False):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim + 2, h_dim)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)
        self.dp = nn.Dropout(0.2)
        _xavier_init(self)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, t, y], dim=1)
        h1 = F.relu(self.fc1(inputs))
        if self.dropout:
            h1 = self.dp(h1)
        h2 = F.relu(self.fc2(F.relu(self.fc2_2(F.relu(self.fc2_1(h1))))))
        if self.dropout:
            h2 = self.dp(h2)
        y0 = self.fc32(F.relu(self.fc31(h2)))
        y1 = self.fc42(F.relu(self.fc41(h2)))
        return torch.cat([y0, y1], dim=1)


class _GANITEDiscriminator(nn.Module):
    def __init__(self, input_dim: int, h_dim: int, dropout: bool = False):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim + 2, h_dim)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.dp = nn.Dropout(0.2)
        _xavier_init(self)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        input0 = (1.0 - t) * y + t * y_hat[:, 0:1]
        input1 = t * y + (1.0 - t) * y_hat[:, 1:2]
        inputs = torch.cat([x, input0, input1], dim=1)
        h1 = F.relu(self.fc1(inputs))
        if self.dropout:
            h1 = self.dp(h1)
        h2 = F.relu(self.fc2(F.relu(self.fc2_2(F.relu(self.fc2_1(h1))))))
        if self.dropout:
            h2 = self.dp(h2)
        return self.fc3(h2)


class _GANITEInference(nn.Module):
    def __init__(self, input_dim: int, h_dim: int, dropout: bool = False):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)
        self.dp = nn.Dropout(0.2)
        _xavier_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        if self.dropout:
            h1 = self.dp(h1)
        h2 = F.relu(self.fc2(F.relu(self.fc2_2(F.relu(self.fc2_1(h1))))))
        if self.dropout:
            h2 = self.dp(h2)
        y0 = self.fc32(F.relu(self.fc31(h2)))
        y1 = self.fc42(F.relu(self.fc41(h2)))
        return torch.cat([y0, y1], dim=1)


class GANITE(BaseDeepEstimator):
    """Generative Adversarial Network for Individual Treatment Effect estimation.

    Parameters
    ----------
    h_dim : int
        Hidden width for generator, discriminator, and inference networks.
    alpha, beta : float
        Weights for the adversarial generator loss and supervised inference loss.
    dropout : bool
        Whether to apply dropout in the networks.
    """

    def __init__(
        self,
        h_dim: int = 50,
        alpha: float = 1.0,
        beta: float = 1.0,
        dropout: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 200)
        kwargs.setdefault("lr", 1e-4)
        super().__init__(**kwargs)
        self.h_dim = h_dim
        self.alpha = alpha
        self.beta = beta
        self.dropout = dropout

    def fit(self, X, treatment, outcome) -> GANITE:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment).reshape(-1, 1)
        y = to_numpy(outcome).astype(np.float64).ravel().reshape(-1, 1)
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        p = x.shape[1]
        self.generator_ = module_to_device(_GANITEGenerator(p, self.h_dim, self.dropout), device)
        self.discriminator_ = module_to_device(
            _GANITEDiscriminator(p, self.h_dim, self.dropout), device
        )
        self.inference_ = module_to_device(_GANITEInference(p, self.h_dim, self.dropout), device)

        opt_g = torch.optim.Adam(self.generator_.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator_.parameters(), lr=self.lr)
        opt_i = torch.optim.Adam(self.inference_.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            d_losses, g_losses, i_losses = [], [], []
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)

                # Discriminator
                self.generator_.eval()
                self.discriminator_.train()
                self.inference_.eval()
                for _ in range(2):
                    opt_d.zero_grad()
                    y_tilde = self.generator_(xb, tb, yb)
                    d_logit = self.discriminator_(xb, tb, yb, y_tilde)
                    d_loss = F.binary_cross_entropy_with_logits(d_logit, tb)
                    d_loss.backward(retain_graph=True)
                    opt_d.step()
                    d_losses.append(d_loss.item())

                # Generator
                self.generator_.train()
                self.discriminator_.eval()
                self.inference_.eval()
                opt_g.zero_grad()
                y_tilde = self.generator_(xb, tb, yb)
                d_logit = self.discriminator_(xb, tb, yb, y_tilde)
                g_loss_gan = -F.binary_cross_entropy_with_logits(d_logit, tb)
                y_est = tb * y_tilde[:, 1:2] + (1 - tb) * y_tilde[:, 0:1]
                g_loss = F.mse_loss(y_est, yb) + self.alpha * g_loss_gan
                g_loss.backward(retain_graph=True)
                opt_g.step()
                g_losses.append(g_loss.item())

                # Inference
                self.generator_.eval()
                self.discriminator_.eval()
                self.inference_.train()
                opt_i.zero_grad()
                y_hat = self.inference_(xb)
                y_tilde = self.generator_(xb, tb, yb)
                y_t0 = tb * yb + (1 - tb) * y_tilde[:, 1:2]
                y_t1 = (1 - tb) * yb + tb * y_tilde[:, 0:1]
                i_loss = (
                    F.mse_loss(y_hat[:, 1:2], y_t0)
                    + F.mse_loss(y_hat[:, 0:1], y_t1)
                    + self.beta
                    * F.mse_loss(
                        y_hat[:, 1].mean().view(1),
                        (tb * yb - (1 - tb) * yb).mean().detach().view(1),
                    )
                )
                i_loss.backward()
                opt_i.step()
                i_losses.append(i_loss.item())

            self._record(
                d_loss=np.mean(d_losses),
                g_loss=np.mean(g_losses),
                i_loss=np.mean(i_losses),
            )
            self._log_epoch(epoch, d_loss=d_losses[-1], g_loss=g_losses[-1], i_loss=i_losses[-1])

        self._fitted = True
        return self

    def predict_potential_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        self.inference_.eval()
        with torch.no_grad():
            y_hat = self.inference_(to_tensor(x, device=device))
        return to_numpy(y_hat[:, 0]), to_numpy(y_hat[:, 1])

    def predict_cate(self, X) -> np.ndarray:
        y0, y1 = self.predict_potential_outcomes(X)
        return y1 - y0

    def predict_ate(self, X) -> float:
        return float(self.predict_cate(X).mean())
