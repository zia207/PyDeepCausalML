"""CEVAE: Causal Effect Variational Autoencoder (Louizos et al., 2017)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator
from ..utils import check_array, check_binary_treatment, module_to_device, to_numpy, to_tensor

__all__ = ["CEVAE"]


def _fc_stack(sizes: list[int], final_act: type | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    if final_act is not None:
        layers.append(final_act())
    return nn.Sequential(*layers)


class _DiagNormalNet(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.dim_out = sizes[-1]
        self.fc = _fc_stack(sizes[:-1] + [sizes[-1] * 2])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(x)
        loc = out[:, : self.dim_out].clamp(-1e2, 1e2)
        scale = F.softplus(out[:, self.dim_out :]).add(1e-3).clamp(max=1e2)
        return loc, scale


class _BernoulliNet(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.fc = _fc_stack(sizes + [1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1).clamp(-10, 10)


class _NormalOutcomeNet(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.fc = _fc_stack(sizes + [2])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(x)
        loc = out[:, 0].clamp(-1e6, 1e6)
        scale = F.softplus(out[:, 1]).clamp(min=1e-3, max=1e6)
        return loc, scale


class _CEVAEModel(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.latent_dim = latent_dim
        h_sizes = [latent_dim] + [hidden_dim] * num_layers
        self.x_nn = _DiagNormalNet(h_sizes + [feature_dim])
        self.t_nn = _BernoulliNet([latent_dim])
        self.y0_nn = _NormalOutcomeNet(h_sizes)
        self.y1_nn = _NormalOutcomeNet(h_sizes)

    def y_mean(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p0_loc, p0_scale = self.y0_nn(z)
        p1_loc, p1_scale = self.y1_nn(z)
        t_bool = t.bool()
        return torch.where(t_bool, p1_loc, p0_loc)


class _CEVAEGuide(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.t_nn = _BernoulliNet([feature_dim])
        y_sizes = [feature_dim] + [hidden_dim] * max(1, num_layers - 1)
        self.y_shared = _fc_stack(y_sizes + [hidden_dim], final_act=nn.ELU)
        self.y0_nn = _NormalOutcomeNet([hidden_dim])
        self.y1_nn = _NormalOutcomeNet([hidden_dim])
        z_in = 1 + feature_dim
        z_sizes = [z_in] + [hidden_dim] * max(1, num_layers - 1)
        self.z_shared = _fc_stack(z_sizes + [hidden_dim], final_act=nn.ELU)
        self.z0_nn = _DiagNormalNet([hidden_dim, latent_dim])
        self.z1_nn = _DiagNormalNet([hidden_dim, latent_dim])

    def z_sample(self, y: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_x = torch.cat([y.view(-1, 1), x], dim=1)
        hidden = self.z_shared(y_x)
        p0_loc, p0_scale = self.z0_nn(hidden)
        p1_loc, p1_scale = self.z1_nn(hidden)
        t_mask = t.bool().unsqueeze(-1)
        loc = torch.where(t_mask, p1_loc, p0_loc)
        scale = torch.where(t_mask, p1_scale, p0_scale)
        eps = torch.randn_like(loc)
        return loc + eps * scale


def _whiten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loc = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return loc, 1.0 / scale


def _whiten_apply(x: torch.Tensor, loc: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
    return (x - loc) * inv_scale


class CEVAE(BaseDeepEstimator):
    """Causal Effect Variational Autoencoder for ITE estimation.

    Parameters
    ----------
    latent_dim, hidden_dim : int
        Latent and hidden layer sizes.
    num_layers : int
        Number of hidden layers in the generative and inference networks.
    num_samples : int
        Monte Carlo samples for ITE prediction.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dim: int = 200,
        num_layers: int = 3,
        num_samples: int = 100,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 50)
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

    def fit(self, X, treatment, outcome) -> CEVAE:
        device = self._setup()
        x = check_array(X, "X")
        t = check_binary_treatment(treatment)
        y = to_numpy(outcome).astype(np.float64).ravel()
        if not (len(x) == len(t) == len(y)):
            raise ValueError("X, treatment, and outcome must have the same length.")

        p = x.shape[1]
        self.model_ = module_to_device(
            _CEVAEModel(p, self.latent_dim, self.hidden_dim, self.num_layers), device
        )
        self.guide_ = module_to_device(
            _CEVAEGuide(p, self.latent_dim, self.hidden_dim, self.num_layers), device
        )
        x_t = to_tensor(x, device=device)
        self.whiten_loc_, self.whiten_inv_ = _whiten(x_t)

        params = list(self.model_.parameters()) + list(self.guide_.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        loader = DataLoader(
            TensorDataset(to_tensor(x), to_tensor(t), to_tensor(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                x_w = _whiten_apply(xb, self.whiten_loc_, self.whiten_inv_)

                optimizer.zero_grad()
                t_logits = self.guide_.t_nn(x_w)
                t_probs = torch.sigmoid(t_logits).clamp(1e-6, 1 - 1e-6)
                t_samp = torch.bernoulli(t_probs)

                y_hidden = self.guide_.y_shared(x_w)
                y0_loc, _ = self.guide_.y0_nn(y_hidden)
                y1_loc, _ = self.guide_.y1_nn(y_hidden)
                y_samp = torch.where(t_samp.bool(), y1_loc, y0_loc)

                z_samp = self.guide_.z_sample(y_samp, t_samp, x_w)

                x_loc, x_scale = self.model_.x_nn(z_samp)
                x_nll = 0.5 * (((x_w - x_loc) / x_scale) ** 2 + 2 * x_scale.log()).sum(-1).mean()

                t_logits_gen = self.model_.t_nn(z_samp)
                t_nll = F.binary_cross_entropy_with_logits(t_logits_gen, t_samp)

                y_loc = self.model_.y_mean(z_samp, t_samp)
                y_nll = 0.5 * (yb - y_loc).pow(2).mean()

                z_prior_loc = torch.zeros_like(z_samp)
                z_prior_scale = torch.ones_like(z_samp)
                kl_z = 0.5 * (
                    ((z_samp - z_prior_loc) / z_prior_scale).pow(2)
                    - 1
                    + 2 * (z_prior_scale.log() - torch.zeros_like(z_samp))
                ).sum(-1).mean()

                loss = x_nll + t_nll + y_nll + kl_z
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            self._record(loss=epoch_loss / max(n_batches, 1))
            self._log_epoch(epoch, loss=epoch_loss / max(n_batches, 1))

        self._fitted = True
        return self

    def predict_cate(self, X, num_samples: int | None = None) -> np.ndarray:
        self._check_fitted()
        device = self._device
        x = check_array(X, "X")
        n_samp = num_samples or self.num_samples
        x_w = _whiten_apply(to_tensor(x, device=device), self.whiten_loc_, self.whiten_inv_)

        self.model_.eval()
        self.guide_.eval()
        ite_acc = np.zeros(len(x))
        with torch.no_grad():
            for _ in range(n_samp):
                t_logits = self.guide_.t_nn(x_w)
                t_samp = torch.bernoulli(torch.sigmoid(t_logits))
                y_hidden = self.guide_.y_shared(x_w)
                y0_loc, _ = self.guide_.y0_nn(y_hidden)
                y1_loc, _ = self.guide_.y1_nn(y_hidden)
                y_samp = torch.where(t_samp.bool(), y1_loc, y0_loc)
                z_samp = self.guide_.z_sample(y_samp, t_samp, x_w)
                y0 = self.model_.y_mean(z_samp, torch.zeros_like(t_samp))
                y1 = self.model_.y_mean(z_samp, torch.ones_like(t_samp))
                ite_acc += to_numpy(y1 - y0)
        return ite_acc / n_samp

    def predict_ate(self, X) -> float:
        return float(self.predict_cate(X).mean())
