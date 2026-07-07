"""CASTLE: CAusal STructure LEarning regularization.

CASTLE (Kyono, Zhang & van der Schaar, 2020) trains a supervised predictor
jointly with a learned adjacency matrix, a NOTEARS acyclicity penalty, an L1
edge-sparsity penalty, and a selective reconstruction loss. One fitted model
yields both a target predictor and a causal graph over all variables.

The implementation follows the tutorial version in the accompanying
``causal_structure_learning_regularization_CASTLE`` notebook, including the
CPU/float64 evaluation of ``matrix_exp`` for numerical stability.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDeepEstimator, EarlyStopping
from ..utils import check_array, to_numpy, to_tensor
from .notears import notears_acyclicity

__all__ = ["CASTLE"]


class _CASTLEModule(nn.Module):
    def __init__(
        self,
        d: int,
        hidden_dim: int,
        num_layers: int,
        neighbor_temp: float,
        y_index: int,
    ):
        super().__init__()
        self.d = d
        self.neighbor_temp = neighbor_temp
        self.y_index = y_index

        self.adjacency = nn.Parameter(torch.randn(d, d) * 0.03)
        self.register_buffer("loop_mask", 1.0 - torch.eye(d))

        layers = []
        inp = d
        for _ in range(num_layers - 1):
            layers += [nn.Linear(inp, hidden_dim), nn.ReLU()]
            inp = hidden_dim
        layers.append(nn.Linear(inp, 1))
        self.predictor = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d)
        )

    def masked_adjacency(self) -> torch.Tensor:
        return self.adjacency * self.loop_mask

    def forward(self, z: torch.Tensor):
        a = self.masked_adjacency()
        z_in = z.clone()
        z_in[:, self.y_index] = 0.0  # target never predicts itself

        importance = torch.sigmoid(a.abs().sum(dim=0))
        h = z_in * importance
        pred = self.predictor(h).squeeze(-1)
        z_hat = self.decoder(h)

        neighbor_score = a.abs().sum(dim=0) + a.abs().sum(dim=1)
        soft_mask = torch.sigmoid(self.neighbor_temp * neighbor_score)
        return pred, a, z_hat, soft_mask


class CASTLE(BaseDeepEstimator):
    """Predictor with causal-graph regularization (CASTLE).

    Fit on the full variable matrix ``Z`` (covariates + treatment + outcome,
    or any variable set); ``y_index`` marks which column is the supervised
    target. After fitting, :meth:`get_adjacency` exposes the learned DAG and
    :meth:`predict` the regularized target predictions.

    Parameters
    ----------
    y_index : int
        Column index of the prediction target inside ``Z`` (negative allowed).
    hidden_dim, num_layers : int
        Predictor width and depth.
    lambda_reg : float
        Weight of the supervised MSE term.
    beta_sparsity : float
        L1 penalty weight on the adjacency.
    acyc_weight : float
        Weight of the acyclicity penalty ``h(A)``.
    recon_weight : float
        Weight of the neighbor-masked reconstruction loss.
    neighbor_temp : float
        Temperature of the soft neighbor mask used by the reconstruction term.
    """

    def __init__(
        self,
        y_index: int = -1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        lambda_reg: float = 1.0,
        beta_sparsity: float = 0.015,
        acyc_weight: float = 0.1,
        recon_weight: float = 0.5,
        neighbor_temp: float = 10.0,
        standardize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.y_index = y_index
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lambda_reg = lambda_reg
        self.beta_sparsity = beta_sparsity
        self.acyc_weight = acyc_weight
        self.recon_weight = recon_weight
        self.neighbor_temp = neighbor_temp
        self.standardize = standardize

    def fit(self, Z) -> CASTLE:
        """Fit CASTLE on an (n, d) matrix containing the target column."""
        device = self._setup()
        z = check_array(Z, "Z")
        d = z.shape[1]
        y_index = self.y_index if self.y_index >= 0 else d + self.y_index
        if not 0 <= y_index < d:
            raise ValueError(f"y_index {self.y_index} is out of range for d={d}.")

        if self.standardize:
            self.z_mean_ = z.mean(axis=0)
            self.z_std_ = z.std(axis=0) + 1e-8
            z = (z - self.z_mean_) / self.z_std_
        else:
            self.z_mean_, self.z_std_ = 0.0, 1.0

        self._y_index_resolved = y_index
        self.module_ = _CASTLEModule(
            d, self.hidden_dim, self.num_layers, self.neighbor_temp, y_index
        ).to(device)
        optimizer = self._make_optimizer(self.module_)
        stopper = (
            EarlyStopping(self.early_stopping_patience)
            if self.early_stopping_patience
            else None
        )

        z_t = to_tensor(z)
        y_t = z_t[:, y_index].clone()
        loader = DataLoader(TensorDataset(z_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.module_.train()
        for epoch in range(self.epochs):
            totals = {"loss": 0.0, "mse": 0.0, "acyc": 0.0, "recon": 0.0}
            for zb, yb in loader:
                zb, yb = zb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred, a, z_hat, soft_mask = self.module_(zb)

                mse = nn.functional.mse_loss(pred, yb)
                sparsity = self.beta_sparsity * a.abs().sum()
                acyc = self.acyc_weight * notears_acyclicity(a)
                recon = self.recon_weight * (soft_mask * (z_hat - zb) ** 2).mean()

                loss = self.lambda_reg * mse + sparsity + acyc + recon
                loss.backward()
                optimizer.step()

                totals["loss"] += loss.item()
                totals["mse"] += mse.item()
                totals["acyc"] += float(acyc)
                totals["recon"] += float(recon)

            averaged = {k: v / len(loader) for k, v in totals.items()}
            self._record(**averaged)
            self._log_epoch(epoch, **averaged)
            if stopper is not None and stopper.step(averaged["loss"]):
                break

        with torch.no_grad():
            self.weights_ = to_numpy(self.module_.masked_adjacency())
        self._fitted = True
        return self

    def predict(self, Z) -> np.ndarray:
        """Predict the target column from an (n, d) matrix (target column is masked)."""
        self._check_fitted()
        z = check_array(Z, "Z")
        z = (z - self.z_mean_) / self.z_std_ if self.standardize else z
        self.module_.eval()
        with torch.no_grad():
            pred, *_ = self.module_(to_tensor(z, device=self._device))
        pred = to_numpy(pred)
        if self.standardize:
            pred = pred * self.z_std_[self._y_index_resolved] + self.z_mean_[self._y_index_resolved]
        return pred

    def get_adjacency(self, threshold: float = 0.05) -> np.ndarray:
        """Binary adjacency over the modeled variables (``A[i, j]``: :math:`Z_j \\to Z_i`)."""
        self._check_fitted()
        a = (np.abs(self.weights_) > threshold).astype(int)
        np.fill_diagonal(a, 0)
        return a

    def adjacency_matrix(self, threshold: float = 0.05) -> np.ndarray:
        """Alias for :meth:`get_adjacency`."""
        return self.get_adjacency(threshold)

    def get_scores(self) -> np.ndarray:
        """Signed learned adjacency weights."""
        self._check_fitted()
        return self.weights_.copy()
