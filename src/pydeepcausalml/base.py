"""Base classes shared across PyDeepCausalML estimators.

All estimators follow an sklearn-style contract:

* ``__init__`` stores hyperparameters only (no computation).
* ``fit(...)`` trains the model and returns ``self``.
* Prediction methods accept and return NumPy arrays.
* ``history_`` (dict of lists) records per-epoch training diagnostics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from .utils import resolve_device, set_seed

logger = logging.getLogger("pydeepcausalml")

__all__ = ["BaseDeepEstimator", "EarlyStopping", "MLP"]


class EarlyStopping:
    """Stop training when a monitored loss stops improving.

    Parameters
    ----------
    patience : int
        Epochs to wait after the last improvement before stopping.
    min_delta : float
        Minimum decrease in loss to count as an improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MLP(nn.Module):
    """Simple fully connected network used as a building block.

    Parameters
    ----------
    in_dim, out_dim : int
        Input and output dimensionality.
    hidden : tuple of int
        Sizes of hidden layers.
    activation : type
        Activation module class (e.g. ``nn.ReLU``, ``nn.ELU``).
    final_activation : nn.Module, optional
        Module appended after the output layer (e.g. ``nn.Sigmoid()``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple = (64, 64),
        activation: type = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class BaseDeepEstimator:
    """Common infrastructure for torch-backed estimators.

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        Adam weight decay (L2).
    early_stopping_patience : int or None
        If set, stop when training loss has not improved for this many epochs.
    device : str or torch.device or None
        ``None`` auto-selects CUDA when available.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        Log per-epoch losses at INFO level every ``log_every`` epochs.
    """

    def __init__(
        self,
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        early_stopping_patience: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        log_every: int = 50,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.log_every = log_every

        self.history_: Dict[str, List[float]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    def _setup(self) -> torch.device:
        if self.random_state is not None:
            set_seed(self.random_state)
        self._device = resolve_device(self.device)
        self.history_ = {}
        return self._device

    def _make_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _record(self, **losses: float) -> None:
        for key, value in losses.items():
            self.history_.setdefault(key, []).append(float(value))

    def _log_epoch(self, epoch: int, **losses: float) -> None:
        if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs - 1):
            msg = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
            logger.info("[%s] epoch %d | %s", type(self).__name__, epoch, msg)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} instance is not fitted yet; call `fit` first."
            )

    # ------------------------------------------------------------------ #
    def get_params(self) -> Dict[str, object]:
        """Return constructor hyperparameters (sklearn-compatible)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.endswith("_") and not k.startswith("_")
        }

    def set_params(self, **params: object) -> BaseDeepEstimator:
        """Set constructor hyperparameters (sklearn-compatible)."""
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k!r} for {type(self).__name__}.")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        params = ", ".join(f"{k}={v!r}" for k, v in list(self.get_params().items())[:6])
        return f"{type(self).__name__}({params})"


def mmd_rbf(a: torch.Tensor, b: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy between two embedding sets with an RBF kernel.

    Used by CFRNet-style representation balancing. Returns a non-negative scalar.
    """

    def rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(0) - y.unsqueeze(1)
        return torch.exp(-(diff**2).sum(-1) / (2 * bandwidth**2))

    na, nb = a.shape[0], b.shape[0]
    k_aa = rbf(a, a).sum() / (na * na)
    k_bb = rbf(b, b).sum() / (nb * nb)
    k_ab = rbf(a, b).sum() / (na * nb)
    return (k_aa + k_bb - 2 * k_ab).clamp(min=0.0)
