"""TCDF: Temporal Causal Discovery Framework.

Attention-based causal discovery with dilated depthwise temporal convolutions
(Nauta, Bucur & Seifert, 2019). This module adapts the reference
``model.py`` / ``runTCDF.py`` scripts into a self-contained, scriptable
estimator:

1. For every target series, an attention-augmented depthwise TCN (ADDSTCN)
   is trained to predict the target one step ahead from all series.
2. Potential causes are read off the learned attention vector using the
   largest-gap heuristic from the paper.
3. Each candidate is validated by Permutation Importance (PIVM): its values
   are shuffled in time and the candidate is kept only when the resulting
   loss increase is a significant fraction of the training loss improvement.
4. The causal delay of each validated edge is estimated from the position of
   the dominant kernel weight along the receptive field.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseDeepEstimator
from ..utils import check_array, to_numpy, to_tensor

__all__ = ["TCDF", "ADDSTCN", "DepthwiseNet"]


class _Chomp1d(nn.Module):
    """Remove trailing padding so convolutions stay strictly causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,  # depthwise: one filter per input series
        )
        self.chomp = _Chomp1d(padding)
        self.relu = nn.PReLU(channels)
        self.conv.weight.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.chomp(self.conv(x)))


class DepthwiseNet(nn.Module):
    """Stack of dilated depthwise causal convolutions (one channel per series).

    Dilation grows as ``dilation_c ** level`` so the receptive field covers
    ``1 + (kernel_size - 1) * sum(dilation_c**l)`` time steps.
    """

    def __init__(self, input_size: int, num_levels: int, kernel_size: int, dilation_c: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                _DepthwiseBlock(input_size, kernel_size, dilation_c**level)
                for level in range(num_levels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ADDSTCN(nn.Module):
    """Attention-based Dilated Depthwise Separable Temporal Convolutional Network.

    Mirrors the reference ``model.py``: a learned per-series attention vector
    gates the input, a depthwise causal TCN models temporal structure, and a
    pointwise convolution mixes channels into the target prediction.
    """

    def __init__(self, input_size: int, num_levels: int, kernel_size: int, dilation_c: int):
        super().__init__()
        self.dwn = DepthwiseNet(input_size, num_levels, kernel_size, dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)
        self.fs_attention = nn.Parameter(torch.ones(input_size, 1))
        self.pointwise.weight.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch=1, series, time) -> (batch, time, 1)
        gated = x * F.softmax(self.fs_attention, dim=0)
        return self.pointwise(self.dwn(gated)).transpose(1, 2)

    def attention_scores(self) -> torch.Tensor:
        return self.fs_attention.detach().squeeze(-1)


class TCDF(BaseDeepEstimator):
    """Temporal causal discovery with attention-based convolutional networks.

    Parameters
    ----------
    kernel_size : int
        Convolution kernel size; the maximum discoverable delay is
        ``receptive_field - 1``. Recommended equal to ``dilation_c``.
    hidden_layers : int
        Extra depthwise levels beyond the first (``num_levels = hidden_layers + 1``).
    dilation_c : int
        Dilation growth coefficient; keep equal to ``kernel_size`` so each
        delay maps to exactly one convolutional path.
    significance : float
        PIVM threshold in (0, 1]: a candidate is validated when permuting it
        removes at least this fraction of the training loss improvement.

    Attributes
    ----------
    causes_ : dict[int, list[int]]
        Validated cause indices per target index.
    delays_ : dict[tuple[int, int], int]
        Estimated delay for each (effect, cause) pair.
    scores_ : ndarray of shape (p, p)
        Attention score matrix (rows = targets, columns = sources).
    columns_ : list[str]
        Series names.

    Examples
    --------
    >>> import numpy as np
    >>> from pydeepcausalml.timeseries import TCDF
    >>> rng = np.random.default_rng(0)
    >>> x0 = rng.standard_normal(500)
    >>> x1 = np.roll(x0, 1) + 0.1 * rng.standard_normal(500)
    >>> model = TCDF(epochs=500, random_state=0).fit(np.column_stack([x0, x1]))
    >>> model.discovered_edges()  # doctest: +SKIP
    [('X0', 'X1', 1)]
    """

    def __init__(
        self,
        kernel_size: int = 4,
        hidden_layers: int = 0,
        dilation_c: Optional[int] = None,
        significance: float = 0.8,
        **kwargs,
    ):
        kwargs.setdefault("epochs", 1000)
        kwargs.setdefault("lr", 1e-2)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.hidden_layers = hidden_layers
        self.dilation_c = dilation_c if dilation_c is not None else kernel_size
        self.significance = significance

    # ------------------------------------------------------------------ #
    @property
    def receptive_field(self) -> int:
        """Number of past steps visible to the network (max delay + 1)."""
        field = 1
        for level in range(self.hidden_layers + 1):
            field += (self.kernel_size - 1) * self.dilation_c**level
        return field

    def _prepare_target(
        self, data: np.ndarray, target_idx: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input: all series shifted so the target is predicted one step ahead."""
        x = data.copy().T  # (series, time)
        y = np.roll(x[target_idx], -1)
        y[-1] = y[-2]  # pad final step, as in the reference implementation
        x_t = to_tensor(x, device=device).unsqueeze(0).contiguous()
        y_t = to_tensor(y, device=device).unsqueeze(0).unsqueeze(-1).contiguous()
        return x_t, y_t

    def _train_target(
        self, x: torch.Tensor, y: torch.Tensor, device: torch.device
    ) -> Tuple[ADDSTCN, float, float]:
        model = ADDSTCN(
            x.shape[1], self.hidden_layers + 1, self.kernel_size, self.dilation_c
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        first_loss = final_loss = 0.0
        model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            if epoch == 1:
                first_loss = loss.item()
            final_loss = loss.item()
        return model, first_loss, final_loss

    @staticmethod
    def _candidate_causes(attention: np.ndarray) -> List[int]:
        """Largest-gap selection on the sorted attention scores (paper heuristic)."""
        order = np.argsort(-attention)
        sorted_scores = attention[order]
        if len(sorted_scores) < 2:
            return list(order)
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        # Only consider gaps above the mean attention score, per the reference code.
        valid = sorted_scores[:-1] >= attention.mean()
        if not valid.any():
            return []
        cut = int(np.argmax(np.where(valid, gaps, -np.inf))) + 1
        return order[:cut].tolist()

    def _validate_pivm(
        self,
        model: ADDSTCN,
        x: torch.Tensor,
        y: torch.Tensor,
        candidate: int,
        first_loss: float,
        final_loss: float,
        rng: np.random.Generator,
    ) -> bool:
        """Permutation Importance: shuffle the candidate series in time and
        keep the edge only if the loss increase is significant."""
        x_perm = x.clone()
        idx = rng.permutation(x.shape[2])
        x_perm[0, candidate] = x[0, candidate][idx]
        model.eval()
        with torch.no_grad():
            perm_loss = F.mse_loss(model(x_perm), y).item()
        improvement = first_loss - final_loss
        if improvement <= 0:
            return False
        recovered = perm_loss - final_loss
        return (recovered / improvement) >= self.significance

    def _estimate_delay(self, model: ADDSTCN, cause: int, target: int) -> int:
        """Delay from the dominant kernel-weight position along the receptive field.

        The network predicts the target one step ahead, so an input tap that
        is ``k`` steps back from the current time corresponds to a causal
        delay of ``k + 1`` in the original series.
        """
        offset = 0
        for level, block in enumerate(model.dwn.blocks):
            w = block.conv.weight.detach()[cause, 0]
            tap = int(torch.argmax(torch.abs(w)))
            offset += (len(w) - 1 - tap) * self.dilation_c**level
        return offset + 1

    # ------------------------------------------------------------------ #
    def fit(self, X, columns: Optional[List[str]] = None) -> TCDF:
        """Discover temporal causal structure in a (T, p) multivariate series.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (T, p)
            One column per time series. DataFrame column names are used as
            series labels.
        columns : list of str, optional
            Series names, overriding any DataFrame columns.
        """
        device = self._setup()
        if isinstance(X, pd.DataFrame) and columns is None:
            columns = [str(c) for c in X.columns]
        data = check_array(X, "X")
        n_series = data.shape[1]
        self.columns_ = columns or [f"X{i}" for i in range(n_series)]
        if len(self.columns_) != n_series:
            raise ValueError("`columns` length does not match the number of series.")

        rng = np.random.default_rng(self.random_state)
        self.causes_: Dict[int, List[int]] = {}
        self.delays_: Dict[Tuple[int, int], int] = {}
        self.scores_ = np.zeros((n_series, n_series))
        self.losses_: Dict[int, float] = {}

        for target in range(n_series):
            x, y = self._prepare_target(data, target, device)
            model, first_loss, final_loss = self._train_target(x, y, device)
            attention = to_numpy(model.attention_scores())
            self.scores_[target] = attention
            self.losses_[target] = final_loss

            validated: List[int] = []
            for candidate in self._candidate_causes(attention):
                if self._validate_pivm(model, x, y, candidate, first_loss, final_loss, rng):
                    validated.append(candidate)
                    self.delays_[(target, candidate)] = self._estimate_delay(
                        model, candidate, target
                    )
            self.causes_[target] = validated
            self._record(target=float(target), loss=final_loss)
            self._log_epoch(target, loss=final_loss)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def get_adjacency(self, include_self_loops: bool = False) -> np.ndarray:
        """Binary adjacency of validated edges (``A[i, j]``: :math:`X_j \\to X_i`)."""
        self._check_fitted()
        p = len(self.columns_)
        a = np.zeros((p, p), dtype=int)
        for effect, causes in self.causes_.items():
            for cause in causes:
                if cause != effect or include_self_loops:
                    a[effect, cause] = 1
        return a

    def get_scores(self) -> np.ndarray:
        """Raw attention-score matrix (rows = targets, columns = sources)."""
        self._check_fitted()
        return self.scores_.copy()

    def discovered_edges(self) -> List[Tuple[str, str, int]]:
        """Validated edges as ``(cause, effect, delay)`` name triples."""
        self._check_fitted()
        edges = []
        for (effect, cause), delay in sorted(self.delays_.items()):
            if cause == effect:
                continue
            edges.append((self.columns_[cause], self.columns_[effect], delay))
        return edges

    def summary(self) -> pd.DataFrame:
        """Tidy DataFrame of validated causal relationships."""
        self._check_fitted()
        rows = [
            {"cause": c, "effect": e, "delay": d} for c, e, d in self.discovered_edges()
        ]
        return pd.DataFrame(rows, columns=["cause", "effect", "delay"])
