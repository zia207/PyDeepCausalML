"""Shared utilities: reproducibility, device handling, tensor conversion, validation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence, torch.Tensor]

__all__ = [
    "set_seed",
    "resolve_device",
    "get_default_device",
    "module_to_device",
    "to_tensor",
    "to_numpy",
    "check_array",
    "check_binary_treatment",
    "notears_acyclicity_h",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_default_device() -> torch.device:
    """Return the best available accelerator: CUDA, then MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device spec.

    ``None`` or ``"auto"`` selects CUDA when available, then Apple MPS, else CPU.
    Explicit strings such as ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``"mps"`` are
    also accepted.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return get_default_device()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if dev.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
    return dev


def module_to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move a module to ``device``, falling back to CPU on accelerator errors."""
    try:
        return module.to(device)
    except RuntimeError:
        if device.type == "cpu":
            raise
        return module.to(torch.device("cpu"))


def notears_acyclicity_h(w: torch.Tensor) -> torch.Tensor:
    """Smooth DAG acyclicity constraint h(W) = tr(exp(W ⊙ W)) − d."""
    d = w.shape[0]
    eye = torch.eye(d, device=w.device, dtype=w.dtype)
    m = w * w
    if w.is_cuda or (hasattr(w, "device") and w.device.type == "mps"):
        m_cpu = m.double().cpu()
        h = torch.trace(torch.linalg.matrix_exp(m_cpu)) - d
        return h.to(w.dtype).to(w.device)
    return torch.trace(torch.linalg.matrix_exp(m)) - d


def to_tensor(
    x: ArrayLike,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert array-like input to a torch tensor without copying when possible."""
    if isinstance(x, torch.Tensor):
        t = x.to(dtype)
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        t = torch.as_tensor(x.to_numpy(), dtype=dtype)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype)
    return t.to(device) if device is not None else t


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert array-like input (including tensors) to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)


def check_array(x: ArrayLike, name: str = "X", ndim: int = 2) -> np.ndarray:
    """Validate and coerce input to a finite float64 NumPy array of given rank."""
    arr = to_numpy(x).astype(np.float64, copy=False)
    if arr.ndim == 1 and ndim == 2:
        arr = arr.reshape(-1, 1)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def check_binary_treatment(t: ArrayLike, name: str = "treatment") -> np.ndarray:
    """Validate a binary treatment vector coded 0/1."""
    arr = to_numpy(t).astype(np.float64, copy=False).ravel()
    uniq = np.unique(arr)
    if not np.isin(uniq, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary (0/1); found values {uniq[:10]}.")
    return arr
