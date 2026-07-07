"""Evaluation metrics for causal effect estimation, graph recovery, and forecasting."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .utils import to_numpy

__all__ = ["pehe", "ate_error", "graph_recovery_metrics", "shd", "mase"]


def pehe(true_cate, est_cate) -> float:
    """Precision in Estimation of Heterogeneous Effects (root mean squared CATE error)."""
    true_cate, est_cate = to_numpy(true_cate).ravel(), to_numpy(est_cate).ravel()
    return float(np.sqrt(np.mean((true_cate - est_cate) ** 2)))


def ate_error(true_ate: float, est_ate: float) -> float:
    """Absolute error of an average-treatment-effect estimate."""
    return float(abs(true_ate - est_ate))


def shd(a_true, a_pred) -> int:
    """Structural Hamming Distance between two binary adjacency matrices.

    Counts edge insertions plus deletions needed to turn ``a_pred`` into
    ``a_true`` (diagonal excluded).
    """
    a_true, a_pred = to_numpy(a_true).astype(int), to_numpy(a_pred).astype(int)
    mask = ~np.eye(a_true.shape[0], dtype=bool)
    return int(np.sum(a_true[mask] != a_pred[mask]))


def graph_recovery_metrics(a_true, a_pred) -> Dict[str, float]:
    """Edge-level precision, recall, F1, and SHD for causal graph recovery.

    Parameters
    ----------
    a_true, a_pred : array-like of shape (p, p)
        Binary adjacency matrices with ``A[i, j] = 1`` meaning
        :math:`X_j \\to X_i`. Diagonals are ignored.
    """
    a_true = to_numpy(a_true).astype(int)
    a_pred = to_numpy(a_pred).astype(int)
    if a_true.shape != a_pred.shape:
        raise ValueError(f"Shape mismatch: {a_true.shape} vs {a_pred.shape}.")

    mask = ~np.eye(a_true.shape[0], dtype=bool)
    y_true, y_pred = a_true[mask].ravel(), a_pred[mask].ravel()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "shd": fp + fn,
    }


def mase(y_true, y_pred) -> float:
    """Mean Absolute Scaled Error for a forecast against a naive one-step baseline.

    Follows the definition used by the TCDF prediction-accuracy evaluation:
    absolute forecast error scaled by the in-sample naive (lag-1) error.
    """
    y_true, y_pred = to_numpy(y_true).ravel(), to_numpy(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    n = len(y_true)
    if n < 2:
        raise ValueError("MASE requires at least two observations.")
    numerator = np.abs(y_true - y_pred).sum()
    denominator = (n / (n - 1)) * np.abs(np.diff(y_true)).sum()
    return float(numerator / denominator) if denominator != 0 else 0.0
