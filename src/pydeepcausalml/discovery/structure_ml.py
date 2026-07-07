"""Unified factory for causal structure learning methods."""

from __future__ import annotations

from ..base import BaseDeepEstimator
from .castle import CASTLE
from .dagma import DagmaLinear, DagmaNonlinearMLP
from .dynotears import DynoTEARS
from .nonlinear import DAGGNN, NOTEARSNonlinearMLP, NOTEARSNonlinearSobolev
from .notears import NOTEARSLinear

__all__ = [
    "causal_structure_ml",
    "causal_structure_ml_model_descriptions",
    "NOTEARSLinear",
    "NOTEARSNonlinearMLP",
    "NOTEARSNonlinearSobolev",
    "DAGGNN",
    "DagmaLinear",
    "DagmaNonlinearMLP",
    "DynoTEARS",
    "CASTLE",
]

_STRUCTURE_MODELS = {
    "notears_linear": NOTEARSLinear,
    "notears_nonlinear_mlp": NOTEARSNonlinearMLP,
    "notears_nonlinear_sobolev": NOTEARSNonlinearSobolev,
    "dag_gnn": DAGGNN,
    "daggnn": DAGGNN,
    "dagma_linear": DagmaLinear,
    "dagma_nonlinear_mlp": DagmaNonlinearMLP,
    "dynotears": DynoTEARS,
    "castle": CASTLE,
}


def causal_structure_ml_model_descriptions() -> dict[str, str]:
    """One-line descriptions for each structure-learning method."""
    return {
        "notears_linear": "Sparse linear DAG via smooth acyclicity constraint.",
        "notears_nonlinear_mlp": "Nonlinear NOTEARS with per-node MLPs.",
        "notears_nonlinear_sobolev": "Sobolev-basis nonlinear NOTEARS.",
        "dag_gnn": "VAE-style DAG-GNN with augmented-Lagrangian penalty.",
        "dagma_linear": "DAGMA linear structure learning via log-det acyclicity.",
        "dagma_nonlinear_mlp": "Nonlinear DAGMA with per-node MLPs.",
        "dynotears": "Lagged time-series DAG discovery.",
        "castle": "Causal structure learning with reconstruction regularization.",
    }


def causal_structure_ml(method: str = "notears_linear", **kwargs) -> BaseDeepEstimator:
    """Unified entry point for DAG discovery (mirrors R ``causalStructureML()``)."""
    key = method.lower().replace("-", "_")
    if key not in _STRUCTURE_MODELS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {list(_STRUCTURE_MODELS)}."
        )
    return _STRUCTURE_MODELS[key](**kwargs)
