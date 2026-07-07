"""Visualization helpers (requires the ``plot`` extra: matplotlib, seaborn)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional

import numpy as np

from .utils import to_numpy

__all__ = ["plot_causal_graph", "plot_score_heatmap", "plot_training_history"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install pydeepcausalml[plot]`."
        ) from exc


def plot_causal_graph(
    adjacency,
    node_names: Optional[Sequence[str]] = None,
    title: str = "Causal graph",
    ax=None,
    node_color: str = "steelblue",
    delays: Optional[dict] = None,
):
    """Draw a directed causal graph from an adjacency matrix.

    Parameters
    ----------
    adjacency : array-like of shape (p, p)
        ``A[i, j] = 1`` draws an edge :math:`X_j \\to X_i`.
    delays : dict, optional
        Mapping ``(effect_idx, cause_idx) -> delay`` used as edge labels
        (matches :attr:`pydeepcausalml.timeseries.TCDF.delays_`).
    """
    plt = _require_matplotlib()
    import networkx as nx

    a = to_numpy(adjacency).astype(int)
    p = a.shape[0]
    names = list(node_names) if node_names is not None else [f"X{i + 1}" for i in range(p)]

    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    edge_labels = {}
    for i in range(p):
        for j in range(p):
            if a[i, j] == 1:
                graph.add_edge(names[j], names[i])
                if delays and (i, j) in delays:
                    edge_labels[(names[j], names[i])] = delays[(i, j)]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_color, node_size=900, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    return ax


def plot_score_heatmap(
    scores,
    node_names: Optional[Sequence[str]] = None,
    title: str = "Causal scores",
    ax=None,
    zero_diagonal: bool = True,
):
    """Heatmap of a causal-score matrix (rows = effects, columns = causes)."""
    plt = _require_matplotlib()

    m = to_numpy(scores).astype(float).copy()
    if zero_diagonal:
        np.fill_diagonal(m, 0.0)
    p = m.shape[0]
    names = list(node_names) if node_names is not None else [f"X{i + 1}" for i in range(p)]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(m, cmap="YlOrRd")
    ax.set_xticks(range(p), names, rotation=45, ha="right")
    ax.set_yticks(range(p), names)
    ax.set_xlabel("Source (cause)")
    ax.set_ylabel("Target (effect)")
    ax.set_title(title, fontweight="bold")
    for i in range(p):
        for j in range(p):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046)
    return ax


def plot_training_history(history: dict, keys: Optional[List[str]] = None, ax=None):
    """Plot per-epoch loss curves stored in an estimator's ``history_``."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for key in keys or list(history):
        values = history.get(key, [])
        if len(values) > 1:
            ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training history", fontweight="bold")
    return ax
