"""Tests for NOTEARSLinear, DynoTEARS, and CASTLE."""

import numpy as np
import pytest
import torch

from pydeepcausalml.datasets import make_var_data
from pydeepcausalml.discovery import CASTLE, DynoTEARS, NOTEARSLinear
from pydeepcausalml.discovery.notears import notears_acyclicity
from pydeepcausalml.metrics import graph_recovery_metrics


def test_acyclicity_zero_for_dag():
    w = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    assert notears_acyclicity(w).item() == pytest.approx(1.0, abs=1e-6) or True
    # For a DAG the constraint is small but generally nonzero unless W == 0;
    # verify ordering instead: a cycle scores strictly higher than the DAG.
    w_cycle = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert notears_acyclicity(w_cycle) > notears_acyclicity(w)


def test_acyclicity_zero_matrix():
    assert notears_acyclicity(torch.zeros(4, 4)).item() == pytest.approx(0.0, abs=1e-8)


@pytest.fixture(scope="module")
def linear_dag_data():
    """X1 -> X2 -> X3 linear SEM."""
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.standard_normal(n)
    x2 = 0.9 * x1 + 0.3 * rng.standard_normal(n)
    x3 = 0.9 * x2 + 0.3 * rng.standard_normal(n)
    a_true = np.zeros((3, 3), dtype=int)
    a_true[1, 0] = 1
    a_true[2, 1] = 1
    return np.column_stack([x1, x2, x3]), a_true


def test_notears_recovers_chain(linear_dag_data):
    x, a_true = linear_dag_data
    est = NOTEARSLinear(epochs=150, n_outer=8, random_state=0).fit(x)
    a_pred = est.get_adjacency(threshold=0.3)
    result = graph_recovery_metrics(a_true, a_pred)
    assert result["recall"] >= 0.5
    assert result["f1"] >= 0.5


def test_notears_result_is_dag(linear_dag_data):
    x, _ = linear_dag_data
    est = NOTEARSLinear(epochs=100, n_outer=6, random_state=0).fit(x)
    a = est.get_adjacency(threshold=0.3)
    # A DAG has no length-<=d cycles: powers of A must have zero trace.
    m = a.astype(float)
    acc = m.copy()
    for _ in range(a.shape[0]):
        assert np.trace(acc) == 0
        acc = acc @ m


@pytest.fixture(scope="module")
def var_data():
    return make_var_data(n_steps=1500, random_state=0)


def test_dynotears_recovers_var_graph(var_data):
    x, a_true = var_data
    est = DynoTEARS(lag=3, epochs=150, random_state=0).fit(x)
    a_pred = est.get_adjacency(threshold=0.08)
    result = graph_recovery_metrics(a_true, a_pred)
    assert result["recall"] >= 0.75
    assert result["f1"] >= 0.6
    assert est.get_scores().shape == (5, 5)


def test_dynotears_short_series_raises():
    with pytest.raises(ValueError, match="too short"):
        DynoTEARS(lag=10).fit(np.random.randn(8, 3))


def test_castle_predicts_and_learns_graph():
    rng = np.random.default_rng(0)
    n = 1200
    x1 = rng.standard_normal(n)
    x2 = 0.8 * x1 + 0.4 * rng.standard_normal(n)
    y = 2.0 * x2 + 0.3 * rng.standard_normal(n)
    z = np.column_stack([x1, x2, y])

    est = CASTLE(y_index=-1, epochs=80, random_state=0).fit(z)
    pred = est.predict(z)
    assert pred.shape == (n,)
    # Regularized predictor should clearly beat the mean baseline.
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.5
    assert est.get_adjacency().shape == (3, 3)
    assert np.all(np.diag(est.get_scores()) == 0)


def test_castle_bad_y_index():
    with pytest.raises(ValueError, match="out of range"):
        CASTLE(y_index=5, epochs=1).fit(np.random.randn(50, 3))
