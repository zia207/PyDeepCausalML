"""Tests for neural Granger, TCDF, CausalLSTMForecaster, and CausalEGM."""

import numpy as np
import pandas as pd
import pytest

from pydeepcausalml import TCDF, CausalEGM, CausalLSTMForecaster, GrangerLSTM, NeuralGrangerCMLP
from pydeepcausalml.datasets import make_confounded_data, make_intervention_series, make_var_data
from pydeepcausalml.metrics import graph_recovery_metrics
from pydeepcausalml.timeseries import make_lagged_sequences


@pytest.fixture(scope="module")
def var_data():
    return make_var_data(n_steps=1500, random_state=0)


def test_make_lagged_sequences_shapes(var_data):
    x, _ = var_data
    xin, xout, mean, std = make_lagged_sequences(x, lag=4)
    assert xin.shape == (len(x) - 4, 4, 5)
    assert xout.shape == (len(x) - 4, 5)
    assert mean.shape == (5,) and std.shape == (5,)


def test_make_lagged_sequences_too_short():
    with pytest.raises(ValueError, match="too short"):
        make_lagged_sequences(np.random.randn(3, 2), lag=5)


def test_cmlp_recovers_var_graph(var_data):
    x, a_true = var_data
    est = NeuralGrangerCMLP(lag=3, epochs=120, lambda_group=0.01, random_state=0).fit(x)
    a_pred = est.get_adjacency(threshold=0.03)
    result = graph_recovery_metrics(a_true, a_pred)
    assert result["recall"] >= 0.75
    assert est.get_scores().shape == (5, 5)


def test_granger_lstm_scores_shape():
    # Small problem so n+1 LSTM fits stay fast.
    x, _ = make_var_data(n_steps=400, random_state=1)
    est = GrangerLSTM(lag=3, epochs=8, hidden_dim=16, n_layers=1, random_state=0).fit(x)
    assert est.get_scores().shape == (5, 5)
    a = est.get_adjacency()
    assert a.shape == (5, 5) and np.all(np.diag(a) == 0)


@pytest.fixture(scope="module")
def tcdf_chain_data():
    """X0 random; X1 caused by X0 at lag 1; X2 caused by X1 at lag 2."""
    rng = np.random.default_rng(0)
    n = 1200
    x0 = rng.standard_normal(n)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for t in range(2, n):
        x1[t] = 0.9 * x0[t - 1] + 0.1 * rng.standard_normal()
        x2[t] = 0.9 * x1[t - 2] + 0.1 * rng.standard_normal()
    return pd.DataFrame({"A": x0, "B": x1, "C": x2})


def test_tcdf_discovers_chain_with_delays(tcdf_chain_data):
    model = TCDF(kernel_size=4, epochs=600, random_state=1).fit(tcdf_chain_data)
    edges = {(c, e): d for c, e, d in model.discovered_edges()}
    assert ("A", "B") in edges
    assert ("B", "C") in edges
    assert edges[("A", "B")] == 1
    assert edges[("B", "C")] == 2
    # No spurious reverse edges.
    assert ("B", "A") not in edges and ("C", "B") not in edges


def test_tcdf_outputs(tcdf_chain_data):
    model = TCDF(kernel_size=4, epochs=200, random_state=0).fit(tcdf_chain_data)
    assert model.get_scores().shape == (3, 3)
    assert model.get_adjacency().shape == (3, 3)
    summary = model.summary()
    assert list(summary.columns) == ["cause", "effect", "delay"]
    assert model.receptive_field == 4  # kernel 4, single level: 1 + 3*1


def test_tcdf_column_mismatch():
    with pytest.raises(ValueError, match="columns"):
        TCDF(epochs=1).fit(np.random.randn(50, 3), columns=["a", "b"])


def _panel_tensors(df, seq_len=12, pred_len=12):
    xs, ts, ys = [], [], []
    for unit in df["unit"].unique():
        u = df[df["unit"] == unit].sort_values("t")
        v, i = u["value"].values, u["intervention"].values
        xs.append(v[:seq_len])
        ts.append(i[:seq_len].astype(float))
        ys.append(v[seq_len : seq_len + pred_len])
    return np.array(xs), np.array(ts), np.array(ys)


def test_causal_lstm_forecaster_effect_direction():
    # Intervention onset inside the history window so the treatment encoder
    # sees informative variation across units.
    df = make_intervention_series(
        n_units=200, horizon=24, intervention_start=6, random_state=0
    )
    x, t, y = _panel_tensors(df)
    est = CausalLSTMForecaster(pred_len=12, epochs=60, random_state=0).fit(x, t, y)

    forecast = est.forecast(x[:10], t[:10])
    assert forecast.shape == (10, 12)

    # Setting the intervention on for everyone should raise forecasts on
    # average relative to setting it off (positive treatment effect).
    effect = est.estimate_effect(x, np.zeros_like(t), np.ones_like(t))
    assert effect.mean() > 0


def test_causal_lstm_forecaster_bad_target_shape():
    with pytest.raises(ValueError, match="pred_len"):
        CausalLSTMForecaster(pred_len=12, epochs=1).fit(
            np.random.randn(5, 12), np.random.randn(5, 12), np.random.randn(5, 6)
        )


def test_causal_egm_ate_direction():
    df = make_confounded_data(n=2500, random_state=0)
    x = df[["age", "education", "prior_income"]].values
    est = CausalEGM(epochs=40, random_state=0).fit(x, df["treatment"], df["outcome"])

    ite = est.predict_ite(x[:100])
    assert ite.shape == (100,)
    true_ate = df["true_cate"].mean()
    assert abs(est.predict_ate(x) - true_ate) < 2000

    e = est.predict_propensity(x[:50])
    assert (e >= 0).all() and (e <= 1).all()
