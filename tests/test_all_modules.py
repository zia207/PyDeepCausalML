"""Smoke tests that every public estimator and support module can fit and predict.

Quality bars live in the family-specific test modules. This file only checks
that each public class trains for a few epochs and returns the documented shape.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import pydeepcausalml
from pydeepcausalml import (
    CFRNet,
    CRN,
    CUTS,
    CEVAE,
    CausalDiscrepancyVAE,
    CausalEGM,
    CausalGAN,
    CausalGNN,
    CausalLSTM,
    CausalLSTMForecaster,
    CausalTransformer,
    CausalVAE,
    DAGGNN,
    DECI,
    DSCM,
    DeepSCM,
    DeepSynth,
    DagmaLinear,
    DagmaNonlinearMLP,
    DragonNet,
    DynoTEARS,
    GANITE,
    GNet,
    GVAR,
    GrangerLSTM,
    IVAE,
    InterventionAwareRNN,
    NeuralDML,
    NeuralGrangerCMLP,
    NeuralGrangerCLSTM,
    NeuralGrangerEconomySRU,
    NeuralRelationalInference,
    NOTEARSLinear,
    NOTEARSNonlinearMLP,
    NOTEARSNonlinearSobolev,
    RETAIN,
    TARNet,
    TCDF,
    TFTNet,
    attn_causal_model,
    causal_structure_ml,
    counterfactual_model,
    gnn_causal_model,
    neural_granger_model,
    rnn_causal_model,
)
from pydeepcausalml.datasets import make_confounded_data, make_var_data
from pydeepcausalml.plotting import plot_causal_graph, plot_score_heatmap, plot_training_history

FAST = dict(epochs=2, batch_size=64, device="cpu", random_state=0)


@pytest.fixture(scope="module")
def effect():
    df = make_confounded_data(n=80, random_state=0)
    x = df[["age", "education", "prior_income"]].values
    return x, df["treatment"].values, df["outcome"].values


@pytest.fixture(scope="module")
def series():
    x, _ = make_var_data(n_steps=40, random_state=0)
    return x


def test_every_public_name_is_exported():
    missing = [name for name in pydeepcausalml.__all__ if not hasattr(pydeepcausalml, name)]
    assert missing == []


@pytest.mark.parametrize(
    "cls",
    [TARNet, CFRNet, DragonNet, GANITE, CEVAE, CausalEGM, CausalGAN, CausalDiscrepancyVAE, DSCM],
)
def test_effect_and_generative_estimators(cls, effect):
    x, t, y = effect
    est = cls(**FAST).fit(x, t, y)
    cate = est.predict_cate(x[:8])
    assert cate.shape == (8,)
    assert np.isfinite(cate).all()
    assert np.isfinite(est.predict_ate(x[:8]))


def test_neural_dml(effect):
    x, t, y = effect
    est = NeuralDML(epochs=2, n_splits=2, batch_size=32, device="cpu", random_state=0).fit(x, t, y)
    assert np.isfinite(est.predict_ate())
    lo, hi = est.confidence_interval()
    assert lo <= est.ate_ <= hi


def test_ivae_and_causalvae(effect):
    x, t, _ = effect
    ivae = IVAE(latent_dim=4, **FAST).fit(x, t.reshape(-1, 1))
    z = ivae.transform(x[:6])
    assert z.shape == (6, 4)
    assert np.isfinite(z).all()
    cvae = CausalVAE(latent_dim=4, **FAST).fit(x)
    assert cvae.adjacency_matrix().shape == (4, 4)
    zc = cvae.transform(x[:6])
    assert zc.shape[0] == 6
    assert np.isfinite(zc).all()


@pytest.mark.parametrize(
    "method",
    [
        "notears_linear",
        "notears_nonlinear_mlp",
        "notears_nonlinear_sobolev",
        "dag_gnn",
        "dagma_linear",
        "dagma_nonlinear_mlp",
        "dynotears",
        "castle",
    ],
)
def test_structure_factory(method, effect, series):
    data = series if method == "dynotears" else effect[0]
    kwargs = dict(epochs=2, device="cpu", random_state=0)
    if method == "dagma_linear":
        kwargs["max_iter"] = 15
    if method == "dynotears":
        kwargs["lag"] = 2
    est = causal_structure_ml(method, **kwargs).fit(data)
    adj = est.adjacency_matrix()
    assert adj.shape[0] == data.shape[1]
    assert np.isfinite(adj).all()


def test_structure_classes(effect, series):
    x = effect[0]
    for cls, kwargs in (
        (NOTEARSLinear, dict(epochs=2, n_outer=1, device="cpu", random_state=0)),
        (NOTEARSNonlinearMLP, FAST),
        (NOTEARSNonlinearSobolev, FAST),
        (DAGGNN, FAST),
        (DagmaLinear, dict(max_iter=15, random_state=0)),
        (DagmaNonlinearMLP, FAST),
    ):
        adj = cls(**kwargs).fit(x).adjacency_matrix()
        assert adj.shape == (x.shape[1], x.shape[1])
    dyno = DynoTEARS(lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    assert dyno.get_adjacency(threshold=0.0).shape == (series.shape[1], series.shape[1])


@pytest.mark.parametrize("method", ["cmlp", "clstm", "economysru", "nri"])
def test_neural_granger_factory(method, series):
    est = neural_granger_model(method, lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    adj = est.adjacency_matrix()
    assert adj.shape == (series.shape[1], series.shape[1])


def test_granger_classes(series):
    cmlp = NeuralGrangerCMLP(lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    assert cmlp.get_scores().shape == (series.shape[1], series.shape[1])
    clstm = NeuralGrangerCLSTM(lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    assert clstm.adjacency_matrix().shape[0] == series.shape[1]
    sru = NeuralGrangerEconomySRU(lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    assert sru.adjacency_matrix().shape[0] == series.shape[1]
    nri = NeuralRelationalInference(lag=2, epochs=2, device="cpu", random_state=0).fit(series)
    assert nri.adjacency_matrix().shape[0] == series.shape[1]
    lstm = GrangerLSTM(lag=2, epochs=2, hidden_dim=8, n_layers=1, device="cpu", random_state=0).fit(series)
    assert lstm.get_scores().shape == (series.shape[1], series.shape[1])


def test_attention_rnn_gnn(series):
    p = series.shape[1]
    n = len(series)
    transformer = CausalTransformer(lag=2, d_model=8, nhead=2, n_layers=1, **FAST).fit(series)
    assert transformer.predict(series).shape == (n - 2, p)
    assert transformer.causal_matrix().shape == (p, p)
    tft = TFTNet(lag=2, hidden=8, **FAST).fit(series)
    assert tft.predict(series).shape == (n - 2, p)
    assert tft.causal_matrix().shape == (p, p)
    lstm = CausalLSTM(lag=2, hidden=8, n_layers=1, **FAST).fit(series)
    assert lstm.predict(series).shape == (n - 2, p)
    retain = RETAIN(lag=2, hidden=8, **FAST).fit(series)
    assert retain.predict(series).shape[0] == n - 2
    irnn = InterventionAwareRNN(lag=2, hidden=8, **FAST).fit(series)
    assert irnn.predict(series).shape[0] == n - 2
    assert irnn.causal_matrix().shape[0] == p
    for cls in (GVAR, CausalGNN, CUTS):
        mat = cls(lag=2, **FAST).fit(series).causal_matrix()
        assert mat.shape[0] == p
    tcdf = TCDF(kernel_size=2, epochs=2, significance=1.0, random_state=0).fit(series)
    assert tcdf.get_adjacency().shape == (p, p)
    assert tcdf.get_scores().shape == (p, p)


def test_attention_rnn_gnn_factories(series):
    for method, kwargs in (
        ("causal_transformer", dict(lag=2, d_model=8, nhead=2, n_layers=1)),
        ("tft", dict(lag=2, hidden=8)),
        ("tcdf", dict(kernel_size=2, significance=1.0)),
    ):
        est = attn_causal_model(method, epochs=2, device="cpu", random_state=0, **kwargs).fit(series)
        assert hasattr(est, "causal_matrix") or hasattr(est, "get_adjacency")
    for method, kwargs in (
        ("causal_lstm", dict(hidden=8, n_layers=1)),
        ("retain", dict(hidden=8)),
        ("intervention_rnn", dict(hidden=8)),
    ):
        est = rnn_causal_model(method, lag=2, epochs=2, device="cpu", **kwargs).fit(series)
        assert est.predict(series).shape[0] == len(series) - 2
    for method in ("gvar", "causal_gnn", "cuts"):
        est = gnn_causal_model(method, lag=2, epochs=2, device="cpu").fit(series)
        assert est.causal_matrix().shape[0] == series.shape[1]


def test_counterfactual_and_scm(series, effect):
    x = series
    n, p = x.shape
    t = np.zeros(n)
    y = x[:, 0]
    synth = DeepSynth(lag=2, **FAST).fit(x, y)
    assert synth.predict_counterfactual(x).shape[0] == n - 2
    crn = CRN(lag=2, **FAST).fit(x, t, y)
    assert crn.predict_ite(x, t).shape[0] == n - 2
    gnet = GNet(lag=2, **FAST).fit(x, t, y)
    assert gnet.predict_ite(x).shape[0] == n - 2
    for method in ("deepsynth", "crn", "gnet"):
        est = counterfactual_model(method, lag=2, epochs=2, device="cpu")
        if method == "deepsynth":
            est.fit(x, y)
            out = est.predict_counterfactual(x)
        elif method == "gnet":
            est.fit(x, t, y)
            out = est.predict_ite(x)
        else:
            est.fit(x, t, y)
            out = est.predict_ite(x, t)
        assert len(out) == n - 2
    scm = DeepSCM(lag=2, **FAST).fit(x)
    intervened = scm.intervene(x, var_idx=0, value=0.0)
    assert intervened.shape == x.shape
    deci = DECI(lag=2, **FAST).fit(x)
    assert deci.adjacency_matrix().shape == (p, p)
    assert np.isfinite(deci.predict_ate(x, intervention_var=0, n_samples=8))


def test_forecaster_factory(series):
    hist = series[:20]
    future = np.random.default_rng(0).standard_normal((1, 4))
    # Panel API: one unit, history length 12, horizon 4.
    outcome = series[:12, 0][None, :]
    treat = np.zeros((1, 12))
    est = rnn_causal_model("causal_lstm_forecaster", pred_len=4, epochs=2, device="cpu").fit(
        outcome, treat, future
    )
    assert isinstance(est, CausalLSTMForecaster)
    assert est.forecast(outcome, treat).shape == (1, 4)
    assert hist.shape[0] == 20


def test_unknown_factory_methods():
    with pytest.raises(ValueError):
        causal_structure_ml("nope")
    with pytest.raises(ValueError):
        neural_granger_model("nope")
    with pytest.raises(ValueError):
        attn_causal_model("nope")
    with pytest.raises(ValueError):
        rnn_causal_model("nope")
    with pytest.raises(ValueError):
        gnn_causal_model("nope")
    with pytest.raises(ValueError):
        counterfactual_model("nope")


def test_plotting_helpers(effect):
    x, t, y = effect
    est = TARNet(**FAST).fit(x, t, y)
    a = np.array([[0, 1], [0, 0]])
    ax = plot_causal_graph(a, node_names=["A", "B"], delays={(0, 1): 2})
    assert ax is not None
    ax = plot_score_heatmap(np.abs(a).astype(float), node_names=["A", "B"])
    assert ax is not None
    ax = plot_training_history(est.history_)
    assert ax is not None
