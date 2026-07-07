"""Smoke tests for newly added models, factory functions, and device handling."""

import importlib

import numpy as np
import pytest
import torch

import pydeepcausalml
from pydeepcausalml import (
    CEVAE,
    CRN,
    CausalDiscrepancyVAE,
    CausalGAN,
    CausalVAE,
    DAGGNN,
    DagmaLinear,
    DSCM,
    GANITE,
    GVAR,
    IVAE,
    RETAIN,
    causal_structure_ml,
    causal_structure_ml_model_descriptions,
    counterfactual_model,
    get_default_device,
    gnn_causal_model,
    neural_granger_model,
    resolve_device,
    rnn_causal_model,
)
from pydeepcausalml.datasets import make_confounded_data, make_var_data


@pytest.fixture(scope="module")
def effect_data():
    df = make_confounded_data(n=500, random_state=0)
    x = df[["age", "education", "prior_income"]].values
    return x, df["treatment"].values, df["outcome"].values


@pytest.fixture(scope="module")
def ts_data():
    x, _ = make_var_data(n_steps=120, random_state=0)
    return x


def test_public_api_exports():
    """Every name in __all__ should be importable from the top-level package."""
    mod = importlib.import_module("pydeepcausalml")
    for name in mod.__all__:
        assert hasattr(mod, name), f"missing export: {name}"


def test_version_matches_pyproject():
    assert pydeepcausalml.__version__ == "0.2.0"


def test_structure_ml_descriptions():
    desc = causal_structure_ml_model_descriptions()
    assert "notears_linear" in desc
    assert "dagma_linear" in desc
    assert len(desc) >= 8


def test_device_auto_and_explicit():
    dev = resolve_device(None)
    assert dev.type in ("cpu", "cuda", "mps")
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    assert get_default_device().type in ("cpu", "cuda", "mps")


def test_cuda_unavailable_raises():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.parametrize("device", [None, "cpu"])
def test_ganite_cevae_fit(effect_data, device):
    x, t, y = effect_data
    for cls in (GANITE, CEVAE):
        est = cls(epochs=3, batch_size=64, device=device, random_state=0).fit(x, t, y)
        cate = est.predict_cate(x[:20])
        assert cate.shape == (20,)


def test_generative_models(effect_data):
    x, t, y = effect_data
    gan = CausalGAN(epochs=3, device="cpu").fit(x, t, y)
    assert gan.predict_cate(x[:10]).shape == (10,)
    ivae = IVAE(epochs=3, device="cpu").fit(x, t.reshape(-1, 1))
    assert ivae.transform(x[:5]).shape[0] == 5
    cvae = CausalVAE(epochs=3, latent_dim=4, device="cpu").fit(x)
    assert cvae.adjacency_matrix().shape == (4, 4)
    cdvae = CausalDiscrepancyVAE(epochs=3, latent_dim=4, device="cpu").fit(x, t, y)
    assert cdvae.predict_cate(x[:5]).shape == (5,)
    dscm = DSCM(epochs=3, device="cpu").fit(x, t, y)
    assert dscm.predict_cate(x[:5]).shape == (5,)


def test_discovery_models(effect_data):
    x, _, _ = effect_data
    for method in ("notears_linear", "dag_gnn", "dagma_linear"):
        est = causal_structure_ml(method, epochs=5, device="cpu").fit(x)
        adj = est.adjacency_matrix()
        assert adj.shape == (x.shape[1], x.shape[1])
    dagma = DagmaLinear(max_iter=50).fit(x)
    assert dagma.adjacency_matrix().shape == (x.shape[1], x.shape[1])
    daggnn = DAGGNN(epochs=3, device="cpu").fit(x)
    assert daggnn.adjacency_matrix().shape == (x.shape[1], x.shape[1])


def test_timeseries_factories(ts_data, effect_data):
    x = ts_data
    x_eff, t, y = effect_data
    for method in ("cmlp", "clstm"):
        est = neural_granger_model(method, epochs=3, lag=3, device="cpu").fit(x)
        assert est.adjacency_matrix().shape[0] == x.shape[1]
    gvar = gnn_causal_model("gvar", epochs=3, lag=3, device="cpu").fit(x)
    assert gvar.causal_matrix().shape[0] == x.shape[1]
    retain = rnn_causal_model("retain", epochs=3, lag=3, device="cpu").fit(x)
    assert retain.predict(x).shape[0] == len(x) - 3
    crn = counterfactual_model("crn", epochs=3, lag=3, device="cpu").fit(x, t[: len(x)], y[: len(x)])
    ite = crn.predict_ite(x, t[: len(x)])
    assert len(ite) == len(x) - 3


def test_counterfactual_model(ts_data, effect_data):
    x_ts = ts_data
    t = np.random.default_rng(0).integers(0, 2, len(x_ts))
    y = np.random.default_rng(1).standard_normal(len(x_ts))
    crn = counterfactual_model("crn", epochs=3, lag=3, device="cpu").fit(x_ts, t, y)
    ite = crn.predict_ite(x_ts, t)
    assert len(ite) == len(x_ts) - 3
    assert CRN(epochs=1, lag=3).get_params()["lag"] == 3
