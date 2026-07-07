"""Tests for TARNet, CFRNet, DragonNet, and NeuralDML.

Estimator quality tests use a confounded simulation with known ground truth
and assert that each estimator beats a wide-but-meaningful accuracy bar with
small budgets, keeping the suite fast on CPU.
"""

import numpy as np
import pytest

from pydeepcausalml import CFRNet, DragonNet, NeuralDML, TARNet
from pydeepcausalml.datasets import make_confounded_data

COVARIATES = ["age", "education", "prior_income"]


@pytest.fixture(scope="module")
def data():
    df = make_confounded_data(n=3000, random_state=0)
    return df[COVARIATES].values, df["treatment"].values, df["outcome"].values, df


@pytest.mark.parametrize("cls", [TARNet, CFRNet, DragonNet])
def test_meta_learner_recovers_ate(cls, data):
    x, t, y, df = data
    est = cls(epochs=60, random_state=0).fit(x, t, y)
    ate_hat = est.predict_ate(x)
    true_ate = df["true_cate"].mean()
    # Naive difference-in-means is biased by thousands of dollars here;
    # a causal estimator should land within $1200 of the truth.
    assert abs(ate_hat - true_ate) < 1200


@pytest.mark.parametrize("cls", [TARNet, CFRNet, DragonNet])
def test_cate_shapes_and_history(cls, data):
    x, t, y, _ = data
    est = cls(epochs=5, random_state=0).fit(x, t, y)
    cate = est.predict_cate(x[:50])
    assert cate.shape == (50,)
    assert len(est.history_["loss"]) == 5


def test_potential_outcomes_consistent(data):
    x, t, y, _ = data
    est = TARNet(epochs=5, random_state=0).fit(x, t, y)
    y0, y1 = est.predict_potential_outcomes(x[:20])
    np.testing.assert_allclose(y1 - y0, est.predict_cate(x[:20]), rtol=1e-5)


def test_dragonnet_propensity_range(data):
    x, t, y, _ = data
    est = DragonNet(epochs=10, random_state=0).fit(x, t, y)
    e = est.predict_propensity(x[:100])
    assert e.shape == (100,)
    assert (e >= 0).all() and (e <= 1).all()


def test_neural_dml_ate_and_ci(data):
    x, t, y, df = data
    est = NeuralDML(epochs=40, n_splits=2, random_state=0).fit(x, t, y)
    assert abs(est.predict_ate() - df["true_cate"].mean()) < 1500
    lo, hi = est.confidence_interval()
    assert lo < est.ate_ < hi


def test_neural_dml_rejects_single_split():
    with pytest.raises(ValueError, match="n_splits"):
        NeuralDML(n_splits=1)


def test_unfitted_raises(data):
    x, *_ = data
    with pytest.raises(RuntimeError, match="not fitted"):
        TARNet().predict_cate(x)


def test_length_mismatch_raises(data):
    x, t, y, _ = data
    with pytest.raises(ValueError, match="same length"):
        TARNet(epochs=1).fit(x[:-5], t, y)


def test_get_set_params():
    est = TARNet(repr_dim=16)
    assert est.get_params()["repr_dim"] == 16
    est.set_params(repr_dim=8)
    assert est.repr_dim == 8
    with pytest.raises(ValueError, match="Unknown parameter"):
        est.set_params(bogus=1)
