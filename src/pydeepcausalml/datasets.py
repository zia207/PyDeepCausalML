"""Synthetic benchmark generators for causal effect estimation and discovery.

Each generator returns data together with its ground truth (CATE or causal
graph), so estimators can be validated before being applied to real data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "make_confounded_data",
    "make_var_data",
    "make_intervention_series",
]


def make_confounded_data(
    n: int = 5000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate a confounded observational study with heterogeneous effects.

    Causal structure::

        Z (age, education, prior income) -> T (program participation)
        Z -> Y (post-program income)
        T -> Y (true causal effect, heterogeneous in Z)

    Returns
    -------
    pandas.DataFrame
        Columns: ``age``, ``education``, ``prior_income``, ``treatment``,
        ``outcome``, ``true_cate``, ``propensity``.
    """
    rng = np.random.default_rng(random_state)

    age = rng.normal(35, 10, n).clip(18, 65)
    education = rng.integers(8, 22, n).astype(float)
    prior_inc = 20000 + 1500 * education + 200 * age + rng.normal(0, 5000, n)

    log_odds = -2 + 0.03 * age + 0.1 * education + 0.00005 * prior_inc
    propensity = 1.0 / (1.0 + np.exp(-log_odds))
    treatment = rng.binomial(1, propensity)

    true_cate = 3000 - 50 * age + 200 * np.clip(16 - education, 0, None)

    outcome = (
        15000
        + 1200 * education
        + 150 * age
        + 0.4 * prior_inc
        + treatment * true_cate
        + rng.normal(0, 4000, n)
    )

    return pd.DataFrame(
        {
            "age": age,
            "education": education,
            "prior_income": prior_inc,
            "treatment": treatment,
            "outcome": outcome,
            "true_cate": true_cate,
            "propensity": propensity,
        }
    )


def make_var_data(
    n_steps: int = 2000,
    noise_std: float = 0.3,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nonlinear VAR(1) process with a known 5-node causal graph.

    Graph (``A[i, j] = 1`` means :math:`X_j \\to X_i`)::

        X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4; X5 independent.

    Returns
    -------
    X : ndarray of shape (n_steps, 5)
        Simulated multivariate series.
    A_true : ndarray of shape (5, 5)
        Ground-truth adjacency matrix (row = effect, column = cause).
    """
    rng = np.random.default_rng(random_state)
    p = 5

    a_true = np.zeros((p, p), dtype=int)
    a_true[1, 0] = 1
    a_true[2, 0] = 1
    a_true[3, 1] = 1
    a_true[3, 2] = 1

    coeffs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.3, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.4, 0.0, 0.0],
            [0.0, 0.4, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
        ]
    )

    x = np.zeros((n_steps, p))
    x[0] = rng.standard_normal(p)
    for t in range(1, n_steps):
        x[t] = np.tanh(coeffs @ x[t - 1]) + rng.standard_normal(p) * noise_std
    return x, a_true


def make_intervention_series(
    n_units: int = 500,
    horizon: int = 24,
    intervention_start: int = 12,
    effect_size: float = 15.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Panel of unit-level time series where some units receive an intervention.

    Units with a high baseline self-select into treatment (confounding by
    baseline); the treatment effect grows linearly after onset.

    Returns
    -------
    pandas.DataFrame
        Long format with columns ``unit``, ``t``, ``value``, ``treated``,
        ``intervention``.
    """
    rng = np.random.default_rng(random_state)
    records = []
    for unit in range(n_units):
        baseline = rng.normal(100, 20)
        trend = rng.normal(0.5, 0.2)
        treated = int(baseline > 105)
        for t in range(horizon):
            active = int(treated and t >= intervention_start)
            effect = effect_size * active * (1 + 0.05 * (t - intervention_start))
            value = baseline + trend * t + effect + rng.normal(0, 5)
            records.append(
                {
                    "unit": unit,
                    "t": t,
                    "value": value,
                    "treated": treated,
                    "intervention": active,
                }
            )
    return pd.DataFrame.from_records(records)
