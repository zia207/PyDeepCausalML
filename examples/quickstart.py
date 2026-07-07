"""PyDeepCausalML end-to-end quickstart.

Runs one representative model from each family on synthetic benchmarks with
known ground truth and prints how well each recovers it.

Usage:  python examples/quickstart.py
"""

import numpy as np

from pydeepcausalml import (
    TCDF,
    CausalLSTMForecaster,
    DragonNet,
    DynoTEARS,
    NeuralDML,
    NeuralGrangerCMLP,
    TARNet,
)
from pydeepcausalml.datasets import (
    make_confounded_data,
    make_intervention_series,
    make_var_data,
)
from pydeepcausalml.metrics import graph_recovery_metrics, pehe

SEED = 0


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# --------------------------------------------------------------- #
section("1. Treatment-effect estimation (confounded observational data)")
df = make_confounded_data(n=4000, random_state=SEED)
X = df[["age", "education", "prior_income"]].values
t, y = df["treatment"].values, df["outcome"].values
true_ate = df["true_cate"].mean()
naive = df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean()
print(f"True ATE: ${true_ate:,.0f} | naive difference-in-means: ${naive:,.0f}")

for est in [TARNet(epochs=150, random_state=SEED), DragonNet(epochs=150, random_state=SEED)]:
    est.fit(X, t, y)
    name = type(est).__name__
    print(
        f"{name:10s} ATE: ${est.predict_ate(X):,.0f} | "
        f"PEHE: {pehe(df['true_cate'], est.predict_cate(X)):,.0f}"
    )

dml = NeuralDML(epochs=80, n_splits=2, random_state=SEED).fit(X, t, y)
lo, hi = dml.confidence_interval()
print(f"NeuralDML  ATE: ${dml.predict_ate():,.0f}  (95% CI ${lo:,.0f} – ${hi:,.0f})")

# --------------------------------------------------------------- #
section("2. Causal discovery on a nonlinear VAR with a known graph")
Xts, A_true = make_var_data(n_steps=2000, random_state=SEED)
for est in [
    NeuralGrangerCMLP(lag=3, epochs=150, random_state=SEED),
    DynoTEARS(lag=3, epochs=200, random_state=SEED),
]:
    est.fit(Xts)
    threshold = 0.03 if isinstance(est, NeuralGrangerCMLP) else 0.08
    result = graph_recovery_metrics(A_true, est.get_adjacency(threshold))
    print(f"{type(est).__name__:18s} F1={result['f1']:.2f} SHD={result['shd']}")

# --------------------------------------------------------------- #
section("3. TCDF: temporal causal discovery with delay estimation")
rng = np.random.default_rng(SEED)
n = 1500
a = rng.standard_normal(n)
b = np.zeros(n)
c = np.zeros(n)
for i in range(2, n):
    b[i] = 0.9 * a[i - 1] + 0.1 * rng.standard_normal()   # A -> B, delay 1
    c[i] = 0.9 * b[i - 2] + 0.1 * rng.standard_normal()   # B -> C, delay 2
tcdf = TCDF(kernel_size=4, epochs=800, random_state=SEED)
tcdf.fit(np.column_stack([a, b, c]), columns=["A", "B", "C"])
print(tcdf.summary().to_string(index=False))

# --------------------------------------------------------------- #
section("4. Counterfactual forecasting with a causal LSTM")
panel = make_intervention_series(n_units=300, horizon=24, intervention_start=6, random_state=SEED)
histories, treatments, futures = [], [], []
for unit in panel["unit"].unique():
    u = panel[panel["unit"] == unit].sort_values("t")
    histories.append(u["value"].values[:12])
    treatments.append(u["intervention"].values[:12].astype(float))
    futures.append(u["value"].values[12:24])
histories, treatments, futures = map(np.array, (histories, treatments, futures))

forecaster = CausalLSTMForecaster(pred_len=12, epochs=80, random_state=SEED)
forecaster.fit(histories, treatments, futures)
effect = forecaster.estimate_effect(
    histories, np.zeros_like(treatments), np.ones_like(treatments)
)
print(f"Estimated mean do(T=1) effect over 12-step horizon: {effect.mean():.2f}")

print("\nDone.")
