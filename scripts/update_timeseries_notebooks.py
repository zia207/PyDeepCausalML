#!/usr/bin/env python3
"""Refactor timeseries tutorial notebooks to use pydeepcausalml (CASTLE-style setup)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

SETUP_INSTALL = '''import importlib
import subprocess
import sys

PACKAGES = [
    "numpy", "pandas", "scipy", "torch", "scikit-learn",
    "matplotlib", "seaborn", "networkx", "yfinance",
]

for pkg in PACKAGES:
    mod = "sklearn" if pkg == "scikit-learn" else pkg
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import pydeepcausalml  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/zia207/PyDeepCausalML.git"]
    )

import pydeepcausalml
print("pydeepcausalml", pydeepcausalml.__version__, "ready.")
'''

IMPORTS_BASE = '''import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pydeepcausalml import set_seed

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
'''

SEED_CELL = '''set_seed(42)
run_fast = True
'''

ETF_TICKERS = '''TICKERS = {
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "ConsumerDisc",
    "XLP": "ConsumerStap",
    "XLU": "Utilities",
}
'''

DATA_LOAD = ETF_TICKERS + '''
import yfinance as yf

raw = yf.download(
    list(TICKERS.keys()),
    start="2018-01-01",
    end="2024-01-01",
    auto_adjust=True,
    progress=False,
)["Close"]

returns = np.log(raw / raw.shift(1)).dropna()
returns.columns = [TICKERS[t] for t in returns.columns]

if returns.shape[0] < 50:
    print("yfinance unavailable; using synthetic demo data.")
    rng = np.random.default_rng(42)
    T, d = 1500, len(TICKERS)
    VAR_NAMES = list(TICKERS.values())
    market = rng.normal(0.0, 0.8, size=(T, 1)).astype(np.float64)
    idio = rng.normal(0.0, 0.6, size=(T, d)).astype(np.float64)
    loading = np.linspace(0.5, 1.1, d, dtype=np.float64)[None, :]
    data_np = (market @ loading + idio).astype(np.float64)
else:
    data_np = returns.values.astype(np.float64)
    T, d = data_np.shape
    VAR_NAMES = returns.columns.tolist()

print(f"Shape: {data_np.shape}")
print(f"Variables ({d}): {VAR_NAMES}")
print(f"Time steps: {T}")
'''

DATA_LOAD_ROBUST = ETF_TICKERS + '''
import yfinance as yf


def fetch_close_prices(tickers, start="2018-01-01", end="2024-01-01"):
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, group_by="ticker", threads=False,
    )
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
    elif "Close" in raw.columns:
        close = raw[["Close"]]
        close.columns = tickers[:1]

    if close is None or close.dropna(how="all").empty:
        cols = []
        for t in tickers:
            one = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if "Close" in one.columns and not one["Close"].dropna().empty:
                cols.append(one["Close"].rename(t))
        close = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    return close.sort_index().dropna(how="all")


close = fetch_close_prices(list(TICKERS.keys()))
if close.empty or close.shape[0] < 50:
    print("yfinance unavailable; using synthetic demo data.")
    rng = np.random.default_rng(42)
    T, d = 1500, len(TICKERS)
    VAR_NAMES = list(TICKERS.values())
    market = rng.normal(0.0, 0.8, size=(T, 1)).astype(np.float64)
    idio = rng.normal(0.0, 0.6, size=(T, d)).astype(np.float64)
    loading = np.linspace(0.5, 1.1, d, dtype=np.float64)[None, :]
    data_np = (market @ loading + idio).astype(np.float64)
else:
    returns = np.log(close / close.shift(1)).dropna()
    returns.columns = [TICKERS[t] for t in returns.columns if t in TICKERS]
    data_np = returns.values.astype(np.float64)
    T, d = data_np.shape
    VAR_NAMES = returns.columns.tolist()

print(f"Shape: {data_np.shape}")
print(f"Variables ({d}): {VAR_NAMES}")
'''


def set_source(cell: dict, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    cell["outputs"] = []
    cell["execution_count"] = None


def update_notebook(path: Path, code_map: Dict[int, str]) -> None:
    nb = json.loads(path.read_text())
    for idx, source in code_map.items():
        set_source(nb["cells"][idx], source)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Updated {path.name} ({len(code_map)} code cells)")


# ------------------------------------------------------------------ #
# 4.1 Neural Granger
# ------------------------------------------------------------------ #
NB_01 = {
    4: SETUP_INSTALL,
    5: IMPORTS_BASE + "\n" + SEED_CELL,
    7: DATA_LOAD,
    9: '''LAG = 5
EPOCHS = 60 if run_fast else 120
BATCH_SIZE = 32
LR = 5e-4
LAM = 0.005
HIDDEN = 32
DEVICE = "cpu"

print(f"LAG={LAG}, EPOCHS={EPOCHS}, device={DEVICE}")
''',
    11: '''from pydeepcausalml import (
    NeuralGrangerCMLP,
    NeuralGrangerCLSTM,
    neural_granger_model,
)

# cMLP and cLSTM live in pydeepcausalml.timeseries (group-sparse Granger models).
print("Neural Granger models imported from pydeepcausalml.")
''',
    13: '''from pydeepcausalml import NeuralGrangerEconomySRU, NeuralRelationalInference

# EconomySRU and NRI are extended neural Granger estimators in the same module family.
print("Extended neural Granger models ready.")
''',
    15: '''def fit_granger(method: str):
    """Fit a neural Granger estimator via pydeepcausalml."""
    common = dict(
        lag=LAG, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        device=DEVICE, random_state=42,
    )
    if method == "cmlp":
        est = neural_granger_model(
            method, hidden_dim=HIDDEN, lambda_group=LAM, **common,
        )
    elif method in ("clstm", "economysru"):
        est = neural_granger_model(
            method, hidden=HIDDEN, lambda_sparse=LAM, **common,
        )
    else:
        est = neural_granger_model(method, hidden=HIDDEN, **common)
    est.fit(data_np)
    return est
''',
    16: '''models = {}
histories = {}

for name, method in [
    ("cMLP", "cmlp"),
    ("cLSTM", "clstm"),
    ("ECONOMY-SRU", "economysru"),
    ("NRI", "nri"),
]:
    print(f"Training {name} ...")
    est = fit_granger(method)
    models[name] = est
    histories[name] = est.history_
    print(f"  done — final loss={est.history_['loss'][-1]:.4f}")
''',
    17: '''print("\\n" + "=" * 45)
print(f"{'Model':<15} {'Final loss':>12}")
print("=" * 45)
for name, hist in sorted(histories.items(), key=lambda x: x[1]["loss"][-1]):
    print(f"{name:<15} {hist['loss'][-1]:>12.6f}")
print("=" * 45)
''',
    18: '''# Training curves
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, name in zip(axes, ["cMLP", "cLSTM", "ECONOMY-SRU", "NRI"]):
    hist = histories[name]
    ax.plot(hist["loss"], lw=2)
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Loss")
plt.tight_layout()
plt.show()

# Causal matrices
def _causal_matrix(est):
    if hasattr(est, "get_scores"):
        return est.get_scores()
    return est.adjacency_matrix()

causal_mats = {name: _causal_matrix(est) for name, est in models.items()}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, mat) in zip(axes.flatten(), causal_mats.items()):
    sns.heatmap(
        mat, ax=ax, cmap="magma",
        xticklabels=VAR_NAMES, yticklabels=VAR_NAMES, cbar=True,
    )
    ax.set_title(f"{name} inferred causal strengths")
    ax.set_xlabel("Source variable")
    ax.set_ylabel("Target variable")
plt.tight_layout()
plt.show()
''',
}

# ------------------------------------------------------------------ #
# 4.2 Structural Causal Models
# ------------------------------------------------------------------ #
NB_02 = {
    5: SETUP_INSTALL + "\n\n" + IMPORTS_BASE + "\n" + SEED_CELL + '''

import networkx as nx
from pydeepcausalml import DeepSCM, DECI, DynoTEARS
from pydeepcausalml.metrics import graph_recovery_metrics
''',
    9: DATA_LOAD_ROBUST + '''
LAG = 1
EPOCHS_SCM = 25 if run_fast else 50
EPOCHS_DECI = 40 if run_fast else 80
EPOCHS_DYNO = 80 if run_fast else 200
DEVICE = "cpu"
''',
    12: '''def plot_dag(adj, names, title, threshold=0.0):
    """Plot a weighted adjacency matrix as a network."""
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for i, tgt in enumerate(names):
        for j, src in enumerate(names):
            w = adj[i, j]
            if w > threshold:
                G.add_edge(src, tgt, weight=float(w))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_size=900, font_size=8, arrows=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def threshold_adjacency(w, threshold=0.35):
    a = (w >= threshold).astype(float)
    np.fill_diagonal(a, 0)
    return a

print("Plotting utilities ready.")
''',
    15: '''# Train DeepSCM (contemporaneous structural equations)
deep_scm = DeepSCM(
    lag=LAG, hidden=64,
    epochs=EPOCHS_SCM, batch_size=64, lr=1e-3,
    device=DEVICE, random_state=42,
)
deep_scm.fit(data_np)
print(f"[DeepSCM] final loss={deep_scm.history_['loss'][-1]:.4f}")

corr = np.abs(np.corrcoef(data_np.T))
np.fill_diagonal(corr, 0.0)
A_fixed = (corr > 0.25).astype(float)
plot_dag(A_fixed, VAR_NAMES, "Correlation heuristic (reference graph)")
''',
    18: '''# Train DECI (joint graph + structural equations)
deci = DECI(
    lag=LAG, hidden=64, lambda_dag=0.1,
    epochs=EPOCHS_DECI, batch_size=64, lr=1e-3,
    device=DEVICE, random_state=42,
)
deci.fit(data_np)
print(f"[DECI] final loss={deci.history_['loss'][-1]:.4f}")

A_soft = deci.adjacency_matrix()
A_bin = threshold_adjacency(A_soft, threshold=0.35)
plot_dag(A_soft, VAR_NAMES, "DECI Soft Adjacency")
plot_dag(A_bin, VAR_NAMES, "DECI Thresholded Adjacency")
''',
    21: '''# Intervention + ATE example
source_name, target_name = "Energy", "Industrials"
source_idx = VAR_NAMES.index(source_name)
target_idx = VAR_NAMES.index(target_name)

x_contemp = data_np[:, :]  # contemporaneous slice for intervention demo
ate = deci.predict_ate(x_contemp, intervention_var=source_idx, n_samples=200)
print(f"Estimated ATE of do({source_name}) on {target_name}: {ate:.4f}")

pred_lo = deep_scm.intervene(x_contemp[:128], var_idx=source_idx, value=-1.0)
pred_hi = deep_scm.intervene(x_contemp[:128], var_idx=source_idx, value=1.0)
delta = (pred_hi[:, target_idx] - pred_lo[:, target_idx]).mean()
print(
    f"Explanation:\\n"
    f"Estimated ATE of do({source_name}) on {target_name}: {ate:.4f}\\n"
    f"DeepSCM intervention delta on {target_name}: {delta:.4f}"
)
''',
    23: '''# DynoTEARS — lag-resolved DAG discovery (pydeepcausalml.discovery)
dyno = DynoTEARS(
    lag=5, lambda_l1=0.02,
    epochs=EPOCHS_DYNO, batch_size=256, lr=3e-3,
    device=DEVICE, random_state=42,
)
dyno.fit(data_np)
print(f"[DynoTEARS] final loss={dyno.history_['loss'][-1]:.4f}")

A_dyno = dyno.get_adjacency(threshold=0.05)
W_agg = dyno.get_scores()
plot_dag(W_agg, VAR_NAMES, "DynoTEARS aggregated lag weights")
plot_dag(A_dyno, VAR_NAMES, "DynoTEARS thresholded adjacency")
''',
    25: '''# Graph comparison across methods available in this notebook
methods = {
    "DECI": deci.adjacency_matrix(),
    "DynoTEARS": dyno.get_adjacency(threshold=0.05),
}

fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
if len(methods) == 1:
    axes = [axes]
for ax, (name, mat) in zip(axes, methods.items()):
    sns.heatmap(mat, ax=ax, cmap="Blues", xticklabels=VAR_NAMES, yticklabels=VAR_NAMES)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# Training dynamics
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for ax, (label, hist) in zip(
    axes,
    [("DeepSCM", deep_scm.history_), ("DECI", deci.history_), ("DynoTEARS", dyno.history_)],
):
    ax.plot(hist["loss"])
    ax.set_title(label)
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
plt.tight_layout()
plt.show()
''',
}

# ------------------------------------------------------------------ #
# 4.3 Attention / Transformer
# ------------------------------------------------------------------ #
NB_03 = {
    8: SETUP_INSTALL + "\n\n" + IMPORTS_BASE + "\n" + SEED_CELL,
    10: DATA_LOAD + '''
LAG = 10
EPOCHS = 40 if run_fast else 80
HIDDEN = 32
DEVICE = "cpu"
''',
    12: '''from pydeepcausalml import TCDF, CausalTransformer, TFTNet, attn_causal_model

print("Attention-based causal models imported from pydeepcausalml.")
''',
    14: '''# Model architectures are provided by pydeepcausalml.timeseries:
#   TCDF            — dilated depthwise conv + attention (Nauta et al., 2019)
#   CausalTransformer — causal-masked transformer encoder
#   TFTNet          — Temporal Fusion Transformer with variable selection
print("See pydeepcausalml.timeseries.attention_models for implementation details.")
''',
    16: '''models = {}
for name, method, extra in [
    ("TCDF", "tcdf", dict(kernel_size=3, hidden_layers=2, epochs=EPOCHS // 2)),
    ("CausalTransformer", "causal_transformer", dict(d_model=HIDDEN, nhead=4, n_layers=2)),
    ("TFT", "tft", dict(hidden=HIDDEN)),
]:
    print(f"Training {name} ...")
    est = attn_causal_model(
        method,
        lag=LAG,
        epochs=extra.pop("epochs", EPOCHS),
        batch_size=64,
        lr=1e-3,
        device=DEVICE,
        random_state=42,
        **extra,
    )
    est.fit(data_np)
    models[name] = est
    print(f"  done — loss={est.history_['loss'][-1]:.4f}")
''',
    18: '''# Causal matrix visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, est) in zip(axes, models.items()):
    if hasattr(est, "causal_matrix"):
        mat = est.causal_matrix()
    elif hasattr(est, "get_adjacency"):
        mat = est.get_adjacency()
    else:
        mat = est.get_scores()
    sns.heatmap(mat, ax=ax, cmap="viridis", xticklabels=VAR_NAMES, yticklabels=VAR_NAMES)
    ax.set_title(name)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for ax, (name, est) in zip(axes, models.items()):
    ax.plot(est.history_["loss"])
    ax.set_title(f"{name} training loss")
    ax.set_xlabel("Epoch")
plt.tight_layout()
plt.show()
''',
}

# ------------------------------------------------------------------ #
# 4.4 RNN / LSTM
# ------------------------------------------------------------------ #
NB_04 = {
    6: SETUP_INSTALL + "\n\n" + IMPORTS_BASE + "\n" + SEED_CELL,
    8: DATA_LOAD + '''
LAG = 10
EPOCHS = 50 if run_fast else 100
HIDDEN = 32
DEVICE = "cpu"
''',
    10: '''from pydeepcausalml import CausalLSTM, RETAIN, InterventionAwareRNN, rnn_causal_model

print("RNN-based causal models imported from pydeepcausalml.")
''',
    12: '''# CausalLSTM — stacked LSTM with learnable soft adjacency mask
print("CausalLSTM: pydeepcausalml.timeseries.rnn_models.CausalLSTM")
''',
    14: '''# RETAIN — reverse-time GRU with temporal and variable attention
print("RETAIN: pydeepcausalml.timeseries.rnn_models.RETAIN")
''',
    16: '''# InterventionAwareRNN — regime-aware LSTM with intervention channel
print("InterventionAwareRNN: pydeepcausalml.timeseries.rnn_models.InterventionAwareRNN")
''',
    18: '''models = {}
for name, method in [
    ("CausalLSTM", "causal_lstm"),
    ("RETAIN", "retain"),
    ("InterventionAwareRNN", "intervention_rnn"),
]:
    print(f"Training {name} ...")
    est = rnn_causal_model(
        method,
        lag=LAG, hidden=HIDDEN,
        epochs=EPOCHS, batch_size=64, lr=1e-3,
        device=DEVICE, random_state=42,
    )
    est.fit(data_np)
    models[name] = est
    print(f"  done — loss={est.history_['loss'][-1]:.4f}")
''',
    20: '''def plot_causal_heatmap(mat, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, cmap="coolwarm", xticklabels=VAR_NAMES, yticklabels=VAR_NAMES)
    plt.title(title)
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.show()
''',
    21: '''for name, est in models.items():
    mat = est.causal_matrix()
    plot_causal_heatmap(mat, f"{name} learned causal structure")

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for ax, (name, est) in zip(axes, models.items()):
    ax.plot(est.history_["loss"])
    ax.set_title(name)
    ax.set_xlabel("Epoch")
plt.tight_layout()
plt.show()
''',
}

# ------------------------------------------------------------------ #
# 4.5 GNN
# ------------------------------------------------------------------ #
NB_05 = {
    5: SETUP_INSTALL + "\n\n" + IMPORTS_BASE + "\n" + SEED_CELL + '''
import networkx as nx
''',
    8: DATA_LOAD + '''
LAG = 5
EPOCHS = 50 if run_fast else 100
HIDDEN = 32
DEVICE = "cpu"
''',
    11: '''from pydeepcausalml import GVAR, CausalGNN, CUTS, gnn_causal_model

def plot_graph(adj, names, title):
    G = nx.DiGraph()
    for i, tgt in enumerate(names):
        for j, src in enumerate(names):
            if adj[i, j] > 0.05:
                G.add_edge(src, tgt, weight=float(adj[i, j]))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_size=800, font_size=8, arrows=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
''',
    15: '''# GVAR — Graph Vector Autoregression with lag-specific soft adjacency
print("GVAR: pydeepcausalml.timeseries.gnn_models.GVAR")
''',
    19: '''# CausalGNN — bilinear graph learner with GRU encoder
print("CausalGNN: pydeepcausalml.timeseries.gnn_models.CausalGNN")
''',
    23: '''# CUTS+ — variational Bernoulli graph posterior for irregular series
print("CUTS: pydeepcausalml.timeseries.gnn_models.CUTS")
''',
    27: '''model_gvar = gnn_causal_model(
    "gvar", lag=LAG, hidden=HIDDEN, lambda_dag=0.1,
    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE, random_state=42,
)
model_cgnn = gnn_causal_model(
    "causal_gnn", lag=LAG, hidden=HIDDEN,
    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE, random_state=42,
)
model_cuts = gnn_causal_model(
    "cuts", lag=LAG, hidden=HIDDEN,
    epochs=EPOCHS, batch_size=64, lr=1e-3, device=DEVICE, random_state=42,
)

for label, est in [("GVAR", model_gvar), ("CausalGNN", model_cgnn), ("CUTS", model_cuts)]:
    print(f"Training {label} ...")
    est.fit(data_np)
    print(f"  done — loss={est.history_['loss'][-1]:.4f}")
''',
    31: '''C_gvar = model_gvar.causal_matrix()
C_cgnn = model_cgnn.causal_matrix()
C_cuts = model_cuts.causal_matrix()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, mat) in zip(axes, [("GVAR", C_gvar), ("CausalGNN", C_cgnn), ("CUTS", C_cuts)]):
    sns.heatmap(mat, ax=ax, cmap="magma", xticklabels=VAR_NAMES, yticklabels=VAR_NAMES)
    ax.set_title(f"{name} causal matrix")
plt.tight_layout()
plt.show()

plot_graph(C_gvar, VAR_NAMES, "GVAR learned graph")
plot_graph(C_cgnn, VAR_NAMES, "CausalGNN learned graph")
plot_graph(C_cuts, VAR_NAMES, "CUTS learned graph")
''',
}

# ------------------------------------------------------------------ #
# 4.6 Counterfactual
# ------------------------------------------------------------------ #
NB_06 = {
    7: SETUP_INSTALL,
    9: IMPORTS_BASE + "\n" + SEED_CELL + '''
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from pydeepcausalml import DeepSynth, CRN, GNet, counterfactual_model
''',
    11: DATA_LOAD + '''
# Treatment: Technology shock (top quartile daily return)
tech_idx = VAR_NAMES.index("Technology")
fin_idx = VAR_NAMES.index("Financials")
tech_ret = data_np[:, tech_idx]
threshold = np.quantile(tech_ret, 0.75)
treatment = (tech_ret >= threshold).astype(np.float64)
outcome = data_np[:, fin_idx]

print(f"Treatment rate (Technology shock): {treatment.mean():.3f}")
''',
    13: '''LAG = 20
EPOCHS = 40 if run_fast else 80
DEVICE = "cpu"
BATCH_SIZE = 64
LR = 3e-4

print(f"LAG={LAG}, outcome=Financials, treatment=Technology shock")
''',
    15: '''# PyDeepCausalML estimators build lagged sequences internally via make_lagged_sequences.
print("DataLoaders handled inside each estimator's fit() method.")
''',
    17: '''# Training utilities — each estimator records loss in history_
def final_loss(est):
    return est.history_["loss"][-1]
''',
    19: '''# DeepSynth — neural synthetic control (pydeepcausalml.timeseries.counterfactual)
print("DeepSynth imported from pydeepcausalml.")
''',
    21: '''# CRN — Counterfactual Recurrent Network with adversarial balancing
print("CRN imported from pydeepcausalml.")
''',
    23: '''# G-Net — deep G-computation
print("GNet imported from pydeepcausalml.")
''',
    25: '''deep_synth = counterfactual_model(
    "deepsynth", lag=LAG, hidden=32,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE, random_state=42,
)
crn = counterfactual_model(
    "crn", lag=LAG, hidden=32, lambda_adv=0.5,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE, random_state=42,
)
gnet = counterfactual_model(
    "gnet", lag=LAG, hidden=32,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE, random_state=42,
)

deep_synth.fit(data_np, outcome)
crn.fit(data_np, treatment, outcome)
gnet.fit(data_np, treatment, outcome)

print(f"DeepSynth final loss: {final_loss(deep_synth):.4f}")
print(f"CRN final loss:       {final_loss(crn):.4f}")
print(f"GNet final loss:      {final_loss(gnet):.4f}")
''',
    27: '''metrics = {
    "DeepSynth": final_loss(deep_synth),
    "CRN": final_loss(crn),
    "GNet": final_loss(gnet),
}
print("\\nValidation training loss (lower is better):")
for name, val in sorted(metrics.items(), key=lambda x: x[1]):
    print(f"  {name:<12} {val:.6f}")
''',
    29: '''plt.figure(figsize=(10, 4))
for name, est in [("DeepSynth", deep_synth), ("CRN", crn), ("GNet", gnet)]:
    plt.plot(est.history_["loss"], label=name, lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
''',
    31: '''ite_crn = crn.predict_ite(data_np, treatment)
ite_gnet = gnet.predict_ite(data_np)
cf_ds = deep_synth.predict_counterfactual(data_np)

ate_crn = float(np.mean(ite_crn[LAG:]))
ate_gnet = float(np.mean(ite_gnet[LAG:]))
ate_ds = float(np.mean(cf_ds) - np.mean(outcome[LAG:]))

print(f"ATE (DeepSynth proxy): {ate_ds:.4f}")
print(f"ATE (CRN):             {ate_crn:.4f}")
print(f"ATE (GNet):            {ate_gnet:.4f}")
''',
    33: '''fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(ite_crn[LAG:], bins=30, alpha=0.7, label="CRN")
axes[0].hist(ite_gnet[LAG:], bins=30, alpha=0.7, label="GNet")
axes[0].set_title("ITE distributions")
axes[0].set_xlabel("ITE")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(cf_ds[:200], label="DeepSynth counterfactual", lw=1.5)
axes[1].plot(outcome[LAG:LAG + 200], label="Observed outcome", lw=1.5, alpha=0.7)
axes[1].set_title("DeepSynth counterfactual vs observed")
axes[1].set_xlabel("Time index")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
''',
}


NOTEBOOKS = [
    (ROOT / "Tutorial/02_08_05_05_04_01_DeepCausalML_timeseries_neural_granger_causality.ipynb", NB_01),
    (ROOT / "Tutorial/02_08_05_05_04_02_DeepCausalML_timeseries_structural_causal_model_SMC.ipynb", NB_02),
    (ROOT / "Tutorial/02_08_05_05_04_03_DeepCausalML_attention_transformer.ipynb", NB_03),
    (ROOT / "Tutorial/02_08_05_05_04_04_DeepCausalML_timeseries_RNN_LSTM_causalML.ipynb", NB_04),
    (ROOT / "Tutorial/02_08_05_05_04_05_DeepCausalML_timeseries_graphNN.ipynb", NB_05),
    (ROOT / "Tutorial/02_08_05_05_04_06_DeepCausalML_timeseries_counterfactual_potential.ipynb", NB_06),
]


def main() -> None:
    for path, code_map in NOTEBOOKS:
        update_notebook(path, code_map)
    print("All notebooks updated.")


if __name__ == "__main__":
    main()
