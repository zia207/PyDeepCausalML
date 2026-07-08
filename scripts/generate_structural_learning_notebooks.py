#!/usr/bin/env python3
"""Generate structural-learning tutorial notebooks from R qmd content."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "Tutorial"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


def nb(cells: list, title: str = "") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = code(
    """import importlib
import subprocess
import sys

PACKAGES = [
    "numpy", "pandas", "scipy", "torch", "scikit-learn",
    "matplotlib", "seaborn", "networkx",
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

import pydeepcausalml
print("pydeepcausalml", pydeepcausalml.__version__, "ready.")"""
)

IMPORTS_CELL = code(
    """import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LinearRegression

from pydeepcausalml import set_seed
from pydeepcausalml.metrics import graph_recovery_metrics, shd, pehe

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())"""
)

SEED_CELL = code("set_seed(42)\nrun_fast = True")

SIM_HELPERS = code(
    '''"""Simulation and evaluation helpers (NOTEARS tutorial utilities)."""

def is_dag(W: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if weighted matrix W encodes a DAG."""
    d = W.shape[0]
    m = (np.abs(W) > tol).astype(float)
    acc = m.copy()
    for _ in range(d):
        if np.trace(acc) > tol:
            return False
        acc = acc @ m
    return True


def simulate_dag(d: int, s0: int, graph_type: str = "ER", seed: int | None = None) -> np.ndarray:
    """Simulate a random binary DAG adjacency (A[i,j]=1 => X_j -> X_i)."""
    rng = np.random.default_rng(seed)
    if graph_type.upper() == "ER":
        prob = min(1.0, s0 / max(d * (d - 1) / 2, 1))
        B = np.triu(rng.random((d, d)) < prob, k=1).astype(int)
    else:  # scale-free-ish upper triangular
        B = np.zeros((d, d), dtype=int)
        edges = 0
        while edges < s0:
            i, j = rng.integers(0, d, size=2)
            if i < j and B[i, j] == 0:
                B[i, j] = 1
                edges += 1
    return B


def simulate_parameter(B: np.ndarray, low: float = 0.5, high: float = 2.0, seed: int | None = None) -> np.ndarray:
    """Draw edge weights for a binary DAG."""
    rng = np.random.default_rng(seed)
    W = np.zeros_like(B, dtype=float)
    mask = B.astype(bool)
    signs = rng.choice([-1.0, 1.0], size=mask.sum())
    W[mask] = signs * rng.uniform(low, high, size=mask.sum())
    return W


def simulate_linear_sem(W: np.ndarray, n: int, sem_type: str = "gauss", seed: int | None = None) -> np.ndarray:
    """Generate data from linear SEM X = X @ W.T + Z (package convention A[i,j]: j->i)."""
    rng = np.random.default_rng(seed)
    d = W.shape[0]
    X = np.zeros((n, d))
    order = list(np.argsort(-W.sum(axis=1)))  # rough topological order
    for _ in range(d):
        Z = rng.standard_normal((n, d)) if sem_type == "gauss" else rng.exponential(size=(n, d))
        X = X @ W.T + Z
    return X


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """Compare estimated graph to ground truth (FDR, TPR, FPR, SHD, nnz)."""
    m = graph_recovery_metrics(B_true, B_est)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    tn = int((~np.eye(B_true.shape[0], dtype=bool)).sum()) - tp - fp - fn
    fdr = fp / (tp + fp) if tp + fp else 0.0
    tpr = m["recall"]
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": m["shd"], "nnz": int(B_est.sum())}


def simulate_nonlinear_sem_custom(W: np.ndarray, n: int, sem_type: str = "mlp", noise_sd: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Nonlinear SEM with additive noise (A[i,j]: j->i)."""
    rng = np.random.default_rng(seed)
    d = W.shape[0]
    G = nx.DiGraph((j, i) for i in range(d) for j in range(d) if W[i, j] != 0)
    order = list(nx.topological_sort(G)) if G.number_of_edges() else list(range(d))
    X = np.zeros((n, d))
    for i in order:
        pa = np.where(W[i, :] != 0)[0]
        if len(pa) == 0:
            X[:, i] = rng.standard_normal(n)
        else:
            z = X[:, pa] @ W[i, pa]
            if sem_type == "mlp":
                X[:, i] = np.tanh(z) + rng.normal(0, noise_sd, n)
            elif sem_type == "mim":
                X[:, i] = z * np.sin(z) + rng.normal(0, noise_sd, n)
            elif sem_type == "gp":
                X[:, i] = np.sin(z) + rng.normal(0, noise_sd, n)
            else:
                X[:, i] = z + rng.standard_normal(n)
    return X'''
)


def build_notears() -> dict:
    cells = [
        md("![Banner](../Image/03_DeepCausalML.png)"),
        md(
            """# 3.1 Continuous Optimization Models (NOTEARS)

> **Note:** NOTEARS requires **PyTorch**. The `NOTEARSLinear`, `NOTEARSNonlinearMLP`, and `NOTEARSNonlinearSobolev` estimators in `pydeepcausalml.discovery` learn DAG structure via a smooth acyclicity constraint.

**NOTEARS** ("DAGs with NO TEARS", Zheng et al., NeurIPS 2018) introduced a landmark reformulation of causal structure learning: instead of exhaustively searching the combinatorial space of $2^{O(d^2)}$ possible directed acyclic graphs (an approach that becomes intractable for more than roughly ten variables), it recasts the problem as a **smooth, gradient-based optimization** over a real-valued weighted adjacency matrix $\\mathbf{W} \\in \\mathbb{R}^{d \\times d}$.

Under the **linear Structural Equation Model (SEM)** assumption

$$\\mathbf{X} = \\mathbf{X}\\mathbf{W} + \\mathbf{Z},$$

where $\\mathbf{Z}$ contains independent noise terms (commonly Gaussian), the algorithm simultaneously (i) fits $\\mathbf{W}$ to the data via a reconstruction loss, (ii) enforces acyclicity through a closed-form differentiable constraint, and (iii) promotes sparsity with an $\\ell_1$ penalty.

**Key innovation — the acyclicity constraint.** The function

$$h(\\mathbf{W}) = \\mathrm{tr}\\!\\left(\\exp(\\mathbf{W} \\odot \\mathbf{W})\\right) - d$$

equals zero *if and only if* the graph encoded by $\\mathbf{W}$ is acyclic. Because $h$ is smooth and differentiable, the NP-hard combinatorial constraint is replaced by an equality constraint amenable to augmented Lagrangian (AL) optimization — no combinatorial search required.

NOTEARS is concise (the core algorithm fits in fewer than 60 lines of code), fast for moderate graph sizes ($d \\leq 100$–$200$), and has become the canonical baseline against which all later continuous causal discovery methods are benchmarked — including GOLEM, DAG-GNN, and GraN-DAG.

![](../Image/NOTEARS.png)"""
        ),
        md(
            """## NOTEARS with PyDeepCausalML

This notebook implements both the **linear** and **nonlinear** NOTEARS variants in Python using the **PyDeepCausalML** package. We apply them to synthetic data (linear and nonlinear SEMs) and evaluate performance against known ground-truth DAGs.

The **PyDeepCausalML** package encodes the full NOTEARS family. The acyclicity constraint $h(W) = \\mathrm{tr}(e^{W \\odot W}) - d = 0$ is enforced via the trace of the matrix exponential, and optimization is handled by the augmented Lagrangian method with Adam inner steps.

**Linear NOTEARS** (`NOTEARSLinear`) learns a weighted adjacency matrix $W$ by minimizing a least-squares loss plus an $\\ell_1$ penalty subject to $h(W) = 0$.

**Nonlinear NOTEARS** (`NOTEARSNonlinearMLP`, `NOTEARSNonlinearSobolev`) replace the linear predictor with either an MLP or a Sobolev-basis model, allowing the algorithm to capture arbitrary smooth nonlinear relationships while keeping the same acyclicity constraint.

### Main Classes

| Class | Role |
|------------------------------------|------------------------------------|
| `NOTEARSLinear` | Linear NOTEARS: L2 loss + L1, augmented Lagrangian |
| `NOTEARSNonlinearMLP` | Nonlinear NOTEARS with MLP per node |
| `NOTEARSNonlinearSobolev` | Nonlinear NOTEARS with Sobolev / polynomial basis |

### Simulation and Evaluation Utilities

| Function | Role |
|------------------------------------|------------------------------------|
| `is_dag` | Check whether a weighted adjacency matrix encodes a DAG |
| `simulate_dag` | Simulate a random DAG (Erdős–Rényi or scale-free) |
| `simulate_parameter` | Draw edge weights for a DAG |
| `simulate_linear_sem` | Generate data from a linear SEM |
| `simulate_nonlinear_sem_custom` | Generate data from a nonlinear SEM |
| `count_accuracy` / `graph_recovery_metrics` | Compare estimated graph to ground truth |

**Reference:** Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS*. <https://arxiv.org/abs/1803.01422>."""
        ),
        md("## Setup\n\n### Check and Install Required Python Packages"),
        SETUP_CELL,
        md("### Verify imports"),
        IMPORTS_CELL,
        SEED_CELL,
        md("### Simulation utilities"),
        SIM_HELPERS,
        md(
            """---

## Part I: Linear NOTEARS on Synthetic Gaussian Data

### Overview

We begin with a **linear Gaussian SEM**: variables are generated by the model $\\mathbf{X} = \\mathbf{X}\\mathbf{W} + \\mathbf{Z}$, where $\\mathbf{Z} \\sim \\mathcal{N}(0, I)$. The ground-truth DAG is known, so we can measure recovery accuracy precisely.

### Data Generation and Preprocessing"""
        ),
        code(
            """# Dataset 1: Linear Gaussian SEM (d = 20)
set_seed(42)

n_lin, d_lin, s0_lin = 2000, 20, 20
graph_type, sem_type_lin = "ER", "gauss"

B_true_lin = simulate_dag(d_lin, s0_lin, graph_type, seed=42)
W_true_lin = simulate_parameter(B_true_lin, seed=42)
X_lin = simulate_linear_sem(W_true_lin, n_lin, sem_type_lin, seed=42)

# Center columns — NOTEARS is sensitive to variable scale
X_lin = X_lin - X_lin.mean(axis=0, keepdims=True)

rng = np.random.default_rng(42)
idx = rng.permutation(n_lin)
tr1 = idx[: int(0.70 * n_lin)]
tmp1 = idx[int(0.70 * n_lin) :]
va1 = tmp1[: len(tmp1) // 2]
te1 = tmp1[len(tmp1) // 2 :]

X_lin_train, X_lin_val, X_lin_test = X_lin[tr1], X_lin[va1], X_lin[te1]
print("Linear SEM — Train:", X_lin_train.shape, "| Val:", X_lin_val.shape, "| Test:", X_lin_test.shape)"""
        ),
        md(
            """### Fitting Linear NOTEARS

The algorithm minimizes the augmented Lagrangian

$$\\mathcal{L}_\\rho(W, \\alpha) = \\frac{1}{2n}\\|X - XW\\|_F^2 + \\lambda_1\\|W\\|_1 + \\alpha\\, h(W) + \\frac{\\rho}{2}h(W)^2,$$

iterating (i) gradient steps on $W$ and (ii) dual-ascent updates until $h(W) < \\epsilon$."""
        ),
        code(
            """from pydeepcausalml.discovery import NOTEARSLinear

model_lin = NOTEARSLinear(
    lambda_l1=0.1,
    h_tol=1e-8,
    rho_max=1e16,
    epochs=100,
    n_outer=100,
    lr=1e-2,
    random_state=42,
)
model_lin.fit(X_lin_train)

W_est_lin = model_lin.weights_
w_threshold = 0.3
B_est_lin = model_lin.get_adjacency(threshold=w_threshold)

print("Acyclic (linear SEM):", is_dag(W_est_lin))
n_val_lin = X_lin_val.shape[0]
val_loss_lin = 0.5 / n_val_lin * np.sum((X_lin_val - X_lin_val @ W_est_lin.T) ** 2)
print(f"Validation L2 loss (linear model | linear SEM): {val_loss_lin:.6f}")"""
        ),
        md(
            """**Hyperparameter guidance.** The two most impactful choices are: (i) the sparsity penalty $\\lambda_1 \\in [0.01, 0.5]$ — higher values prune more edges; and (ii) the edge-threshold $w_{\\mathrm{threshold}} \\in [0.1, 0.5]$ applied after optimization.

### Causal Effect Estimation via Backdoor Adjustment

NOTEARS recovers **graph structure**, not causal effects directly. Once the DAG is in hand, downstream causal inference proceeds in two steps: (i) read off a valid adjustment set from the graph using the backdoor criterion, and (ii) estimate the Average Treatment Effect (ATE) by regression on that set.

The example below treats variable 0 as the treatment $T$ and variable 19 as the outcome $Y$ (0-indexed)."""
        ),
        code(
            """# True ATE via structural path sum: (I - W)^{-1}[Y, T]
I_lin = np.eye(d_lin)
true_total_lin = np.linalg.solve(I_lin - W_true_lin, np.eye(d_lin))
true_ATE_lin = true_total_lin[19, 0]
print(f"True ATE (T -> Y, linear SEM):  {true_ATE_lin:.4f}")

confounders_idx_lin = [i for i in range(d_lin) if i not in (0, 19)]
T_tr_lin = X_lin_train[:, 0:1]
Y_tr_lin = X_lin_train[:, 19]
C_tr_lin = X_lin_train[:, confounders_idx_lin]

model_ate_lin = LinearRegression().fit(np.hstack([T_tr_lin, C_tr_lin]), Y_tr_lin)
est_ATE_lin = model_ate_lin.coef_[0]
print(f"Estimated ATE:  {est_ATE_lin:.4f}  (error: {abs(est_ATE_lin - true_ATE_lin):.4f})")"""
        ),
        md(
            """### Graph Recovery Metrics

When the ground-truth DAG is known, five complementary metrics quantify structural recovery accuracy.

- **FDR** — fraction of predicted edges that are spurious.
- **TPR (Recall)** — fraction of true edges recovered.
- **FPR** — fraction of true non-edges incorrectly predicted.
- **SHD** — edge insertions + deletions to match the true graph.
- **nnz** — number of edges in the estimated graph."""
        ),
        code(
            """acc_lin_result = count_accuracy(B_true_lin, B_est_lin)
print(acc_lin_result)"""
        ),
        md("### Visualization\n\n#### Weighted Adjacency Matrix"),
        code(
            """fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(W_est_lin, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_title("Learned Weighted Adjacency Matrix — Linear SEM (NOTEARS Linear)")
ax.set_xlabel("Cause (source)")
ax.set_ylabel("Effect (destination)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()"""
        ),
        md("#### Learned Causal DAG"),
        code(
            """G_lin = nx.from_numpy_array((np.abs(W_est_lin) > w_threshold).astype(int), create_using=nx.DiGraph)
pos_lin = nx.spring_layout(G_lin, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G_lin, pos_lin, with_labels=True, node_color="lightblue", arrows=True,
        node_size=600, font_size=8, edge_color="gray")
plt.title("Learned Causal DAG — Linear Gaussian SEM (NOTEARS Linear)")
plt.show()"""
        ),
        md(
            """---

## Part II: Nonlinear NOTEARS (MLP and Sobolev) vs. Misspecified Baseline

### Overview

We now stress-test the linear model on data generated from a **nonlinear SEM** and compare it against two nonlinear NOTEARS variants."""
        ),
        code(
            """# Dataset 2: Nonlinear SEM (d = 10)
set_seed(42)
n_nlin, d_nlin, s0_nlin = 2000, 10, 10
B_true_nlin = simulate_dag(d_nlin, s0_nlin, graph_type, seed=42)
W_true_nlin = simulate_parameter(B_true_nlin, seed=42)
X_nlin = simulate_nonlinear_sem_custom(W_true_nlin, n_nlin, sem_type="mlp", seed=42)
X_nlin = X_nlin - X_nlin.mean(axis=0, keepdims=True)

rng = np.random.default_rng(42)
idx = rng.permutation(n_nlin)
tr2 = idx[: int(0.70 * n_nlin)]
tmp2 = idx[int(0.70 * n_nlin) :]
va2 = tmp2[: len(tmp2) // 2]
te2 = tmp2[len(tmp2) // 2 :]
X_nlin_train, X_nlin_val, X_nlin_test = X_nlin[tr2], X_nlin[va2], X_nlin[te2]
print("Nonlinear SEM — Train:", X_nlin_train.shape, "| Val:", X_nlin_val.shape, "| Test:", X_nlin_test.shape)"""
        ),
        md("### Model 1: Linear NOTEARS (Misspecified Baseline)"),
        code(
            """model_lin_nl = NOTEARSLinear(lambda_l1=0.1, epochs=100, n_outer=100, random_state=42)
model_lin_nl.fit(X_nlin_train)
W_lin_nlin = model_lin_nl.weights_
B_lin_nlin = model_lin_nl.get_adjacency(threshold=0.3)
print("Linear NOTEARS on nonlinear SEM — acyclic:", is_dag(W_lin_nlin))
n_val_nlin = X_nlin_val.shape[0]
val_loss_lin_nl = 0.5 / n_val_nlin * np.sum((X_nlin_val - X_nlin_val @ W_lin_nlin.T) ** 2)
print(f"Validation L2 loss (linear model | nonlinear SEM): {val_loss_lin_nl:.6f}")"""
        ),
        md("### Model 2: NOTEARSNonlinearMLP"),
        code(
            """from pydeepcausalml.discovery import NOTEARSNonlinearMLP, NOTEARSNonlinearSobolev

set_seed(42)
model_mlp = NOTEARSNonlinearMLP(hidden=10, lambda1=0.01, rho=1.0, epochs=150, lr=1e-2, random_state=42)
model_mlp.fit(X_nlin_train)
W_mlp_nlin = model_mlp.adjacency_matrix()
B_mlp_nlin = (W_mlp_nlin > 1e-5).astype(int)
print("NOTEARSNonlinearMLP on nonlinear SEM — acyclic:", is_dag(W_mlp_nlin))

if model_mlp.history_.get("loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(model_mlp.history_["loss"], color="steelblue")
    plt.title("NOTEARSNonlinearMLP — Training Loss (Nonlinear SEM)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()"""
        ),
        md("### Model 3: NOTEARSNonlinearSobolev"),
        code(
            """set_seed(42)
model_sob = NOTEARSNonlinearSobolev(degree=4, lambda1=0.01, rho=1.0, epochs=150, lr=1e-2, random_state=42)
model_sob.fit(X_nlin_train)
W_sob_nlin = model_sob.adjacency_matrix()
B_sob_nlin = (W_sob_nlin > 1e-5).astype(int)
print("NOTEARSNonlinearSobolev on nonlinear SEM — acyclic:", is_dag(W_sob_nlin))

if model_sob.history_.get("loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(model_sob.history_["loss"], color="darkorange")
    plt.title("NOTEARSNonlinearSobolev — Training Loss (Nonlinear SEM)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()"""
        ),
        md("### Model Comparison"),
        code(
            """def safe_acc(B_true, W_est, B_est, label):
    if not is_dag(W_est):
        print(f"  {label:30s}  NOT a DAG — skipping metrics")
        return None
    return count_accuracy(B_true, B_est)

acc_lin_on_nlin = safe_acc(B_true_nlin, W_lin_nlin, B_lin_nlin, "Linear NOTEARS (nonlin SEM)")
acc_mlp_on_nlin = safe_acc(B_true_nlin, W_mlp_nlin, B_mlp_nlin, "NOTEARSNonlinearMLP")
acc_sob_on_nlin = safe_acc(B_true_nlin, W_sob_nlin, B_sob_nlin, "NOTEARSNonlinearSobolev")

def print_row(label, acc):
    if acc is None:
        print(f"  {label:34s} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>5s} {'--':>5s}")
    else:
        print(f"  {label:34s} {acc['fdr']:7.4f} {acc['tpr']:7.4f} {acc['fpr']:7.4f} {acc['shd']:5d} {acc['nnz']:5d}")

div = "-" * 72
print("\n", div)
print(f"  {'Model':34s} {'FDR':>7s} {'TPR':>7s} {'FPR':>7s} {'SHD':>5s} {'nnz':>5s}")
print(div)
print(f"  {'Linear NOTEARS — native SEM *':34s} {acc_lin_result['fdr']:7.4f} {acc_lin_result['tpr']:7.4f} {acc_lin_result['fpr']:7.4f} {acc_lin_result['shd']:5d} {acc_lin_result['nnz']:5d}   (d=20, linear SEM)")
print(div)
print_row("Linear NOTEARS  (misspecified baseline)", acc_lin_on_nlin)
print_row("NOTEARSNonlinearMLP (hidden=10)", acc_mlp_on_nlin)
print_row("NOTEARSNonlinearSobolev (degree=4)", acc_sob_on_nlin)
print(div)
print(f"  True edges in nonlinear SEM graph: {int(B_true_nlin.sum())}  |  d = {d_nlin}")"""
        ),
        md("### ATE Estimation on the Nonlinear SEM"),
        code(
            """T_idx_nlin, Y_idx_nlin = 0, d_nlin - 1

if acc_mlp_on_nlin is not None:
    W_best_nlin, label_best = W_mlp_nlin, "NOTEARSNonlinearMLP"
elif acc_sob_on_nlin is not None:
    W_best_nlin, label_best = W_sob_nlin, "NOTEARSNonlinearSobolev"
else:
    W_best_nlin, label_best = W_lin_nlin, "Linear NOTEARS (fallback)"
print(f"Adjustment set derived from: {label_best}")

pa_Y_nlin = np.where(np.abs(W_best_nlin[Y_idx_nlin, :]) > 1e-5)[0]
conf_nlin = [i for i in pa_Y_nlin if i != T_idx_nlin]
if not conf_nlin:
    conf_nlin = [i for i in range(d_nlin) if i not in (T_idx_nlin, Y_idx_nlin)]

T_n = X_nlin_train[:, T_idx_nlin:T_idx_nlin + 1]
Y_n = X_nlin_train[:, Y_idx_nlin]
C_n = X_nlin_train[:, conf_nlin]

X_design = np.hstack([T_n, C_n])
model_ate_nlin = LinearRegression().fit(X_design, Y_n)
ate_nlin = model_ate_nlin.coef_[0]
ate_unadj_nlin = LinearRegression().fit(T_n, Y_n).coef_[0]

rng = np.random.default_rng(42)
ate_boot = []
for _ in range(500):
    idx = rng.integers(0, X_nlin_train.shape[0], X_nlin_train.shape[0])
    Tb = X_nlin_train[idx, T_idx_nlin:T_idx_nlin + 1]
    Yb = X_nlin_train[idx, Y_idx_nlin]
    Cb = X_nlin_train[idx, conf_nlin]
    ate_boot.append(LinearRegression().fit(np.hstack([Tb, Cb]), Yb).coef_[0])
ci_lo, ci_hi = np.quantile(ate_boot, [0.025, 0.975])

try:
    struct_ate = np.linalg.solve(np.eye(d_nlin) - W_best_nlin, np.eye(d_nlin))[Y_idx_nlin, T_idx_nlin]
except np.linalg.LinAlgError:
    struct_ate = np.nan

print(f"\nATE  (T={T_idx_nlin} -> Y={Y_idx_nlin})  on Nonlinear SEM")
print(f"  Adjusted estimate   : {ate_nlin:+.4f}")
print(f"  Unadjusted estimate : {ate_unadj_nlin:+.4f}")
print(f"  Bootstrap 95% CI    : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Structural ATE (W)  : {struct_ate:+.4f}  (linear approx.)")"""
        ),
        md("### Visualization — Adjacency Matrix Heatmaps"),
        code(
            """def make_heatmap(W, title_str, lim=1.0):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_title(title_str, fontsize=9, fontweight="bold")
    ax.set_xlabel("Cause")
    ax.set_ylabel("Effect")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return fig

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, W, title in zip(
    axes,
    [W_lin_nlin, W_mlp_nlin, W_sob_nlin],
    ["Linear NOTEARS\n(nonlinear SEM baseline)", "NOTEARSNonlinearMLP", "NOTEARSNonlinearSobolev"],
):
    im = ax.imshow(W, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Cause")
    ax.set_ylabel("Effect")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()"""
        ),
        md("### Visualization — Learned DAGs"),
        code(
            """fig, axes = plt.subplots(1, 4, figsize=(16, 5))

def plot_dag_ax(ax, B, title, node_color="lightblue", edge_color="gray40"):
    G = nx.from_numpy_array(B.astype(int), create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color, edge_color=edge_color,
            arrows=True, node_size=500, font_size=8)
    ax.set_title(title)

plot_dag_ax(axes[0], B_true_nlin, "Ground Truth\n(Nonlinear SEM)", "lightgreen", "forestgreen")
plot_dag_ax(axes[1], B_lin_nlin, "Linear NOTEARS\n(misspecified baseline)")
B_plot_mlp = B_mlp_nlin if acc_mlp_on_nlin else np.zeros((d_nlin, d_nlin))
B_plot_sob = B_sob_nlin if acc_sob_on_nlin else np.zeros((d_nlin, d_nlin))
plot_dag_ax(axes[2], B_plot_mlp, "NOTEARSNonlinearMLP", "lightsalmon", "tomato")
plot_dag_ax(axes[3], B_plot_sob, "NOTEARSNonlinearSobolev", "plum", "purple")
plt.tight_layout()
plt.show()"""
        ),
        md(
            """### Interpretation

Linear NOTEARS on nonlinear data is a deliberate stress test. Nonlinear variants tend to outperform when sample size is sufficient ($n \\geq 500$ per variable).

DAG validity must be verified with `is_dag()` before interpreting any structural metric.

---

## Summary and Conclusion

NOTEARS transformed causal structure learning by replacing an NP-hard combinatorial search with a smooth continuous optimization problem.

**Strengths.** The linear variant is simple, fast on moderate graphs ($d \\leq 200$), and produces interpretable weighted edges.

**Limitations.** The linear variant assumes additive Gaussian noise with no hidden confounders. Always center or standardize variables before fitting.

---

## Resources

- **Original Paper**: [DAGs with NO TEARS (Zheng et al., 2018)](https://arxiv.org/abs/1803.01422)
- **Official Code**: [github.com/xunzheng/notears](https://github.com/xunzheng/notears)
- **PyDeepCausalML**: `pydeepcausalml.discovery.NOTEARSLinear`, `NOTEARSNonlinearMLP`, `NOTEARSNonlinearSobolev`"""
        ),
    ]
    return nb(cells)


def build_daggnn() -> dict:
    cells = [
        md("![Banner](../Image/03_DeepCausalML.png)"),
        md(
            """# 3.2 DAG-GNN: DAG Structure Learning with Graph Neural Networks

> **Note:** DAG-GNN requires **PyTorch**. The `DAGGNN` estimator in `pydeepcausalml.discovery` learns DAG structure via a VAE-style encoder–decoder with an augmented-Lagrangian acyclicity penalty.

**DAG-GNN** (Yu et al., ICML 2019) learns the structure of a Directed Acyclic Graph (DAG) from observational data by embedding the problem inside a deep generative model. Where NOTEARS replaces a combinatorial graph search with a smooth optimization over a weighted adjacency matrix $\\mathbf{W}$, DAG-GNN wraps that adjacency matrix inside a **Variational Autoencoder (VAE)** driven by **Graph Neural Network (GNN)** encoder and decoder modules.

![](../Image/DAG_GNN.png)"""
        ),
        md(
            """## Implementation in Python

We use **PyDeepCausalML**'s `DAGGNN` class for causal structure discovery and effect estimation on synthetic data.

### Setup"""
        ),
        SETUP_CELL,
        IMPORTS_CELL,
        SEED_CELL,
        md("### Data and Data Processing\n\nWe generate a small **6-node nonlinear DAG**."),
        code(
            """def generate_synthetic_dag_data(n_samples=5000, n_nodes=6, seed=42):
    rng = np.random.default_rng(seed)
    p = 3.0 / (n_nodes - 1) if n_nodes > 1 else 0.0
    G = nx.erdos_renyi_graph(n_nodes, p, directed=True, seed=seed)
    edges = [(u, v) for u, v in G.edges() if u < v]
    G = nx.DiGraph(edges)

    A_true = nx.to_numpy_array(G, dtype=float)
    A_true *= rng.uniform(0.5, 1.5, size=A_true.shape)

    X = np.zeros((n_samples, n_nodes))
    for _ in range(10):
        Z = rng.standard_normal((n_samples, n_nodes))
        X = np.cos(X @ A_true.T + 1) + Z

    T_var = ((rng.standard_normal(n_samples) + 0.5 * X[:, 0] + 0.3 * X[:, 1]) > 0).astype(float)
    Y_var = 2 * T_var + 0.8 * X[:, 0] + np.sin(X[:, 2]) + rng.normal(0, 0.5, n_samples)
    X[:, 4] = T_var
    X[:, 5] = Y_var

    cols = [f"X{i}" for i in range(n_nodes)] + ["T", "Y"]
    df = pd.DataFrame(np.column_stack([X, T_var, Y_var]), columns=cols)
    return df, A_true, G

synth_data = generate_synthetic_dag_data()
df, A_true, true_G = synth_data
print(df.head())

node_cols = [f"X{i}" for i in range(6)]
df_norm = df.copy()
df_norm[node_cols] = (df[node_cols] - df[node_cols].mean()) / (df[node_cols].std() + 1e-8)"""
        ),
        md("### Data split"),
        code(
            """rng = np.random.default_rng(42)
idx = rng.permutation(len(df_norm))
train_idx = idx[: int(0.7 * len(idx))]
temp_idx = idx[int(0.7 * len(idx)) :]
valid_idx = temp_idx[: len(temp_idx) // 2]
test_idx = temp_idx[len(temp_idx) // 2 :]

train_df = df_norm.iloc[train_idx].reset_index(drop=True)
valid_df = df_norm.iloc[valid_idx].reset_index(drop=True)
test_df = df_norm.iloc[test_idx].reset_index(drop=True)

X_train = train_df[node_cols].values.astype(np.float32)
X_valid = valid_df[node_cols].values.astype(np.float32)
X_test = test_df[node_cols].values.astype(np.float32)
print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)"""
        ),
        md(
            """## DAG-GNN Model Architecture

DAG-GNN frames causal structure learning as a **Variational Autoencoder (VAE)** whose latent space is organized by a learnable adjacency matrix $\\mathbf{A}$.

### Acyclicity Constraint $h(\\mathbf{A})$

Acyclicity is enforced via the NOTEARS-style penalty embedded in the training objective.

### Total Objective

$$\\mathcal{L} = \\text{reconstruction} + \\lambda\\, h(\\mathbf{A})^2 + \\text{sparsity}$$"""
        ),
        md("### Model instantiation and training"),
        code(
            """from pydeepcausalml.discovery import DAGGNN

set_seed(42)
model = DAGGNN(hidden=64, latent=16, lambda_dag=0.1, epochs=600, lr=1e-3, random_state=42)
model.fit(X_train)

A_learned = model.adjacency_matrix()
threshold = 0.10
A_binary = (A_learned > threshold).astype(int)
print(f"A_eff: min={A_learned.min():.4f}, max={A_learned.max():.4f}")
print(f"Edges (threshold {threshold}): {int(A_binary.sum())}")"""
        ),
        md("### ATE and CATE Estimation"),
        code(
            """cov_cols = ["X0", "X1", "X2", "X3"]
X_train_cov = train_df[cov_cols].values
X_test_cov = test_df[cov_cols].values
treatment_train = train_df["T"].values
treatment_test = test_df["T"].values
y_train = train_df["Y"].values
y_test = test_df["Y"].values

treatment_train_bin = treatment_train.astype(float)
treatment_test_bin = treatment_test.astype(float)

X_tr = np.column_stack([treatment_train_bin, X_train_cov])
lr_model = LinearRegression().fit(X_tr, y_train)
cate_coef = lr_model.coef_[0]
cate_arr = np.full(len(y_test), cate_coef)
ate_val = cate_arr.mean()

true_ate = 2.0
se_ate = np.sqrt(np.mean((y_test - lr_model.predict(np.column_stack([treatment_test_bin, X_test_cov]))) ** 2) / len(y_test))
ci_lb, ci_ub = ate_val - 1.96 * se_ate, ate_val + 1.96 * se_ate

print("=== ATE on test set ===")
print(f"Estimated ATE (true ATE = {true_ate}): {ate_val:.4f}")
print(f"ATE 95% CI: [{ci_lb:.4f}, {ci_ub:.4f}]")
print(f"PEHE: {pehe(true_ate, cate_arr):.4f}")"""
        ),
        md("## Results visualization\n\n### True vs learned DAG"),
        code(
            """node_labels = [f"X{i}" for i in range(6)]
A_true_bin = (A_true != 0).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, B, title, color in [
    (axes[0], A_binary, "Learned DAG (DAG-GNN)", "steelblue"),
    (axes[1], A_true_bin, "True DAG (data-generating)", "darkseagreen"),
]:
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=color, node_size=700, arrows=True)
    ax.set_title(title)
plt.tight_layout()
plt.show()"""
        ),
        md("### Adjacency matrices: true vs learned"),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, M, title in [
    (axes[0], A_true, "True adjacency (weighted)"),
    (axes[1], A_binary, "Learned adjacency (thresholded)"),
]:
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(6), node_labels, rotation=90)
    ax.set_yticks(range(6), node_labels)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()"""
        ),
        md("### Training curves"),
        code(
            """hist = model.history_
if hist.get("loss"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["loss"], color="steelblue")
    axes[0].set_title("Training: reconstruction + DAG loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    if hist.get("dag"):
        axes[1].plot(hist["dag"], color="darkgreen")
        axes[1].set_title("DAG penalty")
        axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()"""
        ),
        md("### ATE and CATE summary"),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].errorbar([0], [ate_val], yerr=[[ate_val - ci_lb], [ci_ub - ate_val]], fmt="o", color="steelblue", capsize=8)
axes[0].axhline(true_ate, color="gray", linestyle="--")
axes[0].set_title("Estimated ATE with 95% CI")
axes[0].set_ylabel("Effect")
axes[1].hist(cate_arr, bins=30, color="steelblue", alpha=0.7)
axes[1].axvline(true_ate, color="gray", linestyle="--")
axes[1].set_title("CATE on test set")
axes[1].set_xlabel("CATE")
plt.tight_layout()
plt.show()"""
        ),
        md(
            """## Summary and conclusion

DAG-GNN turns the notoriously hard DAG-learning problem into a differentiable VAE + GNN optimization using **PyDeepCausalML**'s `DAGGNN` estimator.

1. Run DAG-GNN on your observational dataset.
2. Threshold the learned adjacency (default **0.10**) to obtain a sparse causal DAG.
3. Feed the DAG into downstream effect estimators for ATE/CATE analysis.

## Resources

- **Paper**: Yu et al. (2019). [DAG-GNN](https://arxiv.org/abs/1904.10098)
- **Official code**: [fishmoon1234/DAG-GNN](https://github.com/fishmoon1234/DAG-GNN)
- **PyDeepCausalML**: `pydeepcausalml.discovery.DAGGNN`"""
        ),
    ]
    return nb(cells)


def build_grandag() -> dict:
    cells = [
        md("![Banner](../Image/03_DeepCausalML.png)"),
        md(
            """# 3.3 GraN-DAG: Gradient-Based Neural DAG Learning

> **Note:** This notebook uses **PyDeepCausalML**'s `NOTEARSNonlinearMLP`, which implements the same masked-MLP + differentiable acyclicity framework as GraN-DAG (Lachapelle et al., NeurIPS 2020).

**GraN-DAG** models every conditional distribution $p(X_i \\mid \\mathbf{X}_{\\mathrm{pa}(i)})$ with a separate **masked MLP** and extracts a continuous adjacency matrix from the neural network weights, retaining the NOTEARS-style acyclicity constraint.

![](../Image/GraN-DAG.png)"""
        ),
        md("## Implementation in Python\n\n### Setup"),
        SETUP_CELL,
        IMPORTS_CELL,
        SEED_CELL,
        md("### Data and data processing\n\nWe simulate a 10-node Erdős–Rényi DAG with nonlinear Gaussian ANM."),
        code(
            """def generate_dag_data(n_nodes=10, n_edges=10, n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    W = np.zeros((n_nodes, n_nodes))
    possible = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    chosen = rng.choice(len(possible), size=min(n_edges, len(possible)), replace=False)
    for k in chosen:
        i, j = possible[k]
        W[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
    X = np.zeros((n_samples, n_nodes))
    for j in range(n_nodes):
        parents = np.where(W[:, j] != 0)[0]
        noise = rng.standard_normal(n_samples)
        if len(parents):
            effect = (X[:, parents] * W[parents, j]).sum(axis=1)
            X[:, j] = np.maximum(effect, 0) + noise
        else:
            X[:, j] = noise
    return W, X

true_causal_matrix, data = generate_dag_data(n_nodes=10, n_edges=10, n_samples=2000, seed=42)
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
print("Data shape:", data.shape)
print("True edges:", int((true_causal_matrix != 0).sum()))"""
        ),
        md("### Data split"),
        code(
            """rng = np.random.default_rng(42)
n = data.shape[0]
train_idx = rng.choice(n, size=int(0.8 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
train_data, test_data = data[train_idx], data[test_idx]
print("Training samples:", train_data.shape[0])
print("Test samples:", test_data.shape[0])"""
        ),
        md("## Training (Structure Learning)\n\nWe fit `NOTEARSNonlinearMLP` — the PyDeepCausalML implementation of masked per-node MLPs with acyclicity constraint."),
        code(
            """from pydeepcausalml.discovery import NOTEARSNonlinearMLP

set_seed(42)
gnd_model = NOTEARSNonlinearMLP(
    hidden=10,
    lambda1=0.001,
    rho=1.0,
    epochs=3000,
    lr=1e-3,
    batch_size=64,
    random_state=42,
)
gnd_model.fit(data)
learned_adj = gnd_model.adjacency_matrix()
learned_bin = (learned_adj != 0).astype(int)
print("Learned edges:", int(learned_bin.sum()))"""
        ),
        md("## CATE Prediction and Validation\n\nNode 0 is **Treatment** (binarized at median) and node 9 is **Outcome**."),
        code(
            """treatment_idx, outcome_idx = 0, 9
T = (data[:, treatment_idx] > np.median(data[:, treatment_idx])).astype(int)
Y = data[:, outcome_idx]
parents_of_Y = np.where(learned_bin[:, outcome_idx] != 0)[0]
adjustment_set = [i for i in parents_of_Y if i != treatment_idx]
print("Adjustment set for backdoor:", adjustment_set)

if adjustment_set:
    X_cate = np.column_stack([T, data[:, adjustment_set]])
else:
    X_cate = T.reshape(-1, 1)

rng = np.random.default_rng(42)
tr = rng.choice(len(Y), size=int(0.8 * len(Y)), replace=False)
reg = LinearRegression().fit(X_cate[tr], Y[tr])
cate_estimate = reg.coef_[0]
naive_corr = np.corrcoef(T, Y)[0, 1]
print("Estimated ATE (via adjustment):", cate_estimate)
print("Validation: naive correlation:", naive_corr)"""
        ),
        md("## Validation of Structure"),
        code(
            """def calculate_dag_metrics(learned, true_adj):
    learned_bin = (learned != 0).astype(int)
    true_bin = (true_adj != 0).astype(int)
    m = graph_recovery_metrics(true_bin, learned_bin)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    tn = int((~np.eye(true_bin.shape[0], dtype=bool)).sum()) - tp - fp - fn
    fdr = fp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": m["precision"],
        "recall": m["recall"],
        "fdr": round(fdr, 4),
        "tpr": m["recall"],
        "fpr": round(fpr, 4),
        "shd": m["shd"],
        "nnz": int(learned_bin.sum()),
        "f1": m["f1"],
    }

metrics = calculate_dag_metrics(learned_adj, true_causal_matrix)
pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})"""
        ),
        md("## Visualization"),
        code(
            """learned_bin = (learned_adj != 0).astype(int)
true_bin = (true_causal_matrix != 0).astype(int)
G_learned = nx.from_numpy_array(learned_bin, create_using=nx.DiGraph)
G_true = nx.from_numpy_array(true_bin, create_using=nx.DiGraph)
pos = nx.spring_layout(G_learned, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
for ax, G, title, color in [
    (axes[0], G_learned, "Learned GraN-DAG (NOTEARSNonlinearMLP)", "lightblue"),
    (axes[1], G_true, "True DAG", "lightgreen"),
]:
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=color, arrows=True, node_size=500)
    ax.set_title(title)
plt.tight_layout()
plt.show()"""
        ),
        md(
            """## Summary and Conclusion

**GraN-DAG** achieves nonlinear causal structure learning via masked MLPs and a differentiable acyclicity constraint. This notebook uses PyDeepCausalML's `NOTEARSNonlinearMLP`, which follows the same gradient-based neural DAG learning paradigm.

## Resources

- **GraN-DAG paper**: [arXiv:1906.02226](https://arxiv.org/abs/1906.02226)
- **PyDeepCausalML**: `pydeepcausalml.discovery.NOTEARSNonlinearMLP`"""
        ),
    ]
    return nb(cells)


IHDP_HELPERS = code(
    '''from pathlib import Path

def data_loading_ihdp(train_rate=0.8, replications=1, seed=42):
    """Load IHDP NPCI replicates (with synthetic fallback)."""
    import urllib.request
    base_url = "https://raw.githubusercontent.com/uber/causalml/master/docs/examples/data"
    repo = Path(".").resolve()
    local_dirs = [
        repo / "inst/examples/data",
        repo / "inst/exdata",
        repo.parent / "examples/data",
    ]
    dfs = []
    for i in range(1, 10):
        fname = f"ihdp_npci_{i}.csv"
        local_path = next((d / fname for d in local_dirs if (d / fname).exists()), None)
        if local_path is not None:
            dfs.append(pd.read_csv(local_path, header=None))
            continue
        url = f"{base_url}/{fname}"
        try:
            dfs.append(pd.read_csv(url, header=None))
        except Exception:
            pass
    if not dfs:
        rng = np.random.default_rng(seed)
        n = 747 * 9
        x_cont = rng.standard_normal((n, 6))
        x_bin = rng.integers(0, 2, size=(n, 19))
        x = np.column_stack([x_cont, x_bin])
        treatment = rng.integers(0, 2, size=n)
        mu0 = 0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.15 * x[:, 2] + rng.normal(0, 0.2, n)
        tau = 1.0 + 0.2 * x[:, 3] - 0.15 * x[:, 4]
        mu1 = mu0 + tau
        y_factual = np.where(treatment == 1, mu1, mu0) + rng.normal(0, 0.3, n)
        y_cfactual = np.where(treatment == 1, mu0, mu1) + rng.normal(0, 0.3, n)
        df = pd.DataFrame(np.column_stack([treatment, y_factual, y_cfactual, mu0, mu1, x]))
    else:
        df = pd.concat(dfs, ignore_index=True)
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    df.columns = cols
    if replications > 1:
        df = pd.concat([df] * replications, ignore_index=True)
    x = df[[f"x{i}" for i in range(1, 26)]].values.astype(float)
    t = df["treatment"].values.astype(float)
    y = df["y_factual"].values.astype(float)
    potential_y = df[["mu0", "mu1"]].values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_train = int(train_rate * len(x))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return {
        "train_x": x[train_idx], "train_t": t[train_idx], "train_y": y[train_idx],
        "train_potential_y": potential_y[train_idx],
        "test_x": x[test_idx], "test_potential_y": potential_y[test_idx],
    }


def preprocess_features(train_x, test_x):
    mu = train_x[:, :6].mean(axis=0)
    sd = train_x[:, :6].std(axis=0)
    sd[sd == 0] = 1.0
    a = train_x.copy()
    b = test_x.copy()
    a[:, :6] = (train_x[:, :6] - mu) / sd
    b[:, :6] = (test_x[:, :6] - mu) / sd
    return a, b


def build_causal_matrix(x, t, y, subsample=5000, seed=42):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([x, t, y])
    idx = rng.choice(len(Z), size=min(subsample, len(Z)), replace=False)
    return Z[idx]'''
)

CASTLE_VALIDATION_HELPERS = code(
    """def adjacency_to_igraph(W, names, threshold=0):
    W2 = W.copy()
    np.fill_diagonal(W2, 0)
    W2[np.abs(W2) <= threshold] = 0
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for i, fr in enumerate(names):
        for j, to in enumerate(names):
            if W2[i, j] != 0:
                G.add_edge(fr, to, weight=W2[i, j])
    return G


def sem_reconstruction_metrics(W, Zt, model_name=""):
    X_hat = Zt @ W
    mse_pv = ((Zt - X_hat) ** 2).mean(axis=0)
    ss_res = ((Zt - X_hat) ** 2).sum()
    ss_tot = ((Zt - Zt.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"model": model_name, "mean_MSE": round(mse_pv.mean(), 4), "global_R2": round(r2, 4),
            "mse_T": round(mse_pv[25], 4), "mse_Y": round(mse_pv[26], 4)}


def estimate_ate_adjustment(W, names, trx, trt, try_, tex, tpy, model_name=""):
    G = adjacency_to_igraph(W, names)
    y_parents = [p for p in G.predecessors("Y")]
    par_idx = [names.index(p) for p in y_parents if p not in ("T", "Y") and p in names[:trx.shape[1]]]
    if not par_idx:
        par_idx = list(range(trx.shape[1]))
    idx_t, idx_c = trt == 1, trt == 0
    fit1 = LinearRegression().fit(trx[idx_t][:, par_idx], try_[idx_t])
    fit0 = LinearRegression().fit(trx[idx_c][:, par_idx], try_[idx_c])
    m1 = fit1.predict(tex[:, par_idx])
    m0 = fit0.predict(tex[:, par_idx])
    ah = (m1 - m0).mean()
    oa = (tpy[:, 1] - tpy[:, 0]).mean()
    pehe_val = np.sqrt(((m1 - m0) - (tpy[:, 1] - tpy[:, 0])) ** 2).mean()
    return {"model": model_name, "n_adj_vars": len(par_idx), "ATE_estimated": round(ah, 4),
            "ATE_oracle": round(oa, 4), "ATE_error": round(abs(ah - oa), 4), "sqrt_PEHE": round(pehe_val, 4)}"""
)


def build_dagma() -> dict:
    cells = [
        md("![Banner](../Image/03_DeepCausalML.png)"),
        md(
            """# 3.4 DAGMA and DAG-NoCurl for Causal Discovery

> **Note:** DAGMA uses **PyDeepCausalML**'s `DagmaLinear` and `DagmaNonlinearMLP`. DAG-NoCurl is implemented inline with PyTorch following Yu et al. (2021).

This notebook compares three continuous-optimization methods — **DAGMA Linear**, **DAGMA Nonlinear (MLP)**, and **DAG-NoCurl** — on the **IHDP** benchmark dataset.

![](../Image/dagma.png)"""
        ),
        md("## Setup"),
        SETUP_CELL,
        IMPORTS_CELL,
        SEED_CELL,
        md("### IHDP data helpers"),
        IHDP_HELPERS,
        md("### Data Loading and Preprocessing"),
        code(
            """print("Loading IHDP data ...")
ihdp = data_loading_ihdp(train_rate=0.8, replications=1 if run_fast else 100)
train_x, test_x = preprocess_features(ihdp["train_x"], ihdp["test_x"])
train_t, train_y = ihdp["train_t"], ihdp["train_y"]
train_potential_y = ihdp["train_potential_y"]
test_potential_y = ihdp["test_potential_y"]
print("Train size :", f"{train_x.shape[0]:,}")
print("Test  size :", f"{test_x.shape[0]:,}")
print("Covariates :", train_x.shape[1])
print(f"Treatment prevalence (train): {train_t.mean():.3f}")"""
        ),
        md(
            """### Assembling the Causal Discovery Matrix

Concatenate covariates, treatment `T`, and factual outcome `Y` into `Z` with shape `(N, 27)`.

| Column range | Content |
|--------------|---------|
| 0 - 24 | Covariates x1 ... x25 |
| 25 | Treatment T |
| 26 | Outcome Y |"""
        ),
        code(
            """VAR_NAMES = [f"x{i}" for i in range(1, 26)] + ["T", "Y"]
Z_train = build_causal_matrix(train_x, train_t, train_y, subsample=5000)
Z_test_eval = build_causal_matrix(test_x, test_potential_y[:, 1] - test_potential_y[:, 0], np.zeros(len(test_x)), subsample=2000)
print("Causal matrix shape (train):", Z_train.shape)
print("Variable names:", ", ".join(VAR_NAMES))"""
        ),
        md("### Exploratory Data Analysis"),
        code(
            """corr_matrix = np.corrcoef(Z_train.T)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(VAR_NAMES)), VAR_NAMES, rotation=90, fontsize=6)
ax.set_yticks(range(len(VAR_NAMES)), VAR_NAMES, fontsize=6)
ax.set_title("Pairwise Pearson Correlation — IHDP (train subsample)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

others = [i for i, n in enumerate(VAR_NAMES) if n != "T"]
print("Top correlations with T:")
print(sorted([(VAR_NAMES[i], abs(corr_matrix[25, i])) for i in others], key=lambda x: -x[1])[:8])"""
        ),
        md("## Model Setup\n\n### DAGMA Linear"),
        code(
            """from pydeepcausalml.discovery import DagmaLinear, DagmaNonlinearMLP

print("Training DAGMA Linear ...")
dagma_lin = DagmaLinear(loss_type="l2", lambda1=0.02, max_iter=4000, random_state=42)
dagma_lin.fit(Z_train)
W_dagma_linear = dagma_lin.adjacency_matrix()
print(f"  Recovered edges : {int((W_dagma_linear != 0).sum())}")
print(f"  W_est shape     : {W_dagma_linear.shape}")"""
        ),
        md("### DAGMA Nonlinear (MLP)"),
        code(
            """Z_nl = build_causal_matrix(train_x, train_t, train_y, subsample=2000)
print("Training DAGMA Nonlinear (MLP) ...")
dagma_nl = DagmaNonlinearMLP(hidden=10, lambda1=0.02, epochs=2500, lr=1e-3, random_state=42)
dagma_nl.fit(Z_nl)
W_dagma_nonlinear = dagma_nl.adjacency_matrix()
print(f"  Recovered edges : {int((W_dagma_nonlinear > 0.25).sum())}")"""
        ),
        md("### DAG-NoCurl"),
        code(
            """class DagNoCurl:
    def __init__(self, d, lambda1=0.02, mu=0.1, device="cpu"):
        self.lambda1, self.mu = lambda1, mu
        self.U = torch.randn(d, d, dtype=torch.float32, device=device, requires_grad=True)

    def get_W(self):
        W = self.U.T @ self.U
        return W - torch.diag(W.diag())

    def curl_penalty(self, W):
        curl = (W - W.T) / 2
        return (curl ** 2).sum()

    def loss(self, X):
        W = self.get_W()
        X_hat = X @ W
        recon = 0.5 / X.shape[0] * ((X - X_hat) ** 2).sum()
        return recon + self.lambda1 * W.abs().sum() + self.mu * self.curl_penalty(W)


def train_dag_nocurl(Z, n_epochs=3000, lr=1e-3, w_threshold=0.2, print_every=500):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(Z, dtype=torch.float32, device=device)
    model = DagNoCurl(Z.shape[1], device=device)
    opt = torch.optim.Adam([model.U], lr=lr)
    losses = []
    print("Training DAG-NoCurl ...")
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = model.loss(X)
        loss.backward()
        opt.step()
        lv = float(loss.item())
        losses.append(lv)
        if epoch % print_every == 0:
            W_now = model.get_W().detach().cpu().numpy()
            print(f"  Epoch {epoch:5d} | loss={lv:.4f} | edges>{w_threshold}: {(np.abs(W_now) > w_threshold).sum()}")
    W_raw = model.get_W().detach().cpu().numpy()
    W_est = np.where(np.abs(W_raw) > w_threshold, W_raw, 0.0)
    print(f"  Final edges : {int((W_est != 0).sum())}")
    return W_est, losses

W_nocurl, nocurl_losses = train_dag_nocurl(Z_train, n_epochs=1500 if run_fast else 3000)
plt.plot(nocurl_losses, color="steelblue")
plt.title("DAG-NoCurl Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()"""
        ),
        md("### Visualizing the Learned DAGs"),
        code(
            """def adjacency_to_digraph(W, names, threshold=0):
    W2 = W.copy()
    W2[np.abs(W2) <= threshold] = 0
    np.fill_diagonal(W2, 0)
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if W2[i, j] != 0:
                G.add_edge(nj, ni, weight=W2[i, j])
    return G

def plot_dag(G, title, highlight=None):
    pos = nx.spring_layout(G, seed=42)
    colors = ["#e74c3c" if highlight and n in highlight else "#3498db" for n in G.nodes()]
    widths = [min(abs(d.get("weight", 1)) * 3, 4) for _, _, d in G.edges(data=True)]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=7,
            edge_color="grey40", width=widths or 1.0, arrows=True)
    plt.title(title)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, W, title in zip(
    axes,
    [W_dagma_linear, W_dagma_nonlinear, W_nocurl],
    ["DAGMA Linear — IHDP", "DAGMA Nonlinear — IHDP", "DAG-NoCurl — IHDP"],
):
    G = adjacency_to_digraph(W, VAR_NAMES, threshold=0.25)
    plt.sca(ax)
    plot_dag(G, title, highlight={"T", "Y"})
plt.tight_layout()
plt.show()"""
        ),
        md("### Validation"),
        code(
            """def graph_stats(G, name=""):
    d, e = G.number_of_nodes(), G.number_of_edges()
    density = e / (d * (d - 1)) if d > 1 else 0
    return {"model": name, "nodes": d, "edges": e, "density": round(density, 4),
            "isDAG": nx.is_directed_acyclic_graph(G)}

def sem_reconstruction_metrics(W, Zt, model_name=""):
    X_hat = Zt @ W
    mse_pv = ((Zt - X_hat) ** 2).mean(axis=0)
    ss_res = ((Zt - X_hat) ** 2).sum()
    ss_tot = ((Zt - Zt.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"model": model_name, "mean_MSE": round(mse_pv.mean(), 4), "global_R2": round(r2, 4),
            "mse_T": round(mse_pv[25], 4), "mse_Y": round(mse_pv[26], 4)}

G_linear = adjacency_to_digraph(W_dagma_linear, VAR_NAMES)
G_nonlinear = adjacency_to_digraph(W_dagma_nonlinear, VAR_NAMES)
G_nocurl = adjacency_to_digraph(W_nocurl, VAR_NAMES)

stats_df = pd.DataFrame([
    graph_stats(G_linear, "DAGMA Linear"),
    graph_stats(G_nonlinear, "DAGMA Nonlinear"),
    graph_stats(G_nocurl, "DAG-NoCurl"),
]).set_index("model")
print("Graph statistics\n", stats_df)

recon_df = pd.DataFrame([
    sem_reconstruction_metrics(W_dagma_linear, Z_test_eval, "DAGMA Linear"),
    sem_reconstruction_metrics(W_dagma_nonlinear, Z_test_eval, "DAGMA Nonlinear"),
    sem_reconstruction_metrics(W_nocurl, Z_test_eval, "DAG-NoCurl"),
]).set_index("model")
print("\nSEM reconstruction\n", recon_df)

def estimate_ate_adjustment(W, names, trx, trt, try_, tex, tpy, model_name=""):
    G = adjacency_to_digraph(W, names)
    y_parents = [p for p in G.predecessors("Y")]
    par_idx = [names.index(p) for p in y_parents if p not in ("T", "Y") and p in names[:trx.shape[1]]]
    if not par_idx:
        par_idx = list(range(trx.shape[1]))
    idx_t, idx_c = trt == 1, trt == 0
    fit1 = LinearRegression().fit(trx[idx_t][:, par_idx], try_[idx_t])
    fit0 = LinearRegression().fit(trx[idx_c][:, par_idx], try_[idx_c])
    m1 = fit1.predict(tex[:, par_idx])
    m0 = fit0.predict(tex[:, par_idx])
    ah = (m1 - m0).mean()
    oa = (tpy[:, 1] - tpy[:, 0]).mean()
    pehe_val = np.sqrt(((m1 - m0) - (tpy[:, 1] - tpy[:, 0])) ** 2).mean()
    return {"model": model_name, "n_adj_vars": len(par_idx), "ATE_estimated": round(ah, 4),
            "ATE_oracle": round(oa, 4), "ATE_error": round(abs(ah - oa), 4), "sqrt_PEHE": round(pehe_val, 4)}

ate_df = pd.DataFrame([
    estimate_ate_adjustment(W_dagma_linear, VAR_NAMES, train_x, train_t, train_y, test_x, test_potential_y, "DAGMA Linear"),
    estimate_ate_adjustment(W_dagma_nonlinear, VAR_NAMES, train_x, train_t, train_y, test_x, test_potential_y, "DAGMA Nonlinear"),
    estimate_ate_adjustment(W_nocurl, VAR_NAMES, train_x, train_t, train_y, test_x, test_potential_y, "DAG-NoCurl"),
]).set_index("model")
print("\nATE\n", ate_df)"""
        ),
        md(
            """## Summary and Conclusion

This notebook applied three continuous-optimization causal discovery methods to the IHDP dataset using **PyDeepCausalML** (`DagmaLinear`, `DagmaNonlinearMLP`) and an inline DAG-NoCurl implementation.

## Resources

- Bello et al. (2022) [DAGMA](https://arxiv.org/abs/2209.08037)
- Yu et al. (2021) [DAG-NoCurl](https://arxiv.org/abs/2106.07197)
- Zheng et al. (2018) [NOTEARS](https://arxiv.org/abs/1803.01422)"""
        ),
    ]
    return nb(cells)


def build_castle() -> dict:
    cells = [
        md("![Banner](../Image/05-03-deep-CausalML-structural-learning.png)"),
        md(
            """# 3.5 CASTLE: CAusal STructure LEarning Regularization

> **Note:** CASTLE requires **PyTorch**. The `CASTLE` estimator in `pydeepcausalml.discovery` jointly learns a causal graph and a supervised predictor.

**CASTLE** (Kyono, Zhang & van der Schaar, NeurIPS 2020) solves **both** causal structure learning and supervised prediction simultaneously — using causal graph discovery as a *regularizer* for the supervised model.

![](../Image/CASTLE_arcitecture.png)"""
        ),
        md(
            """## Overview

CASTLE's architecture is a **multi-objective neural network** that learns a **weighted adjacency matrix** while optimizing supervised prediction and selective reconstruction.

### Core architecture

- **Learnable adjacency matrix** `A` embedded in the input layer
- **Acyclicity constraint** via NOTEARS-style penalty $h(A) = \\mathrm{tr}(e^{A \\odot A}) - d$
- **Selective reconstruction** of features with causal neighbors"""
        ),
        md("## Implementation in Python\n\nWe use `pydeepcausalml.discovery.CASTLE` on the IHDP dataset.\n\n### Setup"),
        SETUP_CELL,
        IMPORTS_CELL,
        SEED_CELL,
        md("### Data loading and preprocessing"),
        IHDP_HELPERS,
        code(
            """print("Loading IHDP data ...")
data_result = data_loading_ihdp(train_rate=0.8, replications=1)
train_x, test_x = preprocess_features(data_result["train_x"], data_result["test_x"])
train_t = data_result["train_t"]
train_y = data_result["train_y"]
train_potential_y = data_result["train_potential_y"]
test_potential_y = data_result["test_potential_y"]

VAR_NAMES = [f"x{i}" for i in range(1, 26)] + ["T", "Y"]
Z_train = build_causal_matrix(train_x, train_t, train_y, subsample=5000)
Z_test = build_causal_matrix(test_x, test_potential_y[:, 1] - test_potential_y[:, 0], np.zeros(len(test_x)), subsample=2000)
Z_test_eval = Z_test.copy()
print(f"Causal matrix shapes -> Train: {Z_train.shape} | Test: {Z_test.shape}")"""
        ),
        md("### Fit CASTLE model"),
        code(
            """from pydeepcausalml.discovery import CASTLE

d = Z_train.shape[1]
castle_fit = CASTLE(
    y_index=-1,
    hidden_dim=64,
    num_layers=3,
    lambda_reg=1.0,
    beta_sparsity=0.015,
    acyc_weight=0.1,
    recon_weight=0.5,
    epochs=300 if run_fast else 600,
    batch_size=256,
    lr=1e-3,
    random_state=42,
)
castle_fit.fit(Z_train)

loss_df = pd.DataFrame(castle_fit.history_)
print("Training finished. Last epoch:")
print(loss_df.tail(1))

plt.figure(figsize=(8, 4))
plt.plot(loss_df["loss"], label="Train Loss")
if "mse" in loss_df:
    plt.plot(loss_df["mse"], label="MSE Component")
plt.legend()
plt.title("CASTLE Training Loss and MSE Component")
plt.xlabel("Epoch")
plt.tight_layout()
plt.show()

A_est = castle_fit.get_scores()
off = A_est.copy()
np.fill_diagonal(off, 0)
amax, amean = np.abs(off).max(), np.abs(off).mean()
print(f"|A| (off-diag): max={amax:.5f}, mean={amean:.6f}")
A_EDGE_TAU = max(1e-4, 0.2 * amax) if amax > 0 else 0.0
A_thresholded = np.where(np.abs(A_est) > A_EDGE_TAU, A_est, 0.0)
np.fill_diagonal(A_thresholded, 0)
print(f"Adaptive edge threshold tau={A_EDGE_TAU:.5f} -> {(A_thresholded != 0).sum()} directed entries")"""
        ),
        md("### Interpretation and visualization"),
        code(
            """node_labels = VAR_NAMES
A = A_est.copy()
np.fill_diagonal(A, 0)
absA = np.abs(A)
in_mass = absA.sum(axis=0)
out_mass = absA.sum(axis=1)
A_inout = in_mass + out_mass

W1 = castle_fit.module_.predictor[0].weight.detach().cpu().numpy()
W1_norm = np.linalg.norm(W1, axis=0)

def minmax(x):
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)

score = 0.5 * minmax(A_inout) + 0.5 * minmax(W1_norm)
xvars = [i for i, n in enumerate(node_labels) if n.startswith("x")]
top5_idxs = sorted(xvars, key=lambda i: score[i], reverse=True)[:5]
highlight_vars = [node_labels[i] for i in top5_idxs]

importance_df = pd.DataFrame({"variable": node_labels, "score": score, "A_inout": A_inout, "W1_norm": W1_norm})
importance_df = importance_df.sort_values("score", ascending=False)
print(importance_df.head(10))

plt.figure(figsize=(8, 6))
colors = ["gold" if v in highlight_vars else "#377eb8" for v in importance_df["variable"]]
plt.barh(importance_df["variable"], importance_df["score"], color=colors)
plt.title("Feature importance — CASTLE-style (IHDP)")
plt.xlabel("Combined score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

edge_rows = []
for i, fr in enumerate(node_labels):
    for j, to in enumerate(node_labels):
        w = A_thresholded[i, j]
        if i != j and abs(w) > 1e-8:
            edge_rows.append((fr, to, w))
edge_df = pd.DataFrame(edge_rows, columns=["from", "to", "weight"])
if edge_df.empty:
    flat = [(node_labels[i], node_labels[j], abs(A_est[i, j])) for i in range(d) for j in range(d) if i != j]
    edge_df = pd.DataFrame(flat, columns=["from", "to", "weight"]).sort_values("weight", ascending=False).head(45)
    print(f"No edges above tau; showing top {len(edge_df)} |A| edges (exploratory).")
else:
    print(f"Graph from adaptive threshold tau={A_EDGE_TAU:.4g}")

G = nx.DiGraph()
G.add_nodes_from(node_labels)
for _, row in edge_df.iterrows():
    G.add_edge(row["from"], row["to"], weight=row["weight"])
pos = nx.spring_layout(G, seed=42)
node_colors = ["gold" if n in highlight_vars else "lightblue" for n in G.nodes()]
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=8, arrows=True)
plt.title("CASTLE-style graph on IHDP")
plt.tight_layout()
plt.show()

pred_test = castle_fit.predict(Z_test)
print("Test predictions shape:", pred_test.shape)"""
        ),
        md("### Validation (DAGMA-style diagnostics)"),
        CASTLE_VALIDATION_HELPERS,
        code(
            """G_castle = adjacency_to_igraph(A_thresholded, VAR_NAMES, threshold=1e-8)
if G_castle.number_of_edges() == 0:
    W_fallback = np.zeros_like(A_est)
    for _, row in edge_df.iterrows():
        i, j = node_labels.index(row["from"]), node_labels.index(row["to"])
        W_fallback[i, j] = row["weight"]
    G_castle = adjacency_to_igraph(W_fallback, VAR_NAMES, threshold=1e-8)

d_n, e_n = G_castle.number_of_nodes(), G_castle.number_of_edges()
density = e_n / (d_n * (d_n - 1)) if d_n > 1 else 0
print("Graph statistics (CASTLE)")
print({"nodes": d_n, "edges": e_n, "density": round(density, 4), "isDAG": nx.is_directed_acyclic_graph(G_castle)})

W_sem = A_est.copy()
np.fill_diagonal(W_sem, 0)
recon_df = pd.DataFrame([sem_reconstruction_metrics(W_sem, Z_test_eval, "CASTLE")]).set_index("model")
print("SEM reconstruction (linear Z @ W, exploratory)")
print(recon_df)

for node in ["T", "Y"]:
    if node in G_castle:
        print(f"CASTLE {node} parents={list(G_castle.predecessors(node))} children={list(G_castle.successors(node))}")

W_ate = A_thresholded.copy()
np.fill_diagonal(W_ate, 0)
if np.abs(W_ate).sum() < 1e-12:
    cut = 1e-6 * max(np.abs(A_est).max(), 1)
    W_ate = np.where(np.abs(A_est) > cut, A_est, 0)
    np.fill_diagonal(W_ate, 0)

ate_df = pd.DataFrame([
    estimate_ate_adjustment(W_ate, VAR_NAMES, train_x, train_t, train_y, test_x, test_potential_y, "CASTLE")
]).set_index("model")
print("ATE (linear adjustment on parents of Y in learned graph, IHDP-style)")
print(ate_df)

dashboard = pd.DataFrame({
    "metric": ["Edges", "Density", "SEM R2", "|ATE error|"],
    "value": [e_n, round(density, 4), recon_df.loc["CASTLE", "global_R2"], ate_df.loc["CASTLE", "ATE_error"]],
})
plt.figure(figsize=(8, 4))
plt.bar(dashboard["metric"], dashboard["value"], color=["#3498db", "#3498db", "#2ecc71", "#e74c3c"])
plt.title("CASTLE validation dashboard (IHDP)")
plt.tight_layout()
plt.show()"""
        ),
        md(
            """## Summary and Conclusions

- **CASTLE (NeurIPS 2020)** ties supervised prediction to a learnable adjacency `A`, selective reconstruction, and a NOTEARS-style acyclicity penalty. This notebook uses **PyDeepCausalML**'s `CASTLE` on IHDP-style matrices with columns $[x_1, \\ldots, x_{25}, T, Y]$.

## Resources (CASTLE and related)

- **Paper (NeurIPS 2020):** [CASTLE: Regularization via Auxiliary Causal Graph Discovery](https://arxiv.org/abs/2009.13180)
- **NOTEARS:** [arXiv:1803.01422](https://arxiv.org/abs/1803.01422)
- **PyDeepCausalML:** `pydeepcausalml.discovery.CASTLE`
- **Companion:** `02_08_05_05_03_04_DeepCausalML_DAGMA_NoCurl.ipynb`"""
        ),
    ]
    return nb(cells)


NOTEBOOKS = {
    "02_08_05_05_03_01_DeepCausalML_NOTEARS.ipynb": build_notears,
    "02_08_05_05_03_02_DeepCausalML_DagGNN.ipynb": build_daggnn,
    "02_08_05_05_03_03_DeepCausalML_GranDag.ipynb": build_grandag,
    "02_08_05_05_03_04_DeepCausalML_DAGMA_NoCurl.ipynb": build_dagma,
    "02_08_05_05_03_05_DeepCausalML_CASTLE.ipynb": build_castle,
}


def main() -> None:
    TUTORIAL.mkdir(parents=True, exist_ok=True)
    for fname, builder in NOTEBOOKS.items():
        path = TUTORIAL / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(builder(), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
