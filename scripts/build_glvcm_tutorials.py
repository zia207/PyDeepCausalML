#!/usr/bin/env python3
"""Build GLVCM tutorial notebooks (02_08_05_05_02_*) from R-Tutorial qmd sources."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R_TUTORIAL = ROOT / "R-Tutorial"
OUT_DIR = ROOT / "Tutorial"

NB_META = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
}


def md(text: str) -> dict:
    lines = [ln + "\n" for ln in text.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(text: str) -> dict:
    lines = [ln + "\n" for ln in text.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }


def nb(*cells) -> dict:
    return {**NB_META, "cells": list(cells)}


def fix_md_prose(text: str) -> str:
    text = re.sub(r"\{\.unnumbered\}", "", text)
    text = re.sub(r'\{width="[^"]*"\}', "", text)
    text = text.replace("{RCausalML}", "`pydeepcausalml`")
    text = text.replace("RCausalML", "pydeepcausalml")
    text = text.replace("in R,", "in Python,")
    text = text.replace("in R ", "in Python ")
    text = text.replace("Implementation in R", "Implementation in Python")
    text = text.replace("R packages", "Python packages")
    text = text.replace("![](..Image/", "![](../Image/")
    text = text.replace("![](Image/", "![](../Image/")
    text = text.replace("![](../Images/", "![](../Image/")
    text = re.sub(r"\[\]\([^)]+\)", "`pydeepcausalml`", text)
    return text


def prose_from_qmd(path: Path, skip_r_blocks: bool = True) -> list[str]:
    """Return markdown sections from a qmd (text outside ```{r} blocks)."""
    raw = path.read_text(encoding="utf-8")
    parts = re.split(r"```\{r\}[\s\S]*?```", raw) if skip_r_blocks else [raw]
    sections = []
    for part in parts:
        part = part.strip()
        if not part or part.startswith("```"):
            continue
        part = fix_md_prose(part)
        sections.append(part)
    return sections


INSTALL = """import importlib
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "numpy", "pandas", "scipy", "torch", "scikit-learn",
    "matplotlib", "seaborn",
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
    repo_root = Path("..").resolve()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)])

print("Packages ready.")"""

IMPORTS = """import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pydeepcausalml import set_seed

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())"""

IHDP_HELPERS = '''
def load_ihdp(replications: int = 2, random_state: int = 1):
    """Load IHDP semi-synthetic benchmark (CausalML format)."""
    base_url = "https://raw.githubusercontent.com/uber/causalml/master/docs/examples/data"
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"X{i}" for i in range(25)]
    parts = []
    for i in range(1, 10):
        url = f"{base_url}/ihdp_npci_{i}.csv"
        tmp = pd.read_csv(url, header=None)
        tmp.columns = cols[: tmp.shape[1]]
        parts.append(tmp)
    df = pd.concat(parts, ignore_index=True)
    if replications > 1:
        df = pd.concat([df] * replications, ignore_index=True)
    perm = list(range(7, 25)) + list(range(6))
    xcols = [f"X{i}" for i in range(25)]
    X = df[xcols].to_numpy(dtype=float)[:, perm]
    treatment = df["treatment"].to_numpy(dtype=int)
    y = df["y_factual"].to_numpy(dtype=float)
    mu0 = df["mu0"].to_numpy(dtype=float)
    mu1 = df["mu1"].to_numpy(dtype=float)
    tau = np.where(
        treatment == 1,
        df["y_factual"] - df["y_cfactual"],
        df["y_cfactual"] - df["y_factual"],
    )
    n = len(X)
    rng = np.random.default_rng(random_state)
    val_idx = rng.choice(n, size=int(0.2 * n), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return df, X, treatment, y, tau, mu0, mu1, train_idx, val_idx


def preprocess_ihdp_features(train_x, test_x, cont_cols=None):
    """Scale continuous covariates (cols 19–24 after binary-first permutation)."""
    cont_cols = cont_cols or list(range(19, 25))
    train = train_x.copy()
    test = test_x.copy()
    means = train[:, cont_cols].mean(axis=0)
    sds = train[:, cont_cols].std(axis=0)
    sds[sds == 0] = 1.0
    train[:, cont_cols] = (train[:, cont_cols] - means) / sds
    test[:, cont_cols] = (test[:, cont_cols] - means) / sds
    return train, test


def synthetic_ihdp_fallback(n=5000, p=25, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    lin = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.15 * X[:, 2]
    tau = 0.5 + 0.2 * np.tanh(X[:, 4])
    mu0 = lin + 0.1 * X[:, 6] ** 2
    mu1 = mu0 + tau
    ps = 1 / (1 + np.exp(-(0.2 * X[:, 0] - 0.1 * X[:, 1])))
    t = rng.binomial(1, ps)
    y = np.where(t == 1, mu1, mu0) + rng.normal(scale=1.0, size=n)
    perm = list(range(7, 25)) + list(range(6))
    return X[:, perm], t, y, mu0, mu1, tau
'''


def build_intro():
    sections = prose_from_qmd(
        R_TUTORIAL / "02-08-05-05-02-00-DeepCausalML-generative-latent-variable-causal-models-introduction-r.qmd"
    )
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    for sec in sections:
        # Skip duplicate leading banner from qmd source
        sec = re.sub(r"^!\[Banner\]\(\.\./Image/02_DeepCuaslaML\.png\)\s*", "", sec.strip())
        if sec.strip():
            cells.append(md(sec))
    return nb(*cells)


def add_prose(cells, sections, skip_last=False):
    end = len(sections) - 1 if skip_last and sections else len(sections)
    for sec in sections[:end]:
        if sec.strip():
            cells.append(md(sec))


def build_causalvae():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-01-DeepCausalML-causalVAE.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe fit **CausalVAE** with `pydeepcausalml.generative.CausalVAE` on synthetic data with a known causal chain."),
        md("### Check and Install Required Python Packages\n\n`numpy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `pydeepcausalml`"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS),
        code("set_seed(42)"),
        md("### Data generation\n\nSynthetic causal chain $z_1 \\to z_2 \\to z_3$ with nonlinear observations."),
        code(
            '''def generate_data(n_samples=5000, latent_dim=3, random_state=42):
    rng = np.random.default_rng(random_state)
    eps = rng.normal(size=(n_samples, latent_dim))
    z = np.zeros((n_samples, latent_dim))
    x = np.zeros((n_samples, latent_dim))
    z[:, 0] = eps[:, 0]
    z[:, 1] = z[:, 0] + eps[:, 1]
    z[:, 2] = z[:, 1] ** 2 + eps[:, 2]
    x[:, 0] = z[:, 0] * z[:, 2]
    x[:, 1] = np.sin(z[:, 1]) + z[:, 0]
    x[:, 2] = z[:, 2] ** 2 + rng.normal(scale=0.1, size=n_samples)
    return x, z, eps


x, true_z, true_eps = generate_data()
x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-8
x_norm = (x - x_mean) / x_std

n = len(x_norm)
train_n, val_n = int(0.8 * n), int(0.1 * n)
x_train = x_norm[:train_n]
x_val = x_norm[train_n : train_n + val_n]
x_test = x_norm[train_n + val_n :]
print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")'''
        ),
        md("### Fit CausalVAE"),
        code(
            '''from pydeepcausalml.generative import CausalVAE

model = CausalVAE(
    latent_dim=3,
    hidden=64,
    beta_kl=1.0,
    lambda_dag=0.1,
    epochs=80,
    batch_size=128,
    lr=1e-3,
    random_state=42,
    verbose=True,
    log_every=10,
)
model.fit(x_train)
z_latent = model.transform(x_test)
print("Learned latent shape:", z_latent.shape)
adj = model.adjacency_matrix()
print("Adjacency (edge weights):\\n", np.round(adj, 3))'''
        ),
        md("### Training loss"),
        code(
            '''hist = model.history_
if hist.get("loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist["loss"]) + 1), hist["loss"], label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CausalVAE: training loss")
    plt.legend()
    plt.tight_layout()
    plt.show()'''
        ),
        md("### ATE estimation via latent interventions\n\nWe extend the setting with treatment $T$ and outcome $Y$ on the synthetic SCM."),
        code(
            '''def generate_data_with_treatment(n_samples=5000, latent_dim=3, random_state=42):
    rng = np.random.default_rng(random_state)
    eps = rng.normal(size=(n_samples, latent_dim))
    z = np.zeros((n_samples, latent_dim))
    z[:, 0] = eps[:, 0]
    z[:, 1] = z[:, 0] + eps[:, 1]
    z[:, 2] = z[:, 1] ** 2 + eps[:, 2]
    T = rng.binomial(1, 0.5, size=n_samples)
    Y = z[:, 2] + T * 0.5 * z[:, 1] + rng.normal(scale=0.1, size=n_samples)
    x = np.zeros((n_samples, latent_dim))
    x[:, 0] = z[:, 0] * z[:, 2]
    x[:, 1] = np.sin(z[:, 1]) + z[:, 0]
    x[:, 2] = z[:, 2] ** 2 + rng.normal(scale=0.1, size=n_samples)
    return x, z, T, Y


x_ate, z_ate, T_ate, Y_ate = generate_data_with_treatment()
x_ate = (x_ate - x_ate.mean(0)) / (x_ate.std(0) + 1e-8)
true_ate = 0.5 * z_ate[:, 1].mean()
print(f"True ATE (population): {true_ate:.4f}")

# Joint model: CausalVAE representation + linear outcome head on latents
from sklearn.linear_model import LinearRegression

cvae_ate = CausalVAE(latent_dim=3, hidden=64, epochs=60, batch_size=256, verbose=False, random_state=42)
cvae_ate.fit(x_ate)
z_hat = cvae_ate.transform(x_ate)
outcome_model = LinearRegression().fit(np.column_stack([z_hat, T_ate]), Y_ate)

rng = np.random.default_rng(0)
z_samples = rng.normal(size=(500, cvae_ate.latent_dim))
y1 = outcome_model.predict(np.column_stack([z_samples, np.ones(500)]))
y0 = outcome_model.predict(np.column_stack([z_samples, np.zeros(500)]))
pred_ate = y1.mean() - y0.mean()
print(f"Predicted ATE: {pred_ate:.4f}")
print(f"Absolute error: {abs(pred_ate - true_ate):.4f}")'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


def build_ivae():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-02-DeepCausalML-CausaliVAEs.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe use `pydeepcausalml.generative.IVAE` on synthetic multi-class data."),
        md("### Check and Install Required Python Packages"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS),
        code("set_seed(42)"),
        md("### Synthetic multi-class data"),
        code(
            '''n_samples, input_dim, n_aux = 400, 13, 3
rng = np.random.default_rng(42)
n_per = [n_samples // n_aux] * n_aux
n_per[-1] += n_samples - sum(n_per)

X_list, y_list = [], []
for k in range(n_aux):
    mean_k = rng.normal(size=input_dim) * 2 + k * 3
    cov = np.eye(input_dim) * 0.5 + rng.uniform(0, 0.1, (input_dim, input_dim))
    cov = (cov + cov.T) / 2
    X_k = rng.multivariate_normal(mean_k, cov, size=n_per[k])
    X_list.append(X_k)
    y_list.append(np.full(n_per[k], k))

X_raw = np.vstack(X_list)
y = np.concatenate(y_list)
X = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

train_idx, test_idx = [], []
for cls in np.unique(y):
    idx = np.where(y == cls)[0]
    n_test = int(0.2 * len(idx))
    test_samp = rng.choice(idx, size=n_test, replace=False)
    train_idx.extend(np.setdiff1d(idx, test_samp))
    test_idx.extend(test_samp)

x_train = X[train_idx]
x_test = X[test_idx]
u_train = pd.get_dummies(y[train_idx], dtype=float).to_numpy()
u_test = pd.get_dummies(y[test_idx], dtype=float).to_numpy()
print(f"Synthetic: {X.shape[0]} samples, {input_dim} features, {n_aux} classes")'''
        ),
        md("### Fit iVAE"),
        code(
            '''from pydeepcausalml.generative import IVAE

ivae = IVAE(
    latent_dim=4,
    hidden=128,
    beta_kl=1.0,
    epochs=60,
    batch_size=32,
    lr=1e-3,
    random_state=42,
    verbose=True,
    log_every=15,
)
ivae.fit(x_train, u_train)
z_test = ivae.transform(x_test)
print("Test latent shape:", z_test.shape)'''
        ),
        md("### Training loss"),
        code(
            '''hist = ivae.history_
if hist.get("loss"):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(hist["loss"]) + 1), hist["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO")
    plt.title("iVAE training loss")
    plt.tight_layout()
    plt.show()'''
        ),
        md("### Latent space structure"),
        code(
            '''plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for k, label in enumerate(np.unique(y[test_idx])):
    mask = y[test_idx] == label
    plt.scatter(z_test[mask, 0], z_test[mask, 1], s=12, alpha=0.7, label=f"Class {label}")
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("Learned latent space")
plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
for k, label in enumerate(np.unique(y[test_idx])):
    mask = y[test_idx] == label
    plt.scatter(x_test[mask, 0], x_test[mask, 1], s=12, alpha=0.7, label=f"Class {label}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Input space (first 2 features)")
plt.tight_layout()
plt.show()'''
        ),
        md("### Latent–feature correlation heatmap"),
        code(
            '''n_show = min(5, input_dim)
corr = np.corrcoef(z_test.T, x_test[:, :n_show].T)[: z_test.shape[1], z_test.shape[1] :]
plt.figure(figsize=(7, 4))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    xticklabels=[f"F{j+1}" for j in range(n_show)],
    yticklabels=[f"z{j+1}" for j in range(z_test.shape[1])],
)
plt.title("Pearson correlation: latent dims vs input features")
plt.tight_layout()
plt.show()'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


def build_cdvae():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-03-DeepCausalML-causaldiscrepancyVAE-r.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe fit **CausalDiscrepancyVAE** with `pydeepcausalml.generative.CausalDiscrepancyVAE` on IHDP."),
        md("### Check and Install Required Python Packages"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS + "\nfrom sklearn.linear_model import LogisticRegression\n"),
        code("set_seed(42)\nrun_fast = True"),
        md("### Load IHDP data"),
        code(
            IHDP_HELPERS
            + """
try:
    _, X, treatment, y, tau, mu0, mu1, train_idx, val_idx = load_ihdp(
        replications=2 if run_fast else 10
    )
except Exception:
    X, treatment, y, mu0, mu1, tau = synthetic_ihdp_fallback()
    n = len(X)
    rng = np.random.default_rng(1)
    val_idx = rng.choice(n, size=int(0.2 * n), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

X_train, X_val = X[train_idx], X[val_idx]
t_train, t_val = treatment[train_idx], treatment[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
tau_val = tau[val_idx]
mu0_val, mu1_val = mu0[val_idx], mu1[val_idx]
print("Train:", len(train_idx), "| Val:", len(val_idx))"""
        ),
        md("### Propensity score overlap"),
        code(
            '''from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression(max_iter=1000).fit(X_train, t_train)
ps_train = ps_model.predict_proba(X_train)[:, 1]

plt.figure(figsize=(7, 4))
for label, mask, color in [(0, t_train == 0, "#185FA5"), (1, t_train == 1, "#993C1D")]:
    sns.kdeplot(ps_train[mask], fill=True, alpha=0.35, color=color, label=["Control", "Treated"][label])
plt.xlabel("P(T=1 | X)")
plt.ylabel("Density")
plt.title("Propensity score overlap (train)")
plt.legend()
plt.tight_layout()
plt.show()'''
        ),
        md("### Feature scaling and subsampling"),
        code(
            '''X_train_s, X_val_s = preprocess_ihdp_features(X_train, X_val)
rng = np.random.default_rng(42)
sub_n = min(5000 if run_fast else len(X_train_s), len(X_train_s))
sub_idx = rng.choice(len(X_train_s), size=sub_n, replace=False)
X_tr = X_train_s[sub_idx]
t_tr = t_train[sub_idx]
y_tr = y_train[sub_idx]
print("Train matrix:", X_tr.shape)'''
        ),
        md("### Fit CausalDiscrepancyVAE"),
        code(
            '''from pydeepcausalml.generative import CausalDiscrepancyVAE

cdvae = CausalDiscrepancyVAE(
    latent_dim=16,
    hidden=128,
    beta_kl=1.0,
    beta_mmd=0.5,
    epochs=40 if run_fast else 100,
    batch_size=128,
    lr=1e-3,
    random_state=42,
    verbose=True,
    log_every=10,
)
cdvae.fit(X_tr, t_tr, y_tr)'''
        ),
        md("### ATE and PEHE on validation set"),
        code(
            '''ite_hat = cdvae.predict_cate(X_val_s)
ite_true = mu1_val - mu0_val
ate_pred, ate_true = ite_hat.mean(), ite_true.mean()
pehe = np.sqrt(np.mean((ite_hat - ite_true) ** 2))
results = pd.DataFrame({
    "Metric": ["True ATE", "Predicted ATE", "ATE bias", "Abs. ATE error", "sqrt(PEHE)"],
    "Value": [ate_true, ate_pred, ate_pred - ate_true, abs(ate_pred - ate_true), pehe],
})
display(results.round(4))'''
        ),
        md("### ITE scatter and distribution"),
        code(
            '''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(ite_true, ite_hat, alpha=0.3, s=10, color="#534AB7")
axes[0].plot([ite_true.min(), ite_true.max()], [ite_true.min(), ite_true.max()], "k--")
axes[0].axhline(ate_pred, color="#BA7517", ls=":")
axes[0].axvline(ate_true, color="#0F6E56", ls=":")
axes[0].set_xlabel("True ITE")
axes[0].set_ylabel("Predicted ITE")
axes[0].set_title("Predicted vs true ITE")

axes[1].hist(ite_true, bins=40, alpha=0.5, label="True ITE", color="#185FA5")
axes[1].hist(ite_hat, bins=40, alpha=0.5, label="Predicted ITE", color="#993C1D")
axes[1].axvline(ate_true, color="#0F6E56", ls="--")
axes[1].legend()
axes[1].set_title("ITE distribution")
plt.tight_layout()
plt.show()'''
        ),
        md("### ITE calibration by decile"),
        code(
            '''df_cal = pd.DataFrame({"ite_hat": ite_hat, "ite_true": ite_true})
df_cal["decile"] = pd.qcut(ite_hat, 10, labels=False, duplicates="drop")
dec = df_cal.groupby("decile").agg(mean_pred=("ite_hat", "mean"), mean_true=("ite_true", "mean"))
plt.figure(figsize=(6, 4))
plt.plot(dec["mean_pred"], dec["mean_true"], "o-", color="#534AB7")
mn, mx = dec[["mean_pred", "mean_true"]].min().min(), dec[["mean_pred", "mean_true"]].max().max()
plt.plot([mn, mx], [mn, mx], "k--")
plt.xlabel("Mean predicted ITE (decile)")
plt.ylabel("Mean true ITE (decile)")
plt.title("ITE calibration by decile")
plt.tight_layout()
plt.show()'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


def build_causalgan():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-04-DeepCausalML-causalGAN-r.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe fit **CausalGAN** with `pydeepcausalml.generative.CausalGAN` on IHDP."),
        md("### Check and Install Required Python Packages"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS),
        code("set_seed(42)\nrun_fast = True"),
        md("### Load and preprocess IHDP"),
        code(
            IHDP_HELPERS
            + """
try:
    _, X, treatment, y, tau, mu0, mu1, train_idx, val_idx = load_ihdp(replications=1)
except Exception:
    X, treatment, y, mu0, mu1, tau = synthetic_ihdp_fallback()
    n = len(X)
    rng = np.random.default_rng(42)
    train_idx = rng.choice(n, size=int(0.8 * n), replace=False)
    val_idx = np.setdiff1d(np.arange(n), train_idx)

X_train, X_test = X[train_idx], X[val_idx]
t_train, t_test = treatment[train_idx], treatment[val_idx]
y_train = y[train_idx]
tau_test = mu1[val_idx] - mu0[val_idx]

# Scale first 6 continuous columns (original X0-X5; after perm they are cols 19-24)
cont = list(range(19, 25))
means = X_train[:, cont].mean(0)
sds = X_train[:, cont].std(0)
sds[sds == 0] = 1
X_train[:, cont] = (X_train[:, cont] - means) / sds
X_test[:, cont] = (X_test[:, cont] - means) / sds

rng = np.random.default_rng(42)
sub = rng.choice(len(X_train), size=min(5000, len(X_train)), replace=False)
X_tr, t_tr, y_tr = X_train[sub], t_train[sub], y_train[sub]
print("Train:", X_tr.shape, "| Test:", X_test.shape)"""
        ),
        md("### Fit CausalGAN"),
        code(
            '''from pydeepcausalml.generative import CausalGAN

cg = CausalGAN(
    hidden=64 if run_fast else 128,
    noise_dim=8,
    lambda_lab=0.5,
    epochs=50 if run_fast else 150,
    batch_size=256,
    lr=2e-4,
    random_state=42,
    verbose=True,
    log_every=10,
)
cg.fit(X_tr, t_tr, y_tr)'''
        ),
        md("### Interventional ATE via do(T)"),
        code(
            '''ate_gen = cg.predict_ate(X_test, n_samples=100)
ate_true = tau_test.mean()
print(f"Estimated ATE (do-calculus): {ate_gen:.4f}")
print(f"True ATE: {ate_true:.4f}")'''
        ),
        md("### CATE PEHE on test set"),
        code(
            '''ite_hat = cg.predict_cate(X_test, n_samples=50)
pehe = np.sqrt(np.mean((ite_hat - tau_test) ** 2))
print(f"sqrt(PEHE): {pehe:.4f}")
print(f"Std of true tau: {tau_test.std():.4f}")'''
        ),
        md("### Training loss and ITE scatter"),
        code(
            '''hist = cg.history_
if hist.get("loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist["loss"]) + 1), hist["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CausalGAN training loss")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(6, 5))
n_sc = min(500, len(X_test))
plt.scatter(tau_test[:n_sc], ite_hat[:n_sc], alpha=0.4, s=10, color="#7b5ce0")
mn = min(tau_test[:n_sc].min(), ite_hat[:n_sc].min())
mx = max(tau_test[:n_sc].max(), ite_hat[:n_sc].max())
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("True tau")
plt.ylabel("Estimated tau")
plt.title("CATE: estimated vs true")
plt.tight_layout()
plt.show()'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


def build_dscm():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-05-DeepCausalML--DSCM.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe fit **DSCM** with `pydeepcausalml.generative.DSCM` on IHDP."),
        md("### Check and Install Required Python Packages"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS + "\nfrom sklearn.linear_model import LogisticRegression\n"),
        code("set_seed(42)\nrun_fast = True"),
        md("### Load IHDP and three-way split"),
        code(
            IHDP_HELPERS
            + """
try:
    _, X, treatment, y, tau, mu0, mu1, _, _ = load_ihdp(replications=2 if run_fast else 10)
except Exception:
    X, treatment, y, mu0, mu1, tau = synthetic_ihdp_fallback()

n = len(X)
rng = np.random.default_rng(42)
idx = rng.permutation(n)
train_end, val_end = int(0.7 * n), int(0.85 * n)
tr_i, va_i, te_i = idx[:train_end], idx[train_end:val_end], idx[val_end:]

def grab(i):
    return X[i], treatment[i], y[i], mu0[i], mu1[i]

X_train, t_train, y_train, mu0_tr, mu1_tr = grab(tr_i)
X_val, t_val, y_val, _, _ = grab(va_i)
X_test, t_test, y_test, mu0_te, mu1_te = grab(te_i)

X_train_s, _ = preprocess_ihdp_features(X_train, X_val)
_, X_val_s = preprocess_ihdp_features(X_train, X_val)
_, X_test_s = preprocess_ihdp_features(X_train, X_test)
print(f"Train: {len(tr_i)} | Val: {len(va_i)} | Test: {len(te_i)}")"""
        ),
        md("### Propensity overlap"),
        code(
            '''ps = LogisticRegression(max_iter=1000).fit(X_train_s, t_train).predict_proba(X_train_s)[:, 1]
plt.figure(figsize=(7, 4))
for m, c, lab in [(t_train == 0, "#185FA5", "Control"), (t_train == 1, "#993C1D", "Treated")]:
    sns.kdeplot(ps[m], fill=True, alpha=0.35, color=c, label=lab)
plt.xlabel("P(T=1 | X)")
plt.title("Propensity score overlap (train)")
plt.legend()
plt.tight_layout()
plt.show()'''
        ),
        md("### Fit DSCM"),
        code(
            '''from pydeepcausalml.generative import DSCM

dscm = DSCM(
    hidden=128,
    epochs=60 if run_fast else 120,
    batch_size=128,
    lr=1e-3,
    random_state=42,
    verbose=True,
    log_every=10,
)
dscm.fit(X_train_s, t_train, y_train)'''
        ),
        md("### Potential outcomes and treatment effects"),
        code(
            '''y0_pred, y1_pred = dscm.predict_potential_outcomes(X_test_s)
ite_pred = y1_pred - y0_pred
ite_true = mu1_te - mu0_te
ate_pred, ate_true = ite_pred.mean(), ite_true.mean()
pehe = np.sqrt(np.mean((ite_pred - ite_true) ** 2))
metrics = pd.DataFrame({
    "Metric": ["True ATE", "Predicted ATE", "ATE bias", "Abs. ATE error", "sqrt(PEHE)"],
    "Value": [ate_true, ate_pred, ate_pred - ate_true, abs(ate_pred - ate_true), pehe],
})
display(metrics.round(4))'''
        ),
        md("### Potential outcomes: predicted vs ground truth"),
        code(
            '''plt.figure(figsize=(6, 5))
plt.scatter(mu0_te, y0_pred, alpha=0.45, s=10, color="#185FA5", label="Y(0)")
plt.scatter(mu1_te, y1_pred, alpha=0.45, s=10, color="#993C1D", label="Y(1)")
lims = [min(mu0_te.min(), y0_pred.min()), max(mu1_te.max(), y1_pred.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("Ground-truth potential outcome")
plt.ylabel("Predicted potential outcome")
plt.title("DSCM potential outcomes")
plt.legend()
plt.tight_layout()
plt.show()'''
        ),
        md("### ITE scatter and calibration"),
        code(
            '''fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(ite_true, ite_pred, alpha=0.4, s=10, color="#534AB7")
axes[0].plot([ite_true.min(), ite_true.max()], [ite_true.min(), ite_true.max()], "k--")
axes[0].set_xlabel("True ITE")
axes[0].set_ylabel("Predicted ITE")
axes[0].set_title("ITE: predicted vs true")

df_cal = pd.DataFrame({"ite_true": ite_true, "ite_pred": ite_pred})
df_cal["decile"] = pd.qcut(ite_pred, 10, labels=False, duplicates="drop")
dec = df_cal.groupby("decile").mean(numeric_only=True)
axes[1].plot(dec["ite_pred"], dec["ite_true"], "o-", color="#534AB7")
axes[1].plot([dec["ite_pred"].min(), dec["ite_pred"].max()],
             [dec["ite_pred"].min(), dec["ite_pred"].max()], "k--")
axes[1].set_xlabel("Mean predicted ITE")
axes[1].set_ylabel("Mean true ITE")
axes[1].set_title("Calibration by decile")
plt.tight_layout()
plt.show()'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


def build_causalegm():
    sections = prose_from_qmd(R_TUTORIAL / "02-08-05-05-02-06-DeepCausalML_CausalEGM.qmd")
    cells = [md("![Banner](../Image/02_DeepCuaslaML.png)")]
    add_prose(cells, sections, skip_last=True)
    cells.extend([
        md("## Implementation in Python\n\nWe fit **CausalEGM** with `pydeepcausalml.generative.CausalEGM` on IHDP."),
        md("### Check and Install Required Python Packages"),
        code(INSTALL),
        md("### Verify imports"),
        code(IMPORTS + "\nfrom sklearn.linear_model import LogisticRegression\n"),
        code("set_seed(42)\nrun_fast = True"),
        md("### Load IHDP"),
        code(
            IHDP_HELPERS
            + """
try:
    _, X, treatment, y, tau, mu0, mu1, _, _ = load_ihdp(replications=2 if run_fast else 10)
except Exception:
    X, treatment, y, mu0, mu1, tau = synthetic_ihdp_fallback()

n = len(X)
rng = np.random.default_rng(42)
idx = rng.permutation(n)
train_n, val_n = int(0.7 * n), int(0.85 * n)
tr_i, te_i = idx[:train_n], idx[val_n:]

X_train, t_train, y_train = X[tr_i], treatment[tr_i], y[tr_i]
X_test, mu0_te, mu1_te = X[te_i], mu0[te_i], mu1[te_i]
X_train_s, X_test_s = preprocess_ihdp_features(X_train, X_test)
print(f"Train: {len(tr_i)} | Test: {len(te_i)}")"""
        ),
        md("### Propensity overlap"),
        code(
            '''ps = LogisticRegression(max_iter=1000).fit(X_train_s, t_train).predict_proba(X_train_s)[:, 1]
plt.figure(figsize=(7, 4))
for m, c, lab in [(t_train == 0, "#185FA5", "Control"), (t_train == 1, "#993C1D", "Treated")]:
    sns.kdeplot(ps[m], fill=True, alpha=0.35, color=c, label=lab)
plt.xlabel("P(T=1 | X)")
plt.title("Propensity score overlap (train)")
plt.legend()
plt.tight_layout()
plt.show()'''
        ),
        md("### Fit CausalEGM"),
        code(
            '''from pydeepcausalml.generative import CausalEGM

egm = CausalEGM(
    dim_c=8,
    dim_t=4,
    dim_y=4,
    hidden=128,
    lambda_recon=1.0,
    lambda_treat=2.0,
    lambda_outcome=2.0,
    lambda_disent=0.5,
    epochs=60 if run_fast else 100,
    batch_size=128,
    lr=1e-3,
    random_state=42,
    verbose=True,
    log_every=10,
)
egm.fit(X_train_s, t_train, y_train)'''
        ),
        md("### Training loss components"),
        code(
            '''hist = egm.history_
cols = [c for c in ["loss", "recon", "treat", "outcome", "disent"] if c in hist]
if cols:
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3))
    if len(cols) == 1:
        axes = [axes]
    epochs = range(1, len(hist[cols[0]]) + 1)
    for ax, c in zip(axes, cols):
        ax.plot(epochs, hist[c])
        ax.set_title(c)
        ax.set_xlabel("Epoch")
    plt.suptitle("CausalEGM training losses")
    plt.tight_layout()
    plt.show()'''
        ),
        md("### Treatment effect evaluation"),
        code(
            '''ite_pred = egm.predict_ite(X_test_s)
ite_true = mu1_te - mu0_te
ate_true, ate_pred = ite_true.mean(), ite_pred.mean()
pehe = np.sqrt(np.mean((ite_pred - ite_true) ** 2))
metrics = pd.DataFrame({
    "Metric": ["True ATE", "Predicted ATE", "ATE bias", "Abs. ATE error", "sqrt(PEHE)"],
    "Value": [ate_true, ate_pred, ate_pred - ate_true, abs(ate_pred - ate_true), pehe],
})
display(metrics.round(4))'''
        ),
        md("### Potential outcomes and ITE plots"),
        code(
            '''# Potential outcomes via outcome head at t=0/1
with torch.no_grad():
    zc, _, zy = egm._encode(X_test_s)
    ones = torch.ones(len(zc), 1, device=egm._device)
    zeros = torch.zeros(len(zc), 1, device=egm._device)
    y1 = egm.module_.outcome_head(torch.cat([zc, zy, ones], dim=-1)).squeeze(-1).cpu().numpy() * egm.y_std_ + egm.y_mean_
    y0 = egm.module_.outcome_head(torch.cat([zc, zy, zeros], dim=-1)).squeeze(-1).cpu().numpy() * egm.y_std_ + egm.y_mean_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(mu0_te, y0, alpha=0.45, s=10, color="#185FA5", label="Y(0)")
axes[0].scatter(mu1_te, y1, alpha=0.45, s=10, color="#993C1D", label="Y(1)")
lims = [min(mu0_te.min(), y0.min()), max(mu1_te.max(), y1.max())]
axes[0].plot(lims, lims, "k--")
axes[0].set_xlabel("Ground truth")
axes[0].set_ylabel("Predicted")
axes[0].legend()
axes[0].set_title("Potential outcomes")

axes[1].scatter(ite_true, ite_pred, alpha=0.4, s=10, color="#534AB7")
axes[1].plot([ite_true.min(), ite_true.max()], [ite_true.min(), ite_true.max()], "k--")
axes[1].set_xlabel("True ITE")
axes[1].set_ylabel("Predicted ITE")
axes[1].set_title(f"ITE scatter (PEHE={pehe:.3f})")
plt.tight_layout()
plt.show()'''
        ),
        md("### Learned propensity scores in latent space"),
        code(
            '''ps_latent = egm.predict_propensity(X_train_s)
plt.figure(figsize=(7, 4))
for m, c, lab in [(t_train == 0, "#185FA5", "Control"), (t_train == 1, "#993C1D", "Treated")]:
    sns.kdeplot(ps_latent[m], fill=True, alpha=0.35, color=c, label=lab)
plt.xlabel("P(T=1 | z_c, z_t)")
plt.title("Learned propensity (latent space)")
plt.legend()
plt.tight_layout()
plt.show()'''
        ),
    ])
    if sections:
        cells.append(md(sections[-1]))
    return nb(*cells)


NOTEBOOKS = [
    ("02_08_05_05_02_00_DeepCausalML_GLVCM_introduction.ipynb", build_intro),
    ("02_08_05_05_02_01_DeepCausalML_CausalVAE.ipynb", build_causalvae),
    ("02_08_05_05_02_02_DeepCausalML_iVAE.ipynb", build_ivae),
    ("02_08_05_05_02_03_DeepCausalML_CausalDiscrepancyVAE.ipynb", build_cdvae),
    ("02_08_05_05_02_04_DeepCausalML_CausalGAN.ipynb", build_causalgan),
    ("02_08_05_05_02_05_DeepCausalML_DSCM.ipynb", build_dscm),
    ("02_08_05_05_02_06_DeepCausalML_CausalEGM.ipynb", build_causalegm),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, builder in NOTEBOOKS:
        path = OUT_DIR / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(builder(), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("Wrote", path)


if __name__ == "__main__":
    main()
