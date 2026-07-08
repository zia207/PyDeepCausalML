# PyDeepCausalML

Deep learning models for causal inference in PyTorch: treatment-effect estimation, causal structure learning, temporal causal discovery, and counterfactual forecasting — behind one consistent, sklearn-style API.

PyDeepCausalML consolidates the model implementations developed across the *Deep Causal ML* tutorial series (neural Granger causality, TCDF, DYNOTEARS, CASTLE, TARNet/CFRNet/DragonNet, neural DML, CausalEGM, causal LSTM forecasting, and more) into a tested, installable package. It is the Python counterpart to the `causalDeepNet.R` / RCausalML model library.

## Installation

```bash
pip install pydeepcausalml            # core
pip install "pydeepcausalml[plot]"    # + matplotlib/seaborn plotting helpers
pip install "pydeepcausalml[dev]"     # + test and lint tooling
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. From source:

```bash
git clone https://github.com/zia207/PyDeepCausalML && cd PyDeepCausalML
pip install -e ".[dev]"
pytest
```

Or install the pre-built wheel:

```bash
pip install pydeepcausalml-0.2.0-py3-none-any.whl
```

## Device support (CPU / GPU / MPS)

Every estimator accepts `device=None` (auto-select) or an explicit device string:

```python
from pydeepcausalml import TARNet, resolve_device, get_default_device

print(get_default_device())          # cuda, mps, or cpu
print(resolve_device("cpu"))         # torch.device('cpu')

est = TARNet(device="cuda", epochs=200).fit(X, t, y)   # NVIDIA GPU
est = TARNet(device="mps", epochs=200).fit(X, t, y)    # Apple Silicon
est = TARNet(device="cpu", epochs=200).fit(X, t, y)    # force CPU
```

Auto-selection prefers CUDA, then Apple MPS, then CPU. Acyclicity terms (`matrix_exp`) are always evaluated in float64 on CPU for numerical stability regardless of the training device.

## Model catalog

### Treatment effects (`pydeepcausalml.effect`)

| Estimator | Purpose | Key methods |
|---|---|---|
| `TARNet` | CATE via shared representation + two outcome heads | `predict_cate`, `predict_ate`, `predict_potential_outcomes` |
| `CFRNet` | TARNet + MMD representation balancing | same as TARNet |
| `DragonNet` | Propensity head + targeted regularization (doubly robust) | + `predict_propensity` |
| `NeuralDML` | Cross-fitted double ML for the ATE with robust SEs | `predict_ate`, `confidence_interval` |
| `GANITE` | GAN-based individual treatment effect estimation | `predict_cate`, `predict_ate` |
| `CEVAE` | Causal effect VAE with latent confounder | `predict_cate`, `predict_ate` |

### Generative causal models (`pydeepcausalml.generative`)

| Estimator | Purpose | Key methods |
|---|---|---|
| `CausalEGM` | Encoding generative model for ITE/ATE | `predict_ite`, `predict_ate`, `predict_propensity` |
| `CausalGAN` | GAN with structural causal equations | `predict_cate`, `predict_ate` |
| `CausalDiscrepancyVAE` | Discrepancy VAE for causal representation | `predict_cate`, `predict_ate` |
| `IVAE` | Identifiable VAE (auxiliary-conditioned prior) | `transform`, `reconstruct` |
| `CausalVAE` | Causal VAE with learned DAG over latents | `adjacency_matrix`, `transform` |
| `DSCM` | Deep structural causal model (X → T → Y) | `predict_cate`, `predict_potential_outcomes` |

### Static structure learning (`pydeepcausalml.discovery`)

| Estimator | Purpose | Key methods |
|---|---|---|
| `NOTEARSLinear` | Linear DAG via smooth acyclicity constraint | `get_adjacency` |
| `NOTEARSNonlinearMLP` | Nonlinear NOTEARS with per-node MLPs | `get_adjacency` |
| `NOTEARSNonlinearSobolev` | Sobolev-basis nonlinear NOTEARS | `get_adjacency` |
| `DAGGNN` | VAE-style DAG-GNN | `adjacency_matrix` |
| `DagmaLinear` | DAGMA linear structure learning | `adjacency_matrix` |
| `DagmaNonlinearMLP` | Nonlinear DAGMA with per-node MLPs | `adjacency_matrix` |
| `DynoTEARS` | Lag-resolved DAG discovery for time series | `get_adjacency`, `get_scores` |
| `CASTLE` | Prediction with causal-structure regularization | `predict`, `get_adjacency` |



```python
from pydeepcausalml import causal_structure_ml, causal_structure_ml_model_descriptions

print(causal_structure_ml_model_descriptions())
est = causal_structure_ml("dagma_linear", device="cuda").fit(X)
A = est.adjacency_matrix()
```

Methods: `notears_linear`, `notears_nonlinear_mlp`, `notears_nonlinear_sobolev`, `dag_gnn`, `dagma_linear`, `dagma_nonlinear_mlp`, `dynotears`, `castle`.

### Time series — Granger causality

| Estimator | Purpose | Key methods |
|---|---|---|
| `NeuralGrangerCMLP` | Component-wise MLPs + group LASSO | `get_adjacency`, `get_scores` |
| `NeuralGrangerCLSTM` | LSTM-based neural Granger | `get_adjacency`, `get_scores` |
| `NeuralGrangerEconomySRU` | Economy-SRU neural Granger | `get_adjacency`, `get_scores` |
| `NeuralRelationalInference` | Relational inference for Granger graphs | `adjacency_matrix` |
| `GrangerLSTM` | Full-vs-reduced LSTM ablation Granger test | `get_adjacency`, `get_scores` |

Factory (mirrors R `neural_granger_ml()`): `neural_granger_model("cmlp" | "clstm" | "economysru" | "nri", ...)`.

### Time series — attention / RNN / GNN

| Estimator | Purpose | Key methods |
|---|---|---|
| `TCDF` | Attention-based temporal causal discovery **with delays** | `discovered_edges`, `summary`, `get_adjacency` |
| `CausalTransformer` | Transformer for temporal causal discovery | `get_adjacency` |
| `TFTNet` | Temporal Fusion Transformer causal model | `predict` |
| `CausalLSTM` | Causal LSTM for time-series modeling | `predict` |
| `CausalLSTMForecaster` | Counterfactual multi-step forecasting | `forecast`, `forecast_counterfactual`, `estimate_effect` |
| `RETAIN` | RETAIN attention for interpretable forecasting | `predict` |
| `InterventionAwareRNN` | RNN with explicit intervention encoding | `predict` |
| `GVAR` | Graph VAR for multivariate time series | `causal_matrix` |
| `CausalGNN` | GNN-based temporal causal discovery | `causal_matrix` |
| `CUTS` | Causal discovery from unstructured time series | `causal_matrix` |


### Time series — counterfactual / SCM

| Estimator | Purpose | Key methods |
|---|---|---|
| `DeepSynth` | Deep synthetic control / counterfactual synthesis | `predict_counterfactual` |
| `CRN` | Counterfactual recurrent network | `predict_ite` |
| `GNet` | G-Net counterfactual estimator | `predict_ite` |
| `DeepSCM` | Deep structural causal model for time series | `intervene` |
| `DECI` | Deep end-to-end causal inference | `adjacency_matrix` |

Factory (mirrors R `counterfactual_model()`): `counterfactual_model("deepsynth" | "crn" | "gnet", ...)`.

Support modules: `datasets` (synthetic benchmarks with ground truth), `metrics` (PEHE, ATE error, graph precision/recall/F1/SHD, MASE), `plotting` (causal graphs, score heatmaps, training curves).

## Quickstart

### 1. Treatment-effect estimation

```python
from pydeepcausalml import TARNet, DragonNet, NeuralDML, GANITE, CEVAE
from pydeepcausalml.datasets import make_confounded_data
from pydeepcausalml.metrics import pehe

df = make_confounded_data(n=5000, random_state=0)
X = df[["age", "education", "prior_income"]].values
t, y = df["treatment"].values, df["outcome"].values

est = DragonNet(epochs=200, random_state=0).fit(X, t, y)
print("ATE:", est.predict_ate(X))
print("PEHE:", pehe(df["true_cate"], est.predict_cate(X)))

dml = NeuralDML(n_splits=2, random_state=0).fit(X, t, y)
print("DML ATE:", dml.predict_ate(), "95% CI:", dml.confidence_interval())

ganite = GANITE(epochs=200, device="cuda", random_state=0).fit(X, t, y)
print("GANITE PEHE:", pehe(df["true_cate"], ganite.predict_cate(X)))
```

### 2. Temporal causal discovery with TCDF

```python
import pandas as pd
from pydeepcausalml import TCDF

series = pd.read_csv("my_timeseries.csv")     # one column per series
model = TCDF(kernel_size=4, epochs=1000, random_state=0).fit(series)

print(model.summary())          # cause -> effect with estimated delay
A = model.get_adjacency()       # binary adjacency: A[i, j] means X_j -> X_i
```

TCDF trains an attention-augmented dilated depthwise TCN per target, selects candidate causes from the attention scores, validates each with permutation importance, and reads causal delays off the kernel weights — following Nauta, Bucur & Seifert (2019).

### 3. Causal graph recovery on time series

```python
from pydeepcausalml import NeuralGrangerCMLP, DynoTEARS, causal_structure_ml
from pydeepcausalml.datasets import make_var_data
from pydeepcausalml.metrics import graph_recovery_metrics

X, A_true = make_var_data(n_steps=2000, random_state=0)

cmlp = NeuralGrangerCMLP(lag=5, lambda_group=0.01, random_state=0).fit(X)
dyno = causal_structure_ml("dynotears", lag=5, random_state=0).fit(X)

for name, est in [("cMLP", cmlp), ("DYNOTEARS", dyno)]:
    print(name, graph_recovery_metrics(A_true, est.get_adjacency()))
```

### 4. Counterfactual forecasting

```python
import numpy as np
from pydeepcausalml import CausalLSTMForecaster

# outcome_hist: (units, seq_len), treat_hist: (units, seq_len), future: (units, 12)
model = CausalLSTMForecaster(pred_len=12, random_state=0)
model.fit(outcome_hist, treat_hist, future)

effect = model.estimate_effect(
    outcome_hist,
    factual_treatment=treat_hist,
    counterfactual_treatment=np.ones_like(treat_hist),   # do(T = 1)
)
print("Mean effect over horizon:", effect.mean())
```

### 5. Generative causal models

```python
from pydeepcausalml import CausalEGM, CausalGAN, IVAE, CausalVAE

egm = CausalEGM(epochs=100, random_state=0).fit(X, t, y)
print("ATE:", egm.predict_ate(X))

ivae = IVAE(epochs=100, device="cpu").fit(X, t.reshape(-1, 1))
z = ivae.transform(X[:100])

cvae = CausalVAE(epochs=100, latent_dim=4).fit(X)
print(cvae.adjacency_matrix())
```

### 6. Visualization

```python
import matplotlib.pyplot as plt
from pydeepcausalml.plotting import plot_causal_graph, plot_score_heatmap

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_causal_graph(model.get_adjacency(), node_names=series.columns,
                  delays=model.delays_, ax=axes[0])
plot_score_heatmap(model.get_scores(), node_names=series.columns, ax=axes[1])
plt.show()
```

## Conventions

- **Adjacency orientation.** Everywhere in the package, `A[i, j] = 1` means *X_j causes X_i* (rows = effects/targets, columns = causes/sources).
- **sklearn-style API.** Constructors store hyperparameters; `fit` returns `self`; predictions take and return NumPy arrays; `history_` records per-epoch diagnostics; `get_params`/`set_params` are supported.
- **Reproducibility.** Pass `random_state=` to any estimator, or call `pydeepcausalml.set_seed`.
- **Devices.** `device=None` auto-selects CUDA → MPS → CPU; pass `device="cpu"` to force CPU. See [Device support](#device-support-cpu--gpu--mps) above.
- **Scaling.** Covariates and outcomes are standardized internally by default (`standardize=False` / `standardize_outcome=False` to disable); predictions are returned in original units.
- **R parity.** Factory functions (`causal_structure_ml`, `neural_granger_model`, `attn_causal_model`, `rnn_causal_model`, `gnn_causal_model`, `counterfactual_model`) mirror the corresponding `causalDeepNet.R` entry points.

## Validating before you trust

Every estimator here can be checked on simulated data with known ground truth before being applied to real data — the `datasets` module exists for exactly that workflow:

```python
from pydeepcausalml.datasets import make_confounded_data, make_var_data, make_intervention_series
```

Effect estimators should recover the known ATE within noise; discovery methods should score well on `graph_recovery_metrics`. If they don't at your data's scale and noise level, tune before drawing conclusions.

## Method selection guide

| Question | Reach for |
|---|---|
| "What was the effect of the treatment?" (i.i.d. data) | `TARNet` → `CFRNet` (imbalance) → `DragonNet` (doubly robust) |
| "GAN / VAE-based ITE with latent confounding" | `GANITE`, `CEVAE` |
| "I only need the ATE with a confidence interval" | `NeuralDML` |
| "Which variables cause which?" (static data) | `NOTEARSLinear`, `DAGGNN`, `DagmaLinear`, `CASTLE` |
| "Which series drives which, and at what lag?" | `TCDF` (delays), `DynoTEARS` (DAG guarantee), `NeuralGrangerCMLP` (fast) |
| "What would this series look like under do(T)?" | `CausalLSTMForecaster`, `counterfactual_model("crn")` |
| "Latent confounding, want ITEs + propensities" | `CausalEGM`, `CausalGAN` |
| "Identifiable / causal representations" | `IVAE`, `CausalVAE`, `CausalDiscrepancyVAE` |
| "Graph neural networks on multivariate series" | `GVAR`, `CausalGNN`, `CUTS` |

## References

- Shalit, Johansson & Sontag (2017). *Estimating individual treatment effect: generalization bounds and algorithms.* ICML. (TARNet/CFRNet)
- Shi, Blei & Veitch (2019). *Adapting neural networks for the estimation of treatment effects.* NeurIPS. (DragonNet)
- Chernozhukov et al. (2018). *Double/debiased machine learning.* Econometrics Journal. (DML)
- Yoon, Jordon & van der Schaar (2018). *GANITE: Estimation of individualized treatment effects using generative adversarial nets.* ICLR. (GANITE)
- Louizos et al. (2017). *Causal effect inference with deep latent-variable models.* NeurIPS. (CEVAE)
- Zheng et al. (2018). *DAGs with NO TEARS.* NeurIPS. (NOTEARS)
- Bello et al. (2022). *DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization.* NeurIPS. (DAGMA)
- Yu et al. (2019). *DAG-GNN: DAG structure learning with graph neural networks.* ICML.
- Pamfil et al. (2020). *DYNOTEARS: structure learning from time-series data.* AISTATS.
- Kyono, Zhang & van der Schaar (2020). *CASTLE: regularization via auxiliary causal graph discovery.* NeurIPS.
- Tank et al. (2021). *Neural Granger causality.* IEEE TPAMI. (cMLP/cLSTM)
- Nauta, Bucur & Seifert (2019). *Causal discovery with attention-based convolutional neural networks.* MAKE. (TCDF)
- Liu et al. (2022). *CausalEGM: a general causal inference framework by encoding generative modeling.*
- Khemakhem et al. (2020). *Variational autoencoders and nonlinear ICA: A unifying framework.* AISTATS. (iVAE)

## License

MIT — see [LICENSE](LICENSE).
