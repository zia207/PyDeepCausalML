"""PyDeepCausalML: deep learning for causal inference in PyTorch.

Model families
--------------
- :mod:`pydeepcausalml.effect` — treatment-effect estimation
  (TARNet, CFRNet, DragonNet, NeuralDML, GANITE, CEVAE).
- :mod:`pydeepcausalml.generative` — generative causal models
  (CausalEGM, CausalGAN, CausalDiscrepancyVAE, IVAE, CausalVAE, DSCM).
- :mod:`pydeepcausalml.discovery` — DAG structure learning
  (NOTEARS, DAGMA, DAG-GNN, DynoTEARS, CASTLE).
- :mod:`pydeepcausalml.timeseries` — temporal causal discovery,
  forecasting, and counterfactual models.

All estimators accept ``device=None`` (auto-select CUDA/MPS/CPU) or an
explicit device string such as ``"cpu"``, ``"cuda"``, or ``"mps"``.

Support modules: :mod:`~pydeepcausalml.datasets`,
:mod:`~pydeepcausalml.metrics`, :mod:`~pydeepcausalml.plotting`.
"""

from . import datasets, metrics
from .discovery import (
    CASTLE,
    DAGGNN,
    DagmaLinear,
    DagmaNonlinearMLP,
    DynoTEARS,
    NOTEARSLinear,
    NOTEARSNonlinearMLP,
    NOTEARSNonlinearSobolev,
    causal_structure_ml,
    causal_structure_ml_model_descriptions,
)
from .effect import CFRNet, CEVAE, DragonNet, GANITE, NeuralDML, TARNet
from .generative import (
    CausalDiscrepancyVAE,
    CausalEGM,
    CausalGAN,
    CausalVAE,
    DSCM,
    IVAE,
)
from .timeseries import (
    CRN,
    CUTS,
    CausalGNN,
    CausalLSTM,
    CausalLSTMForecaster,
    CausalTransformer,
    DECI,
    DeepSCM,
    DeepSynth,
    GNet,
    GVAR,
    GrangerLSTM,
    InterventionAwareRNN,
    NeuralGrangerCMLP,
    NeuralGrangerCLSTM,
    NeuralGrangerEconomySRU,
    NeuralRelationalInference,
    RETAIN,
    TCDF,
    TFTNet,
    attn_causal_model,
    counterfactual_model,
    gnn_causal_model,
    make_lagged_sequences,
    neural_granger_model,
    rnn_causal_model,
)
from .utils import get_default_device, resolve_device, set_seed

__version__ = "0.2.0"

__all__ = [
  # Effect
    "TARNet",
    "CFRNet",
    "DragonNet",
    "NeuralDML",
    "GANITE",
    "CEVAE",
    # Generative
    "CausalEGM",
    "CausalGAN",
    "CausalDiscrepancyVAE",
    "IVAE",
    "CausalVAE",
    "DSCM",
    # Discovery
    "NOTEARSLinear",
    "NOTEARSNonlinearMLP",
    "NOTEARSNonlinearSobolev",
    "DAGGNN",
    "DagmaLinear",
    "DagmaNonlinearMLP",
    "DynoTEARS",
    "CASTLE",
    "causal_structure_ml",
    "causal_structure_ml_model_descriptions",
    # Time series — Granger
    "NeuralGrangerCMLP",
    "NeuralGrangerCLSTM",
    "NeuralGrangerEconomySRU",
    "NeuralRelationalInference",
    "neural_granger_model",
    "GrangerLSTM",
    # Time series — attention / RNN / GNN
    "TCDF",
    "CausalTransformer",
    "TFTNet",
    "attn_causal_model",
    "CausalLSTMForecaster",
    "CausalLSTM",
    "RETAIN",
    "InterventionAwareRNN",
    "rnn_causal_model",
    "GVAR",
    "CausalGNN",
    "CUTS",
    "gnn_causal_model",
    # Time series — counterfactual / SCM
    "DeepSynth",
    "CRN",
    "GNet",
    "counterfactual_model",
    "DeepSCM",
    "DECI",
    "make_lagged_sequences",
    # Utils
    "datasets",
    "metrics",
    "set_seed",
    "resolve_device",
    "get_default_device",
    "__version__",
]
