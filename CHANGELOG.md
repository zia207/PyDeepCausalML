# Changelog

## 0.2.0 — 2026-07-06

Major expansion to match `causalDeepNet.R` model coverage and add first-class CPU / GPU / MPS device support.

### Added
- `effect`: `GANITE`, `CEVAE`.
- `generative`: `CausalGAN`, `CausalDiscrepancyVAE`, `IVAE`, `CausalVAE`, `DSCM`.
- `discovery`: `NOTEARSNonlinearMLP`, `NOTEARSNonlinearSobolev`, `DAGGNN`, `DagmaLinear`, `DagmaNonlinearMLP`, `causal_structure_ml` factory.
- `timeseries`: `NeuralGrangerCLSTM`, `NeuralGrangerEconomySRU`, `NeuralRelationalInference`, `CausalTransformer`, `TFTNet`, `CausalLSTM`, `RETAIN`, `InterventionAwareRNN`, `GVAR`, `CausalGNN`, `CUTS`, `DeepSynth`, `CRN`, `GNet`, `DeepSCM`, `DECI`, and factories `neural_granger_model`, `attn_causal_model`, `rnn_causal_model`, `gnn_causal_model`, `counterfactual_model`.
- Device utilities: `resolve_device`, `get_default_device`, `module_to_device` with auto CUDA → MPS → CPU selection.
- 56-test pytest suite including smoke tests for all new model families and device handling.

### Changed
- All estimators accept `device=None` (auto) or explicit `"cpu"`, `"cuda"`, `"mps"`.
- Added `scipy` as a core dependency (required by DAGMA).

## 0.1.0 — 2026-07-05

Initial release, consolidating the Deep Causal ML tutorial-series models into
a tested package.

### Added
- `effect`: `TARNet`, `CFRNet` (MMD balancing), `DragonNet` (targeted
  regularization), `NeuralDML` (cross-fitting + robust confidence intervals).
- `discovery`: `NOTEARSLinear` (augmented-Lagrangian DAG learning),
  `DynoTEARS` (lag-resolved DAG discovery), `CASTLE` (causal-structure-
  regularized prediction).
- `timeseries`: `NeuralGrangerCMLP` (group-LASSO neural Granger),
  `GrangerLSTM` (ablation Granger test), `TCDF` (attention-based temporal
  causal discovery with delay estimation, adapted from the reference
  ADDSTCN/runTCDF implementation), `CausalLSTMForecaster` (counterfactual
  forecasting).
- `generative`: `CausalEGM` (adversarially disentangled latent blocks).
- `datasets`, `metrics` (PEHE, ATE error, SHD, graph P/R/F1, MASE),
  `plotting`, shared `BaseDeepEstimator` infrastructure with seeding, device
  handling, early stopping, and training histories.
- 49-test pytest suite, including ground-truth recovery checks for every
  estimator family.
