# TRUE-X

TRUE-X is a decision support tool for explainable AI evaluation on multivariate time series. It combines a Streamlit demo interface with the [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) metrics library.

## Repository Structure

```
TRUE-X/
├── app/                    # Streamlit application (entry: frontend.py)
├── data/                   # Dataset folder
├── experiments/            # Experiment pipeline
│   ├── configs/            # Dataset, model and model-dataset configurations
│   ├── datasets/           # Dataset adapters (windowing, splits, standardization)
│   ├── ml_models/          # Model architectures and training wrapper
│   ├── tuning/             # Optuna tuning
│   ├── run_xai_evaluation.py, xai_methods.py
│   ├── analysis/           # PROMETHEE ranking, Table 5, Figures 4 to 6, Table 6, app input
│   └── appendix/           # Appendix C.1 (explainer settings), C.3 (ranking uncertainty)
├── results/                # Benchmark and appendix results
├── saved_models/           # Pre-trained model weights (.pt + .json)
├── submodules/
│   ├── explitest/          # ExpliTest metrics library (git submodule)
│   └── PipelineVis/        # Pipeline visualisation component (git submodule)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> **Note:** `data/` is not versioned in this repository (the folder needs to be created). See the *Data & Models* section for download links.

## Data & Models

`data/` is not versioned in this repository. Download each dataset and place it at the expected path under `data/`:

| Dataset | Task | Expected path | Source |
|---------|------|---------------|--------|
| C-MAPSS FD001 | Regression (RUL) | `data/CMAPPS_Dataset/` | [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| CWRU Bearing | Classification | `data/CWRU/` | [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter/download-data) |
| Hydraulic Systems | Classification | `data/hydraulic_systems/` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| E-coating | Regression | `data/E-coating/` | [Kaggle — process-data-for-predictive-maintenance](https://www.kaggle.com/datasets/boyangs444/process-data-for-predictive-maintenance) |

Pre-trained model weights are in `saved_models/` and versioned in this repository. The single exception is `CWRU_12k_ENCODER.pt` (140 MB), which exceeds GitHub's file size limit and must be downloaded separately:

1. Download `CWRU_12k_ENCODER.pt` from the [Releases](../../releases) page.
2. Place it at `saved_models/CWRU_12k_ENCODER.pt`.

The following result files are larger than 50 MB and are available as a release asset:

| File | Place it at |
|---|---|
| `final_main_benchmark_primary_scope.csv` (80 MB) | `results/main_benchmark_14_merged/` |
| `final_main_benchmark_all_samples.csv` (83 MB) | `results/main_benchmark_14_merged/` |
| `final_main_benchmark_FD001.csv` (59 MB) | `results/main_benchmark_14_merged/` |
| `final_main_benchmark_all_samples_long.csv` (600 MB) | `results/main_benchmark_14_merged/` |
| `uncertainty_scores_20261016.tar.gz` (878 MB) | `results/r1_9_uncertainty_20261016/` |


## Getting Started

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repository-url>
cd TRUE-X
```

Or, if already cloned:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e submodules/explitest
```

### 3. Run the Streamlit app

```bash
streamlit run app/frontend.py
```

The app will open at `http://localhost:8501`. The *Browse Benchmark* mode reads `results/main_benchmark_14_with_performance.csv`.

### 4. Reproduce experiments (optional)

```bash
pip install -r experiments/requirements.txt
cd experiments
python run_xai_evaluation.py
```

## Benchmark

The benchmark covers 4 datasets, 9 models per dataset and 14 explainers (504 configurations, of which 485 are supported; `results/main_benchmark_14_merged/support_matrix.csv`).

| Task | Datasets | Models (`experiments/configs/model_dataset_map.yaml`) |
|---|---|---|
| Regression | C-MAPSS FD001, Ecoating | LSTM, BiLSTM, Attention-LSTM, CNN-LSTM, LSTM-FCN, TCN, TST, TFT, Linear |
| Classification | Hydraulic, CWRU | InceptionTime, ResNet, FCN, Time-CNN, Encoder, TST, LSTM, GRU, Logistic |

Explainers (registry keys of `ExplainerFactory` in `experiments/xai_methods.py`; settings in `run_xai_evaluation.py::compute_attributions_for_method`): Saliency (`saliency`), GradX (`gradient_x_input`), GuidedBP (`guided_back_prop`), SmoothGrad (`smooth_gradient`), IntegGrad (`integrated_gradients`), ExpectGrad (`expected_gradients`), DeepLift (`deeplift`), DeepSHAP (`deepliftshap`), GradSHAP (`gradientshap`), Occlusion (`occlusion`), FeatAbl (`feature_ablation`), LIME (`lime_tabular`), SHAP (`shap_permutation`, `shap.Explainer` with 2D+1 evaluations) and KernelSHAP (`kernelshap`, `shap.KernelExplainer` with 2D+2048 coalition samples, no l1 regularization).

### Trustworthiness metrics

The eight metrics are implemented in ExpliTest (`submodules/explitest`) and instantiated in `run_xai_evaluation.py::_build_xai_metrics`. All metrics use the attribution magnitudes |e|; classification uses the score of the class predicted on the clean input.

- **Faithfulness Correlation (FC):** Spearman correlation, over 100 random subsets of q = 10 input elements, between the attribution mass of the subset and the absolute change of the model output when the subset is removed. Subsets are drawn with a fixed seed.
- **Pixel Flipping (PF):** area under the deletion curve, removing the elements in decreasing order of |e| (trapezoidal rule over the removed fraction). Lower is better.
- Both faithfulness metrics replace a removed element by linear interpolation in time between the nearest kept elements of the same channel, using the original values only (nearest value if only one side exists, channel mean of the window if the whole channel is removed). `replacement="train_mean"` uses the per-channel training mean instead.
- **Average Sensitivity (AvgSen)** and **Continuity (Cont):** Gaussian input noise with σ = 0.01 (inputs are standardized per channel with training statistics) and M = 5 accepted perturbations per sample, fixed seed. A perturbation is accepted if |Δf| ≤ τ = 0.05 (regression) or if the predicted class is unchanged (classification). AvgSen is the mean ratio of the Frobenius norms of the attribution change and the noise; Cont is 1 minus the mean Spearman correlation between the original and perturbed attribution maps.
- **Complexity:** Gini sparseness and normalized entropy, at element level (Spars-E, Compl-E) and channel level (Spars-C, Compl-C).

<!-- robustness/splits: pending -->

Per {model, explainer}, each metric is summarized by its median over the evaluation samples after direction alignment (higher is better), and the explainers are ranked per dataset with PROMETHEE II (`experiments/analysis/promethee_ranking.py`, equal pillar weights).

### Datasets and splits

Windowing and splits are defined by the adapters in `experiments/datasets/adapters/` with the parameters of `experiments/configs/datasets_config.yaml`. Channels are standardized with training statistics.

| Dataset | Window L / hop H | Split in the current code |
|---|---|---|
| C-MAPSS FD001 | 30 / 1 | random 80/20 train/validation split of the training windows; test: last 5 windows of each test engine |
| Ecoating | 30 / 1 | random 80/20 train/validation split of the training windows; test: last 5 windows of the test file |
| Hydraulic | 50 (one cycle) | stratified 90/10 train/validation split of the cycles |
| CWRU (12 kHz) | 2048 / 512 | 10% of the recordings for validation, chosen at random with a fixed seed |

## Reproducing the paper results

All commands run from the repository root unless noted. Each script writes to the `results/` folder given in the table and accepts `--help`.

| Result | Script | Results folder |
|---|---|---|
| PROMETHEE input | `experiments/analysis/build_promethee_input.py --master <final_main_benchmark_primary_scope.csv>` | `results/main_benchmark_14_promethee_input/` |
| Table 5 (ranking) | `experiments/analysis/promethee_ranking.py`, then `experiments/analysis/build_table5.py` | `results/main_benchmark_14_promethee/` |
| Figure 4 (family radars) | `experiments/analysis/radar_family.py` | `results/main_benchmark_14_radars/` |
| Figure 5 (variance attribution) | `experiments/analysis/variance_attribution.py` | `results/main_benchmark_14_anova/` |
| Figure 6, Table 6 (trade-offs) | `experiments/analysis/composites.py`, then `experiments/analysis/tradeoffs.py` | `results/main_benchmark_14_tradeoffs/` |
| App input | `experiments/analysis/build_app_input.py` | `results/main_benchmark_14_with_performance.csv` |
| Appendix C.1 (explainer settings) | from `experiments/`: `appendix/explainer_settings/run_numerical.py`, `run_semantic.py`, `build_numerical_diagnostics.py`, `summarize.py` | `results/r2_6_parameter_sensitivity/` |
| Appendix C.3 (ranking uncertainty) | `experiments/appendix/ranking_uncertainty/run_analysis.py --data-dir <dir>`, `seed_detail.py`, `make_table.py`, `fig_training.py` | `results/r1_9_uncertainty_20261016/` |

The figures used in the paper are in `results/main_benchmark_14_radars_figure4_overleaf/` and `results/main_benchmark_14_tradeoffs_figure6_overleaf/`. Predictive performance per model is in `results/main_benchmark_14_tradeoffs/predictive_performance_36.csv`.

For Appendix C.3, `--data-dir` is a folder containing the contents of `results/r1_9_uncertainty_20261016/uncertainty_completed_20261016.tar.gz` and of `uncertainty_scores_20261016.tar.gz` (release asset), and `data/<dataset>/sample_manifest.csv`.

The baseline and background of IntegGrad, DeepLift, DeepSHAP, ExpectGrad and GradSHAP are computed by `run_xai_evaluation.py::build_train_background` from a shuffled training loader, so regenerated maps of these explainers can differ slightly from the stored ones. The appendix scripts reuse the stored maps in `results/`.

## Docker

```bash
docker build -t true-x:latest .
docker-compose up -d
```

The app will be accessible at `http://localhost:8501`.

## ExpliTest Library

The metrics library is developed independently at [serval-uni-lu/ExpliTest](https://github.com/serval-uni-lu/ExpliTest) and included here as a git submodule. This branch uses the ExpliTest commit with the interpolation replacement and the deletion-curve Pixel Flipping (`submodules/explitest`, branch `neurocom-r1`).
