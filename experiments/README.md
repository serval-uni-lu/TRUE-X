# Experiments

This folder contains the full experimental pipeline behind the paper's evaluation: model training, XAI attribution, and trustworthiness metric computation.

## Pre-computed Artifacts

Results are already included — no re-running required to inspect the data behind the paper's tables and figures.

| Artifact | Location |
|---|---|
| Trained model weights + configs | `saved_models/` |
| Per-combination XAI evaluation outputs | `results/Hydraulic/<MODEL>/` |
| Aggregated results table | `results/Hydraulic.csv` |

## Scope

| | |
|---|---|
| Dataset | Hydraulic condition monitoring (9 channels, L = 50, 1 449 instances) |
| Models | LogCla, IncTime, ResNet, LSTM, TST |
| Explainers | FeatAbl, GradX, DeepSHAP, LIME |
| Metrics | FC ↑, PF ↑ · AvgSen ↓, Cont ↓ · Spars-E ↑, Spars-C ↑, Compl-E ↓, Compl-C ↓ |

## Reproducing from Scratch

### 1. Install dependencies

```bash
pip install -r experiments/requirements.txt
pip install shap lime   # required for DeepSHAP and LIME
```

### 2. Prepare the data

Place the Hydraulic dataset under `experiments/data/hydraulic_systems/`.

### 3. Train a model *(optional — pre-trained weights are included)*

```bash
cd experiments/src
python train.py --dataset Hydraulic --model INCEPTIONTIME
```

Model keys: `LOGISTIC_CLASSIFIER` (LogCla), `INCEPTIONTIME` (IncTime), `RESNET`, `LSTM`, `TST`

### 4. Run XAI evaluation

```bash
cd experiments/src
python run_xai_evaluation.py --dataset Hydraulic --model INCEPTIONTIME --explainer feature_ablation
```

Explainer keys: `feature_ablation` (FeatAbl), `gradient_x_input` (GradX), `deepliftshap` (DeepSHAP), `lime_tabular` (LIME)
