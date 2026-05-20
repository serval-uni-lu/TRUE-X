# TRUE-X

TRUE-X is a decision support tool for explainable AI evaluation on multivariate time series. It combines a Streamlit demo interface with the [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) metrics library.

## Repository Structure

```
TRUE-X/
├── app/                    # Streamlit application (entry: frontend.py)
├── experiments/            # Experiment pipeline (configs, scripts, notebooks)
├── results/                # Benchmark outputs (CSV)
├── saved_models/           # Pre-trained model weights (.pt + .json)
├── submodules/
│   ├── explitest/          # ExpliTest metrics library (git submodule)
│   └── PipelineVis/        # Pipeline visualisation component (git submodule)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> **Note:** `data/` is not versioned in this repository. See the *Data & Models* section for download links.

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

The app will open at `http://localhost:8501`.

### 4. Reproduce experiments (optional)

```bash
pip install -r experiments/requirements.txt
cd experiments
python run_xai_evaluation.py
```

## Docker

```bash
docker build -t true-x:latest .
docker-compose up -d
```

The app will be accessible at `http://localhost:8501`.

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

## ExpliTest Library

The metrics library is developed independently at [serval-uni-lu/ExpliTest](https://github.com/serval-uni-lu/ExpliTest) and included here as a git submodule.
