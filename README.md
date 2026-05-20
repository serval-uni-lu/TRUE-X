# TRUE-X

TRUE-X is a decision support tool for explainable AI evaluation on multivariate time series. It combines a Streamlit demo interface with the [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) metrics library.

## Repository Structure

```
TRUE-X/
├── app/                    # Streamlit application
├── experiments/            # Experiment pipeline (configs, scripts, notebooks)
├── results/                # Benchmark outputs (CSV, figures)
├── submodules/
│   ├── explitest/          # ExpliTest metrics library (git submodule)
│   └── PipelineVis/        # Pipeline visualisation component (git submodule)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> **Note:** `data/` and `saved_models/` are not versioned in this repository. See the *Data & Models* section below.

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
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

## Docker

```bash
docker build -t true-x:latest .
docker-compose up -d
```

The app will be accessible at `http://localhost:8501`.

## Data & Models

`data/` and `saved_models/` are not versioned in this repository. To reproduce experiments, download the artifacts and place them at the expected paths, or update the symlinks to point to your local copies.

## ExpliTest Library

The metrics library is developed independently at [serval-uni-lu/ExpliTest](https://github.com/serval-uni-lu/ExpliTest) and included here as a git submodule. To update it to the latest version:

```bash
git submodule update --remote submodules/explitest
```
