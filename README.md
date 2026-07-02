# TRUE-X

TRUE-X is a decision support tool for explainable AI evaluation on multivariate time series. It combines a Streamlit demo interface with the [ExpliTest](https://github.com/serval-uni-lu/ExpliTest) metrics library and a PipelineVis-based visualisation component.

## Repository Structure

```text
TRUE-X/
├── app/                    # Streamlit application
├── experiments/            # Experiment pipeline: configs, scripts, notebooks
├── results/                # Benchmark outputs: CSV files, figures
├── submodules/
│   ├── explitest/          # ExpliTest metrics library
│   └── PipelineVis/        # Pipeline visualisation component
├── pyproject.toml          # Python project metadata and dependencies
├── uv.lock                 # Locked Python dependency versions
├── Dockerfile
└── docker-compose.yml
```

`data/` and `saved_models/` are not versioned in this repository. See [Data & Models](#data--models).

## Requirements

TRUE-X uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Getting Started

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repository-url>
cd TRUE-X
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
uv sync
```

If ExpliTest is not declared as a local dependency in `pyproject.toml`, install it explicitly:

```bash
uv pip install -e submodules/explitest
```

### 3. Run the Streamlit app

```bash
uv run streamlit run app/app.py
```

The app will be available at:

```text
http://localhost:8501
```

## Docker

Build the image:

```bash
docker build -t true-x:latest .
```


The app will be available at:

```text
http://localhost:8501
```

Stop the service:

```bash
docker-compose down
```

## Data & Models

The following directories are intentionally not versioned:

```text
data/
saved_models/
```

To reproduce experiments, download the required artifacts and place them at the expected paths, or update the project configuration and symlinks to point to your local copies.

## ExpliTest Library

The metrics library is developed independently at [serval-uni-lu/ExpliTest](https://github.com/serval-uni-lu/ExpliTest) and included here as a Git submodule:

```text
submodules/explitest/
```

## PipelineVis

The visualisation component is included as a Git submodule:

```text
submodules/PipelineVis/
```

It is used by the Streamlit interface to provide pipeline visualisation features.

## Updating Submodules

To update all submodules to the commits referenced by this repository:

```bash
git submodule update --init --recursive
```

To fetch newer commits from the submodule remotes:

```bash
git submodule update --remote --recursive
```
