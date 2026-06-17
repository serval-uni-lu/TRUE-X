FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ExpliTest submodule
COPY submodules/explitest/ ./submodules/explitest/
RUN pip install --no-cache-dir -e submodules/explitest

# PipelineVis submodule — build npm bundle then install Python package
COPY submodules/PipelineVis/ ./submodules/PipelineVis/
WORKDIR /app/submodules/PipelineVis/PipelineProfiler
RUN npm install --legacy-peer-deps --no-audit --no-fund regenerator-runtime core-js
RUN npm run build --legacy-peer-deps
WORKDIR /app/submodules/PipelineVis
RUN pip install .

WORKDIR /app

# Application code, models and benchmark results
COPY app/ ./app/
COPY experiments/ ./experiments/
COPY data/ ./data/
COPY saved_models/ ./saved_models/
COPY results/ ./results/

# Streamlit config (sets baseUrlPath for reverse-proxy deployments)
RUN mkdir -p .streamlit
COPY app/streamlit_config.toml .streamlit/config.toml

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["streamlit", "run", "app/frontend.py", "--server.port=8501", "--server.headless=true"]
