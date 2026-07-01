# syntax=docker/dockerfile:1.7

############################
# 0) Source with Git submodules
############################
FROM alpine/git:v2.54.0 AS source

WORKDIR /src

COPY . .

RUN git submodule update --init --recursive


############################
# 1) Build frontend with Node
############################
FROM node:18-bookworm-slim AS frontend-builder

WORKDIR /app/PipelineVis/PipelineProfiler

# Copy only npm manifests first to maximize cache reuse
COPY --from=source /src/PipelineVis/PipelineProfiler/package*.json ./

# Prefer npm ci when package-lock.json exists
RUN --mount=type=cache,target=/root/.npm \
    npm ci --legacy-peer-deps --no-audit --no-fund \
    || npm install --legacy-peer-deps --no-audit --no-fund

# Copy the rest of the submodule only after npm install
COPY --from=source /src/PipelineVis/ /app/PipelineVis/

# Install explicit dependencies only if really needed
RUN --mount=type=cache,target=/root/.npm \
    npm install --legacy-peer-deps --no-audit --no-fund regenerator-runtime@0.14.1 core-js@3.49.0

RUN NODE_OPTIONS=--openssl-legacy-provider npm run build


############################
# 2) Build Python environment with uv
############################
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for cache-friendly layer ordering
COPY --from=source /src/pyproject.toml /src/uv.lock ./

# Install dependencies into a virtual environment, without installing the project yet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy application code
COPY --from=source /src/ /app/

# Copy explitest from the source stage
COPY --from=source /src/submodules/explitest /app/submodules/explitest

# Copy the already-built PipelineVis submodule from the Node stage
COPY --from=frontend-builder /app/PipelineVis /app/PipelineVis

# Final project install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Install the local PipelineVis Python package
WORKDIR /app/PipelineVis

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python .


WORKDIR /app


############################
# 3) Runtime stage
############################
FROM python:3.12-slim AS runtime

LABEL authors="raoni.lourenco"

WORKDIR /app

# Copy the entire venv + app from the builder
COPY --from=builder /app /app

EXPOSE 8501

# Streamlit config: disable telemetry, set default port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["/app/.venv/bin/streamlit", "run", "frontend.py", "--server.fileWatcherType=none"]