FROM python:3.11-slim

LABEL authors="raoni.lourenco"

# Set working directory
WORKDIR /app

# Install system dependencies for Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY frontend.py export_profiler.py ranks_per_block.csv requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the npm submodule
COPY PipelineVis ./PipelineVis

# Build the npm project
WORKDIR /app/PipelineVis/PipelineProfiler
RUN npm install && npm run build

WORKDIR /app/PipelineVis/

RUN pip install .

# Go back to app directory
WORKDIR /app

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

# Entrypoint for the app
ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.headless=true"]
