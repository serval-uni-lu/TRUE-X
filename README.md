# TRUE-X

This repo provides a DEMO of the TRUE-X decision support tool, more details, and a link to the full repo with code and paper coming soon.

## Running the Streamlit Frontend

### Option 1: Local Installation

#### Prerequisites
- Python 3.11 or higher
- Node.js 18.x or higher
- pip (Python package manager)

#### Steps

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd TRUE-X
   ```

2. **Initialize and sync the git submodule**
   ```bash
   git submodule init
   git submodule update --recursive
   ```

   This will pull the PipelineVis submodule which is required for the application.

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the PipelineProfiler package**
   ```bash
   cd PipelineVis
   npm install --legacy-peer-deps
   npm run build
   pip install .
   cd ..
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run frontend.py
   ```

   The application will start and automatically open in your default browser at `http://localhost:8501`

#### Optional Configuration
You can customize the Streamlit configuration by editing `streamlit_config.toml`. Common settings include:
- Server port
- Server address
- Theme settings

### Option 2: Docker

#### Prerequisites
- Docker and Docker Compose installed

#### Build the Docker Image

1. **Build the Docker image**
   ```bash
   docker build -t true-x:latest .
   ```

   This step is required before running the container, whether using Docker Compose or manual docker run commands.

#### Quick Start with Docker Compose

1. **Build the image** (if not already done)
   ```bash
   docker build -t true-x:latest .
   ```

2. **Run the container with Docker Compose**
   ```bash
   docker-compose up -d
   ```

   The application will be accessible based on your docker-compose configuration.

#### Manual Docker Run

1. **Run the container** (after building the image)
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/streamlit_config.toml:/app/.streamlit/config.toml:ro \
     -v $(pwd)/PipelineVis:/app/PipelineVis:rw \
     true-x:latest
   ```

   The application will be accessible at `http://localhost:8501`

#### Docker Configuration Notes
- The container exposes port `8501` (Streamlit default)
- Volume mounts allow you to:
  - Override configuration via `streamlit_config.toml`
  - Access and update the PipelineVis module
- For production deployments, update the Traefik labels in `docker-compose.yml` according to your infrastructure
