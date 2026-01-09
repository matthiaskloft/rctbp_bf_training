# Dockerfile for NPE model training on Fluence.network
# Optimized for interactive Bayesian optimization with GPU support

# Use NVIDIA CUDA base image with Python 3.11
# CUDA 12.1 is well-supported by PyTorch and widely available
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="rctbp_bf_training"
LABEL description="Neural Posterior Estimation training environment for RCT Bayesian Power analysis"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Keras backend - use PyTorch for GPU training
ENV KERAS_BACKEND=torch

# Jupyter settings
ENV JUPYTER_ENABLE_LAB=yes

# Create non-root user for security
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} trainer && \
    useradd -m -u ${UID} -g trainer -s /bin/bash trainer

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=trainer:trainer pyproject.toml ./
COPY --chown=trainer:trainer src/ ./src/
COPY --chown=trainer:trainer examples/ ./examples/
COPY --chown=trainer:trainer tests/ ./tests/
COPY --chown=trainer:trainer README.md ./

# Install PyTorch with CUDA support first (before other dependencies)
# Using PyTorch 2.2+ for best CUDA 12.1 compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the package with all dependencies
RUN pip install -e ".[dev,notebooks]"

# Create directories for persistence
RUN mkdir -p /app/data /app/models /app/notebooks && \
    chown -R trainer:trainer /app

# Copy entrypoint script
COPY --chown=trainer:trainer scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER trainer

# Configure Jupyter
RUN mkdir -p /home/trainer/.jupyter && \
    jupyter notebook --generate-config

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Default command: start Jupyter Lab
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["jupyter", "lab"]
