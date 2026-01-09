# Docker Deployment for NPE Training on Fluence.network

This guide covers deploying the NPE training environment as a Docker container for interactive model training on Fluence.network or any GPU-enabled compute platform.

## Overview

The Docker setup provides:
- **GPU-accelerated training** via NVIDIA CUDA 12.1
- **Interactive Jupyter Lab** for running optimization notebooks
- **Persistent storage** for Optuna databases and trained models
- **Reproducible environment** with all dependencies pre-installed

## Prerequisites

### For GPU Training (Recommended)
- Docker 20.10+
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU with CUDA 12.x support
- At least 8GB GPU memory recommended

### For CPU-only Training
- Docker 20.10+
- At least 8GB RAM recommended

## Quick Start

### GPU Version (Recommended)

```bash
# Build the image
docker compose build npe-training

# Start the container
docker compose up npe-training

# Access Jupyter Lab at http://localhost:8888
```

### CPU Version (Development/Testing)

```bash
# Build the CPU image
docker compose build npe-training-cpu

# Start with CPU profile
docker compose --profile cpu up npe-training-cpu

# Access Jupyter Lab at http://localhost:8889
```

## Building Images

### Build GPU Image

```bash
# Standard build
docker build -t rctbp-bf-training:latest .

# Build with custom UID/GID (for volume permissions)
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t rctbp-bf-training:latest .
```

### Build CPU Image

```bash
docker build -f Dockerfile.cpu -t rctbp-bf-training:cpu .
```

## Running Containers

### Interactive Training (Default)

```bash
# Start Jupyter Lab
docker compose up npe-training

# Or run directly with docker
docker run -it --gpus all \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  rctbp-bf-training:latest
```

### Running Tests

```bash
docker run --rm rctbp-bf-training:latest pytest
```

### Running Custom Scripts

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  rctbp-bf-training:latest \
  python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Interactive Shell

```bash
docker run -it --gpus all rctbp-bf-training:latest bash
```

## Volume Mounts

The container uses three main volume mounts:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Optuna databases, intermediate data |
| `./models` | `/app/models` | Trained model checkpoints |
| `./examples` | `/app/examples` | Notebooks (editable) |

### Creating Data Directories

```bash
mkdir -p data models
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KERAS_BACKEND` | `torch` | Keras backend (torch recommended) |
| `JUPYTER_TOKEN` | (empty) | Jupyter authentication token |
| `JUPYTER_PORT` | `8888` | Host port for Jupyter |
| `OPTUNA_STORAGE` | `sqlite:///...` | Optuna database location |

### Setting a Jupyter Token (Security)

```bash
JUPYTER_TOKEN=your_secret_token docker compose up npe-training
```

## Fluence.network Deployment

### Preparing for Fluence

1. **Build and tag the image:**
   ```bash
   docker build -t your-registry/rctbp-bf-training:latest .
   ```

2. **Push to container registry:**
   ```bash
   docker push your-registry/rctbp-bf-training:latest
   ```

3. **Configure Fluence deployment** with:
   - GPU resource requirements
   - Port 8888 exposed
   - Volume mounts for persistence

### Fluence-specific Considerations

- **GPU Selection**: Request NVIDIA GPUs with sufficient memory (8GB+ recommended)
- **Persistence**: Use Fluence's persistent storage for `/app/data` and `/app/models`
- **Networking**: Ensure port 8888 is accessible for Jupyter Lab
- **Token Security**: Set `JUPYTER_TOKEN` for production deployments

## Training Workflow

### Using the Bayesian Optimization Notebook

1. Start the container and access Jupyter Lab
2. Open `examples/ancova_optimization.ipynb`
3. Configure optimization parameters:
   - `n_trials`: Number of optimization trials (50-200 recommended)
   - `study_name`: Name for Optuna study
   - `storage`: Database path for persistence

4. Run optimization cells
5. View Pareto front and select best configuration
6. Train final model with `train_until_threshold()`
7. Export model with metadata

### Resuming Optimization

Optuna studies are persisted to SQLite. To resume:

```python
# In notebook
study = optuna.load_study(
    study_name="ancova_optimization",
    storage="sqlite:////app/data/optuna_studies.db"
)
# Continue optimization
study.optimize(objective, n_trials=50)
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Permission Issues with Volumes

```bash
# Build with matching UID/GID
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t rctbp-bf-training:latest .
```

### Out of Memory

- Reduce `batch_size` in optimization notebook
- Use gradient checkpointing
- Use CPU-only mode for testing

### Slow Training

- Ensure GPU is being used (check startup logs)
- Increase `batch_size` if GPU memory allows
- Use fewer validation grid points during optimization

## Resource Requirements

### Minimum (CPU)
- 4 CPU cores
- 8GB RAM
- 10GB disk space

### Recommended (GPU)
- 4+ CPU cores
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 20GB disk space

### Production (Fluence)
- 8+ CPU cores
- 32GB RAM
- NVIDIA A100/H100 or similar
- 50GB+ disk space for models and databases

## Security Notes

- The container runs as non-root user `trainer` (UID 1000)
- Jupyter runs without authentication by default (set `JUPYTER_TOKEN` for production)
- Volume mounts use the host user's UID/GID for proper permissions
- No secrets are stored in the image

## Maintenance

### Updating Dependencies

```bash
# Rebuild image to pick up pyproject.toml changes
docker compose build --no-cache npe-training
```

### Cleaning Up

```bash
# Remove containers and volumes
docker compose down -v

# Remove images
docker rmi rctbp-bf-training:latest rctbp-bf-training:cpu
```
