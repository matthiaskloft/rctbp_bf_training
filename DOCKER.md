# Docker Deployment Guide

This guide covers how to build, run, and deploy the RCTBP BayesFlow Training service using Docker.

## Quick Start

### Build and Run Locally

```bash
# Build the Docker image
docker build -t rctbp-bf-training:latest .

# Run the container
docker run -p 8000:8000 rctbp-bf-training:latest

# Access the API
curl http://localhost:8000/health
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f rctbp-api

# Stop the service
docker-compose down
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API server port |
| `HOST` | `0.0.0.0` | API server host |
| `MODEL_PATH` | `/models/default` | Path to trained model |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `KERAS_BACKEND` | `torch` | Keras backend (torch recommended) |

### Loading Models

Mount your trained model directory:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/models:/models:ro \
  -e MODEL_PATH=/models/my_model \
  rctbp-bf-training:latest
```

## Development

### Hot-Reload Development

Use the development Dockerfile for hot-reload:

```bash
# Using docker-compose
docker-compose --profile dev up rctbp-dev

# Or manually
docker build -f Dockerfile.dev -t rctbp-bf-training:dev .
docker run -p 8001:8000 \
  -v $(pwd)/src:/app/src:ro \
  -v $(pwd)/api:/app/api:ro \
  rctbp-bf-training:dev
```

## API Endpoints

Once running, the API provides:

- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /infer` - Run inference on ANCOVA data
- `POST /simulate` - Simulate RCT data
- `POST /power` - Run Bayesian power analysis

Full API documentation available at `http://localhost:8000/docs` (Swagger UI).

## Fluence Network Deployment

### Prerequisites

1. Docker image pushed to a registry
2. Fluence CLI installed
3. Fluence account configured

### Deployment Steps

1. **Build and push the image:**

```bash
# Build
docker build -t your-registry/rctbp-bf-training:v0.1.0 .

# Push
docker push your-registry/rctbp-bf-training:v0.1.0
```

2. **Update fluence.yaml** with your registry:

```yaml
container:
  image: "your-registry/rctbp-bf-training:v0.1.0"
```

3. **Deploy to Fluence:**

```bash
fluence deploy -f fluence.yaml
```

### Model Storage

For production, store trained models in:
- Cloud storage (S3, GCS, Azure Blob)
- IPFS for decentralized storage
- Volume mounts from the Fluence provider

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 0.5 cores | 2 cores |
| Memory | 1 GB | 4 GB |
| Disk | 2 GB | 5 GB |

## Troubleshooting

### Container won't start

Check logs:
```bash
docker logs rctbp-api
```

Common issues:
- Missing `KERAS_BACKEND=torch` environment variable
- Model path doesn't exist
- Port already in use

### Out of memory

Increase container memory limit:
```bash
docker run -m 4g -p 8000:8000 rctbp-bf-training:latest
```

### Slow inference

The container uses CPU by default. For faster inference:
- Use a host with more CPU cores
- Consider GPU deployment for large-scale usage
