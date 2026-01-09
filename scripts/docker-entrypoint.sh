#!/bin/bash
set -e

# Docker entrypoint script for NPE training environment
# Handles startup configuration and GPU detection

echo "=============================================="
echo "NPE Training Environment - Startup"
echo "=============================================="

# Display environment info
echo "Python version: $(python --version)"
echo "Keras backend: ${KERAS_BACKEND:-torch}"

# Check for GPU availability
echo ""
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected - running in CPU mode')
    print('Training will be slower without GPU acceleration')
"

# Verify BayesFlow installation
echo ""
echo "Verifying BayesFlow installation..."
python -c "
import bayesflow
print(f'BayesFlow version: {bayesflow.__version__}')
"

# Check for existing Optuna databases
echo ""
echo "Checking for existing optimization databases..."
for db in /app/examples/*.db /app/data/*.db; do
    if [ -f "$db" ]; then
        echo "  Found: $db"
    fi
done

# Ensure proper permissions on mounted volumes
if [ -d "/app/data" ]; then
    # Only try to change ownership if we have permission
    if [ -w "/app/data" ]; then
        echo ""
        echo "Data directory ready: /app/data"
    fi
fi

if [ -d "/app/models" ]; then
    if [ -w "/app/models" ]; then
        echo "Models directory ready: /app/models"
    fi
fi

echo ""
echo "=============================================="
echo "Starting service..."
echo "=============================================="

# Handle different startup commands
case "$1" in
    jupyter)
        shift
        case "$1" in
            lab)
                shift
                echo "Starting Jupyter Lab..."
                exec jupyter lab \
                    --ip=0.0.0.0 \
                    --port=8888 \
                    --no-browser \
                    --allow-root \
                    --NotebookApp.token='' \
                    --NotebookApp.password='' \
                    --NotebookApp.allow_origin='*' \
                    --NotebookApp.disable_check_xsrf=True \
                    --notebook-dir=/app \
                    "$@"
                ;;
            notebook)
                shift
                echo "Starting Jupyter Notebook..."
                exec jupyter notebook \
                    --ip=0.0.0.0 \
                    --port=8888 \
                    --no-browser \
                    --allow-root \
                    --NotebookApp.token='' \
                    --NotebookApp.password='' \
                    --NotebookApp.allow_origin='*' \
                    --NotebookApp.disable_check_xsrf=True \
                    --notebook-dir=/app \
                    "$@"
                ;;
            *)
                echo "Starting Jupyter Lab (default)..."
                exec jupyter lab \
                    --ip=0.0.0.0 \
                    --port=8888 \
                    --no-browser \
                    --allow-root \
                    --NotebookApp.token='' \
                    --NotebookApp.password='' \
                    --NotebookApp.allow_origin='*' \
                    --NotebookApp.disable_check_xsrf=True \
                    --notebook-dir=/app \
                    "$@"
                ;;
        esac
        ;;
    python)
        shift
        echo "Running Python script..."
        exec python "$@"
        ;;
    pytest)
        shift
        echo "Running tests..."
        exec pytest "$@"
        ;;
    bash)
        shift
        echo "Starting bash shell..."
        exec bash "$@"
        ;;
    *)
        # If no recognized command, assume it's a custom command
        echo "Running custom command: $@"
        exec "$@"
        ;;
esac
