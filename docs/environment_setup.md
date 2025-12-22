# Environment Setup

## Quick Setup (Recommended)

The easiest way to set up your development environment is to use the automated setup script, which handles:
- Virtual environment creation
- GPU detection and PyTorch installation with CUDA support
- Installation of the package with all development and notebook dependencies
- Verification of the installation

### Run the Setup Script

```bash
python setup_env.py
```

This will:
1. Create a virtual environment in `./venv`
2. Detect your CUDA version (if you have an NVIDIA GPU)
3. Install PyTorch with appropriate GPU support (or CPU-only if no GPU detected)
4. Install the package in editable mode with dev and notebook dependencies
5. Verify all packages are installed correctly
6. Register the environment as a Jupyter kernel for use in notebooks

### Script Options

```bash
# Force recreate the environment
python setup_env.py --force

# Install CPU-only PyTorch (skip GPU detection)
python setup_env.py --cpu-only

# Use a specific CUDA version
python setup_env.py --cuda-version 12.6

# Custom virtual environment name
python setup_env.py --name my-env

# Show detailed installation output
python setup_env.py --verbose

# Get help
python setup_env.py --help
```

### After Setup

Activate the virtual environment:

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Unix/macOS:**
```bash
source venv/bin/activate
```

### Verify the Installation

```bash
python -c "import rctbp_bf_training; import torch; print(f'Package version: {rctbp_bf_training.__version__}'); print(f'GPU available: {torch.cuda.is_available()}')"
```

### Using in Jupyter Notebooks

The environment is automatically registered as a Jupyter kernel. You have two options:

**Option 1: Start Jupyter from the activated venv (Recommended)**
```bash
# Activate the venv first
venv\Scripts\activate     # Windows
source venv/bin/activate  # Unix/macOS

# Then start Jupyter
jupyter notebook examples/
```

**Option 2: Use the kernel from any Jupyter instance**
1. Start Jupyter from anywhere: `jupyter notebook`
2. Open or create a notebook
3. Select **Kernel → Change kernel → Python (rctbp_bf_training - venv)**

**Troubleshooting:**
- If the kernel doesn't appear, **restart Jupyter** (the kernel list is cached)
- Verify kernel installation: `jupyter kernelspec list`
- The kernel should appear as `rctbp-venv`

---

## Manual Setup

If you prefer to set up the environment manually or the automated script doesn't work for your system, follow the instructions below.

## Using pip

### Basic Installation

```bash
pip install -e .
```

### Installation with Development Tools

```bash
pip install -e ".[dev]"
```

### Installation with Notebook Support

```bash
pip install -e ".[notebooks]"
```

### Installation with All Optional Dependencies

```bash
pip install -e ".[dev,notebooks]"
```

## Using Conda

### Create and Activate Environment

```bash
conda create -n rctbp python=3.12
conda activate rctbp
```

### Install Package

```bash
pip install -e .
```

Or with optional dependencies:

```bash
pip install -e ".[dev,notebooks]"
```

## Verify Installation

Run the following command to verify the package is installed correctly:

```bash
python -c "import rctbp_bf_training; print(rctbp_bf_training.__version__)"
```

You should see:
```
0.1.0
```

## Test the Public API

```python
python -c "from rctbp_bf_training import ANCOVAConfig; print('Success!')"
```

## Running Tests

After installing with dev dependencies:

```bash
pytest
```

## Code Quality Checks

After installing with dev dependencies:

```bash
# Linting
ruff check src/

# Type checking
mypy src/
```
