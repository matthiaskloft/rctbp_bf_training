# Environment Setup

## Using pip (Recommended)

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
