# CLAUDE.md


## Purpose

Python package for training Neural Posterior Estimation (NPE) models using BayesFlow for Randomized Controlled Trial (RCT) Bayesian power analysis. Implements ANCOVA models for 2-arm continuous outcome trials with Optuna-based multi-objective hyperparameter optimization (calibration error + model complexity).

Key capabilities:
- Train neural networks to approximate posterior distributions for RCT parameters
- Multi-objective hyperparameter optimization via Optuna (Pareto front: calibration error vs param count)
- Simulation-based calibration (SBC) validation
- Docker-based deployment with GPU support (CUDA 12.x)


## Workflow

Always work on a git worktree, not the main repository.

1. **Spec** - Understand the task and read relevant code first
2. **Plan** - Use plan mode to design the approach
3. **Draft** - Implement the changes
4. **Simplify** - Review and simplify the solution
5. **Update Tests** - Add or update tests for changed functionality
6. **Update Docs** - Update docstrings and `docs/` documentation
7. **Test** - Run `pytest`
8. **Verify** - Run `ruff check src/` and `mypy src/`
9. **Quality Check** - Use a subagent to verify code quality and safety
10. **Learnings** - If problems occurred, note mistakes to avoid in the Learnings section below
11. **Commit & PR** - Commit changes and create a pull request
12. **Clean-up worktrees** - After merge, delete the worktree and branch


## Quick Commands

```bash
# Environment setup
python setup_env.py                          # Auto-detect GPU, create venv
python setup_env.py --cpu-only               # CPU-only PyTorch
pip install -e ".[dev,notebooks]"            # Editable install with dev tools

# Testing
pytest                                       # Run all tests
pytest tests/test_core                       # Run core tests only
pytest -v --cov=rctbp_bf_training            # Verbose with coverage

# Code quality
ruff check src/                              # Linting
mypy src/                                    # Type checking

# Training (interactive)
jupyter notebook examples/                   # Start Jupyter for notebooks

# Docker
docker compose build npe-training            # Build GPU image
docker compose up npe-training               # Run (Jupyter at :8888)
docker compose build npe-training-cpu        # Build CPU image
docker compose --profile cpu up npe-training-cpu  # Run CPU (Jupyter at :8889)
```


## Project Structure

```
rctbp_bf_training/
├── src/rctbp_bf_training/              # Main package
│   ├── core/                           # Generic NPE infrastructure
│   │   ├── infrastructure.py           # Network configs, builders, workflow creation
│   │   ├── optimization.py             # Optuna multi-objective optimization
│   │   ├── validation.py               # SBC validation pipeline
│   │   └── utils.py                    # Sampling, distribution utilities
│   ├── models/
│   │   └── ancova/
│   │       └── model.py                # ANCOVA simulators, priors, workflow factory
│   └── plotting/
│       └── diagnostics.py              # SBC diagnostic plots
│
├── tests/                              # Mirror structure of src/
│   ├── test_core/
│   ├── test_models/test_ancova/
│   └── test_plotting/
│
├── examples/                           # Jupyter notebooks
│   ├── ancova_basic.ipynb              # Basic ANCOVA training
│   ├── ancova_calibration_loss.ipynb   # Calibration loss training comparison
│   └── ancova_optimization.ipynb       # Optuna hyperparameter optimization
│
├── scripts/
│   └── docker-entrypoint.sh            # Docker container startup
│
├── docs/                               # Design docs and guides
│
├── Dockerfile                          # GPU image (CUDA 12.1)
├── Dockerfile.cpu                      # CPU-only image
├── docker-compose.yml                  # Compose for local/Fluence deployment
├── setup_env.py                        # Automated environment setup
├── pyproject.toml                      # Package config and dependencies
└── requirements.txt                    # Core dependencies
```


## Reference

### Key dependencies
- `bayesflow>=2.0` - Neural posterior estimation framework
- `keras>=3.9,<3.13` - Deep learning (backend: PyTorch)
- `optuna>=3.0` - Bayesian hyperparameter optimization
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`

### Architecture patterns
- **Generic core + model-specific implementations**: `core/` is model-agnostic; `models/ancova/` contains ANCOVA-specific code
- **Configuration via dataclasses**: All configs use typed dataclasses with defaults
- **Multi-objective optimization**: Optuna studies optimize (calibration_error, param_count) on a Pareto front
- **External calibration loss**: `bayesflow-calibration` ([bfcalloss](https://github.com/matthiaskloft/bfcalloss)) is a separate repo, installed via `pip install -e ".[calibration]"`

### Conventions
- Type hints throughout (enforced by mypy)
- NumPy-style docstrings
- Ruff for linting (line length 88, Python 3.9+)
- Test structure mirrors `src/` layout


## Learnings / Things to avoid

<!-- @Claude: add learnings at the end of each session if necessary -->
