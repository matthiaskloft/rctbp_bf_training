"""
Objective Computation and Parameter Normalization.

Utilities for computing and normalizing objective values used in
hyperparameter optimization (both single- and multi-objective).

Functions here are pure computation with no training/Optuna dependencies.
"""

import importlib.util
from typing import Any, Dict, Optional, Tuple

import numpy as np

KERAS_AVAILABLE = importlib.util.find_spec("keras") is not None


# =============================================================================
# CONSTANTS
# =============================================================================

# Normalization constant: log10(1,000,000) = 6
# Maps param_count to ~0-1 scale: 10K -> 0.67, 100K -> 0.83, 1M -> 1.0
PARAM_COUNT_LOG_SCALE = 6.0

# Failed trial penalty (normalized scale)
FAILED_TRIAL_CAL_ERROR = 1.0
FAILED_TRIAL_PARAM_SCORE = 1.5  # Above any valid normalized value


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def get_param_count(model: Any) -> int:
    """
    Count trainable parameters in a Keras model.

    Parameters
    ----------
    model : keras.Model or BayesFlow approximator
        The model to count parameters for.

    Returns
    -------
    int
        Total number of trainable parameters, or -1 if model not built.

    Raises
    ------
    ImportError
        If Keras is not installed.
    TypeError
        If the model type is not supported.
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras is required for parameter counting")

    # Handle BayesFlow approximator wrapper
    if hasattr(model, 'count_params'):
        try:
            return int(model.count_params())
        except ValueError:
            # Model not built yet - return -1 to indicate deferred counting
            return -1
    elif hasattr(model, 'trainable_weights'):
        if len(model.trainable_weights) == 0:
            return -1  # Model not built
        return sum(np.prod(w.shape) for w in model.trainable_weights)
    else:
        raise TypeError(f"Cannot count parameters for type: {type(model)}")


def estimate_param_count(
    summary_dim: int = 8,
    deepset_width: int = 64,
    deepset_depth: int = 2,
    flow_depth: int = 6,
    flow_hidden: int = 128,
    n_conditions: int = 4,
    n_params: int = 1,
) -> int:
    """
    Estimate parameter count without building the model.

    Useful for quick filtering of configurations before training.

    Parameters
    ----------
    summary_dim : int
        DeepSet output dimension.
    deepset_width : int
        Width of DeepSet MLP layers.
    deepset_depth : int
        Number of DeepSet aggregation stages.
    flow_depth : int
        Number of coupling layers.
    flow_hidden : int
        Hidden size in flow subnets.
    n_conditions : int
        Number of conditioning variables.
    n_params : int
        Number of parameters to infer.

    Returns
    -------
    int
        Estimated parameter count.
    """
    # DeepSet: 4 MLPs (equivariant, inner, outer, last) with depth layers
    # Each MLP: input->hidden->hidden->output
    # Rough estimate: 4 * depth * width^2
    deepset_params = 4 * deepset_depth * (deepset_width ** 2)
    deepset_params += deepset_width * summary_dim  # final projection

    # CouplingFlow: depth subnets
    # Each subnet input: summary_dim + n_conditions + ceil(n_params/2)
    subnet_input = summary_dim + n_conditions + max(1, n_params // 2)
    # Each subnet: input->hidden->hidden->output (2 * ceil(n_params/2))
    subnet_output = 2 * max(1, (n_params + 1) // 2)
    subnet_params = (
        subnet_input * flow_hidden +  # first layer
        flow_hidden * flow_hidden +   # middle layer
        flow_hidden * subnet_output   # output layer
    )
    flow_params = flow_depth * subnet_params

    return int(deepset_params + flow_params)


# =============================================================================
# OBJECTIVE COMPUTATION
# =============================================================================

def compute_composite_objective(
    metrics: Dict,
    param_count: int,
    param_budget: int = 50_000,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a single composite objective from multiple metrics.

    Use this for single-objective optimization when you want to
    combine calibration and model size into one score.

    Parameters
    ----------
    metrics : dict
        Metrics dict from validation pipeline with 'summary' key.
    param_count : int
        Number of trainable parameters.
    param_budget : int
        Target parameter count (penalty kicks in above this).
    weights : dict, optional
        Weights for each component. Default:
        {"calibration": 0.5, "sbc": 0.3, "size": 0.2}

    Returns
    -------
    float
        Composite objective (lower is better).
    """
    if weights is None:
        weights = {"calibration": 0.5, "sbc": 0.3, "size": 0.2}

    summary = metrics.get("summary", metrics)

    # Calibration error (coverage)
    cal_error = summary.get("mean_cal_error", 0.1)

    # SBC uniformity (C2ST deviation from 0.5)
    c2st = summary.get("sbc_c2st_accuracy", 0.5)
    sbc_deviation = abs(c2st - 0.5)

    # Size penalty (soft, logarithmic above budget)
    size_penalty = max(
        0, np.log1p(param_count / param_budget) - np.log1p(1)
    )

    composite = (
        weights["calibration"] * cal_error
        + weights["sbc"] * sbc_deviation
        + weights["size"] * size_penalty
    )

    return float(composite)


# =============================================================================
# OBJECTIVE NORMALIZATION
# =============================================================================

def normalize_param_count(param_count: int) -> float:
    """
    Normalize parameter count to ~0-1 scale using log10.

    Maps parameter counts to a comparable scale with calibration error:
    - 10,000 params -> ~0.67
    - 100,000 params -> ~0.83
    - 1,000,000 params -> 1.0

    Parameters
    ----------
    param_count : int
        Raw parameter count.

    Returns
    -------
    float
        Normalized score in ~0-1 range.
    """
    if param_count <= 0:
        return 0.0
    return np.log10(param_count) / PARAM_COUNT_LOG_SCALE


def denormalize_param_count(normalized: float) -> int:
    """
    Convert normalized param score back to raw parameter count.

    Parameters
    ----------
    normalized : float
        Normalized parameter score.

    Returns
    -------
    int
        Raw parameter count.
    """
    if normalized <= 0:
        return 0
    return int(10 ** (normalized * PARAM_COUNT_LOG_SCALE))


def extract_objective_values(
    metrics: Dict,
    param_count: int,
) -> Tuple[float, float]:
    """
    Extract objective values for multi-objective optimization.

    Returns (calibration_error, normalized_param_score) for Pareto
    optimization. Both objectives are normalized to [0, 1] scale for
    balanced NSGA-II crowding.

    Parameters
    ----------
    metrics : dict
        Metrics dict from validation pipeline.
    param_count : int
        Number of trainable parameters.

    Returns
    -------
    tuple of (float, float)
        (calibration_error, normalized_param_score) both in [0, 1] range.
    """
    summary = metrics.get("summary", metrics)

    # Primary calibration metric: mean absolute calibration error
    # Already bounded [0, 1] - no transformation needed
    cal_error = summary.get("mean_cal_error", 1.0)

    # Normalize param_count to comparable [0, 1] scale
    normalized_params = normalize_param_count(param_count)

    return float(cal_error), float(normalized_params)
