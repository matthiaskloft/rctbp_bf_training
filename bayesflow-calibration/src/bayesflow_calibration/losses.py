"""Differentiable calibration loss functions.

Implements the calibration loss from Falkner et al. (NeurIPS 2023):
"Calibrating Neural Simulation-Based Inference with
Differentiable Coverage Probability".

Core idea: penalize coverage errors during training so that credible intervals
have the advertised coverage by construction.
"""

from __future__ import annotations

import keras


def ste_indicator(x):
    """Hard indicator with straight-through estimator gradients.

    Forward: 1.0 where x > 0, 0.0 elsewhere.
    Backward: identity (gradients flow through as if f(x) = x).

    Parameters
    ----------
    x : tensor
        Input tensor.

    Returns
    -------
    tensor
        Binary indicator with STE gradients.
    """
    hard = keras.ops.cast(x > 0, dtype=x.dtype)
    return x + keras.ops.stop_gradient(hard - x)


def compute_ranks(log_prob_true, log_probs_prior):
    """Compute differentiable fractional ranks via STE indicators.

    For each item in the batch, computes the fraction of prior samples
    that have higher log-density than the true parameter value. Under
    correct calibration, these ranks should be uniformly distributed.

    Parameters
    ----------
    log_prob_true : tensor
        Log-density of true parameters, shape ``(batch,)`` or ``(batch, param_dim)``.
    log_probs_prior : tensor
        Log-density of prior samples, shape ``(batch, n_rank_samples)``
        or ``(batch, n_rank_samples, param_dim)``.

    Returns
    -------
    tensor
        Fractional ranks in [0, 1], shape ``(batch,)`` or ``(batch, param_dim)``.
    """
    # Expand true log-probs for broadcasting: (batch, 1, ...) - (batch, n_samples, ...)
    log_prob_true_expanded = keras.ops.expand_dims(log_prob_true, axis=1)
    diff = log_probs_prior - log_prob_true_expanded
    indicators = ste_indicator(diff)
    ranks = keras.ops.mean(indicators, axis=1)
    return ranks


def coverage_error(ranks, mode=0.0):
    """Compute calibration loss from empirical rank distribution.

    Compares empirical coverage (from sorted ranks) against expected
    uniform coverage. The ``mode`` parameter controls the loss behavior:

    - ``mode=0.0``: conservativeness — penalize under-coverage only
      (ReLU). Produces conservative posteriors (wider credible
      intervals).
    - ``mode=1.0``: calibration — penalize both under- and
      over-coverage. Produces well-calibrated posteriors.
    - ``0 < mode < 1``: mixture of both behaviors.

    Parameters
    ----------
    ranks : tensor
        Fractional ranks, shape ``(batch,)`` or ``(batch, param_dim)``.
        Batch size must be >= 2.
    mode : float, optional
        Loss mode in [0, 1]. 0.0 = conservativeness,
        1.0 = calibration. Default: 0.0.

    Returns
    -------
    tensor
        Scalar calibration loss.
    """
    sorted_ranks = keras.ops.sort(ranks, axis=0)
    batch_size = keras.ops.shape(sorted_ranks)[0]

    # Expected uniform quantiles: (batch_size,)
    expected = keras.ops.linspace(0.0, 1.0, batch_size)

    # Broadcast for multi-parameter case
    if sorted_ranks.ndim > 1:
        n_dims = sorted_ranks.ndim - 1
        for _ in range(n_dims):
            expected = keras.ops.expand_dims(expected, axis=-1)

    diff = sorted_ranks - expected

    if mode == 0.0:
        # Conservativeness: penalize under-coverage only
        error = keras.ops.relu(-diff)
    elif mode == 1.0:
        # Calibration: penalize both directions
        error = diff
    else:
        # Mixture
        error = (1.0 - mode) * keras.ops.relu(-diff) + mode * diff

    return keras.ops.mean(error**2)
