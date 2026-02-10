"""Calibrated continuous approximator for BayesFlow.

Subclasses BayesFlow's ``ContinuousApproximator`` to inject a differentiable
calibration loss into the training objective. Implements the approach from
Falkner et al. (NeurIPS 2023).
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
from bayesflow.approximators import ContinuousApproximator

from bayesflow_calibration.losses import compute_ranks, coverage_error
from bayesflow_calibration.schedules import GammaSchedule


class CalibratedContinuousApproximator(ContinuousApproximator):
    """ContinuousApproximator with differentiable calibration loss.

    Overrides ``compute_metrics`` to add a calibration loss term that
    penalizes coverage errors. The calibration loss weight is controlled
    by a schedulable gamma parameter.

    .. note::
        You must include :class:`CalibrationMonitorCallback` in
        ``callbacks`` when calling ``.fit()`` so the gamma schedule
        advances each epoch. Without it, ``_current_epoch`` stays
        at 0 and the schedule always returns ``gamma(0)``.

    Parameters
    ----------
    prior_fn : callable
        Function ``(n_samples: int) -> np.ndarray`` returning an
        array of shape ``(n_samples, param_dim)`` — i.i.d. samples
        from the marginal prior p(theta).
    gamma_schedule : GammaSchedule, optional
        Schedule for the calibration loss weight.
        Default: constant gamma=100.
    calibration_mode : float, optional
        Loss mode in [0, 1]. 0.0 = conservativeness (penalize
        under-coverage only), 1.0 = calibration (penalize both
        directions). Default: 0.0.
    n_rank_samples : int, optional
        Number of prior samples for rank computation. Must be
        positive. Default: 100.
    subsample_size : int or None, optional
        If set, subsample the batch to this size for calibration
        loss computation (reduces overhead). Must be positive if
        given. Default: None (use full batch).
    **kwargs
        All keyword arguments for
        ``ContinuousApproximator.__init__``.
    """

    def __init__(
        self,
        *,
        prior_fn: Callable[[int], np.ndarray],
        gamma_schedule: GammaSchedule | None = None,
        calibration_mode: float = 0.0,
        n_rank_samples: int = 100,
        subsample_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not (0.0 <= calibration_mode <= 1.0):
            raise ValueError(
                f"calibration_mode must be in [0, 1], "
                f"got {calibration_mode}."
            )
        if n_rank_samples < 1:
            raise ValueError(
                f"n_rank_samples must be >= 1, got {n_rank_samples}."
            )
        if subsample_size is not None and subsample_size < 1:
            raise ValueError(
                f"subsample_size must be >= 1, got {subsample_size}."
            )

        self.prior_fn = prior_fn
        self.gamma_schedule = gamma_schedule or GammaSchedule()
        self.calibration_mode = calibration_mode
        self.n_rank_samples = n_rank_samples
        self.subsample_size = subsample_size
        self._current_epoch = 0

    def compute_metrics(
        self,
        inference_variables,
        inference_conditions=None,
        summary_variables=None,
        sample_weight=None,
        stage="training",
    ):
        """Compute base NLL loss plus calibration loss.

        Calls the parent ``compute_metrics`` for the standard NLL
        loss, then adds the calibration loss term weighted by gamma.

        Parameters
        ----------
        inference_variables : tensor
            True parameter values, shape ``(batch, param_dim)``.
        inference_conditions : tensor or None
            Direct conditioning variables.
        summary_variables : tensor or None
            Set-based data for the summary network.
        sample_weight : tensor or None
            Per-sample weights.
        stage : str
            ``"training"`` or ``"validation"``.

        Returns
        -------
        dict
            Metrics dict with ``"loss"``, ``"calibration_loss"``,
            ``"gamma"``, plus all base metrics.
        """
        # 1. Get base metrics (NLL loss from normalizing flow)
        metrics = super().compute_metrics(
            inference_variables,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            sample_weight=sample_weight,
            stage=stage,
        )

        # 2. Only add calibration loss during training
        if stage != "training":
            return metrics

        gamma = self.gamma_schedule(self._current_epoch)
        if gamma <= 0:
            return metrics

        # 3. Optionally subsample batch for calibration loss
        batch_size = keras.ops.shape(inference_variables)[0]
        if (
            self.subsample_size is not None
            and self.subsample_size < batch_size
        ):
            idx = keras.random.shuffle(
                keras.ops.arange(batch_size)
            )[: self.subsample_size]
            inf_vars_sub = keras.ops.take(
                inference_variables, idx, axis=0
            )
            cond_sub = (
                keras.ops.take(inference_conditions, idx, axis=0)
                if inference_conditions is not None
                else None
            )
            summ_sub = (
                keras.ops.take(summary_variables, idx, axis=0)
                if summary_variables is not None
                else None
            )
        else:
            inf_vars_sub = inference_variables
            cond_sub = inference_conditions
            summ_sub = summary_variables

        # 4. Compute calibration loss
        cal_loss = self._calibration_loss(
            inf_vars_sub, summ_sub, cond_sub
        )

        # 5. Add weighted calibration loss to total
        metrics["loss"] = metrics["loss"] + gamma * cal_loss
        metrics["calibration_loss"] = cal_loss
        metrics["gamma"] = gamma

        return metrics

    def _calibration_loss(
        self, theta_true, summary_variables, inference_conditions
    ):
        """Compute calibration loss for a (sub-)batch.

        Parameters
        ----------
        theta_true : tensor
            True parameter values, shape ``(sub_batch, param_dim)``.
        summary_variables : tensor or None
            Set-based data (through summary network if present).
        inference_conditions : tensor or None
            Direct conditioning variables.

        Returns
        -------
        tensor
            Scalar calibration loss.
        """
        sub_batch = keras.ops.shape(theta_true)[0]

        # Build merged conditions (replicating base class logic)
        conditions = self._build_conditions(
            summary_variables, inference_conditions
        )

        # Log-prob of true theta under each observation's posterior
        log_prob_true = self.inference_network.log_prob(
            theta_true, conditions=conditions
        )

        # Sample from marginal prior and convert to tensor
        prior_samples_np = self.prior_fn(self.n_rank_samples)
        prior_samples = keras.ops.convert_to_tensor(
            prior_samples_np, dtype=theta_true.dtype
        )
        param_dim = keras.ops.shape(prior_samples)[-1]

        # Broadcast prior: (n_rank, dim) -> (sub_batch, n_rank, dim)
        prior_expanded = keras.ops.broadcast_to(
            keras.ops.expand_dims(prior_samples, axis=0),
            (sub_batch, self.n_rank_samples, param_dim),
        )

        # Broadcast cond: (sub_batch, D) -> (sub_batch, n_rank, D)
        cond_dim = keras.ops.shape(conditions)[-1]
        cond_expanded = keras.ops.broadcast_to(
            keras.ops.expand_dims(conditions, axis=1),
            (sub_batch, self.n_rank_samples, cond_dim),
        )

        # Flatten for batch forward pass through inference network
        prior_flat = keras.ops.reshape(
            prior_expanded, (-1, param_dim)
        )
        cond_flat = keras.ops.reshape(
            cond_expanded, (-1, cond_dim)
        )

        # Log-prob of all prior samples under each posterior
        log_probs_prior_flat = self.inference_network.log_prob(
            prior_flat, conditions=cond_flat
        )
        log_probs_prior = keras.ops.reshape(
            log_probs_prior_flat,
            (sub_batch, self.n_rank_samples),
        )

        # Compute ranks via STE
        ranks = compute_ranks(log_prob_true, log_probs_prior)

        # Compute coverage error
        return coverage_error(ranks, mode=self.calibration_mode)

    def _build_conditions(
        self, summary_variables, inference_conditions
    ):
        """Build merged conditions for the inference network.

        Replicates the condition-merging logic from the base
        class's ``compute_metrics`` so we can call
        ``inference_network.log_prob`` directly.

        .. warning::
            This duplicates base class logic and may need updating
            if BayesFlow changes its condition-merging internals.

        Parameters
        ----------
        summary_variables : tensor or None
            Set-based data for the summary network.
        inference_conditions : tensor or None
            Direct conditioning variables.

        Returns
        -------
        tensor
            Merged conditions for the inference network.
        """
        parts = []

        if inference_conditions is not None:
            parts.append(inference_conditions)

        if (
            self.summary_network is not None
            and summary_variables is not None
        ):
            summary_out = self.summary_network(
                summary_variables, training=True
            )
            parts.append(summary_out)

        if not parts:
            raise ValueError(
                "Calibration loss requires at least one of "
                "inference_conditions or summary_variables."
            )

        if len(parts) == 1:
            return parts[0]
        return keras.ops.concatenate(parts, axis=-1)
