"""Training-time calibration monitoring callback.

Provides a Keras callback that updates the epoch counter on
CalibratedContinuousApproximator so the gamma schedule advances correctly.
"""

from __future__ import annotations

import keras


class CalibrationMonitorCallback(keras.callbacks.Callback):
    """Update epoch counter and log calibration metrics during training.

    This callback must be used with ``CalibratedContinuousApproximator``
    to ensure the gamma schedule advances each epoch.

    The callback updates ``model._current_epoch`` at the start of each
    epoch, which the approximator reads when computing calibration loss.

    Examples
    --------
    >>> from bayesflow_calibration import CalibrationMonitorCallback
    >>> approximator.fit(
    ...     ..., callbacks=[CalibrationMonitorCallback()]
    ... )
    """

    def on_epoch_begin(self, epoch, logs=None):
        """Set the current epoch on the model for gamma scheduling."""
        if hasattr(self.model, "_current_epoch"):
            self.model._current_epoch = epoch
