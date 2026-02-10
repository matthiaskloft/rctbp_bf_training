"""bayesflow-calibration: Differentiable calibration loss for BayesFlow NPE.

Implements the calibration loss from Falkner et al. (NeurIPS 2023) as a
standalone add-on for BayesFlow 2.x training.
"""

from bayesflow_calibration.approximator import CalibratedContinuousApproximator
from bayesflow_calibration.diagnostics import CalibrationMonitorCallback
from bayesflow_calibration.losses import compute_ranks, coverage_error, ste_indicator
from bayesflow_calibration.schedules import GammaSchedule

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CalibratedContinuousApproximator",
    "CalibrationMonitorCallback",
    "GammaSchedule",
    "ste_indicator",
    "compute_ranks",
    "coverage_error",
]
