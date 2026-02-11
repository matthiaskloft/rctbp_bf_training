"""Gamma weight scheduling for calibration loss.

Provides schedulable gamma weight to control how strongly the calibration
loss contributes to the total training objective. Starting with pure NLL
(gamma=0) and ramping up calibration pressure prevents the calibration loss
from dominating early when the network is still random.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GammaSchedule:
    """Schedule for the calibration loss weight gamma.

    Parameters
    ----------
    schedule_type : str
        One of ``"constant"``, ``"linear_warmup"``, ``"cosine"``, ``"step"``.
    gamma_max : float
        Maximum (or constant) gamma value.
    warmup_epochs : int
        Number of epochs before gamma reaches ``gamma_max``
        (used by ``linear_warmup`` and ``step``).
    total_epochs : int
        Total training epochs (used by ``cosine``).
    gamma_min : float
        Floor value for gamma.

    Examples
    --------
    >>> schedule = GammaSchedule(
    ...     schedule_type="linear_warmup", gamma_max=100, warmup_epochs=20
    ... )
    >>> schedule(0)
    0.0
    >>> schedule(10)
    50.0
    >>> schedule(20)
    100.0
    >>> schedule(50)
    100.0
    """

    schedule_type: str = "constant"
    gamma_max: float = 100.0
    warmup_epochs: int = 0
    total_epochs: int = 200
    gamma_min: float = 0.0

    def __post_init__(self):
        valid_types = {"constant", "linear_warmup", "cosine", "step"}
        if self.schedule_type not in valid_types:
            raise ValueError(
                f"Unknown schedule_type '{self.schedule_type}'. "
                f"Must be one of {valid_types}."
            )
        if self.gamma_max < self.gamma_min:
            raise ValueError(
                f"gamma_max ({self.gamma_max}) must be >= gamma_min ({self.gamma_min})."
            )

    def __call__(self, epoch: int) -> float:
        """Compute gamma value for the given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).

        Returns
        -------
        float
            Gamma weight for this epoch.
        """
        if self.schedule_type == "constant":
            return self.gamma_max

        elif self.schedule_type == "linear_warmup":
            if epoch < self.warmup_epochs:
                t = epoch / max(self.warmup_epochs, 1)
                return self.gamma_min + (self.gamma_max - self.gamma_min) * t
            return self.gamma_max

        elif self.schedule_type == "cosine":
            progress = min(epoch / max(self.total_epochs, 1), 1.0)
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
            return self.gamma_min + (self.gamma_max - self.gamma_min) * cosine_value

        elif self.schedule_type == "step":
            return self.gamma_max if epoch >= self.warmup_epochs else self.gamma_min

        # Should never reach here due to __post_init__ validation
        raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
