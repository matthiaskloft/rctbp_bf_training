"""
Utility functions for the rctnpe project.
"""

import numpy as np
from keras.callbacks import Callback


def loguniform_int(low, high, alpha=1.0, rng=np.random):
    """
    Sample an integer from a generalized log-uniform distribution.
    
    Samples in log-space with optional skew controlled by alpha.
    
    Parameters:
    -----------
    low : int
        Lower bound (inclusive)
    high : int
        Upper bound (inclusive)
    alpha : float
        Shape parameter controlling the distribution skew:
        - alpha = 1.0: standard log-uniform (equal weight per order of magnitude)
        - alpha < 1.0: more weight toward lower values
        - alpha > 1.0: more weight toward higher values
    rng : numpy random generator
        Random number generator (default: np.random)
    
    Returns:
    --------
    int : Sampled value in [low, high]
    """
    log_low = np.log(low)
    log_high = np.log(high)
    # Sample uniform, apply power transformation for skew, then scale
    # Use 1/alpha so that alpha > 1 gives more weight to higher values
    u = rng.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    return int(np.round(np.exp(log_val)))


def loguniform_float(low, high, alpha=1.0, rng=np.random):
    """
    Sample a float from a generalized log-uniform distribution.
    
    Parameters:
    -----------
    low : float
        Lower bound (inclusive)
    high : float
        Upper bound (inclusive)
    alpha : float
        Shape parameter controlling the distribution skew:
        - alpha = 1.0: standard log-uniform (equal weight per order of magnitude)
        - alpha < 1.0: more weight toward lower values
        - alpha > 1.0: more weight toward higher values
    rng : numpy random generator
        Random number generator (default: np.random)
    
    Returns:
    --------
    float : Sampled value in [low, high]
    """
    log_low = np.log(low)
    log_high = np.log(high)
    # Use 1/alpha so that alpha > 1 gives more weight to higher values
    u = rng.uniform(0, 1) ** (1.0 / alpha)
    log_val = log_low + u * (log_high - log_low)
    return np.exp(log_val)


def sample_t_or_normal(df, scale=1.0, rng=np.random):
    """
    Sample from a Student-t or Normal distribution.
    
    Uses Student-t when df is in [1, 100], otherwise uses Normal.
    As df -> infinity, Student-t converges to Normal.
    
    Parameters:
    -----------
    df : float
        Degrees of freedom for Student-t. If df <= 0 or df > 100,
        uses Normal distribution instead.
    scale : float
        Scale parameter (standard deviation for Normal, scale for t).
    rng : numpy random generator
        Random number generator (default: np.random)
    
    Returns:
    --------
    float : Sampled value with mean 0 and specified scale
    """
    if df <= 0 or df > 100:
        # Use normal distribution
        return rng.normal(0, scale)
    else:
        # Use Student-t distribution
        return rng.standard_t(df) * scale


class MovingAverageEarlyStopping(Callback):
    """
    Stop training when moving average of val_loss increases for given patience.

    This is more stable than standard EarlyStopping which can be sensitive to
    epoch-to-epoch variance.

    Parameters
    ----------
    window : int
        Number of epochs to average over.
    patience : int
        Number of epochs with no improvement to wait before stopping.
    restore_best_weights : bool
        Whether to restore model weights from the epoch with the best
        moving average validation loss.
    """

    def __init__(self, window: int = 5, patience: int = 3, restore_best_weights: bool = True):
        super().__init__()
        self.window = window
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.val_losses = []
        self.moving_averages = []
        self.wait = 0
        self.best_weights = None
        self.best_ma_loss = np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        self.val_losses.append(val_loss)
        if len(self.val_losses) > self.window:
            self.val_losses.pop(0)

        moving_avg = np.mean(self.val_losses)
        self.moving_averages.append(moving_avg)

        if moving_avg < self.best_ma_loss:
            self.best_ma_loss = moving_avg
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

        logs["moving_avg_val_loss"] = moving_avg

    def on_train_end(self, logs=None):
        pass  # Silent
