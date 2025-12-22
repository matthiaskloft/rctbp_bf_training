"""
Bayesian Optimization Infrastructure for Neural Posterior Estimation.

Model-agnostic utilities for hyperparameter optimization using Optuna.
Supports multi-objective optimization (calibration error + parameter count).

Usage:
------
from rctbp_bf_training.core.optimization import (
    create_study,
    compute_composite_objective,
    get_param_count,
    OptunaReportCallback,
    plot_optimization_results,
    get_pareto_trials,
)

# Create multi-objective study
study = create_study(
    study_name="npe_optimization",
    directions=["minimize", "minimize"],  # [calibration, params]
    storage="sqlite:///optuna_study.db",
)

# In your objective function:
def objective(trial):
    # ... build and train model ...
    param_count = get_param_count(model)
    cal_error = metrics["summary"]["mean_cal_error"]
    return cal_error, param_count

study.optimize(objective, n_trials=50)
"""

import gc
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports with graceful fallback
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # type stub

try:
    import keras
    from keras.callbacks import Callback
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    Callback = object  # type stub


# =============================================================================
# STUDY CREATION
# =============================================================================

def create_study(
    study_name: str = "npe_optimization",
    directions: List[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
) -> "optuna.Study":
    """
    Create or load an Optuna study for hyperparameter optimization.
    
    Parameters:
    -----------
    study_name : str
        Name of the study (used for storage/resumption)
    directions : list of str
        Optimization directions. Default: ["minimize", "minimize"]
        for (calibration_error, param_count)
    storage : str, optional
        SQLite URL for persistence. Example: "sqlite:///optuna_study.db"
        If None, study is in-memory only.
    load_if_exists : bool
        If True, resume existing study with same name
    sampler : optuna.samplers.BaseSampler, optional
        Custom sampler. Default: NSGAIISampler for multi-objective
    pruner : optuna.pruners.BasePruner, optional
        Custom pruner for early stopping of bad trials
        
    Returns:
    --------
    optuna.Study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    if directions is None:
        directions = ["minimize", "minimize"]
    
    # Default sampler for multi-objective
    if sampler is None and len(directions) > 1:
        sampler = optuna.samplers.NSGAIISampler(seed=42)
    elif sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)
    
    # Default pruner (median-based)
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        )
    
    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    
    return study


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def get_param_count(model) -> int:
    """
    Count trainable parameters in a Keras model.
    
    Parameters:
    -----------
    model : keras.Model or BayesFlow approximator
        The model to count parameters for
        
    Returns:
    --------
    int : Total number of trainable parameters, or -1 if model not built
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
    
    Parameters:
    -----------
    summary_dim : int
        DeepSet output dimension
    deepset_width : int
        Width of DeepSet MLP layers
    deepset_depth : int
        Number of DeepSet aggregation stages
    flow_depth : int
        Number of coupling layers
    flow_hidden : int
        Hidden size in flow subnets
    n_conditions : int
        Number of conditioning variables
    n_params : int
        Number of parameters to infer
        
    Returns:
    --------
    int : Estimated parameter count
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
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute a single composite objective from multiple metrics.
    
    Use this for single-objective optimization when you want to
    combine calibration and model size into one score.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dict from validation pipeline with 'summary' key
    param_count : int
        Number of trainable parameters
    param_budget : int
        Target parameter count (penalty kicks in above this)
    weights : dict, optional
        Weights for each component. Default:
        {"calibration": 0.5, "sbc": 0.3, "size": 0.2}
        
    Returns:
    --------
    float : Composite objective (lower is better)
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
    size_penalty = max(0, np.log1p(param_count / param_budget) - np.log1p(1))
    
    composite = (
        weights["calibration"] * cal_error +
        weights["sbc"] * sbc_deviation +
        weights["size"] * size_penalty
    )
    
    return float(composite)


def extract_objective_values(
    metrics: Dict,
    param_count: int,
) -> Tuple[float, float]:
    """
    Extract objective values for multi-objective optimization.
    
    Returns (calibration_error, param_count) for Pareto optimization.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dict from validation pipeline
    param_count : int
        Number of trainable parameters
        
    Returns:
    --------
    tuple : (calibration_error, param_count)
    """
    summary = metrics.get("summary", metrics)
    
    # Primary calibration metric: mean absolute calibration error
    cal_error = summary.get("mean_cal_error", 1.0)
    
    # Could also incorporate SBC
    c2st = summary.get("sbc_c2st_accuracy", 0.5)
    sbc_penalty = abs(c2st - 0.5)
    
    # Combined calibration score
    calibration = cal_error + 0.5 * sbc_penalty
    
    return float(calibration), int(param_count)


# =============================================================================
# KERAS CALLBACK FOR OPTUNA
# =============================================================================

class OptunaReportCallback(Callback):
    """
    Keras callback for reporting metrics to Optuna and handling pruning.
    
    Reports validation loss at each epoch, allowing Optuna to prune
    unpromising trials early.
    
    Parameters:
    -----------
    trial : optuna.Trial
        The current Optuna trial
    monitor : str
        Metric to report (default: "val_loss")
    report_frequency : int
        Report every N epochs (default: 1)
        
    Example:
    --------
    callback = OptunaReportCallback(trial)
    history = workflow.fit_online(
        epochs=100,
        callbacks=[callback]
    )
    """
    
    def __init__(
        self,
        trial: "Trial",
        monitor: str = "val_loss",
        report_frequency: int = 1,
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.report_frequency = report_frequency
        self._epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch
        
        if logs is None:
            return
        
        if epoch % self.report_frequency != 0:
            return
        
        value = logs.get(self.monitor)
        if value is None:
            return
        
        # Report to Optuna
        self.trial.report(float(value), step=epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# =============================================================================
# HYPERPARAMETER SAMPLING HELPERS
# =============================================================================

@dataclass
class HyperparameterSpace:
    """
    Define hyperparameter search space for NPE models.
    
    Attributes define ranges for each hyperparameter.
    Use with `sample_hyperparameters(trial, space)`.
    """
    # DeepSet
    summary_dim: Tuple[int, int] = (4, 16)
    deepset_width: Tuple[int, int] = (32, 96)
    deepset_depth: Tuple[int, int] = (1, 3)
    deepset_dropout: Tuple[float, float] = (0.0, 0.15)
    
    # CouplingFlow
    flow_depth: Tuple[int, int] = (3, 8)
    flow_hidden: Tuple[int, int] = (32, 128)
    flow_dropout: Tuple[float, float] = (0.05, 0.25)
    
    # Training
    initial_lr: Tuple[float, float] = (1e-4, 5e-3)
    batch_size: Tuple[int, int] = (128, 384)
    
    # Fixed values (not optimized)
    decay_rate: float = 0.85
    patience: int = 15
    window: int = 15


def sample_hyperparameters(
    trial: "Trial",
    space: HyperparameterSpace = None,
) -> Dict[str, Any]:
    """
    Sample hyperparameters from search space using Optuna trial.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    space : HyperparameterSpace, optional
        Search space definition. Default: HyperparameterSpace()
        
    Returns:
    --------
    dict : Sampled hyperparameters
    """
    if space is None:
        space = HyperparameterSpace()
    
    params = {
        # DeepSet
        "summary_dim": trial.suggest_int(
            "summary_dim", space.summary_dim[0], space.summary_dim[1]
        ),
        "deepset_width": trial.suggest_int(
            "deepset_width", space.deepset_width[0], space.deepset_width[1], step=16
        ),
        "deepset_depth": trial.suggest_int(
            "deepset_depth", space.deepset_depth[0], space.deepset_depth[1]
        ),
        "deepset_dropout": trial.suggest_float(
            "deepset_dropout", space.deepset_dropout[0], space.deepset_dropout[1]
        ),
        
        # CouplingFlow
        "flow_depth": trial.suggest_int(
            "flow_depth", space.flow_depth[0], space.flow_depth[1]
        ),
        "flow_hidden": trial.suggest_int(
            "flow_hidden", space.flow_hidden[0], space.flow_hidden[1], step=16
        ),
        "flow_dropout": trial.suggest_float(
            "flow_dropout", space.flow_dropout[0], space.flow_dropout[1]
        ),
        
        # Training
        "initial_lr": trial.suggest_float(
            "initial_lr", space.initial_lr[0], space.initial_lr[1], log=True
        ),
        "batch_size": trial.suggest_int(
            "batch_size", space.batch_size[0], space.batch_size[1], step=64
        ),
        
        # Fixed
        "decay_rate": space.decay_rate,
        "patience": space.patience,
        "window": space.window,
    }
    
    return params


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def get_pareto_trials(
    study: "optuna.Study",
) -> List["optuna.trial.FrozenTrial"]:
    """
    Get Pareto-optimal trials from a multi-objective study.
    
    Parameters:
    -----------
    study : optuna.Study
        Completed or in-progress study
        
    Returns:
    --------
    list : Pareto-optimal FrozenTrial objects
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required")
    
    return study.best_trials


def trials_to_dataframe(
    study: "optuna.Study",
    include_pruned: bool = False,
) -> pd.DataFrame:
    """
    Convert study trials to a DataFrame for analysis.
    
    Parameters:
    -----------
    study : optuna.Study
        The study to extract trials from
    include_pruned : bool
        Whether to include pruned trials
        
    Returns:
    --------
    pd.DataFrame with columns for params and objectives
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required")
    
    records = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            record = {"trial_number": trial.number}
            record.update(trial.params)
            
            # Handle multi-objective
            if isinstance(trial.values, (list, tuple)):
                record["objective_0"] = trial.values[0]
                if len(trial.values) > 1:
                    record["objective_1"] = trial.values[1]
            else:
                record["objective"] = trial.values
            
            record["duration_s"] = (
                trial.datetime_complete - trial.datetime_start
            ).total_seconds() if trial.datetime_complete else None
            
            records.append(record)
        elif include_pruned and trial.state == optuna.trial.TrialState.PRUNED:
            record = {"trial_number": trial.number, "pruned": True}
            record.update(trial.params)
            records.append(record)
    
    return pd.DataFrame(records)


def summarize_best_trials(
    study: "optuna.Study",
    n_best: int = 5,
) -> pd.DataFrame:
    """
    Summarize the best trials from a study.
    
    For multi-objective: returns Pareto-optimal trials.
    For single-objective: returns top N trials by objective.
    
    Parameters:
    -----------
    study : optuna.Study
        The completed study
    n_best : int
        Number of best trials to return (for single-objective)
        
    Returns:
    --------
    pd.DataFrame with trial parameters and objectives
    """
    if len(study.directions) > 1:
        # Multi-objective: get Pareto front
        best_trials = study.best_trials
    else:
        # Single-objective: get top N
        best_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value,
        )[:n_best]
    
    records = []
    for trial in best_trials:
        record = {"trial": trial.number}
        record.update(trial.params)
        
        if isinstance(trial.values, (list, tuple)):
            record["cal_error"] = trial.values[0]
            if len(trial.values) > 1:
                record["param_count"] = int(trial.values[1])
        else:
            record["objective"] = trial.value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by calibration error if multi-objective
    if "cal_error" in df.columns:
        df = df.sort_values("cal_error")
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_optimization_results(
    study: "optuna.Study",
    figsize: Tuple[int, int] = (14, 5),
) -> "matplotlib.figure.Figure":
    """
    Plot optimization results including Pareto front and parameter importance.
    
    Parameters:
    -----------
    study : optuna.Study
        Completed study
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Pareto front (for multi-objective)
    ax = axes[0]
    if len(study.directions) > 1:
        trials_df = trials_to_dataframe(study)
        if "objective_0" in trials_df.columns and "objective_1" in trials_df.columns:
            # Filter out failed trials (those with penalty values)
            # Failed trials have cal_error=1.0 and param_count=1e9
            valid_mask = (trials_df["objective_0"] < 1.0) & (trials_df["objective_1"] < 1e8)
            valid_df = trials_df[valid_mask]
            
            if len(valid_df) > 0:
                ax.scatter(
                    valid_df["objective_0"],
                    valid_df["objective_1"],
                    alpha=0.5,
                    label=f"Successful trials ({len(valid_df)})",
                )
            
            # Show failed trials count if any
            n_failed = len(trials_df) - len(valid_df)
            if n_failed > 0:
                ax.text(0.95, 0.95, f"Failed: {n_failed}",
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, color='gray')
            
            # Highlight Pareto front (filter valid ones)
            pareto = study.best_trials
            pareto_obj0 = [t.values[0] for t in pareto if t.values[0] < 1.0 and t.values[1] < 1e8]
            pareto_obj1 = [t.values[1] for t in pareto if t.values[0] < 1.0 and t.values[1] < 1e8]
            if pareto_obj0:
                ax.scatter(
                    pareto_obj0, pareto_obj1,
                    c='red', s=100, marker='*',
                    label="Pareto front",
                    zorder=10,
                )
            
            ax.set_xlabel("Calibration Error")
            ax.set_ylabel("Parameter Count")
            ax.set_title("Pareto Front")
            ax.legend()
    else:
        # Single objective: plot optimization history
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in trials]
        ax.plot(range(len(values)), values, 'o-', alpha=0.7)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective")
        ax.set_title("Optimization History")
    
    # 2. Parameter importance
    ax = axes[1]
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())[:10]  # Top 10
        values = [importance[p] for p in params]
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importance")
    except Exception:
        ax.text(0.5, 0.5, "Importance N/A\n(need more trials)",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Parameter Importance")
    
    # 3. Best trial summary
    ax = axes[2]
    ax.axis('off')
    
    best_df = summarize_best_trials(study, n_best=3)
    if len(best_df) > 0:
        # Format as text table
        text_lines = ["Best Configurations:\n"]
        for idx, row in best_df.head(3).iterrows():
            text_lines.append(f"Trial {int(row.get('trial', idx))}:")
            for col in best_df.columns:
                if col != 'trial':
                    val = row[col]
                    if isinstance(val, float):
                        text_lines.append(f"  {col}: {val:.4g}")
                    else:
                        text_lines.append(f"  {col}: {val}")
            text_lines.append("")
        
        ax.text(0.1, 0.9, "\n".join(text_lines[:20]),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace')
    
    ax.set_title("Best Configurations")
    
    plt.tight_layout()
    return fig


def plot_pareto_front(
    study: "optuna.Study",
    ax: Optional[Any] = None,
    highlight_best: bool = True,
) -> Any:
    """
    Plot the Pareto front for a multi-objective study.
    
    Parameters:
    -----------
    study : optuna.Study
        Multi-objective study
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    highlight_best : bool
        Whether to highlight Pareto-optimal points
        
    Returns:
    --------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    trials_df = trials_to_dataframe(study)
    
    if "objective_0" not in trials_df.columns:
        ax.text(0.5, 0.5, "Single objective study",
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # All trials
    ax.scatter(
        trials_df["objective_0"],
        trials_df["objective_1"],
        alpha=0.4,
        s=50,
        label="All trials",
    )
    
    if highlight_best:
        pareto = study.best_trials
        pareto_obj0 = [t.values[0] for t in pareto]
        pareto_obj1 = [t.values[1] for t in pareto]
        
        # Sort for line plot
        sorted_idx = np.argsort(pareto_obj0)
        pareto_obj0 = np.array(pareto_obj0)[sorted_idx]
        pareto_obj1 = np.array(pareto_obj1)[sorted_idx]
        
        ax.plot(pareto_obj0, pareto_obj1, 'r--', alpha=0.7, linewidth=2)
        ax.scatter(
            pareto_obj0, pareto_obj1,
            c='red', s=150, marker='*',
            label="Pareto front",
            zorder=10,
            edgecolors='black',
        )
    
    ax.set_xlabel("Calibration Error", fontsize=12)
    ax.set_ylabel("Parameter Count", fontsize=12)
    ax.set_title("Multi-Objective Optimization: Pareto Front", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


# =============================================================================
# GPU MEMORY CLEANUP
# =============================================================================

def cleanup_trial():
    """
    Clean up GPU memory after a trial completes.
    
    Call this at the end of each trial to prevent memory leaks.
    """
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass


# =============================================================================
# MAIN OPTIMIZATION RUNNER
# =============================================================================

def run_optimization(
    objective_fn: Callable[["Trial"], Union[float, Tuple[float, float]]],
    n_trials: int = 50,
    study_name: str = "npe_optimization",
    directions: List[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
    timeout: Optional[float] = None,
    show_progress_bar: bool = True,
) -> "optuna.Study":
    """
    Run hyperparameter optimization with sensible defaults.
    
    Parameters:
    -----------
    objective_fn : callable
        Objective function taking an Optuna trial and returning objective value(s)
    n_trials : int
        Number of optimization trials
    study_name : str
        Name for the study
    directions : list of str
        Optimization directions. Default: ["minimize", "minimize"]
    storage : str, optional
        SQLite URL for persistence
    n_jobs : int
        Number of parallel jobs (1 = sequential)
    timeout : float, optional
        Time limit in seconds
    show_progress_bar : bool
        Whether to show Optuna progress bar
        
    Returns:
    --------
    optuna.Study : Completed study
    
    Example:
    --------
    def objective(trial):
        params = sample_hyperparameters(trial)
        # ... build model, train, validate ...
        return cal_error, param_count
    
    study = run_optimization(objective, n_trials=50)
    """
    if directions is None:
        directions = ["minimize", "minimize"]
    
    study = create_study(
        study_name=study_name,
        directions=directions,
        storage=storage,
    )
    
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )
    
    return study


# =============================================================================
# THRESHOLD-BASED TRAINING LOOP
# =============================================================================

@dataclass
class QualityThresholds:
    """
    Quality thresholds for convergence checking.
    
    Training continues until all thresholds are met or max_iterations reached.
    """
    max_cal_error: float = 0.02       # Maximum mean calibration error
    max_c2st_deviation: float = 0.05  # Maximum |C2ST - 0.5|
    max_coverage_error: float = 0.03  # Maximum coverage deviation from nominal
    max_iterations: int = 10          # Maximum training attempts
    min_improvement: float = 0.001    # Minimum improvement to continue


def check_thresholds(
    metrics: Dict,
    thresholds: QualityThresholds,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if metrics meet quality thresholds.
    
    Parameters:
    -----------
    metrics : dict
        Metrics from validation pipeline with 'summary' key
    thresholds : QualityThresholds
        Threshold values to check against
        
    Returns:
    --------
    tuple : (passed: bool, scores: dict with individual metric values)
    """
    summary = metrics.get("summary", metrics)
    
    cal_error = summary.get("mean_cal_error", 1.0)
    c2st = summary.get("sbc_c2st_accuracy", 0.5)
    c2st_deviation = abs(c2st - 0.5)
    
    # Coverage at 95% level
    cov_95 = summary.get("coverage_95", 0.95)
    coverage_error = abs(cov_95 - 0.95)
    
    scores = {
        "cal_error": cal_error,
        "c2st_deviation": c2st_deviation,
        "coverage_error": coverage_error,
    }
    
    passed = (
        cal_error <= thresholds.max_cal_error and
        c2st_deviation <= thresholds.max_c2st_deviation and
        coverage_error <= thresholds.max_coverage_error
    )
    
    return passed, scores


def train_until_threshold(
    build_workflow_fn: Callable[[Dict], Any],
    train_fn: Callable[[Any], Any],
    validate_fn: Callable[[Any], Dict],
    hyperparams: Dict,
    thresholds: QualityThresholds = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a model repeatedly until quality thresholds are met.
    
    This function implements a retry loop that:
    1. Builds a fresh model with the given hyperparameters
    2. Trains the model
    3. Validates on a strict evaluation grid
    4. Checks if thresholds are met
    5. If not met and improving, retrain from scratch
    
    Parameters:
    -----------
    build_workflow_fn : callable
        Function(hyperparams) -> workflow/model
        Builds a fresh model from hyperparameters
    train_fn : callable
        Function(workflow) -> history
        Trains the workflow and returns training history
    validate_fn : callable
        Function(workflow) -> metrics_dict
        Validates the trained model on the strict grid
    hyperparams : dict
        Hyperparameters from best Optuna trial
    thresholds : QualityThresholds, optional
        Quality thresholds. Default: QualityThresholds()
    checkpoint_path : str, optional
        Path to save best model checkpoint
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict with keys:
        - 'workflow': trained workflow/model
        - 'metrics': final validation metrics
        - 'history': list of all training histories
        - 'iterations': number of training iterations
        - 'converged': whether thresholds were met
        - 'best_scores': best scores achieved
    """
    if thresholds is None:
        thresholds = QualityThresholds()
    
    best_workflow = None
    best_metrics = None
    best_composite = float('inf')
    best_scores = None
    all_histories = []
    converged = False
    
    for iteration in range(1, thresholds.max_iterations + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Iteration {iteration}/{thresholds.max_iterations}")
            print(f"{'='*60}")
        
        # Build fresh model
        workflow = build_workflow_fn(hyperparams)
        
        # Train
        try:
            history = train_fn(workflow)
            all_histories.append(history)
        except Exception as e:
            if verbose:
                print(f"Training failed: {e}")
            cleanup_trial()
            continue
        
        # Validate on strict grid
        try:
            metrics = validate_fn(workflow)
        except Exception as e:
            if verbose:
                print(f"Validation failed: {e}")
            cleanup_trial()
            continue
        
        # Check thresholds
        passed, scores = check_thresholds(metrics, thresholds)
        
        # Compute composite score for comparison
        composite = scores["cal_error"] + scores["c2st_deviation"] + scores["coverage_error"]
        
        if verbose:
            print(f"\nIteration {iteration} Results:")
            print(f"  Calibration Error: {scores['cal_error']:.4f} (threshold: {thresholds.max_cal_error})")
            print(f"  C2ST Deviation:    {scores['c2st_deviation']:.4f} (threshold: {thresholds.max_c2st_deviation})")
            print(f"  Coverage Error:    {scores['coverage_error']:.4f} (threshold: {thresholds.max_coverage_error})")
            print(f"  Composite Score:   {composite:.4f}")
            print(f"  Passed: {'✓ YES' if passed else '✗ NO'}")
        
        # Track best
        if composite < best_composite:
            improvement = best_composite - composite
            best_composite = composite
            best_workflow = workflow
            best_metrics = metrics
            best_scores = scores
            
            if verbose and iteration > 1:
                print(f"  → New best! Improvement: {improvement:.4f}")
            
            # Save checkpoint
            if checkpoint_path is not None:
                try:
                    if hasattr(workflow, 'approximator'):
                        workflow.approximator.save(checkpoint_path)
                    else:
                        workflow.save(checkpoint_path)
                    if verbose:
                        print(f"  → Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    if verbose:
                        print(f"  → Checkpoint save failed: {e}")
        else:
            improvement = best_composite - composite
        
        # Check convergence
        if passed:
            converged = True
            if verbose:
                print(f"\n✓ Thresholds met at iteration {iteration}!")
            break
        
        # Check if still improving enough to continue
        if iteration > 1 and improvement < thresholds.min_improvement:
            if verbose:
                print(f"\n⚠ Insufficient improvement ({improvement:.4f} < {thresholds.min_improvement})")
                print("  Consider adjusting architecture or training settings.")
        
        # Cleanup for next iteration
        if workflow is not best_workflow:
            del workflow
        cleanup_trial()
    
    if not converged and verbose:
        print(f"\n⚠ Max iterations reached without meeting thresholds.")
        print(f"  Best composite score: {best_composite:.4f}")
        print(f"  Best individual scores: {best_scores}")
    
    return {
        'workflow': best_workflow,
        'metrics': best_metrics,
        'history': all_histories,
        'iterations': len(all_histories),
        'converged': converged,
        'best_scores': best_scores,
    }


def create_strict_validation_grid(
    N_vals: List[int] = None,
    p_alloc_vals: List[float] = None,
    prior_df_vals: List[int] = None,
    prior_scale_vals: List[float] = None,
    b_group_vals: List[float] = None,
    b_covariate_vals: List[float] = None,
) -> List[Dict]:
    """
    Create a strict validation grid covering the full parameter space.
    
    Default creates a comprehensive 144-condition grid for thorough validation.
    
    Parameters:
    -----------
    N_vals : list of int
        Sample sizes to test. Default: [20, 100, 500, 1000]
    p_alloc_vals : list of float
        Allocation probabilities. Default: [0.5, 0.7]
    prior_df_vals : list of int
        Prior degrees of freedom. Default: [0, 3, 10, 30]
    prior_scale_vals : list of float
        Prior scales. Default: [0.5, 2.0, 5.0]
    b_group_vals : list of float
        True treatment effects. Default: [0.0, 0.3, 0.7]
    b_covariate_vals : list of float
        Covariate effects. Default: [0.0]
        
    Returns:
    --------
    list of dict : Condition dictionaries
    """
    from itertools import product
    
    if N_vals is None:
        N_vals = [20, 100, 500, 1000]
    if p_alloc_vals is None:
        p_alloc_vals = [0.5, 0.7]
    if prior_df_vals is None:
        prior_df_vals = [0, 3, 10, 30]
    if prior_scale_vals is None:
        prior_scale_vals = [0.5, 2.0, 5.0]
    if b_group_vals is None:
        b_group_vals = [0.0, 0.3, 0.7]
    if b_covariate_vals is None:
        b_covariate_vals = [0.0]
    
    conditions = []
    for idx, (n, p, pdf, psc, b_grp, b_cov) in enumerate(product(
        N_vals, p_alloc_vals, prior_df_vals, prior_scale_vals, b_group_vals, b_covariate_vals
    )):
        conditions.append({
            "id_cond": idx,
            "n_total": n,
            "p_alloc": p,
            "prior_df": pdf,
            "prior_scale": psc,
            "b_arm_treat": b_grp,
            "b_covariate": b_cov,
        })
    
    return conditions


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Bayesian Optimization Infrastructure")
    print("=" * 40)
    
    # Test parameter estimation
    est = estimate_param_count(
        summary_dim=8,
        deepset_width=64,
        deepset_depth=2,
        flow_depth=6,
        flow_hidden=128,
    )
    print(f"Estimated params (current config): {est:,}")
    
    # Test search space
    space = HyperparameterSpace()
    print(f"\nSearch space:")
    print(f"  summary_dim: {space.summary_dim}")
    print(f"  flow_depth: {space.flow_depth}")
    print(f"  initial_lr: {space.initial_lr}")
    
    # Test study creation (if Optuna available)
    if OPTUNA_AVAILABLE:
        study = create_study(study_name="test", storage=None)
        print(f"\nOptuna study created: {study.study_name}")
        print(f"  Directions: {study.directions}")
    else:
        print("\nOptuna not installed - skipping study test")
