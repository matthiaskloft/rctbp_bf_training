"""
Bayesian Optimization Infrastructure for Neural Posterior Estimation.

Model-agnostic utilities for hyperparameter optimization using Optuna.
Supports multi-objective optimization (calibration error + parameter count).

Usage
-----
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with graceful fallback
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # type stub

try:
    from keras.callbacks import Callback
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    Callback = object  # type stub

# Import from split modules (canonical locations)
from rctbp_bf_training.core.objectives import (  # noqa: I001
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_PARAM_SCORE,
    extract_objective_values,
    get_param_count,
)


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

    Parameters
    ----------
    study_name : str
        Name of the study (used for storage/resumption).
    directions : list of str
        Optimization directions. Default: ["minimize", "minimize"]
        for (calibration_error, normalized_param_score).
    storage : str, optional
        SQLite URL for persistence. Example: "sqlite:///optuna_study.db"
        If None, study is in-memory only.
    load_if_exists : bool
        If True, resume existing study with same name.
    sampler : optuna.samplers.BaseSampler, optional
        Custom sampler. Default: MOTPESampler for multi-objective (better
        sample efficiency than NSGA-II for 50-200 trials).
    pruner : optuna.pruners.BasePruner, optional
        Custom pruner for early stopping of bad trials.

    Returns
    -------
    optuna.Study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required. Install with: pip install optuna"
        )

    if directions is None:
        directions = ["minimize", "minimize"]

    # Default sampler: MOTPE for multi-objective
    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=10,
        )

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
# KERAS CALLBACK FOR OPTUNA
# =============================================================================

class OptunaReportCallback(Callback):
    """
    Keras callback for reporting metrics to Optuna and handling pruning.

    Reports validation loss at each epoch, allowing Optuna to prune
    unpromising trials early.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.
    monitor : str
        Metric to report (default: "val_loss").
    report_frequency : int
        Report every N epochs (default: 1).

    Examples
    --------
    >>> callback = OptunaReportCallback(trial)
    >>> history = workflow.fit_online(
    ...     epochs=100,
    ...     callbacks=[callback]
    ... )
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

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
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
    Use with ``sample_hyperparameters(trial, space)``.

    Attributes
    ----------
    summary_dim : tuple of (int, int)
        Range for DeepSet output dimension.
    deepset_width : tuple of (int, int)
        Range for DeepSet MLP width.
    deepset_depth : tuple of (int, int)
        Range for DeepSet depth.
    deepset_dropout : tuple of (float, float)
        Range for DeepSet dropout.
    flow_depth : tuple of (int, int)
        Range for CouplingFlow depth.
    flow_hidden : tuple of (int, int)
        Range for CouplingFlow hidden size.
    flow_dropout : tuple of (float, float)
        Range for CouplingFlow dropout.
    initial_lr : tuple of (float, float)
        Range for initial learning rate.
    batch_size : tuple of (int, int)
        Range for batch size.
    decay_rate : float
        Fixed learning rate decay (not optimized).
    patience : int
        Fixed early stopping patience (not optimized).
    window : int
        Fixed moving average window (not optimized).
    """
    # DeepSet
    summary_dim: Tuple[int, int] = (4, 16)
    deepset_width: Tuple[int, int] = (32, 128)
    deepset_depth: Tuple[int, int] = (1, 4)
    deepset_dropout: Tuple[float, float] = (0.0, 0.3)

    # CouplingFlow
    flow_depth: Tuple[int, int] = (2, 8)
    flow_hidden: Tuple[int, int] = (32, 128)
    flow_dropout: Tuple[float, float] = (0.05, 0.3)

    # Training
    initial_lr: Tuple[float, float] = (1e-4, 5e-3)
    batch_size: Tuple[int, int] = (64, 1024)

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

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    space : HyperparameterSpace, optional
        Search space definition. Default: HyperparameterSpace().

    Returns
    -------
    dict
        Sampled hyperparameters.
    """
    if space is None:
        space = HyperparameterSpace()

    params = {
        # DeepSet
        "summary_dim": trial.suggest_int(
            "summary_dim", space.summary_dim[0], space.summary_dim[1],
        ),
        "deepset_width": trial.suggest_int(
            "deepset_width",
            space.deepset_width[0], space.deepset_width[1],
            step=16,
        ),
        "deepset_depth": trial.suggest_int(
            "deepset_depth",
            space.deepset_depth[0], space.deepset_depth[1],
        ),
        "deepset_dropout": trial.suggest_float(
            "deepset_dropout",
            space.deepset_dropout[0], space.deepset_dropout[1],
        ),
        # CouplingFlow
        "flow_depth": trial.suggest_int(
            "flow_depth", space.flow_depth[0], space.flow_depth[1],
        ),
        "flow_hidden": trial.suggest_int(
            "flow_hidden",
            space.flow_hidden[0], space.flow_hidden[1],
            step=16,
        ),
        "flow_dropout": trial.suggest_float(
            "flow_dropout",
            space.flow_dropout[0], space.flow_dropout[1],
        ),
        # Training
        "initial_lr": trial.suggest_float(
            "initial_lr",
            space.initial_lr[0], space.initial_lr[1],
            log=True,
        ),
        "batch_size": trial.suggest_int(
            "batch_size",
            space.batch_size[0], space.batch_size[1],
            step=64,
        ),
        # Fixed
        "decay_rate": space.decay_rate,
        "patience": space.patience,
        "window": space.window,
    }

    return params


# =============================================================================
# GPU MEMORY CLEANUP
# =============================================================================

def cleanup_trial() -> None:
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
    objective_fn: Callable[
        ["Trial"], Union[float, Tuple[float, float]]
    ],
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

    Parameters
    ----------
    objective_fn : callable
        Objective function taking an Optuna trial and returning
        objective value(s).
    n_trials : int
        Number of optimization trials.
    study_name : str
        Name for the study.
    directions : list of str
        Optimization directions. Default: ["minimize", "minimize"].
    storage : str, optional
        SQLite URL for persistence.
    n_jobs : int
        Number of parallel jobs (1 = sequential).
    timeout : float, optional
        Time limit in seconds.
    show_progress_bar : bool
        Whether to show Optuna progress bar.

    Returns
    -------
    optuna.Study
        Completed study.

    Examples
    --------
    >>> def objective(trial):
    ...     params = sample_hyperparameters(trial)
    ...     # ... build model, train, validate ...
    ...     return cal_error, param_count
    >>>
    >>> study = run_optimization(objective, n_trials=50)
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
# GENERIC OBJECTIVE FUNCTION BUILDER
# =============================================================================

def create_optimization_objective(
    config: Any,
    simulator: Any,
    adapter: Any,
    search_space: HyperparameterSpace,
    validation_conditions: List[Dict],
    inference_conditions: List[str],
    param_key: str,
    data_keys: List[str],
    context_keys: Dict[str, type],
    true_param_key: str,
    simulate_fn_factory: Callable,
    n_sims: int = 500,
    n_post_draws: int = 500,
    rng: np.random.Generator = None,
) -> Callable:
    """
    Create a generic Optuna objective function for BayesFlow optimization.

    This is a model-agnostic factory that creates objective functions for
    hyperparameter optimization. Model-specific wrappers should call this
    function with their specific parameter keys and configurations.

    Parameters
    ----------
    config : object with workflow.training attributes
        Configuration object containing training settings
        (epochs, batches_per_epoch, validation_sims).
    simulator : bf.Simulator
        BayesFlow simulator for the model.
    adapter : bf.Adapter
        BayesFlow adapter for data transformation.
    search_space : HyperparameterSpace
        Search space for hyperparameter sampling.
    validation_conditions : list of dict
        Conditions grid for validation.
    inference_conditions : list of str
        List of context variable names
        (e.g., ["N", "p_alloc", "prior_df", "prior_scale"]).
    param_key : str
        Key for the parameter in the workflow (e.g., "b_group").
    data_keys : list of str
        Keys for data in inference
        (e.g., ["outcome", "covariate", "group"]).
    context_keys : dict
        Mapping of context keys to their types
        (e.g., {"N": int, "p_alloc": float}).
    true_param_key : str
        Key for true parameter in validation (e.g., "b_arm_treat").
    simulate_fn_factory : callable
        Function that takes rng and returns a simulate function.
    n_sims : int, default=500
        Number of simulations per condition for validation.
    n_post_draws : int, default=500
        Number of posterior draws per simulation.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    callable
        Objective function that takes an Optuna trial and returns
        (calibration_error, parameter_count).

    Examples
    --------
    >>> from rctbp_bf_training.core.optimization import (
    ...     create_optimization_objective,
    ... )
    >>>
    >>> # Model-specific wrapper
    >>> def create_my_model_objective(
    ...     config, simulator, adapter, search_space, conditions
    ... ):
    ...     return create_optimization_objective(
    ...         config=config,
    ...         simulator=simulator,
    ...         adapter=adapter,
    ...         search_space=search_space,
    ...         validation_conditions=conditions,
    ...         inference_conditions=["N", "p_alloc"],
    ...         param_key="theta",
    ...         data_keys=["y", "x"],
    ...         context_keys={"N": int, "p_alloc": float},
    ...         true_param_key="theta_true",
    ...         simulate_fn_factory=my_simulate_fn_factory,
    ...         n_sims=500,
    ...         n_post_draws=500,
    ...     )
    """
    from rctbp_bf_training.core.infrastructure import (
        params_dict_to_workflow_config,
        build_summary_network,
        build_inference_network,
    )
    from rctbp_bf_training.core.validation import (
        run_validation_pipeline,
        make_bayesflow_infer_fn,
    )
    from rctbp_bf_training.core.utils import MovingAverageEarlyStopping
    import bayesflow as bf

    if rng is None:
        rng = np.random.default_rng()

    def objective(trial: "Trial") -> Tuple[float, float]:
        """Optuna objective: returns (calibration_error, param_count)."""
        import keras as _keras

        # Sample hyperparameters
        params = sample_hyperparameters(trial, search_space)

        # Convert to WorkflowConfig and build networks
        workflow_config = params_dict_to_workflow_config(params)
        summary_net = build_summary_network(
            workflow_config.summary_network
        )
        inference_net = build_inference_network(
            workflow_config.inference_network
        )

        # Setup learning rate schedule
        steps_per_epoch = params["batch_size"] * 100
        lr_schedule = _keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["initial_lr"],
            decay_steps=steps_per_epoch,
            decay_rate=params["decay_rate"],
            staircase=True,
        )
        opt = _keras.optimizers.Adam(learning_rate=lr_schedule)

        # Create workflow
        wf = bf.BasicWorkflow(
            simulator=simulator,
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net,
            optimizer=opt,
            inference_conditions=inference_conditions,
            checkpoint_name=f"optuna_trial_{trial.number}",
        )

        try:
            wf.approximator.compile(optimizer=opt)
        except Exception:
            pass

        early_stop = MovingAverageEarlyStopping(
            window=params["window"],
            patience=params["patience"],
            restore_best_weights=True,
        )

        # Train
        try:
            wf.fit_online(
                epochs=config.workflow.training.epochs,
                batch_size=params["batch_size"],
                num_batches_per_epoch=(
                    config.workflow.training.batches_per_epoch
                ),
                validation_data=(
                    config.workflow.training.validation_sims
                ),
                callbacks=[early_stop],
            )
        except Exception as e:
            print(f"Trial {trial.number} FAILED: {e}")
            cleanup_trial()
            return FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE

        param_count = get_param_count(wf.approximator)

        # Validate
        simulate_fn_opt = simulate_fn_factory(rng=rng)
        infer_fn_opt = make_bayesflow_infer_fn(
            wf.approximator,
            param_key=param_key,
            data_keys=data_keys,
            context_keys=context_keys,
        )

        try:
            results = run_validation_pipeline(
                conditions_list=validation_conditions,
                n_sims=n_sims,
                n_post_draws=n_post_draws,
                simulate_fn=simulate_fn_opt,
                infer_fn=infer_fn_opt,
                true_param_key=true_param_key,
                verbose=False,
            )
            cal_error, normalized_params = extract_objective_values(
                results["metrics"], param_count,
            )
        except Exception as e:
            print(f"Trial {trial.number} validation FAILED: {e}")
            cal_error = FAILED_TRIAL_CAL_ERROR
            normalized_params = FAILED_TRIAL_PARAM_SCORE

        print(
            f"Trial {trial.number}: "
            f"cal_error={cal_error:.4f}, "
            f"params={param_count:,} "
            f"(normalized={normalized_params:.3f})"
        )

        cleanup_trial()
        del wf, summary_net, inference_net
        gc.collect()

        return cal_error, normalized_params

    return objective


# =============================================================================
# BACKWARD COMPATIBILITY: Re-export from split modules
# =============================================================================
# These imports ensure that code using
#   from rctbp_bf_training.core.optimization import X
# continues to work after the module split.

from rctbp_bf_training.core.objectives import (  # noqa: F401, E402
    PARAM_COUNT_LOG_SCALE,
    compute_composite_objective,
    denormalize_param_count,
    estimate_param_count,
    normalize_param_count,
)
from rctbp_bf_training.core.results import (  # noqa: F401, E402
    get_pareto_trials,
    trials_to_dataframe,
    summarize_best_trials,
    plot_optimization_results,
    plot_pareto_front,
)
from rctbp_bf_training.core.threshold import (  # noqa: F401, E402
    QualityThresholds,
    check_thresholds,
    train_until_threshold,
    create_strict_validation_grid,
)
from rctbp_bf_training.core.dashboard import (  # noqa: F401, E402
    launch_dashboard,
)
