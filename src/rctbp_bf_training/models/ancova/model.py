"""
ANCOVA 2-Arms Continuous Outcome: Model-Specific Implementation

This module provides ANCOVA-specific implementations:
- Simulator functions (prior, likelihood, meta) for ANCOVA model
- Adapter specification for ANCOVA data structure
- Factory functions for creating ANCOVA workflows
- Validation pipeline helpers
- Model metadata utilities

Generic infrastructure has been extracted to rctbp_bf_training.core.infrastructure
"""

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import bayesflow as bf

from rctbp_bf_training.core.utils import sample_t_or_normal, loguniform_int
from rctbp_bf_training.core.infrastructure import (
    SummaryNetworkConfig,
    InferenceNetworkConfig,
    WorkflowConfig,
    AdapterSpec,
    build_workflow,
    build_summary_network,
    build_inference_network,
    create_simulator as create_generic_simulator,
    get_workflow_metadata,
    save_workflow_with_metadata,
    load_workflow_with_metadata,
    params_dict_to_workflow_config,
)


# =============================================================================
# ANCOVA-Specific Configuration Dataclasses
# =============================================================================

@dataclass
class PriorConfig:
    """ANCOVA-specific prior distribution parameters."""
    b_covariate_scale: float = 2.0  # Scale for b_covariate Normal distribution
    # Note: sigma is fixed at 1.0 in this model (not estimated)


@dataclass
class MetaConfig:
    """ANCOVA-specific meta-parameter sampling ranges for training."""
    n_min: int = 20
    n_max: int = 1000
    p_alloc_min: float = 0.5
    p_alloc_max: float = 0.9
    prior_df_min: int = 0  # 0 means Normal (df > 100 treated as Normal)
    prior_df_max: int = 30
    prior_df_alpha: float = 0.7  # Alpha for log-uniform sampling
    prior_scale_gamma_shape: float = 2.0
    prior_scale_gamma_scale: float = 1.0


@dataclass
class ANCOVAConfig:
    """
    Complete configuration bundle for ANCOVA 2-arms model.

    This wraps the generic WorkflowConfig with ANCOVA-specific configurations.
    """
    prior: PriorConfig = field(default_factory=PriorConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def to_dict(self) -> dict:
        """Serialize all configs to nested dict for JSON storage."""
        return {
            "prior": asdict(self.prior),
            "meta": asdict(self.meta),
            "workflow": self.workflow.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ANCOVAConfig":
        """Reconstruct from dict."""
        return cls(
            prior=PriorConfig(**d.get("prior", {})),
            meta=MetaConfig(**d.get("meta", {})),
            workflow=WorkflowConfig.from_dict(d.get("workflow", {})),
        )


# =============================================================================
# ANCOVA-Specific Simulator Functions
# =============================================================================

def prior(prior_df: float, prior_scale: float, config: PriorConfig, rng: np.random.Generator) -> dict:
    """
    Sample parameters for model: outcome = b_covariate*x + b_group*group + noise.

    Parameters
    ----------
    prior_df : float
        Degrees of freedom for b_group prior. If df <= 0 or > 100, uses Normal.
    prior_scale : float
        Scale parameter for b_group prior distribution.
    config : PriorConfig
        Prior configuration with b_covariate_scale.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with b_covariate and b_group arrays (shape (1,))

    Notes
    -----
    - b_covariate ~ Normal(0, config.b_covariate_scale)
    - b_group ~ t(prior_df, 0, prior_scale) or Normal if df <= 0 or df > 100
    - sigma is fixed at 1.0 (not a parameter)
    """
    b_covariate = rng.normal(loc=0, scale=config.b_covariate_scale, size=1).astype(np.float64)

    b_group = np.array([sample_t_or_normal(
        df=float(np.asarray(prior_df).flat[0]),
        scale=float(np.asarray(prior_scale).flat[0]),
        rng=rng
    )], dtype=np.float64)

    return dict(b_covariate=b_covariate, b_group=b_group)


def likelihood(
    b_covariate: float,
    b_group: float,
    N: int,
    p_alloc: float,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate 2-arm ANCOVA data with fixed sigma = 1.

    Model: outcome = b_covariate * covariate + b_group * group + noise

    Parameters
    ----------
    b_covariate : float
        Coefficient for baseline covariate.
    b_group : float
        Treatment effect (group difference).
    N : int
        Total sample size.
    p_alloc : float
        Probability of treatment allocation (0 to 1).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with outcome, covariate, group arrays (each shape (N,))
    """
    b_cov = float(np.asarray(b_covariate).reshape(-1)[0])
    b_grp = float(np.asarray(b_group).reshape(-1)[0])
    sigma = 1.0  # Fixed
    n_total = int(np.asarray(N).reshape(-1)[0])
    p = float(np.clip(p_alloc, 0.01, 0.99))

    # Ensure both groups represented
    max_tries = 1000
    for _ in range(max_tries):
        group = rng.choice([0, 1], size=n_total, p=[1 - p, p])
        if np.sum(group == 0) > 0 and np.sum(group == 1) > 0:
            break
    else:
        # Fallback: force at least 1 in each group
        n_treat = max(1, int(n_total * p))
        n_ctrl = n_total - n_treat
        group = np.concatenate([np.zeros(n_ctrl), np.ones(n_treat)])
        rng.shuffle(group)

    covariate = rng.normal(0, 1, size=n_total)
    y_mean = b_cov * covariate + b_grp * group
    outcome = rng.normal(y_mean, sigma, size=n_total)

    return dict(outcome=outcome, covariate=covariate, group=group)


def meta(config: MetaConfig, rng: np.random.Generator) -> dict:
    """
    Sample meta parameters (context) including prior hyperparameters.

    Parameters
    ----------
    config : MetaConfig
        Configuration with sampling ranges.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict with N, p_alloc, prior_df, prior_scale
    """
    N = loguniform_int(config.n_min, config.n_max, rng=rng)
    p_alloc = rng.uniform(config.p_alloc_min, config.p_alloc_max)

    # prior_df: log-uniform shifted to allow 0 (Normal)
    prior_df = int(round(
        loguniform_int(1, config.prior_df_max + 1, alpha=config.prior_df_alpha, rng=rng) - 1
    ))

    prior_scale = rng.gamma(
        shape=config.prior_scale_gamma_shape,
        scale=config.prior_scale_gamma_scale,
    )

    return dict(N=N, p_alloc=p_alloc, prior_df=prior_df, prior_scale=prior_scale)


def simulate_cond_batch(
    n_sims: int,
    n_total: int,
    p_alloc: float,
    b_covariate: float,
    b_group: float,
    prior_df: float,
    prior_scale: float,
    rng: np.random.Generator = None,
) -> dict:
    """
    Vectorized batch simulation for a single condition.

    Parameters
    ----------
    n_sims : int
        Number of simulations to run.
    n_total : int
        Sample size per simulation.
    p_alloc : float
        Treatment allocation probability.
    b_covariate : float
        Coefficient for baseline covariate.
    b_group : float
        Treatment effect (true value).
    prior_df : float
        Degrees of freedom for prior (context for inference).
    prior_scale : float
        Scale for prior (context for inference).
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    dict with outcome, covariate, group matrices (n_sims x n_total) and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sims = int(n_sims)
    n_total = int(n_total)
    p = float(np.clip(p_alloc, 0.01, 0.99))
    b_cov = float(b_covariate)
    b_grp = float(b_group)

    group = (rng.random((n_sims, n_total)) < p).astype(np.float64)
    covariate = rng.standard_normal((n_sims, n_total))
    noise = rng.standard_normal((n_sims, n_total))
    outcome = b_cov * covariate + b_grp * group + noise

    return {
        "outcome": outcome,
        "covariate": covariate,
        "group": group,
        "N": n_total,
        "p_alloc": p_alloc,
        "prior_df": prior_df,
        "prior_scale": prior_scale,
    }


# =============================================================================
# ANCOVA-Specific Adapter Specification
# =============================================================================

def get_ancova_adapter_spec() -> AdapterSpec:
    """
    Return the adapter specification for ANCOVA 2-arms model.

    This declaratively defines how ANCOVA data should be processed
    by the BayesFlow adapter. The spec includes:
    - Set-based data: outcome, covariate, group (per-observation)
    - Parameters to infer: b_group (treatment effect)
    - Context variables: N, p_alloc, prior_df, prior_scale
    - Standardization of outcome, covariate, and b_group
    - Broadcasting and transformations for context variables

    Returns
    -------
    AdapterSpec
        Declarative specification for ANCOVA adapter.
    """
    return AdapterSpec(
        set_keys=["outcome", "covariate", "group"],
        param_keys=["b_group"],
        context_keys=["N", "p_alloc", "prior_df", "prior_scale"],
        standardize_keys=["outcome", "covariate", "b_group"],
        broadcast_specs={
            "N": "outcome",
            "p_alloc": "outcome",
            "prior_df": "outcome",
            "prior_scale": "outcome",
        },
        context_transforms={
            "N": (np.sqrt, np.square),
            "prior_df": (np.log1p, np.expm1),
        },
        output_dtype="float32",
    )


# =============================================================================
# ANCOVA-Specific Factory Functions
# =============================================================================

def create_ancova_workflow_components(config: ANCOVAConfig) -> tuple:
    """
    Create all ANCOVA workflow components using infrastructure.

    This is the main factory function for creating ANCOVA-specific
    summary network, inference network, and adapter.

    Parameters
    ----------
    config : ANCOVAConfig
        Complete ANCOVA configuration.

    Returns
    -------
    tuple
        (summary_net, inference_net, adapter)

    Examples
    --------
    >>> config = ANCOVAConfig()
    >>> summary_net, inference_net, adapter = create_ancova_workflow_components(config)
    """
    adapter_spec = get_ancova_adapter_spec()

    return build_workflow(
        summary_network_config=config.workflow.summary_network,
        inference_network_config=config.workflow.inference_network,
        adapter_spec=adapter_spec,
    )


def create_prior_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create prior function with injected config and rng."""
    def _prior(prior_df, prior_scale):
        return prior(prior_df, prior_scale, config.prior, rng)
    return _prior


def create_likelihood_fn(rng: np.random.Generator) -> Callable:
    """Create likelihood function with injected rng."""
    def _likelihood(b_covariate, b_group, N, p_alloc):
        return likelihood(b_covariate, b_group, N, p_alloc, rng)
    return _likelihood


def create_meta_fn(config: ANCOVAConfig, rng: np.random.Generator) -> Callable:
    """Create meta function with injected config and rng."""
    def _meta():
        return meta(config.meta, rng)
    return _meta


def create_simulator(config: ANCOVAConfig, rng: np.random.Generator = None) -> bf.simulators.Simulator:
    """
    Create BayesFlow simulator for ANCOVA model.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration bundle.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    bf.simulators.Simulator configured for ANCOVA 2-arms
    """
    if rng is None:
        rng = np.random.default_rng()

    prior_fn = create_prior_fn(config, rng)
    likelihood_fn = create_likelihood_fn(rng)
    meta_fn = create_meta_fn(config, rng)

    return create_generic_simulator(prior_fn, likelihood_fn, meta_fn)


def create_adapter() -> bf.Adapter:
    """
    Create adapter for ANCOVA 2-arms model.

    This is a convenience wrapper that creates an adapter using the
    ANCOVA adapter specification. It's equivalent to:

        from rctbp_bf_training.core.infrastructure import create_adapter
        adapter = create_adapter(get_ancova_adapter_spec())

    Returns
    -------
    bf.Adapter configured for ANCOVA 2-arms model

    Examples
    --------
    >>> adapter = create_adapter()
    >>> processed = adapter(simulator.sample(100))
    """
    from rctbp_bf_training.core.infrastructure import create_adapter as build_adapter
    return build_adapter(get_ancova_adapter_spec())


def create_ancova_objective(
    config: ANCOVAConfig,
    simulator: bf.Simulator,
    adapter: bf.Adapter,
    search_space: "HyperparameterSpace",
    validation_conditions: List[Dict],
    n_sims: int = 500,
    n_post_draws: int = 500,
    rng: np.random.Generator = None,
) -> Callable:
    """
    Create Optuna objective function for ANCOVA model optimization.

    This factory function returns an objective function closure that can be
    passed directly to `study.optimize()`. The objective function builds,
    trains, and validates models with different hyperparameters sampled by Optuna.

    Parameters
    ----------
    config : ANCOVAConfig
        ANCOVA configuration with training settings
    simulator : bf.Simulator
        BayesFlow simulator for the ANCOVA model
    adapter : bf.Adapter
        BayesFlow adapter for data transformation
    search_space : HyperparameterSpace
        Search space for hyperparameter sampling
    validation_conditions : list of dict
        Conditions grid for validation (from create_validation_grid)
    n_sims : int, default=500
        Number of simulations per condition for validation
    n_post_draws : int, default=500
        Number of posterior draws per simulation
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    objective : Callable[[Trial], Tuple[float, int]]
        Objective function that takes an Optuna trial and returns
        (calibration_error, parameter_count)

    Examples
    --------
    >>> config = ANCOVAConfig()
    >>> simulator = create_simulator(config, RNG)
    >>> adapter = create_adapter()
    >>> search_space = HyperparameterSpace(summary_dim=(4, 16), ...)
    >>> conditions = create_validation_grid(extended=False)
    >>>
    >>> objective = create_ancova_objective(
    ...     config, simulator, adapter, search_space, conditions
    ... )
    >>> study.optimize(objective, n_trials=30)
    """
    from rctbp_bf_training.core.infrastructure import (
        params_dict_to_workflow_config,
        build_summary_network,
        build_inference_network,
    )
    from rctbp_bf_training.core.optimization import (
        sample_hyperparameters,
        get_param_count,
        extract_objective_values,
        cleanup_trial,
    )
    from rctbp_bf_training.core.validation import (
        run_validation_pipeline,
        make_bayesflow_infer_fn,
    )
    from rctbp_bf_training.core.utils import MovingAverageEarlyStopping
    import gc

    if rng is None:
        rng = np.random.default_rng()

    def objective(trial):
        """Optuna objective: returns (calibration_error, param_count)."""
        import keras

        # Sample hyperparameters
        params = sample_hyperparameters(trial, search_space)

        # Convert to WorkflowConfig and build networks
        workflow_config = params_dict_to_workflow_config(params)
        summary_net = build_summary_network(workflow_config.summary_network)
        inference_net = build_inference_network(workflow_config.inference_network)

        # Setup learning rate schedule
        steps_per_epoch = params["batch_size"] * 100
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["initial_lr"],
            decay_steps=steps_per_epoch,
            decay_rate=params["decay_rate"],
            staircase=True,
        )
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)

        # Create workflow
        wf = bf.BasicWorkflow(
            simulator=simulator,
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net,
            optimizer=opt,
            inference_conditions=["N", "p_alloc", "prior_df", "prior_scale"],
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
            history = wf.fit_online(
                epochs=config.workflow.training.epochs,
                batch_size=params["batch_size"],
                num_batches_per_epoch=config.workflow.training.batches_per_epoch,
                validation_data=config.workflow.training.validation_sims,
                callbacks=[early_stop],
            )
        except Exception as e:
            print(f"Trial {trial.number} FAILED: {e}")
            cleanup_trial()
            return 1.0, 1e9

        param_count = get_param_count(wf.approximator)

        # Validate
        simulate_fn_opt = make_simulate_fn(rng=rng)
        infer_fn_opt = make_bayesflow_infer_fn(
            wf.approximator,
            param_key="b_group",
            data_keys=["outcome", "covariate", "group"],
            context_keys={"N": int, "p_alloc": float, "prior_df": float, "prior_scale": float},
        )

        try:
            results = run_validation_pipeline(
                conditions_list=validation_conditions,
                n_sims=n_sims,
                n_post_draws=n_post_draws,
                simulate_fn=simulate_fn_opt,
                infer_fn=infer_fn_opt,
                true_param_key="b_arm_treat",
                verbose=False,
            )
            cal_error, _ = extract_objective_values(results["metrics"], param_count)
        except Exception as e:
            print(f"Trial {trial.number} validation FAILED: {e}")
            cal_error = 1.0

        print(f"Trial {trial.number}: cal_error={cal_error:.4f}, params={param_count:,}")

        cleanup_trial()
        del wf, summary_net, inference_net
        gc.collect()

        return cal_error, param_count

    return objective


# =============================================================================
# Validation Helpers (ANCOVA-Specific)
# =============================================================================

def create_validation_grid(extended: bool = False) -> list[dict]:
    """
    Generate conditions for systematic validation.

    Parameters
    ----------
    extended : bool
        If True, include more conditions for comprehensive validation.

    Returns
    -------
    list of condition dicts
    """
    from itertools import product

    if extended:
        # Extended grid for final validation
        conditions = []
        for idx, (n, pdf, psc, b_cov, b_grp, p_alloc) in enumerate(product(
            [20, 1000],          # N extremes
            [0, 2],              # prior_df: Normal vs low-df t
            [0.1, 5.0],          # prior_scale extremes
            [-0.5, 0.5],         # b_covariate
            [0.0, 0.3, 1.0],     # b_group: null, small, large
            [0.5, 0.9],          # p_alloc
        )):
            conditions.append({
                "id_cond": idx,
                "n_total": n,
                "p_alloc": p_alloc,
                "b_covariate": b_cov,
                "b_arm_treat": b_grp,
                "prior_df": pdf,
                "prior_scale": psc,
            })
    else:
        # Reduced grid for optimization (faster)
        conditions = []
        for idx, (n, pdf, psc, b_grp) in enumerate(product(
            [20, 500],           # N extremes
            [0, 10],             # prior_df: Normal vs moderate t
            [0.5, 5.0],          # prior_scale extremes
            [0.0, 0.5],          # b_group: null vs moderate
        )):
            conditions.append({
                "id_cond": idx,
                "n_total": n,
                "p_alloc": 0.5,
                "b_covariate": 0.0,
                "b_arm_treat": b_grp,
                "prior_df": pdf,
                "prior_scale": psc,
            })

    return conditions


def make_simulate_fn(
    rng: np.random.Generator = None,
    param_mapping: dict = None,
) -> Callable:
    """
    Create simulation function for validation pipeline.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator.
    param_mapping : dict, optional
        Mapping from condition keys to simulate_cond_batch params.

    Returns
    -------
    callable: simulate_fn(condition, n_sims) -> dict
    """
    if param_mapping is None:
        param_mapping = {
            "n_total": "n_total",
            "p_alloc": "p_alloc",
            "b_covariate": "b_covariate",
            "b_arm_treat": "b_group",
            "prior_df": "prior_df",
            "prior_scale": "prior_scale",
        }

    def simulate_fn(condition: dict, n_sims: int) -> dict:
        kwargs = {"n_sims": n_sims, "rng": rng}
        for cond_key, fn_param in param_mapping.items():
            val = condition.get(cond_key, condition.get(fn_param, 0.0))
            if isinstance(val, np.ndarray):
                val = float(val.flat[0]) if val.size == 1 else val
            kwargs[fn_param] = val
        return simulate_cond_batch(**kwargs)

    return simulate_fn


def make_infer_fn(approximator) -> Callable:
    """
    Create inference function for validation pipeline.

    Parameters
    ----------
    approximator : bf.approximators.Approximator
        Trained BayesFlow approximator.

    Returns
    -------
    callable: infer_fn(data, n_samples) -> np.ndarray
    """
    from rctbp_bf_training.core.validation import make_bayesflow_infer_fn

    return make_bayesflow_infer_fn(
        approximator,
        param_key="b_group",
        data_keys=["outcome", "covariate", "group"],
        context_keys={"N": int, "p_alloc": float, "prior_df": float, "prior_scale": float},
    )


# =============================================================================
# ANCOVA-Specific Metadata Utilities
# =============================================================================

def get_model_metadata(
    config: ANCOVAConfig,
    validation_results: dict = None,
    extra: dict = None,
) -> dict:
    """
    Collect all ANCOVA-specific reproducibility metadata.

    Parameters
    ----------
    config : ANCOVAConfig
        Configuration used for training.
    validation_results : dict, optional
        Validation metrics to include.
    extra : dict, optional
        Additional metadata to include.

    Returns
    -------
    dict with complete metadata
    """
    return get_workflow_metadata(
        config=config.workflow,
        model_type="ancova_cont_2arms",
        validation_results=validation_results,
        extra={
            "prior_config": asdict(config.prior),
            "meta_config": asdict(config.meta),
            **(extra or {})
        }
    )


# Alias for backwards compatibility
save_model_with_metadata = save_workflow_with_metadata
load_model_with_metadata = load_workflow_with_metadata
