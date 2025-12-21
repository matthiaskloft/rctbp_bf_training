"""
ANCOVA 2-Arms Continuous Outcome: Model-Specific Implementation

This module provides ANCOVA-specific implementations:
- Simulator functions (prior, likelihood, meta) for ANCOVA model
- Adapter specification for ANCOVA data structure
- Factory functions for creating ANCOVA workflows
- Validation pipeline helpers
- Model metadata utilities

Generic infrastructure has been extracted to bayesflow_infrastructure.py
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


# =============================================================================
# Backwards Compatibility Wrappers
# =============================================================================

# Legacy NetworkConfig for backwards compatibility
@dataclass
class NetworkConfig:
    """
    Legacy network configuration (backwards compatibility).

    New code should use WorkflowConfig from bayesflow_infrastructure.
    This class is maintained for compatibility with existing code.
    """
    summary_dim: int = 10
    deepset_depth: int = 3
    deepset_width: int = 64
    deepset_dropout: float = 0.05
    flow_depth: int = 7
    flow_hidden: int = 128
    flow_dropout: float = 0.20

    def to_workflow_config(self) -> WorkflowConfig:
        """Convert legacy NetworkConfig to new WorkflowConfig."""
        return WorkflowConfig(
            summary_network=SummaryNetworkConfig(
                summary_dim=self.summary_dim,
                depth=self.deepset_depth,
                width=self.deepset_width,
                dropout=self.deepset_dropout,
            ),
            inference_network=InferenceNetworkConfig(
                depth=self.flow_depth,
                hidden_sizes=(self.flow_hidden, self.flow_hidden),
                dropout=self.flow_dropout,
            ),
        )


def build_networks(config: NetworkConfig | WorkflowConfig) -> tuple:
    """
    Legacy function for building networks (backwards compatibility).

    New code should use create_ancova_workflow_components() instead.
    This function is maintained for compatibility with existing code.

    Parameters
    ----------
    config : NetworkConfig or WorkflowConfig
        Network architecture configuration.

    Returns
    -------
    tuple: (summary_net, inference_net)
    """
    if isinstance(config, NetworkConfig):
        # Legacy path: convert to WorkflowConfig
        workflow_config = config.to_workflow_config()
    else:
        workflow_config = config

    summary_net = build_summary_network(workflow_config.summary_network)
    inference_net = build_inference_network(workflow_config.inference_network)

    return summary_net, inference_net


def build_networks_from_params(params: dict) -> tuple:
    """
    BO-compatible wrapper: flat dict -> networks (backwards compatibility).

    New code should use params_dict_to_workflow_config() instead.
    This function is maintained for compatibility with existing code.

    Parameters
    ----------
    params : dict
        Flat dictionary of hyperparameters (from Optuna trial).

    Returns
    -------
    tuple: (summary_net, inference_net)
    """
    workflow_config = params_dict_to_workflow_config(params)
    return build_networks(workflow_config)


def network_config_from_params(params: dict) -> NetworkConfig:
    """
    Convert flat params dict to legacy NetworkConfig (backwards compatibility).

    New code should use params_dict_to_workflow_config() instead.

    Parameters
    ----------
    params : dict
        Flat dictionary of hyperparameters.

    Returns
    -------
    NetworkConfig (legacy)
    """
    return NetworkConfig(
        summary_dim=int(params.get("summary_dim", 10)),
        deepset_depth=int(params.get("deepset_depth", 3)),
        deepset_width=int(params.get("deepset_width", 64)),
        deepset_dropout=float(params.get("deepset_dropout", 0.05)),
        flow_depth=int(params.get("flow_depth", 7)),
        flow_hidden=int(params.get("flow_hidden", 128)),
        flow_dropout=float(params.get("flow_dropout", 0.20)),
    )


def create_adapter() -> bf.Adapter:
    """
    Legacy function for creating adapter (backwards compatibility).

    New code should use get_ancova_adapter_spec() and the infrastructure's
    create_adapter() instead. This function is maintained for compatibility
    with existing code.

    Returns
    -------
    bf.Adapter configured for ANCOVA 2-arms model
    """
    from bayesflow_infrastructure import create_adapter as build_adapter
    return build_adapter(get_ancova_adapter_spec())


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
    from functions_validation import make_bayesflow_infer_fn

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
