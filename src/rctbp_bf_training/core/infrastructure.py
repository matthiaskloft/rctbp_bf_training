"""
BayesFlow Infrastructure: Generic Components for Neural Posterior Estimation

This module provides reusable infrastructure for building BayesFlow workflows:
- Decoupled network configurations (summary and inference networks)
- Flexible adapter builder using declarative specifications
- Generic network builders
- Simulator factory
- Metadata utilities

This infrastructure can be reused across different models by providing
model-specific simulator functions and adapter specifications.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional
import json

import numpy as np
import bayesflow as bf


# =============================================================================
# Decoupled Network Configurations
# =============================================================================

@dataclass
class SummaryNetworkConfig:
    """
    Configuration for summary network (e.g., DeepSet).

    The summary network aggregates set-based data (variable-size observations)
    into a fixed-size summary representation.

    Attributes
    ----------
    summary_dim : int
        Dimensionality of the summary representation (default: 10).
    depth : int
        Number of layers in the network (default: 3).
    width : int
        Width of hidden layers (default: 64).
    dropout : float
        Dropout rate for regularization (default: 0.05).
    network_type : str
        Type of summary network architecture (default: "DeepSet").
        Future options: "Transformer", "GRU", etc.
    """
    summary_dim: int = 10
    depth: int = 3
    width: int = 64
    dropout: float = 0.05
    network_type: str = "DeepSet"


@dataclass
class InferenceNetworkConfig:
    """
    Configuration for inference network (e.g., Normalizing Flow).

    The inference network learns the posterior distribution conditioned on
    the summary representation and context variables.

    Attributes
    ----------
    depth : int
        Number of coupling layers in the flow (default: 7).
    hidden_sizes : tuple
        Hidden layer sizes for coupling subnets (default: (128, 128)).
    dropout : float
        Dropout rate for regularization (default: 0.20).
    network_type : str
        Type of inference network architecture (default: "CouplingFlow").
        Future options: "MAF", "NSF", etc.
    """
    depth: int = 7
    hidden_sizes: Tuple[int, ...] = (128, 128)
    dropout: float = 0.20
    network_type: str = "CouplingFlow"


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for workflow optimization.

    Attributes
    ----------
    initial_lr : float
        Initial learning rate (default: 7e-4).
    decay_rate : float
        Learning rate decay factor (default: 0.85).
    batch_size : int
        Number of simulations per batch (default: 320).
    epochs : int
        Maximum number of training epochs (default: 200).
    batches_per_epoch : int
        Number of batches per epoch (default: 50).
    validation_sims : int
        Number of simulations for validation (default: 1000).
    early_stopping_patience : int
        Patience for early stopping callback (default: 10).
    early_stopping_window : int
        Window size for moving average (default: 10).
    """
    initial_lr: float = 7e-4
    decay_rate: float = 0.85
    batch_size: int = 320
    epochs: int = 200
    batches_per_epoch: int = 50
    validation_sims: int = 1000
    early_stopping_patience: int = 10
    early_stopping_window: int = 10


@dataclass
class WorkflowConfig:
    """
    Complete workflow configuration bundling all network and training settings.

    This is the top-level generic configuration that can be reused across
    different models. Model-specific configurations should wrap this.

    Attributes
    ----------
    summary_network : SummaryNetworkConfig
        Configuration for summary network.
    inference_network : InferenceNetworkConfig
        Configuration for inference network.
    training : TrainingConfig
        Training hyperparameters.
    """
    summary_network: SummaryNetworkConfig = field(default_factory=SummaryNetworkConfig)
    inference_network: InferenceNetworkConfig = field(default_factory=InferenceNetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        """Serialize all configs to nested dict for JSON storage."""
        return {
            "summary_network": asdict(self.summary_network),
            "inference_network": asdict(self.inference_network),
            "training": asdict(self.training),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowConfig":
        """Reconstruct WorkflowConfig from dict."""
        return cls(
            summary_network=SummaryNetworkConfig(**d.get("summary_network", {})),
            inference_network=InferenceNetworkConfig(**d.get("inference_network", {})),
            training=TrainingConfig(**d.get("training", {})),
        )


# =============================================================================
# Flexible Adapter Builder
# =============================================================================

@dataclass
class AdapterSpec:
    """
    Declarative specification for building a BayesFlow adapter.

    This allows model-specific code to declaratively specify adapter
    transformations without hardcoding them in infrastructure code.
    Each model defines its own AdapterSpec based on its data structure.

    Attributes
    ----------
    set_keys : List[str]
        Data keys that form the set input (per-observation variables).
        Example: ["outcome", "covariate", "group"]
    param_keys : List[str]
        Parameter keys to infer (becomes "inference_variables").
        Example: ["b_group"]
    context_keys : List[str]
        Context variables (meta-parameters).
        Example: ["N", "p_alloc", "prior_df", "prior_scale"]
    standardize_keys : List[str]
        Keys to standardize with mean=0, std=1.
        Example: ["outcome", "covariate", "b_group"]
    broadcast_specs : Dict[str, str]
        Context variables to broadcast: {context_key: target_data_key}.
        Example: {"N": "outcome", "p_alloc": "outcome"}
    context_transforms : Dict[str, Tuple[Callable, Callable]]
        Transformations for context: {key: (forward_fn, inverse_fn)}.
        Example: {"N": (np.sqrt, np.square), "prior_df": (np.log1p, np.expm1)}
    output_dtype : str
        Output data type (default: "float32").
    """
    set_keys: List[str]
    param_keys: List[str]
    context_keys: List[str]
    standardize_keys: List[str] = field(default_factory=list)
    broadcast_specs: Dict[str, str] = field(default_factory=dict)
    context_transforms: Dict[str, Tuple[Callable, Callable]] = field(default_factory=dict)
    output_dtype: str = "float32"


def create_adapter(spec: AdapterSpec) -> bf.Adapter:
    """
    Build a BayesFlow Adapter from a declarative specification.

    This generic function can build adapters for any model by following
    the specification provided. The adapter chain performs:
    1. Broadcasting of context variables
    2. Standardization of specified keys
    3. Set formation from data keys
    4. Context transformations
    5. Type conversion
    6. Renaming and concatenation for BayesFlow interface

    Parameters
    ----------
    spec : AdapterSpec
        Declarative adapter specification.

    Returns
    -------
    bf.Adapter
        Configured BayesFlow adapter ready for use.

    Examples
    --------
    >>> spec = AdapterSpec(
    ...     set_keys=["outcome", "covariate", "group"],
    ...     param_keys=["b_group"],
    ...     context_keys=["N", "p_alloc"],
    ...     standardize_keys=["outcome", "covariate", "b_group"],
    ...     broadcast_specs={"N": "outcome"},
    ...     context_transforms={"N": (np.sqrt, np.square)},
    ... )
    >>> adapter = create_adapter(spec)
    """
    adapter = bf.Adapter()

    # Apply broadcasts
    for ctx_key, target_key in spec.broadcast_specs.items():
        adapter = adapter.broadcast(ctx_key, to=target_key)

    # Standardize
    if spec.standardize_keys:
        adapter = adapter.standardize(spec.standardize_keys, mean=0, std=1)

    # Form set
    adapter = adapter.as_set(spec.set_keys)

    # Apply context transforms (only forward, BayesFlow handles inverse internally)
    for key, (forward_fn, inverse_fn) in spec.context_transforms.items():
        adapter = adapter.apply(include=key, forward=forward_fn)

    # Convert dtype
    adapter = adapter.convert_dtype(from_dtype="float64", to_dtype=spec.output_dtype)

    # Rename parameters to inference_variables
    if len(spec.param_keys) == 1:
        adapter = adapter.rename(from_key=spec.param_keys[0], to_key="inference_variables")
    else:
        # Multi-parameter case: concatenate
        adapter = adapter.concatenate(spec.param_keys, into="inference_variables", axis=-1)

    # Concatenate data into summary_variables
    adapter = adapter.concatenate(spec.set_keys, into="summary_variables", axis=-1)

    # Concatenate context into inference_conditions
    adapter = adapter.concatenate(spec.context_keys, into="inference_conditions", axis=-1)

    return adapter


# =============================================================================
# Decoupled Network Builders
# =============================================================================

def build_summary_network(config: SummaryNetworkConfig):
    """
    Build summary network from configuration.

    The summary network is built independently from the inference network,
    enabling independent optimization and architecture swapping.

    Parameters
    ----------
    config : SummaryNetworkConfig
        Summary network configuration.

    Returns
    -------
    Summary network (e.g., bf.networks.DeepSet)

    Raises
    ------
    ValueError
        If network_type is not supported.

    Examples
    --------
    >>> config = SummaryNetworkConfig(summary_dim=10, depth=3, width=64)
    >>> summary_net = build_summary_network(config)
    """
    if config.network_type == "DeepSet":
        return bf.networks.DeepSet(
            summary_dim=config.summary_dim,
            depth=config.depth,
            mlp_widths_equivariant=(config.width,) * 2,
            mlp_widths_invariant_inner=(config.width,) * 2,
            mlp_widths_invariant_outer=(config.width,) * 2,
            mlp_widths_invariant_last=(config.width,) * 2,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown summary network type: {config.network_type}")


def build_inference_network(config: InferenceNetworkConfig):
    """
    Build inference network from configuration.

    The inference network is built independently from the summary network,
    enabling independent optimization and architecture swapping.

    Parameters
    ----------
    config : InferenceNetworkConfig
        Inference network configuration.

    Returns
    -------
    Inference network (e.g., bf.networks.CouplingFlow)

    Raises
    ------
    ValueError
        If network_type is not supported.

    Examples
    --------
    >>> config = InferenceNetworkConfig(depth=7, hidden_sizes=(128, 128))
    >>> inference_net = build_inference_network(config)
    """
    if config.network_type == "CouplingFlow":
        return bf.networks.CouplingFlow(
            depth=config.depth,
            subnet_kwargs=dict(
                hidden_sizes=list(config.hidden_sizes),
                dropout=config.dropout,
            ),
        )
    else:
        raise ValueError(f"Unknown inference network type: {config.network_type}")


def build_workflow(
    summary_network_config: SummaryNetworkConfig,
    inference_network_config: InferenceNetworkConfig,
    adapter_spec: AdapterSpec,
) -> Tuple:
    """
    Build complete workflow components: summary_net, inference_net, adapter.

    This is a convenience function for creating all BayesFlow components
    at once. Returns individual components for maximum flexibility.

    Parameters
    ----------
    summary_network_config : SummaryNetworkConfig
        Configuration for summary network.
    inference_network_config : InferenceNetworkConfig
        Configuration for inference network.
    adapter_spec : AdapterSpec
        Declarative adapter specification.

    Returns
    -------
    tuple
        (summary_net, inference_net, adapter)

    Examples
    --------
    >>> summary_config = SummaryNetworkConfig(summary_dim=10)
    >>> inference_config = InferenceNetworkConfig(depth=7)
    >>> adapter_spec = AdapterSpec(set_keys=[...], param_keys=[...], ...)
    >>> summary_net, inference_net, adapter = build_workflow(
    ...     summary_config, inference_config, adapter_spec
    ... )
    """
    summary_net = build_summary_network(summary_network_config)
    inference_net = build_inference_network(inference_network_config)
    adapter = create_adapter(adapter_spec)

    return summary_net, inference_net, adapter


# =============================================================================
# Generic Simulator Factory
# =============================================================================

def create_simulator(
    prior_fn: Callable,
    likelihood_fn: Callable,
    meta_fn: Optional[Callable] = None,
) -> bf.simulators.Simulator:
    """
    Generic simulator factory.

    Takes already-configured prior, likelihood, and meta functions
    and wraps them in a BayesFlow Simulator. Model-specific code
    should inject their configurations and RNG into these functions
    before passing them here.

    Parameters
    ----------
    prior_fn : Callable
        Function that samples from prior distribution.
        Should have signature matching BayesFlow requirements.
    likelihood_fn : Callable
        Function that samples from likelihood given parameters.
        Should have signature matching BayesFlow requirements.
    meta_fn : Callable, optional
        Function that samples meta-parameters (context).
        If None, no meta-parameters are used.

    Returns
    -------
    bf.simulators.Simulator
        Configured BayesFlow simulator.

    Examples
    --------
    >>> def my_prior(hyperparams):
    ...     return {"theta": np.random.normal(0, 1, size=1)}
    >>>
    >>> def my_likelihood(theta):
    ...     return {"data": np.random.normal(theta, 1, size=10)}
    >>>
    >>> simulator = create_simulator(my_prior, my_likelihood)
    """
    return bf.simulators.make_simulator(
        [prior_fn, likelihood_fn],
        meta_fn=meta_fn
    )


# =============================================================================
# Metadata Utilities
# =============================================================================

def get_workflow_metadata(
    config: WorkflowConfig,
    model_type: str,
    validation_results: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Collect reproducibility metadata for workflow.

    Generic version that works for any model. Model-specific code
    should call this and add their own metadata via the 'extra' parameter.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration used for training.
    model_type : str
        String identifier for model type (e.g., "ancova_cont_2arms").
    validation_results : dict, optional
        Validation metrics to include.
    extra : dict, optional
        Additional model-specific metadata to include.

    Returns
    -------
    dict
        Complete metadata dictionary.

    Examples
    --------
    >>> config = WorkflowConfig()
    >>> metadata = get_workflow_metadata(
    ...     config,
    ...     model_type="my_model",
    ...     extra={"prior_config": {...}}
    ... )
    """
    metadata = {
        "config": config.to_dict(),
        "versions": {
            "bayesflow": bf.__version__,
            "numpy": np.__version__,
        },
        "created_at": datetime.now().isoformat(),
        "model_type": model_type,
    }

    if validation_results is not None:
        metadata["validation"] = validation_results

    if extra is not None:
        metadata.update(extra)

    return metadata


def save_workflow_with_metadata(
    approximator,
    path: str | Path,
    metadata: dict,
) -> Path:
    """
    Save .keras model + .json sidecar with metadata.

    Generic utility for persisting trained models with full reproducibility
    information. Can be used for any BayesFlow approximator.

    Parameters
    ----------
    approximator : bf.approximators.Approximator
        Trained approximator to save.
    path : str or Path
        Base path (will create .keras and .json files).
    metadata : dict
        Metadata to save (from get_workflow_metadata or custom).

    Returns
    -------
    Path
        Path to saved .keras file.

    Examples
    --------
    >>> approximator = bf.approximators.ContinuousApproximator(...)
    >>> metadata = get_workflow_metadata(config, "my_model")
    >>> keras_path = save_workflow_with_metadata(
    ...     approximator, "models/my_model", metadata
    ... )
    """
    path = Path(path)
    keras_path = path.with_suffix(".keras")
    json_path = path.with_suffix(".json")

    # Ensure parent directory exists
    keras_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    approximator.save(keras_path)

    # Save metadata
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return keras_path


def load_workflow_with_metadata(path: str | Path) -> Tuple:
    """
    Load model and metadata from disk.

    Generic utility for loading persisted models. Loads both the .keras
    model and the .json metadata sidecar if present.

    Parameters
    ----------
    path : str or Path
        Base path (expects .keras and optionally .json files).

    Returns
    -------
    tuple
        (approximator, metadata)
        If no metadata file exists, returns empty dict for metadata.

    Examples
    --------
    >>> approximator, metadata = load_workflow_with_metadata("models/my_model")
    >>> print(metadata["model_type"])
    """
    path = Path(path)
    keras_path = path.with_suffix(".keras")
    json_path = path.with_suffix(".json")

    # Load model
    approximator = bf.BasicWorkflow.load(keras_path)

    # Load metadata if exists
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return approximator, metadata


# =============================================================================
# Hyperparameter Conversion for Bayesian Optimization
# =============================================================================

def params_dict_to_workflow_config(params: dict) -> WorkflowConfig:
    """
    Convert flat hyperparameter dict (from Optuna) to WorkflowConfig.

    This provides a standard interface between Bayesian optimization
    and workflow building. Extracts network and training parameters
    from a flat dictionary.

    Parameters
    ----------
    params : dict
        Flat dictionary of hyperparameters (typically from Optuna trial).
        Expected keys: summary_dim, deepset_depth, deepset_width, deepset_dropout,
                      flow_depth, flow_hidden, flow_dropout, initial_lr, batch_size, etc.

    Returns
    -------
    WorkflowConfig
        Structured configuration object.

    Examples
    --------
    >>> params = {
    ...     "summary_dim": 12,
    ...     "deepset_depth": 4,
    ...     "flow_depth": 8,
    ...     "flow_hidden": 256,
    ...     "initial_lr": 1e-3,
    ... }
    >>> config = params_dict_to_workflow_config(params)
    """
    return WorkflowConfig(
        summary_network=SummaryNetworkConfig(
            summary_dim=int(params.get("summary_dim", 10)),
            depth=int(params.get("deepset_depth", 3)),
            width=int(params.get("deepset_width", 64)),
            dropout=float(params.get("deepset_dropout", 0.05)),
        ),
        inference_network=InferenceNetworkConfig(
            depth=int(params.get("flow_depth", 7)),
            hidden_sizes=(int(params.get("flow_hidden", 128)),) * 2,
            dropout=float(params.get("flow_dropout", 0.20)),
        ),
        training=TrainingConfig(
            initial_lr=float(params.get("initial_lr", 7e-4)),
            batch_size=int(params.get("batch_size", 320)),
            decay_rate=float(params.get("decay_rate", 0.85)),
            epochs=int(params.get("epochs", 200)),
            batches_per_epoch=int(params.get("batches_per_epoch", 50)),
            validation_sims=int(params.get("validation_sims", 1000)),
            early_stopping_patience=int(params.get("early_stopping_patience", 10)),
            early_stopping_window=int(params.get("early_stopping_window", 10)),
        ),
    )
