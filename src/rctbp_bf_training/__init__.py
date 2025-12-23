"""RCT Bayesian Power Training using Neural Posterior Estimation."""

__version__ = "0.1.0"

# Core infrastructure (generic, reusable)
from rctbp_bf_training.core.infrastructure import (
    SummaryNetworkConfig,
    InferenceNetworkConfig,
    TrainingConfig,
    WorkflowConfig,
    AdapterSpec,
    build_summary_network,
    build_inference_network,
    build_workflow,
    save_workflow_with_metadata,
    load_workflow_with_metadata,
)

from rctbp_bf_training.core.optimization import (
    create_study,
    HyperparameterSpace,
    create_optimization_objective,
)

from rctbp_bf_training.core.validation import (
    run_validation_pipeline,
)

from rctbp_bf_training.core.utils import (
    loguniform_int,
    loguniform_float,
    sample_t_or_normal,
)

# ANCOVA model
from rctbp_bf_training.models.ancova.model import (
    ANCOVAConfig,
    PriorConfig,
    MetaConfig,
    create_ancova_workflow_components,
)

# Plotting
from rctbp_bf_training.plotting.diagnostics import (
    plot_coverage_diff,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "SummaryNetworkConfig",
    "InferenceNetworkConfig",
    "TrainingConfig",
    "WorkflowConfig",
    "AdapterSpec",
    "build_summary_network",
    "build_inference_network",
    "build_workflow",
    "save_workflow_with_metadata",
    "load_workflow_with_metadata",
    "create_study",
    "HyperparameterSpace",
    "create_optimization_objective",
    "run_validation_pipeline",
    "loguniform_int",
    "loguniform_float",
    "sample_t_or_normal",
    # ANCOVA
    "ANCOVAConfig",
    "PriorConfig",
    "MetaConfig",
    "create_ancova_workflow_components",
    # Plotting
    "plot_coverage_diff",
]
