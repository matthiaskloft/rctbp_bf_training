"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Package version")
    model_loaded: bool = Field(..., description="Whether a model is loaded")


class ModelInfo(BaseModel):
    """Model metadata information."""
    model_type: str = Field(..., description="Type of model (e.g., ancova_cont_2arms)")
    version: str = Field(..., description="Model version")
    created_at: Optional[str] = Field(None, description="Model creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class InferenceRequest(BaseModel):
    """Request for posterior inference on ANCOVA data."""
    outcome: List[float] = Field(..., description="Outcome variable values (N observations)")
    covariate: List[float] = Field(..., description="Covariate values (N observations)")
    group: List[float] = Field(..., description="Group assignment (0=control, 1=treatment)")
    n_samples: int = Field(default=1000, ge=100, le=10000, description="Number of posterior samples")
    prior_df: float = Field(default=0.0, ge=0, description="Prior degrees of freedom (0 = Normal)")
    prior_scale: float = Field(default=1.0, gt=0, description="Prior scale parameter")

    class Config:
        json_schema_extra = {
            "example": {
                "outcome": [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4],
                "covariate": [0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.5, 0.4],
                "group": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_samples": 1000,
                "prior_df": 3.0,
                "prior_scale": 1.0,
            }
        }


class InferenceResponse(BaseModel):
    """Response with posterior inference results."""
    posterior_samples: List[float] = Field(..., description="Posterior samples for b_group")
    posterior_mean: float = Field(..., description="Posterior mean of b_group")
    posterior_std: float = Field(..., description="Posterior standard deviation")
    credible_interval_95: List[float] = Field(..., description="95% credible interval [lower, upper]")
    n_observations: int = Field(..., description="Number of observations in input data")


class SimulationRequest(BaseModel):
    """Request for simulating ANCOVA data."""
    n_sims: int = Field(default=100, ge=1, le=10000, description="Number of simulations")
    n_total: int = Field(default=100, ge=10, le=10000, description="Sample size per simulation")
    p_alloc: float = Field(default=0.5, gt=0, lt=1, description="Treatment allocation probability")
    b_covariate: float = Field(default=0.0, description="Covariate coefficient")
    b_group: float = Field(default=0.5, description="True treatment effect")
    prior_df: float = Field(default=0.0, ge=0, description="Prior degrees of freedom")
    prior_scale: float = Field(default=1.0, gt=0, description="Prior scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "n_sims": 100,
                "n_total": 200,
                "p_alloc": 0.5,
                "b_covariate": 0.3,
                "b_group": 0.5,
                "prior_df": 3.0,
                "prior_scale": 1.0,
                "seed": 42,
            }
        }


class SimulationResponse(BaseModel):
    """Response with simulated data."""
    n_sims: int = Field(..., description="Number of simulations generated")
    n_total: int = Field(..., description="Sample size per simulation")
    outcome_shape: List[int] = Field(..., description="Shape of outcome array")
    true_b_group: float = Field(..., description="True treatment effect used")


class PowerAnalysisRequest(BaseModel):
    """Request for Bayesian power analysis."""
    n_total: int = Field(..., ge=10, le=10000, description="Total sample size")
    p_alloc: float = Field(default=0.5, gt=0, lt=1, description="Treatment allocation probability")
    effect_size: float = Field(..., description="Expected treatment effect (b_group)")
    prior_df: float = Field(default=0.0, ge=0, description="Prior degrees of freedom")
    prior_scale: float = Field(default=1.0, gt=0, description="Prior scale parameter")
    threshold: float = Field(default=0.0, description="Decision threshold for effect")
    n_sims: int = Field(default=500, ge=100, le=5000, description="Number of simulations")
    n_posterior_samples: int = Field(default=1000, ge=100, le=5000, description="Posterior samples per sim")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "n_total": 200,
                "p_alloc": 0.5,
                "effect_size": 0.3,
                "prior_df": 3.0,
                "prior_scale": 1.0,
                "threshold": 0.0,
                "n_sims": 500,
                "n_posterior_samples": 1000,
                "seed": 42,
            }
        }


class PowerAnalysisResponse(BaseModel):
    """Response with Bayesian power analysis results."""
    power: float = Field(..., ge=0, le=1, description="Estimated power (proportion above threshold)")
    n_total: int = Field(..., description="Sample size used")
    effect_size: float = Field(..., description="Effect size used")
    threshold: float = Field(..., description="Decision threshold used")
    n_sims: int = Field(..., description="Number of simulations run")
    mean_posterior_mean: float = Field(..., description="Average posterior mean across simulations")
    mean_posterior_std: float = Field(..., description="Average posterior std across simulations")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
