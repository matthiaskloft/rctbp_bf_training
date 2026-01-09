"""
FastAPI application for RCTBP BayesFlow Training service.

This module provides a REST API for:
- Loading trained BayesFlow models
- Running posterior inference on ANCOVA data
- Simulating RCT data
- Running Bayesian power analysis
"""

import os
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Set Keras backend before importing any Keras/BayesFlow code
os.environ.setdefault("KERAS_BACKEND", "torch")

from api.schemas import (
    HealthResponse,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
    SimulationRequest,
    SimulationResponse,
    PowerAnalysisRequest,
    PowerAnalysisResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for loaded model
_model_state = {
    "workflow": None,
    "approximator": None,
    "metadata": None,
    "loaded": False,
}


def load_model(model_path: Optional[str] = None) -> bool:
    """
    Load a trained BayesFlow model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the model directory. If None, uses MODEL_PATH env var.

    Returns
    -------
    bool : True if model loaded successfully
    """
    from rctbp_bf_training import load_workflow_with_metadata

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "/models/default")

    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning(f"Model path does not exist: {model_path}")
        return False

    try:
        workflow, metadata = load_workflow_with_metadata(model_path)
        _model_state["workflow"] = workflow
        _model_state["approximator"] = workflow.approximator
        _model_state["metadata"] = metadata
        _model_state["loaded"] = True
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def ensure_model_loaded():
    """Raise HTTPException if model is not loaded."""
    if not _model_state["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Please load a model first or check MODEL_PATH."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting RCTBP BayesFlow API service...")

    # Try to load model from environment variable
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        logger.info(f"Attempting to load model from: {model_path}")
        load_model(model_path)
    else:
        logger.info("No MODEL_PATH set, starting without pre-loaded model")

    yield

    # Shutdown
    logger.info("Shutting down RCTBP BayesFlow API service...")


# Create FastAPI application
app = FastAPI(
    title="RCTBP BayesFlow API",
    description="REST API for RCT Bayesian Power analysis using Neural Posterior Estimation",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and model status."""
    from rctbp_bf_training import __version__

    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_model_state["loaded"],
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the currently loaded model."""
    ensure_model_loaded()

    metadata = _model_state["metadata"] or {}

    return ModelInfo(
        model_type=metadata.get("model_type", "ancova_cont_2arms"),
        version=metadata.get("version", "unknown"),
        created_at=metadata.get("created_at"),
        metadata=metadata,
    )


@app.post("/model/load", tags=["Model"])
async def load_model_endpoint(model_path: str):
    """
    Load a model from the specified path.

    Note: In production, models should be pre-loaded via MODEL_PATH env var.
    """
    success = load_model(model_path)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to load model from: {model_path}"
        )
    return {"status": "loaded", "path": model_path}


@app.post(
    "/infer",
    response_model=InferenceResponse,
    tags=["Inference"],
    responses={400: {"model": ErrorResponse}},
)
async def run_inference(request: InferenceRequest):
    """
    Run posterior inference on ANCOVA data.

    Given observed outcome, covariate, and group assignment data,
    returns posterior samples for the treatment effect (b_group).
    """
    ensure_model_loaded()

    # Validate input lengths
    n_obs = len(request.outcome)
    if len(request.covariate) != n_obs or len(request.group) != n_obs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="outcome, covariate, and group must have the same length"
        )

    try:
        # Prepare data for inference
        data = {
            "outcome": np.array(request.outcome, dtype=np.float32),
            "covariate": np.array(request.covariate, dtype=np.float32),
            "group": np.array(request.group, dtype=np.float32),
            "N": n_obs,
            "p_alloc": float(np.mean(request.group)),  # Estimate from data
            "prior_df": request.prior_df,
            "prior_scale": request.prior_scale,
        }

        # Run inference using the approximator
        approximator = _model_state["approximator"]

        # Sample from posterior
        posterior_samples = approximator.sample(
            conditions=data,
            num_samples=request.n_samples,
        )

        # Extract b_group samples (flatten if needed)
        if isinstance(posterior_samples, dict):
            samples = posterior_samples.get("b_group", posterior_samples.get("parameters"))
        else:
            samples = posterior_samples

        samples = np.array(samples).flatten()

        # Compute summary statistics
        posterior_mean = float(np.mean(samples))
        posterior_std = float(np.std(samples))
        ci_lower = float(np.percentile(samples, 2.5))
        ci_upper = float(np.percentile(samples, 97.5))

        return InferenceResponse(
            posterior_samples=samples.tolist(),
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval_95=[ci_lower, ci_upper],
            n_observations=n_obs,
        )

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["Simulation"],
)
async def simulate_data(request: SimulationRequest):
    """
    Simulate ANCOVA RCT data.

    Generates synthetic data from the ANCOVA 2-arms model with
    the specified parameters.
    """
    from rctbp_bf_training.models.ancova.model import simulate_cond_batch

    try:
        rng = np.random.default_rng(request.seed)

        result = simulate_cond_batch(
            n_sims=request.n_sims,
            n_total=request.n_total,
            p_alloc=request.p_alloc,
            b_covariate=request.b_covariate,
            b_group=request.b_group,
            prior_df=request.prior_df,
            prior_scale=request.prior_scale,
            rng=rng,
        )

        return SimulationResponse(
            n_sims=request.n_sims,
            n_total=request.n_total,
            outcome_shape=list(result["outcome"].shape),
            true_b_group=request.b_group,
        )

    except Exception as e:
        logger.exception("Simulation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@app.post(
    "/power",
    response_model=PowerAnalysisResponse,
    tags=["Power Analysis"],
)
async def power_analysis(request: PowerAnalysisRequest):
    """
    Run Bayesian power analysis.

    Estimates the probability that the posterior mass above the threshold
    exceeds 95%, across multiple simulated datasets.
    """
    ensure_model_loaded()

    from rctbp_bf_training.models.ancova.model import simulate_cond_batch

    try:
        rng = np.random.default_rng(request.seed)
        approximator = _model_state["approximator"]

        # Simulate multiple datasets
        sim_data = simulate_cond_batch(
            n_sims=request.n_sims,
            n_total=request.n_total,
            p_alloc=request.p_alloc,
            b_covariate=0.0,  # Standard assumption
            b_group=request.effect_size,
            prior_df=request.prior_df,
            prior_scale=request.prior_scale,
            rng=rng,
        )

        # Run inference on each simulated dataset
        above_threshold_count = 0
        posterior_means = []
        posterior_stds = []

        for i in range(request.n_sims):
            # Prepare data for this simulation
            data = {
                "outcome": sim_data["outcome"][i].astype(np.float32),
                "covariate": sim_data["covariate"][i].astype(np.float32),
                "group": sim_data["group"][i].astype(np.float32),
                "N": request.n_total,
                "p_alloc": request.p_alloc,
                "prior_df": request.prior_df,
                "prior_scale": request.prior_scale,
            }

            # Sample from posterior
            posterior_samples = approximator.sample(
                conditions=data,
                num_samples=request.n_posterior_samples,
            )

            # Extract samples
            if isinstance(posterior_samples, dict):
                samples = posterior_samples.get("b_group", posterior_samples.get("parameters"))
            else:
                samples = posterior_samples

            samples = np.array(samples).flatten()

            # Track statistics
            posterior_means.append(float(np.mean(samples)))
            posterior_stds.append(float(np.std(samples)))

            # Check if posterior mass above threshold exceeds 95%
            prop_above = np.mean(samples > request.threshold)
            if prop_above >= 0.95:
                above_threshold_count += 1

        power = above_threshold_count / request.n_sims

        return PowerAnalysisResponse(
            power=power,
            n_total=request.n_total,
            effect_size=request.effect_size,
            threshold=request.threshold,
            n_sims=request.n_sims,
            mean_posterior_mean=float(np.mean(posterior_means)),
            mean_posterior_std=float(np.mean(posterior_stds)),
        )

    except Exception as e:
        logger.exception("Power analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
