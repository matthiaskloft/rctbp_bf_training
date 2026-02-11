"""
Generic Validation Pipeline for Neural Posterior Estimation.

Model-agnostic functions for validating BayesFlow models
across a grid of conditions.

Usage:
------
from rctbp_bf_training.core.validation import (
    run_validation_pipeline,
    extract_calibration_metrics,
    make_bayesflow_infer_fn,
)

# Define your own simulation function
def my_simulate_fn(condition, n_sims):
    ...
    return {"data_key1": array, "data_key2": array, "N": n, ...}

# Create adapter functions
infer_fn = make_bayesflow_infer_fn(
    model, 
    param_key="my_param",
    data_keys=["data_key1", "data_key2"],
    context_keys={"N": int, "p_alloc": float}
)

# Run the pipeline
results = run_validation_pipeline(
    conditions_list=conditions,
    n_sims=1000,
    n_post_draws=1000,
    simulate_fn=my_simulate_fn,
    infer_fn=infer_fn,
    true_param_key="my_param"
)
"""

import gc
import time
import warnings

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Any, Optional, Union


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _cleanup_gpu_memory():
    """Clean up GPU memory after inference."""
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


def _compute_sbc_uniformity_tests(
    ranks: np.ndarray,
    n_post_draws: int,
    n_bins: int = 20
) -> Dict[str, float]:
    """
    Compute uniformity tests for SBC ranks.

    SBC (Simulation-Based Calibration) checks if posterior inference is correct
    by testing whether rank statistics are uniformly distributed.
    Reference: Talts et al. (2018) "Validating Bayesian Inference Algorithms
    with Simulation-Based Calibration"

    Parameters:
    -----------
    ranks : array of shape (n_sims,)
        SBC ranks in {0, 1, ..., n_post_draws}
    n_post_draws : int
        Number of posterior draws (determines rank range)
    n_bins : int
        Number of bins for chi-squared test (default: 20)

    Returns:
    --------
    dict with:
        - sbc_ks_stat: Kolmogorov-Smirnov statistic
        - sbc_ks_pvalue: KS test p-value (p < 0.05 suggests miscalibration)
        - sbc_chi2_stat: Chi-squared statistic
        - sbc_chi2_pvalue: Chi-squared p-value
    """
    from scipy.stats import kstest, chisquare

    n_sims = len(ranks)

    if n_sims == 0:
        return {
            "sbc_ks_stat": np.nan,
            "sbc_ks_pvalue": np.nan,
            "sbc_chi2_stat": np.nan,
            "sbc_chi2_pvalue": np.nan,
        }

    # 1. Kolmogorov-Smirnov test
    # Normalize ranks to [0, 1] for KS test against Uniform(0, 1)
    # Use (rank + 0.5) / (n_post_draws + 1) for continuity correction
    # This maps discrete ranks {0, ..., M} to continuous (0, 1)
    normalized_ranks = (ranks + 0.5) / (n_post_draws + 1)
    ks_stat, ks_pvalue = kstest(normalized_ranks, 'uniform')

    # 2. Chi-squared test on binned ranks
    n_bins_actual = min(n_bins, n_post_draws + 1)
    hist, _ = np.histogram(ranks, bins=n_bins_actual, range=(-0.5, n_post_draws + 0.5))
    expected_per_bin = n_sims / n_bins_actual

    # Only run chi2 if expected counts >= 5 (assumption for chi-squared validity)
    if expected_per_bin >= 5:
        chi2_stat, chi2_pvalue = chisquare(hist, f_exp=[expected_per_bin] * n_bins_actual)
    else:
        chi2_stat, chi2_pvalue = np.nan, np.nan

    return {
        "sbc_ks_stat": float(ks_stat),
        "sbc_ks_pvalue": float(ks_pvalue),
        "sbc_chi2_stat": float(chi2_stat) if not np.isnan(chi2_stat) else np.nan,
        "sbc_chi2_pvalue": float(chi2_pvalue) if not np.isnan(chi2_pvalue) else np.nan,
    }


def _compute_sbc_c2st(
    ranks: np.ndarray,
    n_post_draws: int,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Classifier Two-Sample Test (C2ST) for SBC ranks.

    Trains a classifier to distinguish observed ranks from uniform samples.
    If the classifier cannot distinguish them (accuracy ~0.5), the posterior
    is well-calibrated. High accuracy (>0.6) indicates miscalibration.

    Reference: Lopez-Paz & Oquab (2016) "Revisiting Classifier Two-Sample Tests"

    Parameters:
    -----------
    ranks : array of shape (n_sims,)
        SBC ranks in {0, 1, ..., n_post_draws}
    n_post_draws : int
        Number of posterior draws (determines rank range)
    n_folds : int
        Number of cross-validation folds (default: 5)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict with:
        - sbc_c2st_accuracy: Mean cross-validation accuracy (~0.5 = good)
        - sbc_c2st_sd: Standard deviation of CV accuracies
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return {"sbc_c2st_accuracy": np.nan, "sbc_c2st_sd": np.nan}

    n_sims = len(ranks)

    if n_sims < 2 * n_folds:
        return {"sbc_c2st_accuracy": np.nan, "sbc_c2st_sd": np.nan}

    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    # Generate uniform reference samples (same size as observed)
    uniform_ranks = rng.randint(0, n_post_draws + 1, size=n_sims)

    # Create dataset: observed (label=1) vs uniform (label=0)
    X = np.concatenate([ranks, uniform_ranks]).reshape(-1, 1)
    y = np.concatenate([np.ones(n_sims), np.zeros(n_sims)])

    # Train classifier with cross-validation
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=random_state
    )
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')

    return {
        "sbc_c2st_accuracy": float(np.mean(scores)),
        "sbc_c2st_sd": float(np.std(scores)),
    }


# =============================================================================
# BAYESFLOW INFERENCE ADAPTER (Generic)
# =============================================================================

def make_bayesflow_infer_fn(
    bf_model,
    param_key: str = "b_group",
    data_keys: Optional[List[str]] = None,
    context_keys: Optional[Dict[str, type]] = None
) -> Callable:
    """
    Create a generic inference function for a BayesFlow model.
    
    Parameters:
    -----------
    bf_model : BayesFlow ContinuousApproximator
        Trained BayesFlow model
    param_key : str
        Key for the parameter in the posterior samples dict
    data_keys : list of str, optional
        Keys for data arrays to pass to the model.
        Default: ["outcome", "covariate", "group"]
    context_keys : dict of {str: type}, optional
        Keys for context variables with their type converters.
        Default: {"N": int, "p_alloc": float}
        
    Returns:
    --------
    callable: infer_fn(sim_data, n_post_draws) -> np.ndarray of shape (n_sims, n_post_draws)
    
    Example:
    --------
    # For ANCOVA model
    infer_fn = make_bayesflow_infer_fn(
        model, 
        param_key="b_group",
        data_keys=["outcome", "covariate", "group"],
        context_keys={"N": int, "p_alloc": float}
    )
    
    # For a different model with different data structure
    infer_fn = make_bayesflow_infer_fn(
        model,
        param_key="theta",
        data_keys=["response", "stimulus"],
        context_keys={"n_trials": int, "condition": float}
    )
    """
    # Set defaults
    if data_keys is None:
        data_keys = ["outcome", "covariate", "group"]
    if context_keys is None:
        context_keys = {"N": int, "p_alloc": float}
    
    def infer_fn(sim_data: Dict[str, Any], n_post_draws: int) -> np.ndarray:
        # Build conditions dict for BayesFlow
        data_cond = {}
        
        # Add data arrays
        for key in data_keys:
            if key in sim_data:
                data_cond[key] = sim_data[key]
        
        # Add context variables with type conversion
        for key, type_fn in context_keys.items():
            if key in sim_data:
                data_cond[key] = type_fn(sim_data[key])
        
        # Run inference
        post_draws = bf_model.sample(conditions=data_cond, num_samples=int(n_post_draws))
        draws = np.squeeze(post_draws[param_key], axis=-1)
        
        # Clean up
        del post_draws, data_cond
        
        return draws
    
    return infer_fn


# =============================================================================
# PER-BATCH METRICS COMPUTATION
# =============================================================================

def compute_batch_metrics(
    draws: np.ndarray,
    true_value: float,
    cond_id: int,
    sim_id_start: int,
    coverage_levels: List[float],
    full_coverage_levels: List[float]
) -> pd.DataFrame:
    """
    Compute per-simulation metrics for a single batch of posterior draws.
    
    This function is called after each condition's inference to immediately
    compute metrics without storing raw draws, enabling memory-efficient
    incremental processing.
    
    Parameters:
    -----------
    draws : np.ndarray
        Posterior draws of shape (n_sims, n_post_draws)
    true_value : float
        True parameter value for this condition
    cond_id : int
        Condition identifier
    sim_id_start : int
        Starting simulation ID for this batch
    coverage_levels : list of float
        Coverage levels for CI bounds (e.g., [0.50, 0.80, 0.90, 0.95, 0.99])
    full_coverage_levels : list of float
        All coverage levels for profile (1-99%)
        
    Returns:
    --------
    pd.DataFrame with per-simulation metrics for this batch
    """
    n_sims, n_post_draws = draws.shape
    
    # Per-simulation posterior summaries
    posterior_mean = np.mean(draws, axis=1)
    posterior_sd = np.std(draws, axis=1)
    posterior_var = np.var(draws, axis=1, ddof=1)
    posterior_median = np.median(draws, axis=1)
    
    # SBC rank: count of posterior draws < true value
    sbc_ranks = np.sum(draws < true_value, axis=1)
    
    # Errors
    errors = posterior_mean - true_value
    
    # Coverage at ALL levels (1-99%) for full coverage profile
    coverage_results = {}
    for level in full_coverage_levels:
        alpha = 1 - level
        lower_q = alpha / 2 * 100
        upper_q = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(draws, lower_q, axis=1)
        upper_bounds = np.percentile(draws, upper_q, axis=1)
        covered = (true_value >= lower_bounds) & (true_value <= upper_bounds)
        
        level_int = int(level * 100)
        coverage_results[f"covered_{level_int}"] = covered
        # Only store CI bounds for the report levels (to save memory)
        if level in coverage_levels:
            coverage_results[f"ci_lower_{level_int}"] = lower_bounds
            coverage_results[f"ci_upper_{level_int}"] = upper_bounds
    
    # Build DataFrame for this batch
    batch_metrics = pd.DataFrame({
        "id_cond": np.full(n_sims, cond_id, dtype=np.int32),
        "id_sim": np.arange(sim_id_start, sim_id_start + n_sims, dtype=np.int32),
        "true_value": np.full(n_sims, true_value),
        "posterior_mean": posterior_mean,
        "posterior_median": posterior_median,
        "posterior_sd": posterior_sd,
        "posterior_var": posterior_var,
        "error": errors,
        "squared_error": errors ** 2,
        "abs_error": np.abs(errors),
        "sbc_rank": sbc_ranks,
        **coverage_results
    })
    
    return batch_metrics


def aggregate_metrics(
    sim_metrics: pd.DataFrame,
    condition_grid: List[Dict],
    true_param_key: str,
    coverage_levels: List[float],
    full_coverage_levels: List[float],
    n_post_draws: int
) -> Dict:
    """
    Aggregate simulation-level metrics into condition-level and overall summaries.
    
    Called after all batches have been processed and simulation_metrics collected.
    
    Parameters:
    -----------
    sim_metrics : pd.DataFrame
        Combined simulation metrics from all batches
    condition_grid : list of dict
        Condition specifications
    true_param_key : str
        Key for true parameter value in conditions
    coverage_levels : list of float
        Reported coverage levels
    full_coverage_levels : list of float
        All coverage levels for profile
    n_post_draws : int
        Number of posterior draws per simulation
        
    Returns:
    --------
    dict with condition_metrics, condition_summary, simulation_metrics, summary
    """
    # Build lookup for condition info
    cond_info_by_id = {}
    for cond in condition_grid:
        cond_id = cond.get('id_cond', cond.get('id'))
        cond_info_by_id[cond_id] = cond
    
    # Prior variance and range from all true values
    true_values = sim_metrics["true_value"].values
    prior_var = np.var(true_values, ddof=1)
    prior_range = np.max(true_values) - np.min(true_values)
    
    # Compute contraction: 1 - (post_var / prior_var)
    sim_metrics = sim_metrics.copy()
    sim_metrics["contraction"] = np.clip(
        1 - (sim_metrics["posterior_var"].values / prior_var), 0, 1
    )
    
    # Aggregate per condition
    cond_metrics_list = []
    for cond_id in sim_metrics["id_cond"].unique():
        cond_data = sim_metrics[sim_metrics["id_cond"] == cond_id]
        n_cond_sims = len(cond_data)
        cond_true = cond_data["true_value"].iloc[0]
        cond_info = cond_info_by_id.get(cond_id, {})
        
        cond_errors = cond_data["error"].values
        rmse = np.sqrt(np.mean(cond_errors ** 2))
        nrmse = rmse / prior_range if prior_range > 0 else np.nan
        
        # Build row with condition grid parameters FIRST, then metrics
        cond_row = {"id_cond": cond_id}
        
        # Add all condition grid parameters first
        for key, val in cond_info.items():
            if key not in ('id', 'id_cond', true_param_key):
                if isinstance(val, np.ndarray):
                    val = float(val.flat[0]) if val.size == 1 else val.tolist()
                cond_row[key] = val
        
        cond_row["true_value"] = cond_true
        cond_row.update({
            "n_sims": n_cond_sims,
            "rmse": rmse,
            "nrmse": nrmse,
            "mae": np.mean(np.abs(cond_errors)),
            "bias": np.mean(cond_errors),
            "mean_posterior_sd": cond_data["posterior_sd"].mean(),
            "mean_contraction": cond_data["contraction"].mean(),
        })
        
        # Coverage metrics
        for level in coverage_levels:
            level_int = int(level * 100)
            empirical_coverage = cond_data[f"covered_{level_int}"].mean()
            cond_row[f"coverage_{level_int}"] = empirical_coverage
            cond_row[f"cal_error_{level_int}"] = np.abs(empirical_coverage - level)
        
        cal_errors = [cond_row[f"cal_error_{int(l*100)}"] for l in coverage_levels]
        cond_row["mean_cal_error"] = np.mean(cal_errors)
        
        # SBC uniformity tests per condition
        cond_ranks = cond_data["sbc_rank"].values
        cond_sbc_tests = _compute_sbc_uniformity_tests(cond_ranks, n_post_draws)
        cond_row["sbc_ks_pvalue"] = cond_sbc_tests["sbc_ks_pvalue"]
        cond_row["sbc_chi2_pvalue"] = cond_sbc_tests["sbc_chi2_pvalue"]
        
        cond_metrics_list.append(cond_row)
    
    cond_metrics = pd.DataFrame(cond_metrics_list).sort_values("id_cond").reset_index(drop=True)
    
    # Reorder columns
    sample_cond = condition_grid[0] if condition_grid else {}
    cond_param_cols = [k for k in sample_cond.keys() if k not in ('id', 'id_cond', true_param_key)]
    metric_cols = ['n_sims', 'rmse', 'nrmse', 'mae', 'bias', 'mean_posterior_sd', 'mean_contraction']
    coverage_cols = []
    for level in coverage_levels:
        level_int = int(level * 100)
        coverage_cols.extend([f"coverage_{level_int}", f"cal_error_{level_int}"])
    coverage_cols.extend(["mean_cal_error", "sbc_ks_pvalue", "sbc_chi2_pvalue"])
    
    ordered_cols = ['id_cond'] + cond_param_cols + ['true_value'] + metric_cols + coverage_cols
    ordered_cols = [c for c in ordered_cols if c in cond_metrics.columns]
    remaining_cols = [c for c in cond_metrics.columns if c not in ordered_cols]
    cond_metrics = cond_metrics[ordered_cols + remaining_cols]
    
    # Parameter recovery metrics
    posterior_mean = sim_metrics["posterior_mean"].values
    posterior_median = sim_metrics["posterior_median"].values
    recovery_corr_mean = np.corrcoef(true_values, posterior_mean)[0, 1]
    recovery_corr_median = np.corrcoef(true_values, posterior_median)[0, 1]
    recovery_r2_mean = recovery_corr_mean ** 2
    recovery_r2_median = recovery_corr_median ** 2
    
    # Summary metrics
    errors = sim_metrics["error"].values
    squared_errors = sim_metrics["squared_error"].values
    contraction = sim_metrics["contraction"].values
    
    summary = {
        "recovery_corr": recovery_corr_mean,
        "recovery_corr_median": recovery_corr_median,
        "recovery_r2": recovery_r2_mean,
        "recovery_r2_median": recovery_r2_median,
        "overall_rmse": np.sqrt(np.mean(squared_errors)),
        "overall_nrmse": np.sqrt(np.mean(squared_errors)) / prior_range if prior_range > 0 else np.nan,
        "overall_mae": np.mean(np.abs(errors)),
        "overall_bias": np.mean(errors),
        "mean_contraction": np.mean(contraction),
    }
    
    for level in coverage_levels:
        level_int = int(level * 100)
        summary[f"coverage_{level_int}"] = sim_metrics[f"covered_{level_int}"].mean()
    
    summary["mean_cal_error"] = cond_metrics["mean_cal_error"].mean()
    
    # SBC uniformity tests (global)
    sbc_ranks = sim_metrics["sbc_rank"].values
    sbc_tests = _compute_sbc_uniformity_tests(sbc_ranks, n_post_draws)
    summary.update(sbc_tests)
    
    sbc_c2st = _compute_sbc_c2st(sbc_ranks, n_post_draws)
    summary.update(sbc_c2st)
    
    summary["n_post_draws"] = n_post_draws
    
    # Full coverage profile for plotting
    coverage_profile = {}
    for level in full_coverage_levels:
        level_int = int(level * 100)
        coverage_profile[level] = float(sim_metrics[f"covered_{level_int}"].mean())
    summary["coverage_profile"] = coverage_profile
    
    # Condition summary (metrics only, exclude grid parameters)
    numeric_cols = cond_metrics.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude id, counts, and condition grid parameters (keep only metrics)
    sample_cond = condition_grid[0] if condition_grid else {}
    cond_param_keys = [k for k in sample_cond.keys() if k not in ('id', 'id_cond')]
    exclude_cols = ['id_cond', 'n_sims', 'true_value'] + cond_param_keys
    summary_cols = [c for c in numeric_cols if c not in exclude_cols]
    condition_summary = cond_metrics[summary_cols].agg(['mean', 'std', 'min', 'median', 'max']).T
    condition_summary.columns = ['mean', 'sd', 'min', 'median', 'max']
    condition_summary.index.name = 'metric'
    
    return {
        "condition_metrics": cond_metrics,
        "condition_summary": condition_summary,
        "simulation_metrics": sim_metrics,
        "summary": summary
    }


# =============================================================================
# CALIBRATION METRICS EXTRACTION (LEGACY - kept for compatibility)
# =============================================================================

def extract_calibration_metrics(
    condition_grid: List[Dict],
    results_dict: Dict[str, np.ndarray],
    true_param_key: str = "b_arm_treat",
    coverage_levels: Optional[List[float]] = None
) -> Dict:
    """
    Extract calibration metrics from simulation results.
    
    This function computes metrics following BayesFlow conventions:
    - NRMSE: Normalized by prior range (max - min of true values)
    - Posterior contraction: 1 - (posterior_var / prior_var)
    - Calibration error: |empirical_coverage - nominal_coverage|
    
    Parameters:
    -----------
    condition_grid : list of dict
        List of condition dictionaries, each containing:
        - id_cond: condition identifier
        - true parameter value under `true_param_key`
        - additional condition parameters
    results_dict : dict
        Dictionary with keys:
        - id_cond: array of condition IDs for each simulation
        - id_sim: array of simulation IDs
        - draws: matrix of posterior draws (n_sims x n_post_draws)
    true_param_key : str
        Key in condition dict containing the true parameter value
    coverage_levels : list of float, optional
        Coverage levels to compute. Default: [0.50, 0.80, 0.90, 0.95, 0.99]
        
    Returns:
    --------
    dict with:
        - condition_metrics: DataFrame with per-condition aggregated metrics
        - condition_summary: DataFrame with min/max/mean/std/median across conditions
        - simulation_metrics: DataFrame with per-simulation metrics
        - summary: dict with overall metrics
    """
    warnings.warn(
        "extract_calibration_metrics() is deprecated. "
        "Use run_validation_pipeline() instead, which computes the "
        "same metrics incrementally with lower memory usage.",
        DeprecationWarning,
        stacklevel=2,
    )

    if coverage_levels is None:
        coverage_levels = [0.50, 0.80, 0.90, 0.95, 0.99]

    # Full coverage profile for plotting (1% to 99% in 1% steps)
    full_coverage_levels = [i / 100 for i in range(1, 100)]

    id_cond = results_dict["id_cond"]
    id_sim = results_dict["id_sim"]
    draws = results_dict["draws"]
    
    # Build lookup for true values by condition id
    true_values_by_cond = {}
    cond_info_by_id = {}
    for cond in condition_grid:
        cond_id = cond.get('id_cond', cond.get('id'))
        true_val = cond.get(true_param_key)
        if true_val is None:
            true_val = cond.get('b_arm_treat', cond.get('b_group'))
        if isinstance(true_val, np.ndarray):
            true_val = float(true_val.flat[0])
        true_values_by_cond[cond_id] = true_val
        cond_info_by_id[cond_id] = cond
    
    # Per-simulation metrics
    posterior_mean = np.mean(draws, axis=1)
    posterior_sd = np.std(draws, axis=1)
    posterior_var = np.var(draws, axis=1, ddof=1)
    posterior_median = np.median(draws, axis=1)
    true_values = np.array([true_values_by_cond[c] for c in id_cond])

    # SBC rank: count of posterior draws < true value
    # If calibrated, ranks are uniform over {0, 1, ..., n_post_draws}
    # Reference: Talts et al. (2018)
    n_post_draws = draws.shape[1]
    sbc_ranks = np.sum(draws < true_values[:, None], axis=1)
    
    # Prior variance estimated from all true values (as BayesFlow does)
    prior_var = np.var(true_values, ddof=1)
    # Prior range for NRMSE normalization (BayesFlow style)
    prior_range = np.max(true_values) - np.min(true_values)
    
    errors = posterior_mean - true_values
    squared_errors = errors ** 2
    
    # Posterior contraction: 1 - (post_var / prior_var)
    # Values near 1 = high contraction (posterior much narrower than prior)
    # Values near 0 = low contraction (posterior similar to prior)
    # This matches BayesFlow's definition
    contraction = np.clip(1 - (posterior_var / prior_var), 0, 1)
    
    # Coverage at ALL levels (1-99%) for full coverage profile
    # This enables coverage plots without needing raw draws
    coverage_results = {}
    for level in full_coverage_levels:
        alpha = 1 - level
        lower_q = alpha / 2 * 100
        upper_q = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(draws, lower_q, axis=1)
        upper_bounds = np.percentile(draws, upper_q, axis=1)
        covered = (true_values >= lower_bounds) & (true_values <= upper_bounds)
        
        level_int = int(level * 100)
        coverage_results[f"covered_{level_int}"] = covered
        # Only store CI bounds for the report levels (to save memory)
        if level in coverage_levels:
            coverage_results[f"ci_lower_{level_int}"] = lower_bounds
            coverage_results[f"ci_upper_{level_int}"] = upper_bounds
    
    # Per-simulation DataFrame
    sim_metrics = pd.DataFrame({
        "id_cond": id_cond,
        "id_sim": id_sim,
        "true_value": true_values,
        "posterior_mean": posterior_mean,
        "posterior_median": posterior_median,
        "posterior_sd": posterior_sd,
        "posterior_var": posterior_var,
        "error": errors,
        "squared_error": squared_errors,
        "abs_error": np.abs(errors),
        "contraction": contraction,  # BayesFlow-style: 1 - (post_var/prior_var)
        "sbc_rank": sbc_ranks,  # SBC rank statistic (Talts et al. 2018)
        **coverage_results
    })
    
    # Aggregate per condition
    cond_metrics_list = []
    for cond_id in np.unique(id_cond):
        mask = id_cond == cond_id
        n_cond_sims = np.sum(mask)
        cond_true = true_values[mask][0]
        cond_info = cond_info_by_id.get(cond_id, {})
        
        cond_errors = errors[mask]
        rmse = np.sqrt(np.mean(cond_errors ** 2))
        nrmse = rmse / prior_range  # BayesFlow normalizes by (max - min) of targets
        
        # Build row with condition grid parameters FIRST, then metrics
        cond_row = {"id_cond": cond_id}
        
        # Add all condition grid parameters first (excluding id variants and true_param_key)
        for key, val in cond_info.items():
            if key not in ('id', 'id_cond', true_param_key):
                if isinstance(val, np.ndarray):
                    val = float(val.flat[0]) if val.size == 1 else val.tolist()
                cond_row[key] = val
        
        # Add true value (from true_param_key)
        cond_row["true_value"] = cond_true
        
        # Add metrics
        cond_row.update({
            "n_sims": n_cond_sims,
            "rmse": rmse,
            "nrmse": nrmse,
            "mae": np.mean(np.abs(cond_errors)),
            "bias": np.mean(cond_errors),
            "mean_posterior_sd": np.mean(posterior_sd[mask]),
            "mean_contraction": np.mean(contraction[mask]),
        })
        
        # Coverage metrics
        for level in coverage_levels:
            level_int = int(level * 100)
            empirical_coverage = np.mean(coverage_results[f"covered_{level_int}"][mask])
            cond_row[f"coverage_{level_int}"] = empirical_coverage
            cond_row[f"cal_error_{level_int}"] = np.abs(empirical_coverage - level)

        cal_errors = [cond_row[f"cal_error_{int(l*100)}"] for l in coverage_levels]
        cond_row["mean_cal_error"] = np.mean(cal_errors)

        # SBC uniformity tests per condition
        cond_ranks = sbc_ranks[mask]
        cond_sbc_tests = _compute_sbc_uniformity_tests(cond_ranks, n_post_draws)
        cond_row["sbc_ks_pvalue"] = cond_sbc_tests["sbc_ks_pvalue"]
        cond_row["sbc_chi2_pvalue"] = cond_sbc_tests["sbc_chi2_pvalue"]

        cond_metrics_list.append(cond_row)
    
    cond_metrics = pd.DataFrame(cond_metrics_list).sort_values("id_cond").reset_index(drop=True)
    
    # Reorder columns: id_cond, condition params, true_value, then metrics
    # Get all condition parameter keys (excluding id variants and true_param_key)
    sample_cond = condition_grid[0] if condition_grid else {}
    cond_param_cols = [k for k in sample_cond.keys() if k not in ('id', 'id_cond', true_param_key)]
    
    # Define column order: id_cond first, then condition params, true_value, then all metrics
    metric_cols = ['n_sims', 'rmse', 'nrmse', 'mae', 'bias', 'mean_posterior_sd', 'mean_contraction']
    coverage_cols = []
    for level in coverage_levels:
        level_int = int(level * 100)
        coverage_cols.extend([f"coverage_{level_int}", f"cal_error_{level_int}"])
    coverage_cols.append("mean_cal_error")
    # SBC columns
    sbc_cols = ['sbc_ks_pvalue', 'sbc_chi2_pvalue']
    coverage_cols.extend(sbc_cols)
    
    ordered_cols = ['id_cond'] + cond_param_cols + ['true_value'] + metric_cols + coverage_cols
    # Only keep columns that exist (in case some are missing)
    ordered_cols = [c for c in ordered_cols if c in cond_metrics.columns]
    # Add any remaining columns that weren't explicitly ordered
    remaining_cols = [c for c in cond_metrics.columns if c not in ordered_cols]
    cond_metrics = cond_metrics[ordered_cols + remaining_cols]
    
    # Parameter recovery metrics (correlation between true values and estimates)
    # Following BayesFlow's recovery plot convention
    recovery_corr_mean = np.corrcoef(true_values, posterior_mean)[0, 1]
    recovery_corr_median = np.corrcoef(true_values, posterior_median)[0, 1]
    recovery_r2_mean = recovery_corr_mean ** 2
    recovery_r2_median = recovery_corr_median ** 2
    
    # Summary metrics
    summary = {
        # Parameter recovery (BayesFlow style)
        "recovery_corr": recovery_corr_mean,       # Pearson r (posterior mean vs true)
        "recovery_corr_median": recovery_corr_median,  # Pearson r (posterior median vs true)
        "recovery_r2": recovery_r2_mean,           # R² (posterior mean)
        "recovery_r2_median": recovery_r2_median,  # R² (posterior median)
        # Error metrics
        "overall_rmse": np.sqrt(np.mean(squared_errors)),
        "overall_nrmse": np.sqrt(np.mean(squared_errors)) / prior_range,  # BayesFlow style
        "overall_mae": np.mean(np.abs(errors)),
        "overall_bias": np.mean(errors),
        # Posterior quality
        "mean_contraction": np.mean(contraction),
    }
    for level in coverage_levels:
        level_int = int(level * 100)
        summary[f"coverage_{level_int}"] = np.mean(coverage_results[f"covered_{level_int}"])
    
    # Mean calibration error: average of condition-wise mean_cal_error
    # This is more accurate than computing from global coverage, as global coverage
    # can mask condition-specific biases that cancel out
    summary["mean_cal_error"] = cond_metrics["mean_cal_error"].mean()

    # SBC uniformity tests (global across all simulations)
    sbc_tests = _compute_sbc_uniformity_tests(sbc_ranks, n_post_draws)
    summary.update(sbc_tests)

    # C2ST test (classifier-based, may be slower)
    sbc_c2st = _compute_sbc_c2st(sbc_ranks, n_post_draws)
    summary.update(sbc_c2st)

    # Store n_post_draws for use in plotting
    summary["n_post_draws"] = n_post_draws
    
    # Full coverage profile for coverage plots (1-99%)
    # This stores the empirical coverage at each level, enabling coverage plots
    # without needing raw posterior draws
    coverage_profile = {}
    for level in full_coverage_levels:
        level_int = int(level * 100)
        coverage_profile[level] = float(np.mean(coverage_results[f"covered_{level_int}"]))
    summary["coverage_profile"] = coverage_profile

    # Condition summary: aggregate statistics across conditions (metrics only)
    # Exclude id, counts, and condition grid parameters
    numeric_cols = cond_metrics.select_dtypes(include=[np.number]).columns.tolist()
    sample_cond = condition_grid[0] if condition_grid else {}
    cond_param_keys = [k for k in sample_cond.keys() if k not in ('id', 'id_cond')]
    exclude_cols = ['id_cond', 'n_sims', 'true_value'] + cond_param_keys
    summary_cols = [c for c in numeric_cols if c not in exclude_cols]

    condition_summary = cond_metrics[summary_cols].agg(
        ['mean', 'std', 'min', 'median', 'max']
    ).T
    condition_summary.columns = ['mean', 'sd', 'min', 'median', 'max']
    condition_summary.index.name = 'metric'

    return {
        "condition_metrics": cond_metrics,
        "condition_summary": condition_summary,
        "simulation_metrics": sim_metrics,
        "summary": summary
    }


# =============================================================================
# GENERIC VALIDATION PIPELINE
# =============================================================================

def run_validation_pipeline(
    conditions_list: List[Dict],
    n_sims: int,
    n_post_draws: int,
    simulate_fn: Callable,
    infer_fn: Callable,
    true_param_key: str = "b_arm_treat",
    coverage_levels: Optional[List[float]] = None,
    verbose: bool = True,
    progress_every: int = 10
) -> Dict:
    """
    Memory-efficient, model-agnostic validation pipeline.
    
    For each condition: simulate -> infer -> compute metrics -> cleanup memory
    Metrics are computed incrementally after each condition to avoid storing
    all raw posterior draws, significantly reducing memory usage.
    
    Parameters:
    -----------
    conditions_list : list of dict
        Each dict should contain condition parameters including:
        - id_cond: condition identifier (optional, defaults to index)
        - true parameter value under `true_param_key`
    n_sims : int
        Number of simulations per condition
    n_post_draws : int
        Number of posterior draws per simulation
    simulate_fn : callable
        Function signature: simulate_fn(condition, n_sims) -> dict
        Must return dict with data arrays suitable for infer_fn.
    infer_fn : callable
        Function signature: infer_fn(sim_data, n_post_draws) -> np.ndarray
        Must return posterior draws array of shape (n_sims, n_post_draws).
    true_param_key : str
        Key in condition dict containing the true parameter value
    coverage_levels : list of float, optional
        Coverage levels to compute. Default: [0.50, 0.80, 0.90, 0.95, 0.99]
    verbose : bool
        Print progress information
    progress_every : int
        Report progress every N conditions
        
    Returns:
    --------
    dict with:
        - metrics: aggregated metrics (condition_metrics, simulation_metrics, summary)
        - timing: dict with timing information
    """
    if coverage_levels is None:
        coverage_levels = [0.50, 0.80, 0.90, 0.95, 0.99]
    
    # Full coverage profile for plotting (1% to 99% in 1% steps)
    full_coverage_levels = [i / 100 for i in range(1, 100)]
    
    timing = {"simulation": 0.0, "inference": 0.0, "metrics": 0.0}
    n_conditions = len(conditions_list)
    total_sims = n_conditions * n_sims
    
    # Collect simulation metrics incrementally (no raw draws stored)
    all_batch_metrics = []
    
    if verbose:
        print(f"Processing {n_conditions} conditions x {n_sims} sims...")
        print(f"  Total simulations: {total_sims:,}")
        print(f"  Memory-efficient mode: metrics computed per-batch")
    
    t_total_start = time.time()
    sim_counter = 0
    
    for i, cond in enumerate(conditions_list):
        cond_id = cond.get('id_cond', cond.get('id', i))
        
        # Get true value for this condition
        true_val = cond.get(true_param_key)
        if true_val is None:
            true_val = cond.get('b_arm_treat', cond.get('b_group', 0.0))
        if isinstance(true_val, np.ndarray):
            true_val = float(true_val.flat[0])
        
        # Step 1: Simulate
        t0 = time.time()
        sim_data = simulate_fn(cond, n_sims)
        timing["simulation"] += time.time() - t0
        
        # Step 2: Infer
        t1 = time.time()
        draws = infer_fn(sim_data, n_post_draws)
        timing["inference"] += time.time() - t1
        
        # Step 3: Compute metrics immediately (no raw draws stored)
        t2 = time.time()
        batch_metrics = compute_batch_metrics(
            draws=draws,
            true_value=true_val,
            cond_id=cond_id,
            sim_id_start=sim_counter,
            coverage_levels=coverage_levels,
            full_coverage_levels=full_coverage_levels
        )
        all_batch_metrics.append(batch_metrics)
        timing["metrics"] += time.time() - t2
        
        sim_counter += n_sims
        
        # Step 4: Cleanup - draws are no longer needed
        del sim_data, draws
        _cleanup_gpu_memory()
        
        # Progress
        if verbose and ((i + 1) % progress_every == 0 or i == n_conditions - 1):
            elapsed = time.time() - t_total_start
            rate = (i + 1) / elapsed
            eta = (n_conditions - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_conditions}] {elapsed:.1f}s elapsed, ETA: {eta:.1f}s")
    
    # Combine all batch metrics
    if verbose:
        print("Aggregating metrics across conditions...")
    
    t3 = time.time()
    sim_metrics = pd.concat(all_batch_metrics, ignore_index=True)
    
    # Aggregate into final metrics structure
    metrics = aggregate_metrics(
        sim_metrics=sim_metrics,
        condition_grid=conditions_list,
        true_param_key=true_param_key,
        coverage_levels=coverage_levels,
        full_coverage_levels=full_coverage_levels,
        n_post_draws=n_post_draws
    )
    timing["metrics"] += time.time() - t3
    timing["total"] = time.time() - t_total_start
    
    if verbose:
        print(f"\nTiming: Sim {timing['simulation']:.2f}s | Infer {timing['inference']:.2f}s | "
              f"Metrics {timing['metrics']:.2f}s | Total {timing['total']:.2f}s")
        s = metrics["summary"]
        print(f"Summary: r={s['recovery_corr']:.4f}, R²={s['recovery_r2']:.4f}, "
              f"NRMSE={s['overall_nrmse']:.4f}, Bias={s['overall_bias']:.4f}, "
              f"Cal.Error={s['mean_cal_error']:.4f}")
        # SBC metrics
        sbc_ks_p = s.get('sbc_ks_pvalue', np.nan)
        sbc_c2st = s.get('sbc_c2st_accuracy', np.nan)
        print(f"SBC: KS p-value={sbc_ks_p:.4f}, C2ST accuracy={sbc_c2st:.4f}")
    
    return {
        "metrics": metrics,
        "timing": timing
    }
