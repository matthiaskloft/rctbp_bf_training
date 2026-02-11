"""
SBC Visualization Functions.

Provides plotting utilities for Simulation-Based Calibration (SBC) diagnostics.
For BayesFlow built-in plots, use:
    from bayesflow.diagnostics import calibration_histogram, calibration_ecdf

Reference: Talts et al. (2018) "Validating Bayesian Inference Algorithms
with Simulation-Based Calibration"
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List, Dict
from scipy import stats as scipy_stats
from matplotlib.ticker import PercentFormatter
import pandas as pd
from PIL import Image


# =============================================================================
# GRID LAYOUT HELPER
# =============================================================================

def _create_condition_grid(
    n_conditions: int,
    max_conditions: int = 16,
    max_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (3.0, 3.0),
) -> Tuple[plt.Figure, np.ndarray, int, int, int]:
    """
    Create a subplot grid for condition-level plots.

    Handles grid sizing, axes normalization, and returns the actual
    number of conditions to plot (capped at max_conditions).

    Parameters
    ----------
    n_conditions : int
        Total number of conditions available.
    max_conditions : int
        Maximum number of conditions to plot.
    max_cols : int
        Maximum number of columns in the grid.
    figsize_per_plot : tuple of (float, float)
        Size per subplot (width, height).

    Returns
    -------
    tuple of (fig, axes_2d, n_conds, n_rows, n_cols)
        fig : matplotlib Figure
        axes_2d : 2D ndarray of Axes
        n_conds : actual number of conditions to plot
        n_rows : number of grid rows
        n_cols : number of grid columns
    """
    n_conds = min(n_conditions, max_conditions)
    n_cols = min(max_cols, n_conds)
    n_rows = int(np.ceil(n_conds / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )
    if n_conds == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    return fig, axes, n_conds, n_rows, n_cols


def _hide_empty_subplots(
    axes: np.ndarray,
    n_used: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """Hide unused subplots in a grid."""
    for idx in range(n_used, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def plot_diagnostic_dashboard(
    estimates: Dict,
    targets: Dict,
    param_key: str = "b_group",
    variable_name: str = r"$b_2$ (treatment effect)",
    num_bins: int = 50,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create a 2x2 diagnostic dashboard using BayesFlow plots.
    
    Panels:
    1. Recovery (BayesFlow)
    2. Coverage Difference
    3. Calibration Histogram (BayesFlow)
    4. Calibration ECDF (BayesFlow)
    
    Parameters:
    -----------
    estimates : dict
        Posterior draws dict from workflow.sample()
    targets : dict
        True values dict from simulator.sample()
    param_key : str
        Key for the parameter in both dicts
    variable_name : str
        Display name for the parameter
    num_bins : int
        Number of bins for histogram
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
    """
    import bayesflow as bf
    
    def fig_to_image(fig, dpi=150):
        """Render a matplotlib Figure to a numpy image array."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        return np.asarray(img)

    # Create each diagnostic figure
    fig_recovery = bf.diagnostics.plots.recovery(
        estimates=estimates, 
        targets=targets,
        variable_names=[variable_name]
    )

    fig_coverage_diff = plot_coverage_diff(
        estimates=estimates[param_key], 
        targets=targets[param_key],
        variable_name=variable_name,
        prob=0.95,
        max_points=100
    )

    fig_hist = bf.diagnostics.plots.calibration_histogram(
        estimates=estimates,
        targets=targets,
        num_bins=num_bins,
        variable_names=[variable_name]
    )

    fig_ecdf = bf.diagnostics.plots.calibration_ecdf(
        estimates=estimates, 
        targets=targets,
        variable_names=[variable_name],
        difference=True,
    )

    # Convert figures to images
    img_recovery = fig_to_image(fig_recovery)
    img_coverage_diff = fig_to_image(fig_coverage_diff)
    img_hist = fig_to_image(fig_hist)
    img_ecdf = fig_to_image(fig_ecdf)

    # Combine into a 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    axes[0].imshow(img_recovery)
    axes[0].axis("off")
    axes[0].set_title("Recovery")

    axes[1].imshow(img_coverage_diff)
    axes[1].axis("off")
    axes[1].set_title("Coverage Difference")

    axes[2].imshow(img_hist)
    axes[2].axis("off")
    axes[2].set_title("Calibration Histogram")

    axes[3].imshow(img_ecdf)
    axes[3].axis("off")
    axes[3].set_title("Calibration ECDF (difference)")

    plt.tight_layout()
    return fig


def plot_sbc_rank_histogram(
    ranks: np.ndarray,
    n_post_draws: int,
    n_bins: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = "SBC Rank Histogram",
    color: str = "#132a70",
    show_ci: bool = True,
    ci_level: float = 0.99
) -> plt.Axes:
    """
    Plot histogram of SBC ranks with expected uniform distribution.

    Interpretation:
    - Uniform = well-calibrated
    - U-shape = posterior underdispersed (too narrow/confident)
    - Inverted U (hump) = posterior overdispersed (too wide/uncertain)
    - Left-skewed = systematic overestimation
    - Right-skewed = systematic underestimation

    Parameters:
    -----------
    ranks : array of shape (n_sims,)
        SBC ranks in {0, 1, ..., n_post_draws}
    n_post_draws : int
        Number of posterior draws (determines rank range)
    n_bins : int
        Number of histogram bins (default: 20)
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title
    color : str
        Histogram bar color
    show_ci : bool
        Whether to show confidence interval band
    ci_level : float
        Confidence level for the band (default: 0.99)

    Returns:
    --------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n_bins = min(n_bins, n_post_draws + 1)
    n_sims = len(ranks)
    expected_per_bin = n_sims / n_bins

    # Convert ranks to fractional ranks (0 to 1)
    fractional_ranks = ranks / n_post_draws

    # Plot histogram with fractional ranks on x-axis
    ax.hist(
        fractional_ranks,
        bins=n_bins,
        range=(0, 1),
        edgecolor='white',
        alpha=0.95,
        color=color,
        label='Observed'
    )

    # Confidence interval band using normal approximation to binomial
    if show_ci:
        from scipy.stats import norm
        z = norm.ppf((1 + ci_level) / 2)
        # Binomial std for each bin: sqrt(n * p * (1-p)) where p = 1/n_bins
        p = 1 / n_bins
        std_per_bin = np.sqrt(n_sims * p * (1 - p))
        ci_low = expected_per_bin - z * std_per_bin
        ci_high = expected_per_bin + z * std_per_bin

        ax.axhspan(
            ci_low,
            ci_high,
            alpha=0.3,
            facecolor='grey',
            label=f'{int(ci_level*100)}% CI'
        )

    # Expected uniform line (on top of band)
    ax.axhline(
        expected_per_bin,
        color='grey',
        linestyle='-',
        linewidth=1,
        alpha=0.9,
        zorder=2
    )

    ax.set_xlabel('Rank statistic', fontsize=16)
    ax.set_ylabel('', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=12)
    ax.get_yaxis().set_ticks([])

    return ax


def plot_sbc_ecdf_diff(
    ranks: np.ndarray,
    n_post_draws: int,
    ax: Optional[plt.Axes] = None,
    title: str = "SBC ECDF Difference",
    color: str = "#132a70",
    show_band: bool = True,
    alpha_level: float = 0.05,
    show_legend: bool = True
) -> plt.Axes:
    """
    Plot ECDF difference mimicking BayesFlow's calibration_ecdf(difference=True).

    Uses step-function ECDF with simultaneous confidence bands.
    A well-calibrated posterior should have ECDF difference hovering around 0.

    Parameters:
    -----------
    ranks : array of shape (n_sims,)
        SBC ranks in {0, 1, ..., n_post_draws}
    n_post_draws : int
        Number of posterior draws (determines rank range)
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title
    color : str
        Line color (default: BayesFlow navy #132a70)
    show_band : bool
        Whether to show simultaneous confidence band
    alpha_level : float
        Significance level for band (default: 0.05 for 95% band)
    show_legend : bool
        Whether to show legend

    Returns:
    --------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n_sims = len(ranks)
    
    # Convert to fractional ranks (0 to 1 scale) - BayesFlow style
    fractional_ranks = ranks / n_post_draws
    sorted_ranks = np.sort(fractional_ranks)
    
    # Create step-function ECDF (BayesFlow approach)
    # Repeat each x value twice to create steps
    xx = np.repeat(sorted_ranks, 2)
    xx = np.pad(xx, (1, 1), constant_values=(0, 1))
    yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
    yy = np.repeat(yy, 2)
    
    # Compute difference from uniform (diagonal)
    yy_diff = yy - xx
    
    # Simultaneous confidence bands (BayesFlow lens-shaped bands)
    # Based on the variance of uniform order statistics: Var ~ z*(1-z)
    if show_band:
        epsilon = np.sqrt(np.log(2 / alpha_level) / (2 * n_sims))
        z = np.linspace(0, 1, 200)
        # Lens-shaped bands: scale by sqrt(z*(1-z)) / 0.5 to match BayesFlow
        # Maximum at z=0.5, zero at z=0 and z=1
        scale = np.sqrt(z * (1 - z)) / 0.5
        L = -epsilon * scale
        U = epsilon * scale
        
        ax.fill_between(
            z, L, U,
            color='grey',
            alpha=0.2,
            label=rf"{int((1-alpha_level)*100)}$\%$ Confidence Bands"
        )
    
    # Plot ECDF difference as step function
    ax.plot(xx, yy_diff, color=color, alpha=0.95, label='Rank ECDF')

    ax.set_xlabel('Fractional rank statistic', fontsize=16)
    ax.set_ylabel('ECDF Difference', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=12)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=14)

    return ax


def plot_sbc_diagnostics(
    metrics_or_ranks: Union[Dict, np.ndarray],
    n_post_draws: Optional[int] = None,
    figsize: tuple = (14, 10),
    title_prefix: str = ""
) -> plt.Figure:
    """
    Create a four-panel SBC diagnostic plot combining all simulations across conditions.
    
    Panels:
    1. Rank Histogram - shows distribution of SBC ranks (should be uniform)
    2. ECDF Difference - deviation from uniform CDF
    3. Coverage Profile - empirical vs nominal coverage (from coverage_profile)
    4. Recovery Scatter - posterior mean vs true value

    Parameters:
    -----------
    metrics_or_ranks : dict or array
        Either the metrics dict from run_validation_pipeline()["metrics"],
        or an array of SBC ranks of shape (n_sims,).
    n_post_draws : int, optional
        Number of posterior draws. Required if passing ranks array.
        Inferred from metrics["summary"]["n_post_draws"] if passing dict.
    figsize : tuple
        Figure size (width, height)
    title_prefix : str
        Prefix for subplot titles

    Returns:
    --------
    matplotlib Figure
    """
    # Handle both input types
    if isinstance(metrics_or_ranks, dict):
        metrics = metrics_or_ranks
        sim_metrics = metrics["simulation_metrics"]
        summary = metrics["summary"]
        ranks = sim_metrics["sbc_rank"].values
        n_post_draws = summary["n_post_draws"]
        has_full_metrics = True
    else:
        ranks = np.asarray(metrics_or_ranks)
        if n_post_draws is None:
            n_post_draws = int(np.max(ranks))
        has_full_metrics = False
        metrics = None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Panel 1: Rank Histogram
    hist_title = f"{title_prefix} Rank Histogram" if title_prefix else "SBC Rank Histogram"
    plot_sbc_rank_histogram(ranks, n_post_draws, ax=axes[0], title=hist_title)

    # Panel 2: ECDF Difference
    ecdf_title = f"{title_prefix} ECDF Difference" if title_prefix else "SBC ECDF Difference"
    plot_sbc_ecdf_diff(ranks, n_post_draws, ax=axes[1], title=ecdf_title)

    # Panel 3: Coverage Profile (styled like plot_coverage_diff)
    if has_full_metrics and "coverage_profile" in summary:
        coverage_profile = summary["coverage_profile"]
        levels = sorted(coverage_profile.keys())
        widths = np.array(levels)
        empirical_coverage = np.array([coverage_profile[l] for l in levels])
        diff = empirical_coverage - widths
        
        # Wilson score CI for each coverage level
        n_sims = summary.get("n_simulations", len(metrics["simulation_metrics"]))
        prob = 0.95
        z = scipy_stats.norm.ppf(0.5 + prob / 2)
        ci_low = []
        ci_high = []
        for cov in empirical_coverage:
            denominator = 1 + z**2 / n_sims
            center = (cov + z**2 / (2 * n_sims)) / denominator
            margin = z * np.sqrt(cov * (1 - cov) / n_sims + z**2 / (4 * n_sims**2)) / denominator
            ci_low.append(max(0, center - margin))
            ci_high.append(min(1, center + margin))
        ci_low = np.array(ci_low)
        ci_high = np.array(ci_high)
        diff_low = ci_low - widths
        diff_high = ci_high - widths
        
        ax = axes[2]
        ax.fill_between(widths, diff_low, diff_high, alpha=0.2, color="grey", label=f"{int(prob*100)}% CI")
        ax.plot(widths, diff, "-", color="#132a70", linewidth=1.5, label="Coverage diff")
        ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.7, zorder=0)
        ax.set_xlabel("Nominal Coverage", fontsize=12)
        ax.set_ylabel("Observed - Nominal", fontsize=12)
        ax.set_title("Coverage Profile", fontsize=14)
        ax.set_xlim(0, 1)
        max_abs = max(np.max(np.abs(diff_low)), np.max(np.abs(diff_high)), 0.05)
        ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
        ax.legend(loc="upper left", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        axes[2].text(0.5, 0.5, "Coverage profile\nnot available", 
                     ha='center', va='center', fontsize=12)
        axes[2].set_title("Coverage Profile")
        axes[2].axis('off')

    # Panel 4: Recovery Scatter (if metrics available)
    if has_full_metrics:
        sim_metrics = metrics["simulation_metrics"]
        true_vals = sim_metrics["true_value"].values
        post_medians = sim_metrics["posterior_median"].values
        
        ax = axes[3]
        ax.scatter(true_vals, post_medians, alpha=0.25, s=8, c='#132a70', edgecolors='none')
        
        lims = [min(true_vals.min(), post_medians.min()), 
                max(true_vals.max(), post_medians.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.8)
        
        # Compute correlation (BayesFlow shows r, not R²)
        corr = np.corrcoef(true_vals, post_medians)[0, 1]
        
        ax.set_xlabel("Ground truth", fontsize=12)
        ax.set_ylabel("Estimate", fontsize=12)
        ax.set_title("Recovery", fontsize=14)
        ax.text(0.05, 0.95, f'$r$ = {corr:.3f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)
    else:
        axes[3].text(0.5, 0.5, "Recovery plot\nnot available", 
                     ha='center', va='center', fontsize=12)
        axes[3].set_title("Recovery")
        axes[3].axis('off')

    plt.suptitle(f"SBC Diagnostics (n={len(ranks):,} simulations)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_sbc_from_metrics(
    metrics: Dict,
    figsize: tuple = (12, 5),
    title_prefix: str = ""
) -> plt.Figure:
    """
    Create SBC diagnostic plots directly from validation metrics output.

    Convenience function that extracts sbc_rank and n_post_draws from
    the metrics dict returned by extract_calibration_metrics() or
    run_validation_pipeline().

    Parameters:
    -----------
    metrics : dict
        Output from extract_calibration_metrics() or run_validation_pipeline()["metrics"]
        Must contain 'simulation_metrics' DataFrame with 'sbc_rank' column
        and 'summary' dict with 'n_post_draws' key.
    figsize : tuple
        Figure size (width, height)
    title_prefix : str
        Prefix for subplot titles

    Returns:
    --------
    matplotlib Figure

    Example:
    --------
    >>> results = run_validation_pipeline(...)
    >>> fig = plot_sbc_from_metrics(results["metrics"])
    """
    sim_metrics = metrics["simulation_metrics"]
    summary = metrics["summary"]

    sbc_ranks = sim_metrics["sbc_rank"].values
    n_post_draws = summary["n_post_draws"]

    return plot_sbc_diagnostics(sbc_ranks, n_post_draws, figsize, title_prefix)


def plot_sbc_by_condition(
    metrics_or_df: Union[Dict, pd.DataFrame],
    n_post_draws: Optional[int] = None,
    condition_col: str = "id_cond",
    max_conditions: int = 9,
    figsize_per_plot: tuple = (4, 3)
) -> plt.Figure:
    """
    Plot SBC rank histograms for multiple conditions in a grid.

    Parameters:
    -----------
    metrics_or_df : dict or DataFrame
        Either the full metrics dict from run_validation_pipeline()["metrics"]
        (which contains simulation_metrics and summary with n_post_draws),
        or a DataFrame with 'sbc_rank' and condition columns.
    n_post_draws : int, optional
        Number of posterior draws. Required if passing DataFrame directly.
        Inferred from metrics["summary"]["n_post_draws"] if passing metrics dict.
    condition_col : str
        Column name for condition identifier
    max_conditions : int
        Maximum number of conditions to plot
    figsize_per_plot : tuple
        Size per subplot

    Returns:
    --------
    matplotlib Figure

    Example:
    --------
    >>> results = run_validation_pipeline(...)
    >>> fig = plot_sbc_by_condition(results["metrics"])
    """
    # Handle full results, metrics dict, or DataFrame inputs
    if isinstance(metrics_or_df, dict):
        if "metrics" in metrics_or_df:
            # Full results from run_validation_pipeline()
            metrics = metrics_or_df["metrics"]
        else:
            # Just the metrics dict
            metrics = metrics_or_df
        simulation_metrics = metrics["simulation_metrics"]
        n_post_draws = metrics["summary"]["n_post_draws"]
    else:
        simulation_metrics = metrics_or_df
        if n_post_draws is None:
            raise ValueError("n_post_draws required when passing DataFrame directly")

    conditions = simulation_metrics[condition_col].unique()[:max_conditions]

    fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(
        len(conditions), max_conditions, max_cols=3,
        figsize_per_plot=figsize_per_plot,
    )

    for idx, cond_id in enumerate(conditions[:n_conds]):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        mask = simulation_metrics[condition_col] == cond_id
        ranks = simulation_metrics.loc[mask, 'sbc_rank'].values

        plot_sbc_rank_histogram(
            ranks,
            n_post_draws,
            ax=ax,
            title=f"Condition {cond_id}",
            n_bins=15
        )
        ax.legend().set_visible(False)  # Hide legend for cleaner grid

    _hide_empty_subplots(axes, n_conds, n_rows, n_cols)

    plt.tight_layout()
    return fig


def plot_recovery(
    estimates: np.ndarray,
    targets: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Parameter Recovery",
    color: str = "#132a70"
) -> plt.Axes:
    """
    Plot posterior mean vs true value (recovery plot).

    A well-recovering model should show points along the diagonal.

    Parameters:
    -----------
    estimates : array of shape (n_sims, n_draws) or (n_sims,)
        Posterior draws or posterior means for each simulation
    targets : array of shape (n_sims,)
        True parameter values
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title
    color : str
        Point color

    Returns:
    --------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    estimates = np.asarray(estimates)
    targets = np.asarray(targets).flatten()

    # Compute posterior means if we have draws
    if estimates.ndim == 2:
        post_means = np.mean(estimates, axis=1)
    else:
        post_means = estimates.flatten()

    # Plot scatter (BayesFlow style: semitransparent points)
    ax.scatter(targets, post_means, alpha=0.25, s=20, color=color, edgecolors='none')

    # Diagonal line (black dashed)
    lims = [
        min(targets.min(), post_means.min()),
        max(targets.max(), post_means.max())
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.8)

    # Compute R² (shown in title like BayesFlow)
    corr = np.corrcoef(targets, post_means)[0, 1]
    r2 = corr ** 2

    ax.set_xlabel('Ground truth', fontsize=16)
    ax.set_ylabel('Estimate', fontsize=16)
    ax.set_title(f'{title}', fontsize=18)
    ax.text(0.05, 0.95, f'$r$ = {corr:.3f}', transform=ax.transAxes, 
            fontsize=16, verticalalignment='top')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)

    return ax


def plot_coverage_from_metrics(
    metrics: Dict,
    prob: float = 0.95,
    figsize: tuple = (8, 6),
    title: str = "Coverage Difference Plot"
) -> plt.Figure:
    """
    Plot global coverage difference using pre-computed coverage_profile from metrics.

    Parameters:
    -----------
    metrics : dict
        The metrics dict from run_validation_pipeline()["metrics"]
    prob : float
        Confidence level for Wilson score interval
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns:
    --------
    matplotlib Figure
    """
    summary = metrics["summary"]
    coverage_profile = summary.get("coverage_profile", {})
    
    if not coverage_profile:
        raise ValueError("coverage_profile not found in summary. Re-run validation pipeline.")
    
    sim_metrics = metrics["simulation_metrics"]
    n_sims = len(sim_metrics)
    
    # Extract coverage profile
    levels = sorted(coverage_profile.keys())
    widths = np.array(levels)
    empirical_coverage = np.array([coverage_profile[l] for l in levels])
    diff = empirical_coverage - widths
    
    # Wilson score confidence intervals
    z = scipy_stats.norm.ppf(0.5 + prob / 2)
    ci_low = []
    ci_high = []
    for cov in empirical_coverage:
        denominator = 1 + z**2 / n_sims
        center = (cov + z**2 / (2 * n_sims)) / denominator
        margin = z * np.sqrt(cov * (1 - cov) / n_sims + z**2 / (4 * n_sims**2)) / denominator
        ci_low.append(max(0, center - margin))
        ci_high.append(min(1, center + margin))
    
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)
    diff_low = ci_low - widths
    diff_high = ci_high - widths
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(widths, diff_low, diff_high, alpha=0.2, color="grey",
                    label=f"{int(prob*100)}% confidence band")
    ax.plot(widths, diff, "-", color="#132a70", linewidth=2, label="Coverage difference")
    ax.axhline(0, linestyle="--", color="black", linewidth=1.5, alpha=0.7, zorder=0)
    
    ax.set_xlabel("Credible Interval Width (Nominal Coverage)", fontsize=16)
    ax.set_ylabel("Coverage Difference (Observed - Expected)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlim(0, 1)
    
    max_abs = max(np.max(np.abs(diff_low)), np.max(np.abs(diff_high)), 0.05)
    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
    
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    return fig


def plot_coverage_diff(
    estimates: np.ndarray,
    targets: np.ndarray,
    variable_name: str = "Parameter",
    prob: float = 0.95,
    max_points: int = 50,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot coverage difference (observed - expected) with Wilson score intervals.

    Shows how well the posterior credible intervals cover the true values
    across different credible interval widths.

    Parameters:
    -----------
    estimates : array of shape (n_sims, n_draws) or (n_sims, n_draws, 1)
        Posterior draws for each simulation
    targets : array of shape (n_sims,) or (n_sims, 1)
        True parameter values
    variable_name : str
        Name for the plot title
    prob : float
        Confidence level for the Wilson score interval (default: 0.95)
    max_points : int
        Number of credible interval widths to evaluate
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns:
    --------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Handle different input shapes
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)

    if estimates.ndim == 3:
        estimates = estimates[:, :, 0]
    if targets.ndim == 2:
        targets = targets[:, 0]

    n_sims = estimates.shape[0]
    widths = np.linspace(0, 1, max_points)

    empirical_coverage = []
    ci_low = []
    ci_high = []

    for width in widths:
        if width == 0:
            empirical_coverage.append(0)
            ci_low.append(0)
            ci_high.append(0)
            continue
        if width == 1:
            empirical_coverage.append(1)
            ci_low.append(1)
            ci_high.append(1)
            continue

        alpha = 1 - width
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        in_interval = []
        for i in range(n_sims):
            lower = np.percentile(estimates[i, :], lower_q * 100)
            upper = np.percentile(estimates[i, :], upper_q * 100)
            true_val = targets[i]
            in_interval.append(lower <= true_val <= upper)

        coverage = np.mean(in_interval)
        empirical_coverage.append(coverage)

        # Wilson score confidence interval
        z = scipy_stats.norm.ppf(0.5 + prob / 2)
        denominator = 1 + z**2 / n_sims
        center = (coverage + z**2 / (2 * n_sims)) / denominator
        margin = z * np.sqrt(coverage * (1 - coverage) / n_sims + z**2 / (4 * n_sims**2)) / denominator

        ci_low.append(max(0, center - margin))
        ci_high.append(min(1, center + margin))

    widths = np.asarray(widths)
    empirical_coverage = np.asarray(empirical_coverage)
    ci_low = np.asarray(ci_low)
    ci_high = np.asarray(ci_high)

    diff = empirical_coverage - widths
    diff_low = ci_low - widths
    diff_high = ci_high - widths

    ax.fill_between(widths, diff_low, diff_high, alpha=0.2, color="grey",
                    label=f"{int(prob*100)}% confidence band")
    ax.plot(widths, diff, "o-", color="#132a70", linewidth=2, markersize=4,
            label="Coverage difference")
    ax.axhline(0, linestyle="--", color="black", linewidth=1.5,
               alpha=0.7, zorder=0)

    ax.set_xlabel("Credible Interval Width (Nominal Coverage)", fontsize=16)
    ax.set_ylabel("Coverage Difference (Observed - Expected)", fontsize=16)
    ax.set_title(f"Coverage Difference Plot: {variable_name}", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlim(0, 1)

    max_abs = max(np.max(np.abs(diff_low)), np.max(np.abs(diff_high)), 1e-6)
    ax.set_ylim(-max_abs * 1.05, max_abs * 1.05)

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    return ax


def plot_recovery_by_condition(
    metrics: Dict,
    max_conditions: int = 16,
    figsize_per_plot: tuple = (3, 3)
) -> plt.Figure:
    """
    Plot recovery (posterior median vs true value) for each condition in a grid.
    Uses posterior_median and true_value from simulation_metrics.
    Style matches BayesFlow diagnostics.plots.recovery.

    Parameters:
    -----------
    metrics : dict
        The metrics dict from run_validation_pipeline()["metrics"]
    max_conditions : int
        Maximum number of conditions to show
    figsize_per_plot : tuple
        Size per subplot

    Returns:
    --------
    matplotlib Figure
    """
    sim_metrics = metrics["simulation_metrics"]

    unique_conds = sim_metrics["id_cond"].unique()[:max_conditions]

    # Compute global axis limits across all conditions for consistent scales
    all_true = sim_metrics["true_value"].values
    all_medians = sim_metrics["posterior_median"].values
    global_min = min(all_true.min(), all_medians.min())
    global_max = max(all_true.max(), all_medians.max())
    margin = (global_max - global_min) * 0.05
    global_lims = [global_min - margin, global_max + margin]

    fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(
        len(unique_conds), max_conditions,
        figsize_per_plot=figsize_per_plot,
    )

    for idx, cond_id in enumerate(unique_conds[:n_conds]):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cond_data = sim_metrics[sim_metrics["id_cond"] == cond_id]
        cond_medians = cond_data["posterior_median"].values
        cond_true = cond_data["true_value"].values

        # BayesFlow-style scatter: semitransparent points, identity line
        ax.scatter(
            cond_true, cond_medians,
            alpha=0.25, s=12, c='#132a70', edgecolors='none',
        )
        ax.plot(global_lims, global_lims, 'k--', linewidth=1, alpha=0.8)
        ax.set_xlim(global_lims)
        ax.set_ylim(global_lims)
        ax.set_aspect('equal', adjustable='box')

        # Compute correlation for subtitle (BayesFlow shows r, not R²)
        corr = np.corrcoef(cond_true, cond_medians)[0, 1]
        ax.set_title(f"Cond {cond_id}", fontsize=12)
        ax.text(
            0.05, 0.95, f'$r$={corr:.2f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
        )

        if row == n_rows - 1:
            ax.set_xlabel("Ground truth", fontsize=10)
        if col == 0:
            ax.set_ylabel("Estimate", fontsize=10)
        ax.tick_params(labelsize=9)

    _hide_empty_subplots(axes, n_conds, n_rows, n_cols)

    fig.suptitle(
        "Recovery by Condition (Posterior Median)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_coverage_by_condition(
    metrics: Dict,
    max_conditions: int = 16,
    figsize_per_plot: tuple = (3.5, 3),
    prob: float = 0.95
) -> plt.Figure:
    """
    Plot coverage difference for each condition in a grid.
    Uses pre-computed covered_* columns from simulation_metrics (1-99%).

    Parameters:
    -----------
    metrics : dict
        The metrics dict from run_validation_pipeline()["metrics"]
    max_conditions : int
        Maximum number of conditions to show
    figsize_per_plot : tuple
        Size per subplot
    prob : float
        Confidence level for Wilson score interval

    Returns:
    --------
    matplotlib Figure
    """
    sim_metrics = metrics["simulation_metrics"]

    # Get all coverage levels from column names (covered_1 to covered_99)
    coverage_cols = [c for c in sim_metrics.columns if c.startswith("covered_")]
    levels = sorted([int(c.split("_")[1]) / 100 for c in coverage_cols])

    unique_conds = sim_metrics["id_cond"].unique()[:max_conditions]

    fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(
        len(unique_conds), max_conditions,
        figsize_per_plot=figsize_per_plot,
    )

    for idx, cond_id in enumerate(unique_conds[:n_conds]):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cond_data = sim_metrics[sim_metrics["id_cond"] == cond_id]
        n_sims = len(cond_data)

        # Compute empirical coverage at each level from pre-computed columns
        empirical_coverage = []
        widths = []
        for level in levels:
            level_int = int(level * 100)
            col_name = f"covered_{level_int}"
            if col_name in cond_data.columns:
                empirical_coverage.append(cond_data[col_name].mean())
                widths.append(level)

        widths = np.array(widths)
        empirical_coverage = np.array(empirical_coverage)
        diff = empirical_coverage - widths

        # Wilson score confidence interval
        z = scipy_stats.norm.ppf(0.5 + prob / 2)
        ci_low = []
        ci_high = []
        for cov in empirical_coverage:
            denominator = 1 + z**2 / n_sims
            center = (cov + z**2 / (2 * n_sims)) / denominator
            margin = (
                z * np.sqrt(
                    cov * (1 - cov) / n_sims
                    + z**2 / (4 * n_sims**2)
                ) / denominator
            )
            ci_low.append(max(0, center - margin))
            ci_high.append(min(1, center + margin))

        ci_low = np.array(ci_low)
        ci_high = np.array(ci_high)
        diff_low = ci_low - widths
        diff_high = ci_high - widths

        ax.fill_between(
            widths, diff_low, diff_high, alpha=0.2, color="grey",
        )
        ax.plot(widths, diff, "-", color="#132a70", linewidth=1.5)
        ax.axhline(
            0, linestyle="--", color="black",
            linewidth=1, alpha=0.7, zorder=0,
        )
        ax.set_title(f"Cond {cond_id}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=9)

        max_abs = max(
            np.max(np.abs(diff_low)),
            np.max(np.abs(diff_high)),
            0.1,
        )
        ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    _hide_empty_subplots(axes, n_conds, n_rows, n_cols)

    fig.suptitle(
        "Coverage Difference by Condition",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_histogram_by_condition(
    results_or_metrics: Dict,
    max_conditions: int = 16,
    figsize_per_plot: tuple = (3, 2.5),
    n_bins: int = 15
) -> plt.Figure:
    """
    Plot SBC rank histograms for each condition in a grid.

    Parameters:
    -----------
    results_or_metrics : dict
        Either full output from run_validation_pipeline() or just metrics dict
    max_conditions : int
        Maximum number of conditions to show
    figsize_per_plot : tuple
        Size per subplot
    n_bins : int
        Number of histogram bins

    Returns:
    --------
    matplotlib Figure
    """
    # Handle both full results and metrics-only input
    if "metrics" in results_or_metrics:
        metrics = results_or_metrics["metrics"]
    else:
        metrics = results_or_metrics

    sim_metrics = metrics["simulation_metrics"]
    summary = metrics["summary"]

    id_cond = sim_metrics["id_cond"].values
    n_post_draws = summary["n_post_draws"]

    unique_conds = np.unique(id_cond)[:max_conditions]

    fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(
        len(unique_conds), max_conditions,
        figsize_per_plot=figsize_per_plot,
    )

    for idx, cond_id in enumerate(unique_conds[:n_conds]):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cond_ranks = sim_metrics.loc[
            sim_metrics["id_cond"] == cond_id, "sbc_rank"
        ].values

        plot_sbc_rank_histogram(
            cond_ranks, n_post_draws, ax=ax,
            title=f"Cond {cond_id}", n_bins=n_bins,
        )
        ax.legend().set_visible(False)

    _hide_empty_subplots(axes, n_conds, n_rows, n_cols)

    fig.suptitle(
        "SBC Rank Histograms by Condition",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_ecdf_by_condition(
    results_or_metrics: Dict,
    max_conditions: int = 16,
    figsize_per_plot: tuple = (3, 2.5)
) -> plt.Figure:
    """
    Plot SBC ECDF difference for each condition in a grid.
    Mimics BayesFlow's calibration_ecdf(difference=True) style.

    Parameters:
    -----------
    results_or_metrics : dict
        Either full output from run_validation_pipeline() or just metrics dict
    max_conditions : int
        Maximum number of conditions to show
    figsize_per_plot : tuple
        Size per subplot

    Returns:
    --------
    matplotlib Figure
    """
    # Handle both full results and metrics-only input
    if "metrics" in results_or_metrics:
        metrics = results_or_metrics["metrics"]
    else:
        metrics = results_or_metrics

    sim_metrics = metrics["simulation_metrics"]
    summary = metrics["summary"]

    id_cond = sim_metrics["id_cond"].values
    n_post_draws = summary["n_post_draws"]

    unique_conds = np.unique(id_cond)[:max_conditions]

    fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(
        len(unique_conds), max_conditions,
        figsize_per_plot=figsize_per_plot,
    )

    for idx, cond_id in enumerate(unique_conds[:n_conds]):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cond_ranks = sim_metrics.loc[
            sim_metrics["id_cond"] == cond_id, "sbc_rank"
        ].values

        # Use BayesFlow-style ECDF with no legend on subplots
        plot_sbc_ecdf_diff(
            cond_ranks, n_post_draws, ax=ax,
            title=f"Cond {cond_id}", show_legend=False,
        )

        # Smaller fonts for grid layout
        ax.set_title(f"Cond {cond_id}", fontsize=12)
        ax.set_xlabel("")  # Remove x-labels on inner plots
        ax.set_ylabel("")  # Remove y-labels on inner plots
        ax.tick_params(labelsize=9)

    # Add shared axis labels
    for idx in range(n_conds):
        row, col = divmod(idx, n_cols)
        if row == n_rows - 1:
            axes[row, col].set_xlabel("Fractional rank", fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel("ECDF diff", fontsize=10)

    _hide_empty_subplots(axes, n_conds, n_rows, n_cols)

    fig.suptitle(
        "Calibration ECDF (difference) by Condition",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    return fig
