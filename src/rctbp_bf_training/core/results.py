"""
Results Analysis and Visualization for Optuna Studies.

Utilities for extracting, summarizing, and plotting optimization results
from Optuna hyperparameter studies.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from rctbp_bf_training.core.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_PARAM_SCORE,
    denormalize_param_count,
)

# Optional imports with graceful fallback
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def get_pareto_trials(
    study: "optuna.Study",
) -> List["optuna.trial.FrozenTrial"]:
    """
    Get Pareto-optimal trials from a multi-objective study.

    Parameters
    ----------
    study : optuna.Study
        Completed or in-progress study.

    Returns
    -------
    list of optuna.trial.FrozenTrial
        Pareto-optimal trials.

    Raises
    ------
    ImportError
        If Optuna is not installed.
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

    Parameters
    ----------
    study : optuna.Study
        The study to extract trials from.
    include_pruned : bool
        Whether to include pruned trials.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for params and objectives.

    Raises
    ------
    ImportError
        If Optuna is not installed.
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
        elif (
            include_pruned
            and trial.state == optuna.trial.TrialState.PRUNED
        ):
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

    Parameters
    ----------
    study : optuna.Study
        The completed study.
    n_best : int
        Number of best trials to return (for single-objective).

    Returns
    -------
    pd.DataFrame
        DataFrame with trial parameters and objectives.
    """
    if len(study.directions) > 1:
        # Multi-objective: get Pareto front
        best_trials = study.best_trials
    else:
        # Single-objective: get top N
        best_trials = sorted(
            [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ],
            key=lambda t: t.value,
        )[:n_best]

    records = []
    for trial in best_trials:
        record = {"trial": trial.number}
        record.update(trial.params)

        if isinstance(trial.values, (list, tuple)):
            record["cal_error"] = trial.values[0]
            if len(trial.values) > 1:
                # Denormalize param_count for human-readable display
                normalized_params = trial.values[1]
                record["param_count"] = denormalize_param_count(
                    normalized_params
                )
                record["param_score"] = normalized_params
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
) -> Any:
    """
    Plot optimization results: Pareto front, parameter importance, best configs.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    figsize : tuple of (int, int)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Pareto front (for multi-objective)
    ax = axes[0]
    if len(study.directions) > 1:
        trials_df = trials_to_dataframe(study)
        if (
            "objective_0" in trials_df.columns
            and "objective_1" in trials_df.columns
        ):
            # Filter out failed trials (those with penalty values)
            valid_mask = (
                (trials_df["objective_0"] < FAILED_TRIAL_CAL_ERROR)
                & (trials_df["objective_1"] < FAILED_TRIAL_PARAM_SCORE)
            )
            valid_df = trials_df[valid_mask]

            if len(valid_df) > 0:
                denorm_params = valid_df["objective_1"].apply(
                    denormalize_param_count
                )
                ax.scatter(
                    valid_df["objective_0"],
                    denorm_params,
                    alpha=0.5,
                    label=f"Successful trials ({len(valid_df)})",
                )

            # Show failed trials count if any
            n_failed = len(trials_df) - len(valid_df)
            if n_failed > 0:
                ax.text(
                    0.95, 0.95, f"Failed: {n_failed}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, color='gray',
                )

            # Highlight Pareto front (filter valid ones)
            pareto = study.best_trials
            pareto_obj0 = [
                t.values[0] for t in pareto
                if t.values[0] < FAILED_TRIAL_CAL_ERROR
                and t.values[1] < FAILED_TRIAL_PARAM_SCORE
            ]
            pareto_obj1 = [
                denormalize_param_count(t.values[1]) for t in pareto
                if t.values[0] < FAILED_TRIAL_CAL_ERROR
                and t.values[1] < FAILED_TRIAL_PARAM_SCORE
            ]
            if pareto_obj0:
                ax.scatter(
                    pareto_obj0, pareto_obj1,
                    c='red', s=100, marker='*',
                    label="Pareto front",
                    zorder=10,
                )

            ax.set_xlabel("Calibration Error")
            ax.set_ylabel("Parameter Count")
            ax.set_title("Pareto Front (normalized internally)")
            ax.legend()
    else:
        # Single objective: plot optimization history
        trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
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
        ax.text(
            0.5, 0.5, "Importance N/A\n(need more trials)",
            ha='center', va='center', transform=ax.transAxes,
        )
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

        ax.text(
            0.1, 0.9, "\n".join(text_lines[:20]),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
        )

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

    Parameters
    ----------
    study : optuna.Study
        Multi-objective study.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    highlight_best : bool
        Whether to highlight Pareto-optimal points.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    trials_df = trials_to_dataframe(study)

    if "objective_0" not in trials_df.columns:
        ax.text(
            0.5, 0.5, "Single objective study",
            ha='center', va='center', transform=ax.transAxes,
        )
        return ax

    # Filter valid trials and denormalize for display
    valid_mask = (
        (trials_df["objective_0"] < FAILED_TRIAL_CAL_ERROR)
        & (trials_df["objective_1"] < FAILED_TRIAL_PARAM_SCORE)
    )
    valid_df = trials_df[valid_mask]

    if len(valid_df) > 0:
        denorm_params = valid_df["objective_1"].apply(
            denormalize_param_count
        )
        ax.scatter(
            valid_df["objective_0"],
            denorm_params,
            alpha=0.4,
            s=50,
            label="All trials",
        )

    if highlight_best:
        pareto = study.best_trials
        # Filter and denormalize Pareto front
        valid_pareto = [
            (t.values[0], denormalize_param_count(t.values[1]))
            for t in pareto
            if t.values[0] < FAILED_TRIAL_CAL_ERROR
            and t.values[1] < FAILED_TRIAL_PARAM_SCORE
        ]

        if valid_pareto:
            pareto_obj0 = [p[0] for p in valid_pareto]
            pareto_obj1 = [p[1] for p in valid_pareto]

            # Sort for line plot
            sorted_idx = np.argsort(pareto_obj0)
            pareto_obj0 = np.array(pareto_obj0)[sorted_idx]
            pareto_obj1 = np.array(pareto_obj1)[sorted_idx]

            ax.plot(
                pareto_obj0, pareto_obj1,
                'r--', alpha=0.7, linewidth=2,
            )
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
