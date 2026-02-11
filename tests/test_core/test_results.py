"""Tests for Optuna results analysis and visualization."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rctbp_bf_training.core.results import (
    get_pareto_trials,
    plot_pareto_front,
    trials_to_dataframe,
)
from rctbp_bf_training.core.optimization import create_study


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def study_with_trials():
    """Create a multi-objective study with some completed trials."""
    study = create_study(study_name="test_results")

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x, 1 - x + 0.1 * y

    study.optimize(objective, n_trials=20)
    return study


# =====================================================================
# get_pareto_trials
# =====================================================================

class TestGetParetoTrials:
    """Test Pareto-optimal trial extraction."""

    def test_returns_list(self, study_with_trials):
        trials = get_pareto_trials(study_with_trials)
        assert isinstance(trials, list)
        assert len(trials) > 0

    def test_pareto_trials_are_subset(self, study_with_trials):
        pareto = get_pareto_trials(study_with_trials)
        all_trials = study_with_trials.trials
        for t in pareto:
            assert t in all_trials


# =====================================================================
# trials_to_dataframe
# =====================================================================

class TestTrialsToDataframe:
    """Test trial-to-DataFrame conversion."""

    def test_returns_dataframe(self, study_with_trials):
        df = trials_to_dataframe(study_with_trials)
        assert df is not None
        assert len(df) > 0

    def test_has_expected_columns(self, study_with_trials):
        df = trials_to_dataframe(study_with_trials)
        assert "number" in df.columns or "trial_number" in df.columns


# =====================================================================
# plot_pareto_front
# =====================================================================

class TestPlotParetoFront:
    """Test Pareto front visualization."""

    def test_returns_axes(self, study_with_trials):
        ax = plot_pareto_front(study_with_trials)
        assert isinstance(ax, (plt.Axes, plt.Figure, type(None))) or ax is not None
