"""Tests for diagnostic plotting functions."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rctbp_bf_training.plotting.diagnostics import (
    _create_condition_grid,
    _hide_empty_subplots,
    plot_coverage_diff,
    plot_recovery,
    plot_sbc_diagnostics,
    plot_sbc_ecdf_diff,
    plot_sbc_rank_histogram,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# =====================================================================
# Grid layout helpers
# =====================================================================

class TestCreateConditionGrid:
    """Test _create_condition_grid helper."""

    def test_returns_five_values(self):
        result = _create_condition_grid(10)
        assert len(result) == 5
        fig, axes, n_conds, n_rows, n_cols = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim == 2

    def test_caps_at_max_conditions(self):
        _, _, n_conds, _, _ = _create_condition_grid(100, max_conditions=9)
        assert n_conds == 9

    def test_single_condition(self):
        fig, axes, n_conds, n_rows, n_cols = _create_condition_grid(1)
        assert n_conds == 1
        assert n_rows == 1
        assert n_cols == 1
        assert axes.shape == (1, 1)

    def test_max_cols_respected(self):
        _, _, _, _, n_cols = _create_condition_grid(
            10, max_cols=3
        )
        assert n_cols <= 3

    def test_figsize_scaling(self):
        fig, _, _, _, _ = _create_condition_grid(
            4, figsize_per_plot=(5.0, 4.0)
        )
        w, h = fig.get_size_inches()
        # 4 conditions → 4 cols × 1 row with max_cols=4
        assert w == pytest.approx(20.0, abs=0.1)
        assert h == pytest.approx(4.0, abs=0.1)


class TestHideEmptySubplots:
    """Test _hide_empty_subplots helper."""

    def test_hides_trailing_axes(self):
        fig, axes = plt.subplots(2, 3)
        axes_2d = np.atleast_2d(axes)
        _hide_empty_subplots(axes_2d, n_used=4, n_rows=2, n_cols=3)
        # Axes 4 and 5 should be hidden
        assert not axes_2d[1, 1].get_visible()
        assert not axes_2d[1, 2].get_visible()
        # Axes 0-3 should be visible
        assert axes_2d[0, 0].get_visible()
        assert axes_2d[1, 0].get_visible()


# =====================================================================
# Rank histogram
# =====================================================================

class TestPlotSbcRankHistogram:
    """Test SBC rank histogram plot."""

    @pytest.fixture
    def uniform_ranks(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 200, size=500)

    def test_returns_axes(self, uniform_ranks):
        ax = plot_sbc_rank_histogram(uniform_ranks, n_post_draws=200)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, uniform_ranks):
        fig, ax = plt.subplots()
        result = plot_sbc_rank_histogram(
            uniform_ranks, n_post_draws=200, ax=ax
        )
        assert result is ax

    def test_no_ci(self, uniform_ranks):
        ax = plot_sbc_rank_histogram(
            uniform_ranks, n_post_draws=200, show_ci=False
        )
        assert isinstance(ax, plt.Axes)


# =====================================================================
# ECDF difference
# =====================================================================

class TestPlotSbcEcdfDiff:
    """Test SBC ECDF difference plot."""

    @pytest.fixture
    def uniform_ranks(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 200, size=500)

    def test_returns_axes(self, uniform_ranks):
        ax = plot_sbc_ecdf_diff(uniform_ranks, n_post_draws=200)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self, uniform_ranks):
        fig, ax = plt.subplots()
        result = plot_sbc_ecdf_diff(
            uniform_ranks, n_post_draws=200, ax=ax
        )
        assert result is ax

    def test_no_band(self, uniform_ranks):
        ax = plot_sbc_ecdf_diff(
            uniform_ranks, n_post_draws=200, show_band=False
        )
        assert isinstance(ax, plt.Axes)


# =====================================================================
# Recovery plot
# =====================================================================

class TestPlotRecovery:
    """Test recovery scatter plot."""

    def test_returns_axes_with_1d(self):
        targets = np.random.randn(100)
        estimates = targets + np.random.randn(100) * 0.1
        ax = plot_recovery(estimates, targets)
        assert isinstance(ax, plt.Axes)

    def test_returns_axes_with_2d(self):
        targets = np.random.randn(100)
        estimates = np.random.randn(100, 50)  # draws
        ax = plot_recovery(estimates, targets)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        fig, ax = plt.subplots()
        targets = np.random.randn(50)
        result = plot_recovery(targets, targets, ax=ax)
        assert result is ax


# =====================================================================
# SBC diagnostics (4-panel)
# =====================================================================

class TestPlotSbcDiagnostics:
    """Test 4-panel SBC diagnostic plot."""

    def test_from_ranks_array(self):
        ranks = np.random.randint(0, 200, size=500)
        fig = plot_sbc_diagnostics(ranks, n_post_draws=200)
        assert isinstance(fig, plt.Figure)

    def test_from_metrics_dict(self):
        n_sims = 100
        metrics = {
            "simulation_metrics": pd.DataFrame({
                "sbc_rank": np.random.randint(0, 200, size=n_sims),
                "true_value": np.random.randn(n_sims),
                "posterior_median": np.random.randn(n_sims),
            }),
            "summary": {
                "n_post_draws": 200,
                "coverage_profile": {
                    l / 100: np.clip(l / 100 + np.random.randn() * 0.02, 0, 1)
                    for l in range(1, 100)
                },
                "n_simulations": n_sims,
            },
        }
        fig = plot_sbc_diagnostics(metrics)
        assert isinstance(fig, plt.Figure)


# =====================================================================
# Coverage difference
# =====================================================================

class TestPlotCoverageDiff:
    """Test coverage difference plot."""

    def test_returns_axes(self):
        n_sims = 100
        n_draws = 200
        estimates = np.random.randn(n_sims, n_draws)
        targets = np.random.randn(n_sims)
        ax = plot_coverage_diff(estimates, targets)
        assert isinstance(ax, plt.Axes)

    def test_3d_estimates(self):
        """Handle (n_sims, n_draws, 1) shape."""
        n_sims = 50
        n_draws = 100
        estimates = np.random.randn(n_sims, n_draws, 1)
        targets = np.random.randn(n_sims, 1)
        ax = plot_coverage_diff(estimates, targets)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        fig, ax = plt.subplots()
        estimates = np.random.randn(50, 100)
        targets = np.random.randn(50)
        result = plot_coverage_diff(estimates, targets, ax=ax)
        assert result is ax
