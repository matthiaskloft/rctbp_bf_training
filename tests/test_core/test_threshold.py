"""Tests for threshold-based training loop and validation grid."""

import pytest

from rctbp_bf_training.core.threshold import (
    QualityThresholds,
    check_thresholds,
    create_strict_validation_grid,
    train_until_threshold,
)


# =====================================================================
# QualityThresholds dataclass
# =====================================================================

class TestQualityThresholds:
    """Test the QualityThresholds dataclass."""

    def test_default_values(self):
        t = QualityThresholds()
        assert t.max_cal_error == 0.02
        assert t.max_c2st_deviation == 0.05
        assert t.max_coverage_error == 0.03
        assert t.max_iterations == 10
        assert t.min_improvement == 0.001

    def test_custom_values(self):
        t = QualityThresholds(max_cal_error=0.05, max_iterations=5)
        assert t.max_cal_error == 0.05
        assert t.max_iterations == 5
        # Others should keep defaults
        assert t.max_c2st_deviation == 0.05


# =====================================================================
# check_thresholds
# =====================================================================

class TestCheckThresholds:
    """Test threshold checking logic."""

    @pytest.fixture
    def thresholds(self):
        return QualityThresholds(
            max_cal_error=0.02,
            max_c2st_deviation=0.05,
            max_coverage_error=0.03,
        )

    def test_all_pass(self, thresholds):
        metrics = {
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.52,  # deviation = 0.02
                "coverage_95": 0.96,  # error = 0.01
            }
        }
        passed, scores = check_thresholds(metrics, thresholds)
        assert passed is True
        assert scores["cal_error"] == 0.01
        assert abs(scores["c2st_deviation"] - 0.02) < 1e-10
        assert abs(scores["coverage_error"] - 0.01) < 1e-10

    def test_cal_error_fails(self, thresholds):
        metrics = {
            "summary": {
                "mean_cal_error": 0.10,  # too high
                "sbc_c2st_accuracy": 0.50,
                "coverage_95": 0.95,
            }
        }
        passed, scores = check_thresholds(metrics, thresholds)
        assert passed is False
        assert scores["cal_error"] == 0.10

    def test_c2st_deviation_fails(self, thresholds):
        metrics = {
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.70,  # deviation = 0.20, too high
                "coverage_95": 0.95,
            }
        }
        passed, scores = check_thresholds(metrics, thresholds)
        assert passed is False

    def test_coverage_fails(self, thresholds):
        metrics = {
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.50,
                "coverage_95": 0.80,  # error = 0.15, too high
            }
        }
        passed, scores = check_thresholds(metrics, thresholds)
        assert passed is False

    def test_returns_scores_dict(self, thresholds):
        metrics = {"summary": {"mean_cal_error": 0.01}}
        _, scores = check_thresholds(metrics, thresholds)
        assert "cal_error" in scores
        assert "c2st_deviation" in scores
        assert "coverage_error" in scores

    def test_missing_keys_use_defaults(self, thresholds):
        """Missing metrics use worst-case defaults."""
        metrics = {"summary": {}}
        passed, scores = check_thresholds(metrics, thresholds)
        assert passed is False
        assert scores["cal_error"] == 1.0  # default

    def test_flat_metrics_dict(self, thresholds):
        """Supports metrics without nested 'summary' key."""
        metrics = {
            "mean_cal_error": 0.01,
            "sbc_c2st_accuracy": 0.50,
            "coverage_95": 0.95,
        }
        passed, _ = check_thresholds(metrics, thresholds)
        assert passed is True


# =====================================================================
# train_until_threshold
# =====================================================================

class TestTrainUntilThreshold:
    """Test threshold-based training loop with mocks."""

    def test_converges_first_try(self):
        """If first attempt passes thresholds, loop exits."""
        build_fn = lambda hp: "mock_workflow"  # noqa: E731
        train_fn = lambda wf: "history"  # noqa: E731
        validate_fn = lambda wf: {  # noqa: E731
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.50,
                "coverage_95": 0.95,
            }
        }

        result = train_until_threshold(
            build_fn, train_fn, validate_fn,
            hyperparams={"lr": 0.001},
            thresholds=QualityThresholds(max_iterations=5),
            verbose=False,
        )

        assert result["converged"] is True
        assert result["iterations"] == 1
        assert result["workflow"] == "mock_workflow"

    def test_max_iterations_exhausted(self):
        """If thresholds never met, returns best after max_iterations."""
        call_count = [0]

        def build_fn(hp):
            call_count[0] += 1
            return f"workflow_{call_count[0]}"

        train_fn = lambda wf: "history"  # noqa: E731
        validate_fn = lambda wf: {  # noqa: E731
            "summary": {
                "mean_cal_error": 0.50,  # never passes
                "sbc_c2st_accuracy": 0.80,
                "coverage_95": 0.70,
            }
        }

        result = train_until_threshold(
            build_fn, train_fn, validate_fn,
            hyperparams={},
            thresholds=QualityThresholds(max_iterations=3),
            verbose=False,
        )

        assert result["converged"] is False
        assert result["iterations"] == 3

    def test_training_failure_skipped(self):
        """If training raises, iteration is skipped but loop continues."""
        attempt = [0]

        def train_fn(wf):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("GPU OOM")
            return "history"

        build_fn = lambda hp: "workflow"  # noqa: E731
        validate_fn = lambda wf: {  # noqa: E731
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.50,
                "coverage_95": 0.95,
            }
        }

        result = train_until_threshold(
            build_fn, train_fn, validate_fn,
            hyperparams={},
            thresholds=QualityThresholds(max_iterations=3),
            verbose=False,
        )

        assert result["converged"] is True
        # First attempt failed, second succeeded
        assert result["iterations"] == 1  # only 1 successful history

    def test_result_keys(self):
        """Check all expected keys present in result."""
        build_fn = lambda hp: "wf"  # noqa: E731
        train_fn = lambda wf: "hist"  # noqa: E731
        validate_fn = lambda wf: {  # noqa: E731
            "summary": {
                "mean_cal_error": 0.01,
                "sbc_c2st_accuracy": 0.50,
                "coverage_95": 0.95,
            }
        }

        result = train_until_threshold(
            build_fn, train_fn, validate_fn,
            hyperparams={},
            thresholds=QualityThresholds(max_iterations=1),
            verbose=False,
        )

        expected_keys = {
            "workflow", "metrics", "history",
            "iterations", "converged", "best_scores",
        }
        assert set(result.keys()) == expected_keys


# =====================================================================
# create_strict_validation_grid
# =====================================================================

class TestCreateStrictValidationGrid:
    """Test validation grid creation."""

    def test_default_grid_size(self):
        grid = create_strict_validation_grid()
        # 4 * 2 * 4 * 3 * 3 * 1 = 288
        assert len(grid) == 288

    def test_custom_grid(self):
        grid = create_strict_validation_grid(
            N_vals=[50, 100],
            p_alloc_vals=[0.5],
            prior_df_vals=[0, 10],
            prior_scale_vals=[1.0],
            b_group_vals=[0.0, 0.5],
            b_covariate_vals=[0.0],
        )
        # 2 * 1 * 2 * 1 * 2 * 1 = 8
        assert len(grid) == 8

    def test_grid_has_required_keys(self):
        grid = create_strict_validation_grid(
            N_vals=[100],
            p_alloc_vals=[0.5],
            prior_df_vals=[3],
            prior_scale_vals=[1.0],
            b_group_vals=[0.0],
            b_covariate_vals=[0.0],
        )
        assert len(grid) == 1
        cond = grid[0]
        assert "id_cond" in cond
        assert "n_total" in cond
        assert "p_alloc" in cond
        assert "prior_df" in cond
        assert "prior_scale" in cond
        assert "b_arm_treat" in cond
        assert "b_covariate" in cond

    def test_ids_are_sequential(self):
        grid = create_strict_validation_grid(
            N_vals=[50, 100, 200],
            p_alloc_vals=[0.5],
            prior_df_vals=[0],
            prior_scale_vals=[1.0],
            b_group_vals=[0.0],
            b_covariate_vals=[0.0],
        )
        ids = [c["id_cond"] for c in grid]
        assert ids == list(range(len(grid)))

    def test_single_condition(self):
        grid = create_strict_validation_grid(
            N_vals=[100],
            p_alloc_vals=[0.5],
            prior_df_vals=[0],
            prior_scale_vals=[1.0],
            b_group_vals=[0.0],
            b_covariate_vals=[0.0],
        )
        assert len(grid) == 1
