"""Tests for objective computation and parameter normalization."""

import numpy as np
import pytest

from rctbp_bf_training.core.objectives import (
    FAILED_TRIAL_CAL_ERROR,
    FAILED_TRIAL_PARAM_SCORE,
    PARAM_COUNT_LOG_SCALE,
    compute_composite_objective,
    denormalize_param_count,
    estimate_param_count,
    extract_objective_values,
    normalize_param_count,
)


# =====================================================================
# Constants
# =====================================================================

class TestConstants:
    """Test module-level constants have sensible values."""

    def test_param_count_log_scale_positive(self):
        assert PARAM_COUNT_LOG_SCALE > 0

    def test_failed_trial_cal_error_is_worst(self):
        """Failed trial cal error should be the worst possible value."""
        assert FAILED_TRIAL_CAL_ERROR == 1.0

    def test_failed_trial_param_score_exceeds_one(self):
        """Failed param score should exceed 1 (worse than any real)."""
        assert FAILED_TRIAL_PARAM_SCORE > 1.0


# =====================================================================
# estimate_param_count
# =====================================================================

class TestEstimateParamCount:
    """Test analytical parameter count estimation."""

    def test_default_params(self):
        count = estimate_param_count()
        assert isinstance(count, int)
        assert count > 0

    def test_increases_with_flow_depth(self):
        small = estimate_param_count(flow_depth=4)
        large = estimate_param_count(flow_depth=10)
        assert large > small

    def test_increases_with_flow_hidden(self):
        small = estimate_param_count(flow_hidden=64)
        large = estimate_param_count(flow_hidden=256)
        assert large > small

    def test_increases_with_deepset_width(self):
        small = estimate_param_count(deepset_width=32)
        large = estimate_param_count(deepset_width=128)
        assert large > small

    def test_increases_with_deepset_depth(self):
        small = estimate_param_count(deepset_depth=1)
        large = estimate_param_count(deepset_depth=4)
        assert large > small

    def test_minimal_architecture(self):
        count = estimate_param_count(
            summary_dim=2,
            deepset_width=16,
            deepset_depth=1,
            flow_depth=2,
            flow_hidden=32,
            n_conditions=1,
            n_params=1,
        )
        assert count > 0
        assert count < 100_000


# =====================================================================
# normalize / denormalize param count
# =====================================================================

class TestNormalizeDenormalize:
    """Test parameter count normalization roundtrip."""

    @pytest.mark.parametrize("param_count", [100, 1_000, 10_000, 100_000, 1_000_000])
    def test_roundtrip(self, param_count):
        normalized = normalize_param_count(param_count)
        recovered = denormalize_param_count(normalized)
        # Allow small rounding error from int conversion
        assert abs(recovered - param_count) / param_count < 0.01

    def test_normalize_zero(self):
        """Zero params should give finite result."""
        result = normalize_param_count(0)
        assert np.isfinite(result)

    def test_normalize_increases_with_count(self):
        n1 = normalize_param_count(1_000)
        n2 = normalize_param_count(100_000)
        assert n2 > n1

    def test_normalize_typical_range(self):
        """Typical param counts should normalize to roughly 0-2."""
        n = normalize_param_count(50_000)
        assert 0 < n < 2.0

    def test_denormalize_returns_int(self):
        result = denormalize_param_count(0.5)
        assert isinstance(result, int)


# =====================================================================
# compute_composite_objective
# =====================================================================

class TestComputeCompositeObjective:
    """Test composite objective computation."""

    @pytest.fixture
    def good_metrics(self):
        return {
            "summary": {
                "mean_cal_error": 0.01,
            }
        }

    @pytest.fixture
    def bad_metrics(self):
        return {
            "summary": {
                "mean_cal_error": 0.5,
            }
        }

    def test_lower_cal_error_gives_lower_objective(self, good_metrics, bad_metrics):
        good_obj = compute_composite_objective(good_metrics, 10_000)
        bad_obj = compute_composite_objective(bad_metrics, 10_000)
        assert good_obj < bad_obj

    def test_more_params_gives_higher_objective(self, good_metrics):
        small = compute_composite_objective(good_metrics, 10_000)
        large = compute_composite_objective(good_metrics, 500_000)
        assert large > small

    def test_custom_weights(self, good_metrics):
        # Cal-only weight
        obj_cal = compute_composite_objective(
            good_metrics, 10_000,
            weights={"calibration": 1.0, "sbc": 0.0, "size": 0.0},
        )
        # Size-only weight
        obj_size = compute_composite_objective(
            good_metrics, 10_000,
            weights={"calibration": 0.0, "sbc": 0.0, "size": 1.0},
        )
        # They should be different
        assert obj_cal != obj_size

    def test_param_budget_effect(self, good_metrics):
        """Larger param budget should yield lower complexity penalty."""
        strict = compute_composite_objective(good_metrics, 50_000, param_budget=10_000)
        relaxed = compute_composite_objective(good_metrics, 50_000, param_budget=500_000)
        assert relaxed < strict

    def test_returns_float(self, good_metrics):
        result = compute_composite_objective(good_metrics, 10_000)
        assert isinstance(result, float)


# =====================================================================
# extract_objective_values
# =====================================================================

class TestExtractObjectiveValues:
    """Test multi-objective value extraction."""

    def test_returns_tuple_of_two_floats(self):
        metrics = {"summary": {"mean_cal_error": 0.05}}
        result = extract_objective_values(metrics, 30_000)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_cal_error_passed_through(self):
        cal_error = 0.123
        metrics = {"summary": {"mean_cal_error": cal_error}}
        result_cal, _ = extract_objective_values(metrics, 10_000)
        assert result_cal == cal_error

    def test_param_score_is_normalized(self):
        metrics = {"summary": {"mean_cal_error": 0.05}}
        _, param_score = extract_objective_values(metrics, 50_000)
        expected = normalize_param_count(50_000)
        assert abs(param_score - expected) < 1e-6

    def test_missing_cal_error_uses_default(self):
        metrics = {"summary": {}}
        cal, _ = extract_objective_values(metrics, 10_000)
        assert cal == FAILED_TRIAL_CAL_ERROR
