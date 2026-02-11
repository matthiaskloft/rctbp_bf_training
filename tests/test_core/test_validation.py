"""Tests for validation pipeline."""

import warnings

import numpy as np
import pytest

from rctbp_bf_training.core.validation import (
    extract_calibration_metrics,
    make_bayesflow_infer_fn,
    run_validation_pipeline,
)


# =====================================================================
# extract_calibration_metrics deprecation
# =====================================================================

class TestExtractCalibrationMetricsDeprecation:
    """Test that the deprecated function warns."""

    def test_emits_deprecation_warning(self):
        """extract_calibration_metrics should emit DeprecationWarning."""
        # Create minimal valid input
        n_sims = 5
        n_draws = 10
        results_dict = {
            "id_cond": np.zeros(n_sims, dtype=int),
            "id_sim": np.arange(n_sims),
            "draws": np.random.randn(n_sims, n_draws),
            "true_values": np.random.randn(n_sims),
        }
        condition_grid = [{"id_cond": 0, "n_total": 100}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                extract_calibration_metrics(
                    condition_grid, results_dict,
                )
            except Exception:
                pass  # May fail on data shape; we just want the warning

            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(
                deprecation_warnings[0].message
            ).lower()


# =====================================================================
# make_bayesflow_infer_fn
# =====================================================================

class TestMakeBayesflowInferFn:
    """Test inference function factory."""

    def test_returns_callable(self):
        """Should return a callable even with a mock model."""

        class MockApproximator:
            def sample(self, *args, **kwargs):
                return {"b_group": np.random.randn(1, 100, 1)}

        fn = make_bayesflow_infer_fn(
            MockApproximator(),
            param_key="b_group",
        )
        assert callable(fn)
