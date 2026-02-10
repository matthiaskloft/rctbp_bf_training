"""Integration tests for CalibratedContinuousApproximator.

These tests verify that the approximator correctly integrates with
BayesFlow and produces the expected metrics during training.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import keras
import numpy as np
import pytest
from bayesflow.approximators import ContinuousApproximator

from bayesflow_calibration.approximator import (
    CalibratedContinuousApproximator,
)
from bayesflow_calibration.diagnostics import CalibrationMonitorCallback
from bayesflow_calibration.schedules import GammaSchedule


def _make_mock_prior_fn(param_dim=1, seed=42):
    """Create a deterministic prior function for testing."""
    rng = np.random.default_rng(seed)

    def prior_fn(n_samples):
        return rng.standard_normal((n_samples, param_dim)).astype(
            np.float32
        )

    return prior_fn


class TestCalibratedApproximatorInit:
    """Tests for approximator initialization."""

    def test_stores_calibration_params(self):
        """Calibration parameters should be stored as attributes."""
        prior_fn = _make_mock_prior_fn()
        schedule = GammaSchedule(
            schedule_type="constant", gamma_max=50.0
        )

        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            approx = CalibratedContinuousApproximator(
                prior_fn=prior_fn,
                gamma_schedule=schedule,
                calibration_mode=0.5,
                n_rank_samples=200,
                subsample_size=64,
            )

        assert approx.prior_fn is prior_fn
        assert approx.gamma_schedule is schedule
        assert approx.calibration_mode == 0.5
        assert approx.n_rank_samples == 200
        assert approx.subsample_size == 64
        assert approx._current_epoch == 0

    def test_default_gamma_schedule(self):
        """Default gamma schedule should be constant gamma=100."""
        prior_fn = _make_mock_prior_fn()
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            approx = CalibratedContinuousApproximator(
                prior_fn=prior_fn
            )

        assert approx.gamma_schedule.schedule_type == "constant"
        assert approx.gamma_schedule.gamma_max == 100.0

    def test_rejects_invalid_calibration_mode(self):
        """calibration_mode outside [0, 1] should raise."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            with pytest.raises(ValueError, match="calibration_mode"):
                CalibratedContinuousApproximator(
                    prior_fn=_make_mock_prior_fn(),
                    calibration_mode=1.5,
                )

    def test_rejects_negative_calibration_mode(self):
        """Negative calibration_mode should raise."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            with pytest.raises(ValueError, match="calibration_mode"):
                CalibratedContinuousApproximator(
                    prior_fn=_make_mock_prior_fn(),
                    calibration_mode=-0.1,
                )

    def test_rejects_zero_rank_samples(self):
        """n_rank_samples < 1 should raise."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            with pytest.raises(ValueError, match="n_rank_samples"):
                CalibratedContinuousApproximator(
                    prior_fn=_make_mock_prior_fn(),
                    n_rank_samples=0,
                )

    def test_rejects_zero_subsample_size(self):
        """subsample_size < 1 should raise."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            with pytest.raises(ValueError, match="subsample_size"):
                CalibratedContinuousApproximator(
                    prior_fn=_make_mock_prior_fn(),
                    subsample_size=0,
                )


class TestComputeMetrics:
    """Tests for compute_metrics with calibration loss."""

    @pytest.fixture()
    def mock_approximator(self):
        """Create a mock approximator with fake networks."""
        prior_fn = _make_mock_prior_fn(param_dim=1, seed=123)

        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            approx = CalibratedContinuousApproximator(
                prior_fn=prior_fn,
                gamma_schedule=GammaSchedule(
                    schedule_type="constant", gamma_max=10.0
                ),
                calibration_mode=0.0,
                n_rank_samples=20,
            )

        approx.inference_network = MagicMock()
        approx.summary_network = None

        rng = np.random.default_rng(456)

        def mock_log_prob(x, conditions=None):
            batch = keras.ops.shape(x)[0]
            return keras.ops.convert_to_tensor(
                rng.standard_normal(batch).astype(np.float32)
            )

        approx.inference_network.log_prob = mock_log_prob
        return approx

    def test_validation_skips_calibration(self, mock_approximator):
        """Calibration loss not added during validation."""
        rng = np.random.default_rng(10)
        batch_size = 16
        inf_vars = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 1)).astype(np.float32)
        )
        cond = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 4)).astype(np.float32)
        )

        base_loss = keras.ops.convert_to_tensor(1.0)
        with patch.object(
            ContinuousApproximator,
            "compute_metrics",
            return_value={"loss": base_loss},
        ):
            metrics = mock_approximator.compute_metrics(
                inf_vars,
                inference_conditions=cond,
                stage="validation",
            )

        assert "calibration_loss" not in metrics

    def test_training_adds_calibration(self, mock_approximator):
        """Calibration loss should be added during training."""
        rng = np.random.default_rng(20)
        batch_size = 32
        inf_vars = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 1)).astype(np.float32)
        )
        cond = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 4)).astype(np.float32)
        )

        base_loss = keras.ops.convert_to_tensor(1.0)
        with patch.object(
            ContinuousApproximator,
            "compute_metrics",
            return_value={"loss": base_loss},
        ):
            metrics = mock_approximator.compute_metrics(
                inf_vars,
                inference_conditions=cond,
                stage="training",
            )

        assert "calibration_loss" in metrics
        assert "gamma" in metrics
        assert float(metrics["gamma"]) == 10.0
        total = float(keras.ops.convert_to_numpy(metrics["loss"]))
        cal = float(
            keras.ops.convert_to_numpy(metrics["calibration_loss"])
        )
        assert total == pytest.approx(1.0 + 10.0 * cal, rel=1e-5)

    def test_zero_gamma_skips_calibration(self):
        """When gamma=0, calibration loss should not be computed."""
        prior_fn = _make_mock_prior_fn()

        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            approx = CalibratedContinuousApproximator(
                prior_fn=prior_fn,
                gamma_schedule=GammaSchedule(
                    schedule_type="step",
                    gamma_max=100.0,
                    warmup_epochs=10,
                ),
            )

        approx._current_epoch = 0  # Before warmup -> gamma=0
        approx.inference_network = MagicMock()
        approx.summary_network = None

        rng = np.random.default_rng(30)
        batch_size = 16
        inf_vars = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 1)).astype(np.float32)
        )
        cond = keras.ops.convert_to_tensor(
            rng.standard_normal((batch_size, 4)).astype(np.float32)
        )

        base_loss = keras.ops.convert_to_tensor(1.0)
        with patch.object(
            ContinuousApproximator,
            "compute_metrics",
            return_value={"loss": base_loss},
        ):
            metrics = approx.compute_metrics(
                inf_vars,
                inference_conditions=cond,
                stage="training",
            )

        assert "calibration_loss" not in metrics


class TestBuildConditions:
    """Tests for the condition-merging helper."""

    @pytest.fixture()
    def approx(self):
        """Create a minimal approximator."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            approx = CalibratedContinuousApproximator(
                prior_fn=_make_mock_prior_fn(),
            )
        approx.summary_network = None
        return approx

    def test_inference_conditions_only(self, approx):
        """Returns inference_conditions when no summary network."""
        rng = np.random.default_rng(40)
        cond = keras.ops.convert_to_tensor(
            rng.standard_normal((8, 4)).astype(np.float32)
        )
        result = approx._build_conditions(None, cond)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(result),
            keras.ops.convert_to_numpy(cond),
        )

    def test_with_summary_network(self, approx):
        """Concatenates inference_conditions and summary output."""
        approx.summary_network = MagicMock()
        approx.summary_network.return_value = (
            keras.ops.convert_to_tensor(
                np.ones((8, 2), dtype=np.float32)
            )
        )

        cond = keras.ops.convert_to_tensor(
            np.zeros((8, 4), dtype=np.float32)
        )
        rng = np.random.default_rng(50)
        summ = keras.ops.convert_to_tensor(
            rng.standard_normal((8, 10, 5)).astype(np.float32)
        )

        result = approx._build_conditions(summ, cond)
        # Should be (8, 6) = 4 + 2
        assert keras.ops.shape(result) == (8, 6)

    def test_raises_without_any_conditions(self, approx):
        """Raises if neither conditions nor summary are provided."""
        with pytest.raises(ValueError, match="requires at least one"):
            approx._build_conditions(None, None)


class TestCalibrationMonitorCallback:
    """Tests for the epoch-counter callback."""

    def test_updates_epoch(self):
        """Callback should set _current_epoch on the model."""
        with patch.object(
            ContinuousApproximator, "__init__", return_value=None
        ):
            model = CalibratedContinuousApproximator(
                prior_fn=_make_mock_prior_fn(),
            )

        cb = CalibrationMonitorCallback()
        cb.set_model(model)

        assert model._current_epoch == 0
        cb.on_epoch_begin(5)
        assert model._current_epoch == 5
        cb.on_epoch_begin(10)
        assert model._current_epoch == 10

    def test_silent_on_non_calibrated_model(self):
        """Callback should not raise on model without _current_epoch."""
        cb = CalibrationMonitorCallback()
        mock_model = MagicMock(spec=[])  # no _current_epoch attr
        cb.set_model(mock_model)
        # Should not raise
        cb.on_epoch_begin(5)
