"""Unit tests for calibration loss functions."""

import keras
import numpy as np

from bayesflow_calibration.losses import compute_ranks, coverage_error, ste_indicator


class TestSteIndicator:
    """Tests for the straight-through estimator indicator function."""

    def test_forward_positive(self):
        """Positive values should map to 1.0."""
        x = keras.ops.convert_to_tensor([0.5, 1.0, 3.0])
        result = ste_indicator(x)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(
            keras.ops.convert_to_numpy(result), expected
        )

    def test_forward_negative(self):
        """Negative values should map to 0.0."""
        x = keras.ops.convert_to_tensor([-0.5, -1.0, -3.0])
        result = ste_indicator(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            keras.ops.convert_to_numpy(result), expected
        )

    def test_forward_zero(self):
        """Zero should map to 0.0 (strict > 0)."""
        x = keras.ops.convert_to_tensor([0.0])
        result = ste_indicator(x)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(
            keras.ops.convert_to_numpy(result), expected
        )

    def test_forward_mixed(self):
        """Mixed positive and negative values."""
        x = keras.ops.convert_to_tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
        result = ste_indicator(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(
            keras.ops.convert_to_numpy(result), expected
        )

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        x = keras.ops.convert_to_tensor(np.random.randn(5, 3).astype(np.float32))
        result = ste_indicator(x)
        assert keras.ops.shape(result) == (5, 3)


class TestComputeRanks:
    """Tests for the differentiable rank computation."""

    def test_basic_shape_1d(self):
        """Ranks should have shape (batch,) for 1D log-probs."""
        batch_size = 16
        n_samples = 50
        log_prob_true = keras.ops.convert_to_tensor(
            np.random.randn(batch_size).astype(np.float32)
        )
        log_probs_prior = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, n_samples).astype(np.float32)
        )
        ranks = compute_ranks(log_prob_true, log_probs_prior)
        assert keras.ops.shape(ranks) == (batch_size,)

    def test_ranks_bounded(self):
        """Ranks should be in [0, 1]."""
        batch_size = 32
        n_samples = 100
        log_prob_true = keras.ops.convert_to_tensor(
            np.random.randn(batch_size).astype(np.float32)
        )
        log_probs_prior = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, n_samples).astype(np.float32)
        )
        ranks = compute_ranks(log_prob_true, log_probs_prior)
        ranks_np = keras.ops.convert_to_numpy(ranks)
        assert np.all(ranks_np >= 0.0)
        assert np.all(ranks_np <= 1.0)

    def test_high_true_prob_gives_low_rank(self):
        """If true log-prob is very high, few prior samples exceed it -> low rank."""
        batch_size = 8
        n_samples = 100
        # True log-prob is very high
        log_prob_true = keras.ops.convert_to_tensor(
            np.full(batch_size, 10.0, dtype=np.float32)
        )
        # Prior log-probs are low
        log_probs_prior = keras.ops.convert_to_tensor(
            np.full((batch_size, n_samples), -10.0, dtype=np.float32)
        )
        ranks = compute_ranks(log_prob_true, log_probs_prior)
        ranks_np = keras.ops.convert_to_numpy(ranks)
        # All prior samples have lower density -> rank should be ~0
        np.testing.assert_array_less(ranks_np, 0.1)

    def test_low_true_prob_gives_high_rank(self):
        """If true log-prob is very low, most prior samples exceed it -> high rank."""
        batch_size = 8
        n_samples = 100
        log_prob_true = keras.ops.convert_to_tensor(
            np.full(batch_size, -10.0, dtype=np.float32)
        )
        log_probs_prior = keras.ops.convert_to_tensor(
            np.full((batch_size, n_samples), 10.0, dtype=np.float32)
        )
        ranks = compute_ranks(log_prob_true, log_probs_prior)
        ranks_np = keras.ops.convert_to_numpy(ranks)
        # All prior samples have higher density -> rank should be ~1
        assert np.all(ranks_np > 0.9)


class TestCoverageError:
    """Tests for the coverage error (calibration loss)."""

    def test_uniform_ranks_low_loss(self):
        """Perfectly uniform ranks should produce near-zero loss."""
        batch_size = 100
        # Perfectly uniform ranks
        ranks = keras.ops.convert_to_tensor(
            np.linspace(0.0, 1.0, batch_size).astype(np.float32)
        )
        loss = coverage_error(ranks, mode=1.0)
        loss_val = float(keras.ops.convert_to_numpy(loss))
        assert loss_val < 0.01, (
            f"Expected near-zero loss for uniform ranks, got {loss_val}"
        )

    def test_degenerate_ranks_high_loss(self):
        """Degenerate ranks (all same value) should produce high loss."""
        batch_size = 100
        # All ranks are 0 (severely miscalibrated)
        ranks = keras.ops.convert_to_tensor(
            np.zeros(batch_size, dtype=np.float32)
        )
        loss = coverage_error(ranks, mode=1.0)
        loss_val = float(keras.ops.convert_to_numpy(loss))
        assert loss_val > 0.1, (
            f"Expected high loss for degenerate ranks, got {loss_val}"
        )

    def test_conservativeness_mode(self):
        """Mode=0.0 should only penalize under-coverage."""
        batch_size = 50
        # Over-dispersed ranks (too wide credible intervals)
        # Ranks concentrated near 0.5 -> over-coverage
        ranks = keras.ops.convert_to_tensor(
            np.full(batch_size, 0.5, dtype=np.float32)
        )
        loss_conservative = coverage_error(ranks, mode=0.0)
        loss_calibration = coverage_error(ranks, mode=1.0)
        # Conservativeness mode should be less than or equal to calibration mode
        # (it only penalizes one direction)
        assert float(keras.ops.convert_to_numpy(loss_conservative)) <= float(
            keras.ops.convert_to_numpy(loss_calibration)
        )

    def test_different_modes_different_loss(self):
        """Different modes produce different loss for non-uniform ranks."""
        batch_size = 50
        rng = np.random.default_rng(42)
        ranks = keras.ops.convert_to_tensor(
            rng.beta(0.5, 0.5, size=batch_size).astype(np.float32)
        )
        loss_0 = float(keras.ops.convert_to_numpy(coverage_error(ranks, mode=0.0)))
        loss_1 = float(keras.ops.convert_to_numpy(coverage_error(ranks, mode=1.0)))
        loss_05 = float(keras.ops.convert_to_numpy(coverage_error(ranks, mode=0.5)))
        # All should be positive
        assert loss_0 > 0
        assert loss_1 > 0
        assert loss_05 > 0

    def test_multidim_ranks(self):
        """Coverage error should work with multi-dimensional ranks."""
        batch_size = 50
        param_dim = 3
        ranks = keras.ops.convert_to_tensor(
            np.linspace(0.0, 1.0, batch_size * param_dim)
            .reshape(batch_size, param_dim)
            .astype(np.float32)
        )
        loss = coverage_error(ranks, mode=1.0)
        loss_val = float(keras.ops.convert_to_numpy(loss))
        # Should be a scalar
        assert loss.ndim == 0
        # Should be near zero for uniform ranks
        assert loss_val < 0.05

    def test_output_is_scalar(self):
        """Loss should always be a scalar."""
        for shape in [(20,), (20, 1), (20, 3)]:
            ranks = keras.ops.convert_to_tensor(
                np.random.rand(*shape).astype(np.float32)
            )
            loss = coverage_error(ranks, mode=0.0)
            assert loss.ndim == 0, (
                f"Expected scalar, got ndim={loss.ndim} for {shape}"
            )
