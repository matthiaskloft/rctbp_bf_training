"""Tests for utility functions."""

import numpy as np
import pytest

from rctbp_bf_training.core.utils import (
    loguniform_float,
    loguniform_int,
    sample_t_or_normal,
)


# =====================================================================
# loguniform_int
# =====================================================================

class TestLoguniformInt:
    """Test log-uniform integer sampling."""

    def test_returns_int(self):
        result = loguniform_int(1, 100)
        assert isinstance(result, (int, np.integer))

    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            val = loguniform_int(10, 1000, rng=rng)
            assert 10 <= val <= 1000

    def test_low_equals_high(self):
        result = loguniform_int(50, 50)
        assert result == 50

    def test_different_alpha(self):
        """Alpha parameter should affect distribution shape."""
        rng = np.random.default_rng(0)
        samples_low = [
            loguniform_int(1, 1000, alpha=0.5, rng=rng)
            for _ in range(500)
        ]
        rng = np.random.default_rng(0)
        samples_high = [
            loguniform_int(1, 1000, alpha=2.0, rng=rng)
            for _ in range(500)
        ]
        # Different alpha → different distribution
        assert np.mean(samples_low) != np.mean(samples_high)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        assert loguniform_int(1, 100, rng=rng1) == loguniform_int(
            1, 100, rng=rng2
        )


# =====================================================================
# loguniform_float
# =====================================================================

class TestLoguniformFloat:
    """Test log-uniform float sampling."""

    def test_returns_float(self):
        result = loguniform_float(0.001, 1.0)
        assert isinstance(result, float)

    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            val = loguniform_float(0.01, 10.0, rng=rng)
            assert 0.01 <= val <= 10.0

    def test_different_alpha(self):
        rng = np.random.default_rng(0)
        samples_low = [
            loguniform_float(0.01, 10.0, alpha=0.5, rng=rng)
            for _ in range(500)
        ]
        rng = np.random.default_rng(0)
        samples_high = [
            loguniform_float(0.01, 10.0, alpha=2.0, rng=rng)
            for _ in range(500)
        ]
        assert np.mean(samples_low) != np.mean(samples_high)


# =====================================================================
# sample_t_or_normal
# =====================================================================

class TestSampleTOrNormal:
    """Test Student-t / Normal sampling."""

    def test_returns_float(self):
        result = sample_t_or_normal(df=5)
        assert isinstance(result, (float, np.floating))

    def test_normal_when_df_zero(self):
        """df=0 should sample from Normal distribution."""
        rng = np.random.default_rng(42)
        samples = [
            sample_t_or_normal(df=0, scale=1.0, rng=rng)
            for _ in range(1000)
        ]
        # Should be approximately standard normal
        assert abs(np.mean(samples)) < 0.15
        assert abs(np.std(samples) - 1.0) < 0.15

    def test_scale_affects_spread(self):
        rng = np.random.default_rng(42)
        narrow = [
            sample_t_or_normal(df=5, scale=0.1, rng=rng)
            for _ in range(1000)
        ]
        rng = np.random.default_rng(42)
        wide = [
            sample_t_or_normal(df=5, scale=10.0, rng=rng)
            for _ in range(1000)
        ]
        assert np.std(wide) > np.std(narrow)

    def test_large_df_approaches_normal(self):
        """With large df, t-distribution approximates normal."""
        rng = np.random.default_rng(42)
        samples = [
            sample_t_or_normal(df=1000, scale=1.0, rng=rng)
            for _ in range(1000)
        ]
        assert abs(np.std(samples) - 1.0) < 0.15

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        assert sample_t_or_normal(
            df=5, rng=rng1
        ) == sample_t_or_normal(df=5, rng=rng2)
