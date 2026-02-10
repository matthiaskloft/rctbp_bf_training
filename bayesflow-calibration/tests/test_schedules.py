"""Unit tests for gamma scheduling."""

import pytest

from bayesflow_calibration.schedules import GammaSchedule


class TestConstantSchedule:
    """Tests for constant gamma schedule."""

    def test_returns_gamma_max(self):
        schedule = GammaSchedule(schedule_type="constant", gamma_max=50.0)
        assert schedule(0) == 50.0
        assert schedule(100) == 50.0
        assert schedule(1000) == 50.0

    def test_default_gamma(self):
        schedule = GammaSchedule()
        assert schedule(0) == 100.0


class TestLinearWarmupSchedule:
    """Tests for linear warmup gamma schedule."""

    def test_starts_at_gamma_min(self):
        schedule = GammaSchedule(
            schedule_type="linear_warmup",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=20,
        )
        assert schedule(0) == 0.0

    def test_reaches_gamma_max_at_warmup(self):
        schedule = GammaSchedule(
            schedule_type="linear_warmup",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=20,
        )
        assert schedule(20) == 100.0

    def test_midpoint_value(self):
        schedule = GammaSchedule(
            schedule_type="linear_warmup",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=20,
        )
        assert schedule(10) == pytest.approx(50.0)

    def test_stays_at_gamma_max_after_warmup(self):
        schedule = GammaSchedule(
            schedule_type="linear_warmup",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=20,
        )
        assert schedule(50) == 100.0
        assert schedule(200) == 100.0

    def test_nonzero_gamma_min(self):
        schedule = GammaSchedule(
            schedule_type="linear_warmup",
            gamma_max=100.0,
            gamma_min=10.0,
            warmup_epochs=10,
        )
        assert schedule(0) == 10.0
        assert schedule(10) == 100.0
        assert schedule(5) == pytest.approx(55.0)


class TestCosineSchedule:
    """Tests for cosine gamma schedule."""

    def test_starts_at_gamma_min(self):
        schedule = GammaSchedule(
            schedule_type="cosine",
            gamma_max=100.0,
            gamma_min=0.0,
            total_epochs=200,
        )
        assert schedule(0) == pytest.approx(0.0)

    def test_reaches_gamma_max_at_end(self):
        schedule = GammaSchedule(
            schedule_type="cosine",
            gamma_max=100.0,
            gamma_min=0.0,
            total_epochs=200,
        )
        assert schedule(200) == pytest.approx(100.0)

    def test_midpoint_is_half(self):
        schedule = GammaSchedule(
            schedule_type="cosine",
            gamma_max=100.0,
            gamma_min=0.0,
            total_epochs=200,
        )
        # At midpoint, cosine schedule should be at 50%
        assert schedule(100) == pytest.approx(50.0)

    def test_monotonically_increasing(self):
        schedule = GammaSchedule(
            schedule_type="cosine",
            gamma_max=100.0,
            gamma_min=0.0,
            total_epochs=100,
        )
        values = [schedule(e) for e in range(101)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_bounded(self):
        schedule = GammaSchedule(
            schedule_type="cosine",
            gamma_max=100.0,
            gamma_min=10.0,
            total_epochs=50,
        )
        for e in range(60):
            val = schedule(e)
            assert val >= 10.0 - 1e-10
            assert val <= 100.0 + 1e-10


class TestStepSchedule:
    """Tests for step gamma schedule."""

    def test_before_warmup(self):
        schedule = GammaSchedule(
            schedule_type="step",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=10,
        )
        assert schedule(0) == 0.0
        assert schedule(5) == 0.0
        assert schedule(9) == 0.0

    def test_at_warmup(self):
        schedule = GammaSchedule(
            schedule_type="step",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=10,
        )
        assert schedule(10) == 100.0

    def test_after_warmup(self):
        schedule = GammaSchedule(
            schedule_type="step",
            gamma_max=100.0,
            gamma_min=0.0,
            warmup_epochs=10,
        )
        assert schedule(50) == 100.0


class TestValidation:
    """Tests for input validation."""

    def test_invalid_schedule_type(self):
        with pytest.raises(ValueError, match="Unknown schedule_type"):
            GammaSchedule(schedule_type="invalid")

    def test_gamma_max_less_than_min(self):
        with pytest.raises(ValueError, match="gamma_max"):
            GammaSchedule(gamma_max=5.0, gamma_min=10.0)
