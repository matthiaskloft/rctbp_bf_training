"""Tests for Bayesian optimization utilities."""

import pytest

from rctbp_bf_training.core.optimization import (
    HyperparameterSpace,
    create_study,
)


# =====================================================================
# HyperparameterSpace
# =====================================================================

class TestHyperparameterSpace:
    """Test HyperparameterSpace dataclass defaults and fields."""

    def test_default_construction(self):
        hs = HyperparameterSpace()
        assert isinstance(hs.summary_dim, tuple)
        assert len(hs.summary_dim) == 2
        assert hs.summary_dim[0] < hs.summary_dim[1]

    def test_flow_depth_range(self):
        hs = HyperparameterSpace()
        low, high = hs.flow_depth
        assert low >= 1
        assert high >= low

    def test_initial_lr_range(self):
        hs = HyperparameterSpace()
        low, high = hs.initial_lr
        assert 0 < low < high


# =====================================================================
# create_study
# =====================================================================

class TestCreateStudy:
    """Test Optuna study creation."""

    def test_creates_study(self):
        study = create_study(study_name="test_study")
        assert study is not None
        assert study.study_name == "test_study"

    def test_default_directions(self):
        study = create_study(study_name="test_multi")
        # Should be multi-objective by default
        assert len(study.directions) == 2

    def test_custom_directions(self):
        study = create_study(
            study_name="test_single",
            directions=["minimize"],
        )
        assert len(study.directions) == 1

    def test_in_memory_storage(self):
        """Default (no storage) creates in-memory study."""
        study = create_study(study_name="inmem")
        assert study is not None

    def test_load_if_exists(self):
        """Creating study twice with same name shouldn't error."""
        study1 = create_study(study_name="duplicate_test")
        study2 = create_study(
            study_name="duplicate_test", load_if_exists=True
        )
        assert study2 is not None
