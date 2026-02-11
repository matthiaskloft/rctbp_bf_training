"""Tests for core BayesFlow infrastructure."""

import pytest

from rctbp_bf_training.core.infrastructure import (
    InferenceNetworkConfig,
    SummaryNetworkConfig,
    TrainingConfig,
    WorkflowConfig,
)


# =====================================================================
# SummaryNetworkConfig
# =====================================================================

class TestSummaryNetworkConfig:
    """Test SummaryNetworkConfig dataclass."""

    def test_default_values(self):
        c = SummaryNetworkConfig()
        assert c.summary_dim == 10
        assert c.depth == 3
        assert c.width == 64
        assert c.dropout == 0.05
        assert c.network_type == "DeepSet"

    def test_custom_values(self):
        c = SummaryNetworkConfig(summary_dim=16, depth=5, width=128)
        assert c.summary_dim == 16
        assert c.depth == 5
        assert c.width == 128


# =====================================================================
# InferenceNetworkConfig
# =====================================================================

class TestInferenceNetworkConfig:
    """Test InferenceNetworkConfig dataclass."""

    def test_default_values(self):
        c = InferenceNetworkConfig()
        assert c.depth == 7
        assert c.hidden_sizes == (128, 128)
        assert c.dropout == 0.20
        assert c.network_type == "CouplingFlow"

    def test_custom_hidden_sizes(self):
        c = InferenceNetworkConfig(hidden_sizes=(64, 64, 64))
        assert c.hidden_sizes == (64, 64, 64)


# =====================================================================
# TrainingConfig
# =====================================================================

class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        c = TrainingConfig()
        assert c.initial_lr == 7e-4
        assert c.epochs == 200
        assert c.batch_size == 320
        assert c.early_stopping_patience == 10

    def test_custom_values(self):
        c = TrainingConfig(initial_lr=1e-3, epochs=100)
        assert c.initial_lr == 1e-3
        assert c.epochs == 100


# =====================================================================
# WorkflowConfig serialization
# =====================================================================

class TestWorkflowConfig:
    """Test WorkflowConfig dataclass and serialization."""

    def test_default_construction(self):
        c = WorkflowConfig()
        assert isinstance(c.summary_network, SummaryNetworkConfig)
        assert isinstance(c.inference_network, InferenceNetworkConfig)
        assert isinstance(c.training, TrainingConfig)

    def test_to_dict(self):
        c = WorkflowConfig()
        d = c.to_dict()
        assert isinstance(d, dict)
        assert "summary_network" in d
        assert "inference_network" in d
        assert "training" in d

    def test_roundtrip_serialization(self):
        """to_dict → from_dict should preserve values."""
        original = WorkflowConfig(
            summary_network=SummaryNetworkConfig(summary_dim=16),
            inference_network=InferenceNetworkConfig(depth=5),
            training=TrainingConfig(epochs=50),
        )
        d = original.to_dict()
        restored = WorkflowConfig.from_dict(d)
        assert restored.summary_network.summary_dim == 16
        assert restored.inference_network.depth == 5
        assert restored.training.epochs == 50

    def test_from_dict_with_defaults(self):
        """from_dict with empty dict should use defaults."""
        d = {
            "summary_network": {},
            "inference_network": {},
            "training": {},
        }
        c = WorkflowConfig.from_dict(d)
        assert c.summary_network.summary_dim == 10
        assert c.training.epochs == 200
