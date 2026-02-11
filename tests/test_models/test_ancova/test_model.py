"""Tests for ANCOVA model configuration and simulator functions."""

import numpy as np
import pytest

from rctbp_bf_training.models.ancova.model import (
    ANCOVAConfig,
    MetaConfig,
    PriorConfig,
    likelihood,
    meta,
    prior,
)


# =====================================================================
# PriorConfig
# =====================================================================

class TestPriorConfig:
    """Test PriorConfig dataclass."""

    def test_default_values(self):
        c = PriorConfig()
        assert c.b_covariate_scale == 2.0

    def test_custom_values(self):
        c = PriorConfig(b_covariate_scale=5.0)
        assert c.b_covariate_scale == 5.0


# =====================================================================
# MetaConfig
# =====================================================================

class TestMetaConfig:
    """Test MetaConfig dataclass."""

    def test_default_values(self):
        c = MetaConfig()
        assert c.n_min == 20
        assert c.n_max == 1000
        assert c.p_alloc_min == 0.5
        assert c.p_alloc_max == 0.9

    def test_ranges_valid(self):
        c = MetaConfig()
        assert c.n_min < c.n_max
        assert c.p_alloc_min < c.p_alloc_max
        assert c.prior_df_min <= c.prior_df_max


# =====================================================================
# ANCOVAConfig serialization
# =====================================================================

class TestANCOVAConfig:
    """Test ANCOVAConfig composite dataclass."""

    def test_default_construction(self):
        c = ANCOVAConfig()
        assert isinstance(c.prior, PriorConfig)
        assert isinstance(c.meta, MetaConfig)

    def test_to_dict(self):
        c = ANCOVAConfig()
        d = c.to_dict()
        assert "prior" in d
        assert "meta" in d
        assert "workflow" in d

    def test_roundtrip_serialization(self):
        original = ANCOVAConfig(
            prior=PriorConfig(b_covariate_scale=3.0),
            meta=MetaConfig(n_min=50, n_max=500),
        )
        d = original.to_dict()
        restored = ANCOVAConfig.from_dict(d)
        assert restored.prior.b_covariate_scale == 3.0
        assert restored.meta.n_min == 50
        assert restored.meta.n_max == 500


# =====================================================================
# prior function
# =====================================================================

class TestPriorFunction:
    """Test the ANCOVA prior sampling function."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def config(self):
        return PriorConfig()

    def test_returns_dict(self, rng, config):
        result = prior(prior_df=5, prior_scale=1.0, config=config, rng=rng)
        assert isinstance(result, dict)
        assert "b_covariate" in result
        assert "b_group" in result

    def test_output_shapes(self, rng, config):
        result = prior(prior_df=5, prior_scale=1.0, config=config, rng=rng)
        assert result["b_covariate"].shape == (1,)
        assert result["b_group"].shape == (1,)

    def test_output_dtype(self, rng, config):
        result = prior(prior_df=5, prior_scale=1.0, config=config, rng=rng)
        assert result["b_covariate"].dtype == np.float64
        assert result["b_group"].dtype == np.float64

    def test_normal_prior_df_zero(self, rng, config):
        """df=0 should use Normal prior."""
        result = prior(prior_df=0, prior_scale=1.0, config=config, rng=rng)
        assert np.isfinite(result["b_group"][0])

    def test_reproducible(self, config):
        r1 = prior(prior_df=5, prior_scale=1.0, config=config,
                    rng=np.random.default_rng(99))
        r2 = prior(prior_df=5, prior_scale=1.0, config=config,
                    rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1["b_group"], r2["b_group"])


# =====================================================================
# likelihood function
# =====================================================================

class TestLikelihoodFunction:
    """Test the ANCOVA likelihood (data generation) function."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_returns_dict(self, rng):
        result = likelihood(
            b_covariate=0.5, b_group=0.3,
            N=100, p_alloc=0.5, rng=rng,
        )
        assert isinstance(result, dict)
        assert "outcome" in result
        assert "covariate" in result
        assert "group" in result

    def test_output_shapes(self, rng):
        n = 200
        result = likelihood(
            b_covariate=0.5, b_group=0.3,
            N=n, p_alloc=0.5, rng=rng,
        )
        assert result["outcome"].shape == (n,)
        assert result["covariate"].shape == (n,)
        assert result["group"].shape == (n,)

    def test_both_groups_present(self, rng):
        result = likelihood(
            b_covariate=0.0, b_group=0.0,
            N=100, p_alloc=0.5, rng=rng,
        )
        groups = result["group"]
        assert 0 in groups
        assert 1 in groups

    def test_extreme_allocation(self, rng):
        """Even extreme p_alloc should produce both groups."""
        result = likelihood(
            b_covariate=0.0, b_group=0.0,
            N=100, p_alloc=0.99, rng=rng,
        )
        groups = result["group"]
        assert 0 in groups
        assert 1 in groups

    def test_reproducible(self):
        r1 = likelihood(
            b_covariate=0.5, b_group=0.3,
            N=50, p_alloc=0.5,
            rng=np.random.default_rng(123),
        )
        r2 = likelihood(
            b_covariate=0.5, b_group=0.3,
            N=50, p_alloc=0.5,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(r1["outcome"], r2["outcome"])


# =====================================================================
# meta function
# =====================================================================

class TestMetaFunction:
    """Test the meta-parameter sampling function."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def config(self):
        return MetaConfig()

    def test_returns_dict(self, rng, config):
        result = meta(config=config, rng=rng)
        assert isinstance(result, dict)

    def test_required_keys(self, rng, config):
        result = meta(config=config, rng=rng)
        assert "N" in result
        assert "p_alloc" in result
        assert "prior_df" in result
        assert "prior_scale" in result

    def test_n_within_range(self, rng, config):
        for _ in range(50):
            result = meta(config=config, rng=rng)
            n = int(result["N"])
            assert config.n_min <= n <= config.n_max

    def test_p_alloc_within_range(self, rng, config):
        for _ in range(50):
            result = meta(config=config, rng=rng)
            p = float(result["p_alloc"])
            assert config.p_alloc_min <= p <= config.p_alloc_max
