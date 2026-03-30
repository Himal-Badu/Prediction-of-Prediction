"""
Tests for PoP Cross-Attention Fusion (Option B).
"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')

import torch
import pytest
from pop.core.pop_cross_attention import (
    PoPCrossAttentionFusion,
    create_pop_cross_attention_fusion,
    CrossAttentionBlock,
    FusionMLP,
    AdaptiveGate,
    extract_features_v1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab_size():
    return 50257


@pytest.fixture
def fusion(vocab_size):
    return PoPCrossAttentionFusion(vocab_size=vocab_size, device="cpu")


@pytest.fixture
def sample_inputs(vocab_size):
    logits = torch.randn(1, vocab_size)
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


@pytest.fixture
def batch_inputs(vocab_size):
    logits = torch.randn(4, vocab_size)
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestCrossAttentionInit:
    """Test initialization with various configurations."""

    def test_default_init(self, vocab_size):
        fusion = PoPCrossAttentionFusion(vocab_size=vocab_size, device="cpu")
        assert fusion.vocab_size == vocab_size
        assert fusion.hidden_dim == 64
        assert fusion.device == "cpu"

    def test_custom_hidden_dim(self, vocab_size):
        fusion = PoPCrossAttentionFusion(
            vocab_size=vocab_size, device="cpu", hidden_dim=128, num_heads=4
        )
        assert fusion.hidden_dim == 128

    def test_factory_function(self, vocab_size):
        fusion = create_pop_cross_attention_fusion(vocab_size=vocab_size, device="cpu")
        assert isinstance(fusion, PoPCrossAttentionFusion)

    def test_components_exist(self, fusion):
        assert fusion.v1_projector is not None
        assert fusion.v2_projector is not None
        assert fusion.cross_attention is not None
        assert fusion.fusion_mlp is not None
        assert fusion.adaptive_gate is not None


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Test v1 feature extraction produces correct shapes."""

    def test_v1_features_shape(self, sample_inputs):
        logits, probs = sample_inputs
        features = extract_features_v1(logits, probs)
        assert features.shape == (1, 16)

    def test_v1_features_batch(self, batch_inputs):
        logits, probs = batch_inputs
        features = extract_features_v1(logits, probs)
        assert features.shape == (4, 16)

    def test_v1_features_no_nan(self, sample_inputs):
        logits, probs = sample_inputs
        features = extract_features_v1(logits, probs)
        assert not torch.isnan(features).any()

    def test_v2_features_shape(self, sample_inputs):
        from pop.core.pop_v2 import extract_features_vectorized
        logits, probs = sample_inputs
        features = extract_features_vectorized(logits, probs)
        assert features.shape == (1, 24)


# ---------------------------------------------------------------------------
# Cross-Attention Block Tests
# ---------------------------------------------------------------------------

class TestCrossAttentionBlock:
    """Test cross-attention dimensions and behavior."""

    def test_output_shapes(self):
        block = CrossAttentionBlock(embed_dim=64, num_heads=2)
        v1 = torch.randn(2, 64)
        v2 = torch.randn(2, 64)
        attended_v1, attended_v2 = block(v1, v2)
        assert attended_v1.shape == (2, 64)
        assert attended_v2.shape == (2, 64)

    def test_residual_connection(self):
        """Output should differ from input (attention modifies it)."""
        block = CrossAttentionBlock(embed_dim=64, num_heads=2)
        v1 = torch.randn(1, 64)
        v2 = torch.randn(1, 64)
        attended_v1, attended_v2 = block(v1, v2)
        # Residual + norm means they won't be identical to pure input
        assert not torch.allclose(attended_v1, v1, atol=1e-6)

    def test_invalid_heads_raises(self):
        with pytest.raises(AssertionError):
            CrossAttentionBlock(embed_dim=64, num_heads=3)  # 64 not divisible by 3


# ---------------------------------------------------------------------------
# Fusion MLP Tests
# ---------------------------------------------------------------------------

class TestFusionMLP:
    """Test fusion MLP output shapes."""

    def test_output_shapes(self):
        mlp = FusionMLP(input_dim=256)
        x = torch.randn(2, 256)
        out = mlp(x)
        assert out["error_magnitude"].shape == (2,)
        assert out["confidence"].shape == (2,)
        assert out["error_direction"].shape == (2,)
        assert out["fused_repr"].shape == (2, 64)

    def test_value_ranges(self):
        mlp = FusionMLP(input_dim=256)
        x = torch.randn(10, 256)
        out = mlp(x)
        # error_magnitude should be in [0, 1] (sigmoid)
        assert (out["error_magnitude"] >= 0).all()
        assert (out["error_magnitude"] <= 1).all()
        # confidence should be in [0, 1] (sigmoid)
        assert (out["confidence"] >= 0).all()
        assert (out["confidence"] <= 1).all()
        # error_direction should be in [-1, 1] (tanh)
        assert (out["error_direction"] >= -1).all()
        assert (out["error_direction"] <= 1).all()


# ---------------------------------------------------------------------------
# Adaptive Gate Tests
# ---------------------------------------------------------------------------

class TestAdaptiveGate:
    """Test adaptive gate produces valid alpha values."""

    def test_alpha_range(self):
        gate = AdaptiveGate(input_dim=104)  # 16 + 24 + 64
        x = torch.randn(5, 104)
        alpha = gate(x)
        assert alpha.shape == (5,)
        assert (alpha >= 0).all()
        assert (alpha <= 1).all()

    def test_alpha_is_sigmoid(self):
        """Alpha should be strictly in (0, 1), not hitting exact 0 or 1."""
        gate = AdaptiveGate(input_dim=104)
        x = torch.randn(100, 104)
        alpha = gate(x)
        assert (alpha > 0).all()
        assert (alpha < 1).all()


# ---------------------------------------------------------------------------
# Main Class Predict Tests
# ---------------------------------------------------------------------------

class TestCrossAttentionPredict:
    """Test the full predict() pipeline."""

    def test_predict_returns_all_keys(self, fusion, sample_inputs):
        logits, probs = sample_inputs
        result = fusion.predict(logits, probs)
        required_keys = [
            "error_magnitude", "confidence", "error_direction",
            "should_correct", "llm_likely_wrong", "llm_overconfident",
            "llm_underconfident", "model_type", "alpha",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_value_ranges(self, fusion, sample_inputs):
        logits, probs = sample_inputs
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert -1 <= result["error_direction"] <= 1
        assert 0 <= result["alpha"] <= 1

    def test_predict_model_type(self, fusion, sample_inputs):
        logits, probs = sample_inputs
        result = fusion.predict(logits, probs)
        assert result["model_type"] == "cross_attention_fusion"

    def test_predict_1d_input(self, fusion, vocab_size):
        """Should handle 1D input (single sample without batch dim)."""
        logits = torch.randn(vocab_size)
        probs = torch.softmax(logits, dim=-1)
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1

    def test_predict_batch_input(self, fusion, batch_inputs):
        """Should handle batched input."""
        logits, probs = batch_inputs
        result = fusion.predict(logits, probs)
        # Single scalar results even for batch (takes first)
        assert isinstance(result["error_magnitude"], float)


# ---------------------------------------------------------------------------
# forward_features Tests
# ---------------------------------------------------------------------------

class TestForwardFeatures:
    """Test the raw forward_features path."""

    def test_output_shapes(self, fusion):
        v1_feat = torch.randn(2, 16)
        v2_feat = torch.randn(2, 24)
        result = fusion.forward_features(v1_feat, v2_feat)
        assert result["error_magnitude"].shape == (2,)
        assert result["confidence"].shape == (2,)
        assert result["error_direction"].shape == (2,)
        assert result["alpha"].shape == (2,)
        assert result["attended_v1"].shape == (2, 64)
        assert result["attended_v2"].shape == (2, 64)
        assert result["fused_repr"].shape == (2, 64)

    def test_alpha_valid_range(self, fusion):
        v1_feat = torch.randn(10, 16)
        v2_feat = torch.randn(10, 24)
        result = fusion.forward_features(v1_feat, v2_feat)
        assert (result["alpha"] >= 0).all()
        assert (result["alpha"] <= 1).all()


# ---------------------------------------------------------------------------
# get_params Tests
# ---------------------------------------------------------------------------

class TestGetParams:
    """Test parameter introspection."""

    def test_get_params_keys(self, fusion):
        params = fusion.get_params()
        assert params["model_type"] == "cross_attention_fusion"
        assert params["vocab_size"] == 50257
        assert params["total_parameters"] > 0
        assert "architecture" in params

    def test_architecture_details(self, fusion):
        params = fusion.get_params()
        arch = params["architecture"]
        assert "v1_projector" in arch
        assert "v2_projector" in arch
        assert "cross_attention" in arch
        assert "fusion_mlp" in arch
        assert "adaptive_gate" in arch


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_uniform_distribution(self, fusion, vocab_size):
        """Uniform distribution (max entropy) should produce valid outputs."""
        logits = torch.zeros(1, vocab_size)
        probs = torch.ones(1, vocab_size) / vocab_size
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_peaked_distribution(self, fusion, vocab_size):
        """Very peaked distribution (near-deterministic) should produce valid outputs."""
        logits = torch.full((1, vocab_size), -10.0)
        logits[0, 42] = 10.0
        probs = torch.softmax(logits, dim=-1)
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1  # Valid range (model not trained yet)

    def test_deterministic_output(self, fusion, sample_inputs):
        """Same input should produce same output (eval mode)."""
        logits, probs = sample_inputs
        r1 = fusion.predict(logits, probs)
        r2 = fusion.predict(logits, probs)
        assert r1["error_magnitude"] == r2["error_magnitude"]
        assert r1["alpha"] == r2["alpha"]
