import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import pytest
from pop.core.pop_fusion import PoPFusion, create_pop_fusion


class TestPoPFusionInit:
    """Test Fusion layer initialization with different configurations."""

    def test_init_distributional_only(self):
        fusion = PoPFusion(vocab_size=50257, model_type="distributional")
        assert fusion.v1 is not None
        assert fusion.v2 is None
        assert fusion.model_type == "distributional"

    def test_init_contextual_only(self):
        fusion = PoPFusion(vocab_size=50257, model_type="contextual")
        assert fusion.v1 is None
        assert fusion.v2 is not None
        assert fusion.model_type == "contextual"

    def test_init_fusion_both(self):
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        assert fusion.v1 is not None
        assert fusion.v2 is not None
        assert fusion.model_type == "fusion"
        assert fusion._fusion_logit is not None

    def test_init_invalid_type_raises(self):
        with pytest.raises(AssertionError):
            PoPFusion(vocab_size=50257, model_type="invalid")

    def test_init_default_is_fusion(self):
        fusion = PoPFusion(vocab_size=50257)
        assert fusion.model_type == "fusion"

    def test_factory_function(self):
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")
        assert isinstance(fusion, PoPFusion)
        assert fusion.model_type == "distributional"


class TestPoPFusionPredict:
    """Test forward pass produces valid outputs."""

    @pytest.fixture
    def sample_inputs(self):
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def test_distributional_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="distributional")
        result = fusion.predict(logits, probs)
        assert "error_magnitude" in result
        assert "confidence" in result
        assert "error_direction" in result
        assert "should_correct" in result
        assert result["model_type"] == "distributional"
        assert result["v1_weight"] == 1.0

    def test_contextual_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="contextual")
        result = fusion.predict(logits, probs)
        assert "error_magnitude" in result
        assert "confidence" in result
        assert "error_direction" in result
        assert result["model_type"] == "contextual"
        assert result["v1_weight"] == 0.0

    def test_fusion_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        result = fusion.predict(logits, probs)
        assert "error_magnitude" in result
        assert "confidence" in result
        assert "error_direction" in result
        assert result["model_type"] == "fusion"
        # Default fusion weight should be ~0.5 (sigmoid(0) = 0.5)
        assert 0.3 <= result["v1_weight"] <= 0.7


class TestPoPFusionOutputShapes:
    """Test output shapes and value ranges."""

    @pytest.fixture
    def sample_inputs(self):
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def test_single_sample_shapes(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        result = fusion.predict(logits, probs)
        # Scalars for single sample
        assert isinstance(result["error_magnitude"], float)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["error_direction"], float)

    def test_output_ranges_distributional(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="distributional")
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert -1 <= result["error_direction"] <= 1

    def test_output_ranges_contextual(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="contextual")
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert -1 <= result["error_direction"] <= 1

    def test_output_ranges_fusion(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        result = fusion.predict(logits, probs)
        assert 0 <= result["error_magnitude"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert -1 <= result["error_direction"] <= 1

    def test_batched_input(self):
        logits = torch.randn(4, 50257)
        probs = torch.softmax(logits, dim=-1)
        fusion = PoPFusion(vocab_size=50257, model_type="contextual")
        result = fusion.predict(logits, probs)
        assert "error_magnitude" in result
        assert "confidence" in result

    def test_get_params(self):
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        params = fusion.get_params()
        assert params["model_type"] == "fusion"
        assert params["vocab_size"] == 50257
        assert "v1_weight" in params
        assert "v1_params" in params
        assert "v2_params" in params


class TestPoPFusionFusionWeight:
    """Test learned fusion weight behavior."""

    def test_default_fusion_weight_is_half(self):
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        assert abs(fusion._get_fusion_weight() - 0.5) < 0.01

    def test_distributional_weight_is_one(self):
        fusion = PoPFusion(vocab_size=50257, model_type="distributional")
        assert fusion._get_fusion_weight() == 1.0

    def test_contextual_weight_is_zero(self):
        fusion = PoPFusion(vocab_size=50257, model_type="contextual")
        assert fusion._get_fusion_weight() == 0.0

    def test_is_trained_reflects_specialists(self):
        fusion = PoPFusion(vocab_size=50257, model_type="fusion")
        assert not fusion.is_trained
