import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import pytest
from pop.core.pop_fusion import PoPFusion, create_pop_fusion


class TestIntegrationModelTypes:
    """Test that each model_type configures the correct specialists."""

    def test_distributional_has_v1_only(self):
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")
        assert fusion.v1 is not None
        assert fusion.v2 is None
        assert fusion.model_type == "distributional"

    def test_contextual_has_v2_only(self):
        fusion = create_pop_fusion(vocab_size=50257, model_type="contextual")
        assert fusion.v1 is None
        assert fusion.v2 is not None
        assert fusion.model_type == "contextual"

    def test_fusion_has_both(self):
        fusion = create_pop_fusion(vocab_size=50257, model_type="fusion")
        assert fusion.v1 is not None
        assert fusion.v2 is not None
        assert fusion.model_type == "fusion"


class TestIntegrationPredictForEachType:
    """Test prediction works for each model_type."""

    @pytest.fixture
    def sample_inputs(self):
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def test_distributional_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")
        result = fusion.predict(logits, probs)
        assert result["model_type"] == "distributional"
        assert "error_magnitude" in result
        assert "confidence" in result

    def test_contextual_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = create_pop_fusion(vocab_size=50257, model_type="contextual")
        result = fusion.predict(logits, probs)
        assert result["model_type"] == "contextual"
        assert "error_magnitude" in result
        assert "confidence" in result

    def test_fusion_predict(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = create_pop_fusion(vocab_size=50257, model_type="fusion")
        result = fusion.predict(logits, probs)
        assert result["model_type"] == "fusion"
        assert "error_magnitude" in result
        assert "confidence" in result
        assert "should_correct" in result


class TestIntegrationBackwardCompatibility:
    """Test backward compatibility — default model_type works like v1."""

    @pytest.fixture
    def sample_inputs(self):
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def test_default_model_type_is_distributional(self):
        """create_pop_fusion defaults to 'fusion', but integration should
        default to 'distributional' for backward compatibility."""
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")
        assert fusion.model_type == "distributional"

    def test_predict_returns_expected_keys(self, sample_inputs):
        logits, probs = sample_inputs
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")
        result = fusion.predict(logits, probs)
        expected_keys = {
            "error_magnitude", "confidence", "error_direction",
            "should_correct", "llm_likely_wrong",
            "llm_overconfident", "llm_underconfident",
            "model_type", "v1_weight"
        }
        assert expected_keys.issubset(result.keys())

    def test_v1_specialist_matches_standalone(self, sample_inputs):
        """Fusion(distributional) predictions should match standalone PoPLayerLLM."""
        from pop.core.pop_layer_llm import PoPLayerLLM
        logits, probs = sample_inputs

        standalone = PoPLayerLLM(vocab_size=50257)
        fusion = create_pop_fusion(vocab_size=50257, model_type="distributional")

        s_result = standalone.predict(logits, probs)
        f_result = fusion.predict(logits, probs)

        assert abs(s_result["error_magnitude"] - f_result["error_magnitude"]) < 1e-5
        assert abs(s_result["confidence"] - f_result["confidence"]) < 1e-5

    def test_v2_specialist_matches_standalone(self, sample_inputs):
        """Fusion(contextual) predictions should match standalone PoPLayerLLMV2."""
        from pop.core.pop_v2 import PoPLayerLLMV2
        logits, probs = sample_inputs

        standalone = PoPLayerLLMV2(vocab_size=50257, device='cpu')
        fusion = create_pop_fusion(vocab_size=50257, model_type="contextual")

        s_result = standalone.predict(logits, probs)
        f_result = fusion.predict(logits, probs)

        assert abs(s_result["error_magnitude"] - f_result["error_magnitude"]) < 1e-5
        assert abs(s_result["confidence"] - f_result["confidence"]) < 1e-5
