import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import pytest
from pop.core.pop_v2 import LLMErrorPredictorV2, PoPLayerLLMV2, extract_features_vectorized


class TestLLMErrorPredictorV2:
    def test_init(self):
        model = LLMErrorPredictorV2(vocab_size=50257)
        assert sum(p.numel() for p in model.parameters()) > 100000
    
    def test_forward(self):
        model = LLMErrorPredictorV2(vocab_size=50257)
        model.eval()
        logits = torch.randn(2, 50257)
        probs = torch.softmax(logits, dim=-1)
        out = model(logits, probs)
        assert 'error_magnitude' in out
        assert 'confidence' in out
        assert 'error_direction' in out
    
    def test_extract_features_24(self):
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        features = extract_features_vectorized(logits, probs)
        assert features.shape == (1, 24)


class TestPoPLayerLLMV2:
    def test_init(self):
        pop = PoPLayerLLMV2(vocab_size=50257, device='cpu')
        assert not pop.is_trained
    
    def test_predict(self):
        pop = PoPLayerLLMV2(vocab_size=50257, device='cpu')
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        result = pop.predict(logits, probs)
        assert 'error_magnitude' in result
        assert 'confidence' in result
