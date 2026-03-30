import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import pytest
from pop.core.pop_layer_llm import LLMErrorPredictor, PoPLayerLLM


class TestLLMErrorPredictor:
    def test_init(self):
        model = LLMErrorPredictor(vocab_size=50257)
        assert model.vocab_size == 50257
    
    def test_forward_shape(self):
        model = LLMErrorPredictor(vocab_size=50257)
        logits = torch.randn(2, 50257)
        probs = torch.softmax(logits, dim=-1)
        out = model(logits, probs)
        assert 'error_magnitude' in out
        assert 'confidence' in out
        assert 'error_direction' in out
        assert out['error_magnitude'].shape == (2,)
    
    def test_extract_features_shape(self):
        model = LLMErrorPredictor(vocab_size=50257)
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        features = model.extract_features(logits, probs)
        assert features.shape == (1, 16)
    
    def test_output_ranges(self):
        model = LLMErrorPredictor(vocab_size=50257)
        model.eval()
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        out = model(logits, probs)
        assert 0 <= out['error_magnitude'].item() <= 1  # sigmoid
        assert 0 <= out['confidence'].item() <= 1  # sigmoid
        assert -1 <= out['error_direction'].item() <= 1  # tanh


class TestPoPLayerLLM:
    def test_init(self):
        pop = PoPLayerLLM(vocab_size=50257)
        assert pop.vocab_size == 50257
        assert not pop.is_trained
    
    def test_predict(self):
        pop = PoPLayerLLM(vocab_size=50257)
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        result = pop.predict(logits, probs)
        assert 'error_magnitude' in result
        assert 'confidence' in result
        assert 'should_correct' in result
    
    def test_train_step(self):
        pop = PoPLayerLLM(vocab_size=50257)
        logits = torch.randn(1, 50257)
        probs = torch.softmax(logits, dim=-1)
        loss = pop.train_step(logits, probs, 1.0, 0.8, 0.5)
        assert 'loss' in loss
        assert loss['loss'] > 0
        assert pop.is_trained
