import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import numpy as np
import pytest
from pop.core.training_data import get_llm_correct_prompts, get_llm_wrong_prompts, get_balanced_facts


class TestTrainingData:
    def test_correct_prompts_format(self):
        prompts = get_llm_correct_prompts()
        assert len(prompts) > 0
        for p in prompts:
            assert 'prompt' in p
            assert 'answer' in p
    
    def test_wrong_prompts_format(self):
        prompts = get_llm_wrong_prompts()
        assert len(prompts) > 0
        for p in prompts:
            assert 'prompt' in p
            assert 'answer' in p
    
    def test_balanced_facts(self):
        facts = get_balanced_facts()
        assert len(facts) > 0
        # Should be balanced
        prompts_set = set(f['prompt'] for f in facts)
        assert len(prompts_set) == len(facts)  # no duplicates
    
    def test_training_data_npy(self):
        data = np.load('/root/.openclaw/workspace-main/pop-repo/training_data.npy', allow_pickle=True).item()
        assert 'features' in data
        assert 'labels' in data
        assert data['features'].shape[1] == 16
        assert len(data['labels']) == data['features'].shape[0]
        # Balanced
        assert abs(sum(data['labels']) - len(data['labels'])/2) < 5
