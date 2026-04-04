"""Tests for the tier detection router."""
from __future__ import annotations

import pytest

from pop.router import TierDetector, TierLevel


class TestTierDetector:
    def setup_method(self):
        self.detector = TierDetector()

    def test_detect_full_with_logits(self):
        result = self.detector.detect({"text": "hello", "logits": [[0.1, 0.9]]})
        assert result == TierLevel.FULL

    def test_detect_lite_with_token_probs(self):
        result = self.detector.detect({
            "text": "hello",
            "token_probs": [{"the": 0.8, "a": 0.2}],
        })
        assert result == TierLevel.LITE

    def test_detect_minimal_text_only(self):
        result = self.detector.detect({"text": "just some text"})
        assert result == TierLevel.MINIMAL

    def test_logits_takes_priority_over_token_probs(self):
        result = self.detector.detect({
            "text": "hello",
            "logits": [[0.1, 0.9]],
            "token_probs": [{"the": 0.8}],
        })
        assert result == TierLevel.FULL

    def test_empty_dict_is_minimal(self):
        result = self.detector.detect({})
        assert result == TierLevel.MINIMAL

    def test_none_logits_falls_through(self):
        result = self.detector.detect({"text": "hi", "logits": None})
        assert result == TierLevel.MINIMAL

    def test_tier_capabilities_full(self):
        caps = TierDetector.tier_capabilities(TierLevel.FULL)
        assert caps["name"] == "Full"
        assert "logit_entropy" in caps["features"]
        assert caps["accuracy"] == "highest"

    def test_tier_capabilities_lite(self):
        caps = TierDetector.tier_capabilities(TierLevel.LITE)
        assert caps["name"] == "Lite"
        assert "top_k_margin" in caps["features"]

    def test_tier_capabilities_minimal(self):
        caps = TierDetector.tier_capabilities(TierLevel.MINIMAL)
        assert caps["name"] == "Minimal"
        assert "hedging_detection" in caps["features"]
