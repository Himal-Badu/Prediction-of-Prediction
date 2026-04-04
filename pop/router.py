"""
Tier detection for PoP v3 three-tier confidence scoring.

Tiers:
  - FULL:    logits available (raw model output probabilities)
  - LITE:    top-k token probabilities available
  - MINIMAL: text only (no probability data)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class TierLevel(str, Enum):
    """Confidence scoring tier levels."""
    FULL = "full"
    LITE = "lite"
    MINIMAL = "minimal"


class TierDetector:
    """Auto-detect available tier from input payload."""

    def detect(self, input_data: Dict[str, Any]) -> TierLevel:
        """Detect the highest available tier from the input data.

        Priority: FULL > LITE > MINIMAL

        Args:
            input_data: Request payload dict.

        Returns:
            The detected TierLevel.
        """
        if input_data.get("logits") is not None:
            return TierLevel.FULL

        if input_data.get("token_probs") is not None:
            return TierLevel.LITE

        return TierLevel.MINIMAL

    @staticmethod
    def tier_capabilities(tier: TierLevel) -> Dict[str, Any]:
        """Return feature descriptions for a given tier."""
        capabilities = {
            TierLevel.FULL: {
                "name": "Full",
                "description": "Raw logits available — highest confidence scoring accuracy.",
                "features": [
                    "logit_entropy",
                    "top_k_margin",
                    "probability_calibration",
                    "token_level_surprise",
                    "attention_weight_analysis",
                ],
                "accuracy": "highest",
            },
            TierLevel.LITE: {
                "name": "Lite",
                "description": "Top-k token probabilities — good accuracy with reduced data.",
                "features": [
                    "top_k_margin",
                    "probability_calibration",
                    "token_level_surprise",
                ],
                "accuracy": "high",
            },
            TierLevel.MINIMAL: {
                "name": "Minimal",
                "description": "Text-only — heuristic and semantic analysis.",
                "features": [
                    "semantic_coherence",
                    "hedging_detection",
                    "factual_consistency_heuristic",
                ],
                "accuracy": "baseline",
            },
        }
        return capabilities[tier]
