"""
PoP Fusion Layer — Combines distributional (v1) and contextual (v2) specialists.

The Fusion layer runs both PoP specialists in parallel and combines their
predictions via learned weighted averaging. When only one specialist is
available (config-driven), it acts as a pass-through.

Usage:
    fusion = PoPFusion(vocab_size=50257)
    result = fusion.predict(logits, probs)
    # result["error_magnitude"], result["confidence"], etc.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .pop_layer_llm import PoPLayerLLM
from .pop_v2 import PoPLayerLLMV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPES = ("distributional", "contextual", "fusion")


@dataclass
class FusionConfig:
    """Configuration for PoP Fusion layer."""
    model_type: str = "fusion"  # "distributional", "contextual", or "fusion"
    v1_hidden_dim: int = 256
    v2_hidden_dim: int = 512


class PoPFusion:
    """
    Fusion layer that combines distributional (v1) and contextual (v2)
    PoP specialists into a single prediction.

    Supports three config-driven modes:
    - "distributional": v1 only
    - "contextual": v2 only
    - "fusion": both, combined via learned weights

    When both specialists are active, their outputs are combined with
    a learnable sigmoid-gated weighting (alpha for v1, 1-alpha for v2).
    """

    def __init__(
        self,
        vocab_size: int,
        device: Optional[str] = None,
        model_type: str = "fusion",
        v1_hidden_dim: int = 256,
        v2_hidden_dim: int = 512,
    ):
        """
        Args:
            vocab_size: LLM vocabulary size
            device: 'cpu' or 'cuda'
            model_type: "distributional" (v1 only), "contextual" (v2 only), or "fusion" (both)
            v1_hidden_dim: Hidden dim for v1 specialist
            v2_hidden_dim: Hidden dim for v2 specialist
        """
        assert model_type in ("distributional", "contextual", "fusion"), (
            f"model_type must be 'distributional', 'contextual', or 'fusion', got '{model_type}'"
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.model_type = model_type

        # Create specialists based on config
        self.v1: Optional[PoPLayerLLM] = None
        self.v2: Optional[PoPLayerLLMV2] = None

        if model_type in ("distributional", "fusion"):
            self.v1 = PoPLayerLLM(vocab_size=vocab_size, device=self.device, hidden_dim=v1_hidden_dim)
            logger.info("Fusion: v1 (distributional) specialist created")

        if model_type in ("contextual", "fusion"):
            self.v2 = PoPLayerLLMV2(vocab_size=vocab_size, device=self.device, hidden_dim=v2_hidden_dim)
            logger.info("Fusion: v2 (contextual) specialist created")

        # Learned fusion weight (logit-space, will be sigmoided)
        # alpha → weight for v1, (1-alpha) → weight for v2
        if model_type == "fusion":
            self._fusion_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        else:
            self._fusion_logit = None

    @property
    def is_trained(self) -> bool:
        if self.model_type == "distributional":
            return self.v1.is_trained
        elif self.model_type == "contextual":
            return self.v2.is_trained
        else:  # fusion
            return self.v1.is_trained and self.v2.is_trained

    def _get_fusion_weight(self) -> float:
        """Get the learned v1 weight (alpha). Returns 1.0 / 0.0 in single-specialist mode."""
        if self.model_type == "distributional":
            return 1.0
        elif self.model_type == "contextual":
            return 0.0
        else:
            return torch.sigmoid(self._fusion_logit).item()

    def predict(self, logits: torch.Tensor, probs: torch.Tensor) -> Dict[str, Any]:
        """
        Predict error likelihood using configured specialists.

        Args:
            logits: LLM logits, shape (V,) or (B, V)
            probs: LLM probabilities, shape (V,) or (B, V)

        Returns:
            Dict with combined predictions:
                - error_magnitude, confidence, error_direction
                - should_correct, llm_likely_wrong, llm_overconfident, llm_underconfident
                - model_type, v1_weight
        """
        alpha = self._get_fusion_weight()

        if self.model_type == "distributional":
            result = self.v1.predict(logits, probs)
            return self._wrap_result(result, alpha)

        elif self.model_type == "contextual":
            result = self.v2.predict(logits, probs)
            return self._wrap_result(result, alpha)

        else:  # fusion
            r1 = self.v1.predict(logits, probs)
            r2 = self.v2.predict(logits, probs)

            combined = {
                "error_magnitude": float(alpha * r1["error_magnitude"] + (1 - alpha) * r2["error_magnitude"]),
                "confidence": float(alpha * r1["confidence"] + (1 - alpha) * r2["confidence"]),
                "error_direction": float(alpha * r1["error_direction"] + (1 - alpha) * r2["error_direction"]),
            }

            # Decision logic: correct if either specialist flags a problem
            combined["should_correct"] = r1["should_correct"] or r2["should_correct"]
            combined["llm_likely_wrong"] = r1["llm_likely_wrong"] or r2["llm_likely_wrong"]
            combined["llm_overconfident"] = r1["llm_overconfident"] or r2["llm_overconfident"]
            combined["llm_underconfident"] = r1["llm_underconfident"] or r2["llm_underconfident"]

            return self._wrap_result(combined, alpha)

    def _wrap_result(self, result: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """Add fusion metadata to result dict."""
        result["model_type"] = self.model_type
        result["v1_weight"] = alpha
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "device": self.device,
            "v1_weight": self._get_fusion_weight(),
            "v1_params": self.v1.get_params() if self.v1 else None,
            "v2_params": self.v2.get_params() if self.v2 else None,
        }


def create_pop_fusion(
    vocab_size: int,
    device: Optional[str] = None,
    model_type: str = "fusion",
) -> PoPFusion:
    """Factory function — creates a PoP Fusion layer."""
    return PoPFusion(vocab_size=vocab_size, device=device, model_type=model_type)
