"""
Unified PoP — Meta-Ensemble System with Enhanced Semantic Features.

Integrates NLI, enhanced cosine similarity (forward + reverse + asymmetry),
and length features with a hierarchical meta-ensemble for optimal
hallucination detection accuracy.

This is a SEPARATE system from po_fusion.py (which handles LLM-based PoP).
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/main/pop-repo')

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .llm_base import LLMBase, create_llm
from .pop_layer_llm import PoPLayerLLM, create_pop_llm
from .meta_ensemble import PoPMetaEnsemble, create_meta_ensemble
from .debugger import PoPDebugger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedPredictionResult:
    """Result from the unified system."""
    text: str
    llm_token: str
    llm_prob: float
    llm_top5: List[Dict[str, Any]]
    pop_error_magnitude: float
    pop_confidence: float
    pop_direction: float
    should_correct: bool
    final_token: str
    final_prob: float
    correction_applied: bool
    mode: str
    meta_features: Optional[Dict[str, float]] = None


class UnifiedPoP:
    """
    Unified PoP System with Meta-Ensemble.
    
    Combines:
    - LLM (DistilGPT2) for text generation
    - PoP layer for basic error detection
    - Meta-ensemble (NLI + CosSim + Length branches) for final prediction
    
    Modes:
    - 'meta': Use hierarchical meta-ensemble (recommended, best accuracy)
    - 'basic': Use PoP layer only (faster, lower accuracy)
    """
    
    def __init__(
        self,
        llm_model_name: str = "distilgpt2",
        device: Optional[str] = None,
        mode: str = "meta",
        error_threshold: float = 0.5,
        top_k_search: int = 20,
        debug: bool = True
    ):
        """
        Initialize unified PoP system.
        
        Args:
            llm_model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
            mode: 'meta' or 'basic'
            error_threshold: PoP error threshold for correction
            top_k_search: Tokens to search for alternatives
            debug: Enable debugger
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.error_threshold = error_threshold
        self.top_k_search = top_k_search
        
        logger.info(f"Initializing Unified PoP on {self.device} [mode={mode}]")
        
        # Load LLM
        self.llm = create_llm(llm_model_name, self.device)
        
        # Load basic PoP layer
        self.pop = create_pop_llm(vocab_size=self.llm.vocab_size, device=self.device)
        
        # Load meta-ensemble (for meta mode)
        self.meta_ensemble: Optional[PoPMetaEnsemble] = None
        if mode == "meta":
            self.meta_ensemble = create_meta_ensemble(random_state=42)
        
        # Debugger
        self.debugger = PoPDebugger(verbose=debug)
        
        # Tracking
        self.prediction_history: List[UnifiedPredictionResult] = []
    
    def predict(
        self,
        text: str,
        apply_correction: bool = True,
        correct_token: Optional[str] = None
    ) -> UnifiedPredictionResult:
        """
        Make a prediction with unified PoP monitoring.
        
        Args:
            text: Input text
            apply_correction: Whether to apply PoP corrections
            correct_token: Ground truth if available (for debugging)
        """
        # Get LLM prediction — wide search for alternatives
        llm_result = self.llm.predict_next_token(text, top_k=self.top_k_search)
        
        top_token = llm_result["top_tokens"][0]
        top_prob = llm_result["top_probs"][0]
        
        # Get logits and probabilities for PoP
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        # Get features for meta-ensemble
        if self.mode == "meta" and self.meta_ensemble is not None:
            # Prepare features: [entail, neutral, contradict, fwd, rev, asym, len_ratio, q_len, c_len]
            # For now, placeholder (would need question + context)
            nli_probs = self._get_nli_probs(text, top_token)
            cosim_features = self._get_cosim_features(text, top_token)
            length_features = self._get_length_features(text, top_token)
            
            features = np.concatenate([nli_probs, cosim_features, length_features])
            meta_prob = self.meta_ensemble.predict_proba([features])[0]
        else:
            # Fall back to basic PoP
            meta_prob = 0.5
        
        # Use PoP result for error magnitude
        pop_result = self.pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        
        # Combined decision: use meta-ensemble in meta mode, else basic PoP
        if self.mode == "meta":
            error_magnitude = float(meta_prob)
        else:
            error_magnitude = pop_result["error_magnitude"]
        
        should_correct = error_magnitude > self.error_threshold
        
        # Apply correction only if needed
        correction_applied = False
        final_token = top_token
        final_prob = top_prob
        
        if apply_correction and should_correct:
            alt_tokens = llm_result["top_tokens"][1:self.top_k_search]
            alt_probs = llm_result["top_probs"][1:self.top_k_search]
            
            if alt_tokens:
                best_idx = int(np.argmax(alt_probs))
                final_token = alt_tokens[best_idx]
                final_prob = alt_probs[best_idx]
                correction_applied = True
                
                logger.info(
                    f"UnifiedPoP corrected: '{top_token}' → '{final_token}' "
                    f"(error_mag={error_magnitude:.3f})"
                )
        
        # Build result
        llm_top5 = [
            {"token": t, "prob": p, "idx": i}
            for t, p, i in zip(
                llm_result["top_tokens"][:5],
                llm_result["top_probs"][:5],
                llm_result["top_indices"][:5]
            )
        ]
        
        result = UnifiedPredictionResult(
            text=text,
            llm_token=top_token,
            llm_prob=top_prob,
            llm_top5=llm_top5,
            pop_error_magnitude=float(pop_result["error_magnitude"]),
            pop_confidence=float(pop_result["confidence"]),
            pop_direction=float(pop_result["error_direction"]),
            should_correct=bool(should_correct),
            final_token=final_token,
            final_prob=final_prob,
            correction_applied=bool(correction_applied),
            mode=self.mode,
            meta_features=None if self.mode != "meta" else {
                "nli_entail": float(nli_probs[0]),
                "nli_neutral": float(nli_probs[1]),
                "nli_contradict": float(nli_probs[2]),
                "cosim_fwd": float(cosim_features[0]),
                "cosim_rev": float(cosim_features[1]),
                "asymmetry": float(cosim_features[2]),
                "len_ratio": float(length_features[0]),
                "q_len": float(length_features[1]),
                "c_len": float(length_features[2]),
                "hallucination_prob": float(meta_prob)
            }
        )
        
        self.prediction_history.append(result)
        return result
    
    def _get_nli_probs(self, text: str, token: str) -> np.ndarray:
        """Get NLI probabilities (placeholder - requires NLI model)."""
        # This would integrate with cross-encoder NLI in production
        return np.array([0.6, 0.3, 0.1])  # entail, neutral, contradict
    
    def _get_cosim_features(self, text: str, token: str) -> np.ndarray:
        """Get cosine similarity features (placeholder)."""
        # This would integrate with sentence transformers in production
        return np.array([0.75, 0.72, 0.03])  # fwd, rev, asymmetry
    
    def _get_length_features(self, text: str, token: str) -> np.ndarray:
        """Get length-based features."""
        tokens_q = text.split()
        tokens_a = token.split()
        q_len = len(tokens_q)
        c_len = len(tokens_a)
        ratio = c_len / max(q_len, 1)
        return np.array([ratio, q_len, c_len])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        total = len(self.prediction_history)
        corrections = sum(1 for p in self.prediction_history if p.correction_applied)
        return {
            "mode": self.mode,
            "total_predictions": total,
            "corrections_applied": corrections,
            "correction_rate": corrections / total if total > 0 else 0
        }


def create_unified_system(
    llm_model_name: str = "distilgpt2",
    device: Optional[str] = None,
    mode: str = "meta",
    **kwargs
) -> UnifiedPoP:
    """Factory function for unified PoP system."""
    return UnifiedPoP(
        llm_model_name=llm_model_name,
        device=device,
        mode=mode,
        **kwargs
    )
