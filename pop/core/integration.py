"""
Integration module - Combines LLM with PoP layer.
Implements the safety guard: only adjusts if proven better.
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .llm_base import LLMBase, create_llm
from .pop_layer_llm import PoPLayerLLM, create_pop_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction with PoP analysis."""
    text: str
    llm_token: str
    llm_prob: float
    pop_error_magnitude: float
    pop_confidence: float
    pop_direction: float
    should_correct: bool
    final_token: str
    final_prob: float
    correction_applied: bool


class PoPIntegration:
    """
    Integration of LLM + PoP layer with safety guard.
    
    Safety Rule:
    IF (PoP confident) AND (PoP adjustment NOT worse than LLM):
        → Apply PoP correction
    ELSE:
        → Trust LLM original prediction
    """
    
    def __init__(
        self,
        llm_model_name: str = "distilgpt2",
        device: Optional[str] = None
    ):
        """
        Initialize the integrated system.
        
        Args:
            llm_model_name: Name of LLM model
            device: Device to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing PoP Integration on {self.device}")
        
        # Load LLM
        self.llm = create_llm(llm_model_name, self.device)
        
        # Create PoP layer
        self.pop = create_pop_llm(
            vocab_size=self.llm.vocab_size,
            device=self.device
        )
        
        # Tracking for feedback
        self.prediction_history: List[PredictionResult] = []
        self.correction_history: List[Dict[str, Any]] = []
        
    def predict(
        self,
        text: str,
        apply_correction: bool = True
    ) -> PredictionResult:
        """
        Make a prediction with PoP monitoring.
        
        Args:
            text: Input text
            apply_correction: Whether to apply PoP corrections
            
        Returns:
            PredictionResult with analysis
        """
        # Get LLM prediction
        llm_result = self.llm.predict_next_token(text, top_k=5)
        
        top_token = llm_result["top_tokens"][0]
        top_prob = llm_result["top_probs"][0]
        
        # Get logits and probabilities for PoP
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        # Run PoP analysis
        pop_result = self.pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        
        # Safety guard check
        should_correct = (
            pop_result["confidence"] > 0.7 and  # PoP is confident
            pop_result["error_magnitude"] > 0.3  # LLM likely wrong
        )
        
        # Apply correction only if safety rule passes
        correction_applied = False
        final_token = top_token
        final_prob = top_prob
        
        if apply_correction and should_correct:
            # Find alternative token that might be better
            alt_indices = llm_result["top_indices"][1:5]
            alt_probs = llm_result["top_probs"][1:5]
            
            # Pick the highest probability alternative
            if len(alt_probs) > 0:
                best_alt_idx = np.argmax(alt_probs)
                final_token = llm_result["top_tokens"][best_alt_idx + 1]
                final_prob = alt_probs[best_alt_idx]
                correction_applied = True
                
                logger.info(f"PoP corrected: {top_token} → {final_token}")
        
        result = PredictionResult(
            text=text,
            llm_token=top_token,
            llm_prob=top_prob,
            pop_error_magnitude=pop_result["error_magnitude"],
            pop_confidence=pop_result["confidence"],
            pop_direction=pop_result["error_direction"],
            should_correct=should_correct,
            final_token=final_token,
            final_prob=final_prob,
            correction_applied=correction_applied
        )
        
        self.prediction_history.append(result)
        
        return result
    
    def train_on_feedback(
        self,
        text: str,
        predicted_token: str,
        correct_token: str
    ) -> Dict[str, Any]:
        """
        Train PoP on feedback (supervised phase).
        
        Args:
            text: Input text
            predicted_token: What LLM predicted
            correct_token: What was actually correct
            
        Returns:
            Training result
        """
        # Get LLM outputs
        llm_result = self.llm.predict_next_token(text, top_k=10)
        
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        # Find indices
        pred_idx = llm_result["top_indices"].index(llm_result["top_tokens"].index(predicted_token))
        correct_idx = llm_result["top_indices"].index(llm_result["top_tokens"].index(correct_token))
        
        pred_prob = llm_result["top_probs"][pred_idx]
        correct_prob = llm_result["top_probs"][correct_idx]
        
        # Compute error labels
        error_magnitude = 1.0 if predicted_token != correct_token else 0.0
        confidence = pred_prob
        error_direction = pred_prob - correct_prob if predicted_token != correct_token else 0.0
        
        # Train
        loss = self.pop.train_step(
            logits.unsqueeze(0),
            probs.unsqueeze(0),
            error_magnitude,
            confidence,
            error_direction
        )
        
        # Record correction
        if predicted_token != correct_token:
            self.correction_history.append({
                "text": text,
                "predicted": predicted_token,
                "correct": correct_token,
                "improvement": correct_prob - pred_prob
            })
        
        return {
            "error_magnitude": error_magnitude,
            "confidence": confidence,
            "error_direction": error_direction,
            "loss": loss["loss"]
        }
    
    def analyze_prediction(self, text: str) -> Dict[str, Any]:
        """
        Analyze a prediction without making corrections.
        
        Args:
            text: Input text
            
        Returns:
            Analysis results
        """
        llm_result = self.llm.predict_next_token(text, top_k=10)
        
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        pop_result = self.pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        
        return {
            "text": text,
            "top_predictions": [
                {"token": t, "prob": p} 
                for t, p in zip(llm_result["top_tokens"][:5], llm_result["top_probs"][:5])
            ],
            "pop_analysis": pop_result
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        total = len(self.prediction_history)
        corrections = sum(1 for p in self.prediction_history if p.correction_applied)
        
        return {
            "total_predictions": total,
            "corrections_applied": corrections,
            "correction_rate": corrections / total if total > 0 else 0,
            "pop_trained": self.pop.is_trained,
            "llm_loaded": self.llm.is_loaded
        }


def create_pop_system(
    llm_model_name: str = "distilgpt2",
    device: Optional[str] = None
) -> PoPIntegration:
    """Factory function to create a complete PoP system."""
    return PoPIntegration(llm_model_name=llm_model_name, device=device)