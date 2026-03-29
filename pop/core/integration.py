"""
Integration module - Combines LLM with PoP layer.
Implements the safety guard: only adjusts if proven better.
Includes debugger for full observability.

Supports three model types via config:
    - "distributional": PoP v1 (simple features, fast)
    - "contextual": PoP v2 (residual blocks, expanded features)
    - "fusion": Both specialists combined (best accuracy)
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .llm_base import LLMBase, create_llm
from .pop_layer_llm import PoPLayerLLM, create_pop_llm
from .pop_v2 import PoPLayerLLMV2, create_pop_v2
from .pop_fusion import PoPFusion, FusionConfig, create_pop_fusion
from .debugger import PoPDebugger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Valid model types
MODEL_TYPES = ("distributional", "contextual", "fusion")


@dataclass
class PredictionResult:
    """Result of a prediction with PoP analysis."""
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
    mode: str  # "passive" (warn only) or "active" (override)
    model_type: str = "distributional"  # which specialist produced the result


def _create_pop_layer(
    model_type: str,
    vocab_size: int,
    device: str,
    fusion_config: Optional[FusionConfig] = None,
) -> Any:
    """
    Factory: create the right PoP specialist or fusion layer.
    
    Args:
        model_type: "distributional", "contextual", or "fusion"
        vocab_size: LLM vocabulary size
        device: torch device string
        fusion_config: Optional FusionConfig (only used when model_type="fusion")
    
    Returns:
        PoPLayerLLM, PoPLayerLLMV2, or PoPFusion instance
    """
    if model_type == "distributional":
        return create_pop_llm(vocab_size=vocab_size, device=device)
    elif model_type == "contextual":
        return create_pop_v2(vocab_size=vocab_size, device=device)
    elif model_type == "fusion":
        config = fusion_config or FusionConfig()
        return create_pop_fusion(vocab_size=vocab_size, device=device, config=config)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {MODEL_TYPES}"
        )


class PoPIntegration:
    """
    Integration of LLM + PoP layer with safety guard.
    
    Supports three model types:
    - "distributional" (default): PoP v1 — simple features, fast inference
    - "contextual": PoP v2 — residual blocks, expanded feature set, better accuracy
    - "fusion": Combines both specialists for best results
    
    Modes:
    - "passive": PoP warns but never overrides LLM (log-only)
    - "active": PoP can override when confident error is high
    
    Safety Rule (active mode):
    IF error_magnitude > threshold → pick best alternative from wider search
    ELSE → trust LLM
    """
    
    def __init__(
        self,
        llm_model_name: str = "distilgpt2",
        device: Optional[str] = None,
        debug: bool = True,
        mode: str = "active",
        error_threshold: float = 0.5,
        top_k_search: int = 20,
        model_type: str = "distributional",
        fusion_config: Optional[FusionConfig] = None,
    ):
        """
        Args:
            llm_model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
            debug: Enable debugger
            mode: "passive" (warn only) or "active" (override)
            error_threshold: PoP error_magnitude above this triggers correction
            top_k_search: How many tokens to search for alternatives
            model_type: "distributional", "contextual", or "fusion"
            fusion_config: Configuration for fusion layer (only used when model_type="fusion")
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Valid options: {MODEL_TYPES}"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.error_threshold = error_threshold
        self.top_k_search = top_k_search
        self.model_type = model_type
        logger.info(
            f"Initializing PoP Integration on {self.device} | "
            f"mode={mode} | model_type={model_type}"
        )
        
        # Load LLM
        self.llm = create_llm(llm_model_name, self.device)
        
        # Create PoP layer via factory
        self.pop = _create_pop_layer(
            model_type=model_type,
            vocab_size=self.llm.vocab_size,
            device=self.device,
            fusion_config=fusion_config,
        )
        
        # Debugger
        self.debugger = PoPDebugger(verbose=debug)
        
        # Tracking
        self.prediction_history: List[PredictionResult] = []
        self.correction_history: List[Dict[str, Any]] = []
    
    def predict(
        self,
        text: str,
        apply_correction: bool = True,
        correct_token: Optional[str] = None
    ) -> PredictionResult:
        """
        Make a prediction with PoP monitoring.
        
        Args:
            text: Input text
            apply_correction: Whether to apply PoP corrections (ignored in passive mode)
            correct_token: Ground truth if available (for debugging)
        """
        # Get LLM prediction — wide search for alternatives
        llm_result = self.llm.predict_next_token(text, top_k=self.top_k_search)
        
        top_token = llm_result["top_tokens"][0]
        top_prob = llm_result["top_probs"][0]
        
        # Get logits and probabilities for PoP
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        # Run PoP analysis
        pop_result = self.pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        
        # Safety guard: PoP says error is HIGH → correct
        should_correct = pop_result["error_magnitude"] > self.error_threshold
        
        # Apply correction only in active mode
        correction_applied = False
        final_token = top_token
        final_prob = top_prob
        
        if self.mode == "active" and apply_correction and should_correct:
            # Smart correction: search wider for the best alternative
            # Strategy: pick the token with highest probability that isn't the top token
            # (since PoP says top token is likely wrong)
            alt_tokens = llm_result["top_tokens"][1:self.top_k_search]
            alt_probs = llm_result["top_probs"][1:self.top_k_search]
            
            if alt_tokens:
                best_idx = int(np.argmax(alt_probs))
                final_token = alt_tokens[best_idx]
                final_prob = alt_probs[best_idx]
                correction_applied = True
                
                logger.info(f"PoP corrected: '{top_token}' → '{final_token}' (searched {len(alt_tokens)} alts)")
        
        decision = "CORRECTED" if correction_applied else "TRUST_LLM"
        
        # Build top-5 for debugger
        llm_top5 = [
            {"token": t, "prob": p, "idx": i}
            for t, p, i in zip(
                llm_result["top_tokens"][:5],
                llm_result["top_probs"][:5],
                llm_result["top_indices"][:5]
            )
        ]
        
        # Log to debugger
        self.debugger.log_prediction(
            input_text=text,
            llm_token=top_token,
            llm_prob=top_prob,
            llm_top5=llm_top5,
            pop_error_magnitude=pop_result["error_magnitude"],
            pop_confidence=pop_result["confidence"],
            pop_direction=pop_result["error_direction"],
            decision=decision,
            final_token=final_token,
            final_prob=final_prob,
            correct_token=correct_token
        )
        
        result = PredictionResult(
            text=text,
            llm_token=top_token,
            llm_prob=top_prob,
            llm_top5=llm_top5,
            pop_error_magnitude=pop_result["error_magnitude"],
            pop_confidence=pop_result["confidence"],
            pop_direction=pop_result["error_direction"],
            should_correct=should_correct,
            final_token=final_token,
            final_prob=final_prob,
            correction_applied=correction_applied,
            mode=self.mode,
            model_type=self.model_type,
        )
        
        self.prediction_history.append(result)
        return result
    
    def train_on_feedback(
        self,
        text: str,
        correct_token: str
    ) -> Dict[str, Any]:
        """
        Train PoP on feedback. Gets LLM prediction, compares to ground truth.
        
        Args:
            text: Input text
            correct_token: What the correct next token should be
            
        Returns:
            Training result
        """
        # Get LLM outputs
        llm_result = self.llm.predict_next_token(text, top_k=10)
        
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        predicted_token = llm_result["top_tokens"][0]
        pred_prob = llm_result["top_probs"][0]
        
        # Find correct token probability
        correct_prob = 0.0
        if correct_token in llm_result["top_tokens"]:
            idx = llm_result["top_tokens"].index(correct_token)
            correct_prob = llm_result["top_probs"][idx]
        else:
            # Token not in top-10, get full distribution
            full_probs = probs[0].cpu().numpy()
            correct_ids = self.llm.tokenizer.encode(correct_token)
            if correct_ids:
                correct_prob = float(full_probs[correct_ids[0]])
        
        # Compute error labels
        is_wrong = predicted_token.strip().lower() != correct_token.strip().lower()
        error_magnitude = 1.0 if is_wrong else 0.0
        confidence = pred_prob
        error_direction = (pred_prob - correct_prob) if is_wrong else 0.0
        
        # Train PoP layer
        loss = self.pop.train_step(
            logits.unsqueeze(0),
            probs.unsqueeze(0),
            error_magnitude,
            confidence,
            error_direction
        )
        
        # Record if wrong
        if is_wrong:
            self.correction_history.append({
                "text": text,
                "predicted": predicted_token,
                "correct": correct_token,
                "pred_prob": pred_prob,
                "correct_prob": correct_prob,
                "improvement": correct_prob - pred_prob
            })
        
        # Normalize loss output across specialist types
        if isinstance(loss, dict):
            loss_val = loss.get("loss", loss.get("total", 0.0))
        else:
            loss_val = float(loss)
        
        return {
            "predicted_token": predicted_token,
            "correct_token": correct_token,
            "is_wrong": is_wrong,
            "error_magnitude": error_magnitude,
            "confidence": confidence,
            "error_direction": error_direction,
            "loss": loss_val,
            "model_type": self.model_type,
        }
    
    def train_batch(
        self,
        examples: List[Dict[str, str]],
        epochs: int = 3
    ) -> Dict[str, Any]:
        """
        Train PoP on a batch of (prompt, correct_answer) pairs.
        
        Args:
            examples: List of {"prompt": str, "answer": str}
            epochs: Number of training passes
            
        Returns:
            Training summary
        """
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            wrong_count = 0
            
            for ex in examples:
                result = self.train_on_feedback(ex["prompt"], ex["answer"])
                epoch_loss += result["loss"]
                if result["is_wrong"]:
                    wrong_count += 1
            
            avg_loss = epoch_loss / len(examples)
            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "llm_errors": wrong_count,
                "total": len(examples)
            })
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} — "
                f"loss: {avg_loss:.4f} | "
                f"LLM errors: {wrong_count}/{len(examples)}"
            )
        
        return {
            "status": "trained",
            "epochs": epochs,
            "examples": len(examples),
            "model_type": self.model_type,
            "history": history
        }
    
    def analyze_prediction(self, text: str) -> Dict[str, Any]:
        """Analyze a prediction without making corrections."""
        llm_result = self.llm.predict_next_token(text, top_k=10)
        
        logits = self.llm.get_logits(text)
        probs = torch.softmax(logits, dim=-1)
        
        pop_result = self.pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        
        analysis = {
            "text": text,
            "model_type": self.model_type,
            "top_predictions": [
                {"token": t, "prob": p} 
                for t, p in zip(llm_result["top_tokens"][:5], llm_result["top_probs"][:5])
            ],
            "pop_analysis": pop_result,
        }
        
        # Include specialist breakdown for fusion mode
        if self.model_type == "fusion" and "specialists" in pop_result:
            analysis["specialist_breakdown"] = pop_result["specialists"]
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        total = len(self.prediction_history)
        corrections = sum(1 for p in self.prediction_history if p.correction_applied)
        
        # Get pop params safely regardless of specialist type
        pop_params = self.pop.get_params()
        
        return {
            "total_predictions": total,
            "corrections_applied": corrections,
            "correction_rate": corrections / total if total > 0 else 0,
            "model_type": self.model_type,
            "pop_params": pop_params,
            "llm_loaded": self.llm.is_loaded,
            "debugger_metrics": self.debugger.get_metrics()
        }
    
    def print_debug_summary(self):
        """Print full debug summary."""
        self.debugger.print_summary()


def create_pop_system(
    llm_model_name: str = "distilgpt2",
    device: Optional[str] = None,
    debug: bool = True,
    mode: str = "active",
    error_threshold: float = 0.5,
    top_k_search: int = 20,
    model_type: str = "distributional",
    fusion_config: Optional[FusionConfig] = None,
) -> PoPIntegration:
    """
    Factory function to create a complete PoP system.
    
    Args:
        llm_model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        debug: Enable debugger
        mode: "passive" or "active"
        error_threshold: Correction trigger threshold
        top_k_search: Tokens to search for alternatives
        model_type: "distributional", "contextual", or "fusion"
        fusion_config: Config for fusion mode
    
    Returns:
        Configured PoPIntegration instance
    
    Backward compatibility:
        Calling with only the original parameters works identically:
            create_pop_system("distilgpt2", device="cpu", mode="active")
        New model_type defaults to "distributional" (same as v1 behavior).
    """
    return PoPIntegration(
        llm_model_name=llm_model_name,
        device=device,
        debug=debug,
        mode=mode,
        error_threshold=error_threshold,
        top_k_search=top_k_search,
        model_type=model_type,
        fusion_config=fusion_config,
    )
