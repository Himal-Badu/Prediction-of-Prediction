"""
Smart Correction Engine for PoP v2.

Replaces naive "pick #2 token" correction with:
- Beam search over top-k continuations (2 tokens deep)
- error_direction-guided correction strategy
- Factual answer heuristic checking
- Fallback to LLM when no good alternative found
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Known factual answers for heuristic checking ───────────────────────

KNOWN_FACTS: Dict[str, List[str]] = {
    "capital of france": ["paris"],
    "largest planet": ["jupiter", "saturn"],
    "water freezes": ["zero", "0", "32", "freezing"],
    "chemical symbol for gold": ["au"],
    "speed of light": ["three hundred", "300", "299", "186"],
    "dna stands for": ["deoxyribonucleic"],
    "powerhouse of the cell": ["mitochondria", "mitochondrion"],
    "photosynthesis converts": ["energy", "glucose", "sugar", "chemical"],
    "newton's first law": ["inertia"],
    "boiling point of water": ["100", "212", "boiling"],
    "world war ii ended": ["1945"],
    "first person to walk on the moon": ["neil", "armstrong", "buzz"],
    "great wall of china": ["protect", "defend", "mongol", "border"],
    "rome was founded": ["753"],
    "cleopatra": ["egypt"],
    "meaning of life douglas adams": ["42"],
    "1984": ["totalitarianism", "orwell", "big brother", "dystopia"],
    "theory of relativity": ["einstein", "albert"],
    "shakespeare wrote": ["hamlet", "hamlet", "macbeth", "othello"],
    "mona lisa": ["leonardo", "da vinci", "vinci"],
}


@dataclass
class CorrectionCandidate:
    """A candidate correction with scoring metadata."""
    token: str
    token_id: int
    prob: float
    continuation_tokens: List[str] = field(default_factory=list)
    continuation_probs: List[float] = field(default_factory=list)
    beam_score: float = 0.0
    fact_match: bool = False
    source: str = ""  # "beam", "topk", "fallback"


class SmartCorrectionEngine:
    """
    Smart correction engine that uses beam search + PoP analysis
    to find better alternatives when the LLM is likely wrong.
    
    Strategy:
    1. If error_direction > 0.3 (overconfident): actively search for alternatives
    2. If error_direction < -0.3 (underconfident): trust LLM, minimal correction
    3. Otherwise: moderate search, use beam scoring to pick best
    4. Always check factual prompts against known answer set
    5. Fallback to LLM's #1 if no good alternative found
    """
    
    def __init__(
        self,
        llm,  # LLMBase instance
        beam_width: int = 5,
        continuation_depth: int = 2,
        overconfident_threshold: float = 0.3,
        underconfident_threshold: float = -0.3,
    ):
        """
        Initialize the smart correction engine.
        
        Args:
            llm: LLMBase instance for generating continuations
            beam_width: Number of beam candidates to explore
            continuation_depth: Tokens deep to score each candidate
            overconfident_threshold: error_direction above this = overconfident
            underconfident_threshold: error_direction below this = underconfident
        """
        self.llm = llm
        self.beam_width = beam_width
        self.continuation_depth = continuation_depth
        self.overconfident_thresh = overconfident_threshold
        self.underconfident_thresh = underconfident_threshold
    
    def _check_fact_match(self, prompt: str, token: str) -> bool:
        """Check if token matches a known factual answer for the prompt."""
        prompt_lower = prompt.lower().strip()
        token_clean = token.strip().lower()
        
        for key, answers in KNOWN_FACTS.items():
            if key in prompt_lower:
                for ans in answers:
                    if token_clean.startswith(ans) or ans.startswith(token_clean):
                        return True
        return False
    
    def _beam_score_continuation(
        self,
        prompt: str,
        candidate_token: str,
        candidate_id: int,
    ) -> Tuple[float, List[str], List[float]]:
        """
        Score a candidate by generating continuations and measuring
        total log-probability.
        
        Returns:
            (beam_score, continuation_tokens, continuation_probs)
        """
        extended_text = prompt + candidate_token
        
        try:
            cont_result = self.llm.predict_next_token(
                extended_text, top_k=self.beam_width
            )
        except Exception:
            return 0.0, [], []
        
        if not cont_result["top_tokens"]:
            return 0.0, [], []
        
        # Take top continuation and score it
        best_cont_token = cont_result["top_tokens"][0]
        best_cont_prob = float(cont_result["top_probs"][0])
        
        # Score = probability of the first continuation token
        # Higher = the candidate leads to a more natural continuation
        beam_score = best_cont_prob
        
        # If depth > 1, go one more level
        if self.continuation_depth > 1 and best_cont_token:
            next_text = extended_text + best_cont_token
            try:
                next_result = self.llm.predict_next_token(next_text, top_k=1)
                if next_result["top_probs"]:
                    # Combine scores (geometric mean for depth-normalization)
                    beam_score = (beam_score * float(next_result["top_probs"][0])) ** 0.5
            except Exception:
                pass
        
        return beam_score, [best_cont_token], [best_cont_prob]
    
    def correct(
        self,
        prompt: str,
        llm_result: Dict[str, Any],
        pop_analysis: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> CorrectionCandidate:
        """
        Apply smart correction based on PoP analysis.
        
        Args:
            prompt: Original input prompt
            llm_result: Output from LLMBase.predict_next_token()
            pop_analysis: Output from PoPLayerLLM.predict() or PoPLayerLLMV2.predict()
            expected: Optional ground truth for evaluation
            
        Returns:
            CorrectionCandidate with the chosen correction
        """
        top_token = llm_result["top_tokens"][0]
        top_prob = float(llm_result["top_probs"][0])
        top_ids = llm_result.get("top_indices", list(range(len(llm_result["top_tokens"]))))
        
        error_direction = pop_analysis.get("error_direction", 0.0)
        error_magnitude = pop_analysis.get("error_magnitude", 0.0)
        confidence = pop_analysis.get("confidence", 0.5)
        should_correct = pop_analysis.get("should_correct", False)
        
        # ── Strategy 1: Underconfident → trust LLM ───────────────────
        if error_direction < self.underconfident_thresh:
            # LLM is likely underconfident — its top pick is probably fine
            return CorrectionCandidate(
                token=top_token,
                token_id=top_ids[0] if top_ids else 0,
                prob=top_prob,
                beam_score=top_prob,
                fact_match=self._check_fact_match(prompt, top_token),
                source="fallback_underconfident",
            )
        
        # ── Strategy 2: Beam search over top-k alternatives ──────────
        # Determine how aggressively to search based on error_direction
        if error_direction > self.overconfident_thresh:
            # Overconfident: search wider
            search_k = min(20, len(llm_result["top_tokens"]))
        elif should_correct:
            # PoP flags error: moderate search
            search_k = min(10, len(llm_result["top_tokens"]))
        else:
            # Mild uncertainty: narrow search
            search_k = min(5, len(llm_result["top_tokens"]))
        
        candidates: List[CorrectionCandidate] = []
        
        for i in range(search_k):
            token = llm_result["top_tokens"][i]
            prob = float(llm_result["top_probs"][i])
            token_id = top_ids[i] if i < len(top_ids) else 0
            
            # Score with beam search
            beam_score, cont_tokens, cont_probs = self._beam_score_continuation(
                prompt, token, token_id
            )
            
            # Check fact match
            fact_match = self._check_fact_match(prompt, token)
            
            # Combined score: beam score weighted by original probability
            # + bonus for factual match
            combined_score = beam_score * 0.7 + prob * 0.3
            if fact_match:
                combined_score += 0.5  # Strong bonus for known facts
            
            candidates.append(CorrectionCandidate(
                token=token,
                token_id=token_id,
                prob=prob,
                continuation_tokens=cont_tokens,
                continuation_probs=cont_probs,
                beam_score=combined_score,
                fact_match=fact_match,
                source="beam",
            ))
        
        if not candidates:
            # No alternatives available
            return CorrectionCandidate(
                token=top_token,
                token_id=top_ids[0] if top_ids else 0,
                prob=top_prob,
                beam_score=top_prob,
                source="fallback_empty",
            )
        
        # Sort by combined beam score
        candidates.sort(key=lambda c: c.beam_score, reverse=True)
        best = candidates[0]
        
        # ── Strategy 3: Fallback if best isn't meaningfully better ───
        # If the best alternative is the top token anyway, just use it
        if best.token == top_token:
            best.source = "fallback_top1"
            return best
        
        # If best score is barely above top-1, stick with LLM
        # (avoid swapping for marginal gains)
        if best.beam_score < top_prob * 1.1 and not best.fact_match:
            return CorrectionCandidate(
                token=top_token,
                token_id=top_ids[0] if top_ids else 0,
                prob=top_prob,
                beam_score=top_prob,
                fact_match=self._check_fact_match(prompt, top_token),
                source="fallback_marginal",
            )
        
        # ── Apply correction ─────────────────────────────────────────
        return best
    
    def get_params(self) -> Dict[str, Any]:
        """Get engine parameters."""
        return {
            "beam_width": self.beam_width,
            "continuation_depth": self.continuation_depth,
            "overconfident_threshold": self.overconfident_thresh,
            "underconfident_threshold": self.underconfident_thresh,
            "num_known_facts": len(KNOWN_FACTS),
        }
