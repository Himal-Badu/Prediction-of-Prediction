"""
PoP Debugger - Real-time observability for the PoP system.
Tracks every prediction, correction, and outcome.
"""
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DebugEntry:
    """Single prediction debug entry."""
    timestamp: float
    input_text: str
    llm_token: str
    llm_prob: float
    llm_top5: List[Dict[str, Any]]
    pop_error_magnitude: float
    pop_confidence: float
    pop_direction: float
    decision: str  # "TRUST_LLM" or "CORRECTED"
    final_token: str
    final_prob: float
    correct_token: Optional[str] = None
    was_correct: Optional[bool] = None


class PoPDebugger:
    """
    Debug/observability layer for PoP predictions.
    Logs everything, computes metrics, prints real-time.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.entries: List[DebugEntry] = []
        self.session_start = time.time()

    def log_prediction(
        self,
        input_text: str,
        llm_token: str,
        llm_prob: float,
        llm_top5: List[Dict[str, Any]],
        pop_error_magnitude: float,
        pop_confidence: float,
        pop_direction: float,
        decision: str,
        final_token: str,
        final_prob: float,
        correct_token: Optional[str] = None
    ) -> DebugEntry:
        """Log a single prediction."""
        was_correct = None
        if correct_token is not None:
            was_correct = (final_token.strip().lower() == correct_token.strip().lower())

        entry = DebugEntry(
            timestamp=time.time(),
            input_text=input_text,
            llm_token=llm_token,
            llm_prob=llm_prob,
            llm_top5=llm_top5,
            pop_error_magnitude=pop_error_magnitude,
            pop_confidence=pop_confidence,
            pop_direction=pop_direction,
            decision=decision,
            final_token=final_token,
            final_prob=final_prob,
            correct_token=correct_token,
            was_correct=was_correct
        )
        self.entries.append(entry)

        if self.verbose:
            self._print_entry(entry)

        return entry

    def _print_entry(self, e: DebugEntry):
        """Pretty-print a debug entry."""
        prompt = e.input_text[:60] + ("..." if len(e.input_text) > 60 else "")
        
        icon = "✓" if e.decision == "TRUST_LLM" else "🔧"
        correct_icon = ""
        if e.was_correct is not None:
            correct_icon = " ✅" if e.was_correct else " ❌"

        print(f"\n{'─' * 60}")
        print(f"  [{icon}] \"{prompt}\"")
        print(f"  LLM  → \"{e.llm_token}\" (prob: {e.llm_prob:.4f})")
        print(f"  PoP  → error: {e.pop_error_magnitude:.3f} | conf: {e.pop_confidence:.3f} | dir: {e.pop_direction:.3f}")
        print(f"  Decision: {e.decision}")
        if e.decision == "CORRECTED":
            print(f"  Result → \"{e.final_token}\" (prob: {e.final_prob:.4f})")
        if e.correct_token:
            print(f"  Answer → \"{e.correct_token}\"{correct_icon}")
        parts = []
        for t in e.llm_top5:
            parts.append(f"'{t['token']}'({t['prob']:.3f})")
        print(f"  Top-5: {', '.join(parts)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Compute session metrics."""
        if not self.entries:
            return {"status": "No predictions logged"}

        total = len(self.entries)
        corrections = sum(1 for e in self.entries if e.decision == "CORRECTED")
        trusts = total - corrections

        # Accuracy where ground truth is available
        labeled = [e for e in self.entries if e.was_correct is not None]
        correct = sum(1 for e in labeled if e.was_correct)
        accuracy = correct / len(labeled) if labeled else None

        # Accuracy split: trusted vs corrected
        trusted_labeled = [e for e in labeled if e.decision == "TRUST_LLM"]
        corrected_labeled = [e for e in labeled if e.decision == "CORRECTED"]

        trusted_correct = sum(1 for e in trusted_labeled if e.was_correct)
        corrected_correct = sum(1 for e in corrected_labeled if e.was_correct)

        # Error patterns
        error_magnitudes = [e.pop_error_magnitude for e in self.entries]
        confidences = [e.pop_confidence for e in self.entries]

        # Correction quality
        correction_helped = 0
        correction_hurt = 0
        for e in corrected_labeled:
            if e.correct_token:
                if e.final_token.strip().lower() == e.correct_token.strip().lower():
                    correction_helped += 1
                elif e.llm_token.strip().lower() == e.correct_token.strip().lower():
                    correction_hurt += 1

        elapsed = time.time() - self.session_start

        metrics = {
            "total_predictions": total,
            "trust_llm": trusts,
            "corrections_applied": corrections,
            "correction_rate": corrections / total,
            "elapsed_seconds": round(elapsed, 2),
            "avg_error_magnitude": float(np.mean(error_magnitudes)),
            "avg_confidence": float(np.mean(confidences)),
        }

        if labeled:
            metrics.update({
                "overall_accuracy": accuracy,
                "labeled_count": len(labeled),
                "trusted_correct": trusted_correct,
                "trusted_total": len(trusted_labeled),
                "trusted_accuracy": trusted_correct / len(trusted_labeled) if trusted_labeled else None,
                "corrected_correct": corrected_correct,
                "corrected_total": len(corrected_labeled),
                "corrected_accuracy": corrected_correct / len(corrected_labeled) if corrected_labeled else None,
                "correction_helped": correction_helped,
                "correction_hurt": correction_hurt,
            })

        return metrics

    def print_summary(self):
        """Print a summary of the session."""
        m = self.get_metrics()
        print(f"\n{'═' * 60}")
        print(f"  PoP DEBUGGER SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total predictions:    {m['total_predictions']}")
        print(f"  Trusted LLM:          {m['trust_llm']}")
        print(f"  Corrections applied:  {m['corrections_applied']}")
        print(f"  Correction rate:      {m['correction_rate']:.1%}")
        print(f"  Avg error magnitude:  {m['avg_error_magnitude']:.4f}")
        print(f"  Avg PoP confidence:   {m['avg_confidence']:.4f}")

        if "overall_accuracy" in m:
            print(f"\n  ── With Ground Truth ({m['labeled_count']} samples) ──")
            print(f"  Overall accuracy:     {m['overall_accuracy']:.1%}")
            if m.get('trusted_total', 0) > 0:
                print(f"  Trusted accuracy:     {m['trusted_accuracy']:.1%} ({m['trusted_correct']}/{m['trusted_total']})")
            if m.get('corrected_total', 0) > 0:
                print(f"  Corrected accuracy:   {m['corrected_accuracy']:.1%} ({m['corrected_correct']}/{m['corrected_total']})")
            print(f"  Corrections helped:   {m['correction_helped']}")
            print(f"  Corrections hurt:     {m['correction_hurt']}")
            
            # PoP impact analysis
            if m.get('trusted_total', 0) > 0 and m.get('corrected_total', 0) > 0:
                print(f"\n  ── PoP Impact ──")
                baseline = m['labeled_count'] - sum(1 for e in self.entries if e.was_correct)
                pop_wrong = m['correction_hurt']
                pop_right = m['correction_helped']
                print(f"  LLM errors (baseline): {baseline}")
                print(f"  PoP caught & fixed:    {pop_right}")
                print(f"  PoP made worse:        {pop_wrong}")
                if baseline > 0:
                    net = pop_right - pop_wrong
                    print(f"  Net improvement:       {'+' if net > 0 else ''}{net}")

        print(f"  Elapsed:              {m['elapsed_seconds']}s")
        print(f"{'═' * 60}\n")

    def to_json(self, path: Optional[str] = None) -> str:
        """Export all entries as JSON."""
        data = {
            "session_start": self.session_start,
            "metrics": self.get_metrics(),
            "entries": [asdict(e) for e in self.entries]
        }
        text = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
            logger.info(f"Debug log saved to {path}")
        return text

    def get_error_distribution(self) -> List[float]:
        """Get error magnitudes for histogram."""
        return [e.pop_error_magnitude for e in self.entries]

    def get_false_corrections(self) -> List[DebugEntry]:
        """Find cases where PoP correction made things worse."""
        bad = []
        for e in self.entries:
            if e.decision == "CORRECTED" and e.correct_token:
                llm_right = e.llm_token.strip().lower() == e.correct_token.strip().lower()
                pop_right = e.final_token.strip().lower() == e.correct_token.strip().lower()
                if llm_right and not pop_right:
                    bad.append(e)
        return bad

    def get_missed_corrections(self) -> List[DebugEntry]:
        """Find cases where PoP should have corrected but didn't."""
        missed = []
        for e in self.entries:
            if e.decision == "TRUST_LLM" and e.was_correct is False:
                missed.append(e)
        return missed
