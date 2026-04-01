"""
PoP v3 — Unified scoring engine.

Routes input through the tier detector, selects the appropriate feature
extractors, and returns a RiskScore dict.  Placeholder logic — real
engines will be wired in by the ML Engineer.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pop.router import TierDetector, TierLevel


class PoPScorer:
    """Unified scorer that takes tier-detected input and returns RiskScore."""

    def __init__(self) -> None:
        self.tier_detector = TierDetector()

    # ── public API ───────────────────────────────────────────────────

    def score(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Score a request dict and return a result dict.

        Args:
            request: Parsed ScoreRequest as dict.

        Returns:
            Dict with keys: risk_score, confidence, tier_used, label,
            flagged_spans, grounded_spans, features_used.
        """
        tier = self.tier_detector.detect(request)
        caps = TierDetector.tier_capabilities(tier)

        text: str = request.get("text", "")

        # ── placeholder scoring logic ────────────────────────────────
        risk_score, confidence = self._placeholder_score(text, tier)
        label = self._label_from_score(risk_score)

        flagged_spans: List[Dict[str, Any]] = []
        grounded_spans: List[Dict[str, Any]] = []

        if risk_score > 0.7:
            flagged_spans.append({
                "text": text[:80],
                "start": 0,
                "end": min(len(text), 80),
                "risk_level": "high" if risk_score > 0.85 else "medium",
                "reason": "Placeholder: high risk score detected.",
                "score": round(risk_score, 4),
            })
        else:
            grounded_spans.append({
                "text": text[:80],
                "start": 0,
                "end": min(len(text), 80),
                "risk_level": "low",
                "reason": "Placeholder: low risk score.",
                "score": round(risk_score, 4),
            })

        return {
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "tier_used": tier.value,
            "label": label,
            "flagged_spans": flagged_spans,
            "grounded_spans": grounded_spans,
            "features_used": caps["features"],
        }

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _placeholder_score(text: str, tier: TierLevel) -> tuple[float, float]:
        """Return (risk_score, confidence) placeholder values.

        Real implementations will call tier-specific feature extractors
        and a trained fusion model.
        """
        # Very naive heuristic: longer text → slightly higher risk
        base_risk = min(len(text) / 2000.0, 0.95)
        tier_confidence = {
            TierLevel.FULL: 0.95,
            TierLevel.LITE: 0.80,
            TierLevel.MINIMAL: 0.55,
        }
        confidence = tier_confidence[tier]
        return base_risk, confidence

    @staticmethod
    def _label_from_score(risk_score: float) -> str:
        if risk_score < 0.3:
            return "safe"
        if risk_score < 0.7:
            return "warning"
        return "dangerous"
