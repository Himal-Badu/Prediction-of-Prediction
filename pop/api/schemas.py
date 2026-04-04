"""
Pydantic schemas for the PoP v3 confidence-scoring API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Input payload for the /score endpoint."""
    text: str = Field(..., description="The LLM output text to score.")
    logits: Optional[List[List[float]]] = Field(
        None, description="Raw token logit distributions (FULL tier)."
    )
    token_probs: Optional[List[Dict[str, float]]] = Field(
        None, description="Top-k token probabilities per position (LITE tier)."
    )
    context: Optional[str] = Field(
        None, description="Optional prompt or conversation context."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Arbitrary caller metadata (model name, etc.)."
    )


# ── Response ─────────────────────────────────────────────────────────

class SpanResult(BaseModel):
    """A flagged or grounded text span with risk details."""
    text: str = Field(..., description="The span text.")
    start: int = Field(..., description="Start character offset.")
    end: int = Field(..., description="End character offset.")
    risk_level: str = Field(..., description="low | medium | high | critical.")
    reason: str = Field(..., description="Human-readable explanation.")
    score: float = Field(..., ge=0.0, le=1.0, description="Span risk score 0-1.")


class ScoreResponse(BaseModel):
    """Response from the /score endpoint."""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall hallucination risk 0-1.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the risk assessment.")
    tier_used: str = Field(..., description="Tier that was used for scoring (full | lite | minimal).")
    label: str = Field(..., description="Classification label: safe | warning | dangerous.")
    flagged_spans: List[SpanResult] = Field(default_factory=list, description="Spans flagged as risky.")
    grounded_spans: List[SpanResult] = Field(default_factory=list, description="Spans assessed as grounded/safe.")
    features_used: List[str] = Field(default_factory=list, description="Feature extractors that ran.")
    latency_ms: float = Field(..., description="Processing latency in milliseconds.")
