"""
WebSocket streaming endpoint for real-time token scoring.

Clients connect to /api/v1/stream and send JSON messages with a
``token`` field.  The server responds with an incremental risk score
after each token.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pop.router import TierDetector, TierLevel

streaming_router = APIRouter()


class StreamScorer:
    """Accumulates tokens and returns incremental risk scores."""

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._token_count: int = 0

    def add_token(self, token: str) -> Dict[str, Any]:
        """Add a token and return the current incremental risk assessment."""
        self._buffer.append(token)
        self._token_count += 1
        full_text = "".join(self._buffer)

        # Placeholder heuristic — real version uses token-level features
        risk_score = min(self._token_count / 300.0, 0.95)
        if risk_score < 0.3:
            label = "safe"
        elif risk_score < 0.7:
            label = "warning"
        else:
            label = "dangerous"

        return {
            "token_index": self._token_count,
            "token": token,
            "cumulative_risk": round(risk_score, 4),
            "label": label,
            "text_so_far": full_text[-200:],  # last 200 chars for context
        }


@streaming_router.websocket("/api/v1/stream")
async def stream_endpoint(ws: WebSocket):
    """Accept token-by-token streaming and return incremental risk scores.

    Client sends:  {"token": "<next token>", "done": false}
    Server sends:  {"token_index": N, "cumulative_risk": X, "label": "...", ...}

    Client sends:  {"done": true}  → server sends final summary and closes.
    """
    await ws.accept()
    scorer = StreamScorer()
    t0 = time.perf_counter()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "invalid JSON"})
                continue

            if msg.get("done"):
                elapsed_ms = (time.perf_counter() - t0) * 1000
                await ws.send_json({
                    "type": "final",
                    "total_tokens": scorer._token_count,
                    "final_risk": round(
                        min(scorer._token_count / 300.0, 0.95), 4
                    ),
                    "latency_ms": round(elapsed_ms, 2),
                })
                await ws.close()
                return

            token = msg.get("token", "")
            result = scorer.add_token(token)
            result["type"] = "token"
            await ws.send_json(result)

    except WebSocketDisconnect:
        pass
