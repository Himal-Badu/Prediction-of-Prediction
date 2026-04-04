"""Tests for the PoP v3 FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pop.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestTiersEndpoint:
    def test_tiers_returns_all_three(self, client):
        resp = client.get("/tiers")
        assert resp.status_code == 200
        data = resp.json()
        assert "full" in data
        assert "lite" in data
        assert "minimal" in data

    def test_tiers_have_features(self, client):
        resp = client.get("/tiers")
        data = resp.json()
        for tier in ("full", "lite", "minimal"):
            assert "features" in data[tier]
            assert len(data[tier]["features"]) > 0


class TestScoreEndpoint:
    def test_minimal_text_only(self, client):
        resp = client.post("/api/v1/score", json={"text": "The sky is blue."})
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["risk_score"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["tier_used"] == "minimal"
        assert data["label"] in ("safe", "warning", "dangerous")
        assert "latency_ms" in data

    def test_full_with_logits(self, client):
        resp = client.post("/api/v1/score", json={
            "text": "Paris is the capital of France.",
            "logits": [[0.1, 0.9], [0.3, 0.7]],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier_used"] == "full"

    def test_lite_with_token_probs(self, client):
        resp = client.post("/api/v1/score", json={
            "text": "Hello world",
            "token_probs": [{"hello": 0.9, "hi": 0.1}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier_used"] == "lite"

    def test_missing_text_returns_422(self, client):
        resp = client.post("/api/v1/score", json={})
        assert resp.status_code == 422

    def test_response_has_spans(self, client):
        resp = client.post("/api/v1/score", json={"text": "A short test."})
        data = resp.json()
        assert isinstance(data["flagged_spans"], list)
        assert isinstance(data["grounded_spans"], list)

    def test_response_has_features_used(self, client):
        resp = client.post("/api/v1/score", json={"text": "Test."})
        data = resp.json()
        assert isinstance(data["features_used"], list)
        assert len(data["features_used"]) > 0

    def test_with_context_and_metadata(self, client):
        resp = client.post("/api/v1/score", json={
            "text": "Some output.",
            "context": "What is 2+2?",
            "metadata": {"model": "gpt-4"},
        })
        assert resp.status_code == 200


class TestStreamingEndpoint:
    def test_websocket_stream(self, client):
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json({"token": "Hello"})
            data = ws.receive_json()
            assert data["type"] == "token"
            assert data["token"] == "Hello"
            assert "cumulative_risk" in data

            ws.send_json({"token": " world"})
            data = ws.receive_json()
            assert data["token_index"] == 2

            ws.send_json({"done": True})
            data = ws.receive_json()
            assert data["type"] == "final"
            assert data["total_tokens"] == 2
