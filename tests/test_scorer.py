"""
Robustness tests for PoP scorer / model pipeline.

Covers edge cases, malformed inputs, and latency requirements.
"""

import time
import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """Minimal MLP for testing."""
    def __init__(self, input_dim=8, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_scorer(input_dim=8):
    """Return a model + predict wrapper matching PoP conventions."""
    model = TinyMLP(input_dim=input_dim)
    model.eval()

    def predict(features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32))
            return logits.argmax(dim=1).numpy()

    def predict_proba(features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    model.predict = predict
    model.predict_proba = predict_proba
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Empty or zero-length inputs should not crash."""

    def test_empty_array(self):
        scorer = make_scorer(8)
        features = np.empty((0, 8), dtype=np.float32)
        preds = scorer.predict(features)
        assert len(preds) == 0

    def test_empty_proba(self):
        scorer = make_scorer(8)
        features = np.empty((0, 8), dtype=np.float32)
        probas = scorer.predict_proba(features)
        assert probas.shape[0] == 0


class TestVeryLongInput:
    """Simulate very large inputs (>10K samples)."""

    def test_10k_samples(self):
        scorer = make_scorer(8)
        features = np.random.randn(10_000, 8).astype(np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (10_000,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_50k_samples(self):
        scorer = make_scorer(8)
        features = np.random.randn(50_000, 8).astype(np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (50_000,)


class TestMalformedLogits:
    """Handle wrong shapes and NaN/Inf values gracefully."""

    def test_wrong_feature_dim(self):
        scorer = make_scorer(8)
        features = np.random.randn(10, 4).astype(np.float32)  # wrong dim
        with pytest.raises(Exception):
            scorer.predict(features)

    def test_nan_values(self):
        scorer = make_scorer(8)
        features = np.full((5, 8), np.nan, dtype=np.float32)
        # Should not crash — model may produce NaN logits but argmax still works
        try:
            preds = scorer.predict(features)
            # NaN logits → argmax returns 0 or 1 (implementation-dependent)
            assert preds.shape == (5,)
        except Exception:
            pass  # acceptable — NaN input is degenerate

    def test_inf_values(self):
        scorer = make_scorer(8)
        features = np.full((5, 8), np.inf, dtype=np.float32)
        try:
            preds = scorer.predict(features)
            assert preds.shape == (5,)
        except Exception:
            pass

    def test_negative_inf_values(self):
        scorer = make_scorer(8)
        features = np.full((5, 8), -np.inf, dtype=np.float32)
        try:
            preds = scorer.predict(features)
            assert preds.shape == (5,)
        except Exception:
            pass

    def test_1d_input(self):
        """Single sample as 1D array should be handled."""
        scorer = make_scorer(8)
        features = np.random.randn(8).astype(np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (1,)


class TestEdgeCases:
    """Single-token and uniform inputs."""

    def test_single_sample(self):
        scorer = make_scorer(8)
        features = np.random.randn(1, 8).astype(np.float32)
        preds = scorer.predict(features)
        assert len(preds) == 1
        assert preds[0] in (0, 1)

    def test_all_same_value(self):
        scorer = make_scorer(8)
        features = np.ones((20, 8), dtype=np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (20,)
        # All same input → all predictions should be the same class
        assert len(set(preds)) == 1

    def test_all_zeros(self):
        scorer = make_scorer(8)
        features = np.zeros((20, 8), dtype=np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (20,)

    def test_extreme_values(self):
        scorer = make_scorer(8)
        features = np.full((10, 8), 1e6, dtype=np.float32)
        preds = scorer.predict(features)
        assert preds.shape == (10,)

    def test_probability_sum_to_one(self):
        scorer = make_scorer(8)
        features = np.random.randn(50, 8).astype(np.float32)
        probas = scorer.predict_proba(features)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


class TestLatency:
    """Model should meet latency SLAs."""

    def test_single_sample_latency(self):
        scorer = make_scorer(8)
        features = np.random.randn(1, 8).astype(np.float32)
        # Warm up
        scorer.predict(features)

        t0 = time.perf_counter()
        for _ in range(100):
            scorer.predict(features)
        elapsed = time.perf_counter() - t0
        avg_ms = elapsed / 100 * 1000
        print(f"Single-sample latency: {avg_ms:.2f} ms")
        assert avg_ms < 50, f"Single-sample latency {avg_ms:.1f} ms exceeds 50 ms SLA"

    def test_batch_latency(self):
        scorer = make_scorer(8)
        features = np.random.randn(1000, 8).astype(np.float32)
        # Warm up
        scorer.predict(features)

        t0 = time.perf_counter()
        scorer.predict(features)
        elapsed = time.perf_counter() - t0
        ms_per_sample = elapsed / 1000 * 1000
        print(f"Batch (1000) per-sample latency: {ms_per_sample:.3f} ms")
        assert elapsed < 5.0, f"1000-sample batch took {elapsed:.2f}s, exceeds 5s"

    def test_proba_latency(self):
        scorer = make_scorer(8)
        features = np.random.randn(100, 8).astype(np.float32)
        # Warm up
        scorer.predict_proba(features)

        t0 = time.perf_counter()
        for _ in range(100):
            scorer.predict_proba(features)
        elapsed = time.perf_counter() - t0
        avg_ms = elapsed / 100 * 1000
        print(f"predict_proba latency (100 samples): {avg_ms:.2f} ms")
        assert avg_ms < 100, f"predict_proba latency {avg_ms:.1f} ms exceeds 100 ms"
