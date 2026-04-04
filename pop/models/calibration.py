"""
Calibration module for PoP v3.
Temperature scaling to improve probability calibration.
"""

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaling:
    """
    Post-hoc calibration via temperature scaling.
    
    Learns a single temperature parameter T on a validation set that
    minimizes negative log-likelihood.  Calibrated probabilities are
    softmax(logits / T).
    """

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over last axis."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
        """Negative log-likelihood given temperature."""
        scaled = logits / max(temperature, 1e-6)
        probs = TemperatureScaling.softmax(scaled)
        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        nll_val = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        return float(nll_val)

    @classmethod
    def calibrate(cls, logits: np.ndarray, labels: np.ndarray,
                  t_range: tuple = (0.01, 100.0)) -> float:
        """
        Learn optimal temperature on validation data.

        Args:
            logits: Raw model logits, shape (N, C).
            labels: Ground-truth class indices, shape (N,).
            t_range: Search range for temperature.

        Returns:
            Optimal temperature scalar.
        """
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if logits.shape[1] == 1:
            # Binary case — create two-column logits
            logits = np.concatenate([-logits, logits], axis=1)

        result = minimize_scalar(
            lambda t: cls.nll(logits, labels, t),
            bounds=t_range,
            method="bounded",
        )
        return float(result.x)

    @classmethod
    def apply(cls, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to produce calibrated probabilities.

        Args:
            logits: Raw model logits, shape (N, C).
            temperature: Learned temperature parameter.

        Returns:
            Calibrated probabilities, shape (N, C).
        """
        return cls.softmax(logits / max(temperature, 1e-6))

    @classmethod
    def ece(cls, probs: np.ndarray, labels: np.ndarray,
            n_bins: int = 15) -> float:
        """
        Expected Calibration Error (ECE).

        Divides confidence into bins and measures the gap between
        mean accuracy and mean confidence in each bin.

        Args:
            probs: Predicted probabilities, shape (N, C).
            labels: Ground-truth class indices, shape (N,).
            n_bins: Number of confidence bins.

        Returns:
            ECE scalar (lower is better, 0 = perfectly calibrated).
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_val = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece_val += (n_in_bin / len(labels)) * abs(avg_acc - avg_conf)

        return float(ece_val)
