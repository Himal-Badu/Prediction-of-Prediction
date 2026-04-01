"""
Evaluation framework for PoP v3.
Provides metrics computation, tier comparison, calibration analysis, and report generation.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


class Evaluator:
    """
    Unified evaluation harness for PoP classifiers.

    Usage:
        ev = Evaluator()
        results = ev.evaluate(model, features, labels)
        ev.generate_report(results, "report.json")
    """

    def evaluate(self, model, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a model on given features / labels.

        Args:
            model: Object with a ``predict(features)`` method returning class predictions.
            features: Feature matrix, shape (N, D).
            labels: Ground-truth labels, shape (N,).

        Returns:
            dict with keys: accuracy, precision, recall, f1, auc_roc,
            confusion_matrix, n_samples.
        """
        preds = model.predict(features)
        preds = np.asarray(preds).astype(int)
        labels = np.asarray(labels).astype(int)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # AUC-ROC requires probability scores; fall back to predictions
        try:
            probas = model.predict_proba(features)[:, 1]
            auc = roc_auc_score(labels, probas)
        except (AttributeError, IndexError, ValueError):
            auc = float("nan")

        cm = confusion_matrix(labels, preds).tolist()

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
            "confusion_matrix": cm,
            "n_samples": int(len(labels)),
        }

    @staticmethod
    def tier_comparison(results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute degradation between tiers relative to the Full tier.

        Args:
            results_dict: ``{"Full": {...}, "Lite": {...}, "Minimal": {...}}``
                          Each value is an ``evaluate()`` output dict.

        Returns:
            Degradation report with per-tier degradation percentages.
        """
        full_f1 = results_dict.get("Full", {}).get("f1", 1.0)
        if full_f1 == 0:
            full_f1 = 1e-8  # avoid div-by-zero

        report: Dict[str, Any] = {"baseline_f1": full_f1, "tiers": {}}
        for tier, res in results_dict.items():
            tier_f1 = res.get("f1", 0.0)
            degradation = (full_f1 - tier_f1) / full_f1 * 100
            report["tiers"][tier] = {
                "f1": tier_f1,
                "degradation_pct": round(degradation, 2),
                "accuracy": res.get("accuracy"),
                "precision": res.get("precision"),
                "recall": res.get("recall"),
                "auc_roc": res.get("auc_roc"),
            }
        return report

    def calibration_curve(self, model, features: np.ndarray,
                          labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Compute reliability-diagram data (calibration curve).

        Args:
            model: Must have ``predict_proba(features)``.
            features: Feature matrix.
            labels: Ground-truth labels.
            n_bins: Number of confidence bins.

        Returns:
            dict with bins (mean_confidence, mean_accuracy, count).
        """
        try:
            probas = model.predict_proba(features)[:, 1]
        except (AttributeError, IndexError):
            # Fall back to 0/1 predictions
            preds = model.predict(features)
            probas = np.clip(preds, 0, 1).astype(float)

        labels = np.asarray(labels).astype(int)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (probas > bin_edges[i]) & (probas <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bins.append({
                "bin_range": [round(bin_edges[i], 3), round(bin_edges[i + 1], 3)],
                "mean_confidence": round(float(probas[mask].mean()), 4),
                "mean_accuracy": round(float((probas[mask].round() == labels[mask]).mean()), 4),
                "count": int(mask.sum()),
            })
        return {"n_bins": n_bins, "bins": bins}

    @staticmethod
    def generate_report(results: Dict[str, Any], output_path: str) -> str:
        """
        Save evaluation results as a formatted JSON report.

        Args:
            results: Arbitrary results dict.
            output_path: File path to write.

        Returns:
            The output path.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return output_path
