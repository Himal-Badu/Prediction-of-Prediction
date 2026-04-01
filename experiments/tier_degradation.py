"""
Tier Degradation Experiment for PoP v3.

Measures how much hallucination-detection performance drops across
Full → Lite → Minimal information tiers.

- Full:    All 24 logits/features (complete model access)
- Lite:    Top-k probabilities only (reduced feature set)
- Minimal: Text-derived features only (no logit access)

Outputs a JSON report with per-tier metrics and degradation percentages.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pop.training.evaluator import Evaluator
from pop.models.calibration import TemperatureScaling

DATA_DIR = REPO
RESULTS_DIR = os.path.join(REPO, "experiments", "results")


# ---------------------------------------------------------------------------
# Simple MLP classifier
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """Two-hidden-layer MLP for binary classification."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x):
        return self.net(x)


class SklearnMLPWrapper:
    """Wraps a PyTorch SimpleMLP to expose predict() / predict_proba() for Evaluator."""

    def __init__(self, model: SimpleMLP):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()


# ---------------------------------------------------------------------------
# Tier simulation
# ---------------------------------------------------------------------------
def simulate_tiers(features: np.ndarray, n_logit_features: int = 8):
    """
    Split a unified feature matrix into tier-specific subsets.

    Assumption (matches real PoP v3 feature layout):
      - First n_logit_features columns = logit-derived (full logits)
      - Next n_logit_features columns = top-k prob features (lite)
      - Remaining columns = text-derived features (minimal)

    Args:
        features: Full feature matrix, shape (N, D).
        n_logit_features: Width of each logit block.

    Returns:
        dict mapping tier name → feature subset.
    """
    D = features.shape[1]
    # Full: everything
    full = features

    # Lite: only the top-k prob block (columns [n_logit : 2*n_logit])
    # Plus entropy / confidence from logits
    lite_end = min(2 * n_logit_features, D)
    lite_cols = list(range(n_logit_features, lite_end))
    # Also grab entropy/confidence-like features if available
    if D > 2 * n_logit_features:
        lite_cols += list(range(2 * n_logit_features, min(2 * n_logit_features + 2, D)))
    lite = features[:, lite_cols]

    # Minimal: only text-derived features (everything after the logit blocks)
    text_start = min(2 * n_logit_features + 2, D)
    if text_start < D:
        minimal = features[:, text_start:]
    else:
        # If no separate text features, use last few features as proxy
        minimal = features[:, -max(4, D // 4):]

    return {"Full": full, "Lite": lite, "Minimal": minimal}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              input_dim: int, epochs: int = 80, lr: float = 1e-3) -> SimpleMLP:
    """Train a SimpleMLP on numpy arrays, return the model."""
    model = SimpleMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(data_path_features: str = None,
                   data_path_labels: str = None,
                   output_path: str = None):
    """
    Run the full tier-degradation experiment.

    Args:
        data_path_features: Path to features .npy file.
        data_path_labels:   Path to labels .npy file.
        output_path:        Where to save the JSON report.
    """
    # Load data
    if data_path_features is None:
        data_path_features = os.path.join(DATA_DIR, "real_features_all.npy")
    if data_path_labels is None:
        data_path_labels = os.path.join(DATA_DIR, "real_labels_all.npy")

    print(f"Loading features from {data_path_features}")
    features = np.load(data_path_features)
    print(f"Loading labels   from {data_path_labels}")
    labels = np.load(data_path_labels).astype(int)

    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Label distribution: 0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}")

    # Simulate tiers
    tiers = simulate_tiers(features)
    for name, feat in tiers.items():
        print(f"  Tier '{name}': {feat.shape[1]} features")

    # Split data
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42,
                                           stratify=labels)

    evaluator = Evaluator()
    results = {}
    tier_models = {}

    for tier_name, tier_features in tiers.items():
        print(f"\n{'='*50}")
        print(f"Training tier: {tier_name}  ({tier_features.shape[1]} features)")
        print(f"{'='*50}")

        X_train = tier_features[train_idx]
        y_train = labels[train_idx]
        X_test = tier_features[test_idx]
        y_test = labels[test_idx]

        t0 = time.time()
        model = train_mlp(X_train, y_train, input_dim=X_train.shape[1])
        train_time = time.time() - t0

        wrapper = SklearnMLPWrapper(model)
        metrics = evaluator.evaluate(wrapper, X_test, y_test)
        metrics["train_time_s"] = round(train_time, 3)
        metrics["feature_dim"] = int(tier_features.shape[1])

        # Calibration
        model.eval()
        with torch.no_grad():
            logits_val = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        temp = TemperatureScaling.calibrate(logits_val, y_test)
        calibrated_probs = TemperatureScaling.apply(logits_val, temp)
        ece_before = TemperatureScaling.ece(
            torch.softmax(torch.tensor(logits_val), dim=1).numpy(), y_test
        )
        ece_after = TemperatureScaling.ece(calibrated_probs, y_test)

        metrics["calibration"] = {
            "temperature": round(temp, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
        }

        results[tier_name] = metrics
        tier_models[tier_name] = wrapper

        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  AUC-ROC:   {metrics['auc_roc']}")
        print(f"  ECE before→after calibration: {ece_before:.4f} → {ece_after:.4f}")
        print(f"  Train time: {train_time:.2f}s")

    # Tier comparison
    print(f"\n{'='*50}")
    print("TIER DEGRADATION REPORT")
    print(f"{'='*50}")
    degradation = evaluator.tier_comparison(results)
    for tier, info in degradation["tiers"].items():
        print(f"  {tier:8s}  F1={info['f1']:.4f}  Degradation={info['degradation_pct']:.1f}%")

    # Calibration curves per tier
    calibration_data = {}
    for tier_name, wrapper in tier_models.items():
        tier_features = tiers[tier_name]
        X_test = tier_features[test_idx]
        y_test = labels[test_idx]
        cal = evaluator.calibration_curve(wrapper, X_test, y_test)
        calibration_data[tier_name] = cal

    # Assemble full report
    full_report = {
        "experiment": "tier_degradation",
        "dataset": {
            "features_path": data_path_features,
            "labels_path": data_path_labels,
            "n_samples": int(features.shape[0]),
            "n_features_full": int(features.shape[1]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "tier_results": results,
        "degradation": degradation,
        "calibration_curves": calibration_data,
    }

    # Save
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "tier_degradation.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nReport saved to {output_path}")

    return full_report


if __name__ == "__main__":
    report = run_experiment()
    # Also print summary JSON to stdout
    print("\n" + json.dumps(report["degradation"], indent=2))
