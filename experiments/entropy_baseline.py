"""
Experiment 1: Entropy Baseline
Does PoP's 24-feature model beat simple entropy thresholding?

Uses pre-extracted 24 features (real_features_all.npy).
Builds a simple MLP on 24 features (not the full logit pipeline).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import json
import os

print("=" * 60)
print("EXPERIMENT 1: Entropy Baseline vs PoP v2 (24 features)")
print("=" * 60)

features = np.load('real_features_all.npy')
labels = np.load('real_labels_all.npy')

print(f"\nData: {features.shape[0]} samples, {features.shape[1]} features")
n_correct = int((labels == 0).sum())
n_errors = int((labels == 1).sum())
print(f"Class balance: {n_correct} correct, {n_errors} errors ({n_errors/len(labels)*100:.1f}% errors)")

# Normalize features (important for MLP)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

# --- BASELINE 1: Entropy Threshold ---
print(f"\n{'='*60}")
print("BASELINE 1: Entropy Threshold")
print(f"{'='*60}")
# Feature 0 is entropy (first extracted feature)
entropy = features[:, 0]

best_f1_ent = 0
best_thresh_ent = 0
for pct in range(5, 96):
    thresh = np.percentile(entropy, pct)
    preds = (entropy > thresh).astype(float)
    if preds.sum() == 0 or preds.sum() == len(preds):
        continue
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    if f1 > best_f1_ent:
        best_f1_ent = f1
        best_thresh_ent = thresh

preds_ent = (entropy > best_thresh_ent).astype(float)
p_ent, r_ent, f1_ent, _ = precision_recall_fscore_support(labels, preds_ent, average='binary', zero_division=0)
acc_ent = accuracy_score(labels, preds_ent)
try:
    auc_ent = roc_auc_score(labels, entropy)
except:
    auc_ent = 0.5

print(f"  Threshold:  {best_thresh_ent:.4f}")
print(f"  Accuracy:   {acc_ent:.4f}")
print(f"  Precision:  {p_ent:.4f}")
print(f"  Recall:     {r_ent:.4f}")
print(f"  F1:         {f1_ent:.4f}")
print(f"  AUC-ROC:    {auc_ent:.4f}")

# --- BASELINE 2: Max Probability Threshold ---
print(f"\n{'='*60}")
print("BASELINE 2: Max Probability Threshold")
print(f"{'='*60}")
# Feature 3 should be max/top-1 probability mass
# Let's check correlation with labels to find the best single feature
best_single_f1 = 0
best_single_idx = 0
best_single_name = ""
for i in range(features.shape[1]):
    feat = features[:, i]
    if np.std(feat) < 1e-8:
        continue
    # Try both directions
    for direction, name in [(1, ">"), (-1, "<")]:
        for pct in range(5, 96):
            thresh = np.percentile(feat, pct)
            if direction == 1:
                preds = (feat > thresh).astype(float)
            else:
                preds = (feat < thresh).astype(float)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            if f1 > best_single_f1:
                best_single_f1 = f1
                best_single_idx = i
                best_single_name = f"feat[{i}] {name} {thresh:.4f}"

print(f"  Best single feature: {best_single_name}")
print(f"  Best single-feature F1: {best_single_f1:.4f}")

# Use the best single feature as max-prob baseline
best_feat = features[:, best_single_idx]
best_f1_mp = 0
best_thresh_mp = 0
best_dir = 1
for pct in range(5, 96):
    thresh = np.percentile(best_feat, pct)
    for direction in [1, -1]:
        if direction == 1:
            preds = (best_feat > thresh).astype(float)
        else:
            preds = (best_feat < thresh).astype(float)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1_mp:
            best_f1_mp = f1
            best_thresh_mp = thresh
            best_dir = direction

if best_dir == 1:
    preds_mp = (best_feat > best_thresh_mp).astype(float)
else:
    preds_mp = (best_feat < best_thresh_mp).astype(float)
p_mp, r_mp, f1_mp, _ = precision_recall_fscore_support(labels, preds_mp, average='binary', zero_division=0)
acc_mp = accuracy_score(labels, preds_mp)

print(f"  Feature index: {best_single_idx}")
print(f"  Direction: {'>' if best_dir == 1 else '<'}")
print(f"  Threshold: {best_thresh_mp:.4f}")
print(f"  Accuracy:  {acc_mp:.4f}")
print(f"  Precision: {p_mp:.4f}")
print(f"  Recall:    {r_mp:.4f}")
print(f"  F1:        {f1_mp:.4f}")

# --- MODEL: Simple MLP on 24 features ---
print(f"\n{'='*60}")
print("MODEL: Simple MLP (24 features)")
print(f"{'='*60}")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device('cpu')
X = torch.FloatTensor(features_norm)
y = torch.FloatTensor(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_norm, labels)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Handle class imbalance with weighted sampling
    class_weights = 1.0 / torch.tensor([n_errors, n_correct], dtype=torch.float)
    sample_weights = class_weights[y_train.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    model = SimpleMLP(input_dim=24, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)
    
    # Train
    model.train()
    for epoch in range(50):
        for xb, yb in train_dl:
            out = model(xb)
            loss = nn.BCEWithLogitsLoss()(out, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        p, r, f1, _ = precision_recall_fscore_support(
            y_val.numpy(), preds.numpy(), average='binary', zero_division=0
        )
        acc = accuracy_score(y_val.numpy(), preds.numpy())
        try:
            auc = roc_auc_score(y_val.numpy(), probs.numpy())
        except:
            auc = 0.5
        
        fold_results.append({'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc, 'auc': auc})
        print(f"  Fold {fold+1}: P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f}")

avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
std = {k: np.std([f[k] for f in fold_results]) for k in fold_results[0]}

print(f"\n  MLP Average:")
print(f"    Accuracy:  {avg['accuracy']:.4f} ± {std['accuracy']:.4f}")
print(f"    Precision: {avg['precision']:.4f} ± {std['precision']:.4f}")
print(f"    Recall:    {avg['recall']:.4f} ± {std['recall']:.4f}")
print(f"    F1:        {avg['f1']:.4f} ± {std['f1']:.4f}")
print(f"    AUC-ROC:   {avg['auc']:.4f} ± {std['auc']:.4f}")

# --- FINAL COMPARISON ---
print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 65)
print(f"{'Entropy Threshold':<25} {p_ent:>10.4f} {r_ent:>10.4f} {f1_ent:>10.4f} {auc_ent:>10.4f}")
print(f"{'Best Single Feature':<25} {p_mp:>10.4f} {r_mp:>10.4f} {f1_mp:>10.4f} {'N/A':>10}")
print(f"{'MLP (24 features)':<25} {avg['precision']:>10.4f} {avg['recall']:>10.4f} {avg['f1']:>10.4f} {avg['auc']:>10.4f}")

f1_gain_vs_entropy = avg['f1'] - f1_ent
f1_gain_vs_best_single = avg['f1'] - f1_mp

print(f"\nMLP vs Entropy:       F1 = {f1_gain_vs_entropy:+.4f} ({f1_gain_vs_entropy/f1_ent*100:+.1f}%)")
print(f"MLP vs Best Single:   F1 = {f1_gain_vs_best_single:+.4f} ({f1_gain_vs_best_single/f1_mp*100:+.1f}%)")

if f1_gain_vs_entropy > 0.05:
    verdict = "PASS"
    msg = "✅ PoP BEATS entropy by >5% F1 — signal is real, proceed with product"
elif f1_gain_vs_entropy > 0:
    verdict = "MARGINAL"
    msg = "⚠️  PoP slightly beats entropy — marginal advantage, investigate further"
else:
    verdict = "FAIL"
    msg = "❌ PoP does NOT beat entropy — fundamental problem, rethink approach"

print(f"\n{msg}")

# Save results
os.makedirs('experiments', exist_ok=True)
results = {
    "experiment": "entropy_baseline",
    "data": {"samples": int(features.shape[0]), "features": int(features.shape[1]),
             "n_correct": n_correct, "n_errors": n_errors},
    "entropy_baseline": {"precision": float(p_ent), "recall": float(r_ent),
                         "f1": float(f1_ent), "accuracy": float(acc_ent), "auc": float(auc_ent),
                         "threshold": float(best_thresh_ent)},
    "best_single_feature": {"index": int(best_single_idx), "precision": float(p_mp),
                            "recall": float(r_mp), "f1": float(f1_mp), "accuracy": float(acc_mp)},
    "mlp_24_features": {"precision": float(avg['precision']), "recall": float(avg['recall']),
                        "f1": float(avg['f1']), "accuracy": float(avg['accuracy']),
                        "auc": float(avg['auc']),
                        "precision_std": float(std['precision']),
                        "recall_std": float(std['recall']),
                        "f1_std": float(std['f1'])},
    "f1_gain_vs_entropy": float(f1_gain_vs_entropy),
    "f1_gain_vs_best_single": float(f1_gain_vs_best_single),
    "verdict": verdict
}

with open('experiments/entropy_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to experiments/entropy_baseline_results.json")
