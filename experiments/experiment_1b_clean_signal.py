"""
Experiment 1b: Clean Signal Test
Following Himal's direction:
1. Correlation matrix → pick top 3-5 independent features
2. Balance classes (downsample)
3. Simple classifier (logistic regression + tiny MLP)
4. Evaluate AUC-ROC, PR-AUC, confusion matrix
5. Compare against entropy baseline
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, 
                             accuracy_score, confusion_matrix, average_precision_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import os

print("=" * 70)
print("EXPERIMENT 1b: Clean Signal Test (Fixed Imbalance)")
print("=" * 70)

features = np.load('real_features_all.npy')
labels = np.load('real_labels_all.npy')
n_correct = int((labels == 0).sum())
n_errors = int((labels == 1).sum())
print(f"\nOriginal data: {features.shape[0]} samples, {features.shape[1]} features")
print(f"Class balance: {n_correct} correct, {n_errors} errors ({n_errors/len(labels)*100:.1f}% errors)")

# --- STEP 1: Correlation Matrix → Pick Independent Features ---
print(f"\n{'='*70}")
print("STEP 1: Correlation Analysis")
print(f"{'='*70}")

# Compute correlation matrix
corr_matrix = np.corrcoef(features.T)
print(f"\nCorrelation matrix shape: {corr_matrix.shape}")

# Find feature-label correlations (which features correlate with the label?)
feature_label_corr = np.array([
    abs(np.corrcoef(features[:, i], labels)[0, 1]) 
    for i in range(features.shape[1])
])

# Rank features by correlation with labels
ranked = np.argsort(feature_label_corr)[::-1]
print(f"\nTop 10 features by |correlation with label|:")
for i, idx in enumerate(ranked[:10]):
    print(f"  feat[{idx:2d}] correlation: {feature_label_corr[idx]:.4f}")

# Select top 5 features that are NOT highly correlated with each other
selected = [ranked[0]]
for idx in ranked[1:]:
    if len(selected) >= 5:
        break
    # Check if this feature is too correlated with already-selected features
    max_corr_with_selected = max(
        abs(corr_matrix[idx, s]) for s in selected
    )
    if max_corr_with_selected < 0.7:  # threshold: 70% correlation
        selected.append(idx)
    else:
        print(f"  Skipping feat[{idx}] (correlated {max_corr_with_selected:.2f} with selected)")

print(f"\nSelected features: {selected}")
for idx in selected:
    print(f"  feat[{idx}]: label_corr={feature_label_corr[idx]:.4f}, "
          f"mean={features[:, idx].mean():.4f}, std={features[:, idx].std():.4f}")

# --- STEP 2: Balance Classes (Downsample) ---
print(f"\n{'='*70}")
print("STEP 2: Balance Classes")
print(f"{'='*70}")

np.random.seed(42)
correct_idx = np.where(labels == 0)[0]
error_idx = np.where(labels == 1)[0]

# Downsample errors to match correct count
n_min = min(n_correct, n_errors)
error_downsampled = np.random.choice(error_idx, size=n_min, replace=False)
balanced_idx = np.concatenate([correct_idx, error_downsampled])
np.random.shuffle(balanced_idx)

X_balanced = features[balanced_idx]
y_balanced = labels[balanced_idx]

print(f"Balanced dataset: {len(y_balanced)} samples")
print(f"  Correct: {(y_balanced == 0).sum()}, Errors: {(y_balanced == 1).sum()}")
print(f"  50/50 split ✓")

# Also prepare selected-features version
X_selected = X_balanced[:, selected]
X_all = X_balanced

# Normalize
scaler_all = StandardScaler()
X_all_norm = scaler_all.fit_transform(X_all)
X_selected_norm = StandardScaler().fit_transform(X_selected)
X_entropy_only = X_balanced[:, [selected[0]]]  # best entropy-like feature
X_entropy_norm = StandardScaler().fit_transform(X_entropy_only)

# --- STEP 3: Train Classifiers ---
print(f"\n{'='*70}")
print("STEP 3: Train & Evaluate")
print(f"{'='*70}")

def evaluate(y_true, y_pred, y_prob, name):
    """Full evaluation with all metrics."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = 0.5
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n  --- {name} ---")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {p:.4f}")
    print(f"    Recall:    {r:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    PR-AUC:    {pr_auc:.4f}")
    print(f"    Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, 
            "auc_roc": auc, "pr_auc": pr_auc,
            "confusion": {"TN": int(cm[0,0]), "FP": int(cm[0,1]), 
                          "FN": int(cm[1,0]), "TP": int(cm[1,1])}}

results = {}

# 5-fold CV on balanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Classifier A: Logistic Regression on entropy only ---
print(f"\n{'='*50}")
print("A: Logistic Regression (1 feature — entropy)")
fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_entropy_norm, y_balanced)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_entropy_norm[train_idx], y_balanced[train_idx])
    y_pred = clf.predict(X_entropy_norm[val_idx])
    y_prob = clf.predict_proba(X_entropy_norm[val_idx])[:, 1]
    r = evaluate(y_balanced[val_idx], y_pred, y_prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in ['accuracy','precision','recall','f1','auc_roc','pr_auc']}
print(f"\n  Average: F1={avg['f1']:.4f} AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f}")
results['logreg_entropy'] = avg

# --- Classifier B: Logistic Regression on selected features ---
print(f"\n{'='*50}")
print(f"B: Logistic Regression ({len(selected)} selected features)")
fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected_norm, y_balanced)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_selected_norm[train_idx], y_balanced[train_idx])
    y_pred = clf.predict(X_selected_norm[val_idx])
    y_prob = clf.predict_proba(X_selected_norm[val_idx])[:, 1]
    r = evaluate(y_balanced[val_idx], y_pred, y_prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in ['accuracy','precision','recall','f1','auc_roc','pr_auc']}
print(f"\n  Average: F1={avg['f1']:.4f} AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f}")
results['logreg_selected'] = avg

# --- Classifier C: Logistic Regression on all 24 features ---
print(f"\n{'='*50}")
print("C: Logistic Regression (all 24 features)")
fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_norm, y_balanced)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_all_norm[train_idx], y_balanced[train_idx])
    y_pred = clf.predict(X_all_norm[val_idx])
    y_prob = clf.predict_proba(X_all_norm[val_idx])[:, 1]
    r = evaluate(y_balanced[val_idx], y_pred, y_prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in ['accuracy','precision','recall','f1','auc_roc','pr_auc']}
print(f"\n  Average: F1={avg['f1']:.4f} AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f}")
results['logreg_all24'] = avg

# --- Classifier D: Tiny MLP on selected features ---
print(f"\n{'='*50}")
print(f"D: Tiny MLP ({len(selected)} selected features)")

class TinyMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

X_sel_t = torch.FloatTensor(X_selected_norm)
y_t = torch.FloatTensor(y_balanced)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected_norm, y_balanced)):
    model = TinyMLP(len(selected))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_ds = TensorDataset(X_sel_t[train_idx], y_t[train_idx])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(100):
        for xb, yb in train_dl:
            loss = nn.BCEWithLogitsLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logits = model(X_sel_t[val_idx])
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(float)
    r = evaluate(y_balanced[val_idx], preds, probs, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in ['accuracy','precision','recall','f1','auc_roc','pr_auc']}
print(f"\n  Average: F1={avg['f1']:.4f} AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f}")
results['mlp_selected'] = avg

# --- Classifier E: Tiny MLP on all 24 features ---
print(f"\n{'='*50}")
print("E: Tiny MLP (all 24 features)")

X_all_t = torch.FloatTensor(X_all_norm)
fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_norm, y_balanced)):
    model = TinyMLP(24)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_ds = TensorDataset(X_all_t[train_idx], y_t[train_idx])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(100):
        for xb, yb in train_dl:
            loss = nn.BCEWithLogitsLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logits = model(X_all_t[val_idx])
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(float)
    r = evaluate(y_balanced[val_idx], preds, probs, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in ['accuracy','precision','recall','f1','auc_roc','pr_auc']}
print(f"\n  Average: F1={avg['f1']:.4f} AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f}")
results['mlp_all24'] = avg

# --- FINAL TABLE ---
print(f"\n{'='*70}")
print("FINAL COMPARISON (Balanced Data)")
print(f"{'='*70}")
print(f"{'Method':<30} {'F1':>8} {'AUC-ROC':>8} {'PR-AUC':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 72)
for name, r in results.items():
    print(f"{name:<30} {r['f1']:>8.4f} {r['auc_roc']:>8.4f} {r['pr_auc']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")

# Key comparison: does selected features beat entropy?
entropy_auc = results['logreg_entropy']['auc_roc']
selected_auc = results['logreg_selected']['auc_roc']
mlp_sel_auc = results['mlp_selected']['auc_roc']

print(f"\n--- KEY FINDING ---")
print(f"Entropy only:        AUC={entropy_auc:.4f} PR-AUC={results['logreg_entropy']['pr_auc']:.4f}")
print(f"Selected features:   AUC={selected_auc:.4f} PR-AUC={results['logreg_selected']['pr_auc']:.4f}")
print(f"MLP selected:        AUC={mlp_sel_auc:.4f} PR-AUC={results['mlp_selected']['pr_auc']:.4f}")

if selected_auc > entropy_auc + 0.03:
    print(f"\n✅ Selected features beat entropy by {selected_auc - entropy_auc:.4f} AUC — REAL SIGNAL")
elif selected_auc > entropy_auc:
    print(f"\n⚠️  Selected features marginally beat entropy (+{selected_auc - entropy_auc:.4f} AUC) — WEAK SIGNAL")
else:
    print(f"\n❌ Selected features DO NOT beat entropy — 24 features ≈ entropy alone")

# Save
os.makedirs('experiments', exist_ok=True)
with open('experiments/experiment_1b_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to experiments/experiment_1b_results.json")
