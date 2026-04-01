"""
Experiment 3: Hybrid Detection at Scale
- Generate 50K-100K synthetic samples across 3 domains
- Add NLI entailment features
- Test: logit features vs NLI features vs HYBRID (both)
- Test cross-domain generalization
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             accuracy_score, confusion_matrix, average_precision_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import json
import os
import sys
import random
import time
sys.path.insert(0, '.')

print("=" * 70)
print("EXPERIMENT 3: Hybrid Detection at Scale")
print("=" * 70)

# ============================================================
# STEP 1: Generate Large-Scale Synthetic Dataset
# ============================================================
print(f"\n{'='*70}")
print("STEP 1: Generating 100K Synthetic Samples")
print(f"{'='*70}")

def generate_synthetic_data(n_samples=100000, n_features=24, seed=42):
    """
    Generate synthetic hallucination detection data.
    
    Simulates realistic logit distribution patterns:
    - Correct outputs: lower entropy, higher top-1 mass, stable margins
    - Incorrect outputs: higher entropy, lower top-1 mass, unstable margins
    - With realistic noise and overlap between classes
    
    Also generates NLI-like features:
    - Entailment score (high = output matches source)
    - Contradiction score (high = output contradicts source)
    - Neutral score (uncertain)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_correct = n_samples // 2
    n_errors = n_samples - n_correct
    
    # --- Logit features (24 features) ---
    # Simulate what real logit distributions look like
    
    # Correct outputs: sharp distributions, high confidence
    correct_features = np.zeros((n_correct, n_features))
    # Feature 0: entropy (lower for correct)
    correct_features[:, 0] = np.random.normal(2.5, 1.0, n_correct).clip(0, 10)
    # Feature 1: top-1 probability mass (higher for correct)
    correct_features[:, 1] = np.random.normal(0.7, 0.15, n_correct).clip(0, 1)
    # Feature 2: top-5 probability mass
    correct_features[:, 2] = np.random.normal(0.9, 0.08, n_correct).clip(0, 1)
    # Feature 3: max logit
    correct_features[:, 3] = np.random.normal(5.0, 2.0, n_correct).clip(-5, 15)
    # Feature 4: mean logit
    correct_features[:, 4] = np.random.normal(-8.0, 3.0, n_correct).clip(-20, 5)
    # Feature 5: logit std
    correct_features[:, 5] = np.random.normal(4.0, 1.5, n_correct).clip(0, 15)
    # Feature 6: top-1/top-2 margin (higher for correct - more decisive)
    correct_features[:, 6] = np.random.normal(2.0, 1.5, n_correct).clip(0, 10)
    # Feature 7: perplexity
    correct_features[:, 7] = np.random.normal(15.0, 8.0, n_correct).clip(1, 100)
    # Features 8-23: other distributional stats (correlated with above)
    for i in range(8, n_features):
        base = correct_features[:, i % 8]
        noise = np.random.normal(0, 0.5 * np.abs(base).mean(), n_correct)
        correct_features[:, i] = base + noise
    
    # Incorrect outputs: flatter distributions, more uncertainty
    error_features = np.zeros((n_errors, n_features))
    # Higher entropy
    error_features[:, 0] = np.random.normal(4.5, 1.5, n_errors).clip(0, 12)
    # Lower top-1 mass
    error_features[:, 1] = np.random.normal(0.4, 0.2, n_errors).clip(0, 1)
    # Lower top-5 mass
    error_features[:, 2] = np.random.normal(0.75, 0.15, n_errors).clip(0, 1)
    # Lower max logit
    error_features[:, 3] = np.random.normal(2.0, 2.5, n_errors).clip(-5, 15)
    # Different mean logit
    error_features[:, 4] = np.random.normal(-10.0, 4.0, n_errors).clip(-25, 5)
    # Higher logit std (more spread)
    error_features[:, 5] = np.random.normal(5.5, 2.0, n_errors).clip(0, 15)
    # Lower margin (less decisive)
    error_features[:, 6] = np.random.normal(0.8, 1.0, n_errors).clip(0, 10)
    # Higher perplexity
    error_features[:, 7] = np.random.normal(30.0, 15.0, n_errors).clip(1, 200)
    # Features 8-23
    for i in range(8, n_features):
        base = error_features[:, i % 8]
        noise = np.random.normal(0, 0.6 * np.abs(base).mean(), n_errors)
        error_features[:, i] = base + noise
    
    # Add realistic overlap (20% of samples have flipped characteristics)
    flip_mask = np.random.random(n_correct) < 0.2
    correct_features[flip_mask] = error_features[:flip_mask.sum()]
    flip_mask2 = np.random.random(n_errors) < 0.15
    error_features[flip_mask2] = correct_features[:flip_mask2.sum()]
    
    # --- NLI features (3 features) ---
    # Simulate NLI model outputs: entailment, contradiction, neutral scores
    
    # Correct outputs: high entailment, low contradiction
    correct_nli = np.zeros((n_correct, 3))
    correct_nli[:, 0] = np.random.beta(8, 2, n_correct)  # entailment (high)
    correct_nli[:, 1] = np.random.beta(2, 8, n_correct)  # contradiction (low)
    correct_nli[:, 2] = 1 - correct_nli[:, 0] - correct_nli[:, 1]  # neutral
    correct_nli[:, 2] = np.clip(correct_nli[:, 2], 0, 1)
    
    # Incorrect outputs: low entailment, high contradiction
    error_nli = np.zeros((n_errors, 3))
    error_nli[:, 0] = np.random.beta(3, 5, n_errors)     # entailment (lower)
    error_nli[:, 1] = np.random.beta(5, 3, n_errors)     # contradiction (higher)
    error_nli[:, 2] = 1 - error_nli[:, 0] - error_nli[:, 1]
    error_nli[:, 2] = np.clip(error_nli[:, 2], 0, 1)
    
    # Add overlap for NLI too
    flip_nli = np.random.random(n_correct) < 0.15
    correct_nli[flip_nli] = error_nli[:flip_nli.sum()]
    flip_nli2 = np.random.random(n_errors) < 0.1
    error_nli[flip_nli2] = correct_nli[:flip_nli2.sum()]
    
    # --- Combine ---
    X_logit = np.vstack([correct_features, error_features])
    X_nli = np.vstack([correct_nli, error_nli])
    X_hybrid = np.hstack([X_logit, X_nli])
    y = np.concatenate([np.zeros(n_correct), np.ones(n_errors)])
    
    # Shuffle
    perm = np.random.permutation(len(y))
    return X_logit[perm], X_nli[perm], X_hybrid[perm], y[perm]

X_logit, X_nli, X_hybrid, y = generate_synthetic_data(100000)
print(f"Generated {len(y)} samples")
print(f"  Logit features: {X_logit.shape[1]}")
print(f"  NLI features: {X_nli.shape[1]}")
print(f"  Hybrid features: {X_hybrid.shape[1]}")
print(f"  Balance: {(y==0).sum()} correct, {(y==1).sum()} errors")

# ============================================================
# STEP 2: Evaluation Framework
# ============================================================
def full_eval(y_true, y_pred, y_prob, name):
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
    
    print(f"  {name}: AUC={auc:.4f} PR-AUC={pr_auc:.4f} F1={f1:.4f} P={p:.4f} R={r:.4f}")
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
            "f1": float(f1), "auc_roc": float(auc), "pr_auc": float(pr_auc)}

# ============================================================
# STEP 3: Test All Combinations
# ============================================================
print(f"\n{'='*70}")
print("STEP 3: Testing Logit vs NLI vs Hybrid")
print(f"{'='*70}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Test A: Logit features only (24)
print(f"\n--- A: Logit Features Only (24) ---")
X_logit_norm = StandardScaler().fit_transform(X_logit)
fold_results = []
for fold, (tr, val) in enumerate(skf.split(X_logit_norm, y)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_logit_norm[tr], y[tr])
    prob = clf.predict_proba(X_logit_norm[val])[:, 1]
    pred = (prob > 0.5).astype(float)
    r = full_eval(y[val], pred, prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"  AVERAGE: AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f} F1={avg['f1']:.4f}")
results['logit_only'] = avg

# Test B: NLI features only (3)
print(f"\n--- B: NLI Features Only (3) ---")
X_nli_norm = StandardScaler().fit_transform(X_nli)
fold_results = []
for fold, (tr, val) in enumerate(skf.split(X_nli_norm, y)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_nli_norm[tr], y[tr])
    prob = clf.predict_proba(X_nli_norm[val])[:, 1]
    pred = (prob > 0.5).astype(float)
    r = full_eval(y[val], pred, prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"  AVERAGE: AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f} F1={avg['f1']:.4f}")
results['nli_only'] = avg

# Test C: Hybrid (logit + NLI = 27 features)
print(f"\n--- C: HYBRID (Logit + NLI = 27 features) ---")
X_hybrid_norm = StandardScaler().fit_transform(X_hybrid)
fold_results = []
for fold, (tr, val) in enumerate(skf.split(X_hybrid_norm, y)):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_hybrid_norm[tr], y[tr])
    prob = clf.predict_proba(X_hybrid_norm[val])[:, 1]
    pred = (prob > 0.5).astype(float)
    r = full_eval(y[val], pred, prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"  AVERAGE: AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f} F1={avg['f1']:.4f}")
results['hybrid'] = avg

# Test D: Hybrid with GradientBoosting (non-linear)
print(f"\n--- D: HYBRID with GradientBoosting ---")
fold_results = []
for fold, (tr, val) in enumerate(skf.split(X_hybrid, y)):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X_hybrid[tr], y[tr])
    prob = clf.predict_proba(X_hybrid[val])[:, 1]
    pred = (prob > 0.5).astype(float)
    r = full_eval(y[val], pred, prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"  AVERAGE: AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f} F1={avg['f1']:.4f}")
results['hybrid_gb'] = avg

# Test E: NLI-only GradientBoosting
print(f"\n--- E: NLI Features with GradientBoosting ---")
fold_results = []
for fold, (tr, val) in enumerate(skf.split(X_nli, y)):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_nli[tr], y[tr])
    prob = clf.predict_proba(X_nli[val])[:, 1]
    pred = (prob > 0.5).astype(float)
    r = full_eval(y[val], pred, prob, f"Fold {fold+1}")
    fold_results.append(r)
avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
print(f"  AVERAGE: AUC={avg['auc_roc']:.4f} PR-AUC={avg['pr_auc']:.4f} F1={avg['f1']:.4f}")
results['nli_gb'] = avg

# ============================================================
# STEP 4: Cross-Domain Test
# ============================================================
print(f"\n{'='*70}")
print("STEP 4: Cross-Domain Test")
print(f"{'='*70}")
print("Simulating: Train on domain A (QA), test on domain B (summarization)")

# Create domain-shifted data (simulating summarization characteristics)
np.random.seed(123)
n_domain_b = 20000

# Domain B has different distributions (simulating summarization)
X_logit_b = np.zeros((n_domain_b, 24))
X_logit_b[:, 0] = np.random.normal(3.5, 1.2, n_domain_b).clip(0, 10)  # different entropy
X_logit_b[:, 1] = np.random.normal(0.55, 0.18, n_domain_b).clip(0, 1)
X_logit_b[:, 2] = np.random.normal(0.85, 0.10, n_domain_b).clip(0, 1)
for i in range(3, 24):
    X_logit_b[:, i] = np.random.normal(X_logit_b[:, i % 3].mean(), 1.0, n_domain_b)

# NLI features stay more consistent across domains (semantic signal)
X_nli_b = np.zeros((n_domain_b, 3))
X_nli_b[:, 0] = np.random.beta(5, 3, n_domain_b)  # entailment
X_nli_b[:, 1] = np.random.beta(3, 5, n_domain_b)  # contradiction
X_nli_b[:, 2] = 1 - X_nli_b[:, 0] - X_nli_b[:, 1]
X_nli_b[:, 2] = np.clip(X_nli_b[:, 2], 0, 1)

y_b = np.concatenate([np.zeros(n_domain_b // 2), np.ones(n_domain_b // 2)])

X_hybrid_b = np.hstack([X_logit_b, X_nli_b])

# Train on domain A, test on domain B
print("\n  Train: Domain A (QA-like) → Test: Domain B (Summarization-like)")

# Logit-only cross-domain
clf_logit = LogisticRegression(max_iter=1000)
clf_logit.fit(X_logit_norm, y)
prob_b = clf_logit.predict_proba(StandardScaler().fit_transform(X_logit_b))[:, 1]
r = full_eval(y_b, (prob_b > 0.5).astype(float), prob_b, "Logit-only cross-domain")
results['logit_cross_domain'] = r

# NLI-only cross-domain
clf_nli = LogisticRegression(max_iter=1000)
clf_nli.fit(X_nli_norm, y)
prob_b = clf_nli.predict_proba(StandardScaler().fit_transform(X_nli_b))[:, 1]
r = full_eval(y_b, (prob_b > 0.5).astype(float), prob_b, "NLI-only cross-domain")
results['nli_cross_domain'] = r

# Hybrid cross-domain
clf_hybrid = LogisticRegression(max_iter=1000)
clf_hybrid.fit(X_hybrid_norm, y)
prob_b = clf_hybrid.predict_proba(StandardScaler().fit_transform(X_hybrid_b))[:, 1]
r = full_eval(y_b, (prob_b > 0.5).astype(float), prob_b, "Hybrid cross-domain")
results['hybrid_cross_domain'] = r

# ============================================================
# STEP 5: Scale Test (does more data help?)
# ============================================================
print(f"\n{'='*70}")
print("STEP 5: Scale Test — Does More Data Help?")
print(f"{'='*70}")

for n in [1000, 5000, 10000, 50000, 100000]:
    Xl, Xn, Xh, ys = generate_synthetic_data(n, seed=n)
    Xh_norm = StandardScaler().fit_transform(Xh)
    
    # Quick 2-fold test
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    aucs = []
    for tr, val in sss.split(Xh_norm, ys):
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xh_norm[tr], ys[tr])
        prob = clf.predict_proba(Xh_norm[val])[:, 1]
        auc = roc_auc_score(ys[val], prob)
        aucs.append(auc)
    print(f"  n={n:>7d}: Hybrid AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    results[f'scale_{n}'] = {"n_samples": n, "auc": float(np.mean(aucs))}

# ============================================================
# FINAL RESULTS
# ============================================================
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"\n{'Method':<30} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1':>8}")
print("-" * 54)
for name, r in results.items():
    if isinstance(r, dict) and 'auc_roc' in r:
        print(f"{name:<30} {r['auc_roc']:>8.4f} {r['pr_auc']:>8.4f} {r['f1']:>8.4f}")
    elif isinstance(r, dict) and 'auc' in r:
        print(f"{name:<30} {r['auc']:>8.4f} {'N/A':>8} {'N/A':>8}")

# Key findings
print(f"\n--- KEY FINDINGS ---")
logit_auc = results['logit_only']['auc_roc']
nli_auc = results['nli_only']['auc_roc']
hybrid_auc = results['hybrid']['auc_roc']

print(f"1. Logit features alone:     AUC = {logit_auc:.4f}")
print(f"2. NLI features alone:       AUC = {nli_auc:.4f}")
print(f"3. HYBRID (logit + NLI):     AUC = {hybrid_auc:.4f}")
print(f"4. Hybrid vs NLI alone:      +{hybrid_auc - nli_auc:.4f} AUC")
print(f"5. Hybrid vs Logit alone:    +{hybrid_auc - logit_auc:.4f} AUC")

if hybrid_auc > nli_auc + 0.02:
    print(f"\n✅ HYBRID adds value over NLI alone — logit features contribute!")
    print(f"   This is the novel finding: combining distributional + semantic signals")
elif hybrid_auc > nli_auc:
    print(f"\n⚠️  Hybrid marginally better than NLI — weak additive effect")
else:
    print(f"\n❌ Hybrid not better than NLI alone — logit features are redundant")

# Cross-domain
logit_cd = results['logit_cross_domain']['auc_roc']
nli_cd = results['nli_cross_domain']['auc_roc']
hybrid_cd = results['hybrid_cross_domain']['auc_roc']
print(f"\n--- CROSS-DOMAIN ---")
print(f"Logit-only cross-domain:   AUC = {logit_cd:.4f}")
print(f"NLI-only cross-domain:     AUC = {nli_cd:.4f}")
print(f"Hybrid cross-domain:       AUC = {hybrid_cd:.4f}")

if hybrid_cd > logit_cd + 0.05:
    print(f"✅ NLI features make cross-domain work — semantic signal transfers!")

# Save
os.makedirs('experiments', exist_ok=True)
with open('experiments/experiment_3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to experiments/experiment_3_results.json")
