"""
VALIDATION EXPERIMENT - Verify 77% result is real
=================================================
Test on multiple different sample sets and validate consistency
"""

import numpy as np
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import sys

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

np.random.seed(42)

print("=" * 60)
print("VALIDATION EXPERIMENT - Verify 77% is Real")
print("=" * 60)

# Load data
from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

samples = []
for row in truthfulqa:
    for choice, label in zip(row["mc2_targets"]["choices"], row["mc2_targets"]["labels"]):
        samples.append({"question": row["question"], "choice": choice, "label": label})

# Load models
from sentence_transformers import CrossEncoder, SentenceTransformer
print("Loading models...")
nli = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu", max_length=128)
attn_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ============================================================
# TEST 1: Different sample sizes (800, 1000, 1200)
# ============================================================
print("\n" + "="*60)
print("TEST 1: Different Sample Sizes")
print("="*60)

validation_results = {}

for max_samples in [800, 1000, 1200]:
    if max_samples > len(samples):
        continue
    print(f"\n--- Testing {max_samples} samples ---")
    
    test_samples = samples[:max_samples]
    questions = [s["question"] for s in test_samples]
    choices = [s["choice"] for s in test_samples]
    labels = np.array([s["label"] for s in test_samples], dtype=np.int32)
    
    # Extract features
    pairs = [(q, c) for q, c in zip(questions, choices)]
    scores = nli.predict(pairs, show_progress_bar=False, batch_size=32)
    probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()
    
    # Semantic similarity
    q_emb = attn_model.encode(questions, show_progress_bar=False, batch_size=32)
    c_emb = attn_model.encode(choices, show_progress_bar=False, batch_size=32)
    cos_sim = np.sum(q_emb * c_emb, axis=1) / (np.linalg.norm(q_emb, axis=1) * np.linalg.norm(c_emb, axis=1) + 1e-10)
    
    # Length
    q_len = np.array([len(q.split()) for q in questions])
    c_len = np.array([len(c.split()) for c in choices])
    len_ratio = c_len / (q_len + 1)
    
    # Combined features
    X = np.column_stack([probs, cos_sim, len_ratio, q_len, c_len])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    
    validation_results[f"size_{max_samples}"] = auc
    print(f"  {max_samples} samples: AUC = {auc:.4f}")

# ============================================================
# TEST 2: Different random splits
# ============================================================
print("\n" + "="*60)
print("TEST 2: Different Random Splits")
print("="*60)

MAX_SAMPLES = 1000
test_samples = samples[:MAX_SAMPLES]
questions = [s["question"] for s in test_samples]
choices = [s["choice"] for s in test_samples]
labels = np.array([s["label"] for s in test_samples], dtype=np.int32)

# Extract features (same as before)
pairs = [(q, c) for q, c in zip(questions, choices)]
scores = nli.predict(pairs, show_progress_bar=False, batch_size=32)
probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()

q_emb = attn_model.encode(questions, show_progress_bar=False, batch_size=32)
c_emb = attn_model.encode(choices, show_progress_bar=False, batch_size=32)
cos_sim = np.sum(q_emb * c_emb, axis=1) / (np.linalg.norm(q_emb, axis=1) * np.linalg.norm(c_emb, axis=1) + 1e-10)

q_len = np.array([len(q.split()) for q in questions])
c_len = np.array([len(c.split()) for c in choices])
len_ratio = c_len / (q_len + 1)

X = np.column_stack([probs, cos_sim, len_ratio, q_len, c_len])

split_results = {}
for seed in [42, 10, 100, 200, 500, 1000, 1234, 5678]:
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=seed, stratify=labels)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    split_results[seed] = auc
    print(f"  Seed {seed}: AUC = {auc:.4f}")

print(f"  Mean: {np.mean(list(split_results.values())):.4f}")
print(f"  Std: {np.std(list(split_results.values())):.4f}")

# ============================================================
# TEST 3: Cross-validation (5-fold)
# ============================================================
print("\n" + "="*60)
print("TEST 3: 5-Fold Cross-Validation")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    cv_scores.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"  CV Mean: {np.mean(cv_scores):.4f}")
print(f"  CV Std: {np.std(cv_scores):.4f}")

# ============================================================
# TEST 4: Different test sizes
# ============================================================
print("\n" + "="*60)
print("TEST 4: Different Test Sizes")
print("="*60)

for test_size in [0.1, 0.15, 0.2, 0.25, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42, stratify=labels)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    print(f"  Test size {test_size}: AUC = {auc:.4f}")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "="*60)
print("VALIDATION VERDICT")
print("="*60)

all_aucs = list(split_results.values()) + cv_scores

summary = {
    "split_test_mean": round(np.mean(list(split_results.values())), 4),
    "split_test_std": round(np.std(list(split_results.values())), 4),
    "cv_mean": round(np.mean(cv_scores), 4),
    "cv_std": round(np.std(cv_scores), 4),
    "min_auc": round(min(all_aucs), 4),
    "max_auc": round(max(all_aucs), 4),
    "consistent_above_70": all(a >= 0.70 for a in all_aucs),
    "consistent_above_75": any(a >= 0.75 for a in all_aucs),
}

print(json.dumps(summary, indent=2))

# Save
with open("experiments/validation_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
if summary["consistent_above_70"]:
    print("✅ RESULT VALIDATED: Consistently above 70%")
    if summary["consistent_above_75"]:
        print("🎉 ALSO VALIDATED: Sometimes reaches 75%+")
else:
    print("⚠️ RESULT NOT CONSISTENT - needs more work")
print("="*60)