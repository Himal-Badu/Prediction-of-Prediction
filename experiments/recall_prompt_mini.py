"""
Recall Prompt Experiment (MINI VERSION - 1k samples)
====================================================
Fast test with small sample size.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import json
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import torch

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("RECALL PROMPT EXPERIMENT (1k samples)")
print("=" * 70)

# ============================================================
# Load Data (1k samples)
# ============================================================
print("\n[1/5] Loading data...")

from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

samples = []
for i, row in enumerate(truthfulqa):
    question = row["question"]
    choices = row["mc2_targets"]["choices"]
    labels = row["mc2_targets"]["labels"]
    for j, (choice, label) in enumerate(zip(choices, labels)):
        samples.append({
            "question": question,
            "choice": choice,
            "label": label
        })

# Use 1k samples
MAX_PROCESS = 1000
all_samples = samples[:MAX_PROCESS]

print(f"Loaded {len(all_samples)} samples")

# ============================================================
# Load NLI Model
# ============================================================
print("\n[2/5] Loading NLI model...")

from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu", max_length=512)
print("Loaded: MiniLM NLI")

# ============================================================
# Extract features for all prompt variants (batch)
# ============================================================
print("\n[3/5] Extracting features for all variants (batch processing)...")

def get_nli_features_batch(questions, choices, nli_model):
    """Batch NLI feature extraction."""
    pairs = [(q, c) for q, c in zip(questions, choices)]
    scores = nli_model.predict(pairs, show_progress_bar=False)
    probs = torch.softmax(torch.tensor(scores), dim=-1)
    entailment = probs[:, 1].numpy()
    contradiction = probs[:, 0].numpy()
    neutral = probs[:, 2].numpy()
    return np.column_stack([entailment, contradiction, neutral])

# Prepare questions/choices
questions = [s["question"] for s in all_samples]
choices = [s["choice"] for s in all_samples]
labels = np.array([s["label"] for s in all_samples])

# Standard
print("  Processing standard...")
features_standard = get_nli_features_batch(questions, choices, nli_model)

# Verify
print("  Processing verify...")
questions_verify = [f"Verify: {q}" for q in questions]
features_verify = get_nli_features_batch(questions_verify, choices, nli_model)

# Recall
print("  Processing recall...")
questions_recall = [f"Recall facts about: {q}" for q in questions]
features_recall = get_nli_features_batch(questions_recall, choices, nli_model)

print(f"Features shape: {features_standard.shape}")

# ============================================================
# Evaluate each variant
# ============================================================
print("\n[4/5] Evaluating variants...")

results = {}

for name, features in [("standard", features_standard), 
                       ("verify", features_verify),
                       ("recall", features_recall)]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    results[name] = {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    print(f"  {name}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

# ============================================================
# Ensemble all variants
# ============================================================
print("\n[5/5] Testing ensemble of all variants...")

# Combine all features (3 variants * 3 features = 9)
all_features_combined = np.hstack([features_standard, features_verify, features_recall])

X_train, X_test, y_train, y_test = train_test_split(
    all_features_combined, labels, test_size=0.2, random_state=42, stratify=labels
)

clf_ensemble = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_ensemble.fit(X_train, y_train)
y_prob_ensemble = clf_ensemble.predict_proba(X_test)[:, 1]
y_pred_ensemble = clf_ensemble.predict(X_test)

try:
    auc_ensemble = roc_auc_score(y_test, y_prob_ensemble)
except:
    auc_ensemble = 0.5

acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
prec_ensemble, rec_ensemble, f1_ensemble, _ = precision_recall_fscore_support(y_test, y_pred_ensemble, average='binary')

results["ensemble"] = {"auc": auc_ensemble, "accuracy": acc_ensemble, "precision": prec_ensemble, "recall": rec_ensemble, "f1": f1_ensemble}

print(f"  ensemble: AUC={auc_ensemble:.4f}, Acc={acc_ensemble:.4f}, F1={f1_ensemble:.4f}")

# ============================================================
# Save Results
# ============================================================
with open("experiments/recall_prompt_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY (1k samples)")
print("=" * 70)
for name, res in results.items():
    print(f"  {name}: AUC={res['auc']:.4f}, F1={res['f1']:.4f}")
print("=" * 70)

# Compare with baseline
best_single = max([results[k]["auc"] for k in ["standard", "verify", "recall"]])
if results["ensemble"]["auc"] > best_single:
    print(f"✅ Recall prompt improves: {results['ensemble']['auc']:.4f} > {best_single:.4f}")
else:
    print(f"❌ No improvement: {results['ensemble']['auc']:.4f} <= {best_single:.4f}")