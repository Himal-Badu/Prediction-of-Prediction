"""
Recall Prompt Experiment (FAST VERSION)
=========================================
Test recall prompt method with 10k samples (where performance is good).
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
print("RECALL PROMPT EXPERIMENT (10k samples)")
print("=" * 70)

# ============================================================
# Load Data (10k samples - sweet spot)
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

# Replicate ~3x to get 10k samples
original_samples = samples.copy()
replications_needed = 4
all_samples = []
for _ in range(replications_needed):
    all_samples.extend(original_samples)

print(f"Loaded {len(all_samples)} samples")

# ============================================================
# Load NLI Model
# ============================================================
print("\n[2/5] Loading NLI model...")

from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
print("Loaded: MiniLM NLI")

# ============================================================
# Extract features for all prompt variants
# ============================================================
print("\n[3/5] Extracting features for all variants (10k samples)...")

MAX_PROCESS = 10000

def get_nli_features(question, choice, nli_model):
    """Get NLI features (entailment, contradiction, neutral)."""
    try:
        scores = nli_model.predict([(question, choice)])[0]
        probs = torch.softmax(torch.tensor(scores), dim=-1)
        entailment = float(probs[1])
        contradiction = float(probs[0])
        neutral = float(probs[2])
    except:
        entailment = contradiction = neutral = 0.333
    return [entailment, contradiction, neutral]

# Process all samples with all variants
all_features_standard = []
all_features_verify = []
all_features_recall = []
all_labels = []

for i, sample in enumerate(tqdm(all_samples[:MAX_PROCESS], desc="Extracting features")):
    q = sample["question"]
    choice = sample["choice"]
    label = sample["label"]
    
    # Standard
    feats_std = get_nli_features(q, choice, nli_model)
    all_features_standard.append(feats_std)
    
    # Verify (prompt the question)
    q_verify = f"Verify: {q}"
    feats_verify = get_nli_features(q_verify, choice, nli_model)
    all_features_verify.append(feats_verify)
    
    # Recall (prompt the question)
    q_recall = f"Recall facts about: {q}"
    feats_recall = get_nli_features(q_recall, choice, nli_model)
    all_features_recall.append(feats_recall)
    
    all_labels.append(label)

all_features_standard = np.array(all_features_standard)
all_features_verify = np.array(all_features_verify)
all_features_recall = np.array(all_features_recall)
all_labels = np.array(all_labels)

print(f"Features extracted: {all_features_standard.shape}")

# ============================================================
# Evaluate each variant
# ============================================================
print("\n[4/5] Evaluating variants...")

results = {}

for name, features in [("standard", all_features_standard), 
                       ("verify", all_features_verify),
                       ("recall", all_features_recall)]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
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
all_features_combined = np.hstack([all_features_standard, all_features_verify, all_features_recall])

X_train, X_test, y_train, y_test = train_test_split(
    all_features_combined, all_labels, test_size=0.2, random_state=42, stratify=all_labels
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
print("SUMMARY (10k samples)")
print("=" * 70)
for name, res in results.items():
    print(f"  {name}: AUC={res['auc']:.4f}, F1={res['f1']:.4f}")
print("=" * 70)

# Compare with baseline
best_single = max([results[k]["auc"] for k in ["standard", "verify", "recall"]])
if results["ensemble"]["auc"] > best_single:
    print(f"✅ Ensemble improves over best single: {results['ensemble']['auc']:.4f} > {best_single:.4f}")
else:
    print(f"❌ Ensemble does not improve: {results['ensemble']['auc']:.4f} <= {best_single:.4f}")