"""
Ensemble NLI Experiment (Simplified)
====================================
Use already-cached sentence-transformers models for ensemble.

Since we have limited disk space, we'll use:
1. all-MiniLM-L6-v2 (already loaded, small)
2. nli-bert-base (already downloaded)

The ensemble approach: Get embeddings from both and compute NLI-style features.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import json
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import torch

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("ENSEMBLE NLI EXPERIMENT (Simplified)")
print("=" * 70)

# ============================================================
# Load Data & Models
# ============================================================
print("\n[1/5] Loading data...")

from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

MAX_SAMPLES = 500
samples = []
for i, row in enumerate(truthfulqa):
    if i >= MAX_SAMPLES:
        break
    question = row["question"]
    choices = row["mc2_targets"]["choices"]
    labels = row["mc2_targets"]["labels"]
    for j, (choice, label) in enumerate(zip(choices, labels)):
        samples.append({
            "question": question,
            "choice": choice,
            "label": label
        })

print(f"Loaded {len(samples)} samples")

# Load models (all should be cached)
print("\n[2/5] Loading NLI models...")

# Primary NLI model (MiniLM)
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
print("Loaded: MiniLM NLI")

# Sentence transformers for similarity
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded: MiniLM similarity")

# ============================================================
# Extract Features
# ============================================================
print("\n[3/5] Extracting features...")

def get_nli_features(question, choice, nli_model):
    """Get NLI features from MiniLM."""
    try:
        scores = nli_model.predict([(question, choice)])[0]
        probs = torch.softmax(torch.tensor(scores), dim=-1)
        entailment = float(probs[1])
        contradiction = float(probs[0])
        neutral = float(probs[2])
    except:
        entailment = contradiction = neutral = 0.333
    return [entailment, contradiction, neutral]

def get_semantic_features(question, choice, sim_model):
    """Get semantic similarity features."""
    try:
        embeddings = sim_model.encode([question, choice])
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Additional stats
        diff = embeddings[0] - embeddings[1]
        euclidean = float(np.linalg.norm(diff))
        
        # Manhattan
        manhattan = float(np.sum(np.abs(diff)))
    except:
        cos_sim = 0.5
        euclidean = 1.0
        manhattan = 1.0
    
    return [cos_sim, euclidean, manhattan]

# Process samples
MAX_PROCESS = 400
all_features = []
all_labels = []

print(f"Processing {MAX_PROCESS} samples...")

for i, sample in enumerate(tqdm(samples[:MAX_PROCESS])):
    q = sample["question"]
    choice = sample["choice"]
    label = sample["label"]
    
    # NLI features (3)
    nli_feats = get_nli_features(q, choice, nli_model)
    
    # Semantic features (3)
    sem_feats = get_semantic_features(q, choice, sim_model)
    
    # Combined: [semantic(3) + NLI(3)] = 6 features
    all_features.append(sem_feats + nli_feats)
    all_labels.append(label)

all_features = np.array(all_features)
all_labels = np.array(all_labels)

feature_names = ["cos_sim", "euclidean", "manhattan", "entailment", "contradiction", "neutral"]

print(f"Features shape: {all_features.shape}")
print(f"Labels: correct={np.sum(all_labels==0)}, hallucinated={np.sum(all_labels==1)}")

# ============================================================
# Train & Evaluate
# ============================================================
print("\n[4/5] Training detectors...")

X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# 1. Semantic Only (features 0-2)
X_train_sem = X_train[:, 0:3]
X_test_sem = X_test[:, 0:3]

clf_sem = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_sem.fit(X_train_sem, y_train)
y_prob_sem = clf_sem.predict_proba(X_test_sem)[:, 1]

try:
    auc_sem = roc_auc_score(y_test, y_prob_sem)
except:
    auc_sem = 0.5

print(f"\n--- Semantic Features Only ---")
print(f"AUC: {auc_sem:.3f}")

# 2. NLI Only (features 3-5)
X_train_nli = X_train[:, 3:6]
X_test_nli = X_test[:, 3:6]

clf_nli = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_nli.fit(X_train_nli, y_train)
y_prob_nli = clf_nli.predict_proba(X_test_nli)[:, 1]

try:
    auc_nli = roc_auc_score(y_test, y_prob_nli)
except:
    auc_nli = 0.5

print(f"\n--- NLI Features Only ---")
print(f"AUC: {auc_nli:.3f}")

# 3. Ensemble (average probabilities)
y_prob_ensemble = (y_prob_sem + y_prob_nli) / 2
y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)

acc_ens = accuracy_score(y_test, y_pred_ensemble)
prec_ens, rec_ens, f1_ens, _ = precision_recall_fscore_support(y_test, y_pred_ensemble, average='binary')
try:
    auc_ensemble = roc_auc_score(y_test, y_prob_ensemble)
except:
    auc_ensemble = 0.5

print(f"\n--- Ensemble (Semantic + NLI) ---")
print(f"Accuracy: {acc_ens:.3f}")
print(f"Precision: {prec_ens:.3f}")
print(f"Recall: {rec_ens:.3f}")
print(f"F1: {f1_ens:.3f}")
print(f"AUC: {auc_ensemble:.3f}")

# 4. Full Hybrid (all features)
clf_full = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_full.fit(X_train, y_train)
y_pred_full = clf_full.predict(X_test)
y_prob_full = clf_full.predict_proba(X_test)[:, 1]

acc_full = accuracy_score(y_test, y_pred_full)
prec_full, rec_full, f1_full, _ = precision_recall_fscore_support(y_test, y_pred_full, average='binary')
try:
    auc_full = roc_auc_score(y_test, y_prob_full)
except:
    auc_full = 0.5

print(f"\n--- Full Hybrid (All Features) ---")
print(f"Accuracy: {acc_full:.3f}")
print(f"Precision: {prec_full:.3f}")
print(f"Recall: {rec_full:.3f}")
print(f"F1: {f1_full:.3f}")
print(f"AUC: {auc_full:.3f}")

# Feature importance
print(f"\nFeature Importance (Full Hybrid):")
for name, imp in zip(feature_names, clf_full.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# ============================================================
# Save Results
# ============================================================
print("\n[5/5] Saving results...")

results = {
    "experiment": "Ensemble NLI (Simplified)",
    "max_samples": MAX_PROCESS,
    "semantic_only": {
        "auc": auc_sem
    },
    "nli_only": {
        "auc": auc_nli
    },
    "ensemble_prob_avg": {
        "accuracy": acc_ens,
        "precision": prec_ens,
        "recall": rec_ens,
        "f1": f1_ens,
        "auc": auc_ensemble
    },
    "full_hybrid": {
        "accuracy": acc_full,
        "precision": prec_full,
        "recall": rec_full,
        "f1": f1_full,
        "auc": auc_full
    },
    "feature_importance": dict(zip(feature_names, clf_full.feature_importances_.tolist())),
    "comparison": {
        "nli_only_ref": 0.716,
        "semantic_only": auc_sem,
        "nli_only": auc_nli,
        "ensemble": auc_ensemble,
        "full_hybrid": auc_full
    }
}

with open("experiments/ensemble_nli_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Semantic Features AUC: {auc_sem:.3f}")
print(f"NLI Features AUC: {auc_nli:.3f}")
print(f"Ensemble (avg) AUC: {auc_ensemble:.3f}")
print(f"Full Hybrid AUC: {auc_full:.3f}")
print(f"Previous best (NLI): 0.716")
print("=" * 70)

if auc_ensemble > max(auc_sem, auc_nli):
    print("✅ Ensemble improves over single method!")
else:
    print("❌ Ensemble does not improve")
