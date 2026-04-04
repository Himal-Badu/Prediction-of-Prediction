"""
Semantic Similarity Experiment
==============================
Test if simple cosine similarity between question and answer can detect hallucinations.

Hypothesis: Correct answers are more semantically similar to the question than hallucinated ones.
Method: Compare embedding(question) vs embedding(answer) using sentence-transformers.
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

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("SEMANTIC SIMILARITY EXPERIMENT")
print("=" * 70)

# ============================================================
# Load Data & Models
# ============================================================
print("\n[1/4] Loading data...")

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

# Models
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")

print("Models loaded")

# ============================================================
# Extract Features
# ============================================================
print("\n[2/4] Extracting semantic similarity features...")

def get_similarity_features(question, choice, sim_model, nli_model):
    """Extract semantic similarity + NLI features."""
    
    # 1. Semantic similarity (question vs choice)
    try:
        embeddings = sim_model.encode([question, choice])
        q_emb, c_emb = embeddings[0], embeddings[1]
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity([q_emb], [c_emb])[0][0]
    except:
        cos_sim = 0.5
    
    # 2. NLI features (question -> choice)
    try:
        nli_scores = nli_model.predict([(question, choice)])[0]
        nli_probs = torch.softmax(torch.tensor(nli_scores), dim=-1)
        entailment = float(nli_probs[1])
        contradiction = float(nli_probs[0])
        neutral = float(nli_probs[2])
    except:
        entailment = contradiction = neutral = 0.333
    
    # Combined: [cos_sim, entailment, contradiction, neutral]
    return [cos_sim, entailment, contradiction, neutral]

import torch

# Process samples
MAX_PROCESS = 400
all_features = []
all_labels = []

print(f"Processing {MAX_PROCESS} samples...")

for i, sample in enumerate(tqdm(samples[:MAX_PROCESS])):
    q = sample["question"]
    choice = sample["choice"]
    label = sample["label"]
    
    feats = get_similarity_features(q, choice, sim_model, nli_model)
    all_features.append(feats)
    all_labels.append(label)

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"Features shape: {all_features.shape}")
print(f"Labels: correct={np.sum(all_labels==0)}, hallucinated={np.sum(all_labels==1)}")

# ============================================================
# Train & Evaluate
# ============================================================
print("\n[3/4] Training detectors...")

X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Semantic Similarity Only (feature 0)
X_train_sim = X_train[:, 0:1]
X_test_sim = X_test[:, 0:1]

clf_sim = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_sim.fit(X_train_sim, y_train)
y_pred_sim = clf_sim.predict(X_test_sim)
y_prob_sim = clf_sim.predict_proba(X_test_sim)[:, 1]

acc_sim = accuracy_score(y_test, y_pred_sim)
prec_sim, rec_sim, f1_sim, _ = precision_recall_fscore_support(y_test, y_pred_sim, average='binary')
try:
    auc_sim = roc_auc_score(y_test, y_prob_sim)
except:
    auc_sim = 0.5

print(f"\n--- SEMANTIC SIMILARITY ONLY ---")
print(f"Accuracy: {acc_sim:.3f}")
print(f"Precision: {prec_sim:.3f}")
print(f"Recall: {rec_sim:.3f}")
print(f"F1: {f1_sim:.3f}")
print(f"AUC: {auc_sim:.3f}")

# NLI Only (features 1-3)
X_train_nli = X_train[:, 1:4]
X_test_nli = X_test[:, 1:4]

clf_nli = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_nli.fit(X_train_nli, y_train)
y_pred_nli = clf_nli.predict(X_test_nli)
y_prob_nli = clf_nli.predict_proba(X_test_nli)[:, 1]

acc_nli = accuracy_score(y_test, y_pred_nli)
prec_nli, rec_nli, f1_nli, _ = precision_recall_fscore_support(y_test, y_pred_nli, average='binary')
try:
    auc_nli = roc_auc_score(y_test, y_prob_nli)
except:
    auc_nli = 0.5

print(f"\n--- NLI ONLY ---")
print(f"Accuracy: {acc_nli:.3f}")
print(f"Precision: {prec_nli:.3f}")
print(f"Recall: {rec_nli:.3f}")
print(f"F1: {f1_nli:.3f}")
print(f"AUC: {auc_nli:.3f}")

# Hybrid (Sim + NLI)
clf_hybrid = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_hybrid.fit(X_train, y_train)
y_pred_h = clf_hybrid.predict(X_test)
y_prob_h = clf_hybrid.predict_proba(X_test)[:, 1]

acc_h = accuracy_score(y_test, y_pred_h)
prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_test, y_pred_h, average='binary')
try:
    auc_h = roc_auc_score(y_test, y_prob_h)
except:
    auc_h = 0.5

print(f"\n--- HYBRID (Semantic Sim + NLI) ---")
print(f"Accuracy: {acc_h:.3f}")
print(f"Precision: {prec_h:.3f}")
print(f"Recall: {rec_h:.3f}")
print(f"F1: {f1_h:.3f}")
print(f"AUC: {auc_h:.3f}")

# Feature importance
feature_names = ["cos_sim", "entailment", "contradiction", "neutral"]
print(f"\nFeature Importance (Hybrid):")
for name, imp in zip(feature_names, clf_hybrid.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# ============================================================
# Save Results
# ============================================================
print("\n[4/4] Saving results...")

results = {
    "experiment": "Semantic Similarity",
    "max_samples": MAX_PROCESS,
    "semantic_similarity_only": {
        "accuracy": acc_sim,
        "precision": prec_sim,
        "recall": rec_sim,
        "f1": f1_sim,
        "auc": auc_sim
    },
    "nli_only": {
        "accuracy": acc_nli,
        "precision": prec_nli,
        "recall": rec_nli,
        "f1": f1_nli,
        "auc": auc_nli
    },
    "hybrid_sim_nli": {
        "accuracy": acc_h,
        "precision": prec_h,
        "recall": rec_h,
        "f1": f1_h,
        "auc": auc_h
    },
    "feature_importance": dict(zip(feature_names, clf_hybrid.feature_importances_.tolist())),
    "comparison": {
        "consistency_only": 0.526,
        "nli_only_ref": 0.716,
        "semantic_sim_only": auc_sim,
        "hybrid_sim_nli": auc_h
    }
}

with open("experiments/semantic_similarity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Semantic Similarity-only AUC: {auc_sim:.3f}")
print(f"NLI-only AUC: {auc_nli:.3f}")
print(f"Hybrid (Sim + NLI) AUC: {auc_h:.3f}")
print(f"Previous best (NLI): 0.716")
print("=" * 70)

if auc_sim > 0.55:
    print("✅ Semantic Similarity has signal!")
else:
    print("❌ Semantic Similarity alone not sufficient")
