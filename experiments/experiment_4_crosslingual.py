"""
Experiment 4: Multi-NLI Model Ensemble + Hybrid
================================================
Test if combining different NLI models improves detection.
Uses already-cached models: MiniLM, RoBERTa, MPNet

Using 10k samples (sweet spot).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import json
import random
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import torch

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("EXPERIMENT 4: MULTI-NLI MODEL ENSEMBLE")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================
MAX_SAMPLES = 10000

# Already cached models
MODELS = {
    "minilm": "cross-encoder/nli-MiniLM2-L6-H768",
    "roberta": "cross-encoder/nli-roberta-base-v2",
    "mpnet": "cross-encoder/nli-mpnet-base-v2"
}

print(f"\n[Config] Samples: {MAX_SAMPLES}")
print(f"[Config] Models: {list(MODELS.keys())}")

# ============================================================
# Load Data
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
        samples.append({"question": question, "choice": choice, "label": label})

# Replicate to 10k
original_samples = samples.copy()
replications_needed = (MAX_SAMPLES // len(samples)) + 1
all_samples = []
for _ in range(replications_needed):
    all_samples.extend(original_samples)
all_samples = all_samples[:MAX_SAMPLES]

print(f"Loaded {len(all_samples)} samples")

# ============================================================
# Extract Features from All Models
# ============================================================
print("\n[2/5] Extracting features from all NLI models...")

from sentence_transformers import CrossEncoder

questions = [s["question"] for s in all_samples]
choices = [s["choice"] for s in all_samples]
labels = np.array([s["label"] for s in all_samples])

def get_nli_features(questions, choices, model):
    pairs = [(q, c) for q, c in zip(questions, choices)]
    scores = model.predict(pairs, show_progress_bar=False)
    probs = torch.softmax(torch.tensor(scores), dim=-1)
    # Assume: 0=contradiction, 1=entailment, 2=neutral
    return np.column_stack([
        probs[:, 1].numpy(),  # entailment
        probs[:, 0].numpy(),  # contradiction  
        probs[:, 2].numpy()   # neutral
    ])

features_dict = {}

for model_name, model_path in MODELS.items():
    print(f"  Loading {model_name}...", end=" ")
    try:
        model = CrossEncoder(model_path, device="cpu", max_length=256)
        features = get_nli_features(questions, choices, model)
        features_dict[model_name] = features
        print(f"✅ shape: {features.shape}")
    except Exception as e:
        print(f"❌ {e}")

print(f"\nLoaded {len(features_dict)} models")

# ============================================================
# Evaluate Each Model
# ============================================================
print("\n[3/5] Evaluating each NLI model...")

results = {}

for model_name, features in features_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    results[model_name] = {"auc": round(auc, 4)}
    print(f"  {model_name}: AUC = {auc:.4f}")

# ============================================================
# Hybrid (All Models Combined)
# ============================================================
print("\n[4/5] Testing hybrid (all models combined)...")

all_features = np.hstack(list(features_dict.values()))

X_train, X_test, y_train, y_test = train_test_split(
    all_features, labels, test_size=0.2, random_state=42, stratify=labels
)

clf_hybrid = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_hybrid.fit(X_train, y_train)
y_prob_hybrid = clf_hybrid.predict_proba(X_test)[:, 1]
y_pred_hybrid = clf_hybrid.predict(X_test)

try:
    auc_hybrid = roc_auc_score(y_test, y_prob_hybrid)
except:
    auc_hybrid = 0.5

acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
prec_hybrid, rec_hybrid, f1_hybrid, _ = precision_recall_fscore_support(y_test, y_pred_hybrid, average='binary')

results["hybrid_all"] = {
    "auc": round(auc_hybrid, 4),
    "accuracy": round(acc_hybrid, 4),
    "precision": round(prec_hybrid, 4),
    "recall": round(rec_hybrid, 4),
    "f1": round(f1_hybrid, 4)
}

print(f"  hybrid_all: AUC = {auc_hybrid:.4f}")

# ============================================================
# Feature Importance
# ============================================================
print("\n[5/5] Feature importance (top features):")
feature_names = []
for m in features_dict.keys():
    feature_names.extend([f"{m}_entail", f"{m}_contra", f"{m}_neutral"])

importance = clf_hybrid.feature_importances_
indices = np.argsort(importance)[::-1][:6]
for i in indices:
    print(f"  {feature_names[i]}: {importance[i]:.4f}")

# ============================================================
# Save Results
# ============================================================
output = {
    "experiment": "Multi-NLI Model Ensemble",
    "samples": MAX_SAMPLES,
    "models": list(MODELS.keys()),
    "results": results,
    "comparison": {
        "best_single": max([results[m]["auc"] for m in features_dict.keys()]),
        "hybrid": results["hybrid_all"]["auc"],
        "reference_experiment_3_nli": 0.7445,
        "reference_experiment_3_hybrid": 0.7537
    }
}

with open("experiments/experiment_4_crosslingual_results.json", "w") as f:
    json.dump(output, f, indent=2)

# ============================================================
# Reaction Panel
# ============================================================
print("\n" + "=" * 70)
print("REACTION PANEL - EXPERIMENT 4")
print("=" * 70)
print("\n| Method      |   AUC |")
print("|-------------|-------|")
for method, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    print(f"| {method:11} | {res['auc']:.4f} |")

print(f"\n📊 Reference (experiment_3): NLI=0.7445, Hybrid=0.7537")
print(f"📈 This run - Best Single: {max([results[m]['auc'] for m in features_dict.keys()]):.4f}")
print(f"📈 This run - Hybrid: {results['hybrid_all']['auc']:.4f}")

if results["hybrid_all"]["auc"] > max([results[m]["auc"] for m in features_dict.keys()]):
    print("\n✅ Hybrid improves over single models!")
else:
    print("\nℹ️ Single model competitive with hybrid")

print("=" * 70)
print("✅ Experiment 4 Complete!")