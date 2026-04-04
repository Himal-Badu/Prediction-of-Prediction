"""
Recall Prompt Experiment - 2k Samples (Fast Version)
=====================================================
Testing prompt variants at smaller scale to get results quickly.
"""

import os
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

HF_TOKEN = os.environ.get("HF_TOKEN", None)
print("HF_TOKEN:", "found" if HF_TOKEN else "not set")

print("=" * 70)
print("RECALL PROMPT EXPERIMENT - 2k SAMPLES")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================
MAX_SAMPLES = 2000  # Smaller for speed
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

PROMPT_VARIANTS = {
    "standard": lambda q: q,
    "verify": lambda q: f"Verify: {q}",
    "recall": lambda q: f"Recall facts: {q}",
}

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

all_samples = samples[:MAX_SAMPLES]
print(f"Loaded {len(all_samples)} samples")

# ============================================================
# Load NLI Model
# ============================================================
print("\n[2/5] Loading NLI model...")

from sentence_transformers import CrossEncoder

nli_model = CrossEncoder(NLI_MODEL, device="cpu", max_length=256)
print(f"Loaded: {NLI_MODEL}")

# ============================================================
# Extract Features (All variants)
# ============================================================
print("\n[3/5] Extracting features...")

def get_nli_features_batch(questions, choices, nli_model):
    pairs = [(q, c) for q, c in zip(questions, choices)]
    scores = nli_model.predict(pairs, show_progress_bar=False)
    probs = torch.softmax(torch.tensor(scores), dim=-1)
    return np.column_stack([probs[:, 1].numpy(), probs[:, 0].numpy(), probs[:, 2].numpy()])

questions_base = [s["question"] for s in all_samples]
choices_base = [s["choice"] for s in all_samples]
labels = np.array([s["label"] for s in all_samples])

features_dict = {}
for variant_name, prompt_fn in PROMPT_VARIANTS.items():
    print(f"  {variant_name}...", end=" ")
    questions_prompted = [prompt_fn(q) for q in questions_base]
    features_dict[variant_name] = get_nli_features_batch(questions_prompted, choices_base, nli_model)
    print("done")

# ============================================================
# Evaluate
# ============================================================
print("\n[4/5] Evaluating...")

results = {}
for variant_name, features in features_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    results[variant_name] = {"auc": round(auc, 4)}
    print(f"  {variant_name}: AUC={auc:.4f}")

# Ensemble
print("\n[5/5] Ensemble...")
all_combined = np.hstack(list(features_dict.values()))
X_train, X_test, y_train, y_test = train_test_split(all_combined, labels, test_size=0.2, random_state=42, stratify=labels)
clf_ens = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
clf_ens.fit(X_train, y_train)
y_prob_ens = clf_ens.predict_proba(X_test)[:, 1]
try:
    auc_ens = roc_auc_score(y_test, y_prob_ens)
except:
    auc_ens = 0.5

results["ensemble"] = {"auc": round(auc_ens, 4)}
print(f"  ensemble: AUC={auc_ens:.4f}")

# Save
output = {
    "experiment": "Recall Prompt - 2k Samples",
    "samples": MAX_SAMPLES,
    "results": results,
    "reference_10k": {"nli": 0.7445, "hybrid": 0.7537}
}

with open("experiments/recall_prompt_10k_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("REACTION PANEL")
print("=" * 70)
for m, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    print(f"  {m:12}: AUC = {r['auc']:.4f}")
print(f"\nReference (10k, experiment_3): NLI=0.7445, Hybrid=0.7537")
print("=" * 70)