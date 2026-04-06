"""
Experiment 2: Recall Prompt Method (FAST)
==========================================
Test 3 prompt variants + ensemble with 200 samples.
"""

import numpy as np
import json
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import torch

np.random.seed(42)
random.seed(42)

MAX_SAMPLES = 200

print("=" * 50)
print("EXPERIMENT 2: RECALL PROMPT (FAST)")
print("=" * 50)

# Load data
from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

samples = []
for row in truthfulqa:
    for choice, label in zip(row["mc2_targets"]["choices"], row["mc2_targets"]["labels"]):
        samples.append({"question": row["question"], "choice": choice, "label": label})

all_samples = samples[:MAX_SAMPLES]
questions = [s["question"] for s in all_samples]
choices = [s["choice"] for s in all_samples]
labels = np.array([s["label"] for s in all_samples])

print(f"Loaded {len(all_samples)} samples")

# Load model
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu", max_length=128)
print("✅ Model loaded")

def get_feats(qs, cs, m):
    pairs = [(q, c) for q, c in zip(qs, cs)]
    s = m.predict(pairs, show_progress_bar=False)
    p = torch.softmax(torch.tensor(s), dim=-1)
    return np.column_stack([p[:,1].numpy(), p[:,0].numpy(), p[:,2].numpy()])

# Test 3 prompts
PROMPTS = {
    "standard": lambda q: q,
    "verify": lambda q: f"Verify: {q}",
    "recall": lambda q: f"Recall: {q}",
}

results = {}
all_feats = []

for name, fn in PROMPTS.items():
    qs = [fn(q) for q in questions]
    f = get_feats(qs, choices, model)
    all_feats.append(f)
    
    X_tr, X_te, y_tr, y_te = train_test_split(f, labels, test_size=0.2, random_state=42, stratify=labels)
    clf = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    results[name] = {"auc": round(auc, 4)}
    print(f"  {name}: {auc:.4f}")

# Ensemble
ens = np.hstack(all_feats)
X_tr, X_te, y_tr, y_te = train_test_split(ens, labels, test_size=0.2, random_state=42, stratify=labels)
clf = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
clf.fit(X_tr, y_tr)
auc_ens = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
results["ensemble"] = {"auc": round(auc_ens, 4)}
print(f"  ensemble: {auc_ens:.4f}")

# Save
with open("experiments/recall_prompt_full_results.json", "w") as f:
    json.dump({"experiment": "Recall Prompt", "samples": MAX_SAMPLES, "results": results, "reference_10k": 0.7445}, f, indent=2)

print("\n" + "=" * 50)
print("REACTION PANEL")
print("=" * 50)
for m, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    print(f"  {m:10}: {r['auc']:.4f}")
print(f"Reference 10k: 0.7445")
print("✅ Done!")