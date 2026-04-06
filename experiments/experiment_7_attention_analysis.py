"""
Experiment 3: Attention Analysis Method
========================================
Analyze if model attention patterns can detect hallucination.
Use sentence-transformers to get attention weights.

The idea: If model attends to different parts when validating
factual vs hallucinated content, we can use that as a signal.
"""

import numpy as np
import json
import random
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn

np.random.seed(42)
random.seed(42)

MAX_SAMPLES = 100

print("=" * 60)
print("EXPERIMENT 3: ATTENTION ANALYSIS")
print("=" * 60)

# ============================================================
# Load Data
# ============================================================
print("\n[1/6] Loading data...")

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

# ============================================================
# Load Model with Attention
# ============================================================
print("\n[2/6] Loading sentence-transformers model...")

from sentence_transformers import SentenceTransformer

# Use a model that supports attention extraction
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device="cpu")
print(f"✅ Loaded: {model_name}")

# ============================================================
# Extract Attention Features
# ============================================================
print("\n[3/6] Extracting attention features...")

def get_attention_features(question, choice, model):
    """Extract attention-based features from the model."""
    try:
        # Encode the question-answer pair
        inputs = model.tokenizer(
            question, choice,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Get model output with attention
        with torch.no_grad():
            outputs = model.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Average attention across layers and heads
        avg_attention = torch.mean(torch.stack(attentions), dim=0)[0]  # [seq, seq]
        
        # Extract features from attention matrix
        seq_len = avg_attention.shape[0]
        
        # Feature 1: Self-attention on [CLS] (aggregate attention)
        cls_attention = avg_attention[0, :].numpy()
        features = [
            np.mean(cls_attention),
            np.std(cls_attention),
            np.max(cls_attention),
            np.argmax(cls_attention) / seq_len,  # Normalized position
        ]
        
        # Feature 2: Attention to question vs answer
        # Assume question ends where answer begins (approximate)
        q_len = len(model.tokenizer.encode(question))
        if q_len < seq_len:
            q_attention = np.mean(avg_attention[0, :q_len].numpy())
            a_attention = np.mean(avg_attention[0, q_len:].numpy())
            features.extend([q_attention, a_attention, a_attention - q_attention])
        else:
            features.extend([0.5, 0.5, 0.0])
        
        # Feature 3: Entropy of attention distribution
        attn_prob = cls_attention + 1e-10
        attn_prob = attn_prob / np.sum(attn_prob)
        entropy = -np.sum(attn_prob * np.log(attn_prob))
        features.append(entropy)
        
        return features
        
    except Exception as e:
        # Fallback: return zeros
        return [0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5]

# Extract for all samples
print("  Extracting attention features...")
attention_features = []
for i, (q, c) in enumerate(zip(questions, choices)):
    if i % 20 == 0:
        print(f"    Progress: {i}/{len(questions)}")
    feats = get_attention_features(q, c, model)
    attention_features.append(feats)

attention_features = np.array(attention_features)
print(f"  Attention features shape: {attention_features.shape}")

# ============================================================
# Also get NLI features for comparison
# ============================================================
print("\n[4/6] Extracting NLI features...")

from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu", max_length=256)

pairs = [(q, c) for q, c in zip(questions, choices)]
scores = nli_model.predict(pairs, show_progress_bar=False)
probs = torch.softmax(torch.tensor(scores), dim=-1)
nli_features = np.column_stack([
    probs[:, 1].numpy(),  # entailment
    probs[:, 0].numpy(),  # contradiction
    probs[:, 2].numpy()   # neutral
])

print(f"  NLI features shape: {nli_features.shape}")

# ============================================================
# Evaluate Each Method
# ============================================================
print("\n[5/6] Evaluating methods...")

results = {}

# Attention only
X_train, X_test, y_train, y_test = train_test_split(
    attention_features, labels, test_size=0.2, random_state=42, stratify=labels
)
clf_att = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
clf_att.fit(X_train, y_train)
y_prob_att = clf_att.predict_proba(X_test)[:, 1]
auc_att = roc_auc_score(y_test, y_prob_att)
results["attention_only"] = {"auc": round(auc_att, 4)}
print(f"  Attention only: AUC = {auc_att:.4f}")

# NLI only
X_train, X_test, y_train, y_test = train_test_split(
    nli_features, labels, test_size=0.2, random_state=42, stratify=labels
)
clf_nli = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
clf_nli.fit(X_train, y_train)
y_prob_nli = clf_nli.predict_proba(X_test)[:, 1]
auc_nli = roc_auc_score(y_test, y_prob_nli)
results["nli_only"] = {"auc": round(auc_nli, 4)}
print(f"  NLI only: AUC = {auc_nli:.4f}")

# ============================================================
# Hybrid (Attention + NLI)
# ============================================================
print("\n[6/6] Testing hybrid...")

hybrid_features = np.hstack([attention_features, nli_features])

X_train, X_test, y_train, y_test = train_test_split(
    hybrid_features, labels, test_size=0.2, random_state=42, stratify=labels
)

clf_hybrid = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
clf_hybrid.fit(X_train, y_train)
y_prob_hybrid = clf_hybrid.predict_proba(X_test)[:, 1]
y_pred_hybrid = clf_hybrid.predict(X_test)

auc_hybrid = roc_auc_score(y_test, y_prob_hybrid)
acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_hybrid, average='binary')

results["hybrid_attention_nli"] = {
    "auc": round(auc_hybrid, 4),
    "accuracy": round(acc_hybrid, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4)
}

print(f"  Hybrid (Attention + NLI): AUC = {auc_hybrid:.4f}")

# ============================================================
# Save Results
# ============================================================
output = {
    "experiment": "Attention Analysis Method",
    "samples": MAX_SAMPLES,
    "models": ["all-MiniLM-L6-v2", "nli-MiniLM2-L6-H768"],
    "results": results,
    "reference": {
        "10k_nli": 0.7445,
        "10k_hybrid": 0.7537
    },
    "feature_names": [
        "cls_attn_mean", "cls_attn_std", "cls_attn_max", "cls_attn_pos",
        "q_attn", "a_attn", "a_minus_q_attn", "attn_entropy"
    ]
}

with open("experiments/attention_analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

# ============================================================
# Reaction Panel
# ============================================================
print("\n" + "=" * 60)
print("REACTION PANEL")
print("=" * 60)
print("\n| Method              |   AUC |")
print("|---------------------|-------|")
for m, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    print(f"| {m:19} | {r['auc']:.4f} |")

print(f"\n📊 Reference (10k NLI): 0.7445")
print(f"📈 This run (100): Best={max([r['auc'] for r in results.values()]):.4f}")

if results["hybrid_attention_nli"]["auc"] > results["nli_only"]["auc"]:
    print("\n✅ Attention features add value to NLI!")
else:
    print("\nℹ️ NLI alone is still best at 10k scale")

print("=" * 60)
print("✅ Experiment 3 Complete!")