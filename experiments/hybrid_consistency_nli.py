"""
Hybrid Consistency + NLI Experiment
===================================
Task 2: Combine consistency features with NLI (instead of logits)

Previous results:
- Consistency-only: AUC 0.526 (failed)
- NLI-only: AUC 0.716 (works!)
- Hybrid Consistency+Logits: AUC 0.526 (failed)

This test: Hybrid (Consistency + NLI)
Hypothesis: NLI carries the signal, consistency adds nothing. Let's confirm.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import json
import random
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("HYBRID: CONSISTENCY + NLI EXPERIMENT")
print("=" * 70)

# ============================================================
# Load Data & Models
# ============================================================
print("\n[1/4] Loading data and models...")

from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

MAX_SAMPLES = 300
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
from pop.core.llm_base import LLMBase
llm = LLMBase()
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")

# ============================================================
# Extract Features: Consistency + NLI
# ============================================================
print("\n[2/4] Extracting features...")

N_SAMPLES = 5

def get_consistency_nli_features(question, choice, llm, sim_model, nli_model, n_samples=5):
    """Extract consistency features + NLI features separately."""
    
    prompts = [
        f"Q: {question}\nA: {choice} is",
        f"Question: {question}\nAnswer: {choice}",
        f"{question} What is {choice}?",
        f"Based on: {question}\nIs {choice} correct?",
        f"Considering: {question}\nThe answer {choice}",
    ]
    
    answers = []
    for i in range(n_samples):
        try:
            if i == 0:
                result = llm.generate(prompts[i], max_new_tokens=20, temperature=0.0)
            elif i == 1:
                result = llm.generate(prompts[i], max_new_tokens=20, temperature=0.7)
            elif i == 2:
                result = llm.generate(prompts[i], max_new_tokens=20, top_k=50)
            elif i == 3:
                result = llm.generate(prompts[i], max_new_tokens=20, temperature=1.0)
            else:
                result = llm.generate(prompts[i], max_new_tokens=20, top_p=0.9)
            answers.append(result.strip())
        except:
            answers.append(choice)
    
    # Consistency features
    if len(answers) < 2:
        consistency_feats = [1.0, 0.0, 1.0, 1.0]
    else:
        embeddings = sim_model.encode(answers)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        n = len(answers)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(sim_matrix[i, j])
        
        similarities = np.array(similarities)
        mean_sim = float(np.mean(similarities))
        variance = float(np.var(similarities))
        min_sim = float(np.min(similarities))
        agreement_rate = float(np.mean(similarities > 0.8))
        consistency_feats = [mean_sim, variance, min_sim, agreement_rate]
    
    # NLI features (question -> choice)
    try:
        nli_scores = nli_model.predict([(question, choice)])[0]
        nli_probs = torch.softmax(torch.tensor(nli_scores), dim=-1)
        # [entailment, contradiction, neutral]
        nli_feats = [float(nli_probs[1]), float(nli_probs[0]), float(nli_probs[2])]
    except:
        nli_feats = [0.5, 0.5, 0.5]
    
    # Combined: [consistency(4) + NLI(3)] = 7 features
    return consistency_feats + nli_feats

# Process samples
MAX_PROCESS = 200
all_features = []
all_labels = []

print(f"Processing {MAX_PROCESS} samples...")

for i, sample in enumerate(tqdm(samples[:MAX_PROCESS])):
    q = sample["question"]
    choice = sample["choice"]
    label = sample["label"]
    
    feats = get_consistency_nli_features(q, choice, llm, sim_model, nli_model, n_samples=N_SAMPLES)
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

# Combined Hybrid
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

print(f"\n--- HYBRID (Consistency + NLI) ---")
print(f"Accuracy: {acc_h:.3f}")
print(f"Precision: {prec_h:.3f}")
print(f"Recall: {rec_h:.3f}")
print(f"F1: {f1_h:.3f}")
print(f"AUC: {auc_h:.3f}")

# NLI-only (features 4-6)
X_train_nli = X_train[:, 4:7]
X_test_nli = X_test[:, 4:7]

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

print(f"\n--- NLI-ONLY (for comparison) ---")
print(f"Accuracy: {acc_nli:.3f}")
print(f"Precision: {prec_nli:.3f}")
print(f"Recall: {rec_nli:.3f}")
print(f"F1: {f1_nli:.3f}")
print(f"AUC: {auc_nli:.3f}")

# Feature importance
feature_names = ["mean_sim", "variance", "min_sim", "agreement_rate", "entailment", "contradiction", "neutral"]
print(f"\nFeature Importance (Hybrid):")
for name, imp in zip(feature_names, clf_hybrid.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# ============================================================
# Save Results
# ============================================================
print("\n[4/4] Saving results...")

results = {
    "experiment": "Hybrid Consistency + NLI",
    "n_samples": N_SAMPLES,
    "max_samples": MAX_PROCESS,
    "hybrid_consistency_nli": {
        "accuracy": acc_h,
        "precision": prec_h,
        "recall": rec_h,
        "f1": f1_h,
        "auc": auc_h
    },
    "nli_only": {
        "accuracy": acc_nli,
        "precision": prec_nli,
        "recall": rec_nli,
        "f1": f1_nli,
        "auc": auc_nli
    },
    "feature_importance": dict(zip(feature_names, clf_hybrid.feature_importances_.tolist())),
    "comparison": {
        "consistency_only": 0.526,
        "nli_only": 0.716,
        "hybrid_consistency_logits": 0.526,
        "hybrid_consistency_nli": auc_h
    }
}

with open("experiments/hybrid_consistency_nli_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Consistency-only AUC: 0.526 (baseline)")
print(f"NLI-only AUC: 0.716 (reference)")
print(f"Hybrid Consistency+NLI AUC: {auc_h:.3f}")
print(f"Previous: Hybrid Consistency+Logits: 0.526")
print("=" * 70)

if auc_h > 0.55:
    print("✅ Signal detected!")
else:
    print("❌ Consistency adds nothing to NLI")
