"""
Multi-Sample Consistency Experiment
====================================
Test if running the same prompt multiple times and checking consistency detects hallucinations.

Hypothesis: Hallucinating models give inconsistent answers to same prompt.
Signal: Low agreement between multiple samples = likely hallucination.

Steps:
1. Load TruthfulQA data
2. For each question, generate multiple answers (5 samples)
3. Compute pairwise semantic similarity between answers
4. Create consistency features (mean similarity, variance, agreement rate)
5. Test consistency-only detector
6. Combine with NLI features (hybrid)
7. Compare results
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("MULTI-SAMPLE CONSISTENCY EXPERIMENT")
print("=" * 70)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1/6] Loading TruthfulQA data...")

from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

# Limit for speed
MAX_SAMPLES = 500
samples = []
for i, row in enumerate(truthfulqa):
    if i >= MAX_SAMPLES:
        break
    question = row["question"]
    # Use mc2_targets (simpler format)
    choices = row["mc2_targets"]["choices"]
    labels = row["mc2_targets"]["labels"]
    for j, (choice, label) in enumerate(zip(choices, labels)):
        samples.append({
            "question": question,
            "choice": choice,
            "label": label  # 1 = correct, 0 = hallucinated
        })

print(f"Loaded {len(samples)} question-choice pairs")

# ============================================================
# STEP 2: Load Models
# ============================================================
print("\n[2/6] Loading models...")

# LLM for generation (GPT-2 for speed)
from pop.core.llm_base import LLMBase
llm = LLMBase()
print("GPT-2 loaded")

# Sentence transformer for semantic similarity
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer loaded")

# NLI model for hybrid
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
print("NLI model loaded")

# ============================================================
# STEP 3: Multi-Sample Generation & Consistency Features
# ============================================================
print("\n[3/6] Generating multi-sample answers and computing consistency...")

N_SAMPLES = 5  # Generate 5 samples per prompt

def get_consistency_features(question, choice, llm, sim_model, n_samples=5):
    """
    Generate multiple answers and compute consistency features.
    
    Returns:
    - consistency_features: [mean_sim, variance, min_sim, agreement_rate, nli_entailment]
    """
    # For each question-choice pair, we check if answering consistently
    # Since GPT-2 is deterministic, we add temperature variation
    # and use top-k sampling for variation
    
    answers = []
    prompts = [
        f"Q: {question}\nA: {choice} is",  # Original
        f"Question: {question}\nAnswer: {choice}",
        f"{question} What is {choice}?",
        f"Based on: {question}\nIs {choice} correct?",
        f"Considering: {question}\nThe answer {choice}",
    ]
    
    for i in range(n_samples):
        try:
            # Vary generation parameters for diversity
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
        except Exception as e:
            answers.append(choice)  # Fallback
    
    # Compute pairwise similarity
    if len(answers) < 2:
        return [1.0, 0.0, 1.0, 1.0, 0.5]
    
    embeddings = sim_model.encode(answers)
    
    # Pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Get upper triangle (excluding diagonal)
    n = len(answers)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(sim_matrix[i, j])
    
    similarities = np.array(similarities)
    
    mean_sim = float(np.mean(similarities))
    variance = float(np.var(similarities))
    min_sim = float(np.min(similarities))
    
    # Agreement rate: how many pairs have similarity > 0.8
    agreement_rate = float(np.mean(similarities > 0.8))
    
    # Also compute NLI entailment (question -> answer)
    try:
        nli_scores = nli_model.predict([(question, answers[0])])[0]
        nli_probs = torch.softmax(torch.tensor(nli_scores), dim=-1)
        entailment = float(nli_probs[1])  # entailment score
    except:
        entailment = 0.5
    
    return [mean_sim, variance, min_sim, agreement_rate, entailment]

# Process samples (limited for speed)
CONSISTENCY_FEATURES = ["mean_sim", "variance", "min_sim", "agreement_rate", "nli_entailment"]

# Process a subset
MAX_PROCESS = 200  # Limit for speed
all_features = []
all_labels = []

print(f"Processing {MAX_PROCESS} samples with {N_SAMPLES}-sample consistency...")

for i, sample in enumerate(tqdm(samples[:MAX_PROCESS])):
    q = sample["question"]
    choice = sample["choice"]
    label = sample["label"]
    
    feats = get_consistency_features(q, choice, llm, sim_model, n_samples=N_SAMPLES)
    all_features.append(feats)
    all_labels.append(label)

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"Features shape: {all_features.shape}")
print(f"Label distribution: correct={np.sum(all_labels==0)}, hallucinated={np.sum(all_labels==1)}")

# ============================================================
# STEP 4: Train & Evaluate Consistency-Only Detector
# ============================================================
print("\n[4/6] Training consistency-only detector...")

X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train classifier
clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
try:
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = 0.5

print(f"\n--- CONSISTENCY-ONLY RESULTS ---")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1: {f1:.3f}")
print(f"AUC: {auc:.3f}")

# Feature importance
print(f"\nFeature Importance:")
for name, imp in zip(CONSISTENCY_FEATURES, clf.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# ============================================================
# STEP 5: Hybrid (Consistency + NLI + Logits)
# ============================================================
print("\n[5/6] Training hybrid detector...")

def extract_logit_features(logits, probs):
    """Extract 10 logit-based features."""
    probs = probs[0] if len(probs.shape) > 1 else probs
    
    # Basic stats
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    top1 = torch.max(probs).item()
    top5 = torch.sum(torch.topk(probs, min(5, len(probs))).values).item()
    
    # Logit stats
    logit = torch.log(probs + 1e-8)
    max_logit = torch.max(logit).item()
    mean_logit = torch.mean(logit).item()
    std_logit = torch.std(logit).item()
    
    # Margins
    sorted_logit, _ = torch.sort(logit, descending=True)
    margin = (sorted_logit[0] - sorted_logit[1]).item() if len(sorted_logit) > 1 else 0
    
    # Gini coefficient
    sorted_probs, _ = torch.sort(probs, descending=True)
    n = len(sorted_probs)
    gini = (2 * torch.sum(torch.arange(1, n+1) * sorted_probs).item() - (n + 1)) / (n * torch.sum(sorted_probs).item())
    
    # Concentration (Herfindahl)
    concentration = torch.sum(probs ** 2).item()
    
    return [entropy, top1, top5, margin, gini, max_logit, mean_logit, std_logit, concentration]

# Re-extract with logit features
print("Extracting logit features for hybrid...")

hybrid_features = []
for i, sample in enumerate(tqdm(samples[:MAX_PROCESS])):
    q = sample["question"]
    choice = sample["choice"]
    
    # Get logit features
    try:
        _, logits, probs = llm.generate(q, return_logits=True)
        logit_feats = extract_logit_features(logits, probs)
    except:
        logit_feats = [0.5] * 9
    
    # Consistency features (already extracted)
    consistency_feats = all_features[i].tolist()
    
    # Combine: [consistency(5) + logit(9)] = 14 features
    hybrid_feats = consistency_feats + logit_feats
    hybrid_features.append(hybrid_feats)

hybrid_features = np.array(hybrid_features)

# Train hybrid
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    hybrid_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

clf_hybrid = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_hybrid.fit(X_train_h, y_train_h)

y_pred_h = clf_hybrid.predict(X_test_h)
y_prob_h = clf_hybrid.predict_proba(X_test_h)[:, 1]

acc_h = accuracy_score(y_test_h, y_pred_h)
prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_test_h, y_pred_h, average='binary')
try:
    auc_h = roc_auc_score(y_test_h, y_prob_h)
except:
    auc_h = 0.5

print(f"\n--- HYBRID (Consistency + Logits) RESULTS ---")
print(f"Accuracy: {acc_h:.3f}")
print(f"Precision: {prec_h:.3f}")
print(f"Recall: {rec_h:.3f}")
print(f"F1: {f1_h:.3f}")
print(f"AUC: {auc_h:.3f}")

# ============================================================
# STEP 6: Compare and Save Results
# ============================================================
print("\n[6/6] Saving results...")

results = {
    "experiment": "Multi-Sample Consistency",
    "n_samples": N_SAMPLES,
    "max_samples": MAX_PROCESS,
    "consistency_only": {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "features": CONSISTENCY_FEATURES
    },
    "hybrid_consistency_logits": {
        "accuracy": acc_h,
        "precision": prec_h,
        "recall": rec_h,
        "f1": f1_h,
        "auc": auc_h,
        "n_features": 14
    },
    "feature_importance": {
        "consistency": dict(zip(CONSISTENCY_FEATURES, clf.feature_importances_.tolist())),
        "hybrid": dict(zip(
            CONSISTENCY_FEATURES + ["logit_" + str(i) for i in range(9)],
            clf_hybrid.feature_importances_.tolist()
        ))
    }
}

with open("experiments/consistency_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Consistency-only AUC: {auc:.3f}")
print(f"Hybrid (Consistency + Logits) AUC: {auc_h:.3f}")
print(f"Previous NLI-only AUC: 0.716 (reference)")
print("=" * 70)

if auc > 0.55:
    print("✅ CONSISTENCY SIGNAL DETECTED!")
else:
    print("❌ Consistency alone not sufficient")

print(f"\nResults saved to experiments/consistency_results.json")
