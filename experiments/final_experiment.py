"""
FINAL EXPERIMENT - Cross-Validate Best Method (NLI + CosSim + Length)
======================================================================
Get final, definitive results to push to GitHub
"""

import numpy as np
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import sys

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

np.random.seed(42)

print("=" * 60)
print("FINAL EXPERIMENT - DEFINITIVE RESULTS")
print("=" * 60)

# Load data - UNIQUE samples only
from datasets import load_dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")["validation"]

# Get unique QA pairs
samples = []
seen = set()
for row in truthfulqa:
    for choice, label in zip(row["mc2_targets"]["choices"], row["mc2_targets"]["labels"]):
        key = (row["question"], choice)
        if key not in seen:
            seen.add(key)
            samples.append({"question": row["question"], "choice": choice, "label": label})

MAX_SAMPLES = 800
samples = samples[:MAX_SAMPLES]
questions = [s["question"] for s in samples]
choices = [s["choice"] for s in samples]
labels = np.array([s["label"] for s in samples], dtype=np.int32)

print(f"Data: {len(samples)} unique samples")
print(f"Labels: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

# Load models
from sentence_transformers import CrossEncoder, SentenceTransformer
print("Loading models...")
nli = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu", max_length=128)
attn_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ============================================================
# EXTRACT ALL FEATURES
# ============================================================
print("\nExtracting features...")

# NLI features
pairs = [(q, c) for q, c in zip(questions, choices)]
scores = nli.predict(pairs, show_progress_bar=False, batch_size=32)
probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()

# Semantic similarity
q_emb = attn_model.encode(questions, show_progress_bar=False, batch_size=32)
c_emb = attn_model.encode(choices, show_progress_bar=False, batch_size=32)
cos_sim = np.sum(q_emb * c_emb, axis=1) / (np.linalg.norm(q_emb, axis=1) * np.linalg.norm(c_emb, axis=1) + 1e-10)

# Length features
q_len = np.array([len(q.split()) for q in questions])
c_len = np.array([len(c.split()) for c in choices])
len_ratio = c_len / (q_len + 1)

# ============================================================
# COMPARE METHODS
# ============================================================
print("\n" + "="*60)
print("COMPARING ALL METHODS")
print("="*60)

feature_sets = {
    "NLI_only": probs,
    "NLI_CosSim": np.column_stack([probs, cos_sim]),
    "NLI_Length": np.column_stack([probs, len_ratio, q_len, c_len]),
    "NLI_CosSim_Length": np.column_stack([probs, cos_sim, len_ratio, q_len, c_len]),
}

# Also test attention one more time to confirm
def get_attn(q, c, m):
    try:
        inp = m.tokenizer(q, c, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = m.model(**inp, output_attentions=True)
            attn = torch.mean(torch.stack(out.attentions), dim=0)[0,0].numpy()
        return [np.mean(attn), np.std(attn), np.max(attn)]
    except:
        return [0.5, 0.0, 0.5]

attn_feats = np.array([get_attn(q,c,attn_model) for q,c in zip(questions,choices)])
feature_sets["NLI_Attention"] = np.column_stack([probs, attn_feats])
feature_sets["All_Features"] = np.column_stack([probs, cos_sim, len_ratio, q_len, c_len, attn_feats])

results = {}

for name, X in feature_sets.items():
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in cv.split(X, labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        clf.fit(X_train_s, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
        cv_scores.append(auc)
    
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    results[name] = {"mean": mean_auc, "std": std_auc}
    print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")

# ============================================================
# BEST METHOD - DETAILED ANALYSIS
# ============================================================
print("\n" + "="*60)
print("BEST METHOD DETAILED ANALYSIS")
print("="*60)

best_method = "NLI_CosSim_Length"
X_best = feature_sets[best_method]

# Multiple random seeds for robustness
print("\nTesting with different random seeds...")
seed_results = []
for seed in [42, 0, 1, 123, 456, 789, 1000]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []
    
    for train_idx, test_idx in cv.split(X_best, labels):
        X_train, X_test = X_best[train_idx], X_best[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=seed)
        clf.fit(X_train_s, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
        fold_scores.append(auc)
    
    seed_results.append(np.mean(fold_scores))
    print(f"  Seed {seed}: {np.mean(fold_scores):.4f}")

print(f"\nFinal Mean: {np.mean(seed_results):.4f}")
print(f"Final Std: {np.std(seed_results):.4f}")
print(f"Range: {min(seed_results):.4f} - {max(seed_results):.4f}")

# ============================================================
# ADDITIONAL CLASSIFIERS
# ============================================================
print("\n" + "="*60)
print("TESTING DIFFERENT CLASSIFIERS")
print("="*60)

classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "GradientBoost": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}

clf_results = {}
for clf_name, clf in classifiers.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in cv.split(X_best, labels):
        X_train, X_test = X_best[train_idx], X_best[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf.fit(X_train_s, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
        cv_scores.append(auc)
    
    clf_results[clf_name] = np.mean(cv_scores)
    print(f"  {clf_name}: {np.mean(cv_scores):.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

best_name = max(results, key=lambda k: results[k]["mean"])
best_score = results[best_name]["mean"]

# Attention comparison
attn_score = results["NLI_Attention"]["mean"]
nli_score = results["NLI_only"]["mean"]

print(f"""
METHOD COMPARISON:
{'-'*40}""")
for name, res in sorted(results.items(), key=lambda x: -x[1]["mean"]):
    marker = " 👑 BEST" if name == best_name else ""
    attn_marker = " ⚠️" if "Attention" in name else ""
    print(f"  {name}: {res['mean']:.4f} (+/- {res['std']:.4f}){marker}{attn_marker}")

print(f"""
FINDINGS:
{'-'*40}
- Best method: {best_name} ({best_score:.4f} AUC)
- NLI only: {nli_score:.4f} AUC
- NLI + Attention: {attn_score:.4f} AUC
- Attention is {'BETTER' if attn_score > nli_score else 'WORSE'} than NLI only

ATTENTION VERDICT:
{'-'*40}
- NLI alone: {nli_score:.4f}
- NLI + Attention: {attn_score:.4f}
- Difference: {attn_score - nli_score:+.4f}
- Attention {'helps' if attn_score > nli_score else 'does NOT help'}
""")

# Final result for paper
final_result = {
    "best_method": best_name,
    "best_auc": round(best_score, 4),
    "cv_mean": round(np.mean(seed_results), 4),
    "cv_std": round(np.std(seed_results), 4),
    "range": f"{round(min(seed_results), 4)} - {round(max(seed_results), 4)}",
    "all_methods": {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)} for k, v in results.items()},
    "attention_verdict": "NOT USEFUL" if attn_score <= nli_score else "USEFUL"
}

with open("experiments/final_results.json", "w") as f:
    json.dump(final_result, f, indent=2)

print(f"""
FINAL RESULTS FOR PAPER:
{'-'*40}
Method: {best_name}
Average AUC: {np.mean(seed_results):.4f}
Variance: ±{np.std(seed_results):.4f}
Range: {min(seed_results):.4f} - {max(seed_results):.4f}
""")

print("✅ FINAL EXPERIMENT COMPLETE!")