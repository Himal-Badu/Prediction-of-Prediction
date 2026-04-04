"""
SPLICE HYBRID v5: Cross-domain with NLI only (skip slow GPT-2 on HaluEval).
"""
import torch, numpy as np, json, os, time, logging
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD TRUTHFULQA FEATURES
# ============================================================
log.info("Loading TruthfulQA features...")
X_raw = np.load(os.path.join(BASE, "..", "real_features_all.npy"))
y_tq = np.load(os.path.join(BASE, "..", "real_labels_all.npy"))
X_logit = X_raw[:, :10].astype(np.float32)
X_nli_tq = np.load(os.path.join(BASE, "splice_nli_features.npy"))
X_hybrid_tq = np.hstack([X_logit, X_nli_tq])
log.info(f"TruthfulQA: {len(y_tq)} samples, Logit={X_logit.shape}, NLI={X_nli_tq.shape}")

# ============================================================
# HALUEVAL NLI ONLY
# ============================================================
log.info("\nComputing HaluEval NLI...")
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")

ds_h = load_dataset("pminervini/HaluEval", "summarization", split="data")
ds_h = ds_h.select(range(min(500, len(ds_h))))

pairs_he = []
labels_he = []
for item in ds_h:
    doc = item["document"][-200:]
    pairs_he.append((doc, item["right_summary"][:150]))
    labels_he.append(0)
    pairs_he.append((doc, item["hallucinated_summary"][:150]))
    labels_he.append(1)

y_he = np.array(labels_he, dtype=np.float32)
log.info(f"HaluEval: {len(y_he)} pairs")

import gc
t0 = time.time()
all_scores = []
for i in range(0, len(pairs_he), 16):
    batch = pairs_he[i:i+16]
    scores = nli_model.predict(batch)
    all_scores.append(scores)
    if i % 256 == 0:
        gc.collect()
        log.info(f"  NLI: {i}/{len(pairs_he)} ({i/(time.time()-t0):.1f} pairs/s)")

all_scores = np.vstack(all_scores)
probs = torch.softmax(torch.tensor(all_scores), dim=-1).numpy()
X_nli_he = np.stack([probs[:, 1], probs[:, 0], probs[:, 2]], axis=1).astype(np.float32)
log.info(f"NLI done: {time.time()-t0:.1f}s ({len(pairs_he)/(time.time()-t0):.1f} pairs/s)")

# ============================================================
# TRAIN ON TRUTHFULQA, TEST ON HALUEVAL
# ============================================================
def eval_model(clf, Xtr, ytr, Xte, yte, name):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf.fit(Xtr_s, ytr)
    probs = clf.predict_proba(Xte_s)[:, 1]
    preds = (probs > 0.5).astype(float)
    m = {
        "auc_roc": round(float(roc_auc_score(yte, probs)), 4),
        "pr_auc": round(float(average_precision_score(yte, probs)), 4),
        "f1": round(float(f1_score(yte, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(yte, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, preds, zero_division=0)), 4),
    }
    log.info(f"  [{name}] AUC={m['auc_roc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
    return m

log.info("\n" + "="*60)
log.info("IN-DOMAIN (TruthfulQA, 80/20 split)")
log.info("="*60)

results = {}
for X, name in [(X_logit, "logit"), (X_nli_tq, "nli"), (X_hybrid_tq, "hybrid")]:
    for clf_name, clf in [("LR", LogisticRegression(max_iter=1000, class_weight='balanced')),
                           ("GBM", GradientBoostingClassifier(n_estimators=100, max_depth=3))]:
        key = f"{name}_{clf_name}"
        Xtr, Xte, ytr, yte = train_test_split(X, y_tq, test_size=0.2, random_state=42, stratify=y_tq)
        results[key] = eval_model(clf, Xtr, ytr, Xte, yte, key)

log.info("\n" + "="*60)
log.info("CROSS-DOMAIN (Train: TruthfulQA → Test: HaluEval)")
log.info("="*60)

for X_tq, X_he, name in [(X_logit, None, "logit"), 
                           (X_nli_tq, X_nli_he, "nli"), 
                           (X_hybrid_tq, None, "hybrid")]:
    if name == "logit":
        # Can't do cross-domain with logit-only (different GPT-2 runs)
        log.info(f"  [{name}] Skipped (no cross-domain logit features)")
        continue
    if name == "hybrid":
        # Use NLI-only for cross-domain (skip logit)
        X_he = X_nli_he  # Just test NLI transfer
        X_tq = X_nli_tq  # Train on NLI-only
        name = "hybrid_nli_only"
    
    for clf_name, clf in [("LR", LogisticRegression(max_iter=1000, class_weight='balanced')),
                           ("GBM", GradientBoostingClassifier(n_estimators=100, max_depth=3))]:
        key = f"cd_{name}_{clf_name}"
        results[key] = eval_model(clf, X_tq, y_tq, X_he, y_he, key)

# ============================================================
# ANALYSIS
# ============================================================
log.info("\n" + "="*60)
log.info("NLI FEATURE ANALYSIS")
log.info("="*60)

print("\n--- TruthfulQA NLI (error vs correct) ---")
for j, name in enumerate(["entailment", "contradiction", "neutral"]):
    err = float(X_nli_tq[y_tq==1, j].mean())
    cor = float(X_nli_tq[y_tq==0, j].mean())
    print(f"  {name:15s}: err={err:.4f} cor={cor:.4f} Δ={abs(err-cor):.4f}")

print("\n--- HaluEval NLI (hallucinated vs correct) ---")
for j, name in enumerate(["entailment", "contradiction", "neutral"]):
    hall = float(X_nli_he[y_he==1, j].mean())
    right = float(X_nli_he[y_he==0, j].mean())
    print(f"  {name:15s}: hall={hall:.4f} right={right:.4f} Δ={abs(hall-right):.4f}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*70)
print("SPLICE HYBRID v5 — FINAL RESULTS")
print("="*70)

print(f"\n{'Method':<25} {'Domain':<10} {'AUC':>7} {'F1':>7} {'P':>7} {'R':>7}")
print("-" * 60)
for key, r in results.items():
    domain = "HaluEval" if key.startswith("cd_") else "TruthfulQA"
    print(f"{key:<25} {domain:<10} {r['auc_roc']:>7.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

best_id = max([k for k in results if not k.startswith("cd_")], key=lambda k: results[k]["auc_roc"])
best_cd = max([k for k in results if k.startswith("cd_")], key=lambda k: results[k]["auc_roc"])
print(f"\nBest in-domain:    {best_id} (AUC={results[best_id]['auc_roc']:.4f})")
print(f"Best cross-domain: {best_cd} (AUC={results[best_cd]['auc_roc']:.4f})")

with open(os.path.join(BASE, "splice_hybrid_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved.")
print("\n✅ SPLICE HYBRID v5 COMPLETE")
