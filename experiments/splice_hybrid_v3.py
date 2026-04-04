"""
SPLICE HYBRID v3: Use pre-extracted GPT-2 features + batched NLI.
No GPT-2 inference needed — just NLI on text pairs.
"""
import torch
import numpy as np
import json
import os
import time
import logging
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD PRE-EXTRACTED GPT-2 FEATURES
# ============================================================
log.info("Loading pre-extracted GPT-2 features...")
X_raw = np.load(os.path.join(BASE, "..", "real_features_all.npy"))
y = np.load(os.path.join(BASE, "..", "real_labels_all.npy"))
log.info(f"Features: {X_raw.shape}, Labels: {y.shape}")

# Use first 10 features (the original logit features, not augmented)
X_logit = X_raw[:, :10].astype(np.float32)
log.info(f"Logit features: {X_logit.shape}")

# ============================================================
# LOAD NLI MODEL
# ============================================================
log.info("Loading MiniLM-L6 NLI...")
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
log.info("MiniLM loaded")

def extract_batch_nli(pairs, batch_size=16):
    import gc
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        scores = nli_model.predict(batch)
        all_scores.append(scores)
        if i % 512 == 0:
            gc.collect()
    result = np.vstack(all_scores)
    probs = torch.softmax(torch.tensor(result), dim=-1).numpy()
    del all_scores, result
    gc.collect()
    # 0=contradiction, 1=entailment, 2=neutral
    return np.stack([probs[:, 1], probs[:, 0], probs[:, 2]], axis=1)

# ============================================================
# RECONSTRUCT QUESTION-ANSWER PAIRS FOR NLI
# ============================================================
log.info("Reconstructing QA pairs for NLI...")
ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

nli_pairs = []
nli_labels = []
for item in ds:
    q = item["question"]
    mc1 = item["mc1_targets"]
    if not mc1 or not mc1.get("choices"):
        continue
    for choice_text, label in zip(mc1["choices"], mc1["labels"]):
        nli_pairs.append((q[:200], choice_text[:200]))
        nli_labels.append(1 if label == 0 else 0)

log.info(f"QA pairs: {len(nli_pairs)}")

# Verify label alignment
nli_labels = np.array(nli_labels, dtype=np.float32)
assert len(nli_labels) == len(y), f"Label mismatch: {len(nli_labels)} vs {len(y)}"
assert np.allclose(nli_labels, y), "Labels don't match!"
log.info("✅ Labels aligned")

# ============================================================
# BATCH NLI INFERENCE
# ============================================================
log.info(f"Running batched NLI on {len(nli_pairs)} pairs...")
t0 = time.time()
X_nli = extract_batch_nli(nli_pairs)
nli_time = time.time() - t0
log.info(f"NLI done: {nli_time:.1f}s ({len(nli_pairs)/nli_time:.1f} pairs/s)")

X_nli = X_nli.astype(np.float32)
X_hybrid = np.hstack([X_logit, X_nli])

log.info(f"\nFeature shapes:")
log.info(f"  Logit: {X_logit.shape}")
log.info(f"  NLI:   {X_nli.shape}")
log.info(f"  Hybrid: {X_hybrid.shape}")

# Save
np.save(os.path.join(BASE, "splice_nli_features.npy"), X_nli)
np.save(os.path.join(BASE, "splice_hybrid_features.npy"), X_hybrid)

# ============================================================
# TRAIN & EVALUATE
# ============================================================
class Detector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_eval(X, y, name, epochs=300):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = Detector(X.shape[1])
    pw = torch.tensor([(y==0).sum() / max((y==1).sum(), 1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)
    Xtr_t, ytr_t = torch.FloatTensor(Xtr), torch.FloatTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    best_f1, best_state = 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr_t), ytr_t); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(Xte_t)).numpy() > 0.5).astype(float)
            vf1 = f1_score(yte, preds, zero_division=0)
        sched.step(1 - vf1)
        if vf1 > best_f1: best_f1 = vf1; best_state = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).numpy()
        preds = (probs > 0.5).astype(float)
    m = {
        "accuracy": round(float(accuracy_score(yte, preds)), 4),
        "precision": round(float(precision_score(yte, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, preds, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(yte, probs)), 4),
        "pr_auc": round(float(average_precision_score(yte, probs)), 4),
    }
    log.info(f"  [{name}] AUC={m['auc_roc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
    return m, model, scaler

log.info("\n" + "=" * 60)
log.info("TRAINING DETECTORS")
log.info("=" * 60)

results = {}
log.info("\n--- A: Logit-Only (10 feat) ---")
results["logit_only"], m_logit, s_logit = train_eval(X_logit, y, "Logit")
log.info("\n--- B: NLI-Only (3 feat) ---")
results["nli_only"], m_nli, s_nli = train_eval(X_nli, y, "NLI")
log.info("\n--- C: SPLICE HYBRID (13 feat) ---")
results["splice_hybrid"], m_hybrid, s_hybrid = train_eval(X_hybrid, y, "Hybrid")

# Also test with GBM for comparison
from sklearn.ensemble import GradientBoostingClassifier
log.info("\n--- D: Hybrid GBM ---")
scaler_h = StandardScaler()
Xh_s = scaler_h.fit_transform(X_hybrid)
Xtr, Xte, ytr, yte = train_test_split(Xh_s, y, test_size=0.2, random_state=42, stratify=y)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
gbm.fit(Xtr, ytr)
probs_gbm = gbm.predict_proba(Xte)[:, 1]
preds_gbm = (probs_gbm > 0.5).astype(float)
results["hybrid_gbm"] = {
    "accuracy": round(float(accuracy_score(yte, preds_gbm)), 4),
    "precision": round(float(precision_score(yte, preds_gbm, zero_division=0)), 4),
    "recall": round(float(recall_score(yte, preds_gbm, zero_division=0)), 4),
    "f1": round(float(f1_score(yte, preds_gbm, zero_division=0)), 4),
    "auc_roc": round(float(roc_auc_score(yte, probs_gbm)), 4),
    "pr_auc": round(float(average_precision_score(yte, probs_gbm)), 4),
}
r = results["hybrid_gbm"]
log.info(f"  [Hybrid GBM] AUC={r['auc_roc']:.4f} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

# Feature importance analysis
log.info("\n" + "=" * 60)
log.info("FEATURE ANALYSIS")
log.info("=" * 60)
feat_names = ["entropy","top1","top5","margin","gini","max_logit","mean_logit","std_logit","concentration","spread"]
print("\nLogit features (error vs correct):")
for j, name in enumerate(feat_names):
    err = X_logit[y==1, j].mean()
    cor = X_logit[y==0, j].mean()
    print(f"  {name:15s}: err={float(err):.4f} cor={float(cor):.4f} diff={abs(float(err-cor)):.4f}")
print("\nNLI features (error vs correct):")
for j, name in enumerate(["entailment","contradiction","neutral"]):
    err = X_nli[y==1, j].mean()
    cor = X_nli[y==0, j].mean()
    print(f"  {name:15s}: err={float(err):.4f} cor={float(cor):.4f} diff={abs(float(err-cor)):.4f}")

# FINAL REPORT
print("\n" + "=" * 60)
print("SPLICE HYBRID v3 — FINAL RESULTS")
print("=" * 60)
print(f"\n{'Method':<20} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'P':>7} {'R':>7}")
print("-" * 55)
for key, r in results.items():
    print(f"{key:<20} {r['auc_roc']:>7.4f} {r['pr_auc']:>7.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

h, l, n = results["splice_hybrid"], results["logit_only"], results["nli_only"]
print(f"\n--- KEY FINDINGS ---")
print(f"Logit-Only:   AUC={l['auc_roc']:.4f} F1={l['f1']:.4f}")
print(f"NLI-Only:     AUC={n['auc_roc']:.4f} F1={n['f1']:.4f}")
print(f"Splice Hybrid: AUC={h['auc_roc']:.4f} F1={h['f1']:.4f}")
print(f"Hybrid vs Logit: {'+' if h['auc_roc']>=l['auc_roc'] else ''}{h['auc_roc']-l['auc_roc']:.4f} AUC")
print(f"Hybrid vs NLI:   {'+' if h['auc_roc']>=n['auc_roc'] else ''}{h['auc_roc']-n['auc_roc']:.4f} AUC")

results["timing"] = {"nli_seconds": round(nli_time, 1), "nli_pairs_per_sec": round(len(nli_pairs)/nli_time, 1)}
with open(os.path.join(BASE, "splice_hybrid_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nNLI: {len(nli_pairs)} pairs in {nli_time:.1f}s ({len(nli_pairs)/nli_time:.0f} pairs/s)")
print("\n✅ SPLICE HYBRID EXPERIMENT COMPLETE")
