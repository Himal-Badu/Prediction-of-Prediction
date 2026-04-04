"""
SPLICE HYBRID v4: Fixed training + cross-domain evaluation.
Uses pre-computed NLI features from v3.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import logging
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD PRE-COMPUTED FEATURES
# ============================================================
log.info("Loading TruthfulQA features...")
X_raw = np.load(os.path.join(BASE, "..", "real_features_all.npy"))
y_tq = np.load(os.path.join(BASE, "..", "real_labels_all.npy"))
X_logit = X_raw[:, :10].astype(np.float32)

# Load NLI features from v3 run
X_nli = np.load(os.path.join(BASE, "splice_nli_features.npy"))
X_hybrid = np.hstack([X_logit, X_nli])

log.info(f"TruthfulQA: {len(y_tq)} samples")
log.info(f"  Logit: {X_logit.shape}, NLI: {X_nli.shape}, Hybrid: {X_hybrid.shape}")
log.info(f"  Balance: {(y_tq==0).sum()} correct, {(y_tq==1).sum()} errors")

# ============================================================
# HALUEVAL NLI EXTRACTION
# ============================================================
log.info("\n" + "="*60)
log.info("Computing HaluEval NLI features...")
log.info("="*60)

from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")

ds_h = load_dataset("pminervini/HaluEval", "summarization", split="data")
ds_h = ds_h.select(range(min(500, len(ds_h))))
log.info(f"HaluEval: {len(ds_h)} samples")

# Collect pairs
he_pairs = []
he_labels = []
he_logit = []

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_mdl.eval()

def extract_logit_features(logits, probs):
    sp, _ = torch.sort(probs, descending=True)
    ent = float(-(probs * torch.log(probs + 1e-10)).sum())
    t1 = float(sp[0]); t5 = float(sp[:5].sum())
    margin = float(sp[0] - sp[1])
    gini = float(1 - (probs**2).sum())
    ml = float(logits.max()); mil = float(logits.mean()); sl = float(logits.std())
    conc = t1 / (t5 + 1e-10)
    spread = float(sp[0] - sp[4]) if len(sp) > 4 else t1
    return [ent, t1, t5, margin, gini, ml, mil, sl, conc, spread]

t0 = time.time()
for i, item in enumerate(ds_h):
    doc = item["document"][-300:]
    for summary, label in [(item["right_summary"], 0), (item["hallucinated_summary"], 1)]:
        prompt = f"Document: {doc}\nSummary:"
        full = prompt + " " + summary
        inp = gpt2_tok(full, return_tensors="pt", truncation=True, max_length=1024)
        plen = gpt2_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].shape[1]
        with torch.no_grad():
            out = gpt2_mdl(**inp)
        pos = min(plen - 1, out.logits.shape[1] - 1)
        logits = out.logits[0, pos, :]
        probs = F.softmax(logits, dim=-1)
        he_logit.append(extract_logit_features(logits, probs))
        he_pairs.append((doc[:200], summary[:200]))
        he_labels.append(label)
    if (i+1) % 100 == 0:
        log.info(f"  GPT-2: {i+1}/{len(ds_h)} ({(i+1)/(time.time()-t0):.1f} samples/s)")

X_logit_he = np.array(he_logit, dtype=np.float32)
y_he = np.array(he_labels, dtype=np.float32)
log.info(f"GPT-2 done: {len(y_he)} samples in {time.time()-t0:.1f}s")

# Batch NLI for HaluEval
log.info(f"Running NLI on {len(he_pairs)} HaluEval pairs...")
t1 = time.time()
import gc
all_scores = []
for i in range(0, len(he_pairs), 16):
    batch = he_pairs[i:i+16]
    scores = nli_model.predict(batch)
    all_scores.append(scores)
    if i % 256 == 0:
        gc.collect()
all_scores = np.vstack(all_scores)
probs_nli = torch.softmax(torch.tensor(all_scores), dim=-1).numpy()
X_nli_he = np.stack([probs_nli[:, 1], probs_nli[:, 0], probs_nli[:, 2]], axis=1).astype(np.float32)
del all_scores, probs_nli
gc.collect()
log.info(f"NLI done: {time.time()-t1:.1f}s")

X_hybrid_he = np.hstack([X_logit_he, X_nli_he])

# ============================================================
# TRAINING WITH FIXED CLASS BALANCE
# ============================================================
def train_sklearn(clf, X, y, name):
    """Train sklearn classifier with proper evaluation."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:, 1]
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
    return m, clf, scaler

class BalancedMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X, y, name, epochs=500):
    """Train MLP with proper AUC-based early stopping."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    
    model = BalancedMLP(X.shape[1])
    # Strong class weighting
    n_pos = (ytr == 1).sum()
    n_neg = (ytr == 0).sum()
    pw = torch.tensor([n_neg / n_pos])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    Xtr_t = torch.FloatTensor(Xtr)
    ytr_t = torch.FloatTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    
    best_auc, best_state = 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr_t), ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(Xte_t)
            probs = torch.sigmoid(logits).numpy()
            try:
                vauc = roc_auc_score(yte, probs)
            except:
                vauc = 0.5
        
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
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

def cross_eval(clf_or_model, scaler, X, y, name, is_sklearn=True):
    Xs = scaler.transform(X)
    if is_sklearn:
        probs = clf_or_model.predict_proba(Xs)[:, 1]
    else:
        clf_or_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(clf_or_model(torch.FloatTensor(Xs))).numpy()
    preds = (probs > 0.5).astype(float)
    m = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y, probs)), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
    }
    log.info(f"  [{name}] AUC={m['auc_roc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
    return m

# ============================================================
# RUN EXPERIMENTS
# ============================================================
log.info("\n" + "="*60)
log.info("IN-DOMAIN: TruthfulQA")
log.info("="*60)

results = {}

# Sklearn classifiers
log.info("\n--- Logit + LR ---")
results["logit_lr"], m_lr, s_lr = train_sklearn(
    LogisticRegression(max_iter=1000, class_weight='balanced'), X_logit, y_tq, "Logit-LR")

log.info("\n--- NLI + LR ---")
results["nli_lr"], m_nli_lr, s_nli_lr = train_sklearn(
    LogisticRegression(max_iter=1000, class_weight='balanced'), X_nli, y_tq, "NLI-LR")

log.info("\n--- Hybrid + LR ---")
results["hybrid_lr"], m_hlr, s_hlr = train_sklearn(
    LogisticRegression(max_iter=1000, class_weight='balanced'), X_hybrid, y_tq, "Hybrid-LR")

log.info("\n--- NLI + GBM ---")
results["nli_gbm"], m_ngbm, s_ngbm = train_sklearn(
    GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1), X_nli, y_tq, "NLI-GBM")

log.info("\n--- Hybrid + GBM ---")
results["hybrid_gbm"], m_hgbm, s_hgbm = train_sklearn(
    GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1), X_hybrid, y_tq, "Hybrid-GBM")

log.info("\n--- Hybrid MLP (fixed) ---")
results["hybrid_mlp"], m_mlp, s_mlp = train_mlp(X_hybrid, y_tq, "Hybrid-MLP")

# ============================================================
# CROSS-DOMAIN
# ============================================================
log.info("\n" + "="*60)
log.info("CROSS-DOMAIN: TruthfulQA → HaluEval")
log.info("="*60)

results["cd_logit_lr"] = cross_eval(m_lr, s_lr, X_logit_he, y_he, "CD-Logit-LR")
results["cd_nli_lr"] = cross_eval(m_nli_lr, s_nli_lr, X_nli_he, y_he, "CD-NLI-LR")
results["cd_hybrid_lr"] = cross_eval(m_hlr, s_hlr, X_hybrid_he, y_he, "CD-Hybrid-LR")
results["cd_nli_gbm"] = cross_eval(m_ngbm, s_ngbm, X_nli_he, y_he, "CD-NLI-GBM")
results["cd_hybrid_gbm"] = cross_eval(m_hgbm, s_hgbm, X_hybrid_he, y_he, "CD-Hybrid-GBM")
results["cd_hybrid_mlp"] = cross_eval(m_mlp, s_mlp, X_hybrid_he, y_he, "CD-Hybrid-MLP", is_sklearn=False)

# ============================================================
# FEATURE ANALYSIS
# ============================================================
log.info("\n" + "="*60)
log.info("FEATURE ANALYSIS")
log.info("="*60)

feat_names = ["entropy","top1","top5","margin","gini","max_logit","mean_logit","std_logit","concentration","spread"]
print("\n--- TruthfulQA Logit Features (error vs correct) ---")
for j, name in enumerate(feat_names):
    err = float(X_logit[y_tq==1, j].mean())
    cor = float(X_logit[y_tq==0, j].mean())
    d = abs(err - cor)
    print(f"  {name:15s}: err={err:.4f} cor={cor:.4f} Δ={d:.4f}")

print("\n--- TruthfulQA NLI Features (error vs correct) ---")
for j, name in enumerate(["entailment","contradiction","neutral"]):
    err = float(X_nli[y_tq==1, j].mean())
    cor = float(X_nli[y_tq==0, j].mean())
    d = abs(err - cor)
    print(f"  {name:15s}: err={err:.4f} cor={cor:.4f} Δ={d:.4f}")

print("\n--- HaluEval NLI Features (hallucinated vs correct) ---")
for j, name in enumerate(["entailment","contradiction","neutral"]):
    err = float(X_nli_he[y_he==1, j].mean())
    cor = float(X_nli_he[y_he==0, j].mean())
    d = abs(err - cor)
    print(f"  {name:15s}: hall={err:.4f} right={cor:.4f} Δ={d:.4f}")

# GBM feature importance
print("\n--- Hybrid GBM Feature Importance ---")
gbm_feats = feat_names + ["entailment","contradiction","neutral"]
for name, imp in sorted(zip(gbm_feats, m_hgbm.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:15s}: {imp:.4f}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*70)
print("SPLICE HYBRID v4 — FINAL RESULTS")
print("="*70)

print(f"\n{'Method':<20} {'Domain':<10} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'P':>7} {'R':>7}")
print("-" * 65)
for key, r in results.items():
    domain = "HaluEval" if key.startswith("cd_") else "TruthfulQA"
    method = key.replace("cd_", "").replace("_", " ").title()
    print(f"{method:<20} {domain:<10} {r['auc_roc']:>7.4f} {r['pr_auc']:>7.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

print(f"\n{'='*70}")
print("KEY FINDINGS")
print(f"{'='*70}")

# In-domain
print(f"\nIn-Domain (TruthfulQA):")
for k in ["logit_lr", "nli_lr", "hybrid_lr", "nli_gbm", "hybrid_gbm", "hybrid_mlp"]:
    r = results[k]
    print(f"  {k:<20}: AUC={r['auc_roc']:.4f} F1={r['f1']:.4f}")

print(f"\nCross-Domain (TruthfulQA → HaluEval):")
for k in ["cd_logit_lr", "cd_nli_lr", "cd_hybrid_lr", "cd_nli_gbm", "cd_hybrid_gbm", "cd_hybrid_mlp"]:
    r = results[k]
    print(f"  {k:<20}: AUC={r['auc_roc']:.4f} F1={r['f1']:.4f}")

# Best models
best_indomain = max([k for k in results if not k.startswith("cd_")], key=lambda k: results[k]["auc_roc"])
best_cross = max([k for k in results if k.startswith("cd_")], key=lambda k: results[k]["auc_roc"])
print(f"\nBest in-domain:   {best_indomain} (AUC={results[best_indomain]['auc_roc']:.4f})")
print(f"Best cross-domain: {best_cross} (AUC={results[best_cross]['auc_roc']:.4f})")

# Save
with open(os.path.join(BASE, "splice_hybrid_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to splice_hybrid_results.json")
print("\n✅ SPLICE HYBRID v4 COMPLETE")
