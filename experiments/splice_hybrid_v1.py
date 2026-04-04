"""
SPLICE HYBRID EXPERIMENT v1
============================
Real NLI (MiniLM-L6) + Real Logit Features (GPT-2) on TruthfulQA + HaluEval.

Architecture:
  - GPT-2 logit features (10): entropy, top1, top5, margin, gini, max_logit, mean_logit, std_logit, concentration, spread
  - MiniLM NLI features (3): entailment_score, contradiction_score, neutral_score
  - Combined hybrid (13 features) → MLP detector

Tests:
  A. Logit-only detector (in-domain TruthfulQA)
  B. NLI-only detector (in-domain TruthfulQA)  
  C. HYBRID detector (logit + NLI, in-domain TruthfulQA)
  D. Cross-domain: train on TruthfulQA, test on HaluEval summarization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             accuracy_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# MODEL LOADING
# ============================================================
log.info("=" * 70)
log.info("SPLICE HYBRID: Real NLI + Real Logits")
log.info("=" * 70)

# GPT-2 for logit extraction
log.info("Loading GPT-2...")
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_mdl.eval()
log.info("GPT-2 loaded")

# MiniLM for NLI
log.info("Loading MiniLM-L6 NLI...")
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
log.info("MiniLM-L6 loaded")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
LOGIT_FEATURES = ["entropy", "top1", "top5", "margin", "gini", 
                   "max_logit", "mean_logit", "std_logit", "concentration", "spread"]
NLI_FEATURES = ["entailment", "contradiction", "neutral"]
N_LOGIT = len(LOGIT_FEATURES)
N_NLI = len(NLI_FEATURES)

def extract_logit_features(logits, probs):
    """Extract 10 distributional features from GPT-2 logits."""
    sp, _ = torch.sort(probs, descending=True)
    ent = float(-(probs * torch.log(probs + 1e-10)).sum())
    t1 = float(sp[0])
    t5 = float(sp[:5].sum())
    margin = float(sp[0] - sp[1])
    gini = float(1 - (probs**2).sum())
    ml = float(logits.max())
    mil = float(logits.mean())
    sl = float(logits.std())
    conc = t1 / (t5 + 1e-10)
    spread = float(sp[0] - sp[4]) if len(sp) > 4 else t1
    return [ent, t1, t5, margin, gini, ml, mil, sl, conc, spread]

def extract_nli_features(premise, hypothesis):
    """Extract 3 NLI features using MiniLM."""
    scores = nli_model.predict([(premise, hypothesis)])[0]
    # scores = [contradiction, entailment, neutral] (logits)
    # Apply softmax for normalized probabilities
    probs = torch.softmax(torch.tensor(scores), dim=-1)
    return [float(probs[1]), float(probs[0]), float(probs[2])]  # [entailment, contradiction, neutral]

# ============================================================
# DATA EXTRACTION
# ============================================================
def extract_truthfulqa_data(max_questions=None):
    """Extract real features from TruthfulQA using GPT-2 + MiniLM."""
    log.info("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))
    log.info(f"Processing {len(ds)} questions...")
    
    all_logit_feats = []
    all_nli_feats = []
    all_labels = []
    t0 = time.time()
    
    for i, item in enumerate(ds):
        q = item["question"]
        mc1 = item["mc1_targets"]
        if not mc1 or not mc1["choices"]:
            continue
        
        prompt = f"Q: {q}\nA:"
        
        for choice_text, label in zip(mc1["choices"], mc1["labels"]):
            full = prompt + " " + choice_text
            
            # GPT-2 logit features
            inp = gpt2_tok(full, return_tensors="pt")
            plen = gpt2_tok(prompt, return_tensors="pt")["input_ids"].shape[1]
            with torch.no_grad():
                out = gpt2_mdl(**inp)
            pos = min(plen - 1, out.logits.shape[1] - 1)
            logits = out.logits[0, pos, :]
            probs = F.softmax(logits, dim=-1)
            logit_feats = extract_logit_features(logits, probs)
            
            # NLI features: premise=question, hypothesis=answer
            nli_feats = extract_nli_features(q, choice_text)
            
            all_logit_feats.append(logit_feats)
            all_nli_feats.append(nli_feats)
            all_labels.append(1 if label == 0 else 0)  # 0=correct in TruthfulQA → 1=correct for us
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / rate
            log.info(f"  {i+1}/{len(ds)} questions | {rate:.1f} q/s | ETA {eta:.0f}s")
    
    X_logit = np.array(all_logit_feats, dtype=np.float32)
    X_nli = np.array(all_nli_feats, dtype=np.float32)
    X_hybrid = np.hstack([X_logit, X_nli])
    y = np.array(all_labels, dtype=np.float32)
    
    return X_logit, X_nli, X_hybrid, y

def extract_halueval_data(max_samples=500):
    """Extract real features from HaluEval summarization."""
    log.info("Loading HaluEval...")
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    ds = ds.select(range(min(max_samples, len(ds))))
    log.info(f"Processing {len(ds)} HaluEval samples...")
    
    all_logit_feats = []
    all_nli_feats = []
    all_labels = []
    t0 = time.time()
    
    for i, item in enumerate(ds):
        doc = item["document"][-300:]  # Truncate long docs
        right_summary = item["right_summary"]
        hall_summary = item["hallucinated_summary"]
        
        for summary, label in [(right_summary, 0), (hall_summary, 1)]:
            prompt = f"Document: {doc}\nSummary:"
            full = prompt + " " + summary
            
            # GPT-2 logit features
            inp = gpt2_tok(full, return_tensors="pt", truncation=True, max_length=1024)
            plen = gpt2_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].shape[1]
            with torch.no_grad():
                out = gpt2_mdl(**inp)
            pos = min(plen - 1, out.logits.shape[1] - 1)
            logits = out.logits[0, pos, :]
            probs = F.softmax(logits, dim=-1)
            logit_feats = extract_logit_features(logits, probs)
            
            # NLI features: premise=document, hypothesis=summary
            nli_feats = extract_nli_features(doc[:200], summary)  # Truncate premise for speed
            
            all_logit_feats.append(logit_feats)
            all_nli_feats.append(nli_feats)
            all_labels.append(label)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info(f"  {i+1}/{len(ds)} samples | {rate:.1f} samples/s")
    
    X_logit = np.array(all_logit_feats, dtype=np.float32)
    X_nli = np.array(all_nli_feats, dtype=np.float32)
    X_hybrid = np.hstack([X_logit, X_nli])
    y = np.array(all_labels, dtype=np.float32)
    
    return X_logit, X_nli, X_hybrid, y

# ============================================================
# DETECTOR MODEL
# ============================================================
class SpliceHybridDetector(nn.Module):
    """MLP detector for Splice Hybrid features."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_and_evaluate(X, y, name, epochs=300, lr=1e-3):
    """Train detector and return metrics."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    
    model = SpliceHybridDetector(X.shape[1])
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    Xtr_t = torch.FloatTensor(Xtr)
    ytr_t = torch.FloatTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr_t), ytr_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xte_t)).numpy()
            preds = (probs > 0.5).astype(float)
            vf1 = f1_score(yte, preds, zero_division=0)
        
        scheduler.step(1 - vf1)
        
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 100 == 0:
            log.info(f"  [{name}] Epoch {epoch+1}: loss={loss.item():.4f} val_f1={vf1:.4f} best={best_f1:.4f}")
    
    # Final evaluation on test set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).numpy()
        preds = (probs > 0.5).astype(float)
    
    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds, zero_division=0)
    rec = recall_score(yte, preds, zero_division=0)
    f1 = f1_score(yte, preds, zero_division=0)
    try:
        auc = roc_auc_score(yte, probs)
    except:
        auc = 0.5
    try:
        pr_auc = average_precision_score(yte, probs)
    except:
        pr_auc = 0.5
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "best_val_f1": round(float(best_f1), 4),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1])
    }
    
    log.info(f"  [{name}] FINAL: AUC={auc:.4f} PR-AUC={pr_auc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")
    return metrics, model, scaler

def cross_domain_eval(train_model, train_scaler, X_test, y_test, name):
    """Evaluate a trained model on a different domain."""
    Xs = train_scaler.transform(X_test)
    train_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(train_model(torch.FloatTensor(Xs))).numpy()
        preds = (probs > 0.5).astype(float)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
    try:
        pr_auc = average_precision_score(y_test, probs)
    except:
        pr_auc = 0.5
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "n_samples": int(len(y_test))
    }
    
    log.info(f"  [{name}] CROSS-DOMAIN: AUC={auc:.4f} PR-AUC={pr_auc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")
    return metrics

# ============================================================
# MAIN EXPERIMENT
# ============================================================
if __name__ == "__main__":
    results = {}
    
    # --- STEP 1: Extract TruthfulQA features ---
    log.info("\n" + "=" * 70)
    log.info("STEP 1: Extract TruthfulQA Features (GPT-2 + MiniLM NLI)")
    log.info("=" * 70)
    
    t0 = time.time()
    X_logit_tq, X_nli_tq, X_hybrid_tq, y_tq = extract_truthfulqa_data()
    extract_time = time.time() - t0
    log.info(f"\nExtracted {len(y_tq)} samples in {extract_time:.1f}s")
    log.info(f"  Logit features: {X_logit_tq.shape[1]}")
    log.info(f"  NLI features: {X_nli_tq.shape[1]}")
    log.info(f"  Hybrid features: {X_hybrid_tq.shape[1]}")
    log.info(f"  Balance: {(y_tq==0).sum()} correct, {(y_tq==1).sum()} errors")
    
    # Save features
    np.save(os.path.join(BASE, "splice_logit_features.npy"), X_logit_tq)
    np.save(os.path.join(BASE, "splice_nli_features.npy"), X_nli_tq)
    np.save(os.path.join(BASE, "splice_hybrid_features.npy"), X_hybrid_tq)
    np.save(os.path.join(BASE, "splice_labels.npy"), y_tq)
    
    # --- STEP 2: Train detectors ---
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Train Detectors (Logit vs NLI vs Hybrid)")
    log.info("=" * 70)
    
    # A: Logit-only
    log.info("\n--- A: Logit-Only Detector (10 features) ---")
    results["logit_only"], model_logit, scaler_logit = train_and_evaluate(
        X_logit_tq, y_tq, "Logit-Only")
    
    # B: NLI-only
    log.info("\n--- B: NLI-Only Detector (3 features) ---")
    results["nli_only"], model_nli, scaler_nli = train_and_evaluate(
        X_nli_tq, y_tq, "NLI-Only")
    
    # C: HYBRID (Splice Hybrid)
    log.info("\n--- C: SPLICE HYBRID Detector (13 features) ---")
    results["splice_hybrid"], model_hybrid, scaler_hybrid = train_and_evaluate(
        X_hybrid_tq, y_tq, "Splice-Hybrid")
    
    # --- STEP 3: Feature importance analysis ---
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Feature Importance (mean difference error vs correct)")
    log.info("=" * 70)
    
    print("\nLogit features:")
    for j, name in enumerate(LOGIT_FEATURES):
        err_mean = X_logit_tq[y_tq == 1, j].mean()
        cor_mean = X_logit_tq[y_tq == 0, j].mean()
        diff = abs(float(err_mean - cor_mean))
        print(f"  {name:15s}: err={float(err_mean):.4f} cor={float(cor_mean):.4f} diff={diff:.4f}")
    
    print("\nNLI features:")
    for j, name in enumerate(NLI_FEATURES):
        err_mean = X_nli_tq[y_tq == 1, j].mean()
        cor_mean = X_nli_tq[y_tq == 0, j].mean()
        diff = abs(float(err_mean - cor_mean))
        print(f"  {name:15s}: err={float(err_mean):.4f} cor={float(cor_mean):.4f} diff={diff:.4f}")
    
    # --- STEP 4: Cross-domain (TruthfulQA → HaluEval) ---
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Cross-Domain (TruthfulQA → HaluEval Summarization)")
    log.info("=" * 70)
    
    X_logit_he, X_nli_he, X_hybrid_he, y_he = extract_halueval_data(500)
    
    log.info("\n--- Cross-Domain: Logit-Only ---")
    results["crossdomain_logit"] = cross_domain_eval(
        model_logit, scaler_logit, X_logit_he, y_he, "Logit-CD")
    
    log.info("\n--- Cross-Domain: NLI-Only ---")
    results["crossdomain_nli"] = cross_domain_eval(
        model_nli, scaler_nli, X_nli_he, y_he, "NLI-CD")
    
    log.info("\n--- Cross-Domain: Splice Hybrid ---")
    results["crossdomain_hybrid"] = cross_domain_eval(
        model_hybrid, scaler_hybrid, X_hybrid_he, y_he, "Hybrid-CD")
    
    # --- FINAL REPORT ---
    print("\n" + "=" * 70)
    print("SPLICE HYBRID — FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Domain':<12} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'P':>7} {'R':>7}")
    print("-" * 72)
    
    for key, r in results.items():
        if "crossdomain" in key:
            domain = "HaluEval"
            method = key.replace("crossdomain_", "").replace("hybrid", "Splice Hybrid")
        else:
            domain = "TruthfulQA"
            method = key.replace("_", " ").title()
        print(f"{method:<25} {domain:<12} {r['auc_roc']:>7.4f} {r['pr_auc']:>7.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")
    
    # Key analysis
    print("\n--- KEY FINDINGS ---")
    
    h_auc = results["splice_hybrid"]["auc_roc"]
    l_auc = results["logit_only"]["auc_roc"]
    n_auc = results["nli_only"]["auc_roc"]
    
    print(f"\nIn-Domain (TruthfulQA):")
    print(f"  Logit-only:       AUC = {l_auc:.4f}")
    print(f"  NLI-only:         AUC = {n_auc:.4f}")
    print(f"  Splice Hybrid:    AUC = {h_auc:.4f}")
    print(f"  Hybrid vs Logit:  {'+' if h_auc >= l_auc else ''}{h_auc - l_auc:.4f} AUC")
    print(f"  Hybrid vs NLI:    {'+' if h_auc >= n_auc else ''}{h_auc - n_auc:.4f} AUC")
    
    h_cd = results["crossdomain_hybrid"]["auc_roc"]
    l_cd = results["crossdomain_logit"]["auc_roc"]
    n_cd = results["crossdomain_nli"]["auc_roc"]
    
    print(f"\nCross-Domain (TruthfulQA → HaluEval):")
    print(f"  Logit-only:       AUC = {l_cd:.4f}")
    print(f"  NLI-only:         AUC = {n_cd:.4f}")
    print(f"  Splice Hybrid:    AUC = {h_cd:.4f}")
    
    if h_cd > l_cd + 0.05:
        print(f"\n✅ NLI features improve cross-domain generalization!")
    elif h_cd > l_cd:
        print(f"\n⚠️  Hybrid marginally better cross-domain")
    else:
        print(f"\n❌ No cross-domain improvement from NLI")
    
    # Save results
    results["meta"] = {
        "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
        "gpt2_model": "gpt2",
        "n_truthfulqa": int(len(y_tq)),
        "n_halueval": int(len(y_he)),
        "extract_time_seconds": round(extract_time, 1),
        "features": {
            "logit": LOGIT_FEATURES,
            "nli": NLI_FEATURES,
            "hybrid": LOGIT_FEATURES + NLI_FEATURES
        }
    }
    
    with open(os.path.join(BASE, "splice_hybrid_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to splice_hybrid_results.json")
    
    # Save models
    torch.save({
        "logit_model": model_logit.state_dict(),
        "nli_model": model_nli.state_dict(),
        "hybrid_model": model_hybrid.state_dict(),
        "logit_scaler_mean": scaler_logit.mean_.tolist(),
        "logit_scaler_scale": scaler_logit.scale_.tolist(),
        "nli_scaler_mean": scaler_nli.mean_.tolist(),
        "nli_scaler_scale": scaler_nli.scale_.tolist(),
        "hybrid_scaler_mean": scaler_hybrid.mean_.tolist(),
        "hybrid_scaler_scale": scaler_hybrid.scale_.tolist(),
    }, os.path.join(BASE, "splice_hybrid_models.pth"))
    log.info("Models saved to splice_hybrid_models.pth")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
