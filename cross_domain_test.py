"""
Cross-domain test: Train PoP v2 features on HaluEval data directly.
Tests if the 24 features are general-purpose or QA-specific.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, sys, logging, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pop.core.pop_v2 import extract_features_vectorized

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        r = x
        h = self.norm1(x); h = F.relu(self.fc1(h)); h = self.dropout(h)
        h = self.norm2(h); h = self.fc2(h); h = self.dropout(h)
        return F.relu(h + r)


class Detector(nn.Module):
    def __init__(self, d=24, h=256, n=3, drop=0.15):
        super().__init__()
        self.inorm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop))
        self.blocks = nn.Sequential(*[ResBlock(h, drop) for _ in range(n)])
        self.head = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64,1))
    def forward(self, x):
        x = self.inorm(x); x = self.proj(x); x = self.blocks(x); return self.head(x).squeeze(-1)


def main():
    MAX_SAMPLES = 300  # 300 samples × 2 (right+hall) = 600 training points

    log.info("Loading GPT-2...")
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2"); mdl.eval()

    log.info(f"Extracting HaluEval features ({MAX_SAMPLES} samples)...")
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    features_list = []
    labels_list = []  # 0 = right, 1 = hallucinated
    t0 = time.time()

    for i, item in enumerate(ds):
        doc = item["document"][-200:]
        prompt = f"Summarize: {doc}\nAnswer:"
        for text, label in [(item["right_summary"], 0), (item["hallucinated_summary"], 1)]:
            full = prompt + " " + text
            inp = tok(full, return_tensors="pt", truncation=True, max_length=512)
            plen = tok(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].shape[1]
            with torch.no_grad(): out = mdl(**inp)
            pos = min(plen - 1, out.logits.shape[1] - 1)
            logits = out.logits[0, pos, :]
            probs = F.softmax(logits, dim=-1)
            feats = extract_features_vectorized(logits.unsqueeze(0), probs.unsqueeze(0))
            features_list.append(feats.squeeze(0).numpy())
            labels_list.append(label)
        if (i+1) % 50 == 0:
            r = (i+1)/(time.time()-t0)
            log.info(f"  [{i+1}/{len(ds)}] ({r:.1f} samples/s)")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    log.info(f"Extracted: {X.shape[0]} samples, {y.mean():.1%} hallucinated")

    # Train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    Xtr_t, ytr_t = torch.FloatTensor(Xtr), torch.FloatTensor(ytr)
    Xte_t, yte_t = torch.FloatTensor(Xte), torch.FloatTensor(yte)

    model = Detector(X.shape[1])
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

    best_f1 = 0; best_state = None
    for ep in range(1, 301):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr_t), ytr_t); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(Xte_t)).numpy()
            preds = (scores > 0.5).astype(float)
            vf1 = f1_score(yte, preds, zero_division=0)
            disc = np.mean(scores[yte==1] > scores[yte==0]) if (yte==1).any() and (yte==0).any() else 0
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            log.info(f"  Epoch {ep}: loss={loss.item():.4f} val_f1={vf1:.4f} disc={disc:.1%} best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Final eval
    with torch.no_grad():
        scores = torch.sigmoid(model(Xte_t)).numpy()
    preds = (scores > 0.5).astype(float)
    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds, zero_division=0)
    rec = recall_score(yte, preds, zero_division=0)
    f1 = f1_score(yte, preds, zero_division=0)
    disc = np.mean(scores[yte==1] > scores[yte==0]) if (yte==1).any() and (yte==0).any() else 0

    right_mean = float(np.mean(scores[yte==0]))
    hall_mean = float(np.mean(scores[yte==1]))

    log.info(f"\n{'='*60}")
    log.info(f"HaluEval TRAINED on HaluEval (in-domain test)")
    log.info(f"{'='*60}")
    log.info(f"  Accuracy:    {acc:.1%}")
    log.info(f"  Precision:   {prec:.1%}")
    log.info(f"  Recall:      {rec:.1%}")
    log.info(f"  F1:          {f1:.1%}")
    log.info(f"  Discrimination: {disc:.1%}")
    log.info(f"  Right mean:  {right_mean:.4f}")
    log.info(f"  Hall mean:   {hall_mean:.4f}")

    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN TEST RESULTS")
    print(f"{'='*60}")
    print(f"  HaluEval (trained on HaluEval):")
    print(f"    Prec={prec:.1%} Rec={rec:.1%} F1={f1:.1%} Disc={disc:.1%}")
    print(f"")
    print(f"  vs HaluEval (trained on TruthfulQA):")
    print(f"    Prec=50.0% Rec=100% F1=66.7% Disc=0.0%")
    print(f"")
    if disc > 0.6:
        print(f"  ✅ FEATURES ARE GENERAL — just need domain-specific training")
        print(f"  → Train on multiple domains → ensemble → production")
    elif disc > 0.5:
        print(f"  ⚡ PARTIAL — features have some cross-domain signal")
        print(f"  → Need better feature engineering or more training data")
    else:
        print(f"  ❌ FEATURES ARE TASK-SPECIFIC — 24 features don't capture")
        print(f"     summarization hallucination patterns")
        print(f"  → Need fundamentally different features for cross-domain")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
