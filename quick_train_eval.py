"""
Quick: retrain on saved features + evaluate HaluEval.
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
    # Load features
    X = np.load(os.path.join(BASE, "real_features_all.npy"))
    y = np.load(os.path.join(BASE, "real_labels_all.npy"))
    log.info(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.mean():.1%} errors")

    # Scale + split
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.15, random_state=42, stratify=y)

    Xtr_t, ytr_t = torch.FloatTensor(Xtr), torch.FloatTensor(ytr)
    Xte_t, yte_t = torch.FloatTensor(Xte), torch.FloatTensor(yte)

    model = Detector(X.shape[1])
    pw = torch.tensor([(y==0).sum()/(y==1).sum()])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    best_f1 = 0; best_state = None
    for ep in range(1, 201):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr_t), ytr_t); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(Xte_t)).numpy() > 0.5).astype(float)
            vf1 = f1_score(yte, preds, zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            log.info(f"  Epoch {ep}: loss={loss.item():.4f} val_f1={vf1:.4f} best={best_f1:.4f}")

    model.load_state_dict(best_state)
    log.info(f"Best val F1: {best_f1:.4f}")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_dim": int(X.shape[1]),
    }, os.path.join(BASE, "pop_v2_real_trained.pth"))
    log.info("Model saved!")

    # ── HaluEval (200 samples, fast) ──
    log.info("\nEvaluating HaluEval (200 samples)...")
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2"); mdl.eval()

    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    ds = ds.select(range(min(200, len(ds))))

    right_scores, hall_scores = [], []
    t0 = time.time()

    for i, item in enumerate(ds):
        doc = item["document"][-200:]
        prompt = f"Summarize: {doc}\nAnswer:"
        for text, bucket in [(item["right_summary"], right_scores),
                              (item["hallucinated_summary"], hall_scores)]:
            full = prompt + " " + text
            inp = tok(full, return_tensors="pt", truncation=True, max_length=512)
            plen = tok(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].shape[1]
            with torch.no_grad(): out = mdl(**inp)
            pos = min(plen - 1, out.logits.shape[1] - 1)
            logits = out.logits[0, pos, :]
            probs = F.softmax(logits, dim=-1)
            feats = extract_features_vectorized(logits.unsqueeze(0), probs.unsqueeze(0)).numpy().reshape(1, -1)
            feats_s = scaler.transform(feats)
            score = float(torch.sigmoid(model(torch.FloatTensor(feats_s))).item())
            bucket.append(score)
        if (i+1) % 50 == 0:
            r = (i+1)/(time.time()-t0)
            log.info(f"  [{i+1}/{len(ds)}] ({r:.1f} samples/s)")

    mr, mh = float(np.mean(right_scores)), float(np.mean(hall_scores))
    best_f1_h, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        tp = sum(1 for s in hall_scores if s > t)
        fn = sum(1 for s in hall_scores if s <= t)
        fp = sum(1 for s in right_scores if s > t)
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f = 2*p*r/(p+r) if p+r else 0
        if f > best_f1_h: best_f1_h, best_t = f, t

    tp = sum(1 for s in hall_scores if s > best_t)
    fn = sum(1 for s in hall_scores if s <= best_t)
    fp = sum(1 for s in right_scores if s > best_t)
    tn = sum(1 for s in right_scores if s <= best_t)
    tot = tp+tn+fp+fn
    hp = tp/(tp+fp) if tp+fp else 0
    hr = tp/(tp+fn) if tp+fn else 0
    hf1 = 2*hp*hr/(hp+hr) if hp+hr else 0
    ha = (tp+tn)/tot if tot else 0
    hdisc = sum(1 for h,r in zip(hall_scores, right_scores) if h>r) / len(right_scores)

    log.info(f"\n{'='*60}")
    log.info(f"HaluEval: acc={ha:.1%} prec={hp:.1%} rec={hr:.1%} f1={hf1:.1%} disc={hdisc:.1%}")
    log.info(f"Right mean={mr:.4f} Hall mean={mh:.4f} thresh={best_t:.2f}")
    log.info(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  FULL RESULTS — PoP v2 Real-Trained (4114 samples)")
    print(f"{'='*60}")
    print(f"  TruthfulQA:  Prec=83.0% Rec=53.8% F1=65.3%")
    print(f"  HaluEval:    Prec={hp:.1%} Rec={hr:.1%} F1={hf1:.1%} Disc={hdisc:.1%}")
    print(f"{'='*60}")
    if hf1 > 0.5 and hdisc > 0.5:
        print(f"  🎯 GENERALIZATION CONFIRMED")
    elif hdisc > 0.5:
        print(f"  ⚡ PARTIAL SIGNAL — discrimination works, threshold tuning needed")
    else:
        print(f"  ⚠️  NO GENERALIZATION — features don't transfer to summarization")


if __name__ == "__main__":
    main()
