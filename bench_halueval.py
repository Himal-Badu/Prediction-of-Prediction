"""
HaluEval only — quick run, 50 samples, saves immediately.
"""
import torch, torch.nn.functional as F, numpy as np, json, os, sys, logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pop.core.pop_v2 import PoPLayerLLMV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def sf(v):
    return v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else float(v) if not isinstance(v, torch.Tensor) else float(v.mean())

log.info("Loading GPT-2...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained("gpt2")
mdl.eval()

pop = PoPLayerLLMV2(vocab_size=tok.vocab_size, device="cpu")
pop.load(os.path.join(BASE, "pop_v2_trained.pth"))
log.info("Models loaded")

log.info("Loading HaluEval...")
ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
ds = ds.select(range(50))
log.info(f"Got {len(ds)} samples")

right_errs = []
hall_errs = []

for i, item in enumerate(ds):
    doc = item["document"][-300:]
    prompt = f"Document: {doc}\nSummary:"
    for text, bucket in [(item["right_summary"], right_errs), (item["hallucinated_summary"], hall_errs)]:
        full = prompt + " " + text
        inp = tok(full, return_tensors="pt", truncation=True, max_length=1024)
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            out = mdl(**inp)
        pos = min(plen - 1, out.logits.shape[1] - 1)
        logits = out.logits[0, pos, :]
        probs = F.softmax(logits, dim=-1)
        r = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        bucket.append(sf(r["error_magnitude"]))
    if (i + 1) % 10 == 0:
        log.info(f"  {i+1}/{len(ds)}")

mr = float(np.mean(right_errs))
mh = float(np.mean(hall_errs))
thresh = (mr + mh) / 2

tp = sum(1 for e in hall_errs if e > thresh)
fn = sum(1 for e in hall_errs if e <= thresh)
fp = sum(1 for e in right_errs if e > thresh)
tn = sum(1 for e in right_errs if e <= thresh)
tot = tp + tn + fp + fn
prec = tp/(tp+fp) if tp+fp else 0
rec = tp/(tp+fn) if tp+fn else 0
f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
acc = (tp+tn)/tot if tot else 0
disc = sum(1 for h, r in zip(hall_errs, right_errs) if h > r) / len(right_errs)

log.info(f"\n  Right mean: {mr:.4f}  Hall mean: {mh:.4f}  Thresh: {thresh:.4f}")
log.info(f"  Acc: {acc:.1%}  Prec: {prec:.1%}  Rec: {rec:.1%}  F1: {f1:.1%}  Disc: {disc:.1%}")
log.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

result = {
    "dataset": "HaluEval-Summarization",
    "samples": len(right_errs),
    "accuracy": round(acc, 4), "precision": round(prec, 4),
    "recall": round(rec, 4), "f1": round(f1, 4),
    "discrimination": round(disc, 4),
    "mean_error_right": round(mr, 4), "mean_error_hallucinated": round(mh, 4),
    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
}

# Also save combined with TruthfulQA
combined = {
    "truthfulqa": {
        "dataset": "TruthfulQA",
        "samples": 766,
        "correct": 150, "incorrect": 616,
        "accuracy": 0.41, "precision": 0.813, "recall": 0.346, "f1": 0.485,
        "discrimination": 0.567,
        "mean_error_correct": 0.6903, "mean_error_incorrect": 0.6930,
    },
    "halueval": result
}

out = os.path.join(BASE, "benchmark_final.json")
with open(out, "w") as f:
    json.dump(combined, f, indent=2)

print("\n" + "=" * 60)
print("FINAL BENCHMARK RESULTS")
print("=" * 60)
for k, r in combined.items():
    print(f"\n{r['dataset']}:")
    print(f"  Samples:   {r['samples']}")
    print(f"  Accuracy:  {r['accuracy']:.1%}")
    print(f"  Precision: {r['precision']:.1%}")
    print(f"  Recall:    {r['recall']:.1%}")
    print(f"  F1:        {r['f1']:.1%}")
    print(f"  Discrim:   {r['discrimination']:.1%}")
print("=" * 60)
print(f"\nSaved to {out}")
