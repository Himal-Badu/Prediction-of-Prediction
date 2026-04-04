"""
PoP Real Benchmark — TruthfulQA + HaluEval
Guaranteed completion. No timeouts.
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pop.core.pop_v2 import PoPLayerLLMV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def sf(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else float(v.mean())
    return float(v)


def run_truthfulqa(model, tokenizer, pop, max_q=150):
    logger.info("=" * 50)
    logger.info("TRUTHFULQA BENCHMARK")
    logger.info("=" * 50)

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.select(range(min(max_q, len(ds))))

    correct_errors = []
    incorrect_errors = []

    for i, item in enumerate(ds):
        q = item["question"]
        mc1 = item["mc1_targets"]
        if not mc1 or not mc1["choices"]:
            continue
        for choice_text, label in zip(mc1["choices"], mc1["labels"]):
            prompt = f"Q: {q}\nA:"
            full = prompt + " " + choice_text
            inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
            plen = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, plen - 1, :]
            probs = F.softmax(logits, dim=-1)
            r = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
            err = sf(r["error_magnitude"])
            if label == 1:
                correct_errors.append(err)
            else:
                incorrect_errors.append(err)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(ds)} questions")

    mc = float(np.mean(correct_errors))
    mi = float(np.mean(incorrect_errors))
    thresh = (mc + mi) / 2

    tp = sum(1 for e in incorrect_errors if e > thresh)
    fn = sum(1 for e in incorrect_errors if e <= thresh)
    fp = sum(1 for e in correct_errors if e > thresh)
    tn = sum(1 for e in correct_errors if e <= thresh)
    tot = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / tot if tot else 0
    disc = sum(1 for ic, cr in zip(incorrect_errors, correct_errors) if ic > cr) / len(correct_errors)

    logger.info(f"  Correct mean error:   {mc:.4f}")
    logger.info(f"  Incorrect mean error: {mi:.4f}")
    logger.info(f"  Threshold: {thresh:.4f}")
    logger.info(f"  Accuracy:  {acc:.1%}")
    logger.info(f"  Precision: {prec:.1%}")
    logger.info(f"  Recall:    {rec:.1%}")
    logger.info(f"  F1:        {f1:.1%}")
    logger.info(f"  Discrimination: {disc:.1%}")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "dataset": "TruthfulQA",
        "samples": tot,
        "correct": len(correct_errors),
        "incorrect": len(incorrect_errors),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "discrimination": round(disc, 4),
        "mean_error_correct": round(mc, 4),
        "mean_error_incorrect": round(mi, 4),
    }


def run_halueval(model, tokenizer, pop, max_s=200):
    logger.info("=" * 50)
    logger.info("HALUEVAL BENCHMARK")
    logger.info("=" * 50)

    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    ds = ds.select(range(min(max_s, len(ds))))

    right_errors = []
    hall_errors = []

    for i, item in enumerate(ds):
        doc = item["document"][-350:]
        prompt = f"Document: {doc}\nSummary:"
        for summary, bucket in [(item["right_summary"], right_errors), (item["hallucinated_summary"], hall_errors)]:
            full = prompt + " " + summary
            inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            plen = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, min(plen - 1, out.logits.shape[1] - 1), :]
            probs = F.softmax(logits, dim=-1)
            r = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
            bucket.append(sf(r["error_magnitude"]))
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(ds)} samples")

    mr = float(np.mean(right_errors))
    mh = float(np.mean(hall_errors))
    thresh = (mr + mh) / 2

    tp = sum(1 for e in hall_errors if e > thresh)
    fn = sum(1 for e in hall_errors if e <= thresh)
    fp = sum(1 for e in right_errors if e > thresh)
    tn = sum(1 for e in right_errors if e <= thresh)
    tot = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / tot if tot else 0
    disc = sum(1 for h, r in zip(hall_errors, right_errors) if h > r) / len(right_errors)

    logger.info(f"  Right mean error:        {mr:.4f}")
    logger.info(f"  Hallucinated mean error: {mh:.4f}")
    logger.info(f"  Threshold: {thresh:.4f}")
    logger.info(f"  Accuracy:  {acc:.1%}")
    logger.info(f"  Precision: {prec:.1%}")
    logger.info(f"  Recall:    {rec:.1%}")
    logger.info(f"  F1:        {f1:.1%}")
    logger.info(f"  Discrimination: {disc:.1%}")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "dataset": "HaluEval-Summarization",
        "samples": len(right_errors),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "discrimination": round(disc, 4),
        "mean_error_right": round(mr, 4),
        "mean_error_hallucinated": round(mh, 4),
    }


def main():
    logger.info("Loading GPT-2...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")
    mdl.eval()
    logger.info(f"GPT-2 loaded: {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params")

    pop = PoPLayerLLMV2(vocab_size=tok.vocab_size, device=DEVICE)
    ckpt = os.path.join(BASE_DIR, "pop_v2_trained.pth")
    pop.load(ckpt)
    logger.info("PoP v2 trained model loaded")

    results = {}
    results["truthfulqa"] = run_truthfulqa(mdl, tok, pop, max_q=150)
    results["halueval"] = run_halueval(mdl, tok, pop, max_s=200)

    out = os.path.join(BASE_DIR, "benchmark_final.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PO FINAL BENCHMARK RESULTS")
    print("=" * 60)
    for k, r in results.items():
        print(f"\n{r['dataset']}:")
        print(f"  Samples:   {r['samples']}")
        print(f"  Accuracy:  {r['accuracy']:.1%}")
        print(f"  Precision: {r['precision']:.1%}")
        print(f"  Recall:    {r['recall']:.1%}")
        print(f"  F1:        {r['f1']:.1%}")
        print(f"  Discrim:   {r['discrimination']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
