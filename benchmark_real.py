"""
PoP Benchmark — Real Hallucination Detection Test

Tests PoP on TruthfulQA and HaluEval against GPT-2.
Reduced sample sizes for fast completion on CPU.
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pop.core.pop_v2 import PoPLayerLLMV2, extract_features_vectorized

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_name):
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_name}: {params/1e6:.1f}M params")
    return model, tokenizer


def load_pop(vocab_size, device):
    pop = PoPLayerLLMV2(vocab_size=vocab_size, device=device)
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pop_v2_trained.pth")
    if os.path.exists(ckpt):
        pop.load(ckpt)
        logger.info("Trained PoP v2 loaded")
    else:
        logger.warning("No checkpoint found — untrained")
    return pop


def sf(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else float(v.mean())
    return float(v)


def get_logits_at_boundary(model, tokenizer, prompt, completion, device="cpu"):
    """Get logits at the position predicting the first completion token."""
    inputs = tokenizer(prompt + completion, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, prompt_len - 1, :]
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def get_logits_for_generation(model, tokenizer, prompt, device="cpu"):
    """Get logits at the generation position (what token will model produce?)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def benchmark_truthfulqa(model, tokenizer, pop, device, max_samples=200):
    """
    Test PoP on TruthfulQA.
    For each question, test both correct and incorrect answers.
    PoP should give higher error_magnitude for incorrect answers.
    """
    logger.info("=" * 50)
    logger.info("TruthfulQA Benchmark")
    logger.info("=" * 50)

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.select(range(min(max_samples, len(ds))))

    correct_errors = []
    incorrect_errors = []
    tp = fp = tn = fn = 0

    for i, item in enumerate(ds):
        q = item["question"]
        mc1 = item["mc1_targets"]
        if not mc1 or not mc1["choices"]:
            continue
        choices = mc1["choices"]
        labels = mc1["labels"]

        for choice_text, label in zip(choices, labels):
            prompt = f"Q: {q}\nA:"
            logits, probs = get_logits_at_boundary(model, tokenizer, prompt, f" {choice_text}", device)
            result = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
            err_mag = sf(result["error_magnitude"])

            if label == 1:
                correct_errors.append(err_mag)
            else:
                incorrect_errors.append(err_mag)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(ds)} questions processed")

    # Compute threshold and metrics
    mean_correct = np.mean(correct_errors) if correct_errors else 0.5
    mean_incorrect = np.mean(incorrect_errors) if incorrect_errors else 0.5
    threshold = (mean_correct + mean_incorrect) / 2

    # Classify using threshold
    for e in correct_errors:
        if e > threshold:  # predicted error, but actually correct
            fp += 1
        else:
            tn += 1
    for e in incorrect_errors:
        if e > threshold:  # predicted error, and actually incorrect
            tp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Discrimination: how often is incorrect > correct?
    discrim = sum(1 for ic, cr in zip(incorrect_errors, correct_errors) if ic > cr) / len(correct_errors) if correct_errors else 0

    logger.info(f"\n  Mean error (correct answers):   {mean_correct:.4f}")
    logger.info(f"  Mean error (incorrect answers): {mean_incorrect:.4f}")
    logger.info(f"  Threshold:                      {threshold:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.1%}")
    logger.info(f"  Precision: {precision:.1%}")
    logger.info(f"  Recall:    {recall:.1%}")
    logger.info(f"  F1:        {f1:.1%}")
    logger.info(f"  Discrimination: {discrim:.1%}")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"  Samples: {len(correct_errors)} correct, {len(incorrect_errors)} incorrect")

    return {
        "dataset": "TruthfulQA",
        "samples": tp + tn + fp + fn,
        "correct_samples": len(correct_errors),
        "incorrect_samples": len(incorrect_errors),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "discrimination_rate": round(discrim, 4),
        "mean_error_correct": round(mean_correct, 4),
        "mean_error_incorrect": round(mean_incorrect, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def benchmark_halueval(model, tokenizer, pop, device, max_samples=300):
    """
    Test PoP on HaluEval summarization.
    Compare error_magnitude on right vs hallucinated summaries.
    """
    logger.info("=" * 50)
    logger.info("HaluEval Benchmark (Summarization)")
    logger.info("=" * 50)

    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    ds = ds.select(range(min(max_samples, len(ds))))

    right_errors = []
    hall_errors = []

    for i, item in enumerate(ds):
        doc = item["document"][-400:]  # truncate for GPT-2 context
        prompt = f"Document: {doc}\nSummary:"

        # Right summary
        logits_r, probs_r = get_logits_at_boundary(model, tokenizer, prompt, f" {item['right_summary']}", device)
        r_r = pop.predict(logits_r.unsqueeze(0), probs_r.unsqueeze(0))
        err_r = sf(r_r["error_magnitude"])

        # Hallucinated summary
        logits_h, probs_h = get_logits_at_boundary(model, tokenizer, prompt, f" {item['hallucinated_summary']}", device)
        r_h = pop.predict(logits_h.unsqueeze(0), probs_h.unsqueeze(0))
        err_h = sf(r_h["error_magnitude"])

        right_errors.append(err_r)
        hall_errors.append(err_h)

        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(ds)} samples processed")

    mean_right = np.mean(right_errors)
    mean_hall = np.mean(hall_errors)
    threshold = (mean_right + mean_hall) / 2

    # Classification
    tp = fp = tn = fn = 0
    # Right summaries should be "ok" (below threshold)
    for e in right_errors:
        if e > threshold:
            fp += 1
        else:
            tn += 1
    # Hallucinated summaries should be "error" (above threshold)
    for e in hall_errors:
        if e > threshold:
            tp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    discrim = sum(1 for h, r in zip(hall_errors, right_errors) if h > r) / len(right_errors)

    logger.info(f"\n  Mean error (right summaries):        {mean_right:.4f}")
    logger.info(f"  Mean error (hallucinated summaries): {mean_hall:.4f}")
    logger.info(f"  Threshold:                           {threshold:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.1%}")
    logger.info(f"  Precision: {precision:.1%}")
    logger.info(f"  Recall:    {recall:.1%}")
    logger.info(f"  F1:        {f1:.1%}")
    logger.info(f"  Discrimination: {discrim:.1%}")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "dataset": "HaluEval-Summarization",
        "samples": len(right_errors),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "discrimination_rate": round(discrim, 4),
        "mean_error_right": round(mean_right, 4),
        "mean_error_hallucinated": round(mean_hall, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    device = "cpu"
    model_name = "gpt2"
    model, tokenizer = load_model(model_name)
    pop = load_pop(tokenizer.vocab_size, device)

    results = {}
    results["truthfulqa"] = benchmark_truthfulqa(model, tokenizer, pop, device, max_samples=200)
    results["halueval"] = benchmark_halueval(model, tokenizer, pop, device, max_samples=300)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_real_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PO REAL BENCHMARK RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{r['dataset']}:")
        print(f"  Samples:   {r['samples']}")
        print(f"  Accuracy:  {r['accuracy']:.1%}")
        print(f"  Precision: {r['precision']:.1%}")
        print(f"  Recall:    {r['recall']:.1%}")
        print(f"  F1:        {r['f1']:.1%}")
        print(f"  Discrim:   {r['discrimination_rate']:.1%}")
    print("=" * 60)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
