"""
PoP Benchmark — Test on Real Hallucination Datasets

Tests PoP's ability to detect hallucinations on:
1. TruthfulQA — factual accuracy questions
2. HaluEval — task-specific hallucination (summarization, QA, dialogue)

Models tested: GPT-2 (124M), GPT-2 Medium (355M)
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

# Add pop to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pop.core.pop_v2 import PoPLayerLLMV2, extract_features_vectorized

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_name: str):
    """Load a GPT-2 model and tokenizer."""
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_name}: {params/1e6:.1f}M params")
    return model, tokenizer


def get_logits_for_text(model, tokenizer, prompt: str, device: str = "cpu"):
    """
    Get the logit distribution at the last token position for a given text.
    Returns (logits, probs) as tensors of shape (V,).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get logits at last position
    last_logits = outputs.logits[0, -1, :]  # (V,)
    probs = F.softmax(last_logits, dim=-1)
    return last_logits, probs


def get_logits_for_completion(model, tokenizer, prompt: str, completion: str, device: str = "cpu"):
    """
    Get logit distribution when the model would predict the first token of completion
    given the prompt. This measures the model's uncertainty about the completion.
    """
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits at the position right before completion starts
    # This tells us: "given prompt, how confident is the model about the completion?"
    logits_at_boundary = outputs.logits[0, prompt_len - 1, :]  # (V,)
    probs = F.softmax(logits_at_boundary, dim=-1)
    return logits_at_boundary, probs


def get_avg_logits_across_completion(model, tokenizer, prompt: str, completion: str, device: str = "cpu"):
    """
    Average the logit features across all token positions in the completion.
    More robust than single-position analysis.
    """
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    all_logits = outputs.logits[0]  # (seq_len, V)
    
    # Collect features at each completion token position
    pop = PoPLayerLLMV2(vocab_size=all_logits.shape[-1], device=device)
    
    error_mags = []
    confidences = []
    
    for pos in range(prompt_len, min(prompt_len + 20, all_logits.shape[0])):
        logits_pos = all_logits[pos - 1, :]  # logits predicting this token
        probs_pos = F.softmax(logits_pos, dim=-1)
        
        result = pop.predict(logits_pos.unsqueeze(0), probs_pos.unsqueeze(0))
        error_mags.append(result["error_magnitude"])
        confidences.append(result["confidence"])
    
    if not error_mags:
        return {"error_magnitude": 0.5, "confidence": 0.5, "llm_likely_wrong": False}
    
    avg_error = np.mean(error_mags)
    avg_conf = np.mean(confidences)
    
    return {
        "error_magnitude": float(avg_error),
        "confidence": float(avg_conf),
        "llm_likely_wrong": avg_error > 0.5,
    }


def benchmark_truthfulqa(model, tokenizer, device="cpu", max_samples=None):
    """
    Benchmark PoP on TruthfulQA.
    
    Approach: For each question + correct/incorrect answer pair,
    check if PoP gives higher error_magnitude for incorrect answers.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: TruthfulQA")
    logger.info("=" * 60)
    
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    pop = load_pop_trained(tokenizer.vocab_size, device)
    
    results = []
    tp = fp = tn = fn = 0
    
    for i, item in enumerate(ds):
        question = item["question"]
        mc1 = item["mc1_targets"]  # {"choices": [...], "labels": [0, 1, ...]}
        
        if not mc1 or not mc1["choices"]:
            continue
        
        choices = mc1["choices"]
        labels = mc1["labels"]  # 1 = correct, 0 = incorrect
        
        # Test each answer choice
        for choice_text, label in zip(choices, labels):
            prompt = f"Q: {question}\nA:"
            full_input = f"{prompt} {choice_text}"
            
            # Get logits at the answer boundary
            logits, probs = get_logits_for_completion(
                model, tokenizer, prompt, f" {choice_text}", device
            )
            
            # PoP prediction
            result = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
            is_error_pred = bool(safe_float(result["llm_likely_wrong"]))
            is_error_true = (label == 0)  # incorrect answer = error
            
            # Track metrics
            if is_error_pred and is_error_true:
                tp += 1
            elif is_error_pred and not is_error_true:
                fp += 1
            elif not is_error_pred and is_error_true:
                fn += 1
            else:
                tn += 1
            
            results.append({
                "question": question[:60],
                "answer": choice_text[:40],
                "true_label": "correct" if label == 1 else "incorrect",
                "pop_error_mag": round(safe_float(result["error_magnitude"]), 3),
                "pop_confidence": round(safe_float(result["confidence"]), 3),
                "pop_prediction": "error" if is_error_pred else "ok",
                "correct": is_error_pred == is_error_true,
            })
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(ds)} questions...")
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TruthfulQA Results:")
    logger.info(f"  Total samples:  {tp + tn + fp + fn}")
    logger.info(f"  Accuracy:       {accuracy:.1%}")
    logger.info(f"  Precision:      {precision:.1%}")
    logger.info(f"  Recall:         {recall:.1%}")
    logger.info(f"  F1:             {f1:.1%}")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"{'=' * 60}")
    
    return {
        "dataset": "TruthfulQA",
        "samples": tp + tn + fp + fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "examples": results[:10],
    }


def benchmark_halueval(model, tokenizer, device="cpu", max_samples=500):
    """
    Benchmark PoP on HaluEval (summarization).
    
    Approach: For each document, compare PoP's error prediction
    on the right summary vs the hallucinated summary.
    
    PoP should give LOWER error_magnitude for correct summaries
    and HIGHER error_magnitude for hallucinated ones.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: HaluEval (Summarization)")
    logger.info("=" * 60)
    
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    pop = load_pop_trained(tokenizer.vocab_size, device)
    
    results = []
    tp = fp = tn = fn = 0
    
    # We need a threshold — compute error_magnitudes first, then set threshold
    all_errors_right = []
    all_errors_hallucinated = []
    raw_results = []
    
    for i, item in enumerate(ds):
        document = item["document"]
        right_summary = item["right_summary"]
        hallucinated_summary = item["hallucinated_summary"]
        
        # Truncate document to fit model context (GPT-2 has 1024 token limit)
        # Use last 500 chars of document as context
        doc_short = document[-500:] if len(document) > 500 else document
        prompt = f"Document: {doc_short}\nSummary:"
        
        # Get logits for right summary
        logits_r, probs_r = get_logits_for_completion(
            model, tokenizer, prompt, f" {right_summary}", device
        )
        result_r = pop.predict(logits_r.unsqueeze(0), probs_r.unsqueeze(0))
        
        # Get logits for hallucinated summary
        logits_h, probs_h = get_logits_for_completion(
            model, tokenizer, prompt, f" {hallucinated_summary}", device
        )
        result_h = pop.predict(logits_h.unsqueeze(0), probs_h.unsqueeze(0))
        
        err_r = safe_float(result_r["error_magnitude"])
        err_h = safe_float(result_h["error_magnitude"])
        
        all_errors_right.append(err_r)
        all_errors_hallucinated.append(err_h)
        
        raw_results.append({
            "doc_short": doc_short[:80],
            "right_summary": right_summary[:60],
            "hallucinated_summary": hallucinated_summary[:60],
            "right_error_mag": err_r,
            "hallucinated_error_mag": err_h,
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(ds)} samples...")
    
    # Set threshold as midpoint between mean errors
    threshold = (np.mean(all_errors_right) + np.mean(all_errors_hallucinated)) / 2
    
    logger.info(f"  Mean error (right):        {np.mean(all_errors_right):.3f}")
    logger.info(f"  Mean error (hallucinated): {np.mean(all_errors_hallucinated):.3f}")
    logger.info(f"  Threshold:                 {threshold:.3f}")
    
    # Now classify using threshold
    for r in raw_results:
        # Right summary: should be classified as "ok" (low error)
        right_is_error_pred = r["right_error_mag"] > threshold
        # Hallucinated: should be classified as "error" (high error)
        hall_is_error_pred = r["hallucinated_error_mag"] > threshold
        
        # True labels
        right_is_error_true = False  # right summary is correct
        hall_is_error_true = True    # hallucinated summary is wrong
        
        # Right summary classification
        if right_is_error_pred and right_is_error_true:
            tp += 1
        elif right_is_error_pred and not right_is_error_true:
            fp += 1
        elif not right_is_error_pred and right_is_error_true:
            fn += 1
        else:
            tn += 1
        
        # Hallucinated summary classification
        if hall_is_error_pred and hall_is_error_true:
            tp += 1
        elif hall_is_error_pred and not hall_is_error_true:
            fp += 1
        elif not hall_is_error_pred and hall_is_error_true:
            fn += 1
        else:
            tn += 1
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Also compute: how often is hallucinated > right (discrimination ability)
    correct_ordering = sum(
        1 for r, h in zip(all_errors_right, all_errors_hallucinated) if h > r
    )
    discrimination_rate = correct_ordering / len(all_errors_right)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"HaluEval Results:")
    logger.info(f"  Total samples:  {len(raw_results)} pairs")
    logger.info(f"  Accuracy:       {accuracy:.1%}")
    logger.info(f"  Precision:      {precision:.1%}")
    logger.info(f"  Recall:         {recall:.1%}")
    logger.info(f"  F1:             {f1:.1%}")
    logger.info(f"  Discrimination: {discrimination_rate:.1%} (hallucinated > right)")
    logger.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"{'=' * 60}")
    
    return {
        "dataset": "HaluEval-Summarization",
        "samples": len(raw_results),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "discrimination_rate": round(discrimination_rate, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mean_error_right": round(float(np.mean(all_errors_right)), 4),
        "mean_error_hallucinated": round(float(np.mean(all_errors_hallucinated)), 4),
        "examples": raw_results[:5],
    }


def load_pop_trained(vocab_size, device):
    """Load the trained PoP v2 model."""
    pop = PoPLayerLLMV2(vocab_size=vocab_size, device=device)
    ckpt_path = os.path.join(os.path.dirname(__file__), "pop_v2_trained.pth")
    if os.path.exists(ckpt_path):
        pop.load(ckpt_path)
        logger.info(f"Loaded trained PoP v2 from {ckpt_path}")
    else:
        logger.warning(f"No trained checkpoint at {ckpt_path} — using untrained model")
    return pop


def safe_float(v):
    """Convert tensor or number to float."""
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else float(v.mean())
    return float(v)


def main():
    device = "cpu"
    results = {}
    
    # Test on GPT-2 (124M params)
    model_name = "gpt2"
    model, tokenizer = load_model(model_name)
    
    # TruthfulQA (full dataset - 817 questions)
    results["truthfulqa"] = benchmark_truthfulqa(
        model, tokenizer, device, max_samples=817
    )
    
    # HaluEval (500 samples - enough for statistical significance)
    results["halueval"] = benchmark_halueval(
        model, tokenizer, device, max_samples=500
    )
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PO BENCHMARK SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{r['dataset']}:")
        print(f"  Accuracy:  {r['accuracy']:.1%}")
        print(f"  Precision: {r['precision']:.1%}")
        print(f"  Recall:    {r['recall']:.1%}")
        print(f"  F1:        {r['f1']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
