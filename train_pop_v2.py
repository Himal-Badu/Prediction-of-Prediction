"""
PoP v2 Training Script
Trains PoP v2 on diverse prompts using DistilGPT2, then benchmarks against v1.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pop.core.llm_base import LLMBase
from pop.core.pop_v2 import PoPLayerLLMV2, TrainingExampleV2, TrainingConfig
from pop.core.pop_layer_llm import LLMErrorPredictor


# ── 50 Diverse Training Prompts ───────────────────────────────────────────

TRAINING_PROMPTS = [
    # Original training_data.py prompts (balanced correct/wrong)
    # Completion-style (LLM tends to get these right)
    "She opened the", "He picked up the", "The cat sat on the",
    "I went to the", "I have a", "The movie was", "I think it is",
    "He went to the", "The book is", "We went to the", "She is very",
    "The food was", "I need to", "The children were", "She looked at the",
    "The dog ran", "I like to", "The train arrived", "It was a",
    "He said it was", "They went to the", "She said the", "I saw a",
    "The house was", "He looked at the",

    # Factual prompts (LLM tends to get these wrong)
    "The capital of France is", "The chemical symbol for gold is",
    "The largest planet is", "Shakespeare wrote", "World War II ended in",
    "The pyramids are in", "The Mona Lisa was painted by",
    "Newton's first law is about", "The square root of 144 is",
    "The opposite of hot is", "The speed of light is approximately",
    "The atomic number of carbon is", "The first man on the moon was",
    "Albert Einstein was born in", "The largest country by area is",
    "The longest river is the", "Pi is approximately", "Water boils at",
    "The chemical symbol for iron is", "Mount Everest is in",
    "The Great Wall of", "2 + 2 equals", "The color of grass is",
    "The sky is", "The smallest country is",
]

# Answers for correctness checking
CORRECT_ANSWERS = {
    "She opened the": " door", "He picked up the": " phone",
    "The cat sat on the": " floor", "I went to the": " hospital",
    "I have a": " lot", "The movie was": " a", "I think it is": " a",
    "He went to the": " hospital", "The book is": " a",
    "We went to the": " hospital", "She is very": " good",
    "The food was": " good", "I need to": " go",
    "The children were": " playing", "She looked at the": " door",
    "The dog ran": " away", "I like to": " play",
    "The train arrived": " at", "It was a": " great",
    "He said it was": " a", "They went to the": " store",
    "She said the": " same", "I saw a": " man",
    "The house was": " a", "He looked at the": " door",
    "The capital of France is": " Paris",
    "The chemical symbol for gold is": " Au",
    "The largest planet is": " Jupiter", "Shakespeare wrote": " Hamlet",
    "World War II ended in": " 1945", "The pyramids are in": " Egypt",
    "The Mona Lisa was painted by": " Leonardo",
    "Newton's first law is about": " inertia",
    "The square root of 144 is": " 12", "The opposite of hot is": " cold",
    "The speed of light is approximately": " 3",
    "The atomic number of carbon is": " 6",
    "The first man on the moon was": " Neil",
    "Albert Einstein was born in": " 1879",
    "The largest country by area is": " Russia",
    "The longest river is the": " Nile", "Pi is approximately": " 3",
    "Water boils at": " 100", "The chemical symbol for iron is": " Fe",
    "Mount Everest is in": " Nepal", "The Great Wall of": " China",
    "2 + 2 equals": " 4", "The color of grass is": " green",
    "The sky is": " blue", "The smallest country is": " Vatican",
}

# ── 15 Held-out Test Prompts ──────────────────────────────────────────────

TEST_PROMPTS = [
    # Completions (usually correct)
    "She ran to the", "The bird flew", "He opened his",
    "They walked to the", "I need a",
    # Factual (usually wrong)
    "Photosynthesis converts sunlight into",
    "DNA stands for deoxyribonucleic",
    "The Roman Empire fell in the year",
    "The French Revolution began in",
    "The printing press was invented by",
    "E equals mc",
    "The inventor of the telephone was",
    "An even number is divisible by",
    "The mitochondria is the",
    "Light year is a unit of",
]

TEST_ANSWERS = {
    "She ran to the": " door", "The bird flew": " away",
    "He opened his": " mouth", "They walked to the": " store",
    "I need a": " new", "Photosynthesis converts sunlight into": " glucose",
    "DNA stands for deoxyribonucleic": " acid",
    "The Roman Empire fell in the year": " 476",
    "The French Revolution began in": " 1789",
    "The printing press was invented by": " Gutenberg",
    "E equals mc": " squared",
    "The inventor of the telephone was": " Alexander",
    "An even number is divisible by": " two",
    "The mitochondria is the": " powerhouse",
    "Light year is a unit of": " distance",
}

# Known factual prompts where LLM typically fails
FACTUAL_PROMPTS = set([
    "The capital of France is", "The chemical symbol for gold is",
    "The largest planet is", "Shakespeare wrote", "World War II ended in",
    "The pyramids are in", "The Mona Lisa was painted by",
    "Newton's first law is about", "The square root of 144 is",
    "The opposite of hot is", "The speed of light is approximately",
    "The atomic number of carbon is", "The first man on the moon was",
    "Albert Einstein was born in", "The largest country by area is",
    "The longest river is the", "Pi is approximately", "Water boils at",
    "The chemical symbol for iron is", "Mount Everest is in",
    "The Great Wall of", "2 + 2 equals", "The color of grass is",
    "The sky is", "The smallest country is",
    "Photosynthesis converts sunlight into", "DNA stands for deoxyribonucleic",
    "The Roman Empire fell in the year", "The French Revolution began in",
    "The printing press was invented by", "E equals mc",
    "The inventor of the telephone was", "An even number is divisible by",
    "The mitochondria is the", "Light year is a unit of",
])


def create_training_examples(prompts, llm, answers_dict):
    """Create TrainingExampleV2 list from prompts using LLM logits/probs."""
    examples = []
    for prompt in prompts:
        logits = llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)

        answer = answers_dict.get(prompt, "")
        pred = llm.predict_next_token(prompt, top_k=1)
        top_token = pred["top_tokens"][0] if pred["top_tokens"] else ""
        is_correct = (top_token == answer)

        # Label: error_magnitude=1 if LLM is wrong, 0 if correct
        error_mag = 1.0 if not is_correct else 0.0
        # Confidence: based on top prob
        top_prob = float(pred["top_probs"][0]) if pred["top_probs"] else 0.5
        confidence = top_prob
        # Direction: positive = overconfident (wrong but high conf), negative = underconfident
        if is_correct:
            error_dir = -0.3 if top_prob > 0.5 else -0.7  # slightly underconfident direction
        else:
            error_dir = 0.7 if top_prob > 0.3 else 0.3    # overconfident if wrong with high prob

        examples.append(TrainingExampleV2(
            logits=logits.cpu(),
            probs=probs.cpu(),
            error_magnitude=error_mag,
            confidence=confidence,
            error_direction=error_dir,
        ))
    return examples


def evaluate_model(pop_layer, llm, prompts, answers_dict):
    """Evaluate PoP model on prompts. Returns predictions dict."""
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "details": []}

    for prompt in prompts:
        logits = llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)

        answer = answers_dict.get(prompt, "")
        pred = llm.predict_next_token(prompt, top_k=1)
        top_token = pred["top_tokens"][0] if pred["top_tokens"] else ""
        llm_correct = (top_token == answer)

        pop_result = pop_layer.predict(logits.unsqueeze(0), probs.unsqueeze(0))
        error_mag_val = pop_result["error_magnitude"]
        if hasattr(error_mag_val, 'item'):
            error_mag_val = error_mag_val.item()
        conf_val = pop_result["confidence"]
        if hasattr(conf_val, 'item'):
            conf_val = conf_val.item()
        pop_says_wrong = bool(pop_result["llm_likely_wrong"])

        # pop_says_wrong=True means PoP thinks LLM is wrong
        # llm_correct=False means LLM is actually wrong
        if pop_says_wrong and not llm_correct:
            results["tp"] += 1
        elif pop_says_wrong and llm_correct:
            results["fp"] += 1
        elif not pop_says_wrong and llm_correct:
            results["tn"] += 1
        else:
            results["fn"] += 1

        results["details"].append({
            "prompt": prompt,
            "llm_correct": llm_correct,
            "pop_says_wrong": pop_says_wrong,
            "error_mag": error_mag_val,
            "confidence": conf_val,
        })

    return results


def calc_metrics(results):
    """Calculate precision, recall, F1 from confusion matrix."""
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_v1(model, llm, prompts, answers_dict):
    """Evaluate v1 model on prompts."""
    model.eval()
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "details": []}

    with torch.no_grad():
        for prompt in prompts:
            logits = llm.get_logits(prompt)
            probs = torch.softmax(logits, dim=-1)

            answer = answers_dict.get(prompt, "")
            pred = llm.predict_next_token(prompt, top_k=1)
            top_token = pred["top_tokens"][0] if pred["top_tokens"] else ""
            llm_correct = (top_token == answer)

            # V1 expects logits/probs and extracts 16 features internally
            outputs = model(logits.unsqueeze(0), probs.unsqueeze(0))
            error_mag = outputs["error_magnitude"].item()
            pop_says_wrong = error_mag > 0.5

            if pop_says_wrong and not llm_correct:
                results["tp"] += 1
            elif pop_says_wrong and llm_correct:
                results["fp"] += 1
            elif not pop_says_wrong and llm_correct:
                results["tn"] += 1
            else:
                results["fn"] += 1

            results["details"].append({
                "prompt": prompt, "llm_correct": llm_correct,
                "pop_says_wrong": pop_says_wrong, "error_mag": error_mag,
            })

    return results


def main():
    print("=" * 70)
    print("PoP v2 Training & Benchmark")
    print("=" * 70)

    # ── Load LLM ──────────────────────────────────────────────────────
    print("\n[1/6] Loading DistilGPT2...")
    llm = LLMBase(model_name="distilgpt2")
    llm.load()
    print(f"  Vocab size: {llm.vocab_size}")

    # ── Generate Training Data ─────────────────────────────────────────
    print("\n[2/6] Generating training examples from 50 prompts...")
    train_examples = create_training_examples(TRAINING_PROMPTS, llm, CORRECT_ANSWERS)
    n_correct = sum(1 for e in train_examples if e.error_magnitude == 0.0)
    n_wrong = sum(1 for e in train_examples if e.error_magnitude == 1.0)
    print(f"  Total: {len(train_examples)} | Correct: {n_correct} | Wrong: {n_wrong}")

    # ── Train PoP v2 ──────────────────────────────────────────────────
    print("\n[3/6] Training PoP v2 (100 epochs)...")
    pop_v2 = PoPLayerLLMV2(
        vocab_size=llm.vocab_size,
        hidden_dim=512,
        num_residual_blocks=3,
        learning_rate=1e-3,
    )

    config = TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        lr_scheduler="cosine",
        validation_split=0.15,
        log_every=10,
    )

    history = pop_v2.train_batched(train_examples, config)
    print(f"\n  Training complete!")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Final train loss: {history['final_train_loss']:.4f}")

    # Print loss curve summary
    print("\n  Loss curve (selected epochs):")
    for rec in history["history"]:
        if rec["epoch"] % 20 == 0 or rec["epoch"] == 1 or rec["epoch"] == 100:
            print(f"    Epoch {rec['epoch']:3d}: train={rec['train_loss']:.4f}  val={rec['val_loss']:.4f}  lr={rec['lr']:.6f}")

    # Save v2 weights
    v2_path = "pop_v2_trained.pth"
    pop_v2.save(v2_path)
    print(f"\n  ✓ Saved v2 weights to {v2_path}")

    # ── Load v1 for comparison ─────────────────────────────────────────
    print("\n[4/6] Loading PoP v1 for comparison...")
    v1_model = LLMErrorPredictor(vocab_size=llm.vocab_size, hidden_dim=256, num_layers=2, dropout=0.2)
    v1_model.load_state_dict(torch.load("pop_trained.pth", map_location=pop_v2.device))
    v1_model.to(pop_v2.device)
    v1_model.eval()
    v1_params = sum(p.numel() for p in v1_model.parameters())
    print(f"  v1 params: {v1_params:,}")

    # ── Benchmark: Training Set Evaluation ─────────────────────────────
    print("\n[5/6] Benchmark on training set (50 prompts)...")
    v2_train_results = evaluate_model(pop_v2, llm, TRAINING_PROMPTS, CORRECT_ANSWERS)
    v1_train_results = evaluate_v1(v1_model, llm, TRAINING_PROMPTS, CORRECT_ANSWERS)

    v2_train_metrics = calc_metrics(v2_train_results)
    v1_train_metrics = calc_metrics(v1_train_results)

    # ── Benchmark: Held-out Test Set ───────────────────────────────────
    print("\n[6/6] Benchmark on held-out test set (15 prompts)...")
    v2_test_results = evaluate_model(pop_v2, llm, TEST_PROMPTS, TEST_ANSWERS)
    v1_test_results = evaluate_v1(v1_model, llm, TEST_PROMPTS, TEST_ANSWERS)

    v2_test_metrics = calc_metrics(v2_test_results)
    v1_test_metrics = calc_metrics(v1_test_results)

    # ── Print Results ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    TRAINING SET (50 prompts)                   │")
    print("├──────────────┬────────────┬────────────┬────────────┬─────────┤")
    print("│ Model        │ Precision  │ Recall     │ F1 Score   │ Accuracy│")
    print("├──────────────┼────────────┼────────────┼────────────┼─────────┤")
    print(f"│ PoP v1       │ {v1_train_metrics['precision']:>9.2%} │ {v1_train_metrics['recall']:>9.2%} │ {v1_train_metrics['f1']:>9.2%} │ {v1_train_metrics['accuracy']:>6.2%} │")
    print(f"│ PoP v2       │ {v2_train_metrics['precision']:>9.2%} │ {v2_train_metrics['recall']:>9.2%} │ {v2_train_metrics['f1']:>9.2%} │ {v2_train_metrics['accuracy']:>6.2%} │")
    print("└──────────────┴────────────┴────────────┴────────────┴─────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                  HELD-OUT TEST SET (15 prompts)                │")
    print("├──────────────┬────────────┬────────────┬────────────┬─────────┤")
    print("│ Model        │ Precision  │ Recall     │ F1 Score   │ Accuracy│")
    print("├──────────────┼────────────┼────────────┼────────────┼─────────┤")
    print(f"│ PoP v1       │ {v1_test_metrics['precision']:>9.2%} │ {v1_test_metrics['recall']:>9.2%} │ {v1_test_metrics['f1']:>9.2%} │ {v1_test_metrics['accuracy']:>6.2%} │")
    print(f"│ PoP v2       │ {v2_test_metrics['precision']:>9.2%} │ {v2_test_metrics['recall']:>9.2%} │ {v2_test_metrics['f1']:>9.2%} │ {v2_test_metrics['accuracy']:>6.2%} │")
    print("└──────────────┴────────────┴────────────┴────────────┴─────────┘")

    # Confusion matrices
    print("\nConfusion Matrix (Test Set):")
    for label, m in [("PoP v1", v1_test_metrics), ("PoP v2", v2_test_metrics)]:
        print(f"  {label}: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

    # Per-prompt test results for v2
    print("\nPer-prompt test results (PoP v2):")
    for d in v2_test_results["details"]:
        status = "✅" if d["llm_correct"] else "❌"
        pop_flag = "⚠️ WRONG" if d["pop_says_wrong"] else "✓ OK"
        print(f'  {status} "{d["prompt"]}" → PoP: {pop_flag} (err={d["error_mag"]:.3f})')

    # Model comparison
    v2_params = pop_v2.model.count_parameters()
    print(f"\nModel sizes:")
    print(f"  v1: {v1_params:,} parameters")
    print(f"  v2: {v2_params:,} parameters ({v2_params/v1_params:.1f}x)")
    print(f"  v2 features: 24 (vs v1's 16)")

    # Improvement summary
    delta_f1 = v2_test_metrics["f1"] - v1_test_metrics["f1"]
    delta_prec = v2_test_metrics["precision"] - v1_test_metrics["precision"]
    print(f"\n{'🎉' if delta_f1 > 0 else '📉'} v2 vs v1 on test set:")
    print(f"  Precision: {'+' if delta_prec >= 0 else ''}{delta_prec:.2%}")
    print(f"  F1 Score:  {'+' if delta_f1 >= 0 else ''}{delta_f1:.2%}")

    print("\n" + "=" * 70)
    print("Done!")

    # Save results to JSON for later reference
    import json
    results_summary = {
        "v1_train": v1_train_metrics,
        "v2_train": v2_train_metrics,
        "v1_test": v1_test_metrics,
        "v2_test": v2_test_metrics,
        "v1_params": v1_params,
        "v2_params": v2_params,
    }
    with open("pop_v2_benchmark.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n✓ Results saved to pop_v2_benchmark.json")


if __name__ == "__main__":
    main()
