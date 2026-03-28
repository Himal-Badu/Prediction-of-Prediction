"""
PoP Benchmark: Raw LLM vs LLM + Trained PoP
=============================================
Compares DistilGPT2 raw predictions against predictions corrected
by the trained PoP (Prediction of Prediction) layer.

Metrics: error detection precision, recall, F1, accuracy comparison.
"""

import torch
import numpy as np
import json
import sys
import os
import time
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pop.core.llm_base import LLMBase
from pop.core.pop_layer_llm import LLMErrorPredictor


# ─── Test Prompts with Ground Truth ──────────────────────────────────────

TEST_PROMPTS = [
    # Factual
    {"prompt": "The capital of France is", "expected": "Paris"},
    {"prompt": "The largest planet in our solar system is", "expected": "Jupiter"},
    {"prompt": "Water freezes at a temperature of", "expected": "zero"},
    {"prompt": "The chemical symbol for gold is", "expected": "Au"},
    {"prompt": "The speed of light is approximately", "expected": "three hundred"},

    # Scientific
    {"prompt": "DNA stands for", "expected": "deoxyribonucleic"},
    {"prompt": "The powerhouse of the cell is the", "expected": "mitochondria"},
    {"prompt": "Photosynthesis converts sunlight into", "expected": "energy"},
    {"prompt": "Newton's first law describes", "expected": "inertia"},
    {"prompt": "The boiling point of water is", "expected": "100"},

    # Historical
    {"prompt": "World War II ended in the year", "expected": "1945"},
    {"prompt": "The first person to walk on the moon was", "expected": "Neil"},
    {"prompt": "The Great Wall of China was built to", "expected": "protect"},
    {"prompt": "The ancient city of Rome was founded in", "expected": "753"},
    {"prompt": "Cleopatra was the last pharaoh of", "expected": "Egypt"},

    # Creative / Challenging
    {"prompt": "The meaning of life according to Douglas Adams is", "expected": "42"},
    {"prompt": "In the novel 1984, the author warns about", "expected": "totalitarianism"},
    {"prompt": "The theory of relativity was proposed by", "expected": "Einstein"},
    {"prompt": "Shakespeare wrote the famous play called", "expected": "Hamlet"},
    {"prompt": "The Mona Lisa was painted by", "expected": "Leonardo"},
]


class PoPBenchmark:
    """
    Benchmark comparing raw LLM predictions vs LLM + trained PoP.
    """

    def __init__(self, model_path: str = "pop_trained.pth", device: str = "cpu"):
        self.device = device
        self.model_path = model_path

        print("=" * 70)
        print("  PoP BENCHMARK: Raw LLM vs LLM + Trained PoP")
        print("=" * 70)

        # Load LLM
        print("\n[1/3] Loading DistilGPT2...")
        self.llm = LLMBase(model_name="distilgpt2", device=device)
        self.llm.load()
        print(f"      ✓ Vocab size: {self.llm.vocab_size}")

        # Load trained PoP v1
        print("\n[2/3] Loading trained PoP model...")
        self.pop_model = LLMErrorPredictor(
            vocab_size=self.llm.vocab_size,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        self.pop_model.load_state_dict(state_dict)
        self.pop_model.eval()
        print(f"      ✓ Weights loaded from {model_path}")
        print(f"      ✓ Parameters: {sum(p.numel() for p in self.pop_model.parameters()):,}")

        # Error threshold (same as integration.py)
        self.error_threshold = 0.5

        print("\n[3/3] Ready to benchmark.\n")

    def _check_match(self, predicted_token: str, expected: str) -> bool:
        """Check if predicted token contains or matches expected answer."""
        pred_clean = predicted_token.strip().lower()
        exp_clean = expected.strip().lower()
        return exp_clean in pred_clean or pred_clean.startswith(exp_clean[:3])

    def _run_single(self, prompt_data: Dict[str, str], idx: int) -> Dict[str, Any]:
        """Run a single prompt through LLM and PoP."""
        prompt = prompt_data["prompt"]
        expected = prompt_data["expected"]

        # LLM prediction
        llm_result = self.llm.predict_next_token(prompt, top_k=20)
        top_token = llm_result["top_tokens"][0]
        top_prob = float(llm_result["top_probs"][0])
        top5_tokens = llm_result["top_tokens"][:5]
        top5_probs = [float(p) for p in llm_result["top_probs"][:5]]

        # Ground truth: is LLM correct?
        llm_correct = self._check_match(top_token, expected)
        top5_correct = any(self._check_match(t, expected) for t in top5_tokens)

        # PoP analysis
        logits = self.llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)

        with torch.no_grad():
            pop_out = self.pop_model(logits.unsqueeze(0), probs.unsqueeze(0))

        error_magnitude = pop_out["error_magnitude"].item()
        pop_confidence = pop_out["confidence"].item()
        error_direction = pop_out["error_direction"].item()

        # PoP decision
        pop_flags_error = error_magnitude > self.error_threshold

        # Simulate correction: if PoP flags, pick next best from top-20
        corrected_token = top_token
        corrected_prob = top_prob
        correction_applied = False

        if pop_flags_error:
            # Pick highest-probability alternative
            for i in range(1, min(20, len(llm_result["top_tokens"]))):
                alt_token = llm_result["top_tokens"][i]
                alt_prob = float(llm_result["top_probs"][i])
                corrected_token = alt_token
                corrected_prob = alt_prob
                correction_applied = True
                break

        corrected_correct = self._check_match(corrected_token, expected)

        # Determine if PoP's flag was useful
        # True Positive: PoP flags AND LLM was actually wrong
        # False Positive: PoP flags BUT LLM was correct
        # True Negative: PoP doesn't flag AND LLM was correct
        # False Negative: PoP doesn't flag BUT LLM was wrong
        if pop_flags_error and not llm_correct:
            pop_classification = "TP"
        elif pop_flags_error and llm_correct:
            pop_classification = "FP"
        elif not pop_flags_error and llm_correct:
            pop_classification = "TN"
        else:
            pop_classification = "FN"

        return {
            "index": idx,
            "prompt": prompt,
            "expected": expected,
            "llm_token": top_token,
            "llm_prob": top_prob,
            "llm_correct": llm_correct,
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs,
            "top5_correct": top5_correct,
            "error_magnitude": round(error_magnitude, 4),
            "pop_confidence": round(pop_confidence, 4),
            "error_direction": round(error_direction, 4),
            "pop_flags_error": pop_flags_error,
            "pop_classification": pop_classification,
            "corrected_token": corrected_token,
            "corrected_prob": round(corrected_prob, 4),
            "correction_applied": correction_applied,
            "corrected_correct": corrected_correct,
        }

    def run(self, prompts: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run the full benchmark."""
        prompts = prompts or TEST_PROMPTS

        print(f"Running benchmark on {len(prompts)} prompts...\n")
        print("-" * 70)

        results = []
        for i, p in enumerate(prompts):
            result = self._run_single(p, i + 1)
            results.append(result)

            status = "✓" if result["llm_correct"] else "✗"
            pop_flag = "⚠ FLAG" if result["pop_flags_error"] else "  ok "
            corrected = "→ " + result["corrected_token"].strip() if result["correction_applied"] else ""
            cls = result["pop_classification"]

            print(
                f"  [{i+1:2d}] {status} | PoP {pop_flag} [{cls}] | "
                f"LLM: '{result['llm_token'].strip()}' ({result['llm_prob']:.3f}) "
                f"{'✓' if result['llm_correct'] else '✗ expected: ' + result['expected']} "
                f"{corrected}"
            )

        # ─── Calculate Metrics ────────────────────────────────────────
        metrics = self._calculate_metrics(results)
        self._print_results(results, metrics)

        # ─── Save Results ─────────────────────────────────────────────
        output = {
            "benchmark": "PoP v1 — Raw LLM vs LLM + Trained PoP",
            "model": "distilgpt2",
            "pop_model": "LLMErrorPredictor v1 (16 features, 256 hidden)",
            "pop_weights": self.model_path,
            "num_prompts": len(prompts),
            "error_threshold": self.error_threshold,
            "metrics": metrics,
            "detailed_results": results,
        }

        save_path = os.path.join(os.path.dirname(self.model_path), "benchmark_results.json")
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {save_path}")

        return output

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate benchmark metrics."""
        n = len(results)

        # Raw LLM stats
        llm_correct = sum(1 for r in results if r["llm_correct"])
        llm_accuracy = llm_correct / n

        # PoP error detection stats
        tp = sum(1 for r in results if r["pop_classification"] == "TP")
        fp = sum(1 for r in results if r["pop_classification"] == "FP")
        tn = sum(1 for r in results if r["pop_classification"] == "TN")
        fn = sum(1 for r in results if r["pop_classification"] == "FN")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # After PoP correction
        # Only count corrections that were actually applied
        corrections_applied = [r for r in results if r["correction_applied"]]
        corrections_helped = sum(1 for r in corrections_applied if r["corrected_correct"])
        corrections_hurt = sum(1 for r in corrections_applied if not r["corrected_correct"] and r["llm_correct"])

        # Simulate corrected accuracy: where correction was applied, use corrected result
        corrected_correct = 0
        for r in results:
            if r["correction_applied"]:
                if r["corrected_correct"]:
                    corrected_correct += 1
            else:
                if r["llm_correct"]:
                    corrected_correct += 1
        corrected_accuracy = corrected_correct / n

        # Top-5 accuracy
        top5_correct = sum(1 for r in results if r["top5_correct"])

        return {
            "total_prompts": n,
            "llm_correct": llm_correct,
            "llm_accuracy": round(llm_accuracy, 4),
            "top5_accuracy": round(top5_correct / n, 4),

            # Error detection
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "error_detection_precision": round(precision, 4),
            "error_detection_recall": round(recall, 4),
            "error_detection_f1": round(f1, 4),

            # Correction impact
            "corrections_applied": len(corrections_applied),
            "corrections_helped": corrections_helped,
            "corrections_hurt": corrections_hurt,
            "corrected_accuracy": round(corrected_accuracy, 4),

            # Improvement
            "accuracy_delta": round(corrected_accuracy - llm_accuracy, 4),
        }

    def _print_results(self, results: List[Dict], metrics: Dict):
        """Print formatted results table."""
        print("\n" + "=" * 70)
        print("  BENCHMARK RESULTS")
        print("=" * 70)

        # Accuracy comparison
        print("\n┌─────────────────────────────────────────────┐")
        print("│          ACCURACY COMPARISON                │")
        print("├─────────────────────────────────────────────┤")
        print(f"│  Raw LLM Top-1 Accuracy:     {metrics['llm_accuracy']:.1%}           │")
        print(f"│  Raw LLM Top-5 Accuracy:     {metrics['top5_accuracy']:.1%}           │")
        print(f"│  LLM + PoP Accuracy:         {metrics['corrected_accuracy']:.1%}           │")
        delta = metrics['accuracy_delta']
        sign = "+" if delta >= 0 else ""
        print(f"│  Accuracy Delta:             {sign}{delta:.1%}           │")
        print("└─────────────────────────────────────────────┘")

        # Error detection
        print("\n┌─────────────────────────────────────────────┐")
        print("│       PoP ERROR DETECTION METRICS           │")
        print("├─────────────────────────────────────────────┤")
        print(f"│  True Positives  (TP):  {metrics['true_positives']:3d}                  │")
        print(f"│  False Positives (FP):  {metrics['false_positives']:3d}                  │")
        print(f"│  True Negatives  (TN):  {metrics['true_negatives']:3d}                  │")
        print(f"│  False Negatives (FN):  {metrics['false_negatives']:3d}                  │")
        print("├─────────────────────────────────────────────┤")
        print(f"│  Precision:             {metrics['error_detection_precision']:.4f}              │")
        print(f"│  Recall:                {metrics['error_detection_recall']:.4f}              │")
        print(f"│  F1 Score:              {metrics['error_detection_f1']:.4f}              │")
        print("└─────────────────────────────────────────────┘")

        # Corrections
        print("\n┌─────────────────────────────────────────────┐")
        print("│          CORRECTION ANALYSIS                │")
        print("├─────────────────────────────────────────────┤")
        print(f"│  Corrections Applied:       {metrics['corrections_applied']:3d}              │")
        print(f"│  Corrections Helped:        {metrics['corrections_helped']:3d}              │")
        print(f"│  Corrections Hurt:          {metrics['corrections_hurt']:3d}              │")
        print("└─────────────────────────────────────────────┘")

        # Detailed table
        print(f"\n{'─' * 70}")
        print(f"  {'#':>3}  {'Status':>6}  {'PoP':>6}  {'LLM Token':<15} {'Expected':<15} {'Error Mag':>9}")
        print(f"{'─' * 70}")
        for r in results:
            status = "✓" if r["llm_correct"] else "✗"
            pop_s = "FLAG" if r["pop_flags_error"] else "—"
            print(
                f"  {r['index']:>3}  {status:>6}  {pop_s:>6}  "
                f"{r['llm_token'].strip():<15} {r['expected']:<15} "
                f"{r['error_magnitude']:>9.4f}"
            )
        print(f"{'─' * 70}")


def main():
    benchmark = PoPBenchmark(
        model_path="pop_trained.pth",
        device="cpu"
    )
    results = benchmark.run()


if __name__ == "__main__":
    main()
