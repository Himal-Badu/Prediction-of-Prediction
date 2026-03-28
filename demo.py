"""
PoP Demo — Interactive demonstration of the PoP (Prediction of Prediction) layer.

Shows: Input → LLM Prediction → PoP Analysis → Final Output
Clean, formatted output suitable for presentations and demos.
"""

import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pop.core.llm_base import LLMBase
from pop.core.pop_layer_llm import LLMErrorPredictor


DEMO_PROMPTS = [
    "The capital of France is",
    "Albert Einstein was born in the year",
    "The chemical formula for water is",
    "In computer science, an algorithm is",
    "The speed of sound in air is approximately",
    "Shakespeare wrote the tragedy called",
    "The mitochondria is the",
    "The first president of the United States was",
    "Light travels faster than",
    "The Great Wall of China was built to",
]


class PoPDemo:
    """
    Clean demo of LLM + PoP integration.
    """

    BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██████╗  ██████╗ ██████╗                                ║
║     ██╔══██╗██╔═══██╗██╔══██╗                               ║
║     ██████╔╝██║   ██║██████╔╝                               ║
║     ██╔═══╝ ██║   ██║██╔═══╝                                ║
║     ██║     ╚██████╔╝██║                                    ║
║     ╚═╝      ╚═════╝ ╚═╝                                    ║
║                                                              ║
║     Prediction of Prediction                                 ║
║     Meta-Learning Layer for LLMs                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

    def __init__(self, model_path: str = "pop_trained.pth", device: str = "cpu"):
        self.device = device
        self.error_threshold = 0.5

        print(self.BANNER)
        print("  Loading models...")

        # Load LLM
        self.llm = LLMBase(model_name="distilgpt2", device=device)
        self.llm.load()

        # Load trained PoP
        self.pop_model = LLMErrorPredictor(
            vocab_size=self.llm.vocab_size,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        self.pop_model.load_state_dict(state_dict)
        self.pop_model.eval()

        print(f"  ✓ DistilGPT2 loaded (vocab: {self.llm.vocab_size:,})")
        print(f"  ✓ PoP v1 loaded (trained weights)")
        print()

    def analyze(self, prompt: str) -> None:
        """Run a single prompt through the full pipeline and display results."""

        # ── Step 1: LLM Prediction ──────────────────────────────────
        print("┌" + "─" * 58 + "┐")
        print(f"│  INPUT: \"{prompt}\"".ljust(59) + "│")
        print("└" + "─" * 58 + "┘")

        llm_result = self.llm.predict_next_token(prompt, top_k=10)
        top_token = llm_result["top_tokens"][0]
        top_prob = float(llm_result["top_probs"][0])

        print("\n  ┌─ LLM PREDICTION (DistilGPT2)")
        print(f"  │  Top-1 Token:  '{top_token.strip()}'  (prob: {top_prob:.4f})")
        print(f"  │")
        print(f"  │  Top-5 candidates:")
        for i in range(5):
            t = llm_result["top_tokens"][i]
            p = float(llm_result["top_probs"][i])
            bar = "█" * int(p * 40)
            print(f"  │    {i+1}. '{t.strip():<15}' {p:.4f}  {bar}")
        print(f"  └─")

        # ── Step 2: PoP Analysis ────────────────────────────────────
        logits = self.llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)

        with torch.no_grad():
            pop_out = self.pop_model(logits.unsqueeze(0), probs.unsqueeze(0))

        error_mag = pop_out["error_magnitude"].item()
        pop_conf = pop_out["confidence"].item()
        direction = pop_out["error_direction"].item()
        likely_wrong = error_mag > self.error_threshold

        # Direction label
        if direction > 0.3:
            dir_label = "OVERCONFIDENT ↑"
        elif direction < -0.3:
            dir_label = "UNDERCONFIDENT ↓"
        else:
            dir_label = "CALIBRATED ─"

        # Verdict
        if likely_wrong:
            verdict = "⚠️  HIGH ERROR RISK — LLM may be wrong"
            verdict_color = "⚠"
        else:
            verdict = "✅ LOW ERROR RISK — LLM likely correct"
            verdict_color = "✓"

        print(f"\n  ┌─ PoP ANALYSIS")
        print(f"  │  Error Magnitude:  {error_mag:.4f}  {'████' if error_mag > 0.7 else '██▓░' if error_mag > 0.5 else '██░░' if error_mag > 0.3 else '█░░░'}")
        print(f"  │  PoP Confidence:   {pop_conf:.4f}")
        print(f"  │  Error Direction:  {direction:+.4f}  ({dir_label})")
        print(f"  │")
        print(f"  │  {verdict}")
        print(f"  └─")

        # ── Step 3: Decision ────────────────────────────────────────
        print(f"\n  ┌─ DECISION")
        if likely_wrong:
            # Find best alternative
            best_alt = llm_result["top_tokens"][1]
            best_alt_prob = float(llm_result["top_probs"][1])
            print(f"  │  Action:   CORRECT (PoP override)")
            print(f"  │  Original: '{top_token.strip()}' ({top_prob:.4f})")
            print(f"  │  Corrected: '{best_alt.strip()}' ({best_alt_prob:.4f})")
            final_token = best_alt.strip()
        else:
            print(f"  │  Action:   TRUST LLM")
            print(f"  │  Output:   '{top_token.strip()}' ({top_prob:.4f})")
            final_token = top_token.strip()

        print(f"  └─")

        # ── Full generated text ─────────────────────────────────────
        print(f"\n  ── Generated Text ──")
        generated = self.llm.generate(prompt, max_new_tokens=20, temperature=0.7)
        print(f"  {generated}")
        print(f"\n{'─' * 60}\n")

    def run_demo(self, prompts=None):
        """Run demo on a list of prompts."""
        prompts = prompts or DEMO_PROMPTS

        print(f"  Running {len(prompts)} demo prompts...\n")

        for i, prompt in enumerate(prompts):
            print(f"  ══ DEMO [{i+1}/{len(prompts)}] ══\n")
            self.analyze(prompt)

        print("\n  ✓ Demo complete.\n")

    def interactive(self):
        """Interactive mode — take user input."""
        print("  Interactive mode. Type a prompt, or 'quit' to exit.\n")

        while True:
            try:
                prompt = input("  Your prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break

            print()
            self.analyze(prompt)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PoP Demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt to analyze")
    parser.add_argument("--weights", "-w", type=str, default="pop_trained.pth", help="Path to trained weights")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    args = parser.parse_args()

    demo = PoPDemo(model_path=args.weights, device=args.device)

    if args.prompt:
        demo.analyze(args.prompt)
    elif args.interactive:
        demo.interactive()
    else:
        demo.run_demo()


if __name__ == "__main__":
    main()
