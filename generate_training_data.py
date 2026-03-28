"""
Generate training data for PoP — FIX: bad examples use different features.

Strategy:
- GOOD (label=0): Use features from the LLM's natural next-token prediction.
  The model is confident (top-1 high prob), entropy is low.
- BAD (label=1): Take the wrong token, prepend it to context, get the LLM's
  NEXT prediction. This produces a confused distribution (higher entropy,
  lower top-1 prob, etc.) because the LLM is now predicting after an
  unnatural token.

This ensures the two classes have genuinely different feature distributions.
"""

import sys
import random
import numpy as np
import torch

sys.path.insert(0, "/root/.openclaw/workspace-main/pop-repo")
from pop.core.llm_base import create_llm

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water is composed of hydrogen and",
    "The speed of light is approximately",
    "The tallest mountain in the world is",
    "The chemical symbol for gold is",
    "The human body has",
    "The Great Wall of",
    "Shakespeare wrote the play",
    "The first president of the United States was",
    "Once upon a time there was a",
    "The early bird catches the",
    "A penny saved is a penny",
    "Actions speak louder than",
    "All that glitters is not",
    "Beauty is in the eye of the",
    "Every cloud has a silver",
    "Fortune favors the",
    "Good things come to those who",
    "Honesty is the best",
    "Photosynthesis converts sunlight into",
    "DNA stands for deoxyribonucleic",
    "The mitochondria is the powerhouse of the",
    "Newton's first law states that an object in motion",
    "Evolution by natural selection was proposed by",
    "The periodic table organizes elements by",
    "Quantum mechanics describes the behavior of",
    "The speed of sound in air is approximately",
    "E equals mc",
    "The theory of general relativity was published by",
    "The Roman Empire fell in the year",
    "World War II ended in",
    "The Declaration of Independence was signed in",
    "The French Revolution began in",
    "The Renaissance started in",
    "The pyramids of Giza are located in",
    "The ancient Greeks invented",
    "The printing press was invented by",
    "Christopher Columbus sailed to America in",
    "The Industrial Revolution began in",
    "The square root of 144 is",
    "Pi is approximately equal to",
    "A triangle has",
    "The derivative of x squared is",
    "Zero factorial equals",
    "The sum of angles in a triangle is",
    "An even number is divisible by",
    "A prime number has exactly",
    "The Pythagorean theorem states that a squared plus b squared equals",
    "The area of a circle is pi times the",
]


def extract_features_np(logits_np, probs_np):
    """Extract 16 features, numpy in, numpy out."""
    logits = torch.tensor(logits_np, dtype=torch.float32)
    probs = torch.tensor(probs_np, dtype=torch.float32)
    feat = []

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    feat.append(entropy)
    top1_prob, _ = torch.max(probs, dim=-1)
    feat.append(top1_prob)
    top3, _ = torch.topk(probs, 3, dim=-1)
    feat.append(torch.sum(top3, dim=-1))
    top10, _ = torch.topk(probs, 10, dim=-1)
    feat.append(torch.sum(top10, dim=-1))
    feat.append(torch.max(logits) - torch.min(logits))
    feat.append(torch.mean(logits))
    feat.append(torch.std(logits))
    feat.append(torch.sum(probs > 0.01, dim=-1).float())
    feat.append(torch.sum(probs > 0.1, dim=-1).float())
    sp, _ = torch.sort(probs)
    n = sp.shape[-1]
    feat.append(sp[int(0.25 * n)])
    feat.append(sp[int(0.50 * n)])
    feat.append(sp[int(0.75 * n)])
    feat.append(torch.var(probs))
    sd, _ = torch.sort(probs, descending=True)
    cs = torch.cumsum(sd, dim=-1)
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    gini = (2 * torch.sum(idx * sd) - (n + 1) * cs[-1]) / (n * cs[-1] + 1e-10)
    feat.append(gini)
    min_p = torch.min(probs)
    feat.append(torch.log(top1_prob + 1e-10) - torch.log(min_p + 1e-10))
    feat.append(torch.logsumexp(logits, dim=-1))

    return torch.cat([f.unsqueeze(-1) if f.dim() == 0 else f.unsqueeze(-1) for f in feat], dim=-1).numpy().astype(np.float32)


def main():
    print("Loading DistilGPT2...")
    llm = create_llm("distilgpt2")
    vocab = llm.vocab_size
    print(f"Model loaded. Vocab size: {vocab}")

    records = []

    for idx, prompt in enumerate(PROMPTS):
        if idx % 10 == 0:
            print(f"  Processing {idx+1}/{len(PROMPTS)}...")

        # --- GOOD example ---
        result = llm.predict_next_token(prompt, top_k=10)
        logits_np = llm.get_logits(prompt).cpu().numpy()
        probs_np = result["full_probs"]
        features_good = extract_features_np(logits_np, probs_np)

        top1_token = result["top_tokens"][0]

        records.append({
            "features": features_good,
            "label": 0,
            "prompt": prompt,
            "predicted_token": top1_token,
            "correct_token": top1_token,
        })

        # --- BAD example ---
        # Pick a wrong token (outside top-10) and append it to the prompt.
        # Then get the NEXT prediction — the LLM will be confused.
        top10_set = set(result["top_indices"])
        wrong_id = random.randint(0, vocab - 1)
        tries = 0
        while wrong_id in top10_set and tries < 200:
            wrong_id = random.randint(0, vocab - 1)
            tries += 1

        wrong_token = llm.tokenizer.decode(wrong_id)
        bad_context = prompt + wrong_token

        result_bad = llm.predict_next_token(bad_context, top_k=10)
        logits_bad = llm.get_logits(bad_context).cpu().numpy()
        probs_bad = result_bad["full_probs"]
        features_bad = extract_features_np(logits_bad, probs_bad)

        records.append({
            "features": features_bad,
            "label": 1,
            "prompt": prompt,
            "predicted_token": top1_token,
            "correct_token": wrong_token,
        })

    n = len(records)
    features_arr = np.stack([r["features"] for r in records])
    labels_arr = np.array([r["label"] for r in records], dtype=np.int32)
    prompts_arr = np.array([r["prompt"] for r in records], dtype=object)
    predicted_arr = np.array([r["predicted_token"] for r in records], dtype=object)
    correct_arr = np.array([r["correct_token"] for r in records], dtype=object)

    save_path = "/root/.openclaw/workspace-main/pop-repo/training_data.npy"
    np.save(save_path, {
        "features": features_arr,
        "labels": labels_arr,
        "prompts": prompts_arr,
        "predicted_tokens": predicted_arr,
        "correct_tokens": correct_arr,
    })

    # Verify
    f = features_arr
    l = labels_arr
    print(f"\n{'='*60}")
    print(f"Saved: {save_path}")
    print(f"Total: {n} (good={int((l==0).sum())}, bad={int((l==1).sum())})")
    print(f"Features shape: {f.shape}")
    print(f"\nGood (label=0) feature means (first 8): {f[l==0].mean(axis=0)[:8]}")
    print(f"Bad  (label=1) feature means (first 8): {f[l==1].mean(axis=0)[:8]}")
    print(f"Feature means differ: {not np.allclose(f[l==0].mean(axis=0), f[l==1].mean(axis=0))}")
    print(f"{'='*60}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
