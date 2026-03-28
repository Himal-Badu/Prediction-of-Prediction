"""
PoP Supervised Training Script
Trains LLMErrorPredictor on pre-computed feature data.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pop.core.llm_base import LLMBase
from pop.core.pop_layer_llm import LLMErrorPredictor


def load_training_data(path):
    data = np.load(path, allow_pickle=True).item()
    print(f"Training data loaded from {path}")
    print(f"  Features shape: {data['features'].shape}")
    print(f"  Labels shape:   {data['labels'].shape}")
    print(f"  Labels distribution: {np.bincount(data['labels'].astype(int))}")
    return data


def train(model, X, y, epochs=50, lr=0.001, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    n = X.shape[0]

    print(f"\n{'='*60}")
    print(f"Training PoP: {epochs} epochs, {n} samples, batch_size={batch_size}")
    print(f"{'='*60}")

    model.train()

    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = torch.randperm(n)
        X_shuf = X[perm]
        y_shuf = y[perm]

        total_loss = 0.0
        total_correct = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            optimizer.zero_grad()

            # Use full model forward pass (not manual layer chaining)
            # Pass pre-extracted features as both logits and probs inputs
            # The model will re-extract features from the 16-dim vectors
            outputs = model(xb, xb)

            # Clamp sigmoid output for numerical stability (avoid log(0))
            error_pred = outputs["error_magnitude"].clamp(min=1e-7, max=1 - 1e-7)

            loss = criterion(error_pred, yb)
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * (end - start)
            total_correct += ((error_pred > 0.5).float() == yb).sum().item()

        avg_loss = total_loss / n
        accuracy = total_correct / n

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    return model


def test_on_prompts(model, llm, test_prompts):
    print(f"\n{'='*60}")
    print(f"Testing on {len(test_prompts)} held-out prompts (not in training data)")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            logits = llm.get_logits(prompt)
            probs = torch.softmax(logits, dim=-1)
            outputs = model(logits.unsqueeze(0), probs.unsqueeze(0))

            em = outputs['error_magnitude'].item()
            conf = outputs['confidence'].item()
            ed = outputs['error_direction'].item()
            top_p = probs.max().item()

            status = "⚠️  LIKELY WRONG" if em > 0.5 else "✅ Looks OK"
            print(f"\n  [{i+1}] \"{prompt}\"")
            print(f"      {status}")
            print(f"      error_mag={em:.3f}  confidence={conf:.3f}  direction={ed:.3f}  top_prob={top_p:.3f}")


def main():
    data = load_training_data("training_data.npy")
    features = data["features"]
    labels = data["labels"]

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    print("\nLoading DistilGPT2...")
    llm = LLMBase(model_name="distilgpt2")
    llm.load()
    print(f"LLM loaded. Vocab size: {llm.vocab_size}")

    model = LLMErrorPredictor(vocab_size=llm.vocab_size, hidden_dim=256, num_layers=2, dropout=0.2)
    print(f"\nLLMErrorPredictor: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    model = train(model, X, y, epochs=50, lr=0.001, batch_size=16)

    # Final eval
    model.eval()
    with torch.no_grad():
        fn = model.feature_norm(X)
        h = model.hidden(fn)
        pred = torch.sigmoid(model.error_head(h)).squeeze(-1)
        acc = ((pred > 0.5).float() == y).sum().item() / len(y)
    print(f"\nFinal training accuracy: {acc:.2%}")

    # Held-out test prompts — NONE of these appear in training_data.py
    test_prompts = [
        "The inventor of the telephone was",
        "Mount Everest is in",
        "The internet was invented in",
        "Photosynthesis converts sunlight into",
        "DNA stands for deoxyribonucleic",
        "The Roman Empire fell in the year",
        "The French Revolution began in",
        "The printing press was invented by",
        "An even number is divisible by",
        "E equals mc",
    ]
    test_on_prompts(model, llm, test_prompts)

    save_path = "pop_trained.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Model weights saved to {save_path}")


if __name__ == "__main__":
    main()
