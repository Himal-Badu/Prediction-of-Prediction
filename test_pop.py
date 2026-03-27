#!/usr/bin/env python3
"""
Test script to demonstrate PoP working with LLM.
This tests the complete system: LLM + PoP Layer + Safety Guard.
"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')

from pop.core.llm_base import create_llm
from pop.core.pop_layer_llm import create_pop_llm
from pop.core.integration import create_pop_system
import torch

print("=" * 60)
print("🔮 PoP (Prediction-of-Prediction) Test")
print("=" * 60)

# Step 1: Load the LLM
print("\n📥 Step 1: Loading Base LLM (distilgpt2)...")
llm = create_llm("distilgpt2")
print(f"✅ LLM loaded! Vocab size: {llm.vocab_size}")

# Step 2: Create PoP layer
print("\n🔧 Step 2: Creating PoP Layer...")
pop = create_pop_llm(vocab_size=llm.vocab_size)
print(f"✅ PoP layer created! Device: {pop.device}")

# Step 3: Test LLM inference
print("\n🔍 Step 3: Testing LLM inference...")
test_texts = [
    "The capital of France is",
    "Once upon a time in a",
    "The chemical symbol for gold is",
    "2 + 2 =",
    "The largest planet in our solar system is"
]

for text in test_texts:
    result = llm.predict_next_token(text, top_k=5)
    print(f"\n  Input: \"{text}...\"")
    print(f"  Top prediction: \"{result['top_tokens'][0]}\" (prob: {result['top_probs'][0]:.3f})")

# Step 4: Test PoP analysis
print("\n" + "=" * 60)
print("📊 Step 4: Testing PoP Analysis")
print("=" * 60)

for text in test_texts:
    logits = llm.get_logits(text)
    probs = torch.softmax(logits, dim=-1)
    
    pop_result = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))
    
    print(f"\n  Text: \"{text}...\"")
    print(f"    Error magnitude: {pop_result['error_magnitude']:.3f}")
    print(f"    Confidence:      {pop_result['confidence']:.3f}")
    print(f"    Error direction: {pop_result['error_direction']:.3f}")
    print(f"    LLM likely wrong: {pop_result['llm_likely_wrong']}")
    print(f"    Should correct:   {pop_result['should_correct']}")

# Step 5: Train PoP on synthetic examples (simulated)
print("\n" + "=" * 60)
print("📚 Step 5: Training PoP on synthetic examples")
print("=" * 60)

# Create some synthetic training examples
# These simulate cases where the LLM was wrong
training_examples = []

# Example 1: LLM predicted wrong token
text = "The capital of France is"
logits = llm.get_logits(text)
probs = torch.softmax(logits, dim=-1)

# Simulate wrong prediction (predicted "Paris" but should be "Paris")
# In practice, we'd collect real examples
print("  Training on synthetic examples...")
for i in range(5):
    loss = pop.train_step(
        logits.unsqueeze(0),
        probs.unsqueeze(0),
        error_magnitude=0.8,  # High error
        confidence=0.9,      # LLM was confident but wrong
        error_direction=0.7  # Overconfident
    )
    print(f"    Step {i+1}: loss = {loss['loss']:.4f}")

print("\n✅ PoP trained on synthetic examples!")

# Step 6: Test integrated system with safety guard
print("\n" + "=" * 60)
print("🛡️ Step 6: Testing Integrated System (with Safety Guard)")
print("=" * 60)

system = create_pop_system("distilgpt2")

test_phrases = [
    "The quick brown",
    "In the year 2025,",
    "Machine learning is a",
    "The meaning of life is"
]

for phrase in test_phrases:
    result = system.predict(phrase)
    
    print(f"\n  Input: \"{phrase}...\"")
    print(f"    LLM prediction: \"{result.llm_token}\" (prob: {result.llm_prob:.3f})")
    print(f"    PoP error magnitude: {result.pop_error_magnitude:.3f}")
    print(f"    PoP confidence: {result.pop_confidence:.3f}")
    print(f"    Should correct: {result.should_correct}")
    print(f"    Final token: \"{result.final_token}\" (prob: {result.final_prob:.3f})")
    print(f"    Correction applied: {result.correction_applied}")

# Step 7: Statistics
print("\n" + "=" * 60)
print("📈 Statistics")
print("=" * 60)

stats = system.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("Built by Himal Badu, 16-year-old AI founder")
print("=" * 60)