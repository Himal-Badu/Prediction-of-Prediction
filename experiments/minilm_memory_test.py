"""Memory test: Can MiniLM load alongside GPT-2 in this environment?"""
import torch
import gc
import os

def mem_usage(label):
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    print(f"  [{label}] RSS: {mem:.0f} MB")

mem_usage("start")

# Load GPT-2
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading GPT-2...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained("gpt2")
mdl.eval()
mem_usage("GPT-2 loaded")

# Load MiniLM-L6 (smallest viable NLI cross-encoder)
from sentence_transformers import CrossEncoder
print("Loading MiniLM-L6...")
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")
mem_usage("MiniLM-L6 loaded")

# Test inference
print("\nTesting NLI inference...")
pairs = [("The cat sat on the mat", "A cat is on a mat")]
scores = nli_model.predict(pairs)
print(f"  NLI scores shape: {scores.shape}")
print(f"  Scores: {scores}")
mem_usage("after NLI inference")

# Test with GPT-2 too
print("\nTesting GPT-2 + NLI together...")
inp = tok("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = mdl(**inp)
logits = out.logits[0, -1, :]
probs = torch.softmax(logits, dim=-1)
print(f"  GPT-2 top-1: {probs.max():.4f}")

pairs2 = [("What is the capital of France?", "Paris")]
scores2 = nli_model.predict(pairs2)
print(f"  NLI scores: {scores2}")
mem_usage("both models active")

print("\n✅ Both models loaded and running!")
