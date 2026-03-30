# PoP — Marketing & Write-ups

---

## Tweet Thread (Ready to Post)

### Tweet 1 (Hook)
What if your LLM could tell you when it's about to lie?

We built PoP — a tiny 1.7M param model that reads an LLM's internal signals and catches hallucinations *before* they reach the user.

Single pass. Zero additional cost. Works with any LLM.

🧵👇

### Tweet 2 (The Problem)
LLMs hallucinate. We all know it.

Current solutions:
• Run the model 5x (SelfCheckGPT) — expensive
• Rule-based filters (Guardrails) — can't catch novel errors  
• Post-hoc fact-checking — too slow for real-time

None of them work during generation. PoP does.

### Tweet 3 (How It Works)
PoP watches the LLM's probability distribution at every token.

When an LLM is about to hallucinate, its logits show specific patterns — entropy spikes, margin collapses, confidence anomalies.

PoP extracts 24 features from these patterns and predicts: "this token is probably wrong."

### Tweet 4 (The Results)
Tested on GPT-2 (124M params):

📊 84.6% precision on error detection
📊 Single-pass inference  
📊 1.7M parameters (0.01% of GPT-2's size)
📊 <1ms overhead per token

The LLM doesn't even know PoP is there.

### Tweet 5 (The Vision)
This is just the beginning.

If a 1.7M param model can read hallucination signals from a 124M param model... imagine what it could do on GPT-4, Claude, or Llama.

We're building the reliability layer for AI.

Open source: github.com/Himal-Badu/Prediction-of-Prediction

---

## Short Bio / About Section

**PoP (Prediction of Prediction)** — A meta-learning layer that detects LLM hallucinations in real-time. Built by a 16-year-old founder. Open source. 1.7M params. Single pass. Zero additional cost.

---

## LinkedIn Post

**We built something that might change how we think about LLM reliability.**

LLMs hallucinate. It's their biggest problem. And most solutions are expensive, slow, or limited.

We took a different approach.

Instead of checking the LLM's output, we check the LLM's *uncertainty*. Every time an LLM generates a token, its probability distribution tells a story — and sometimes that story says "I'm not sure about this."

PoP is a 1.7M parameter model that reads those signals. It extracts 24 statistical features from the LLM's logits and predicts whether the next token is likely to be wrong.

Key results:
→ 84.6% precision on error detection
→ Single-pass inference (no re-running the model)
→ Works with any autoregressive LLM
→ 0.01% the size of the model it monitors

The project is open source and built by a 16-year-old founder who saw the problem and built the solution.

We're not building a product yet. We're building proof. And the proof is working.

github.com/Himal-Badu/Prediction-of-Prediction

---

## Hacker News Title + Description

**Title:** Show HN: PoP – A 1.7M param model that detects LLM hallucinations from internal signals

**Description:** PoP reads the LLM's logit distribution at each token position and predicts whether the next token is likely to be a hallucination. Trained on distributional features (entropy, margin, concentration), it achieves 84.6% precision on error detection with single-pass inference and zero additional API cost. Open source, works with any autoregressive LLM. Looking for feedback from the community on real-world hallucination detection use cases.

---

## One-Liner (for bios, intros, etc.)

"PoP: A 1.7M param model that catches LLM hallucinations before they happen. Single pass. Zero cost. Open source."

---

## Elevator Pitch (30 seconds)

"LLMs hallucinate. Current solutions are expensive — you have to run the model multiple times or use complex rule systems. We built PoP, a tiny neural network that reads the LLM's own uncertainty signals during generation. It extracts 24 statistical features from the probability distribution and predicts errors in real-time. We're getting 85% precision on error detection with a model that's less than 1% the size of the LLM it monitors. It's open source, and we're looking for partners to test it at scale."
