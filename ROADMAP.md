# PoP Roadmap — Solve Hallucination, Then Scale

> **The only question that matters:** Does PoP actually solve the hallucination problem?
> **If yes:** Fund it, build an AI lab, change the world.
> **If no:** Find out fast, iterate, prove it.

---

## Phase 1: Prove It (Now → April)
**Goal:** Demonstrate PoP catches real hallucinations on a real benchmark.

### Step 1: Test on DistilGPT-2 (we're here)
- [x] v2 trained: 84.6% F1 on synthetic data
- [ ] **Next: Run PoP on TruthfulQA** — 817 questions, real factual errors
  - This is the moment of truth. Synthetic benchmarks ≠ real hallucinations.
  - If F1 stays above 80% → core hypothesis holds
  - If F1 drops below 70% → investigate, iterate

### Step 2: Test on HaluEval
- 35K samples across QA, dialogue, summarization
- Different error types: factual, logical, fabricated details
- Tests if PoP generalizes beyond factual QA

### Step 3: Document Results
- Clean benchmark report: precision, recall, F1 on each dataset
- Example: "PoP caught this hallucination, here's the logit signal that gave it away"
- This becomes your proof. No product, no pricing. Just proof.

**Exit criteria:** PoP achieves >75% F1 on TruthfulQA AND HaluEval. If not, iterate on architecture before moving on.

---

## Phase 2: Scale It (May → June)
**Goal:** Retrain with more parameters, test on a harder model.

### Step 1: Scale PoP Architecture
- Current: 400K params, 24 features, residual blocks
- Target: 2-5M params, deeper architecture
- Add: multi-depth feature extraction (read intermediate layers, not just final logits)
- Add: sequence-level detection (not just per-token, but "this whole sentence is wrong")

### Step 2: Test on Llama-3-8B
- Real production model, 8B parameters, much harder than DistilGPT-2
- If PoP features transfer → you've proven model-agnostic capability
- If they don't → retrain specialist on Llama internals, still a win

### Step 3: Test on GPT-4 via API
- Limited to final-logit features (no internal access)
- This is the hardest test: can PoP work with only what the API gives you?
- Even 70% F1 here would be a massive result

**Exit criteria:** >80% F1 on Llama-3-8B. GPT-4 results are bonus, not required.

---

## Phase 3: Publish It (July)
**Goal:** Go public. Share the proof.

### Step 1: arXiv Paper
- Title: "PoP: Meta-Learning for Real-Time Hallucination Detection"
- Sections: Method, Experiments (TruthfulQA, HaluEval, Llama-3), Results, Ablations
- Key claim: "400K-param model catches 85% of hallucinations in a single pass at zero additional cost"
- Publish on arXiv regardless. The proof speaks for itself.

### Step 2: Open-Source Everything
- Clean repo, documentation, pip-installable package
- Colab notebook: "See PoP catch hallucinations in 5 minutes"
- Let the code do the talking

### Step 3: Share
- Twitter/X: results, demos, "we built this"
- Hacker News, Reddit ML, AI Twitter
- Not marketing — just showing what it does

**Exit criteria:** Paper on arXiv, code public, community can reproduce results.

---

## Phase 4: Fund It (August → September)
**Goal:** Raise funding to build the AI lab.

### The Pitch (30 seconds)
> "We built a 400K parameter model that catches LLM hallucinations with 85% accuracy. Single pass. Zero additional cost. Works with any LLM. Here's the paper. Here's the code. Here's the benchmark. We need funding to scale this to production models and build the reliability layer for AI."

### What You Need
- Proof from Phase 1-3 (benchmarks, paper, working code)
- Clear ask: $500K-$1.5M pre-seed
- Use of funds: GPU compute, hire 1-2 engineers, expand to more model families

### Who to Talk To
- AI-focused VCs (if you want to go VC route)
- AI safety orgs (if aligned with their mission)
- Strategic acquirers who need hallucination detection (enterprise AI companies)

### What NOT to Do Yet
- Don't build a product
- Don't hire a team
- Don't worry about revenue
- Just prove the science, then fund the science

**Exit criteria:** Funding commitment or clear interest from investors.

---

## Phase 5: Build the Lab (Post-Funding)
**Goal:** Expand PoP into a real AI research lab.

### With Funding, You Can:
- Train on more model families (Llama-70B, Mistral, Mixtral, Claude)
- Hire researchers to explore: multimodal PoP, online learning, adversarial robustness
- Build the infrastructure: training clusters, evaluation pipelines
- Publish at NeurIPS, ICML, ACL
- Eventually: build the reliability layer that every LLM uses

### The Lab Vision
- **Not** a product company (at first)
- **An AI research lab** focused on making LLMs reliable
- PoP is the first project. More will come.
- Think: Anthropic's safety focus, but for reliability/accuracy

---

## Current Status — Where Are We?

| Phase | Status | Key Metric |
|-------|--------|------------|
| **1. Prove It** | 🟡 In Progress | v2: 84.6% F1 on synthetic. Need real benchmarks. |
| **2. Scale It** | ⬜ Not Started | Waiting on Phase 1 results |
| **3. Publish It** | ⬜ Not Started | Waiting on Phase 2 results |
| **4. Fund It** | ⬜ Not Started | Waiting on Phase 3 proof |
| **5. Build Lab** | ⬜ Not Started | Waiting on Phase 4 funding |

### Immediate Next Action
**Run PoP on TruthfulQA.** This is the first real test. Everything depends on this result.

---

## What We're NOT Doing Right Now

- ❌ Building a product
- ❌ Designing APIs
- ❌ Pricing models
- ❌ Community strategy
- ❌ Marketing
- ❌ Hiring

All of that comes AFTER we prove the science works. First: proof. Then: everything else.

---

*Updated: March 30, 2026*
*Owner: Himal Badu*
*Next review: After TruthfulQA results*
