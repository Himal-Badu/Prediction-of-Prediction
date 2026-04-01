# PoP v3 — Product Requirements Document (PRD)
## Confidence Scoring Layer for LLM Outputs
**Version:** 1.0 | **Date:** April 1, 2026 | **Author:** Romy (CEO), PoP Labs

---

## 1. PRODUCT VISION

### What We Are
PoP Labs — a lightweight, model-agnostic confidence scoring layer for LLM outputs.

### What We Build
**PoP v3: Universal Confidence Scorer**
A plug-in layer that takes whatever signal is available from an LLM (logits, token probabilities, or raw text) and returns a calibrated trust score with span-level risk flags.

### One-Line Pitch
> "Plug into any model. Get a trust score."

### Competitive Moat
> How well we extract signal under constraints — not the signal itself.

---

## 2. THREE-TIER ARCHITECTURE

### Tier System

| Tier | Signal Required | Input | Target Models | Expected Performance |
|------|----------------|-------|---------------|---------------------|
| **Full** | Raw logits | Per-token logit vectors | Open-weight (Llama, Mistral, Qwen) | Baseline (best) |
| **Lite** | Top-k token probabilities | Top-k probs per position | APIs with prob access (some OpenAI tiers, open APIs) | ~5-15% degradation expected |
| **Minimal** | Text only | Generated text + optional context | ALL models (GPT-4, Claude, Gemini, any) | ~15-30% degradation expected |

### Key Design Principle
Auto-detect available tier from input. Same API contract. Same output format. User doesn't need to specify tier.

---

## 3. ENGINEERING ARCHITECTURE

### System Components

```
┌─────────────────────────────────────────────┐
│                 PoP v3 API                  │
│  (FastAPI, streaming support)               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Tier     │  │ Tier     │  │ Tier     │  │
│  │ Detector │→ │ Router   │→ │ Scorer   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                             │
├──────────┬──────────┬───────────────────────┤
│ Full     │ Lite     │ Minimal               │
│ Engine   │ Engine   │ Engine                │
│ (logits) │ (top-k)  │ (text-only)           │
├──────────┴──────────┴───────────────────────┤
│          Unified Scoring Layer              │
│    (calibration + span detection)           │
└─────────────────────────────────────────────┘
```

### Module Breakdown

#### Module 1: Feature Extractor (`pop/features/`)
- `full_extractor.py` — 24 logit-based features (existing PoP v2 features)
- `lite_extractor.py` — Reduced feature set from top-k probabilities
- `minimal_extractor.py` — Text-only features (repetition, hedging, consistency heuristics)
- `base.py` — Abstract base class, shared utilities

#### Module 2: Tier Detector (`pop/router.py`)
- Inspects input payload
- Detects: logits present → Full, top-k probs present → Lite, text only → Minimal
- Routes to appropriate engine
- Fallback chain: Full → Lite → Minimal

#### Module 3: Scoring Engines (`pop/engines/`)
- `full_engine.py` — PoP v3 model (Transformer + domain conditioning)
- `lite_engine.py` — Reduced model for top-k input
- `minimal_engine.py` — Text-analysis engine (NLP heuristics + lightweight classifier)

#### Module 4: Unified Scorer (`pop/scorer.py`)
- Takes raw engine output
- Applies calibration (temperature scaling)
- Generates span-level risk flags
- Returns standardized `RiskScore` object

#### Module 5: API Layer (`pop/api/`)
- `main.py` — FastAPI app
- `routes.py` — Score endpoint
- `streaming.py` — Streaming support for real-time scoring
- `schemas.py` — Pydantic models for input/output

#### Module 6: Training Pipeline (`pop/training/`)
- `dataset.py` — Multi-domain dataset loader
- `trainer.py` — Training loop with focal loss
- `evaluator.py` — Evaluation with 3×3 matrix (tier × domain)
- `calibration.py` — Temperature scaling training

---

## 4. API CONTRACT

### Endpoint

```
POST /api/v1/score
```

### Request

```json
{
  "text": "The LLM generated output to evaluate",
  "logits": [[0.8, 0.1, 0.05], [0.7, 0.2, 0.1], ...],  // optional
  "token_probs": {"top_k": [0.8, 0.1], "tokens": ["the", "cat"]},  // optional
  "context": "Source documents if available",  // optional
  "metadata": {
    "model": "llama-3-8b",  // optional
    "domain": "general"     // optional, auto-detect if absent
  }
}
```

### Response

```json
{
  "risk_score": 0.72,
  "confidence": "medium",
  "tier_used": "lite",
  "label": "RISKY",
  "flagged_spans": [
    {
      "text": "specific claim",
      "start": 45,
      "end": 62,
      "risk_level": "high",
      "reason": "low token confidence + unstable phrasing",
      "score": 0.85
    }
  ],
  "grounded_spans": [
    {
      "text": "well-supported claim",
      "start": 10,
      "end": 30,
      "confidence": 0.91
    }
  ],
  "features_used": ["entropy", "top_k_margin", "repetition_score"],
  "latency_ms": 8.3
}
```

### Error Handling
- Missing logits + top-k → auto-fallback to Minimal tier
- Empty text → 400 error
- Malformed logits → 422 with detail

---

## 5. DATA & TRAINING

### Dataset Construction

**Target: 50K labeled samples across 3 domains**

| Domain | Size | Source | Labeling Method |
|--------|------|--------|-----------------|
| QA | 20K | TruthfulQA + HotpotQA + NaturalQuestions | Reference-based (correct/incorrect) |
| Summarization | 20K | XSum + CNN/DM | NLI entailment against source |
| Open-ended | 10K | Synthetic generation from multiple LLMs | Cross-model agreement |

### Per-Tier Feature Sets

**Full (24 features):**
- Entropy (sequence + token level)
- Top-k probability mass (k=1,5,10,20)
- Logit statistics (max, mean, std, margin)
- Perplexity
- Token-level entropy stats (mean, max, min, std)
- Probability concentration (Gini)

**Lite (12 features):**
- Approximated entropy from top-k probs
- Top-k mass distribution
- Probability spread
- Token confidence consistency
- Positional confidence decay

**Minimal (8 features):**
- Repetition score (n-gram self-overlap)
- Hedging language detection
- Sentence length variance
- Named entity consistency (within output)
- Factual specificity score (numbers, dates, names count)
- Syntactic coherence (dependency parse stability)
- Sentiment stability across sentences
- Self-contradiction detection (simple rules)

---

## 6. EXPERIMENT PLAN

### Experiment 1: Entropy Baseline (Week 1, Day 1-2)
**Question:** Does PoP's 24-feature model beat simple entropy?
**Setup:**
- Baseline: entropy threshold classifier
- Baseline: max-probability threshold classifier
- Model: PoP v2 (24 features + MLP)
- Dataset: TruthfulQA (4,114 samples, existing)
- Metrics: Precision, Recall, F1, AUC-ROC

**Decision gate:**
- PoP beats entropy by >5% F1 → proceed
- PoP ≈ entropy → investigate features, try new ones
- PoP < entropy → fundamental problem, rethink approach

### Experiment 2: Tier Degradation (Week 1-2)
**Question:** How much performance drops across tiers?
**Setup:**
- Train on Full (logits)
- Test on Full, Lite (simulated top-k), Minimal (text-only)
- Run on TruthfulQA first, then expand

**Decision gate:**
- <10% drop Full→Minimal → massive win, product is robust
- 10-20% drop → viable for Full+Lite, Minimal is fallback
- >20% drop → Minimal tier needs redesign

### Experiment 3: Multi-Domain × Multi-Tier Matrix (Week 2-3)
**3×3 matrix:**

| | QA | Summarization | Open-ended |
|--|-----|---------------|------------|
| Full | ? | ? | ? |
| Lite | ? | ? | ? |
| Minimal | ? | ? | ? |

This is the core result. Everything else follows from this.

---

## 7. TESTING STRATEGY

### Unit Tests
- Feature extractor correctness (each feature returns expected range)
- Tier detection logic (input → correct tier)
- Scoring engine output format validation
- Calibration curve (reliability diagram)

### Integration Tests
- Full pipeline: input → tier detect → score → output
- Fallback chain: Full fails → Lite → Minimal
- Streaming: real-time scoring with simulated token stream

### Robustness Tests
- Adversarial inputs (empty, very long, malformed logits)
- Edge cases (single token, all same token, non-English)
- Latency under load (target: <10ms per token for Full tier)

---

## 8. PROJECT STRUCTURE

```
pop-repo/
├── pop/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract feature extractor
│   │   ├── full_extractor.py    # 24 logit features
│   │   ├── lite_extractor.py    # 12 top-k features
│   │   └── minimal_extractor.py # 8 text-only features
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── full_engine.py       # Transformer-based scorer
│   │   ├── lite_engine.py       # Reduced model
│   │   └── minimal_engine.py    # Text heuristics + lightweight model
│   ├── router.py                # Tier detection & routing
│   ├── scorer.py                # Unified scoring + calibration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pop_v3.py            # Main architecture
│   │   └── calibration.py       # Temperature scaling
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Data loading
│   │   ├── trainer.py           # Training loop
│   │   └── evaluator.py         # 3×3 evaluation matrix
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── routes.py            # Endpoints
│       ├── streaming.py         # Real-time scoring
│       └── schemas.py           # Pydantic models
├── tests/
│   ├── test_features.py
│   ├── test_engines.py
│   ├── test_router.py
│   ├── test_scorer.py
│   └── test_api.py
├── experiments/
│   ├── entropy_baseline.py      # Experiment 1
│   ├── tier_degradation.py      # Experiment 2
│   └── multi_domain_matrix.py   # Experiment 3
├── data/
│   └── README.md                # Dataset sources & loading instructions
├── docs/
│   └── prd.md                   # This document
├── benchmarks/
│   └── results/                 # Experiment results (gitignored from GitHub)
├── scripts/
│   ├── generate_data.py         # Synthetic data generation
│   └── run_experiments.py       # Run all experiments
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## 9. FOUR-WEEK SPRINT PLAN

### Week 1: Foundation & Signal Strength
| Day | Task | Owner | Output |
|-----|------|-------|--------|
| 1 | Implement feature extractors (Full/Lite/Minimal) | ML Engineer | `pop/features/` module |
| 1 | Implement tier router | API Engineer | `pop/router.py` |
| 2 | Entropy baseline experiment | ML Engineer | Experiment 1 results |
| 3 | Train PoP v3 on existing TruthfulQA data | ML Engineer | Trained model |
| 3-4 | Build tier degradation experiment | ML Engineer | Experiment 2 results |
| 5 | Code review, fix issues, document findings | All | Week 1 report |

### Week 2: Multi-Domain & Architecture
| Day | Task | Owner | Output |
|-----|------|-------|--------|
| 6-7 | Generate multi-domain dataset (QA + summarization + open-ended) | ML Engineer | 50K samples |
| 8-9 | Train on multi-domain, run 3×3 matrix | ML Engineer | Experiment 3 results |
| 10 | Implement PoP v3 architecture (Transformer) if needed | ML Engineer | `pop/models/pop_v3.py` |

### Week 3: Product Build
| Day | Task | Owner | Output |
|-----|------|-------|--------|
| 11-12 | Build FastAPI service | API Engineer | `pop/api/` module |
| 13 | Streaming real-time scoring | API Engineer | Streaming endpoint |
| 14-15 | Integration testing + calibration | AI Safety Engineer | Test suite passing |

### Week 4: Polish & Deliverables
| Day | Task | Owner | Output |
|-----|------|-------|--------|
| 16-17 | Demo UI (Gradio/Streamlit) | API Engineer | Live demo |
| 18 | Benchmark report + README update | ML Engineer | Documentation |
| 19 | Paper draft (honest claims, real numbers) | Research Scientist | arXiv-ready draft |
| 20 | Commit clean code to GitHub, update README | All | Public repo |

---

## 10. GITHUB STRATEGY

### What Goes on GitHub (Public)
- Clean, production code (`pop/` module)
- Tests (`tests/`)
- Professional README with architecture diagram
- API documentation
- LICENSE

### What Stays Local (Never Push)
- Benchmark raw results (JSON files)
- Training reports
- Internal analysis
- Marketing content
- Data files (.npy, .csv)

### .gitignore Additions
```
benchmarks/results/
marketing/
memory/
*.npy
*.csv
benchmark_*.json
*-REPORT.md
```

### Branch Strategy
- `main` — protected, CI must pass
- `develop` — integration branch
- `feature/*` — individual features
- `experiment/*` — experiment branches (deleted after merge or rejection)

---

## 11. FUTURE SCOPE & MARKET

### Immediate (Month 1-2)
- Open-source the core scorer
- Get first 10 users (developers integrating PoP)
- Publish paper on arXiv

### Short-term (Month 3-6)
- Expand to more models (API adapters for popular providers)
- SDK packages (Python, JavaScript)
- LangChain / LlamaIndex integration plugins
- Pursue Nepal government AI safety grant

### Medium-term (Month 6-12)
- Enterprise tier: dashboard, batch scoring, custom calibration
- Domain-specific fine-tuning (legal, medical, code)
- Real-world deployment metrics and case studies

### Long-term Vision
- Industry standard for LLM output trust scoring
- "The Stripe for LLM reliability"
- Pre-deployment trust scoring for AI pipelines
- Regulatory compliance layer (EU AI Act, etc.)

---

## 12. SUCCESS METRICS

### Experiment Success (Week 1-2)
- [ ] PoP beats entropy baseline by >5% F1
- [ ] Tier degradation <20% Full→Minimal
- [ ] At least 2 domains show >60% discrimination

### Product Success (Week 3-4)
- [ ] API responds in <10ms (Full tier)
- [ ] All three tiers functional
- [ ] Demo works end-to-end with 3 different models
- [ ] Paper draft complete

### Business Success (Month 1-2)
- [ ] 10+ GitHub stars
- [ ] 1+ external contributor
- [ ] Grant application submitted
- [ ] arXiv paper published

---

*This is a living document. Update as experiments reveal new information.*
*All claims must be backed by experiment results. No hype.*
