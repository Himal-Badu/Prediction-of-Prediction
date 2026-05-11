# Prediction of Prediction (PoP)

A meta-learning framework for detecting hallucinations in large language model outputs.

PoP works as a lightweight post-processing layer that analyzes LLM probability distributions
to estimate whether each generated token is likely correct or hallucinated — without
modifying the base model.

**Current result:** 76.46% AUC on a combined NLI + semantic similarity + length feature set,
validated with 5-fold cross-validation (±0.9% variance).

---

## About the Author

**Himal Badu** — 16-year-old ML engineer and generative AI specialist from Nepal.

I started working with machine learning at 14 and have been building AI systems ever since.
My work focuses on making LLMs more reliable and trustworthy — particularly in high-stakes
domains where hallucination isn't just inconvenient, it's dangerous.

PoP is my most ambitious project to date. It started as a curiosity-driven experiment
(Can I detect when an LLM is wrong by watching its internal signals?) and grew into
a full research effort with a novel architecture, rigorous benchmarks, and published results.

I build in public. Everything I do is open-source because I believe AI safety tools
shouldn't be locked behind paywalls or restricted to well-funded labs.

- [GitHub](https://github.com/Himal-Badu)
- [LinkedIn](https://linkedin.com/in/himalbadu)

---

## Why This Matters

LLMs generate fluent text by predicting the next token. This process produces
confident-sounding outputs even when the model is wrong — a problem known as hallucination.
Existing mitigation approaches (RLHF, fine-tuning, RAG, temperature scaling) address
symptoms but don't give the model awareness of its own uncertainty.

PoP takes a different approach: instead of changing how the model generates, it watches
what the model produces and learns to recognize the statistical signatures of errors.

---

## How It Works

PoP extracts features from the base LLM at each decoding step and feeds them to a
small classifier that outputs a hallucination probability.

### Feature Categories (9 features, v1.2)

| Category | Features | Rationale |
|----------|----------|-----------|
| **NLI** | Entailment, contradiction, neutral probabilities | Semantic relationship between input and output |
| **Semantic Similarity** | Forward cosine sim, reverse cosine sim, asymmetry | Detects topic drift and irrelevant content |
| **Length** | Question length, answer length, length ratio | Identifies hedging and evasion patterns |

### Architecture

```
Question + LLM Answer
         │
         ▼
  Feature Extraction (9 features)
         │
    ┌────┴────┐
    │ Branches │  NLI branch (RF)
    │          │  CosSim branch (RF)
    │          │  Length branch (RF)
    └────┬─────┘
         │
         ▼
  Meta-Ensemble (GradientBoosting)
         │
         ▼
  Hallucination Score (0–1)
```

Three specialized classifiers are combined via a gradient-boosted meta-learner.
This hierarchical design outperforms any single classifier on the task.

---

## Results

### Detection Performance (AUC)

| Configuration | AUC | Notes |
|--------------|-----|-------|
| NLI only (v1.0) | 67.4% | Baseline |
| NLI + CosSim (v1.1) | 70.2% | +2.8% |
| NLI + Length | 73.3% | +5.9% |
| **Full system (v1.2)** | **76.46%** | **+9.06% from baseline** |
| Meta-ensemble (v2.0, expected) | 77%+ | Not yet merged |

### Validation

- 5-fold cross-validation: 75.5% ± 0.9%
- Stable across multiple random seeds (74–76%)
- No data leakage detected
- Attention-based features confirmed non-predictive (r < 0.1 across 10+ tests)

### Key Findings

1. NLI features provide the strongest single signal for hallucination detection
2. Reverse semantic similarity catches topic drift that forward similarity misses
3. Asymmetry between forward/reverse similarity is a useful signal
4. Attention weights do not correlate with factual correctness
5. A meta-ensemble of specialized classifiers outperforms a single model

---

## Limitations

- **76.46% AUC is not production-grade.** The system is a research prototype, not a deployable product.
- All benchmarks use synthetic or standard academic datasets (TruthfulQA, HaluEval pending).
- The system has not been tested on production-scale models (GPT-4, Claude, etc.) with API-only access.
- Feature set is relatively small (9 features). Richer representations from model internals could improve results.
- Class imbalance in training data (most tokens are correct) may bias the classifier.

---

## Project Structure

```
pop/
├── core/
│   ├── llm_base.py              # LLM integration
│   ├── pop_layer_llm.py        # PoP layer
│   ├── meta_ensemble.py        # Hierarchical meta-ensemble
│   ├── pop_fusion.py           # Unified integration
│   └── correction_engine.py    # Smart correction
├── experiments/
│   ├── final_experiment.py     # Final validation
│   ├── benchmark_meta_ensemble.py
│   └── ...
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── BENCHMARKS.md
└── tests/
```

---

## Quick Start

```bash
git clone https://github.com/Himal-Badu/Prediction-of-Prediction.git
cd Prediction-of-Prediction
pip install -r requirements.txt
```

```python
from pop.core.meta_ensemble import PoPMetaEnsemble
import numpy as np

# Features: [entail, neutral, contradict, fwd_sim, rev_sim, asymmetry, len_ratio, q_len, c_len]
features = np.array([[0.8, 0.1, 0.1, 0.95, 0.92, 0.03, 1.2, 45, 52]])

meta_ensemble = PoPMetaEnsemble(random_state=42)
prob = meta_ensemble.predict_proba(features)[0]
print(f"Hallucination probability: {prob:.2%}")
```

---

## What This Is Not

- This is not a product, API, or service.
- This is not a replacement for RAG, RLHF, or fine-tuning.
- This is not claiming to "solve" hallucination. It's a research contribution showing one approach works at a defined accuracy level.

---

## Future Work

See [ROADMAP.md](ROADMAP.md) for planned work. Current focus: testing on TruthfulQA and HaluEval benchmarks, merging v2.0 meta-ensemble, and preparing an arXiv paper.

---

## License

AGPL-3.0 — see [LICENSE](LICENSE)