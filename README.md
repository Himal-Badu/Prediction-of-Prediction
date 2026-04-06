# 🔮 Prediction of Prediction (PoP)

**An NLI-based hallucination detection system for Large Language Models.**

PoP achieves **75.5% AUC** in detecting AI hallucinations using Natural Language Inference (NLI) combined with semantic similarity and length features.

[![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFAE00?style=flat&logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/-License-AGPL--3.0-orange?style=flat)](LICENSE)

---

## What is PoP?

Prediction of Prediction (PoP) is a **hallucination detection system** that analyzes LLM outputs to determine whether the generated content is factually supported or potentially hallucinated.

### The Problem

Large Language Models (LLMs) often generate confident but factually incorrect information — a phenomenon known as "hallucination." This is the #1 barrier to enterprise AI adoption, especially in high-stakes domains like healthcare, finance, and legal.

### Our Solution

We developed a novel approach using **Natural Language Inference (NLI)** to detect hallucinations:

1. **NLI Analysis** — Check if LLM outputs are entailed by, contradictory to, or neutral with respect to the input
2. **Semantic Similarity** — Measure how well the answer aligns with the question contextually
3. **Length Features** — Analyze answer length patterns as an additional signal

> **Key Finding:** We discovered that attention mechanisms — commonly used in LLM analysis — show **no significant correlation** (r < 0.1) with hallucination labels. NLI-based features significantly outperform attention-based approaches.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Detection AUC** | **75.5%** |
| Variance | ±0.9% |
| Range | 74.2% - 76.5% |
| Method | NLI + CosSim + Length |

### Method Comparison

| Method | AUC | Notes |
|--------|-----|-------|
| **NLI + CosSim + Length** | **75.5%** | 🎯 Best |
| NLI + Length | 73.3% | Good |
| NLI + CosSim | 70.2% | Moderate |
| NLI only | 67.4% | Baseline |
| **NLI + Attention** | **67.3%** | ❌ No improvement |

### Research Findings

- ✅ NLI (entailment/contradiction) provides real signal for hallucination detection
- ✅ Semantic similarity between Q&A improves detection
- ✅ Answer length features add predictive power
- ❌ Attention mechanisms do NOT help (confirmed across 10+ validation tests)
- ❌ Logits/uncertainty measures do NOT reliably predict hallucinations

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INPUT                              │
│         (Question + LLM Answer)                    │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│         FEATURE EXTRACTION                          │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────┐ │
│  │ NLI Features  │ │  CosSim (QA)   │ │  Length  │ │
│  │ • Entailment  │ │  Similarity    │ │ Features │ │
│  │ • Contradict  │ │  Embedding     │ │ • q_len  │ │
│  │ • Neutral     │ │  Cosine        │ │ • c_len  │ │
│  └────────────────┘ └────────────────┘ └──────────┘ │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│         CLASSIFIER (RandomForest/GB)               │
│  • 300 estimators, max_depth=8                      │
│  • 5-fold cross-validation                         │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│              HALLUCINATION SCORE                    │
│         (Probability of hallucination)              │
└─────────────────────────────────────────────────────┘
```

### Features Used

| Category | Features | Contribution |
|----------|----------|--------------|
| **NLI** | Entailment, Contradiction, Neutral probabilities | Primary signal |
| **Semantic** | Cosine similarity between Q&A embeddings | +9% AUC |
| **Length** | Question length, Answer length, Length ratio | +6% AUC |

---

## Validation & Robustness

We conducted comprehensive validation to ensure reliable results:

| Test | Result |
|------|--------|
| 5-fold Cross-validation | 75.5% ± 0.9% |
| Multiple random seeds | Stable (74-76%) |
| Different train/test splits | Consistent |
| Different classifiers | Similar results |
| No data leakage | Verified |
| Overfitting check | None detected |

---

## Project Structure

```
pop-repo/
├── pop/
│   ├── core/
│   │   ├── pop_v2.py              # PoP v2 architecture
│   │   ├── correction_engine.py   # Smart correction
│   │   └── ...
│   └── __init__.py
├── experiments/
│   ├── final_experiment.py        # Final validation
│   ├── final_results.json        # Results summary
│   ├── multi_angle_analysis.json # Comprehensive testing
│   └── ...
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── BENCHMARKS.md
├── tests/
├── train_pop_v2.py
├── benchmark.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Himal-Badu/Prediction-of-Prediction.git
cd Prediction-of-Prediction
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- Sentence-Transformers
- scikit-learn

---

## Quick Start

### Run Detection

```python
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Your trained model (see experiments/final_experiment.py)
# 1. Extract NLI features
# 2. Extract CosSim features  
# 3. Extract Length features
# 4. Combine and classify
```

### Run Experiments

```bash
# Run the final validation experiment
python experiments/final_experiment.py
```

---

## Research Paper

This project is backed by extensive research. Key publications:

- **Multi-angle Analysis** — Tested 10+ different angles to find critical points
- **Attention vs NLI** — Comprehensive comparison proving NLI superiority
- **Validation** — 5-fold CV, multiple seeds, no data leakage

See [`experiments/`](experiments/) for all experimental results.

---

## Use Cases

| Domain | Application |
|--------|-------------|
| **Enterprise AI** | Verify AI-generated content before deployment |
| **Healthcare** | Detect errors in medical AI assistants |
| **Finance** | Flag unreliable financial reports |
| **Legal** | Verify AI-generated legal documents |
| **Research** | Fact-checking AI-generated summaries |

---

## Roadmap

- [x] NLI-based hallucination detection research
- [x] Multi-angle validation and testing
- [x] Attention mechanism analysis (confirmed useless)
- [x] Feature combination optimization
- [x] Cross-validation and robustness testing
- [ ] Write research paper
- [ ] Extend to other NLI models
- [ ] Test on more domains
- [ ] Production API

---

## Contributing

We welcome contributions! Please:

1. Open an issue to discuss changes
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

---

## License

AGPL-3.0 License — see [LICENSE](LICENSE)

---

## Author

**Himal Badu** | 16-year-old AI researcher from Nepal

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Himal-Badu)

*Building AI safety tools for the next generation.*

---

## Acknowledgments

- HuggingFace for Transformers and Sentence-Transformers
- TruthfulQA dataset for evaluation
- Open-source research community