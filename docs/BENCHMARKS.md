# PoP Benchmark Framework

**Evaluation Protocol for Meta-Learning Hallucination Detection**

---

## 1. Overview

This document defines the benchmarking methodology for evaluating PoP's effectiveness at detecting and mitigating hallucinations in large language models. All benchmarks are designed to be reproducible, comparable to existing literature, and extensible to new base models and domains.

---

## 2. Metrics

### 2.1 Core Error Detection Metrics

PoP is evaluated as a binary classifier at the token level: does the base LLM's predicted token match ground truth?

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness of PoP's error detection |
| **Precision** | TP / (TP + FP) | Of tokens PoP flags as errors, how many actually are? |
| **Recall** | TP / (TP + FN) | Of actual errors, how many does PoP catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean — balances precision and recall |
| **AUC-ROC** | Area under ROC curve | Threshold-independent discrimination ability |
| **AUC-PR** | Area under precision-recall curve | More informative under class imbalance |

### 2.2 Calibration Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Expected Calibration Error (ECE)** | Σ (Bₘ / N) \|acc(Bₘ) - conf(Bₘ)\| | Average gap between confidence and accuracy |
| **Maximum Calibration Error (MCE)** | maxₘ \|acc(Bₘ) - conf(Bₘ)\| | Worst-case calibration gap |
| **Brier Score** | mean((p - y)²) | Proper scoring rule for probabilistic predictions |

### 2.3 Downstream Task Metrics

| Metric | Interpretation |
|--------|----------------|
| **Hallucination Rate Reduction** | % decrease in hallucinated tokens in final output |
| **Accuracy Preservation** | % of correct tokens retained after PoP intervention |
| **Abstention Quality** | When PoP abstains, what % of abstained tokens were actually wrong? |
| **Overcorrection Rate** | % of correct tokens incorrectly flagged or modified |

### 2.4 Efficiency Metrics

| Metric | Interpretation |
|--------|----------------|
| **Inference Latency Overhead** | Added time per token (ms) |
| **Parameter Overhead** | PoP parameters / base model parameters |
| **Memory Overhead** | Additional GPU memory required (MB) |

---

## 3. Comparison Baselines

### 3.1 No Intervention (Raw LLM)

The base model's output with no hallucination detection or mitigation. Establishes the floor.

### 3.2 Temperature Scaling

Post-hoc calibration by scaling logits before softmax. Optimal temperature found on validation set.

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

Single parameter; cannot capture token-level uncertainty variation.

### 3.3 Platt Scaling

Logistic regression on logits to recalibrate confidence. More flexible than temperature scaling but still response-level.

### 3.4 Ensemble Methods

Multiple forward passes with different dropout masks (MC Dropout) or sampling with different temperatures. Majority vote determines correctness.

- **MC Dropout** (5 samples): ~5× inference cost
- **MC Dropout** (10 samples): ~10× inference cost

### 3.5 SelfCheckGPT

Zero-resource method: sample multiple responses, check consistency. Flags content that appears in only one sample as likely hallucination.

- 5 samples, 10 samples configurations

### 3.6 Semantic Entropy

Compute entropy over semantically clustered outputs rather than token-level distributions. Groups equivalent answers before computing uncertainty.

### 3.7 Verbalized Confidence

Prompt the model to express its own confidence (e.g., "I am 80% confident"). Tests model self-awareness without external tooling.

---

## 4. Test Suite Design

### 4.1 Datasets

| Dataset | Domain | Size | Purpose |
|---------|--------|------|---------|
| **TriviaQA** | Factual QA | 95K questions | Factual accuracy, entity hallucination |
| **TruthfulQA** | Adversarial QA | 817 questions | Designed to elicit hallucinations |
| **HaluEval** | General | 35K samples | Multi-type hallucination detection |
| **CNN/DailyMail** | News summarization | 312K articles | Summary hallucination, entity substitution |
| **HumanEval** | Code generation | 164 problems | Code correctness, syntax vs. logic errors |
| **MMLU** | Multi-domain | 14K questions | Knowledge boundary detection |
| **Custom: Adversarial Prompts** | Mixed | 1K prompts | Prompts specifically designed to trigger hallucinations |

### 4.2 Test Categories

#### Category A: Known Facts
Questions where the ground truth is well-established and the model has likely seen it during training.
- *Expected: High base LLM accuracy, PoP adds marginal value*

#### Category B: Boundary Knowledge
Questions at the edge of the model's training distribution.
- *Expected: Mixed base accuracy, PoP provides significant value*

#### Category C: Adversarial / Trick Questions
Questions designed to elicit confident wrong answers.
- *Expected: Low base accuracy, PoP's highest value-add*

#### Category D: Long-Form Generation
Open-ended generation where factuality must be maintained over hundreds of tokens.
- *Expected: Error compounding in base LLM, PoP catches early drift*

#### Category E: Code Generation
Syntactically valid but semantically incorrect code.
- *Expected: Syntax errors caught by compiler, logic errors caught by PoP*

### 4.3 Evaluation Protocol

For each test case:

1. **Generate** output from base LLM (greedy decoding, temperature=1.0)
2. **Extract** PoP features at each decoding step
3. **Score** each token with PoP's confidence estimate
4. **Apply** Safety Guard with threshold τ ∈ {0.3, 0.5, 0.7, 0.9}
5. **Compare** to ground truth at token level
6. **Aggregate** metrics across test set

### 4.4 Threshold Analysis

For each threshold τ, compute:
- True positive rate (errors correctly flagged)
- False positive rate (correct tokens incorrectly flagged)
- Net accuracy change (accuracy_with_pop - accuracy_baseline)

Plot the operating characteristic curve to find the Pareto-optimal threshold.

---

## 5. Expected Results Template

*Results to be filled after Phase 1 supervised training completion.*

### 5.1 Error Detection Performance

| Model Config | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------------|----------|-----------|--------|-----|---------|
| PoP + DistilGPT-2 | — | — | — | — | — |
| PoP + GPT-2 | — | — | — | — | — |
| PoP + GPT-2 Medium | — | — | — | — | — |

### 5.2 Comparison with Baselines (DistilGPT-2, TruthfulQA)

| Method | Hallucination Rate ↓ | Accuracy Preserved ↑ | F1 (Error Detection) | Inference Overhead |
|--------|---------------------|---------------------|---------------------|-------------------|
| Raw LLM | — | — | N/A | 0% |
| Temperature Scaling | — | — | — | ~0% |
| Platt Scaling | — | — | — | ~0% |
| MC Dropout (5) | — | — | — | 500% |
| MC Dropout (10) | — | — | — | 1000% |
| SelfCheckGPT (5) | — | — | — | 500% |
| Semantic Entropy | — | — | — | 300% |
| Verbalized Confidence | — | — | — | ~20% |
| **PoP (τ=0.5)** | — | — | — | **<5%** |
| **PoP (τ=0.7)** | — | — | — | **<5%** |
| **PoP (τ=0.9)** | — | — | — | **<5%** |

### 5.3 Calibration Performance

| Model | ECE ↓ | MCE ↓ | Brier Score ↓ |
|-------|-------|-------|---------------|
| Raw LLM (DistilGPT-2) | — | — | — |
| Temperature-scaled | — | — | — |
| PoP-calibrated | — | — | — |

### 5.4 Per-Category Performance

| Category | Raw Accuracy | PoP Accuracy ↑ | PoP Recall ↑ | PoP Precision ↑ |
|----------|-------------|----------------|--------------|-----------------|
| A: Known Facts | — | — | — | — |
| B: Boundary Knowledge | — | — | — | — |
| C: Adversarial | — | — | — | — |
| D: Long-Form | — | — | — | — |
| E: Code Generation | — | — | — | — |

### 5.5 Scalability

| Base Model | Parameters | PoP Overhead | Latency Added | GPU Memory Added |
|-----------|------------|-------------|---------------|-----------------|
| DistilGPT-2 | 82M | 50K (0.06%) | — ms | — MB |
| GPT-2 | 124M | 50K (0.04%) | — ms | — MB |
| GPT-2 Medium | 355M | 50K (0.01%) | — ms | — MB |
| GPT-J | 6B | 50K (0.001%) | — ms | — MB |
| LLaMA-7B | 7B | 50K (0.001%) | — ms | — MB |

### 5.6 Self-Supervised Learning Progress

| Training Stage | F1 (Error Detection) | ECE ↓ | Notes |
|---------------|---------------------|-------|-------|
| Supervised only | — | — | Baseline |
| + 1 day online | — | — | |
| + 1 week online | — | — | |
| + 1 month online | — | — | |

---

## 6. Reproducibility

All benchmarks use:
- **Fixed random seeds** (42 for all experiments)
- **Deterministic decoding** (greedy, temperature=1.0) for base evaluations
- **Versioned datasets** (pinned to specific HF Datasets versions)
- **Logged configurations** (all hyperparameters saved to JSON alongside results)
- **Public result artifacts** (metrics JSON, ROC curves, calibration plots)

Benchmark runner: `pop-repo/src/evaluation/run_benchmarks.py`

---

*Document version: 1.0 | Last updated: 2026-03-28 | PoP Research Team*
