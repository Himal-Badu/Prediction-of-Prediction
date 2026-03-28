# PoP Technical Architecture

**System Design for the Prediction of Prediction Meta-Learning Layer**

---

## 1. System Overview

PoP is designed as a modular, non-invasive layer that intercepts the internal representations of a base LLM and produces per-token confidence estimates. The architecture prioritizes minimal overhead, composability with any transformer-based LLM, and strict safety guarantees.

### Design Constraints

- **<5% inference overhead** — PoP must not meaningfully slow down generation
- **<100K parameters** — The meta-learned network must be tiny relative to the base model
- **Model-agnostic** — Works with any transformer LLM via a standardized logit/hidden-state interface
- **Stateless inference** — PoP's per-step computation depends only on current + recent features, not full history

---

## 2. Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                          │
└──────────────────────────────────────────────────────────────────────┘

  User Prompt
       │
       ▼
┌─────────────────┐
│   Tokenizer     │  prompt → token_ids
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BASE LLM                                  │
│                                                                  │
│  Transformer Layers                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 0   → hidden_0, attn_0                            │   │
│  │  Layer 1   → hidden_1, attn_1                            │   │
│  │  ...                                                     │   │
│  │  Layer N   → hidden_N, attn_N   (final hidden state)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│                    LM Head (linear)                               │
│                            │                                     │
│                            ▼                                     │
│                     logits ∈ ℝ^V                                 │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    logits (V-dim)  hidden_N (d-dim)  attn maps (N×S)
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PoP FEATURE EXTRACTION                          │
│                                                                  │
│  From logits:                                                    │
│    → softmax → p(V)                                             │
│    → entropy, max_prob, top_gap, perplexity, eff_vocab,         │
│      gini, kl_uniform, logit_variance                           │
│                                                                  │
│  From temporal context:                                          │
│    → entropy_trend, maxprob_trend, dist_shift                   │
│      (requires last k feature vectors from ring buffer)          │
│                                                                  │
│  From hidden states:                                             │
│    → hidden_norm, attn_entropy, layer_stability                 │
│                                                                  │
│  From prompt/generation context:                                 │
│    → prompt_perplexity, gen_length_ratio                        │
│                                                                  │
│                          │                                       │
│                          ▼                                       │
│              f(t) ∈ ℝ^16  (feature vector)                      │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PoP PREDICTION NETWORK                          │
│                                                                  │
│  Input: f(t) ∈ ℝ^16                                            │
│                                                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │  Linear(16, 64) + ReLU + Dropout(0.2)       │               │
│  │  Linear(64, 32) + ReLU + Dropout(0.2)       │               │
│  │  Linear(32, 1) + Sigmoid                     │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  Output: p(correct) ∈ [0, 1]                                   │
│                                                                  │
│  Parameters: 16×64 + 64 + 64×32 + 32 + 32×1 + 1 = 3,169       │
│  (With expanded hidden: ~50K params at 256→128→64)              │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAFETY GUARD                                  │
│                                                                  │
│  Input: p(correct), base_logits, threshold τ                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  IF p(correct) < τ:                                      │    │
│  │    Mode = "flag"  → mark token as potentially wrong       │    │
│  │    Mode = "abstain" → replace with [UNCERTAIN] marker     │    │
│  │    Mode = "rerank" → boost alternative tokens             │    │
│  │  ELSE:                                                    │    │
│  │    Pass through base LLM token unchanged                  │    │
│  │                                                           │    │
│  │  Guard checks:                                            │    │
│  │    1. Is p(correct) in valid range [0, 1]?               │    │
│  │    2. Are features within training distribution?          │    │
│  │    3. Is rolling accuracy >= baseline?                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Output: final_token + confidence_metadata                      │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   Final Output Token
                   + Confidence Score
                   + Guard Metadata
```

---

## 3. Neural Network Architecture Details

### 3.1 PoP Prediction Network

```
Layer           Input Dim   Output Dim   Activation   Params
─────────────── ─────────── ──────────── ──────────── ──────────
Linear_1        16          256          ReLU         4,352
Dropout_1       —           —            p=0.2        —
Linear_2        256         128          ReLU         32,896
Dropout_2       —           —            p=0.2        —
Linear_3        128         64           ReLU         8,256
Dropout_3       —           —            p=0.2        —
Linear_4        64          1            Sigmoid      65
─────────────── ─────────── ──────────── ──────────── ──────────
TOTAL                                                          45,569
```

### 3.2 Weight Initialization

- Linear layers: Kaiming uniform (fan_in, ReLU gain)
- Final layer: Xavier uniform (sigmoid-compatible gain)
- Biases: Zero

### 3.3 Regularization

| Technique | Parameter | Purpose |
|-----------|-----------|---------|
| Dropout | p=0.2 | Prevent co-adaptation of features |
| Weight decay | 1e-4 | L2 regularization via AdamW |
| Gradient clipping | max_norm=1.0 | Training stability |
| Early stopping | patience=10 epochs | Prevent overfitting |

### 3.4 Loss Function

Binary cross-entropy with positive class weighting:

```
L = -[w_pos × y × log(p) + (1-y) × log(1-p)]
```

where `w_pos = N_neg / N_pos` compensates for class imbalance (most tokens are predicted correctly).

---

## 4. Feature Extraction Module

### 4.1 Distributional Features

```python
def extract_distributional_features(logits):
    p = softmax(logits)
    
    entropy = -sum(p * log(p + eps))
    max_prob = max(p)
    top_gap = sorted(p)[-1] - sorted(p)[-2]
    perplexity = exp(entropy)
    eff_vocab = count(p > 1e-5)
    gini = 1 - sum(p ** 2)
    kl_uniform = sum(p * log(p * V + eps))
    logit_var = var(logits)
    
    return [entropy, max_prob, top_gap, perplexity, 
            eff_vocab, gini, kl_uniform, logit_var]
```

### 4.2 Temporal Features

Maintained via a fixed-size ring buffer (k=5):

```python
class TemporalFeatureExtractor:
    def __init__(self, window_size=5):
        self.buffer = deque(maxlen=window_size)
    
    def update(self, current_features):
        if len(self.buffer) >= 2:
            entropy_trend = current_features[0] - self.buffer[-1][0]
            maxprob_trend = current_features[1] - self.buffer[-1][1]
            dist_shift = kl_divergence(current_probs, self.buffer[-1].probs)
        self.buffer.append(current_features)
```

### 4.3 Hidden State Features

```python
def extract_hidden_features(hidden_states, attention_weights):
    hidden_norm = norm(hidden_states[-1])  # Last layer norm
    attn_entropy = mean([entropy(attn) for attn in attention_weights])
    layer_stability = correlation(logits_per_layer)
    return [hidden_norm, attn_entropy, layer_stability]
```

### 4.4 Contextual Features

```python
def extract_context_features(prompt_ids, current_step, model):
    prompt_logits = model(prompt_ids).logits
    prompt_pplx = exp(mean(ce_loss(prompt_logits, prompt_ids)))
    gen_ratio = current_step / AVG_GENERATION_LENGTH
    return [prompt_pplx, gen_ratio]
```

---

## 5. Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                        │
└──────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │ Training      │  Wikipedia, C4, OpenWebText, code corpora
  │ Corpus        │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Base LLM      │  Forward pass, collect logits + hidden states
  │ Inference     │  at each decoding step
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Feature       │  16-dim feature vector per step
  │ Extraction    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Label         │  1 = token matches ground truth
  │ Generation    │  0 = token is incorrect
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         Labeled Dataset                   │
  │  {(f(t), y(t))} for millions of steps    │
  └──────┬───────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         Train/Val/Test Split              │
  │  80% train, 10% val, 10% test            │
  │  Stratified by correctness label          │
  └──────┬───────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         PoP Network Training              │
  │                                           │
  │  Optimizer: AdamW (lr=1e-3, wd=1e-4)    │
  │  Scheduler: Cosine annealing              │
  │  Epochs: 50 (early stop, patience=10)    │
  │  Batch size: 2048                         │
  │  Gradient clip: 1.0                       │
  └──────┬───────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         Calibration                       │
  │  Temperature scaling on validation set    │
  │  Ensures p(correct) is well-calibrated    │
  └──────┬───────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         Trained PoP Model                 │
  │  Serialized: pop_model.pt + config.json   │
  └──────────────────────────────────────────┘
```

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | Standard for small MLPs |
| Weight decay | 1e-4 | Light L2 regularization |
| Batch size | 2048 | Stable gradients with class imbalance |
| Max epochs | 50 | Sufficient with early stopping |
| Early stop patience | 10 | Prevents overfitting |
| Dropout | 0.2 | Moderate regularization |
| Class weight | N_neg/N_pos | Handles ~90%+ accuracy baseline |
| Gradient clip | 1.0 | Training stability |

---

## 6. Inference Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                       │
└──────────────────────────────────────────────────────────────┘

  For each decoding step t:

  ┌─────────────────────────┐
  │ 1. Base LLM forward     │  ~95% of compute
  │    → logits, hidden     │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 2. Feature extraction   │  ~3% of compute
  │    → f(t) ∈ ℝ^16       │  (vectorized, GPU-friendly)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 3. PoP forward          │  ~1% of compute
  │    (3-layer MLP)        │  (tiny network)
  │    → p(correct)         │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 4. Safety guard         │  ~1% of compute
  │    threshold check      │  (deterministic logic)
  │    → action decision    │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ 5. Output               │
  │    token + confidence   │
  └─────────────────────────┘

  Total overhead: <5% of base LLM inference time
  Additional memory: ~2MB (PoP weights + feature ring buffer)
```

### Latency Budget (DistilGPT-2)

| Component | Time (ms/token) | % of Total |
|-----------|-----------------|------------|
| Base LLM forward | ~15.0 | 95.5% |
| Feature extraction | ~0.4 | 2.5% |
| PoP MLP forward | ~0.2 | 1.3% |
| Safety guard | ~0.1 | 0.7% |
| **Total** | **~15.7** | **100%** |
| **Overhead** | **~0.7** | **4.5%** |

---

## 7. Scalability

### 7.1 Architecture Scaling

The PoP network size is independent of the base model size. The same 50K-parameter MLP works with any transformer because:

- Feature extraction always produces a 16-dimensional vector
- The MLP input size is fixed at 16
- Hidden state features are aggregated to scalars (norm, entropy)

| Base Model | Size | PoP Size | Overhead Ratio |
|-----------|------|----------|----------------|
| DistilGPT-2 | 82M | 50K | 0.06% |
| GPT-2 | 124M | 50K | 0.04% |
| GPT-2 Medium | 355M | 50K | 0.01% |
| GPT-J 6B | 6B | 50K | 0.001% |
| LLaMA-7B | 7B | 50K | 0.001% |
| LLaMA-70B | 70B | 50K | 0.0001% |

### 7.2 Feature Extraction Scaling

Feature extraction cost depends on:
- **Vocabulary size (V):** Softmax + entropy computation is O(V). For large vocabularies (32K–100K), this is dominated by the softmax.
- **Number of layers (N):** Layer-wise stability computation requires logits from multiple layers. With hooks, this adds one correlation computation per layer pair.
- **Sequence length (S):** Attention entropy is O(N × S²) per step. For long sequences, use only the last layer's attention or sampled attention heads.

Optimization for large models:
- Cache softmax computation (already done for token sampling)
- Use only last 4 layers for stability computation
- Sample 4 of N attention heads for entropy estimation

### 7.3 Batch Inference

For batch generation (multiple prompts simultaneously):
- Feature extraction: fully parallelizable across batch dimension
- PoP MLP: naturally batched (batch × 16 → batch × 1)
- Safety guard: element-wise threshold check

No additional overhead for batched inference beyond single-sample inference.

### 7.4 Distributed Deployment

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GPU Node 1  │     │  GPU Node 2  │     │  GPU Node 3  │
│              │     │              │     │              │
│  Base LLM    │────▶│  PoP Layer   │────▶│  Safety      │
│  (sharded)   │     │  (replicated)│     │  Guard + API │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

PoP can be colocated with the base LLM (same GPU) or deployed on a separate lightweight node. Since PoP's compute is dominated by feature extraction (which accesses base model internals), colocation is preferred to avoid serialization overhead.

---

## 8. Implementation Stack

| Component | Technology |
|-----------|-----------|
| Base LLM integration | HuggingFace Transformers |
| Feature extraction | PyTorch (custom modules) |
| PoP network | PyTorch (nn.Module) |
| Training | PyTorch + WandB logging |
| Safety guard | Python (deterministic, no ML) |
| API serving | FastAPI + TorchServe |
| Benchmarking | Custom harness + HuggingFace Datasets |

---

## 9. Module Structure

```
pop-repo/
├── src/
│   ├── features/
│   │   ├── distributional.py    # Entropy, max_prob, gini, etc.
│   │   ├── temporal.py          # Trends, shifts (ring buffer)
│   │   ├── hidden_state.py      # Norm, attn entropy, stability
│   │   ├── contextual.py        # Prompt perplexity, gen ratio
│   │   └── extractor.py         # Unified feature extractor
│   ├── model/
│   │   ├── pop_network.py       # The 3-layer MLP
│   │   ├── guard.py             # Safety guard logic
│   │   └── pop_pipeline.py      # End-to-end PoP pipeline
│   ├── training/
│   │   ├── dataset.py           # Labeled dataset generation
│   │   ├── train.py             # Training loop
│   │   ├── calibrate.py         # Temperature scaling
│   │   └── config.py            # Hyperparameters
│   ├── evaluation/
│   │   ├── metrics.py           # All metric computations
│   │   ├── baselines.py         # Baseline implementations
│   │   ├── run_benchmarks.py    # Benchmark runner
│   │   └── visualize.py         # Plot generation
│   └── api/
│       ├── server.py            # FastAPI application
│       └── schemas.py           # Request/response models
├── docs/
│   ├── METHODOLOGY.md
│   ├── BENCHMARKS.md
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── configs/
│   ├── distilgpt2.yaml
│   ├── gpt2.yaml
│   └── defaults.yaml
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_guard.py
│   └── test_pipeline.py
└── README.md
```

---

*Document version: 1.0 | Last updated: 2026-03-28 | PoP Research Team*
