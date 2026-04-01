# LLM System Architecture Design

## Document Control

| Field | Value |
|-------|-------|
| **Project** | NeuronLM – Enterprise LLM Platform |
| **Version** | 1.0.0 |
| **Status** | Approved |

---

## 1. High-Level Architecture

### 1.1 Layered Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  Chat UI   │  │  Admin UI  │  │  REST API  │  │  gRPC API │ │
│  │  (React)   │  │  (React)   │  │  (FastAPI) │  │  (Proto)  │ │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                        GATEWAY LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              API Gateway (Kong / Envoy)                   │   │
│  │  • Rate Limiting  • Auth (JWT/OAuth2)  • Request Routing │   │
│  │  • Request/Response Transformation  • Circuit Breaking   │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                       SERVICE LAYER                              │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │   Inference    │  │   Conversation │  │   RAG Engine      │  │
│  │   Service      │  │   Manager      │  │   Service         │  │
│  │                │  │                │  │                   │  │
│  │  • Completion  │  │  • Sessions    │  │  • Embedding      │  │
│  │  • Streaming   │  │  • History     │  │  • Retrieval      │  │
│  │  • Batching    │  │  • Context     │  │  • Re-ranking     │  │
│  └────────┬───────┘  └────────┬───────┘  └────────┬──────────┘  │
│           │                   │                    │             │
│  ┌────────▼───────┐  ┌───────▼────────┐  ┌───────▼──────────┐  │
│  │   Model        │  │   Fine-Tuning  │  │   Data Pipeline  │  │
│  │   Registry     │  │   Service      │  │   Service        │  │
│  │                │  │                │  │                   │  │
│  │  • Versioning  │  │  • LoRA/QLoRA  │  │  • Ingestion     │  │
│  │  • Metadata    │  │  • RLHF        │  │  • Chunking      │  │
│  │  • A/B Testing │  │  • Evaluation  │  │  • Indexing       │  │
│  └────────────────┘  └────────────────┘  └───────────────────┘  │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │   Auth &       │  │   Usage &      │  │   Content        │  │
│  │   RBAC         │  │   Billing      │  │   Moderation     │  │
│  │   Service      │  │   Service      │  │   Service        │  │
│  └────────────────┘  └────────────────┘  └───────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                         │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ │
│  │PostgreSQL│ │  Redis   │ │  Qdrant  │ │ Kafka  │ │ MinIO  │ │
│  │  (OLTP)  │ │  (Cache) │ │ (Vector) │ │ (Msgs) │ │(Object)│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ └────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           GPU Compute Cluster (NVIDIA A100/H100)         │   │
│  │  • Training Nodes  • Inference Nodes  • Auto-Scaling     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. LLM Training Flow

The end-to-end training pipeline that produces a usable language model:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     LLM TRAINING PIPELINE                                │
│                                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ Raw Data │──▶│ Tokenization │──▶│    Model     │──▶│ Prediction  │  │
│  │          │   │              │   │  (Forward    │   │ (Logits)    │  │
│  │ • Text   │   │ • BPE / SPM │   │   Pass)      │   │             │  │
│  │ • Code   │   │ • Vocab Map │   │              │   │ Next-token  │  │
│  │ • Books  │   │ • Token IDs │   │ Transformer  │   │ probability │  │
│  │ • Web    │   │ • Attention │   │ Decoder      │   │ distribution│  │
│  │          │   │   Masks     │   │ Stack        │   │             │  │
│  └──────────┘   └──────────────┘   └──────────────┘   └──────┬──────┘  │
│                                                               │         │
│       ┌──────────────────────────────────────────────────────┘         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │    Loss      │──▶│  Backprop    │──▶│   Weight     │               │
│  │  Computation │   │  (Gradient   │   │   Update     │               │
│  │              │   │   Calc)      │   │              │               │
│  │ Cross-       │   │              │   │ Optimizer:   │               │
│  │ Entropy      │   │ ∂Loss/∂W for │   │ AdamW        │──┐            │
│  │ between      │   │ every layer  │   │              │  │            │
│  │ predicted &  │   │ via chain    │   │ W = W - lr · │  │            │
│  │ actual next  │   │ rule         │   │     ∇Loss    │  │            │
│  │ token        │   │              │   │              │  │            │
│  └──────────────┘   └──────────────┘   └──────────────┘  │            │
│                                                           │            │
│       ┌───────────────────────────────────────────────────┘            │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │                         REPEAT                               │      │
│  │                                                               │      │
│  │  For each mini-batch across all epochs:                       │      │
│  │    1. Sample next batch of token sequences                    │      │
│  │    2. Forward pass → Prediction → Loss                        │      │
│  │    3. Backward pass → Gradients                               │      │
│  │    4. Optimizer step → Updated weights                        │      │
│  │    5. Learning rate schedule (cosine decay with warmup)       │      │
│  │    6. Gradient clipping (max_norm = 1.0)                      │      │
│  │    7. Log metrics (loss, perplexity, throughput)              │      │
│  │                                                               │      │
│  │  Convergence: loss stabilizes ~ 2-3 trillion tokens          │      │
│  └─────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

Note: The training flow described above is intentionally architecture-agnostic — it describes the procedural lifecycle for training any autoregressive sequence model (Transformer, RNN, or similar). Section 3 provides the detailed Transformer Decoder architecture used by NeuronLM. The following mapping shows how the training steps correspond to Transformer components:

- Raw Data / Tokenization  → Token Embedding + Position Embeddings (Section 3)
- Forward / Prediction     → Transformer Decoder Blocks + Output Head (Section 3)
- Loss (Cross-Entropy)    → Computed on logits produced by the Output Head
- Backprop / Gradients    → Gradients flow through attention and FFN weights (W_Q, W_K, W_V, W_O, W_up, W_down, embeddings)
- Weight Update (AdamW)  → Applies to all trainable parameters defined in the Transformer stack
- Repeat (batches/epochs) → Training loop, optimizer steps, LR schedule, checkpointing

### 2.1 Training Stages

| Stage | Description | Data Size | Compute |
|-------|------------|-----------|---------|
| **Pre-training** | Next-token prediction on raw corpus | 2-15 T tokens | 1000s of GPU-hours |
| **Supervised Fine-Tuning (SFT)** | Instruction-response pairs | 100K–1M examples | 10s of GPU-hours |
| **RLHF / DPO** | Human preference alignment | 50K–200K comparisons | 100s of GPU-hours |
| **Task-Specific Fine-Tuning** | Domain adaptation (LoRA/QLoRA) | 1K–100K examples | 1–10 GPU-hours |

### 2.2 Training Hyperparameters (Reference)

```yaml
training_config:
  model_size: 7B
  precision: bfloat16
  batch_size: 2048           # sequences per step (micro × gradient_accum × GPUs)
  sequence_length: 4096
  learning_rate: 3.0e-4
  lr_scheduler: cosine
  warmup_steps: 2000
  weight_decay: 0.1
  gradient_clip: 1.0
  optimizer: AdamW
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  total_steps: 500_000
  checkpoint_interval: 1000
  eval_interval: 500
  distributed:
    strategy: FSDP             # Fully Sharded Data Parallel
    tensor_parallel: 4
    pipeline_parallel: 2
```

---

## 3. Transformer Architecture

The core neural network architecture powering NeuronLM.

### 3.1 Full Transformer Decoder Block

```
 Input Token IDs: [4521, 318, 257, 1332, ...]
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                    TOKEN EMBEDDING LAYER                          │
│                                                                   │
│  Token Embedding (vocab_size × d_model)  ──┐                    │
│  + Rotary Position Embedding (RoPE)     ───┤──▶  X₀ ∈ ℝ^(n×d)  │
│                                             │                    │
└─────────────────────────────────────────────┼────────────────────┘
                                              │
        ┌─────────────────────────────────────┘
        │
        ▼   ×N layers (e.g. N=32 for 7B model)
┌───────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER DECODER BLOCK                      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Layer Norm (RMSNorm)                                        │ │
│  │  x_norm = RMSNorm(x)                                        │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │         GROUPED MULTI-QUERY ATTENTION (GQA)                  │ │
│  │                                                               │ │
│  │   Q = x_norm · W_Q    (n_heads × d_head)                    │ │
│  │   K = x_norm · W_K    (n_kv_heads × d_head)   ← fewer heads │ │
│  │   V = x_norm · W_V    (n_kv_heads × d_head)   ← fewer heads │ │
│  │                                                               │ │
│  │   Apply RoPE to Q, K                                         │ │
│  │                                                               │ │
│  │   ┌───────────────────────────────────────────────────┐      │ │
│  │   │  Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V│      │ │
│  │   │                                                    │      │ │
│  │   │  With causal mask:                                 │      │ │
│  │   │  ┌───┬───┬───┬───┬───┐                            │      │ │
│  │   │  │ 1 │ 0 │ 0 │ 0 │ 0 │  token 1 sees only itself │      │ │
│  │   │  │ 1 │ 1 │ 0 │ 0 │ 0 │  token 2 sees 1-2         │      │ │
│  │   │  │ 1 │ 1 │ 1 │ 0 │ 0 │  token 3 sees 1-3         │      │ │
│  │   │  │ 1 │ 1 │ 1 │ 1 │ 0 │  token 4 sees 1-4         │      │ │
│  │   │  │ 1 │ 1 │ 1 │ 1 │ 1 │  token 5 sees 1-5         │      │ │
│  │   │  └───┴───┴───┴───┴───┘                            │      │ │
│  │   └───────────────────────────────────────────────────┘      │ │
│  │                                                               │ │
│  │   Output Projection: attn_out = concat(heads) · W_O          │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│           x = x + attn_out   │  ← Residual Connection            │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │  Layer Norm (RMSNorm)                                        │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │              FEED-FORWARD NETWORK (SwiGLU)                   │ │
│  │                                                               │ │
│  │   gate   = x_norm · W_gate                                   │ │
│  │   up     = x_norm · W_up                                     │ │
│  │   hidden = SiLU(gate) ⊙ up       ← element-wise multiply    │ │
│  │   output = hidden · W_down                                    │ │
│  │                                                               │ │
│  │   Dimensions: d_model → 4×d_model (intermediate) → d_model  │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │                                    │
│           x = x + ffn_out    │  ← Residual Connection            │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
        Repeat ×N layers       │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                    OUTPUT HEAD                                     │
│                                                                   │
│  Final RMSNorm → Linear Projection (d_model → vocab_size)        │
│                                                                   │
│  logits ∈ ℝ^(vocab_size)                                         │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  SAMPLING / DECODING STRATEGY                         │        │
│  │                                                        │        │
│  │  Temperature scaling:  logits = logits / T             │        │
│  │  Top-p (nucleus):      keep tokens summing to p        │        │
│  │  Top-k:                keep top k tokens               │        │
│  │  Repetition penalty:   reduce repeated token logits    │        │
│  │                                                        │        │
│  │  probs = softmax(filtered_logits)                      │        │
│  │  next_token = sample(probs)                            │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
│  Output: next token ID → detokenize → stream to user             │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 NeuronLM Model Configurations

| Parameter | 1.5B | 7B | 13B | 70B |
|-----------|------|-----|------|------|
| Layers (N) | 28 | 32 | 40 | 80 |
| Model Dim (d_model) | 2048 | 4096 | 5120 | 8192 |
| Attention Heads | 16 | 32 | 40 | 64 |
| KV Heads (GQA) | 4 | 8 | 8 | 8 |
| Head Dim | 128 | 128 | 128 | 128 |
| FFN Intermediate | 5504 | 11008 | 13824 | 28672 |
| Vocab Size | 32000 | 32000 | 32000 | 32000 |
| Max Context | 8192 | 32768 | 32768 | 131072 |
| RoPE θ | 10000 | 500000 | 500000 | 500000 |
| Norm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Activation | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| Precision | BF16 | BF16 | BF16 | BF16 |

### 3.3 Key Architectural Innovations

**Grouped Query Attention (GQA)**
```
Standard MHA:   32 Q heads, 32 K heads, 32 V heads  → high KV cache memory
Multi-Query:    32 Q heads,  1 K head,   1 V head   → fast but quality loss
Grouped Query:  32 Q heads,  8 K heads,  8 V heads  → balanced tradeoff

KV Cache savings at 32K context, 7B model:
  MHA:  32 × 2 × 32768 × 128 × 2 bytes = 512 MB per request
  GQA:   8 × 2 × 32768 × 128 × 2 bytes = 128 MB per request  ← 4× reduction
```

**Rotary Position Embeddings (RoPE)**
```
Encodes position by rotating Q and K vectors:

  RoPE(x, pos) = x · [cos(pos·θ), -sin(pos·θ)]
                     [sin(pos·θ),  cos(pos·θ)]

Benefits:
  • Relative position awareness (attention decays with distance)
  • Extrapolation to longer sequences via NTK-aware scaling
  • No learned position embeddings → fewer parameters
```

**SwiGLU Activation**
```
Standard FFN:    ReLU(x · W₁) · W₂
SwiGLU FFN:      (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down

SiLU(x) = x · σ(x)    where σ = sigmoid

Benefits:
  • Smoother gradients than ReLU
  • Gating mechanism improves expressiveness
  • ~1% improvement on benchmarks vs standard FFN
```

### 3.4 Inference Optimization Techniques

```
┌─────────────────────────────────────────────────────────────────┐
│                 INFERENCE OPTIMIZATION STACK                     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Quantization (AWQ / GPTQ / GGUF)                       │   │
│  │  FP16 → INT4: 4× memory reduction, ~1% quality loss     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  KV Cache with PagedAttention (vLLM)                     │   │
│  │  Virtual memory paging for KV cache blocks               │   │
│  │  Eliminates memory fragmentation → 2-4× more throughput  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Continuous Batching                                      │   │
│  │  Dynamically add/remove requests mid-generation           │   │
│  │  vs. static batching: 10-20× throughput improvement       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Speculative Decoding                                     │   │
│  │  Draft model (1.5B) proposes K tokens → verified by 7B   │   │
│  │  Acceptance rate ~70-80% → 2-3× faster generation        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  FlashAttention-2                                         │   │
│  │  Fused CUDA kernel: tiling + recomputation                │   │
│  │  2-4× faster attention, O(n) memory instead of O(n²)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tensor Parallelism (TP) + Pipeline Parallelism (PP)     │   │
│  │  TP: split attention heads across GPUs (intra-node)       │   │
│  │  PP: split layers across GPUs (inter-node)                │   │
│  │  70B model: TP=8 within node, PP=2 across nodes          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 2.1 Inference Service

The core service responsible for running forward passes through the trained Transformer model to generate text.

```
┌─────────────────────────────────────────────────┐
│                Inference Service                 │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐ │
│  │ Request  │───▶│ Prompt   │───▶│ Model     │ │
│  │ Validator│    │ Builder  │    │ Router    │ │
│  └──────────┘    └──────────┘    └─────┬─────┘ │
│                                        │       │
│           ┌────────────────────────────┘       │
│           ▼                                     │
│  ┌──────────────┐  ┌────────────┐              │
│  │  KV Cache    │  │  Token     │              │
│  │  Manager     │  │  Streamer  │              │
│  └──────┬───────┘  └─────┬──────┘              │
│         │                │                      │
│         ▼                ▼                      │
│  ┌──────────────────────────────┐              │
│  │      Response Assembler      │              │
│  │  • Detokenization            │              │
│  │  • Safety Filtering          │              │
│  │  • Usage Metrics             │              │
│  └──────────────────────────────┘              │
└─────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Continuous Batching**: Dynamic batching of requests to maximize GPU utilization
- **PagedAttention**: Efficient KV-cache memory management (via vLLM)
- **Speculative Decoding**: Use a smaller draft model for faster generation
- **Tensor Parallelism**: Split large models across multiple GPUs

### 4.2 Conversation Manager

Manages multi-turn conversations with context window optimization.

```python
# Conversation Flow
User Message → Context Assembly → [System Prompt + History + RAG Context + User Message]
             → Token Budget Allocation → Truncation Strategy → Inference Service
             → Response → Store in History → Return to User
```

**Context Window Strategy:**
| Priority | Component | Token Budget |
|----------|-----------|-------------|
| 1 (Fixed) | System Prompt | 500 tokens |
| 2 (Fixed) | Safety Instructions | 200 tokens |
| 3 (Dynamic) | RAG Context | 2,000 tokens |
| 4 (Dynamic) | Conversation History | Remaining budget |
| 5 (Fixed) | Current User Message | As needed |

### 4.3 RAG Engine

```
┌──────────────────────────────────────────────────────┐
│                    RAG Pipeline                       │
│                                                       │
│  ┌──────────┐  ┌───────────┐  ┌───────────────────┐ │
│  │ Document │  │  Chunker  │  │  Embedding        │ │
│  │ Loader   │──▶│  (512tok) │──▶│  Generator        │ │
│  └──────────┘  └───────────┘  └────────┬──────────┘ │
│                                         │            │
│                                         ▼            │
│                                  ┌─────────────┐    │
│                                  │  Vector DB  │    │
│                                  │  (Qdrant)   │    │
│                                  └──────┬──────┘    │
│                                         │            │
│  ┌──────────┐  ┌───────────┐           │            │
│  │ Context  │◀─│ Re-Ranker │◀──────────┘            │
│  │ Builder  │  │ (Cross-   │                        │
│  └──────────┘  │ Encoder)  │                        │
│                └───────────┘                        │
└──────────────────────────────────────────────────────┘
```

### 4.4 Model Registry

```
┌─────────────────────────────────────────────┐
│              Model Registry                  │
│                                              │
│  Model Metadata:                             │
│  ├── model_id: "neuronlm-7b-v2"            │
│  ├── architecture: "transformer-decoder"     │
│  ├── parameters: 7_000_000_000              │
│  ├── quantization: "AWQ-4bit"               │
│  ├── max_context: 32768                     │
│  ├── version: "2.1.0"                       │
│  ├── status: "production"                   │
│  ├── serving_config:                        │
│  │   ├── gpu_memory: "24GB"                 │
│  │   ├── tensor_parallel: 1                 │
│  │   └── max_batch_size: 64                 │
│  └── evaluation_scores:                     │
│      ├── mmlu: 0.72                         │
│      ├── humaneval: 0.65                    │
│      └── hellaswag: 0.81                    │
└─────────────────────────────────────────────┘
```

---

## 5. Data Architecture

### 5.1 Data Flow Diagram

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────▶│  API    │────▶│Inference│────▶│Response │
│ Request │     │ Gateway │     │ Service │     │  Cache  │
└─────────┘     └─────────┘     └────┬────┘     └─────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────┐    ┌──────────┐
              │Conversation│  │  Model   │    │  Usage   │
              │  Store    │  │  Weights  │    │  Metrics │
              │(Postgres) │  │  (MinIO)  │    │  (Kafka) │
              └──────────┘   └──────────┘    └──────────┘
```

### 5.2 Database Schema (Core Tables)

```sql
-- Conversations
CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    title           VARCHAR(500),
    model_id        VARCHAR(100) NOT NULL,
    system_prompt   TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Messages
CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role            VARCHAR(20) NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    content         TEXT NOT NULL,
    token_count     INTEGER,
    model_id        VARCHAR(100),
    latency_ms      INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    key_hash        VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of key
    key_prefix      VARCHAR(8) NOT NULL,           -- First 8 chars for identification
    name            VARCHAR(100),
    scopes          TEXT[] DEFAULT '{}',
    rate_limit      INTEGER DEFAULT 60,
    expires_at      TIMESTAMPTZ,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Usage Tracking
CREATE TABLE usage_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    api_key_id      UUID REFERENCES api_keys(id),
    model_id        VARCHAR(100) NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens    INTEGER GENERATED ALWAYS AS (prompt_tokens + completion_tokens) STORED,
    latency_ms      INTEGER,
    status          VARCHAR(20) DEFAULT 'success',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 6. API Design

### 6.1 Chat Completions API (OpenAI-Compatible)

```
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <api-key>

{
    "model": "neuronlm-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": true,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

### 6.2 API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming/non-streaming) |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Generate embeddings |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model details |
| POST | `/v1/moderations` | Content moderation |
| POST | `/v1/fine-tuning/jobs` | Create fine-tuning job |
| GET | `/v1/fine-tuning/jobs/{id}` | Get fine-tuning status |
| GET | `/v1/usage` | Get usage statistics |
| POST | `/v1/rag/documents` | Upload RAG documents |
| POST | `/v1/rag/query` | Query RAG knowledge base |

---

## 7. Security Architecture

### 7.1 Authentication & Authorization Flow

```
Client → API Gateway → JWT Validation → RBAC Check → Service
                │                │              │
                ▼                ▼              ▼
          Rate Limiter    Token Introspection  Policy Engine
          (per API key)   (Redis cache)        (OPA/Cedar)
```

### 7.2 Security Controls

| Control | Implementation |
|---------|---------------|
| Authentication | OAuth 2.0 + JWT tokens, API key (hashed with SHA-256) |
| Authorization | RBAC with fine-grained permissions |
| Encryption (Transit) | TLS 1.3 for all communications |
| Encryption (Rest) | AES-256-GCM for stored data |
| Input Validation | Pydantic models, max token limits, content filtering |
| Rate Limiting | Token bucket algorithm per API key / user |
| Audit Logging | Immutable audit trail for all API calls |
| Secret Management | HashiCorp Vault / K8s secrets with rotation |
| Network Security | Network policies, private subnets, no public DB access |

---

## 8. Deployment Architecture

### 8.1 Kubernetes Topology

```
┌─────────────────────── Kubernetes Cluster ──────────────────────┐
│                                                                  │
│  ┌─── Namespace: neuronlm-gateway ──┐                          │
│  │  Ingress Controller (2 replicas) │                          │
│  │  API Gateway (3 replicas)        │                          │
│  └──────────────────────────────────┘                          │
│                                                                  │
│  ┌─── Namespace: neuronlm-services ─┐                          │
│  │  Inference Service (auto-scaled)  │  ← GPU node pool        │
│  │  Conversation Service (3 rep.)    │  ← CPU node pool        │
│  │  RAG Service (2 replicas)         │  ← CPU node pool        │
│  │  Auth Service (2 replicas)        │  ← CPU node pool        │
│  └──────────────────────────────────┘                          │
│                                                                  │
│  ┌─── Namespace: neuronlm-data ─────┐                          │
│  │  PostgreSQL (HA - 3 nodes)        │                          │
│  │  Redis Cluster (6 nodes)          │                          │
│  │  Qdrant (3 nodes, sharded)        │                          │
│  │  Kafka (3 brokers + ZooKeeper)    │                          │
│  └──────────────────────────────────┘                          │
│                                                                  │
│  ┌─── Namespace: neuronlm-monitoring ┐                          │
│  │  Prometheus + Grafana              │                          │
│  │  Jaeger (distributed tracing)      │                          │
│  │  ELK Stack (logging)              │                          │
│  └──────────────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 GPU Auto-Scaling Strategy

| Metric | Scale Up | Scale Down | Cooldown |
|--------|----------|------------|----------|
| GPU Utilization | > 75% avg 2 min | < 30% avg 10 min | 5 min |
| Request Queue | > 100 pending | < 10 pending | 5 min |
| Latency P95 | > 500ms | < 100ms | 3 min |

---

## 9. Observability Design

### 9.1 Metrics (Prometheus)

| Metric | Type | Description |
|--------|------|-------------|
| `llm_request_total` | Counter | Total inference requests |
| `llm_request_duration_seconds` | Histogram | Request latency |
| `llm_tokens_generated_total` | Counter | Total tokens generated |
| `llm_tokens_per_second` | Gauge | Current throughput |
| `llm_gpu_utilization` | Gauge | GPU utilization % |
| `llm_kv_cache_utilization` | Gauge | KV cache memory % |
| `llm_active_requests` | Gauge | Currently processing |
| `llm_queue_depth` | Gauge | Requests waiting |
| `llm_model_load_time_seconds` | Histogram | Model loading latency |
| `llm_error_total` | Counter | Error count by type |

### 9.2 Structured Logging Format

```json
{
    "timestamp": "2026-04-01T10:30:00.000Z",
    "level": "INFO",
    "service": "inference-service",
    "trace_id": "abc123def456",
    "span_id": "span789",
    "user_id": "usr_xxx",
    "model_id": "neuronlm-7b",
    "event": "inference_complete",
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "latency_ms": 320,
    "gpu_id": "gpu-0"
}
```

---

## 10. Failure Modes & Resilience

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| GPU OOM | Memory monitoring | Request rejection, smaller batch |
| Model loading failure | Health check | Fallback to cached model |
| Database outage | Connection pool monitoring | Read replica failover |
| High latency | P95 alerting | Auto-scale, request shedding |
| Kafka partition loss | Consumer lag monitoring | Rebalance, replay |
| Cache miss storm | Hit rate monitoring | Circuit breaker, warm-up |
