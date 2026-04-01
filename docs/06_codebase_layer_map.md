# 06 - Codebase Layer Map Template

Use this template to analyze any LLM codebase end-to-end and map implementation details into five architecture layers.

## How To Use

1. Replace placeholders in angle brackets, for example `<PROJECT_NAME>`.
2. For each layer, list concrete files and symbols first.
3. Explain control flow in sequence form (`1 -> 2 -> 3`).
4. Map each code artifact to an LLM concept.
5. Capture performance trade-offs and missing capabilities.
6. Finish with the Mental Model Summary block.

## Analysis Metadata

- Project: `<PROJECT_NAME>`
- Repository root: `<REPO_PATH>`
- Commit/branch analyzed: `<COMMIT_OR_BRANCH>`
- Date: `<YYYY-MM-DD>`
- Analyst: `<NAME>`

---

## Layer 1 - Model Type (Design)

### 1.1 Model Type Identification

- Model family: `<GPT | MoE | encoder-only | encoder-decoder | multimodal | hybrid>`
- Deployment model variants: `<7B/13B/70B/etc>`
- Training objective (if visible in code/docs): `<autoregressive | masked LM | seq2seq | RLHF>`

### 1.2 Where Architecture Is Defined

- Key files/modules:
  - `<path/to/file_1>`
  - `<path/to/file_2>`
- Key symbols/classes/functions:
  - `<ClassOrFunctionName>` in `<path/to/file>`

### 1.3 Design Decisions Affecting Computation

- Decision: `<example: decoder-only causal mask>`
  - Code location: `<path/to/file#symbol>`
  - Impact: `<latency/memory/quality implications>`
- Decision: `<example: RoPE vs absolute positions>`
  - Code location: `<...>`
  - Impact: `<...>`

### 1.4 Trade-offs

- Strengths:
  - `<bullet>`
- Risks/limitations:
  - `<bullet>`

---

## Layer 2 - One Pass (Forward Computation)

### 2.1 Single Forward Pass Trace

- Entry function: `<symbol>` in `<path>`
- Pass sequence:
  1. `<input preprocessing/tokenization>`
  2. `<embedding lookup/projection>`
  3. `<attention block stack>`
  4. `<feedforward/MLP>`
  5. `<final norm + lm_head/logits>`

### 2.2 Core Model Layers and forward() Paths

- File map:
  - Embeddings: `<path#symbol>`
  - Attention: `<path#symbol>`
  - FFN/MLP: `<path#symbol>`
  - Residual + norm: `<path#symbol>`
  - Output head/logits: `<path#symbol>`

### 2.3 Code -> Concept Mapping

| Code Artifact | Concept | Notes |
|---|---|---|
| `<path#symbol>` | `<embedding/attention/ffn/logits>` | `<details>` |
| `<path#symbol>` | `<...>` | `<...>` |

### 2.4 Trade-offs

- Compute complexity notes: `<O(n^2) attention, flash attention, etc>`
- Numerical/runtime concerns: `<precision, stability, kernel choices>`

---

## Layer 3 - Inference Loop (Token Generation)

### 3.1 Generation Loop Location

- Main loop function: `<symbol>` in `<path>`
- Control variables: `<max_tokens, eos, stop sequences, timeout>`

### 3.2 Decoding Strategy

- Implemented decoding: `<greedy | top-k | top-p | beam | contrastive | speculative>`
- Sampling location: `<path#symbol>`
- Logits processing: `<temperature, repetition penalty, bad words, etc>`

### 3.3 Token Append and Loop Control

- Token append path: `<path#symbol>`
- EOS/stop handling: `<path#symbol>`
- Stream chunk formatting: `<path#symbol>`

### 3.4 Code -> Concept Mapping

| Code Artifact | Concept | Notes |
|---|---|---|
| `<path#symbol>` | `decoding` | `<details>` |
| `<path#symbol>` | `termination criteria` | `<details>` |

### 3.5 Trade-offs

- Latency vs quality trade-offs:
  - `<example: greedy is faster but lower diversity>`
- Throughput impacts:
  - `<example: beam search increases compute per token>`

---

## Layer 4 - System Architecture (Orchestration)

### 4.1 API Entry Points

- API files:
  - `<path/to/router_or_handler>`
- Endpoint(s):
  - `<METHOD /path>` -> `<handler symbol>`

### 4.2 Request Handling Flow

1. `<auth/middleware>`
2. `<validation/moderation>`
3. `<routing to inference service>`
4. `<response shaping/streaming>`

### 4.3 Scheduling and Batching

- Scheduler present: `<yes/no>`
- Batching strategy: `<static/dynamic/none>`
- Queueing model: `<in-process queue/external broker/none>`

### 4.4 GPU Worker Execution

- Runtime backend: `<vLLM/TensorRT-LLM/Transformers/custom/none>`
- Worker topology: `<single process/multi worker/distributed>`
- Placement controls: `<device map, affinity, MIG, etc>`

### 4.5 Streaming Responses

- Protocol: `<SSE/WebSocket/gRPC stream>`
- Chunk schema file: `<path>`
- Backpressure/retry behavior: `<details>`

### 4.6 Trade-offs

- Operational strengths:
  - `<bullet>`
- Scalability/reliability gaps:
  - `<bullet>`

---

## Layer 5 - Optimization Layer

### 5.1 KV Cache

- Implemented: `<yes/no/partial>`
- Code location: `<path#symbol>`
- Cache policy: `<per request/per conversation/eviction strategy>`

### 5.2 Batching Strategy

- Prefill batching: `<yes/no>`
- Decode batching: `<yes/no>`
- Dynamic batch controls: `<max batch, wait window, priorities>`

### 5.3 Parallelism

- Tensor parallelism: `<yes/no>`
- Pipeline parallelism: `<yes/no>`
- Data parallelism: `<yes/no>`
- Expert parallelism (MoE): `<yes/no>`

### 5.4 Memory Management

- Precision/quantization: `<fp16/bf16/int8/int4/etc>`
- Offload strategy: `<CPU/NVMe/off>`
- Fragmentation controls: `<allocator settings/pools>`

### 5.5 Code -> Concept Mapping

| Code Artifact | Concept | Notes |
|---|---|---|
| `<path#symbol>` | `kv cache` | `<details>` |
| `<path#symbol>` | `batch scheduler` | `<details>` |
| `<path#symbol>` | `parallel strategy` | `<details>` |

### 5.6 Trade-offs

- Latency trade-offs:
  - `<bullet>`
- Throughput trade-offs:
  - `<bullet>`
- Cost trade-offs:
  - `<bullet>`

---

## Cross-Layer Gap Analysis

### Present vs Missing Capabilities

- Present:
  - `<capability>`
- Missing:
  - `<capability>`
- Impact severity: `<high/medium/low>`

### Priority Recommendations

1. `<highest impact improvement>`
2. `<second improvement>`
3. `<third improvement>`

---

## Mental Model Summary

Model Type:
- `<one to three lines>`

ONE PASS:
- `<one to three lines>`

INFERENCE LOOP:
- `<one to three lines>`

SYSTEM:
- `<one to three lines>`

OPTIMIZATION:
- `<one to three lines>`

---

## Optional Evidence Appendix

### A. File Inventory Used

- `<path>`
- `<path>`

### B. Functions Traced

- `<symbol>` in `<path>`
- `<symbol>` in `<path>`

### C. Open Questions

- `<question requiring deeper inspection>`
- `<question requiring runtime benchmark>`
