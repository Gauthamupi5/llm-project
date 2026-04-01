# 13 - Current Project Mapping with Code Links (Transformers Path)

Date: 2026-04-01

Purpose:
This document maps the current project implementation to the requested architecture when using distilgpt2, gpt2, or gpt2-medium, and explicitly shows what is missing.

---

## 1) Models in Scope

Configured model backend and model name are defined in:
- [src/neuronlm/config.py](src/neuronlm/config.py#L68)
- [src/neuronlm/config.py](src/neuronlm/config.py#L70)

Transformers backend implementation is in:
- [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L24)

Supported names in this project path:
1. distilgpt2
2. gpt2
3. gpt2-medium

---

## 2) Layer Mapping (Implemented vs Missing)

| Requested Layer | Current Project Status | Code Link(s) | Notes |
|---|---|---|---|
| Inference System (Core) | Partially implemented | [src/neuronlm/main.py](src/neuronlm/main.py#L28), [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L46), [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L59) | API + middleware + engine exist, but no enterprise scheduler-worker split |
| Inference Loop (token-by-token generation) | Partially implemented | [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L192), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124) | Loop is handled inside HF generate, not explicit per-token engine loop |
| One Pass: Embedding -> Transformer Layers -> Logits | Implemented (model-internal) | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L58), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59) | One-pass happens in HuggingFace model forward internals |
| Decoding Strategy -> Next Token | Partially implemented | [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L87), [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L88), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L77), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124) | temperature/top_p/seed supported; advanced enterprise decode controls are missing |
| KV Cache | Missing as enterprise layer | [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L59) | No explicit KV lifecycle API, policy, or observability in project runtime |
| Dynamic Batching | Missing | [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L59) | No continuous/dynamic batching module |
| Scheduler / Admission Control | Missing | [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L59) | No queue policy, priority, backpressure, or admission controller |
| Model Type: GPT | Implemented | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59) | distilgpt2/gpt2/gpt2-medium are GPT-style causal LMs |
| Model Type: MoE | Missing | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L24) | No MoE model path in current implementation |
| Model Type: Multimodal | Missing | [src/neuronlm/models/schemas.py](src/neuronlm/models/schemas.py#L34), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L77) | Current request path and backend are text-first only |

---

## 3) Verified Flow in Current Code

### 3.1 API and middleware ingress
- App creation and middleware chain:
  - [src/neuronlm/main.py](src/neuronlm/main.py#L28)
  - [src/neuronlm/main.py](src/neuronlm/main.py#L57)
  - [src/neuronlm/main.py](src/neuronlm/main.py#L68)
- LLM API endpoints:
  - [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L46)
  - [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L152)
  - [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L199)
  - [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L225)
- Request controls:
  - [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L23)
  - [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L61)
  - [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L93)

### 3.2 Backend selection and generation
- Backend selected from settings and transformers backend initialized:
  - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L93)
  - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L94)
- Transformers dispatch path for response generation:
  - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L192)
- Transformers model load + generate call:
  - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L40)
  - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59)
  - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124)

---

## 4) What Is Missing (Enterprise Mapping Gaps)

Missing layers and missing code modules in this project today:
1. KV cache lifecycle manager module (create/reuse/evict/ttl APIs).
2. Scheduler module for request admission and queueing.
3. Dynamic batching module with queue coalescing.
4. Worker runtime boundary (separate scheduler-worker protocol).
5. MoE and multimodal model execution paths.

Current effect:
1. Throughput and latency under concurrency are limited.
2. Runtime behavior is less predictable under spikes.
3. Model-type coverage is GPT-text only.

---

## 5) Direct Answer

Does current project use the requested mapping?
1. Yes, partially for GPT transformers path.
2. No, not fully for enterprise runtime layers (KV cache, batching, scheduler, worker orchestration, MoE, multimodal).

So for distilgpt2, gpt2, and gpt2-medium:
1. One-pass and decoding exist via HuggingFace internals.
2. Enterprise inference infrastructure layers are still missing.
