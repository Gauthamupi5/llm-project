# NeuronLM — Complete System Architecture

## Purpose

This document captures the complete system architecture and end-to-end flows for the NeuronLM project: request flows, training and model lifecycle, data flows, deployment topology, observability, security, and operational practices.

---

## 1. Overview

NeuronLM is an enterprise LLM platform composed of the following logical subsystems:

- Presentation: Chat UI, Admin UI, CLI/SDK
- API Gateway: authentication, rate-limiting, routing
- Inference Service: model serving (streaming + non-streaming)
- Conversation Service: session management, history
- RAG Engine: embeddings, retrieval, re-ranking
- Model Registry & Training: model artifacts, fine-tuning, RLHF
- Infrastructure: Kubernetes, GPU node pools, Postgres, Redis, Vector DB (Qdrant), Object store (MinIO)
- Observability & Ops: Prometheus, Grafana, Jaeger, ELK

All modules are designed for modular deployment and horizontal scaling.

---

## 2. High-level Component Diagram

```
User/UI  →  API Gateway  →  Service Mesh  →  {Inference, Conversation, RAG, Auth, Usage}
                                   │                 │
                                   ▼                 ▼
                           Model Registry         Data Stores (Postgres, Redis, Qdrant, MinIO)
                                   │
                                   ▼
                           Training Fleet (GPU Cluster)
```

---

## 3. Core Flows (step-by-step)

### 3.1 External Request → Response (Inference) Flow

1. Client (Web/SDK) makes request to `POST /v1/chat/completions` with `Authorization: Bearer <api-key>`.
2. Request hits API Gateway (Kong/Envoy): validate JWT/API key, apply rate limit, add request ID, route to Inference Service.
3. Authentication middleware maps API key → user, quotas, scopes.
4. Conversation Service (if conversation_id provided) returns conversation history and system prompt.
5. Context assembler composes `[system prompt + safety instructions + RAG context + history + new message]` and applies token budget/truncation.
6. Tokenizer converts text → token IDs; TokenizerService returns token counts.
7. Inference Engine selects model (`Model Registry`): verifies model status and serving config.
8. Forward path:
   - If `stream=true`: Inference Engine runs `stream_complete()` streaming SSE/chunked responses.
   - Else: runs `complete()` to generate full response.
   - Under the hood: embeddings → stacked Transformer decoder blocks → logits → sampling/decoding.
9. Content Moderation runs on inputs and (post-filter) on outputs.
10. UsageTracker records prompt/completion tokens and latency.
11. Conversation Store persists messages and metadata.
12. Response returned to client; gateway returns logs/metrics.

Key notes: KV cache and batching are applied to group simultaneous requests for GPU efficiency.

### 3.2 Embeddings Flow

1. Client calls `POST /v1/embeddings` with input text.
2. Tokenizer → tokens → embedding generator (`generate_embedding()` returns deterministic vector).
3. Vector returned and optionally stored in Vector DB (Qdrant) by RAG ingestion pipeline.

### 3.3 RAG (Retrieval-Augmented Generation) Flow

1. Document ingestion pipeline: fetch → dedupe → chunk → embed → index in Qdrant (with metadata).
2. Query flow: on chat request, RAG builder performs similarity search (Qdrant) → re-ranker (cross-encoder optional) → selected contexts appended to prompt.
3. Retrieval results tracked in logs for auditing and provenance.

### 3.4 Training / Model Lifecycle Flow

1. Data pipeline: raw sources → cleaning/dedupe → tokenization → dataset shards stored (with lineage metadata).
2. Pre-training (if executed): distributed training across GPU fleet (FSDP/ZeRO), checkpointing to object store.
3. SFT / Fine-tuning: smaller jobs (LoRA/QLoRA) for targeted updates, evaluation, validation.
4. RLHF: reward model training, preference collection, PPO/DPO loops (optional roadmap).
5. Model Registry: versions, evaluation metrics, canary flags, serving config.
6. Deployment: build container image for model server → push to registry → deploy via `model rollout manager` (canary → gradual → full) with health checks and automatic rollback on SLO violations.

---

## 4. Data Flow & Lineage

- All training and ingestion pipelines tag artifacts with provenance metadata (source, transform steps, hashes).
- Conversation data persisted with user IDs, token counts and model version used for traceability.
- Sensitive data handling: PII detection during ingestion; redact/omit per policy before indexing.

---

## 5. Deployment Topology

- Kubernetes clusters split into namespaces: `gateway`, `services`, `data`, `monitoring`.
- GPU node pool (A100/H100) for inference/training jobs.
- Multi-zone / multi-region design recommended for production.
- `docker-compose` included for dev/local emulation.

---

## 6. Observability, Metrics & SLOs

- Metrics exposed by services (Prometheus): request_count, request_latency_p50/p95/p99, tokens_generated, model_load_time, kv_cache_hit_ratio.
- Traces: Jaeger for request & model inference spans.
- Logs: structured JSON logs pushed to ELK; include trace_id, model_id, user_id, token_counts.
- SLO examples: 99.9% availability, p95 latency < 500ms (non-streaming baseline), error rate < 0.1%.
- Alerts: SLO breaches, model OOM, high queue depth, data pipeline failures.

---

## 7. Security & Compliance

- Transport: TLS 1.3
- Keys: API keys hashed with SHA-256, jwt tokens for session auth
- Secrets: HashiCorp Vault or K8s secrets with rotation
- RBAC via Auth service, OPA/Cedar for policy controls
- Audit logs for model access and dataset changes

---

## 8. Operational Patterns

- Canary deployments for models with automated health checks and scripted rollback.
- Continuous evaluation during rollout: run a small benchmark set on canary traffic.
- Autoscaling: HPA for CPU services; custom scaler for GPU pools based on queue depth and GPU utilization.
- Runbook highlights: steps for GPU OOM, model rollback, data pipeline failure, and security incident response.

---

## 9. Failure Modes & Mitigations (summary)

- GPU OOM: reject large requests, fallback to smaller models, use paged KV cache.
- Slow model load: keep warm replicas, prefetch weights, use lazy-loading with warm pool.
- Data corruption in pipeline: validation gate, checksum verification, revert to last known good dataset.
- Unsafe output: moderation + policy engine blocking and operator alerting.

---

## 10. File & Service Map (where to find code)

- API & routes: `src/neuronlm/api/routes.py`
- Middleware: `src/neuronlm/api/middleware.py`
- Inference engine: `src/neuronlm/services/inference_engine.py`
- Conversation store: `src/neuronlm/services/conversation.py`
- Tokenizer: `src/neuronlm/core/tokenizer.py`
- Auth: `src/neuronlm/core/auth.py`
- Model config & settings: `src/neuronlm/config.py`
- Docs: `docs/02_architecture_design.md`, `docs/03_training_transformer_mapping.md`, `docs/04_gaps_gpt_vs_neuronlm.md`

---

## 11. Next Steps (recommended immediate actions)

- Add `model rollout manager` design and implementation (canary + auto-rollback).
- Add dataset registry + lineage metadata collection in ingestion pipeline.
- Implement production-grade KV caching (vLLM style) and benchmark.
- Define SLOs and wire automated canary evaluation for model releases.

---

Document created: `docs/05_system_architecture.md` — let me know if you want a Mermaid visual, a runbook, or this converted into a presentation.
