# 10 - Interface Mapping: Current NeuronLM vs Enterprise GPT

Date: 2026-04-01

This document maps the **interface layer** of this project to a real enterprise GPT system (OpenAI-style contract + enterprise serving controls).

Scope of interface in this doc:
1. External API contract (endpoints, request/response schema, streaming behavior).
2. Control interface (auth, rate-limit, policy, usage, observability).
3. Runtime-facing interface (how API hands off to model execution).

---

## 1) Interface Components in Current Project

Current interface components:
1. API app assembly and middleware chain in `src/neuronlm/main.py`.
2. OpenAI-style endpoints in `src/neuronlm/api/routes.py`.
3. Conversation endpoints in `src/neuronlm/api/conversations.py`.
4. Auth/rate-limit/logging middleware in `src/neuronlm/api/middleware.py`.
5. Request/response schemas in `src/neuronlm/models/schemas.py`.
6. Backend dispatch contract in `src/neuronlm/services/inference_engine.py`.

---

## 2) Endpoint-Level Mapping

| Interface Area | Current NeuronLM | Enterprise GPT Pattern | Gap Level | Impact |
|---|---|---|---|---|
| `POST /v1/chat/completions` | Present, OpenAI-like payload and response | Present, usually with broader fields/tool protocol support | Medium | Basic compatibility exists, advanced agent/tool workflows limited |
| Chat streaming | SSE via `text/event-stream`, chunk objects | SSE/WebSocket with token-level streaming and stronger event guarantees | Medium | Works for dev usage; weaker real-time token semantics |
| `POST /v1/embeddings` | Present | Present with real embedding model families and versioned behavior | Medium | Contract exists; quality depends on backend |
| `GET /v1/models` + `GET /v1/models/{id}` | Present | Present with richer metadata, capabilities, and lifecycle states | Low-Medium | Discoverability works; capability negotiation is limited |
| Usage endpoint (`GET /v1/usage`) | Present | Enterprise often has metering/billing APIs + quota policy APIs | Medium | Good start, not full tenancy/billing model |
| Conversation endpoints | Present (`/v1/conversations/...`) | Often separate product API or session API with retention policy controls | Medium | Useful app feature, but governance controls are thin |
| Health/ready | Present (`/health`, `/ready`) | Present with deep dependency/runtime probes and SLO indicators | Low-Medium | Good baseline; limited production depth |

---

## 3) Request/Response Schema Mapping

### 3.1 Chat Request

Current request supports:
1. `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stream`, `stop`, penalties, `seed`, `user`.

Enterprise GPT request expectations typically add:
1. Tool/function calling schema (strict JSON schema, tool choice policy).
2. Structured output modes (JSON schema constrained decoding).
3. Multi-modal message parts (text/image/audio/video references).
4. Per-request policy flags and compliance tags.
5. Metadata for tenancy/project/billing lineage.

Gap summary:
1. Core chat fields are aligned.
2. Tooling, structured outputs, multimodal, and governance metadata are missing.

### 3.2 Chat Response

Current response supports:
1. `id`, `object`, `created`, `model`, `choices`, `usage`, finish reason.

Enterprise GPT response typically adds:
1. Tool call payloads and invocation traces.
2. Safety annotations and moderation traces.
3. Cache accounting (prompt cache hit/miss) and richer token classes.
4. System-level trace/correlation IDs exposed for supportability.

Gap summary:
1. Shape is compatible for basic clients.
2. Enterprise debugging and governance metadata is incomplete.

---

## 4) Security and Governance Interface Mapping

| Control Plane Interface | Current NeuronLM | Enterprise GPT Pattern | Gap Level |
|---|---|---|---|
| Authentication | JWT/API key style auth middleware | Org/project-scoped keys, rotation, scoped tokens, mTLS in internal hops | Medium |
| Authorization | Basic user context in request state | Fine-grained scopes (model access, tools, data domains) | Medium-High |
| Rate limiting | Per-user check + headers | Multi-dimensional quotas (RPM/TPM/day/concurrency/spend), adaptive limits | High |
| Content policy gate | Input moderation check | Input + output + tool-result policy chain with auditable outcomes | High |
| Error model | Standard error envelope | Taxonomy with retry classes, policy classes, and localized diagnostics | Medium |
| Auditability | Request logging + usage tracker | Immutable audit logs, policy decision logs, retention controls | High |

---

## 5) Runtime Handoff Interface Mapping

Current handoff:
1. API route validates request and passes normalized fields into `InferenceEngine`.
2. Engine dispatches by backend (`simulated`, `bigram`, `transformers`).
3. Streaming currently emits chunked text segments.

Enterprise GPT handoff pattern:
1. API gateway -> scheduler/admission controller.
2. Scheduler -> model workers with dynamic batching and queue policy.
3. Workers manage KV cache/session state and token-by-token emission.
4. Runtime returns token events + telemetry events.

Main interface gap:
1. Missing formal scheduler/worker protocol interface between API and runtime.
2. Missing token-event telemetry contract.
3. Missing cache/session lifecycle contract (for KV reuse and long dialogs).

---

## 6) Compatibility Scorecard (Interface-Only)

Scoring guide:
1. 0 = absent
2. 1 = minimal
3. 2 = partial
4. 3 = production-ready baseline
5. 4 = enterprise-strong

| Interface Capability | Score | Notes |
|---|---|---|
| OpenAI-style endpoint shape | 3/4 | Good baseline compatibility |
| Chat request parameter coverage | 2/4 | Core decoding fields present; missing tools/JSON schema |
| Streaming protocol robustness | 2/4 | SSE exists; token/event semantics are basic |
| Auth and request identity | 2/4 | Works, but limited tenancy/scope model |
| Quota and limit controls | 1/4 | Basic rate limiting only |
| Policy + moderation interface | 1/4 | Input-only moderation path |
| Usage/billing interface | 1/4 | Basic usage endpoint, no billing-grade metering |
| Runtime handoff interface | 1/4 | No scheduler-worker contract |
| Operability interface (traceability) | 2/4 | Request IDs + logs, limited deep telemetry |

Overall interface maturity vs enterprise GPT: **~1.9/4 (partial)**.

---

## 7) Target Interface Blueprint (What to Add)

### Phase 1: Enterprise-Compatible API Contract
1. Add tool-calling request/response fields and validation.
2. Add structured JSON response mode (schema-constrained output).
3. Add standardized metadata fields (`project_id`, `tenant_id`, `request_tags`).
4. Extend error taxonomy with retry guidance and policy categories.

### Phase 2: Governance and Quota Interfaces
1. Add output moderation and policy decision envelope.
2. Add quota APIs (usage buckets, resets, spend guards).
3. Add audit log export interface with request and policy correlation IDs.

### Phase 3: Runtime Interface Separation
1. Introduce an internal scheduler-worker API boundary.
2. Add token-event protocol for streaming (`token`, `logprob`, `latency`, `cache_hit`).
3. Add session/cache lifecycle interface (create/resume/evict conversation cache state).

### Phase 4: Operability and SRE Interfaces
1. Add per-endpoint SLO reporting interface.
2. Add trace propagation headers and runtime span IDs.
3. Add admin/status APIs for model worker pools and queue depth.

---

## 8) Concrete Mapping to Current Files

| File | Role in Interface | Enterprise Alignment | Next Upgrade |
|---|---|---|---|
| `src/neuronlm/main.py` | App assembly, middleware ordering, health/readiness | Good skeleton | Add deeper readiness probes and trace propagation |
| `src/neuronlm/api/routes.py` | Main OpenAI-style LLM endpoints | Good basic shape | Add tools/structured outputs/policy metadata |
| `src/neuronlm/api/conversations.py` | Session-style conversation APIs | Useful extension | Add retention policy, export/delete governance, pagination cursors |
| `src/neuronlm/api/middleware.py` | Auth, limits, logging | Baseline control plane | Add scope-based authz and multidimensional quotas |
| `src/neuronlm/models/schemas.py` | External contract schema | Baseline compatibility | Add multimodal/tool/schema constraints |
| `src/neuronlm/services/inference_engine.py` | API-to-runtime handoff | Basic backend dispatch | Introduce scheduler/worker contract and token telemetry hooks |

---

## 9) Summary

Current NeuronLM already has a strong **enterprise-shaped API skeleton**:
1. OpenAI-style chat/embedding/model endpoints.
2. Middleware for auth, rate limiting, and request logging.
3. Streaming path and usage tracking.

To match an actual enterprise GPT interface, the biggest upgrades are:
1. Tool and structured-output contracts.
2. Governance-grade controls (authz, quota, policy/audit).
3. Formal runtime handoff interface (scheduler/worker/token events/cache lifecycle).
