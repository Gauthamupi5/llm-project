# 11 - Enterprise GPT Interface Upgrade Checklist (Execution Plan)

Date: 2026-04-01

Purpose:
This is an implementation-ready checklist that turns the interface gap mapping into concrete engineering work.

Effort scale:
- S = 0.5 to 1.5 days
- M = 2 to 5 days
- L = 1 to 3 weeks

---

## 1) API Contract Upgrades

### 1.1 Tool Calling Support
- File(s):
  - `src/neuronlm/models/schemas.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/services/inference_engine.py`
- Effort: L
- Tasks:
  1. Add tool/function definitions to chat request schema.
  2. Add assistant tool-call response structure (tool call id, name, arguments).
  3. Add tool-choice policy (`none`, `auto`, forced tool).
  4. Validate tool payloads and produce deterministic error messages.
- Done criteria:
  1. `/v1/chat/completions` accepts tool specs and returns tool-call objects.
  2. Invalid tool schema returns consistent `invalid_request_error`.
  3. Streaming mode emits tool-call chunks correctly.

### 1.2 Structured JSON Output Mode
- File(s):
  - `src/neuronlm/models/schemas.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/services/inference_engine.py`
- Effort: M
- Tasks:
  1. Add `response_format` or JSON schema field to request.
  2. Add output post-validation layer (JSON parse + schema validation).
  3. Return explicit error type when generated output violates schema.
- Done criteria:
  1. Request can enforce JSON object output.
  2. Non-conformant outputs fail with typed error.
  3. Happy-path returns valid schema-conformant JSON.

### 1.3 Metadata and Tenant Fields
- File(s):
  - `src/neuronlm/models/schemas.py`
  - `src/neuronlm/api/middleware.py`
  - `src/neuronlm/api/routes.py`
- Effort: M
- Tasks:
  1. Add optional request metadata fields (`project_id`, `tenant_id`, `request_tags`).
  2. Validate size/shape and redact unsafe values from logs.
  3. Propagate metadata to usage tracking.
- Done criteria:
  1. Metadata accepted and available throughout request lifecycle.
  2. Logs and usage include safe correlation fields.

---

## 2) Governance and Security Upgrades

### 2.1 Scope-Based Authorization
- File(s):
  - `src/neuronlm/api/middleware.py`
  - `src/neuronlm/core/auth.py`
  - `src/neuronlm/api/routes.py`
- Effort: L
- Tasks:
  1. Add scope model (`chat:write`, `embedding:write`, `models:read`, `admin:*`).
  2. Enforce scope checks by route.
  3. Return standardized authz-denied errors.
- Done criteria:
  1. Routes reject tokens missing required scopes.
  2. Error payloads include stable authz code.

### 2.2 Multi-Dimensional Quotas
- File(s):
  - `src/neuronlm/core/rate_limiter.py`
  - `src/neuronlm/api/middleware.py`
  - `src/neuronlm/services/usage_tracker.py`
- Effort: L
- Tasks:
  1. Extend limiter to RPM + TPM + daily token budget + concurrent requests.
  2. Add model-specific limits.
  3. Return full rate-limit headers for each dimension.
- Done criteria:
  1. Limits enforce independently and together.
  2. Retry and remaining values are accurate under load tests.

### 2.3 Output Moderation + Policy Trace
- File(s):
  - `src/neuronlm/core/moderation.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/models/schemas.py`
- Effort: M
- Tasks:
  1. Add output moderation pass before final response.
  2. Add policy decision metadata object (internal and optional external view).
  3. Add policy refusal reason taxonomy.
- Done criteria:
  1. Harmful outputs are blocked/rewritten per configured behavior.
  2. Policy decision is logged and correlated with request id.

### 2.4 Audit Log Interface
- File(s):
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/services/usage_tracker.py`
  - `src/neuronlm/main.py`
- Effort: M
- Tasks:
  1. Add immutable audit event writer (append-only abstraction).
  2. Add admin export endpoint for audit slices by time window.
  3. Include request id, user id, model id, policy action.
- Done criteria:
  1. Audit events are queryable and traceable to requests.
  2. Export endpoint returns paginated deterministic slices.

---

## 3) Runtime Interface Separation

### 3.1 Scheduler-Worker Contract
- File(s):
  - `src/neuronlm/services/inference_engine.py`
  - `src/neuronlm/services/` (new files: `scheduler.py`, `worker_protocol.py`)
  - `src/neuronlm/models/schemas.py`
- Effort: L
- Tasks:
  1. Define internal request object for runtime jobs.
  2. Add queue/admission logic (priority, timeout, cancellation).
  3. Convert current direct engine call to scheduler dispatch.
- Done criteria:
  1. API layer no longer directly manages backend-specific execution flow.
  2. Scheduler metrics expose queue depth and wait time.

### 3.2 Token Event Streaming Protocol
- File(s):
  - `src/neuronlm/services/inference_engine.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/models/schemas.py`
- Effort: M
- Tasks:
  1. Emit normalized token events (`token`, `finish_reason`, `latency_ms`).
  2. Preserve OpenAI-compatible SSE shape while adding internal trace hooks.
  3. Add robust stream termination and cancellation handling.
- Done criteria:
  1. Streaming is token-consistent across backends.
  2. Interrupted clients cancel generation cleanly.

### 3.3 Session and Cache Lifecycle Interface
- File(s):
  - `src/neuronlm/api/conversations.py`
  - `src/neuronlm/services/conversation.py`
  - `src/neuronlm/services/inference_engine.py`
- Effort: L
- Tasks:
  1. Add session metadata for cache eligibility and context reuse.
  2. Add explicit cache control policy (`reuse`, `evict`, TTL).
  3. Add administrative cache invalidation endpoint.
- Done criteria:
  1. Session/cache state is managed with explicit lifecycle semantics.
  2. Cache controls are observable and testable.

---

## 4) Observability and Operability Interfaces

### 4.1 Trace Propagation and Correlation
- File(s):
  - `src/neuronlm/api/middleware.py`
  - `src/neuronlm/main.py`
  - `src/neuronlm/api/routes.py`
- Effort: M
- Tasks:
  1. Support `traceparent` header passthrough and generation.
  2. Include trace id in logs, errors, and audit events.
  3. Add request->stream correlation id consistency checks.
- Done criteria:
  1. Single request can be traced from ingress to response chunks.

### 4.2 SLO Status and Admin Runtime Endpoints
- File(s):
  - `src/neuronlm/main.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/services/inference_engine.py`
- Effort: M
- Tasks:
  1. Add endpoint(s) for runtime health by component (scheduler, workers, cache).
  2. Publish simple SLO snapshots (p50/p95 latency, error rate).
  3. Add guarded admin route access.
- Done criteria:
  1. Runtime status and service quality are externally inspectable.

### 4.3 Usage/Billing Grade Metrics Contract
- File(s):
  - `src/neuronlm/services/usage_tracker.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/models/schemas.py`
- Effort: M
- Tasks:
  1. Record prompt/completion by model and endpoint with timestamps.
  2. Add aggregate query options (window, model, tenant).
  3. Add idempotency strategy for duplicated request submission.
- Done criteria:
  1. Usage reports are stable under retries/replays.

---

## 5) Error Contract Standardization

### 5.1 Error Taxonomy Upgrade
- File(s):
  - `src/neuronlm/models/schemas.py`
  - `src/neuronlm/core/exceptions.py`
  - `src/neuronlm/api/routes.py`
  - `src/neuronlm/main.py`
- Effort: M
- Tasks:
  1. Define canonical error classes (`invalid_request`, `policy_denied`, `quota_exceeded`, `transient_runtime`, etc.).
  2. Add retryability hint and support correlation id in error body.
  3. Ensure consistent mapping from raised exceptions to error envelopes.
- Done criteria:
  1. Every endpoint emits consistent typed errors.
  2. Error contract is documented and test-covered.

---

## 6) Test Checklist (Must Add)

### 6.1 API Contract Tests
- File(s):
  - `tests/` (new: `tests/integration/test_api_interface_contract.py`)
- Effort: M
- Tasks:
  1. Validate all mandatory fields and major optional fields.
  2. Validate streaming chunk shape and final terminator behavior.
  3. Validate error payload consistency for invalid inputs.

### 6.2 Security and Quota Tests
- File(s):
  - `tests/` (new: `tests/integration/test_authz_and_limits.py`)
- Effort: M
- Tasks:
  1. Missing scope failures.
  2. Per-dimension quota failures and headers.
  3. Policy gate behavior on blocked outputs.

### 6.3 Runtime Boundary Tests
- File(s):
  - `tests/` (new: `tests/unit/test_scheduler_worker_contract.py`)
- Effort: M
- Tasks:
  1. Queue timeout behavior.
  2. Cancellation path.
  3. Event emission ordering and finish reasons.

---

## 7) Suggested Delivery Sequence

1. Sprint 1 (Contract Baseline):
   - Tool calling scaffold
   - Structured JSON outputs
   - Error taxonomy standardization
2. Sprint 2 (Governance):
   - Scope authz
   - Multi-dimensional quotas
   - Output moderation and policy trace
3. Sprint 3 (Runtime Interface):
   - Scheduler-worker boundary
   - Token event protocol
   - Session/cache lifecycle
4. Sprint 4 (Operability):
   - Trace propagation
   - Admin/SLO endpoints
   - Billing-grade usage metrics

---

## 8) Quick Wins (Start This Week)

1. Add canonical error envelope and correlation ids (M).
2. Add response-format JSON mode with validation (M).
3. Add basic scope-based route guards (M).
4. Add integration tests for streaming contract (M).

These four items provide the fastest jump in enterprise interface readiness without major runtime refactor risk.
