# 15 - Full LLM Architecture Analysis of Current Codebase

Date: 2026-04-02

Scope:
This document analyzes the current NeuronLM codebase across seven architecture layers and maps it to real-world LLM system concepts.

---

## 1) Model Architecture

### Key modules and files
- Model routing and backend selection: [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L59)
- Transformers backend wrapper: [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L24)
- Bigram model architecture: [src/neuronlm/training/bigram_lm.py](src/neuronlm/training/bigram_lm.py#L18)
- Tokenization and vocabulary: [src/neuronlm/core/tokenizer.py](src/neuronlm/core/tokenizer.py#L11)
- Model metadata schema: [src/neuronlm/models/schemas.py](src/neuronlm/models/schemas.py#L126)

### Control flow
1. Runtime selects backend by configuration from settings.
2. If backend is transformers, model wrapper loads a HuggingFace causal LM lazily.
3. If backend is bigram, model behavior comes from a trained transition table.
4. Tokenizer service handles token counting, encode, and decode for request/response accounting.

### Real-world mapping
- Transformers path maps to decoder-only GPT-style architecture where embeddings, attention blocks, and logits are internal to the HuggingFace model.
- Bigram path is a statistical language model, useful as a training and serving scaffold but not semantically equivalent to modern transformer LLMs.

### Design patterns used
1. Strategy pattern via backend switch in inference engine.
2. Lazy initialization in transformers backend for first-request model load.
3. Singleton service access pattern for tokenizer and engine.

### Trade-offs and limitations
1. Fast path to working GPT inference through HuggingFace abstraction, but lower visibility into internal layer-level controls.
2. Bigram backend is extremely lightweight and trainable locally, but has low quality ceiling and weak long-context coherence.
3. Model registry exists, but physical model isolation and capability metadata are minimal.

---

## 2) Inference Flow

### Key modules and files
- Prompt assembly and generation dispatch: [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L168)
- Transformers generation call: [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L77)
- Bigram token loop: [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L260)
- Chat endpoint entry: [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L46)

### Control flow
1. API receives chat request and validates schema.
2. Route normalizes stop sequences and forwards decode parameters to inference engine.
3. Engine builds prompt from chat messages and chooses backend.
4. Transformers path calls model generate with temperature and top-p.
5. Bigram path explicitly samples next tokens in a loop.
6. Non-streaming path returns a full response object; streaming path emits SSE chunks.

### Forward pass, decoding loop, KV cache usage
- Forward pass:
  - Transformers: executed internally by HuggingFace model generate.
  - Bigram: no neural forward pass, only transition sampling.
- Decoding loop:
  - Transformers: loop is internal to model generate.
  - Bigram: explicit loop in project code.
- KV cache:
  - Not exposed as a first-class lifecycle interface in project runtime.
  - Any internal transformer caching is managed inside library defaults, not by platform policy.

### Real-world mapping
- Closest concept is a simple LLM serving gateway calling a local runtime adapter.
- Not yet equivalent to production token-scheduler engines such as vLLM continuous batching or custom decoder workers.

### Design patterns used
1. Adapter boundary between route layer and model backend.
2. Shared request parameter contract across streaming and non-streaming paths.

### Trade-offs and limitations
1. Rapid implementation with low complexity.
2. Limited token-event control and no explicit queue-driven scheduler.
3. Word-split streaming in engine rather than guaranteed token-boundary streaming.

---

## 3) Training Pipeline

### Key modules and files
- Bigram training orchestration: [src/neuronlm/training/trainer.py](src/neuronlm/training/trainer.py#L21)
- Bigram fit/save/load: [src/neuronlm/training/bigram_lm.py](src/neuronlm/training/bigram_lm.py#L27)
- Training CLI: [src/neuronlm/training/cli.py](src/neuronlm/training/cli.py#L22)

### Control flow
1. CLI reads corpus lines.
2. Trainer concatenates text and tokenizes corpus.
3. Trainer validates minimum token count.
4. Bigram fit computes transition counts and smoothed probabilities.
5. Checkpoint is saved to compressed artifact.
6. Inference engine can load checkpoint and switch to bigram backend.

### Optimization and distributed training status
- Present:
  1. Basic statistical fitting with smoothing.
  2. Persisted checkpoint artifacts.
- Missing:
  1. Neural optimization stack (loss, backpropagation, optimizer, scheduler).
  2. Mixed precision and distributed training.
  3. Data-parallel or pipeline-parallel training.

### Real-world mapping
- This is a lightweight local training path for experimentation and explainability, not an enterprise LLM pretraining or fine-tuning pipeline.

### Design patterns used
1. Separation of concerns between CLI, trainer, and model object.
2. Config object for training parameters.

### Trade-offs and limitations
1. Very low resource footprint and high reproducibility.
2. Not suitable for modern model capability growth.

---

## 4) System Design

### Key modules and files
- Application assembly: [src/neuronlm/main.py](src/neuronlm/main.py#L28)
- API endpoints: [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L46)
- Conversation APIs: [src/neuronlm/api/conversations.py](src/neuronlm/api/conversations.py)
- Middleware: [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L23)
- Schemas: [src/neuronlm/models/schemas.py](src/neuronlm/models/schemas.py#L39)
- Deployment artifacts: [docker/docker-compose.yml](docker/docker-compose.yml), [docker/Dockerfile](docker/Dockerfile)

### Control flow
1. FastAPI app starts and mounts middleware.
2. Authentication and rate limiting run before route handlers.
3. Route handlers invoke inference engine and usage tracker.
4. Responses are returned as JSON or SSE stream.
5. Health and readiness endpoints expose service state.

### Batching and scaling status
- Current:
  1. Single-process serving by default worker count.
  2. No explicit batch scheduler in inference core.
- Available scaffolding:
  1. Container setup with Postgres and Redis dependencies.
  2. Config placeholders for pool sizes and batch size.

### Real-world mapping
- Architectural shape resembles an OpenAI-compatible gateway service.
- Runtime internals are still development-grade relative to large-scale model serving.

### Design patterns used
1. Layered architecture: API, services, core utilities, models.
2. Middleware chain for cross-cutting concerns.
3. Dependency inversion through service getters.

### Trade-offs and limitations
1. Clean and readable structure for extension.
2. Missing scheduler-worker decomposition and autoscaling-aware model runtime topology.

---

## 5) Performance Optimizations

### Key modules and files
- Transformers generation settings: [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124)
- Bigram sparse transitions: [src/neuronlm/training/bigram_lm.py](src/neuronlm/training/bigram_lm.py#L49)
- Tokenizer/token accounting: [src/neuronlm/core/tokenizer.py](src/neuronlm/core/tokenizer.py#L22)
- Optional GPU dependencies list: [pyproject.toml](pyproject.toml)

### Existing optimizations
1. Lazy model loading avoids startup stalls when backend is unused.
2. Prompt truncation before generation avoids context overflow.
3. Sparse bigram representation avoids dense matrix memory blowups.
4. Deterministic pseudo-embedding path is lightweight for dev behavior.

### Missing production optimizations
1. Explicit GPU memory management policy.
2. Continuous batching and queue-aware token scheduling.
3. KV cache policy and reuse metrics.
4. Tensor-level performance tuning and kernel specialization.
5. Multi-instance sharding and load-aware routing.

### Trade-offs and limitations
1. Good developer ergonomics on CPU and small deployments.
2. Throughput and tail latency will degrade under concurrent traffic compared with production-grade runtimes.

---

## 6) Extensibility

### Key modules and files
- Backend switch point: [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L168)
- Transformers adapter boundary: [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L24)
- API contracts: [src/neuronlm/models/schemas.py](src/neuronlm/models/schemas.py#L39)
- Conversation persistence abstraction: [src/neuronlm/services/conversation.py](src/neuronlm/services/conversation.py#L14)

### How to add new models, plugins, tools
1. Add a new backend adapter class with generate and optional streaming methods.
2. Extend inference engine dispatch for the new backend.
3. Add settings fields for backend selection and model identifiers.
4. Extend schemas for any new request controls.
5. Add route-layer validation and usage tracking hooks.

### Plugin and tool status
- Tool role exists in schemas, but formal tool-calling runtime orchestration is not implemented.
- No explicit plugin registry abstraction exists yet.

### Real-world mapping
- Good base for adapter-based model integration.
- Not yet an agent platform with tool execution graph, planner, or retrieval orchestration.

### Design patterns used
1. Adapter pattern for backend integration.
2. Configuration-driven behavior selection.

### Trade-offs and limitations
1. Easy to add basic backends.
2. Harder to add enterprise-grade features without introducing a scheduler and policy bus boundary.

---

## 7) Failure Handling and Observability

### Key modules and files
- Error taxonomy: [src/neuronlm/core/exceptions.py](src/neuronlm/core/exceptions.py)
- Global exception handler: [src/neuronlm/main.py](src/neuronlm/main.py#L77)
- Request logging middleware: [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L23)
- Authentication and rate-limit middleware: [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L61), [src/neuronlm/api/middleware.py](src/neuronlm/api/middleware.py#L93)
- Usage metrics tracker: [src/neuronlm/services/usage_tracker.py](src/neuronlm/services/usage_tracker.py#L11)
- Moderation checks: [src/neuronlm/core/moderation.py](src/neuronlm/core/moderation.py#L18)

### Control flow
1. Request middleware assigns correlation id and timing headers.
2. Authentication middleware validates bearer token.
3. Rate limiter enforces per-user request quotas.
4. Routes catch model-specific errors and return typed API responses.
5. Global exception handler catches uncaught exceptions and returns safe 500 envelope.
6. Usage tracker records token counts and latency for successful flows.

### Retry and metrics status
- Present:
  1. Request timing and id headers.
  2. Per-user usage accumulation.
  3. Health and readiness checks.
- Missing:
  1. Structured distributed tracing despite dependency readiness.
  2. Centralized metrics export wiring in runtime path.
  3. Retry policies for transient model/runtime failures.
  4. Circuit breakers and fallback policies beyond backend-level fallback.

### Trade-offs and limitations
1. Good baseline safety and debugging for development.
2. Limited SRE-grade telemetry and resilience controls for production incidents.

---

## Mental Model: How Everything Connects

Think of the current platform as four rings:
1. Contract ring:
   - Pydantic schemas define OpenAI-like API payloads and responses.
2. Gateway ring:
   - FastAPI routes plus middleware perform auth, moderation, limits, logging, and endpoint dispatch.
3. Runtime ring:
   - Inference engine routes generation to simulated, bigram, or transformers adapters.
4. Data/ops ring:
   - Usage tracking, conversation store, config, and container scaffolding support operations.

Data path summary:
1. Client request enters gateway.
2. Policies and limits execute.
3. Prompt is built and sent to selected backend.
4. Backend generates output.
5. Response and usage metadata are returned and recorded.

---

## Comparison with OpenAI and Anthropic Style Production Systems

### Where this codebase aligns
1. OpenAI-compatible endpoint shapes and streaming behavior.
2. Clear service boundaries for auth, moderation, and usage tracking.
3. Configurable model backend abstraction.

### Where it differs materially
1. Runtime core:
   - Production systems use highly optimized multi-worker token schedulers with continuous batching and cache management.
   - Current system uses direct per-request execution without scheduler-worker separation.
2. Model operations:
   - Production systems operate large distributed GPU fleets with load-aware routing.
   - Current system is primarily local CPU-oriented execution.
3. Reliability and observability:
   - Production systems run full tracing, SLO-driven alerting, and robust retry/circuit-breaker controls.
   - Current system has baseline logs and health checks but limited deep telemetry.
4. Training stack:
   - Production systems include large-scale distributed optimization pipelines and post-training stacks.
   - Current system provides a lightweight bigram training flow and inference adapter integration.

### Practical bottom line
The codebase is a strong educational and prototyping architecture with realistic API and service boundaries.
To reach OpenAI or Anthropic class production behavior, the largest missing pieces are scheduler-driven inference runtime, cache and batching infrastructure, distributed model operations, and deeper reliability instrumentation.
