# 07 - Real Enterprise LLM vs Current NeuronLM Mapping

This document maps a real enterprise LLM stack to the current NeuronLM implementation.

## Scope

- Compare architecture and runtime behavior.
- Show what is implemented now.
- Show what is simulated.
- Provide a practical upgrade path.

---

## Layer A - Model Design

### Real Enterprise LLM

- Trained model weights (checkpointed parameters).
- Concrete architecture (decoder-only GPT / MoE / encoder-decoder).
- Explicit layer stack (embeddings, attention blocks, FFN/MLP, norm, lm_head).
- Versioned artifacts and model registry.

### Current NeuronLM

- Model IDs exist in registry (`neuronlm-7b`, `neuronlm-13b`, `neuronlm-7b-chat`, `neuronlm-embed`).
- No trained checkpoint loading path in runtime.
- Output behavior is template/simulated logic.

### Code Mapping

- Model registry and config:
  - `src/neuronlm/services/inference_engine.py` (`ModelConfig`, `_register_default_models`)
- API schema contracts:
  - `src/neuronlm/models/schemas.py`

### Gap Summary

- Present: model identity and API contracts.
- Missing: real trained model graph and weight loading.

---

## Layer B - ONE PASS (Forward Computation)

### Real Enterprise LLM

Single pass per token step:
1. Token IDs -> embedding vectors.
2. Positional encoding/rotation (for example RoPE).
3. Multi-head self-attention (+ causal mask, optional KV reuse).
4. FFN/MLP.
5. Residual + normalization.
6. LM head projection -> logits.

### Current NeuronLM

Single pass is simplified:
1. Build prompt text from messages.
2. Count tokens using tokenizer.
3. Produce response text via deterministic template logic.
4. Truncate by `max_tokens` and apply `stop` strings.

### Code Mapping

- Prompt assembly:
  - `src/neuronlm/services/inference_engine.py` (`_build_prompt`)
- Token count/truncate:
  - `src/neuronlm/core/tokenizer.py`
- Simulated generation:
  - `src/neuronlm/services/inference_engine.py` (`_generate_response_text`)

### Gap Summary

- Present: prompt processing and token accounting.
- Missing: attention, FFN, logits, and true `forward()` model layers.

---

## Layer C - INFERENCE LOOP (Token Generation)

### Real Enterprise LLM

Iterative decode loop:
1. Run forward for current context (or with KV cache).
2. Get logits for next token.
3. Apply decoding policy (greedy/top-k/top-p/beam).
4. Select token, append to sequence.
5. Stop on EOS/max tokens/stop conditions.
6. Stream token deltas to client.

### Current NeuronLM

Streaming loop is simulated:
1. Generate full response text first.
2. Split text by words.
3. Emit SSE chunks word by word.
4. Emit final finish chunk and `[DONE]`.

### Code Mapping

- Stream generator:
  - `src/neuronlm/services/inference_engine.py` (`stream_complete`)
- Stream transport:
  - `src/neuronlm/api/routes.py` (`StreamingResponse` for `/v1/chat/completions`)

### Gap Summary

- Present: streaming protocol and chunk schema.
- Missing: logits-based decoding and token-level generation loop.

---

## Layer D - System Architecture (Serving)

### Real Enterprise LLM

- API gateway + auth + policy checks.
- Request scheduler and dynamic batching.
- GPU worker pool (vLLM/TensorRT-LLM/custom runtime).
- Observability, retries, admission control.

### Current NeuronLM

- API, middleware, and request handling are implemented.
- In-process singleton inference engine.
- No scheduler or GPU worker execution path.

### Code Mapping

- App bootstrap:
  - `src/neuronlm/main.py`
- Middleware (auth/rate-limit/logging):
  - `src/neuronlm/api/middleware.py`
- Request handlers:
  - `src/neuronlm/api/routes.py`

### Gap Summary

- Present: clean service interface and middleware pipeline.
- Missing: enterprise orchestration for throughput and multi-GPU execution.

---

## Layer E - Optimization Layer

### Real Enterprise LLM

- KV cache lifecycle and eviction policies.
- Continuous/dynamic batching.
- Tensor/pipeline/expert parallelism.
- Precision and memory strategy (fp16/bf16/int8/int4).

### Current NeuronLM

- Basic token limits and usage tracking.
- No KV cache.
- No dynamic batching.
- No parallel inference strategy.

### Code Mapping

- Token handling and limits:
  - `src/neuronlm/core/tokenizer.py`
  - `src/neuronlm/services/inference_engine.py`

### Gap Summary

- Present: baseline controls for API-level behavior.
- Missing: all major LLM performance optimizations.

---

## Quick Matrix: Real vs Current

| Area | Real Enterprise LLM | Current NeuronLM |
|---|---|---|
| Model weights | Trained checkpoints | Not loaded |
| Forward pass | True transformer ops | Simulated text generation |
| Decoding | Logits + sampling/beam | Word-split from prebuilt text |
| Streaming | Token-level deltas | Word-level SSE deltas |
| KV cache | Yes | No |
| Batching | Dynamic/continuous | No |
| GPU workers | Yes | No |
| Parallelism | Tensor/pipeline/expert | No |
| Embeddings | Learned model vectors | Deterministic pseudo-vectors |

---

## Upgrade Path (Practical)

1. Add backend adapter interface in `InferenceEngine`:
   - `simulated`, `transformers`, `vllm` modes.
2. Replace `_generate_response_text` with runtime call:
   - real token-by-token decode from model logits.
3. Implement true embeddings for `neuronlm-embed`:
   - map to a sentence-transformer or hosted embedding model.
4. Add KV cache + batching abstraction:
   - request scheduler and decode loop state.
5. Keep API contracts stable:
   - no client changes required for `/v1/chat/completions` and `/v1/embeddings`.

---

## Mental Model

- Current NeuronLM is an enterprise-style service skeleton with simulated model behavior.
- API layer is production-shaped; model internals are not yet production-grade.
- This is suitable for interface development, testing, and integration readiness.
- To become a real enterprise LLM stack, the backend must be replaced with trained model runtimes and optimization primitives.
