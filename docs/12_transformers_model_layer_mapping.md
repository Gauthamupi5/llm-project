# 12 - Transformers Model Layer Mapping (distilgpt2 / gpt2 / gpt2-medium)

Date: 2026-04-01

Purpose:
This document answers: when NeuronLM uses `distilgpt2`, `gpt2`, or `gpt2-medium`, which layers in the requested enterprise inference stack are mapped, partially mapped, or not mapped.

Reference architecture being checked:
1. Inference System (Core)
2. Inference Loop (token-by-token)
3. One Pass (Embedding -> Transformer Layers -> Logits)
4. Decoding Strategy -> Next Token
5. KV Cache / Batching / Scheduler
6. Model Layer (GPU/TPU, GPT/MoE/Multimodal)

---

## 1) Scope and Assumptions

Current code path for these models:
1. `src/neuronlm/services/inference_engine.py` dispatches to transformers backend.
2. `src/neuronlm/training/transformers_backend.py` loads HF causal LM and calls `.generate(...)`.
3. Supported model names include `distilgpt2`, `gpt2`, `gpt2-medium`.

Important interpretation:
1. "Mapped" here means available in the current serving path.
2. "Partially mapped" means present but abstracted by HF internals or missing enterprise-grade controls.
3. "Not mapped" means absent from this project runtime layer.

---

## 2) Layer Mapping Matrix (Common Across distilgpt2 / gpt2 / gpt2-medium)

| Architecture Layer | distilgpt2 | gpt2 | gpt2-medium | Status | Notes |
|---|---|---|---|---|---|
| Inference System (Core) | Yes | Yes | Yes | Partially mapped | API + backend dispatch exist; runtime still monolithic |
| Inference Loop (token-by-token) | Yes (inside HF generate) | Yes (inside HF generate) | Yes (inside HF generate) | Partially mapped | Loop exists but not explicitly controlled in NeuronLM engine |
| One Pass: Embedding | Yes | Yes | Yes | Mapped (model-internal) | Implemented by model forward pass, not explicit project module |
| One Pass: Transformer Layers | Yes | Yes | Yes | Mapped (model-internal) | Executed inside HF model |
| One Pass: Logits output | Yes | Yes | Yes | Mapped (model-internal) | Used internally by generate/decoding |
| Decoding Strategy (temperature/top_p) | Yes | Yes | Yes | Partially mapped | Basic sampling controls available |
| Next Token selection | Yes | Yes | Yes | Partially mapped | Controlled via HF generation utilities |
| KV Cache lifecycle interface | Limited | Limited | Limited | Not mapped (enterprise view) | No explicit cache API/policy in project |
| Dynamic/continuous batching | No | No | No | Not mapped | No scheduler/worker batching loop |
| Scheduler/admission control | No | No | No | Not mapped | No queue policy, priority, backpressure interfaces |
| GPU/TPU runtime orchestration | Optional local only | Optional local only | Optional local only | Not mapped (enterprise view) | No production worker pool/runtime manager |
| Model Type: GPT | Yes | Yes | Yes | Mapped | All three are GPT-style causal LMs |
| Model Type: MoE | No | No | No | Not mapped | Not in these model families |
| Model Type: Multimodal | No | No | No | Not mapped | Text-only models |

---

## 3) Mapped vs Not Mapped (Direct Answer)

### Fully/Functionally Mapped in Current Setup
1. GPT model loading and text generation via HuggingFace causal LMs.
2. One-pass internals (embedding, transformer blocks, logits) inside each model.
3. Basic decoding controls (`temperature`, `top_p`, max tokens, seed).

### Partially Mapped
1. Inference loop is present but hidden inside `.generate(...)` rather than explicit NeuronLM token loop.
2. Decoding and token event control are limited to high-level API knobs.
3. Inference core works, but not yet separated into scheduler-worker architecture.

### Not Mapped (Enterprise Layers Missing)
1. Explicit KV cache lifecycle management interface.
2. Continuous batching and queue scheduler.
3. Admission control/backpressure and worker orchestration.
4. MoE and multimodal inference model types.
5. Production-grade GPU/TPU serving fabric.

---

## 4) Per-Model Notes

### distilgpt2
1. Smallest among listed options, fastest for development CPU usage.
2. Same interface layer mapping as other two models.
3. Lower output quality ceiling vs larger GPT variants.

### gpt2
1. Better quality than distilgpt2 in many prompts.
2. Same mapped/not-mapped layer profile.
3. Higher memory and latency requirements.

### gpt2-medium
1. Highest quality of the three in general.
2. Same mapped/not-mapped layer profile.
3. Significantly heavier memory and startup cost.

Conclusion:
1. Switching among these three changes capability/quality and compute cost.
2. It does not change enterprise layer coverage in this codebase.

---

## 5) Gap Impact by Missing Layer

| Missing Layer | Practical Effect Today |
|---|---|
| KV cache interface | Higher latency/cost for long or repeated contexts |
| Dynamic batching | Lower throughput under concurrent traffic |
| Scheduler/admission control | Less predictable latency under load spikes |
| Worker orchestration | Harder to scale horizontally and isolate failures |
| MoE/multimodal support | Cannot serve those model classes through same platform path |

---

## 6) Minimal Next Steps to Improve Layer Coverage

1. Add an explicit scheduler-worker boundary in the inference service.
2. Add token-event streaming abstraction independent of backend internals.
3. Add cache/session policy abstraction (reuse/evict/ttl) even before full KV implementation.
4. Add concurrency-aware micro-batching for queued requests.
5. Keep API contract stable while evolving runtime internals.

---

## 7) Final Yes/No Answer to "Do you have all these layers?"

No, not all layers are fully present in enterprise form.

You currently have:
1. GPT model execution and one-pass internals (via HF models).
2. Basic inference and decoding path.

You do not yet have:
1. Enterprise KV cache lifecycle.
2. Dynamic batching/scheduler stack.
3. Full distributed production runtime controls.
