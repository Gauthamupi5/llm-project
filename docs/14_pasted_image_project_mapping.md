# 14 - Pasted Transformer Image Mapping to Current Project

Date: 2026-04-01

Purpose:
This document maps the pasted Transformer architecture image to the current NeuronLM project, with focus on distilgpt2, gpt2, and gpt2-medium usage.

---

## 1) What the Pasted Image Represents

The image is the classic Transformer diagram with:
1. Left stack: Encoder blocks (self-attention + feed-forward + add/norm), repeated Nx.
2. Right stack: Decoder blocks (masked self-attention + cross-attention + feed-forward + add/norm), repeated Nx.
3. Top head: Linear + Softmax for token probabilities.

The red highlighted region corresponds to an Encoder block.

---

## 2) Important Model-Type Clarification

For this project path (distilgpt2/gpt2/gpt2-medium):
1. These are GPT-style decoder-only causal language models.
2. They do not use the encoder stack shown on the left side of the original Transformer image.
3. They use decoder-style transformer layers to produce logits token by token.

So, the red boxed encoder block is conceptually useful for Transformer history, but it is not the runtime block used by GPT-2 family generation in this project.

---

## 3) Mapping Image Blocks to Current Project

| Image Component | Applies to GPT-2 Family in Project | Current Project Mapping | Code Link(s) |
|---|---|---|---|
| Input Embedding + Positional Encoding | Yes | Implemented inside HuggingFace model internals | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59) |
| Encoder Block (red box, left side) | No (for GPT-2 path) | Not used directly in current GPT inference path | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L24) |
| Decoder Transformer Layers (right side concept) | Yes | Implemented inside model forward pass (internal to HF model) | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59) |
| Linear + Softmax output head | Yes | Part of model forward/generation internals | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124) |
| Token-by-token generation loop | Yes | Performed by HF generate utility, invoked from backend | [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124), [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L192) |
| Decoding strategy (temperature, top_p) | Yes | Exposed through API and passed to model generation | [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L87), [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L88), [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L77) |

---

## 4) Project-Level Inference Mapping (with Image Context)

Current runtime flow:
1. Request enters API layer and is validated.
   - [src/neuronlm/api/routes.py](src/neuronlm/api/routes.py#L46)
2. Inference engine routes to transformers backend when configured.
   - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L93)
   - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L94)
3. Backend loads chosen GPT model and calls generation.
   - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L58)
   - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L59)
   - [src/neuronlm/training/transformers_backend.py](src/neuronlm/training/transformers_backend.py#L124)
4. Response returns through completion or streaming interface.
   - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L304)
   - [src/neuronlm/services/inference_engine.py](src/neuronlm/services/inference_engine.py#L360)

---

## 5) Mapped vs Not Mapped Relative to the Image and Enterprise Stack

Mapped now:
1. GPT-model generation path (decoder-only internals).
2. Logits-to-token generation via HF runtime.
3. Basic decoding controls (temperature/top_p/seed).

Not mapped as explicit project subsystems:
1. Encoder stack implementation (not needed for GPT-2 generation path).
2. Explicit, custom one-pass modules outside HF internals.
3. Enterprise runtime layers: KV cache lifecycle API, dynamic batching, scheduler/admission control.

---

## 6) Direct Answer for the Pasted Image

With distilgpt2/gpt2/gpt2-medium in this project:
1. You are using the decoder-side Transformer generation idea from the image.
2. You are not using the encoder block highlighted in red.
3. The generation core is present, but enterprise serving layers are still missing.
