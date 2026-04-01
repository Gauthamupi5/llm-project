# Mapping: GPT Training Flow ↔ Transformer Architecture

## Purpose

Provide a concise, example-driven mapping between the procedural GPT-style training loop (Raw Data → Tokenization → Forward → Loss → Backprop → Update → Repeat) and the Transformer Decoder components described in `docs/02_architecture_design.md` (Section 2 & Section 3). Includes a small worked example and a runnable-style pseudocode training loop that highlights exactly where Transformer components are invoked.

---

## 1. Short Mapping Table

| Training Step | Transformer Component(s) | Where in doc(s) |
|---------------|--------------------------|-----------------|
| Raw data (text, code, docs) | Tokenizer (BPE / SPM) → Token IDs | `docs/02_architecture_design.md` Section 2 |
| Tokenization → Input tensors | Token Embedding + Position Embeddings (RoPE) | Section 3 |
| Forward pass (prediction) | Transformer decoder blocks (RMSNorm → GQA Attention → SwiGLU FFN) → Output head (linear → logits) | Section 3 |
| Compute loss | Cross-entropy between logits and target token IDs | Section 2 |
| Backpropagate loss | Gradients through Output head, FFN, Attention (W_Q,W_K,W_V,W_O,W_up,W_down), Embeddings | Section 3 |
| Optimizer step (AdamW) | Update all trainable parameters above | Section 2 (training loop) |
| Checkpoint / Logging / LR schedule | Checkpointing, metrics, LR scheduler (cosine + warmup) | Section 2 |

---

## 2. Compact Pseudocode Training Loop (annotated)

This pseudocode is intentionally small and sequential for readability — real training uses distributed strategies (FSDP, tensor/pipeline parallelism), gradient accumulation and mixed precision.

```python
# Pseudocode: single-step view (micro-batch)
# Shapes: batch_size=B, seq_len=L, d_model=D, vocab=V

for batch in dataloader:                  # Raw data → tokenized batches
    token_ids = tokenizer(batch)          # (B, L)  -- maps raw text → token IDs

    # 1) Embedding + Positional encodings
    x = token_embedding(token_ids)        # (B, L, D)
    x = apply_rope(x)                    # RoPE on token embeddings

    # 2) Forward through N decoder layers
    for layer in transformer_layers:
        x_norm = rmsnorm(x)

        # Grouped Multi-Query Attention (GQA)
        Q = x_norm @ W_Q                 # (B, L, n_heads, d_head)
        K = x_norm @ W_K                 # (B, L, n_kv_heads, d_head)
        V = x_norm @ W_V                 # (B, L, n_kv_heads, d_head)
        attn_out = causal_attention(Q, K, V)  # masks applied
        attn_proj = attn_out @ W_O
        x = x + attn_proj

        x_norm = rmsnorm(x)
        gate = x_norm @ W_gate
        up   = x_norm @ W_up
        ffn_out = siLU(gate) * up @ W_down
        x = x + ffn_out

    # 3) Output head → logits
    x_final = rmsnorm(x)
    logits = x_final @ W_out               # (B, L, V)

    # 4) Prediction & Loss
    # Typically next-token prediction: targets are token_ids shifted left
    targets = token_ids[:, 1:]
    pred_logits = logits[:, :-1, :]
    loss = cross_entropy(pred_logits, targets)

    # 5) Backprop
    loss.backward()                        # gradients flow through W_out, W_* etc.

    # 6) Optimizer step
    optimizer.step()                       # AdamW updates all params
    optimizer.zero_grad()

    # 7) Logging/checkpointing
    log_step_metrics(loss, lr_schedule.step())
```

Notes:
- `token_embedding`, `W_Q`, `W_K`, etc., correspond to the parameter matrices described in Section 3.
- `causal_attention` applies a causal mask so token t cannot attend to future tokens > t.
- In practice: gradient accumulation, mixed precision (BF16/FP16), and distributed sharding are used.

---

## 3. Concrete Mini Example (numbers)

Assume:
- `vocab_size = 10_000`, `d_model = 512`, `L = 8`, `B = 2` (tiny toy example), `n_heads = 8`, `d_head = 64`.

1) After tokenization: `token_ids` shape = `(2, 8)`.
2) Token embeddings: `x` shape = `(2, 8, 512)`.
3) For one attention head: Q,K,V per head have shape `(2, 8, 64)`; concatenated across 8 heads makes `(2, 8, 512)`.
4) `logits` computed at output head: `(2, 8, 10000)` → softmax → probabilities.
5) Loss (cross-entropy) computed on `pred_logits[:, :-1, :]` vs `targets[:, 1:]`.
6) Backprop computes gradients such as `dW_out` of shape `(512, 10000)` and `dW_Q` of shape `(512, 512)` (depending on implementation ordering).
7) Optimizer applies parameter updates; e.g.,

   W_out_new = W_out - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * W_out

(where `m_hat`, `v_hat` are Adam moment estimates).

This concrete example helps with sanity-checking tensor shapes, memory usage, and where per-layer gradients are produced.

---

## 4. Example: Where a Bug in Training Shows Up in Model Components

- Symptom: Loss does not decrease (stalls).
  - Possible causes:
    - LR too high/low (training loop / scheduler)
    - Bug in tokenization (bad token IDs or misalignment) — check `tokenizer()` and `token_embedding`
    - Broken causal mask (future tokens visible) — check `causal_attention()`
    - Incorrect parameter initialization or accidental weight freezing — inspect `requires_grad`

- Symptom: Model repeats tokens (degenerate outputs).
  - Possible causes:
    - Sampling/decoding policy misconfiguration (temperature too low, top-k/top-p wrong)
    - Training data quality (over-represented sequences) — data pipeline issue

Mapping symptoms to the components above accelerates debugging.

---

## 5. Cross-References

- Training lifecycle: see `docs/02_architecture_design.md` Section 2 (LLM Training Flow).
- Transformer details: see `docs/02_architecture_design.md` Section 3 (Transformer Architecture).

---

## 6. Next steps (suggested)

- Convert pseudocode into a minimal runnable training harness that trains a tiny Transformer on synthetic data (I can create this in `examples/` if you want).
- Add a small visualization showing data flow and gradient flow (SVG/mermaid).



