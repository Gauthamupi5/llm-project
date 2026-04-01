# 09 - How To Train, Flow Comparison, and Out-of-Training Behavior

This guide explains:
1. How to train the current NeuronLM training layer.
2. How training flow compares to interface flow.
3. How both compare with a real enterprise LLM system.
4. What happens when interface prompts are outside training data.

---

## A. Step-by-Step: How To Train (Current Project)

### Prerequisites

- Python environment is set up.
- Project dependencies are installed.
- Training corpus exists in `data/corpus.txt`.

### Step 1: Prepare training data

- Add clean text lines to `data/corpus.txt`.
- Use one sentence or one QA pair per line.
- Prefer domain-focused examples if you want domain behavior.

Example line format:
- `Question: What is KV cache? Answer: KV cache stores attention keys/values from prior tokens to speed decoding.`

### Step 2: Run training

```bash
python -m neuronlm.training.cli --corpus data/corpus.txt --checkpoint artifacts/bigram_model.npz
```

What this does:
1. Reads corpus text lines.
2. Tokenizes data using project tokenizer.
3. Builds bigram transition probabilities.
4. Saves checkpoint to `artifacts/bigram_model.npz`.

### Step 3: Start API with trained backend

```bash
set NEURONLM_INFERENCE_BACKEND=bigram
set NEURONLM_BIGRAM_CHECKPOINT_PATH=artifacts/bigram_model.npz
python -m uvicorn neuronlm.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Call interface after training

Generate a JWT token first if you do not already have one:

```bash
python -c "from neuronlm.core.auth import get_auth_service; print(get_auth_service().create_jwt_token('demo-user'))"
```

Use PowerShell like this:

```powershell
$TOKEN = (python -c "from neuronlm.core.auth import get_auth_service; print(get_auth_service().create_jwt_token('demo-user'))").Trim()

$body = @{
  model = "neuronlm-7b"
  messages = @(
    @{
      role = "user"
      content = "Explain KV cache"
    }
  )
  max_tokens = 64
  seed = 42
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/v1/chat/completions" `
  -Headers @{ Authorization = "Bearer $TOKEN" } `
  -ContentType "application/json" `
  -Body $body
```

If you want to use `curl` in PowerShell, call `curl.exe` explicitly:

```bash
curl.exe -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer <TOKEN>" -d "{\"model\":\"neuronlm-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain KV cache\"}],\"max_tokens\":64,\"seed\":42}"
```

If the checkpoint is loaded correctly, output is generated from trained bigram transitions.

Common reason the old command failed on Windows:
- `^` is for `cmd.exe`, not PowerShell.
- In PowerShell, `curl` is often an alias for `Invoke-WebRequest`, not real curl.
- The token from `python -c` may include a trailing newline, so use `.Trim()`.

---

## B. Training Flow vs Interface Flow (Current Project)

### Training Flow (Current)

1. Corpus text -> tokenizer IDs.
2. Count token-to-next-token transitions.
3. Normalize to probabilities.
4. Save checkpoint (`.npz`).

In code:
- `src/neuronlm/training/cli.py`
- `src/neuronlm/training/trainer.py`
- `src/neuronlm/training/bigram_lm.py`

### Interface Flow (Current)

1. Client request -> FastAPI endpoint.
2. Middleware (auth, rate limit, logging).
3. Inference engine selects backend (`simulated` or `bigram`).
4. Generate response (stream or non-stream).
5. Return JSON/SSE response.

In code:
- `src/neuronlm/main.py`
- `src/neuronlm/api/middleware.py`
- `src/neuronlm/api/routes.py`
- `src/neuronlm/services/inference_engine.py`

### Key Difference

- Training flow creates model behavior (probability table/checkpoint).
- Interface flow serves requests using whichever behavior is loaded.

---

## C. Comparison with Enterprise LLM

## 1) Training Flow Comparison

Current project:
- Bigram statistical training.
- CPU-friendly.
- No backpropagation, optimizer, gradients, or deep layers.

Enterprise LLM:
- Transformer training with forward + loss + backprop + optimizer step.
- Distributed GPU training (data/tensor/pipeline parallelism).
- Checkpoint orchestration, mixed precision, data curriculum.

## 2) Interface Flow Comparison

Current project:
- API layer is enterprise-shaped (auth, middleware, streaming).
- Backend can be simulated or trained bigram.
- No GPU scheduler or dynamic batching.

Enterprise LLM:
- API gateway + request queue/scheduler.
- Runtime backend (vLLM/TensorRT-LLM/optimized Transformers).
- KV cache, continuous batching, multi-GPU worker pools.

## 3) Decoding Comparison

Current project (bigram mode):
- Next-token sampled from bigram transition probabilities.

Enterprise LLM:
- Next-token from neural logits with top-k/top-p/temperature/beam and advanced penalties.

---

## D. What Happens If Prompt Is Outside Training Data?

### Current project (bigram backend)

You still get a response, but quality may degrade:
- More generic or repetitive output.
- Lower factual accuracy.
- Weak long-range coherence.
- May mirror local token patterns from corpus.

Why:
- Bigram model only uses local token transition statistics.
- It does not learn deep semantics like transformer models.

### Current project (simulated backend)

- You still get a deterministic template response.
- Output is not learned from data.

### Enterprise LLM (well-trained)

- Often provides better generalization for unseen prompts.
- Still can hallucinate if domain is missing or ambiguous.

---

## E. Practical Data Guidance

To improve outputs in this project:
1. Add more domain-specific lines to `data/corpus.txt`.
2. Keep style consistent with expected responses.
3. Include Q/A examples matching your interface use-cases.
4. Retrain checkpoint after corpus updates.
5. Evaluate with a fixed prompt set before deployment.

---

## F. Quick Verification Checklist

1. Train command succeeded and checkpoint exists.
2. API started with `NEURONLM_INFERENCE_BACKEND=bigram`.
3. `/v1/chat/completions` returns non-empty responses.
4. Changing corpus + retraining changes output patterns.
5. Fallback still works if checkpoint is unavailable.

---

## G. Mental Model

- Training flow builds behavior.
- Interface flow serves behavior.
- If training data is weak, interface still works but output quality drops.
- Enterprise quality requires both strong training and strong serving infrastructure.

---

## H. Training + Interface Gaps vs Enterprise LLM (Table)

| Area | Current NeuronLM | Enterprise LLM | Gap Impact | Recommended Next Step |
|---|---|---|---|---|
| Training objective | Bigram transition statistics | Transformer loss optimization (cross-entropy, RLHF variants) | High (quality ceiling) | Add transformer fine-tuning backend (LoRA/PEFT) |
| Model architecture | No deep neural layers in training path | Multi-layer transformer (attention + FFN + norms) | High | Integrate real checkpoint inference/training runtime |
| Backprop + optimizer | Not present | Full gradient updates with optimizer/scheduler | High | Add training pipeline using `transformers` + `accelerate` |
| Data scale handling | Small local corpus text file | Large curated datasets, data versioning, filtering | High | Introduce dataset pipeline and quality gates |
| Generalization outside training data | Limited, local token-pattern behavior | Better semantic generalization (still imperfect) | High | Expand diverse corpus and move to transformer model |
| Decoding quality | Bigram next-token sampling | Logits-based decoding (top-k/top-p/beam, penalties) | High | Replace decode path with neural logits decoding |
| Interface contract | Good API shape (chat/embedding/stream) | OpenAI-compatible or internal enterprise contracts | Low | Keep interface stable; evolve backend only |
| Streaming | SSE word-level chunks | Token-level streaming from decoder loop | Medium | Stream model tokens directly from decoding loop |
| KV cache | Not present | Standard for low-latency autoregressive decoding | High (latency/cost) | Add KV cache lifecycle in runtime layer |
| Batching | Not present | Continuous/dynamic batching | High (throughput) | Add scheduler + dynamic batcher |
| GPU workers | Not present | Multi-GPU worker pool and routing | High (scale) | Add vLLM/TensorRT-LLM serving workers |
| Parallelism | Not present | Tensor/pipeline/expert parallelism | Medium to High | Add distributed runtime strategy |
| Embeddings backend | Deterministic pseudo-vectors | Learned embedding models | Medium | Plug real embedding model (sentence-transformers or hosted) |
| Observability depth | Basic service-level visibility | End-to-end tracing, token metrics, GPU telemetry | Medium | Add per-request/token metrics and runtime tracing |
| Reliability controls | Basic fallback behavior | Retries, admission control, circuit breaking, canary rollout | Medium | Add production SRE controls for serving path |
