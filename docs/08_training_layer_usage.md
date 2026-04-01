# 08 - Training Layer Usage (Bigram Backend)

This project now includes a lightweight training layer that can produce
trained checkpoints and serve responses through the existing API interface.

## What Was Implemented

- Trainable backend:
  - `src/neuronlm/training/bigram_lm.py`
  - `src/neuronlm/training/trainer.py`
  - `src/neuronlm/training/cli.py`
- Inference integration:
  - `src/neuronlm/services/inference_engine.py`
- Config switches:
  - `src/neuronlm/config.py`

## Train a Checkpoint

1. Create a corpus file (one sentence per line), for example `data/corpus.txt`.
2. Run training CLI.

```bash
python -m neuronlm.training.cli --corpus data/corpus.txt --checkpoint artifacts/bigram_model.npz
```

## Serve Using Trained Checkpoint

Set environment variables before starting the API:

```bash
set NEURONLM_INFERENCE_BACKEND=bigram
set NEURONLM_BIGRAM_CHECKPOINT_PATH=artifacts/bigram_model.npz
python -m uvicorn neuronlm.main:app --host 0.0.0.0 --port 8000 --reload
```

## Call Interface After Training

Use your normal API contract (`/v1/chat/completions`).

```bash
curl -X POST http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <TOKEN>" ^
  -d "{\"model\":\"neuronlm-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about data\"}],\"max_tokens\":64,\"seed\":42}"
```

If backend is configured correctly, generation will come from the trained bigram
checkpoint rather than the simulated template generator.

## Validation

Run targeted tests:

```bash
python -m pytest -q tests/unit/test_training_layer.py tests/unit/test_inference_engine.py
```

Expected result: all tests pass.
