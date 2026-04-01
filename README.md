# NeuronLM – Enterprise LLM Platform

An enterprise-grade Large Language Model platform providing scalable inference, RAG, fine-tuning, and multi-tenant API services.

## Project Structure

```
neuronlm/
├── docs/                          # Architecture & design documents
├── src/
│   └── neuronlm/
│       ├── api/                   # FastAPI routes and middleware
│       ├── core/                  # Core business logic
│       ├── models/                # Pydantic data models
│       ├── services/              # Service layer
│       ├── db/                    # Database access layer
│       └── utils/                 # Utilities
├── tests/
│   ├── unit/                      # Unit tests
│   ├── component/                 # Component/API tests
│   ├── integration/               # Integration tests
│   └── conftest.py                # Shared fixtures
├── config/                        # Configuration files
├── docker/                        # Dockerfiles
├── pyproject.toml                 # Project metadata & dependencies
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Set environment variables
cp config/.env.example config/.env

# Run the server
uvicorn neuronlm.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=neuronlm
```

## API Usage

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nlm-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuronlm-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nlm-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuronlm-7b",
    "messages": [{"role": "user", "content": "Explain AI"}],
    "stream": true
  }'
```

## Documentation

- [Architecture Definition](docs/01_architecture_definition.md)
- [Architecture Design](docs/02_architecture_design.md)
- [Requirements](docs/03_requirements.md)
- [Specification](docs/04_specification.md)
- [Testing Strategy](docs/05_testing_strategy.md)
- [Test Cases](docs/06_test_cases.md)
