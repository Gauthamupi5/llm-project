# LLM System Technical Specification

## Document Control

| Field | Value |
|-------|-------|
| **Project** | NeuronLM – Enterprise LLM Platform |
| **Version** | 1.0.0 |
| **Status** | Approved |

---

## 1. API Specification

### 1.1 Base URL & Versioning
```
Production:  https://api.neuronlm.example.com/v1
Staging:     https://api-staging.neuronlm.example.com/v1
```

### 1.2 Authentication
All requests must include one of:
- `Authorization: Bearer <api_key>` header
- `Authorization: Bearer <jwt_token>` header

API keys follow the format: `nlm-<32 random alphanumeric characters>`
Example: `nlm-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

### 1.3 Chat Completions Endpoint

#### Request

```
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
    "model": "string (required) — Model ID from the registry",
    "messages": [
        {
            "role": "string (required) — one of: system, user, assistant, tool",
            "content": "string (required) — message content",
            "name": "string (optional) — participant name"
        }
    ],
    "temperature": "float (optional, default: 0.7) — range [0.0, 2.0]",
    "top_p": "float (optional, default: 1.0) — range [0.0, 1.0]",
    "max_tokens": "integer (optional, default: 2048) — range [1, 32768]",
    "stream": "boolean (optional, default: false)",
    "stop": "string|array (optional) — up to 4 stop sequences",
    "frequency_penalty": "float (optional, default: 0.0) — range [-2.0, 2.0]",
    "presence_penalty": "float (optional, default: 0.0) — range [-2.0, 2.0]",
    "user": "string (optional) — end-user identifier for abuse monitoring"
}
```

#### Response (Non-Streaming)

```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1711929600,
    "model": "neuronlm-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response text here..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 150,
        "total_tokens": 200
    }
}
```

#### Response (Streaming — SSE)

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711929600,"model":"neuronlm-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711929600,"model":"neuronlm-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711929600,"model":"neuronlm-7b","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711929600,"model":"neuronlm-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 1.4 Embeddings Endpoint

```
POST /v1/embeddings
```

```json
{
    "model": "neuronlm-embed",
    "input": "string | array<string>",
    "encoding_format": "float | base64 (default: float)"
}
```

Response:
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.0023, -0.0091, ...],
            "index": 0
        }
    ],
    "model": "neuronlm-embed",
    "usage": {
        "prompt_tokens": 8,
        "total_tokens": 8
    }
}
```

### 1.5 Models Endpoint

```
GET /v1/models
```

Response:
```json
{
    "object": "list",
    "data": [
        {
            "id": "neuronlm-7b",
            "object": "model",
            "created": 1711929600,
            "owned_by": "neuronlm",
            "context_window": 32768,
            "capabilities": ["chat", "completion", "embedding"]
        }
    ]
}
```

### 1.6 Error Responses

```json
{
    "error": {
        "message": "Human-readable error description",
        "type": "invalid_request_error | authentication_error | rate_limit_error | server_error",
        "code": "specific_error_code",
        "param": "field_name (if applicable)"
    }
}
```

| HTTP Code | Error Type | Description |
|-----------|-----------|-------------|
| 400 | `invalid_request_error` | Malformed request, invalid parameters |
| 401 | `authentication_error` | Invalid or missing API key |
| 403 | `permission_error` | Insufficient permissions |
| 404 | `not_found_error` | Model or resource not found |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `server_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

---

## 2. Data Models

### 2.1 Core Domain Models

```python
# User
class User:
    id: UUID
    email: str                    # unique, validated
    name: str
    organization_id: UUID
    role: Enum[admin, developer, viewer]
    is_active: bool
    created_at: datetime
    updated_at: datetime

# Conversation
class Conversation:
    id: UUID
    user_id: UUID
    title: str                    # max 500 chars
    model_id: str
    system_prompt: Optional[str]  # max 4096 tokens
    metadata: dict
    message_count: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime

# Message
class Message:
    id: UUID
    conversation_id: UUID
    role: Enum[system, user, assistant, tool]
    content: str
    token_count: int
    model_id: Optional[str]
    latency_ms: Optional[int]
    metadata: dict
    created_at: datetime

# Model
class Model:
    id: str                       # e.g., "neuronlm-7b"
    name: str
    architecture: str
    parameter_count: int
    quantization: Optional[str]
    max_context_length: int
    status: Enum[loading, ready, deprecated]
    serving_config: ServingConfig
    created_at: datetime

# API Key
class ApiKey:
    id: UUID
    user_id: UUID
    key_hash: str                 # SHA-256
    key_prefix: str               # first 8 chars
    name: str
    scopes: List[str]
    rate_limit_rpm: int           # requests per minute
    token_quota_daily: Optional[int]
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime
```

### 2.2 Configuration Models

```python
class ServingConfig:
    gpu_memory_gb: int
    tensor_parallel_size: int
    max_batch_size: int
    max_num_seqs: int
    dtype: str                    # "float16", "bfloat16", "int4"
    enforce_eager: bool

class GenerationConfig:
    temperature: float            # [0.0, 2.0]
    top_p: float                  # [0.0, 1.0]
    top_k: int                    # [1, 100]
    max_tokens: int               # [1, 32768]
    stop_sequences: List[str]     # max 4
    frequency_penalty: float      # [-2.0, 2.0]
    presence_penalty: float       # [-2.0, 2.0]
    repetition_penalty: float     # [1.0, 2.0]
    seed: Optional[int]

class RateLimitConfig:
    requests_per_minute: int
    tokens_per_minute: int
    tokens_per_day: int
    concurrent_requests: int
```

---

## 3. Service Specifications

### 3.1 Inference Service

| Property | Specification |
|----------|--------------|
| **Protocol** | HTTP/2 + gRPC |
| **Port** | 8000 (HTTP), 8001 (gRPC) |
| **Health Check** | `GET /health` returns 200 with model status |
| **Readiness** | `GET /ready` returns 200 when model is loaded |
| **Startup Probe** | Allow 300s for model loading |
| **Resource Limits** | 1 GPU (A100 80GB), 32GB RAM, 8 CPU cores |
| **Scaling** | HPA on GPU utilization and queue depth |
| **Timeout** | 120s request timeout, 30s idle stream timeout |

### 3.2 Conversation Service

| Property | Specification |
|----------|--------------|
| **Protocol** | HTTP/1.1 + HTTP/2 |
| **Port** | 8010 |
| **Database** | PostgreSQL (read/write split) |
| **Cache** | Redis (conversation metadata, 1h TTL) |
| **Resource Limits** | 2GB RAM, 2 CPU cores |
| **Scaling** | HPA on CPU utilization |

### 3.3 RAG Service

| Property | Specification |
|----------|--------------|
| **Protocol** | HTTP/2 |
| **Port** | 8020 |
| **Vector DB** | Qdrant (HNSW index, cosine similarity) |
| **Chunk Size** | 512 tokens with 50-token overlap |
| **Embedding Model** | all-MiniLM-L6-v2 (384 dimensions) |
| **Top-K** | Default 5, max 20 |
| **Resource Limits** | 4GB RAM, 4 CPU cores |

---

## 4. Message Queue Specifications

### 4.1 Kafka Topics

| Topic | Partitions | Retention | Schema |
|-------|-----------|-----------|--------|
| `inference.requests` | 12 | 24h | InferenceRequest (Avro) |
| `inference.responses` | 12 | 24h | InferenceResponse (Avro) |
| `usage.events` | 6 | 7d | UsageEvent (Avro) |
| `audit.logs` | 6 | 90d | AuditEvent (Avro) |
| `model.events` | 3 | 30d | ModelEvent (Avro) |
| `finetuning.jobs` | 3 | 7d | FineTuningJob (Avro) |

---

## 5. Caching Specification

### 5.1 Cache Layers

| Layer | Technology | TTL | Use Case |
|-------|-----------|-----|----------|
| L1 | In-process (LRU) | 5 min | Hot conversation metadata |
| L2 | Redis | 1 hour | Session data, rate limit counters |
| L3 | Semantic Cache | 24 hours | Similar query results (embedding similarity > 0.95) |

### 5.2 Cache Key Format
```
neuronlm:{service}:{entity}:{id}:{version}
```
Examples:
- `neuronlm:conv:metadata:uuid-123:v1`
- `neuronlm:model:config:neuronlm-7b:v2`
- `neuronlm:ratelimit:apikey:prefix-abc:minute`

---

## 6. Tokenizer Specification

| Property | Value |
|----------|-------|
| Type | BPE (Byte-Pair Encoding) |
| Vocabulary Size | 32,000 tokens |
| Special Tokens | `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|sep|>` |
| Chat Template | `<|bos|><|system|>\n{system}<|sep|>\n<|user|>\n{user}<|sep|>\n<|assistant|>\n` |
| Max Input Length | 32,768 tokens |
| Encoding | UTF-8 |

---

## 7. Infrastructure Specifications

### 7.1 Compute Resources

| Component | Instance Type | Count | Purpose |
|-----------|--------------|-------|---------|
| Inference GPU | A100 80GB | 2-20 (auto) | Model serving |
| Training GPU | A100 80GB x8 | 1-4 | Fine-tuning |
| API Services | c6i.2xlarge | 3-10 (auto) | API, conversation, RAG |
| Database | r6g.2xlarge | 3 (HA) | PostgreSQL |
| Cache | r6g.xlarge | 6 | Redis Cluster |
| Vector DB | r6g.2xlarge | 3 | Qdrant |
| Message Queue | m6i.xlarge | 3 | Kafka |

### 7.2 Storage

| Type | Size | IOPS | Use Case |
|------|------|------|----------|
| EBS gp3 | 500GB | 16,000 | Database |
| EBS io2 | 1TB | 64,000 | Model weights (fast loading) |
| S3 Standard | 10TB | N/A | Model archive, training data |
| S3 Glacier | Unlimited | N/A | Audit logs, old backups |

### 7.3 Networking

| Specification | Value |
|--------------|-------|
| VPC CIDR | 10.0.0.0/16 |
| Public Subnets | 10.0.1.0/24, 10.0.2.0/24 |
| Private Subnets | 10.0.10.0/24, 10.0.20.0/24 |
| GPU Subnet | 10.0.100.0/24 |
| Inter-service | mTLS via service mesh |
| External | TLS 1.3 via ALB |
