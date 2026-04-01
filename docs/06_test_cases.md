# LLM System Test Cases

## Document Control

| Field | Value |
|-------|-------|
| **Project** | NeuronLM – Enterprise LLM Platform |
| **Version** | 1.0.0 |
| **Status** | Approved |

---

## 1. Unit Test Cases

### 1.1 Request Validation

| TC ID | Test Case | Input | Expected Output |
|-------|-----------|-------|-----------------|
| UT-VAL-001 | Valid chat completion request | Complete valid request body | Request parsed successfully |
| UT-VAL-002 | Missing required field (model) | Request without `model` | ValidationError with field name |
| UT-VAL-003 | Missing required field (messages) | Request without `messages` | ValidationError with field name |
| UT-VAL-004 | Empty messages array | `messages: []` | ValidationError: at least 1 message required |
| UT-VAL-005 | Invalid role in messages | `role: "invalid"` | ValidationError: role must be system/user/assistant/tool |
| UT-VAL-006 | Temperature out of range (high) | `temperature: 3.0` | ValidationError: must be between 0.0 and 2.0 |
| UT-VAL-007 | Temperature out of range (negative) | `temperature: -0.1` | ValidationError: must be >= 0.0 |
| UT-VAL-008 | Temperature at boundary (0.0) | `temperature: 0.0` | Valid — accepted |
| UT-VAL-009 | Temperature at boundary (2.0) | `temperature: 2.0` | Valid — accepted |
| UT-VAL-010 | max_tokens exceeds limit | `max_tokens: 100000` | ValidationError: must be <= 32768 |
| UT-VAL-011 | max_tokens negative | `max_tokens: -1` | ValidationError: must be >= 1 |
| UT-VAL-012 | Invalid model ID | `model: "nonexistent"` | ModelNotFoundError |
| UT-VAL-013 | Too many stop sequences | 5 stop sequences | ValidationError: max 4 stop sequences |
| UT-VAL-014 | Valid stop sequences | 4 stop sequences | Valid — accepted |
| UT-VAL-015 | top_p out of range | `top_p: 1.5` | ValidationError |

### 1.2 Token Operations

| TC ID | Test Case | Input | Expected Output |
|-------|-----------|-------|-----------------|
| UT-TOK-001 | Count tokens in simple text | "Hello world" | 2 tokens |
| UT-TOK-002 | Count tokens in empty string | "" | 0 tokens |
| UT-TOK-003 | Count tokens with special chars | "Hello! @#$%" | Correct token count |
| UT-TOK-004 | Count tokens in multi-language | "Hello 你好 مرحبا" | Correct token count |
| UT-TOK-005 | Truncate messages to fit context | 50K tokens of messages, 32K context | Truncated to fit within 32K |
| UT-TOK-006 | Truncation preserves system prompt | System prompt + long history | System prompt intact |
| UT-TOK-007 | Truncation preserves latest message | Long history + current message | Latest message intact |
| UT-TOK-008 | Token budget allocation | Mixed message types | Correct budget per priority |

### 1.3 Authentication

| TC ID | Test Case | Input | Expected Output |
|-------|-----------|-------|-----------------|
| UT-AUTH-001 | Valid API key format | `nlm-a1b2c3...` (32 chars) | Valid |
| UT-AUTH-002 | Invalid API key prefix | `xxx-a1b2c3...` | Invalid format |
| UT-AUTH-003 | API key too short | `nlm-abc` | Invalid format |
| UT-AUTH-004 | API key hash verification | Known key → known hash | Hash matches |
| UT-AUTH-005 | Expired API key | Key with past expiry date | AuthenticationError |
| UT-AUTH-006 | Deactivated API key | `is_active: false` | AuthenticationError |
| UT-AUTH-007 | Valid JWT token | Valid signed JWT | Claims extracted correctly |
| UT-AUTH-008 | Expired JWT token | JWT with past `exp` | AuthenticationError |
| UT-AUTH-009 | Invalid JWT signature | Tampered JWT | AuthenticationError |
| UT-AUTH-010 | JWT missing required claims | JWT without `sub` | AuthenticationError |

### 1.4 Rate Limiting

| TC ID | Test Case | Input | Expected Output |
|-------|-----------|-------|-----------------|
| UT-RL-001 | Under rate limit | 30 requests (limit: 60/min) | All allowed |
| UT-RL-002 | At rate limit | 60 requests (limit: 60/min) | All allowed |
| UT-RL-003 | Over rate limit | 61 requests (limit: 60/min) | 61st rejected with 429 |
| UT-RL-004 | Rate limit reset | Wait 60s after hitting limit | Requests allowed again |
| UT-RL-005 | Token-based rate limit | 1M tokens (limit: 500K/min) | Rejected after 500K |
| UT-RL-006 | Concurrent request limit | 20 concurrent (limit: 10) | 11th queued or rejected |

### 1.5 Prompt Building

| TC ID | Test Case | Input | Expected Output |
|-------|-----------|-------|-----------------|
| UT-PB-001 | Build with system prompt | System + user message | Correct template format |
| UT-PB-002 | Build without system prompt | User message only | Default system prompt used |
| UT-PB-003 | Build with conversation history | 5-turn conversation | All turns in correct order |
| UT-PB-004 | Build with RAG context | User message + RAG chunks | Context inserted correctly |
| UT-PB-005 | Build respects token budget | Long history, limited budget | Truncated appropriately |

---

## 2. Component Test Cases

### 2.1 Chat Completions API

| TC ID | Test Case | Method | Endpoint | Expected |
|-------|-----------|--------|----------|----------|
| CT-CHAT-001 | Successful chat completion | POST | /v1/chat/completions | 200, valid response schema |
| CT-CHAT-002 | Streaming chat completion | POST | /v1/chat/completions (stream=true) | 200, SSE stream with valid chunks |
| CT-CHAT-003 | Stream ends with [DONE] | POST | /v1/chat/completions (stream=true) | Last event is `data: [DONE]` |
| CT-CHAT-004 | Response includes usage | POST | /v1/chat/completions | usage.prompt_tokens + completion_tokens correct |
| CT-CHAT-005 | Missing auth header | POST | /v1/chat/completions | 401 AuthenticationError |
| CT-CHAT-006 | Invalid auth header | POST | /v1/chat/completions | 401 AuthenticationError |
| CT-CHAT-007 | Invalid request body | POST | /v1/chat/completions | 400 with validation errors |
| CT-CHAT-008 | Model not found | POST | /v1/chat/completions (bad model) | 404 ModelNotFoundError |
| CT-CHAT-009 | Rate limit exceeded | POST | /v1/chat/completions (burst) | 429 RateLimitError |
| CT-CHAT-010 | Server error handling | POST | /v1/chat/completions (mock failure) | 500 with error details |
| CT-CHAT-011 | Concurrent requests | POST | /v1/chat/completions (10 parallel) | All return valid responses |
| CT-CHAT-012 | Large input (max context) | POST | /v1/chat/completions (32K tokens) | 200 or appropriate truncation |
| CT-CHAT-013 | Stop sequence works | POST | /v1/chat/completions (stop=["."]) | Response stops at "." |
| CT-CHAT-014 | Temperature 0 deterministic | POST | /v1/chat/completions (temp=0, seed=42) | Same response for same input |
| CT-CHAT-015 | Content-Type validation | POST | /v1/chat/completions (text/plain) | 415 Unsupported Media Type |

### 2.2 Models API

| TC ID | Test Case | Method | Endpoint | Expected |
|-------|-----------|--------|----------|----------|
| CT-MOD-001 | List all models | GET | /v1/models | 200, array of model objects |
| CT-MOD-002 | Get specific model | GET | /v1/models/neuronlm-7b | 200, model details |
| CT-MOD-003 | Model not found | GET | /v1/models/nonexistent | 404 |
| CT-MOD-004 | Model status is ready | GET | /v1/models/neuronlm-7b | status: "ready" |

### 2.3 Embeddings API

| TC ID | Test Case | Method | Endpoint | Expected |
|-------|-----------|--------|----------|----------|
| CT-EMB-001 | Single text embedding | POST | /v1/embeddings | 200, embedding vector (384 dims) |
| CT-EMB-002 | Batch embeddings | POST | /v1/embeddings (array input) | 200, multiple embedding vectors |
| CT-EMB-003 | Empty input | POST | /v1/embeddings (empty string) | 400 ValidationError |
| CT-EMB-004 | Very long input | POST | /v1/embeddings (>8K tokens) | 400 or truncated with warning |

### 2.4 Conversation Management

| TC ID | Test Case | Expected |
|-------|-----------|----------|
| CT-CONV-001 | Create conversation | 201, conversation object with ID |
| CT-CONV-002 | Get conversation by ID | 200, conversation with message history |
| CT-CONV-003 | List user conversations | 200, paginated list |
| CT-CONV-004 | Delete conversation | 204, conversation removed |
| CT-CONV-005 | Add message to conversation | 201, message stored |
| CT-CONV-006 | Get conversation by wrong user | 404 (not 403, to prevent enumeration) |
| CT-CONV-007 | Create with system prompt | System prompt stored and returned |
| CT-CONV-008 | Conversation title auto-generation | Title generated from first message |

### 2.5 Health & Readiness

| TC ID | Test Case | Endpoint | Expected |
|-------|-----------|----------|----------|
| CT-HEALTH-001 | Health check (all healthy) | GET /health | 200 `{"status": "healthy"}` |
| CT-HEALTH-002 | Health check (DB down) | GET /health | 503 `{"status": "unhealthy", "checks": {...}}` |
| CT-HEALTH-003 | Readiness (model loaded) | GET /ready | 200 |
| CT-HEALTH-004 | Readiness (model loading) | GET /ready | 503 |

---

## 3. Integration Test Cases

### 3.1 End-to-End Chat Flow

| TC ID | Test Case | Steps | Expected |
|-------|-----------|-------|----------|
| IT-CHAT-001 | Complete chat flow | 1. Auth → 2. Create conversation → 3. Send message → 4. Get response → 5. Verify stored | Full round trip works |
| IT-CHAT-002 | Multi-turn conversation | 1. Send 5 messages → 2. Verify context used | Each response considers prior context |
| IT-CHAT-003 | Streaming multi-turn | 1. Stream response → 2. Send follow-up → 3. Stream again | Context maintained across streams |
| IT-CHAT-004 | Conversation with system prompt | 1. Set system prompt → 2. Chat → 3. Verify behavior | Response follows system prompt |

### 3.2 RAG Pipeline

| TC ID | Test Case | Steps | Expected |
|-------|-----------|-------|----------|
| IT-RAG-001 | Document ingestion | 1. Upload PDF → 2. Verify chunked → 3. Verify embedded | Document searchable in vector DB |
| IT-RAG-002 | RAG-augmented chat | 1. Ingest doc → 2. Ask question about doc | Response includes doc information |
| IT-RAG-003 | RAG with no relevant docs | 1. Ask unrelated question | Response without hallucinated RAG context |
| IT-RAG-004 | Multiple document RAG | 1. Ingest 3 docs → 2. Ask cross-doc question | Response synthesizes from multiple docs |

### 3.3 Authentication & Rate Limiting

| TC ID | Test Case | Steps | Expected |
|-------|-----------|-------|----------|
| IT-AUTH-001 | Full auth flow | 1. Create user → 2. Create API key → 3. Make API call | Request succeeds |
| IT-AUTH-002 | Rate limit enforcement | 1. Send requests at rate limit → 2. Exceed limit | 429 returned, then allowed after window |
| IT-AUTH-003 | Key rotation | 1. Create key → 2. Use key → 3. Rotate → 4. Old key fails | Old key rejected, new key works |
| IT-AUTH-004 | Audit trail | 1. Make API calls → 2. Query audit logs | All calls recorded with metadata |

---

## 4. Performance Test Cases

| TC ID | Test Case | Config | Pass Criteria |
|-------|-----------|--------|---------------|
| PT-001 | Baseline latency | 1 user, 100-token prompt | TTFT < 100ms |
| PT-002 | Concurrent load | 100 users, 100-token prompts | TTFT P95 < 200ms |
| PT-003 | High concurrency | 1000 users, 100-token prompts | TTFT P95 < 500ms, error < 1% |
| PT-004 | Streaming throughput | 100 users, streaming | > 50 tokens/sec per user |
| PT-005 | Long context | 10 users, 16K-token prompts | Response within 30s |
| PT-006 | Sustained load (1h) | 500 users steady | No degradation over time |
| PT-007 | Spike test | 0 → 2000 users in 30s | Error rate < 5% during spike |
| PT-008 | Embedding throughput | 1000 texts batch | < 5s total processing |

---

## 5. Security Test Cases

| TC ID | Test Case | Attack Vector | Expected Defense |
|-------|-----------|---------------|-----------------|
| ST-001 | SQL injection in search | `' OR 1=1 --` in query | Parameterized query blocks injection |
| ST-002 | XSS in message content | `<script>alert(1)</script>` | Content sanitized in output |
| ST-003 | API key brute force | 1000 random keys/min | IP blocked after 10 failures |
| ST-004 | JWT token tampering | Modified payload, same signature | 401 rejection |
| ST-005 | Prompt injection | "Ignore previous instructions..." | Content moderation flags it |
| ST-006 | PII extraction | "What's user X's email?" | Model refuses, audit logged |
| ST-007 | Rate limit bypass (header spoof) | Forged X-Forwarded-For | Server-side IP used |
| ST-008 | Large payload DoS | 100MB request body | 413 Payload Too Large |
| ST-009 | Slowloris connection | Slow trickle of bytes | Connection timeout (30s) |
| ST-010 | Unauthorized model access | User A accesses User B's fine-tuned model | 404 (not 403) |
| ST-011 | Directory traversal | `../../etc/passwd` in doc upload path | Path sanitized |
| ST-012 | CORS validation | Request from unauthorized origin | Blocked by CORS policy |

---

## 6. Chaos / Resilience Test Cases

| TC ID | Test Case | Failure Injected | Expected Behavior |
|-------|-----------|-----------------|-------------------|
| CH-001 | Inference pod crash | Kill 1 of 3 inference pods | Requests routed to remaining pods, < 5s recovery |
| CH-002 | Database primary failure | Kill PostgreSQL primary | Failover to replica within 30s |
| CH-003 | Redis cluster node failure | Kill 1 Redis node | Other nodes handle traffic, no data loss |
| CH-004 | Network partition | Block inference ↔ DB | Graceful error response, no hang |
| CH-005 | GPU OOM | Send very large batch | Request rejected, other requests unaffected |
| CH-006 | Disk full | Fill model storage disk | New requests rejected, existing responses complete |
| CH-007 | DNS failure | Block DNS resolution | Cached connections work, new fail gracefully |

---

## 7. Regression Test Suite

These tests run on every release to prevent regressions:

| TC ID | Category | Test Case | Automated |
|-------|----------|-----------|-----------|
| RG-001 | API | Chat completion returns valid schema | Yes |
| RG-002 | API | Streaming returns valid SSE format | Yes |
| RG-003 | API | All error codes correct | Yes |
| RG-004 | Auth | Valid key accepted | Yes |
| RG-005 | Auth | Invalid key rejected (401) | Yes |
| RG-006 | Auth | Rate limiting enforced | Yes |
| RG-007 | Model | Model listing returns all models | Yes |
| RG-008 | Model | Inference produces coherent output | Yes |
| RG-009 | RAG | Document ingestion succeeds | Yes |
| RG-010 | RAG | Retrieval returns relevant chunks | Yes |
| RG-011 | Perf | TTFT under threshold | Yes |
| RG-012 | Perf | Throughput above threshold | Yes |
| RG-013 | Conv | Conversation CRUD works | Yes |
| RG-014 | Conv | Multi-turn context maintained | Yes |
| RG-015 | Health | Health endpoint returns 200 | Yes |
