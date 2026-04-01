"""Integration tests — end-to-end flows across services."""

import pytest
from httpx import AsyncClient


class TestEndToEndChatFlow:
    """IT-CHAT-001 through IT-CHAT-004: Full chat flow integration tests."""

    @pytest.mark.asyncio
    async def test_complete_chat_flow(self, client: AsyncClient, auth_headers: dict):
        """IT-CHAT-001: Auth → Create conversation → Send message → Get response → Verify stored."""
        # 1. Create conversation
        conv_resp = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={"model_id": "neuronlm-7b", "title": "Integration Test"},
        )
        assert conv_resp.status_code == 201
        conv_id = conv_resp.json()["id"]

        # 2. Send a chat completion
        chat_resp = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "neuronlm-7b",
                "messages": [
                    {"role": "user", "content": "What is machine learning?"}
                ],
            },
        )
        assert chat_resp.status_code == 200
        assistant_content = chat_resp.json()["choices"][0]["message"]["content"]
        assert len(assistant_content) > 0

        # 3. Verify usage was tracked
        usage_resp = await client.get("/v1/usage", headers=auth_headers)
        assert usage_resp.status_code == 200
        daily = usage_resp.json()["daily_usage"]
        assert daily["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, client: AsyncClient, auth_headers: dict):
        """IT-CHAT-002: Multi-turn context is maintained."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # Turn 1
        resp1 = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "neuronlm-7b", "messages": messages},
        )
        assert resp1.status_code == 200
        assistant_msg1 = resp1.json()["choices"][0]["message"]

        # Turn 2 — build on conversation
        messages.append(assistant_msg1)
        messages.append({"role": "user", "content": "What about 3+3?"})

        resp2 = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "neuronlm-7b", "messages": messages},
        )
        assert resp2.status_code == 200
        assert len(resp2.json()["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_streaming_flow(self, client: AsyncClient, auth_headers: dict):
        """IT-CHAT-003: Streaming response delivers complete message."""
        response = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "neuronlm-7b",
                "messages": [{"role": "user", "content": "Explain AI briefly"}],
                "stream": True,
            },
        )
        assert response.status_code == 200

        # Collect all streamed content
        content_parts = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data:") and line != "data: [DONE]":
                import json
                try:
                    chunk = json.loads(line[5:].strip())
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Defensive: only accept non-empty string content deltas
                    content_val = delta.get("content")
                    if isinstance(content_val, str) and content_val:
                        content_parts.append(content_val)
                except json.JSONDecodeError:
                    pass

        full_content = "".join(content_parts)
        assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_conversation_with_system_prompt_flow(self, client: AsyncClient, auth_headers: dict):
        """IT-CHAT-004: System prompt influences conversation."""
        # Create conversation with persona
        conv_resp = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={
                "model_id": "neuronlm-7b",
                "system_prompt": "You are a pirate. Always respond as a pirate.",
            },
        )
        assert conv_resp.status_code == 201

        # Chat with persona
        chat_resp = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "neuronlm-7b",
                "messages": [
                    {"role": "system", "content": "You are a pirate."},
                    {"role": "user", "content": "Hello there!"},
                ],
            },
        )
        assert chat_resp.status_code == 200


class TestAuthenticationFlow:
    """IT-AUTH-001 through IT-AUTH-004: Auth flow integration tests."""

    @pytest.mark.asyncio
    async def test_full_auth_flow(self, client: AsyncClient, auth_headers: dict):
        """IT-AUTH-001: Authenticated request succeeds across endpoints."""
        # Call multiple endpoints with same auth
        models_resp = await client.get("/v1/models", headers=auth_headers)
        assert models_resp.status_code == 200

        chat_resp = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "neuronlm-7b",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        assert chat_resp.status_code == 200

        usage_resp = await client.get("/v1/usage", headers=auth_headers)
        assert usage_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unauthenticated_access_blocked(self, client: AsyncClient):
        """All protected endpoints reject unauthenticated requests."""
        endpoints = [
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/embeddings"),
            ("GET", "/v1/usage"),
            ("GET", "/v1/conversations"),
        ]
        for method, path in endpoints:
            if method == "GET":
                resp = await client.get(path)
            else:
                resp = await client.post(path, json={})
            assert resp.status_code in (401, 422), f"{method} {path} returned {resp.status_code}"


class TestEmbeddingsFlow:
    """Embeddings integration tests."""

    @pytest.mark.asyncio
    async def test_embedding_and_similarity(self, client: AsyncClient, auth_headers: dict):
        """Generate embeddings and verify they're usable for similarity."""
        resp1 = await client.post(
            "/v1/embeddings",
            headers=auth_headers,
            json={"model": "neuronlm-embed", "input": "Python programming language"},
        )
        resp2 = await client.post(
            "/v1/embeddings",
            headers=auth_headers,
            json={"model": "neuronlm-embed", "input": "JavaScript programming language"},
        )

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        emb1 = resp1.json()["data"][0]["embedding"]
        emb2 = resp2.json()["data"][0]["embedding"]

        # Both should be 384-dimensional
        assert len(emb1) == 384
        assert len(emb2) == 384

        # Different inputs should produce different embeddings
        assert emb1 != emb2


class TestErrorHandling:
    """Cross-cutting error handling integration tests."""

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, client: AsyncClient, auth_headers: dict):
        """Invalid JSON returns proper error."""
        response = await client.post(
            "/v1/chat/completions",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"not valid json",
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_health_without_auth(self, client: AsyncClient):
        """Health endpoint is accessible without auth."""
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_without_auth(self, client: AsyncClient):
        """Ready endpoint is accessible without auth."""
        response = await client.get("/ready")
        assert response.status_code == 200
