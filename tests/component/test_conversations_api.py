"""Component tests for the Conversations API endpoints."""

import pytest
from httpx import AsyncClient


class TestConversationsAPI:
    """Conversations endpoint tests."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, client: AsyncClient, auth_headers: dict):
        """Create returns 201 with conversation object."""
        response = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={"model_id": "neuronlm-7b", "title": "Test Chat"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["model_id"] == "neuronlm-7b"
        assert data["title"] == "Test Chat"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_list_conversations(self, client: AsyncClient, auth_headers: dict):
        """List returns user's conversations."""
        # Create a conversation first
        await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={"model_id": "neuronlm-7b"},
        )

        response = await client.get("/v1/conversations", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert len(data["conversations"]) >= 1

    @pytest.mark.asyncio
    async def test_get_conversation(self, client: AsyncClient, auth_headers: dict):
        """Get by ID returns conversation details."""
        # Create
        create_resp = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={"model_id": "neuronlm-7b", "title": "Fetch Test"},
        )
        conv_id = create_resp.json()["id"]

        # Get
        response = await client.get(f"/v1/conversations/{conv_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == conv_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, client: AsyncClient, auth_headers: dict):
        """Non-existent conversation returns 404."""
        response = await client.get(
            "/v1/conversations/nonexistent-id", headers=auth_headers
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_conversation(self, client: AsyncClient, auth_headers: dict):
        """Delete returns 204."""
        create_resp = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={"model_id": "neuronlm-7b"},
        )
        conv_id = create_resp.json()["id"]

        response = await client.delete(
            f"/v1/conversations/{conv_id}", headers=auth_headers
        )
        assert response.status_code == 204

        # Verify deleted
        get_resp = await client.get(
            f"/v1/conversations/{conv_id}", headers=auth_headers
        )
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_with_system_prompt(self, client: AsyncClient, auth_headers: dict):
        """Conversation with system prompt stores it."""
        response = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={
                "model_id": "neuronlm-7b",
                "system_prompt": "You are a coding assistant.",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["system_prompt"] == "You are a coding assistant."

    @pytest.mark.asyncio
    async def test_get_messages(self, client: AsyncClient, auth_headers: dict):
        """Messages endpoint returns conversation messages."""
        # Create conversation with system prompt (auto-adds a message)
        create_resp = await client.post(
            "/v1/conversations",
            headers=auth_headers,
            json={
                "model_id": "neuronlm-7b",
                "system_prompt": "Be helpful.",
            },
        )
        conv_id = create_resp.json()["id"]

        response = await client.get(
            f"/v1/conversations/{conv_id}/messages", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: AsyncClient):
        """Conversations without auth return 401."""
        response = await client.get("/v1/conversations")
        assert response.status_code == 401
