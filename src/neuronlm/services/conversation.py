"""Conversation management service — handles multi-turn dialogues."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from neuronlm.core.exceptions import ConversationNotFoundError
from neuronlm.core.tokenizer import get_tokenizer
from neuronlm.models.schemas import (
    ChatMessage,
    ConversationResponse,
    MessageResponse,
    MessageRole,
)


class ConversationStore:
    """In-memory conversation store.

    In production, this is backed by PostgreSQL + Redis cache.
    """

    def __init__(self) -> None:
        self._conversations: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._tokenizer = get_tokenizer()

    def create_conversation(
        self,
        user_id: str,
        model_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ConversationResponse:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title or "New Conversation",
            "model_id": model_id,
            "system_prompt": system_prompt,
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
        }

        self._conversations[conversation_id] = conversation
        self._messages[conversation_id] = []

        # Add system prompt as first message if provided
        if system_prompt:
            self.add_message(conversation_id, MessageRole.SYSTEM, system_prompt)

        return ConversationResponse(**conversation)

    def get_conversation(self, conversation_id: str, user_id: str) -> ConversationResponse:
        """Get a conversation by ID, scoped to user."""
        conv = self._conversations.get(conversation_id)
        if conv is None or conv["user_id"] != user_id:
            raise ConversationNotFoundError(conversation_id)
        return ConversationResponse(**conv)

    def list_conversations(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[ConversationResponse]:
        """List conversations for a user with pagination."""
        user_convs = [
            ConversationResponse(**c)
            for c in self._conversations.values()
            if c["user_id"] == user_id
        ]
        # Sort by updated_at descending
        user_convs.sort(key=lambda c: c.updated_at, reverse=True)
        return user_convs[offset : offset + limit]

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation and its messages."""
        conv = self._conversations.get(conversation_id)
        if conv is None or conv["user_id"] != user_id:
            raise ConversationNotFoundError(conversation_id)

        del self._conversations[conversation_id]
        self._messages.pop(conversation_id, None)
        return True

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        model_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
    ) -> MessageResponse:
        """Add a message to a conversation."""
        if conversation_id not in self._conversations:
            raise ConversationNotFoundError(conversation_id)

        message_id = str(uuid.uuid4())
        token_count = self._tokenizer.count_tokens(content)
        now = datetime.now(timezone.utc).isoformat()

        message = {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "token_count": token_count,
            "model_id": model_id,
            "latency_ms": latency_ms,
            "created_at": now,
        }

        self._messages[conversation_id].append(message)

        # Update conversation metadata
        conv = self._conversations[conversation_id]
        conv["message_count"] = len(self._messages[conversation_id])
        conv["updated_at"] = now

        # Auto-generate title from first user message
        if conv["title"] == "New Conversation" and role == MessageRole.USER:
            conv["title"] = content[:80] + ("..." if len(content) > 80 else "")

        return MessageResponse(**message)

    def get_messages(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> list[MessageResponse]:
        """Get messages for a conversation."""
        if conversation_id not in self._messages:
            raise ConversationNotFoundError(conversation_id)

        messages = self._messages[conversation_id]
        if limit:
            messages = messages[-limit:]

        return [MessageResponse(**m) for m in messages]

    def get_chat_messages(self, conversation_id: str) -> list[ChatMessage]:
        """Get messages in ChatMessage format for inference."""
        if conversation_id not in self._messages:
            raise ConversationNotFoundError(conversation_id)

        return [
            ChatMessage(role=m["role"], content=m["content"])
            for m in self._messages[conversation_id]
        ]


# Module-level singleton
_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Return the global conversation store instance."""
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
