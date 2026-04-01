"""Unit tests for conversation management."""

import pytest

from neuronlm.core.exceptions import ConversationNotFoundError
from neuronlm.models.schemas import MessageRole
from neuronlm.services.conversation import ConversationStore


class TestConversationCRUD:
    """CT-CONV-001 through CT-CONV-008: Conversation management tests."""

    def test_create_conversation(self, conversation_store: ConversationStore):
        """CT-CONV-001: Create returns conversation with ID."""
        conv = conversation_store.create_conversation(
            user_id="user-1",
            model_id="neuronlm-7b",
            title="Test Chat",
        )
        assert conv.id is not None
        assert conv.user_id == "user-1"
        assert conv.model_id == "neuronlm-7b"
        assert conv.title == "Test Chat"

    def test_get_conversation_by_id(self, conversation_store: ConversationStore):
        """CT-CONV-002: Get by ID returns conversation with details."""
        created = conversation_store.create_conversation("user-1", "neuronlm-7b")
        fetched = conversation_store.get_conversation(created.id, "user-1")
        assert fetched.id == created.id

    def test_list_conversations(self, conversation_store: ConversationStore):
        """CT-CONV-003: List returns user's conversations."""
        conversation_store.create_conversation("user-1", "neuronlm-7b", title="Chat 1")
        conversation_store.create_conversation("user-1", "neuronlm-7b", title="Chat 2")
        conversation_store.create_conversation("user-2", "neuronlm-7b", title="Other")

        user1_convs = conversation_store.list_conversations("user-1")
        assert len(user1_convs) == 2

    def test_delete_conversation(self, conversation_store: ConversationStore):
        """CT-CONV-004: Delete removes conversation."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        result = conversation_store.delete_conversation(conv.id, "user-1")
        assert result is True

        with pytest.raises(ConversationNotFoundError):
            conversation_store.get_conversation(conv.id, "user-1")

    def test_add_message(self, conversation_store: ConversationStore):
        """CT-CONV-005: Messages are added and stored."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        msg = conversation_store.add_message(
            conv.id, MessageRole.USER, "Hello!"
        )
        assert msg.content == "Hello!"
        assert msg.role == MessageRole.USER
        assert msg.token_count > 0

    def test_get_by_wrong_user(self, conversation_store: ConversationStore):
        """CT-CONV-006: Wrong user gets NotFound (not Forbidden)."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        with pytest.raises(ConversationNotFoundError):
            conversation_store.get_conversation(conv.id, "user-2")

    def test_create_with_system_prompt(self, conversation_store: ConversationStore):
        """CT-CONV-007: System prompt is stored."""
        conv = conversation_store.create_conversation(
            user_id="user-1",
            model_id="neuronlm-7b",
            system_prompt="You are a helpful assistant.",
        )
        assert conv.system_prompt == "You are a helpful assistant."

        # System prompt should be in messages
        messages = conversation_store.get_messages(conv.id)
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == "You are a helpful assistant."

    def test_title_auto_generated(self, conversation_store: ConversationStore):
        """CT-CONV-008: Title generated from first user message."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        assert conv.title == "New Conversation"

        conversation_store.add_message(conv.id, MessageRole.USER, "What is quantum computing?")
        updated = conversation_store.get_conversation(conv.id, "user-1")
        assert "quantum" in updated.title.lower()


class TestMessageOperations:
    """Message storage and retrieval tests."""

    def test_get_messages(self, conversation_store: ConversationStore):
        """Messages are returned in order."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        conversation_store.add_message(conv.id, MessageRole.USER, "Hello")
        conversation_store.add_message(conv.id, MessageRole.ASSISTANT, "Hi!")
        conversation_store.add_message(conv.id, MessageRole.USER, "How are you?")

        messages = conversation_store.get_messages(conv.id)
        assert len(messages) == 3
        assert messages[0].content == "Hello"
        assert messages[2].content == "How are you?"

    def test_get_messages_with_limit(self, conversation_store: ConversationStore):
        """Limit returns only last N messages."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        for i in range(10):
            conversation_store.add_message(conv.id, MessageRole.USER, f"Message {i}")

        messages = conversation_store.get_messages(conv.id, limit=3)
        assert len(messages) == 3
        assert messages[0].content == "Message 7"

    def test_get_chat_messages(self, conversation_store: ConversationStore):
        """chat messages format returned for inference."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        conversation_store.add_message(conv.id, MessageRole.USER, "Hello")
        conversation_store.add_message(conv.id, MessageRole.ASSISTANT, "Hi!")

        chat_msgs = conversation_store.get_chat_messages(conv.id)
        assert len(chat_msgs) == 2
        assert chat_msgs[0].role == MessageRole.USER

    def test_message_count_updated(self, conversation_store: ConversationStore):
        """Message count in conversation is updated."""
        conv = conversation_store.create_conversation("user-1", "neuronlm-7b")
        conversation_store.add_message(conv.id, MessageRole.USER, "Hello")
        conversation_store.add_message(conv.id, MessageRole.ASSISTANT, "Hi!")

        updated = conversation_store.get_conversation(conv.id, "user-1")
        assert updated.message_count == 2

    def test_message_to_nonexistent_conversation(self, conversation_store: ConversationStore):
        """Adding message to non-existent conversation raises error."""
        with pytest.raises(ConversationNotFoundError):
            conversation_store.add_message(
                "nonexistent-id", MessageRole.USER, "Hello"
            )

    def test_delete_nonexistent_conversation(self, conversation_store: ConversationStore):
        """Deleting non-existent conversation raises error."""
        with pytest.raises(ConversationNotFoundError):
            conversation_store.delete_conversation("nonexistent-id", "user-1")

    def test_pagination(self, conversation_store: ConversationStore):
        """Pagination works correctly."""
        for i in range(5):
            conversation_store.create_conversation("user-1", "neuronlm-7b", title=f"Chat {i}")

        page1 = conversation_store.list_conversations("user-1", limit=2, offset=0)
        page2 = conversation_store.list_conversations("user-1", limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id
