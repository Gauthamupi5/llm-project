"""Unit tests for request/response validation (Pydantic models)."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from neuronlm.models.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingRequest,
    MessageRole,
    ConversationCreateRequest,
)


class TestChatCompletionRequest:
    """UT-VAL-001 through UT-VAL-015: Request validation tests."""

    def test_valid_request(self):
        """UT-VAL-001: Complete valid request parses successfully."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            temperature=0.7,
            max_tokens=100,
        )
        assert req.model == "neuronlm-7b"
        assert len(req.messages) == 1
        assert req.temperature == 0.7

    def test_missing_model(self):
        """UT-VAL-002: Missing model field raises ValidationError."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ChatCompletionRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            )
        assert "model" in str(exc_info.value)

    def test_missing_messages(self):
        """UT-VAL-003: Missing messages field raises ValidationError."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ChatCompletionRequest(model="neuronlm-7b")
        assert "messages" in str(exc_info.value)

    def test_empty_messages(self):
        """UT-VAL-004: Empty messages array raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(model="neuronlm-7b", messages=[])

    def test_invalid_role(self):
        """UT-VAL-005: Invalid role raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatMessage(role="invalid_role", content="Hello")

    def test_temperature_too_high(self):
        """UT-VAL-006: Temperature > 2.0 raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                temperature=3.0,
            )

    def test_temperature_negative(self):
        """UT-VAL-007: Negative temperature raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                temperature=-0.1,
            )

    def test_temperature_boundary_zero(self):
        """UT-VAL-008: Temperature 0.0 is valid."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            temperature=0.0,
        )
        assert req.temperature == 0.0

    def test_temperature_boundary_two(self):
        """UT-VAL-009: Temperature 2.0 is valid."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            temperature=2.0,
        )
        assert req.temperature == 2.0

    def test_max_tokens_exceeds_limit(self):
        """UT-VAL-010: max_tokens > 32768 raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                max_tokens=100000,
            )

    def test_max_tokens_negative(self):
        """UT-VAL-011: Negative max_tokens raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                max_tokens=-1,
            )

    def test_too_many_stop_sequences(self):
        """UT-VAL-013: More than 4 stop sequences raises error."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                stop=["a", "b", "c", "d", "e"],
            )

    def test_valid_stop_sequences(self):
        """UT-VAL-014: 4 stop sequences is valid."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            stop=[".", "!", "?", "\n"],
        )
        assert len(req.stop) == 4

    def test_top_p_out_of_range(self):
        """UT-VAL-015: top_p > 1.0 raises ValidationError."""
        with pytest.raises(PydanticValidationError):
            ChatCompletionRequest(
                model="neuronlm-7b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                top_p=1.5,
            )

    def test_defaults_applied(self):
        """Verify default values are correctly applied."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        )
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.max_tokens == 2048
        assert req.stream is False
        assert req.frequency_penalty == 0.0
        assert req.presence_penalty == 0.0

    def test_all_message_roles(self):
        """All valid roles are accepted."""
        for role in MessageRole:
            msg = ChatMessage(role=role, content="Hello")
            assert msg.role == role

    def test_string_stop_sequence(self):
        """Stop can be a single string."""
        req = ChatCompletionRequest(
            model="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            stop=".",
        )
        assert req.stop == "."


class TestEmbeddingRequest:
    """Embedding request validation tests."""

    def test_valid_single_input(self):
        req = EmbeddingRequest(model="neuronlm-embed", input="Hello world")
        assert req.input == "Hello world"

    def test_valid_batch_input(self):
        req = EmbeddingRequest(model="neuronlm-embed", input=["Hello", "World"])
        assert len(req.input) == 2

    def test_invalid_encoding_format(self):
        with pytest.raises(PydanticValidationError):
            EmbeddingRequest(model="neuronlm-embed", input="Hello", encoding_format="xml")


class TestConversationCreateRequest:
    """Conversation creation request validation tests."""

    def test_valid_request(self):
        req = ConversationCreateRequest(model_id="neuronlm-7b")
        assert req.model_id == "neuronlm-7b"
        assert req.title is None

    def test_with_title_and_system_prompt(self):
        req = ConversationCreateRequest(
            model_id="neuronlm-7b",
            title="Test Conversation",
            system_prompt="You are helpful.",
        )
        assert req.title == "Test Conversation"
        assert req.system_prompt == "You are helpful."
