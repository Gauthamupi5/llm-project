"""Unit tests for tokenization and message truncation."""

import pytest

from neuronlm.core.tokenizer import TokenizerService


class TestTokenCounting:
    """UT-TOK-001 through UT-TOK-004: Token counting tests."""

    def test_count_simple_text(self, tokenizer: TokenizerService):
        """UT-TOK-001: Simple text produces non-zero token count."""
        count = tokenizer.count_tokens("Hello world")
        assert count > 0

    def test_count_empty_string(self, tokenizer: TokenizerService):
        """UT-TOK-002: Empty string produces 0 tokens."""
        assert tokenizer.count_tokens("") == 0

    def test_count_special_chars(self, tokenizer: TokenizerService):
        """UT-TOK-003: Special characters are tokenized."""
        count = tokenizer.count_tokens("Hello! @#$%^&*()")
        assert count > 0

    def test_count_multilingual(self, tokenizer: TokenizerService):
        """UT-TOK-004: Multi-language text is tokenized."""
        count = tokenizer.count_tokens("Hello 你好 مرحبا")
        assert count > 0

    def test_count_long_text(self, tokenizer: TokenizerService):
        """Longer text produces more tokens."""
        short_count = tokenizer.count_tokens("Hello")
        long_count = tokenizer.count_tokens("Hello " * 100)
        assert long_count > short_count


class TestTokenEncoding:
    """Token encode/decode tests."""

    def test_encode_decode_roundtrip(self, tokenizer: TokenizerService):
        """Encoding then decoding should return original text."""
        text = "Hello, this is a test message!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_returns_ints(self, tokenizer: TokenizerService):
        """Encoding should return a list of integers."""
        tokens = tokenizer.encode("Hello")
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0


class TestTextTruncation:
    """Text truncation tests."""

    def test_truncate_within_limit(self, tokenizer: TokenizerService):
        """Text within limit is returned as-is."""
        text = "Hello world"
        result = tokenizer.truncate_text(text, max_tokens=100)
        assert result == text

    def test_truncate_exceeding_limit(self, tokenizer: TokenizerService):
        """Text exceeding limit is truncated."""
        text = "Hello world " * 100  # Long text
        result = tokenizer.truncate_text(text, max_tokens=5)
        result_tokens = tokenizer.count_tokens(result)
        assert result_tokens <= 5


class TestMessageTruncation:
    """UT-TOK-005 through UT-TOK-008: Message truncation tests."""

    def test_truncate_preserves_system_prompt(self, tokenizer: TokenizerService):
        """UT-TOK-006: System prompt is preserved during truncation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "msg1 " * 500},
            {"role": "assistant", "content": "reply1 " * 500},
            {"role": "user", "content": "msg2 " * 500},
            {"role": "assistant", "content": "reply2 " * 500},
            {"role": "user", "content": "What is AI?"},
        ]

        result = tokenizer.truncate_messages(messages, max_tokens=200)

        # System prompt should still be there
        assert any(m["role"] == "system" for m in result)

    def test_truncate_preserves_latest_message(self, tokenizer: TokenizerService):
        """UT-TOK-007: Latest message is preserved during truncation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "old msg " * 500},
            {"role": "user", "content": "What is AI?"},
        ]

        result = tokenizer.truncate_messages(messages, max_tokens=100)

        # Last message should be present
        assert result[-1]["content"] == "What is AI?"

    def test_no_truncation_when_within_budget(self, tokenizer: TokenizerService):
        """Messages within budget are returned unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = tokenizer.truncate_messages(messages, max_tokens=10000)
        assert len(result) == len(messages)

    def test_empty_messages(self, tokenizer: TokenizerService):
        """Empty message list returns empty."""
        result = tokenizer.truncate_messages([], max_tokens=100)
        assert result == []

    def test_count_message_tokens(self, tokenizer: TokenizerService):
        """Message token counting includes per-message overhead."""
        messages = [{"role": "user", "content": "Hello"}]
        count = tokenizer.count_message_tokens(messages)
        # Should be more than just the word "Hello" due to overhead
        assert count > tokenizer.count_tokens("Hello")
