"""Tokenizer service for token counting and text encoding/decoding."""

from __future__ import annotations

import hashlib
from typing import Optional

import tiktoken


class TokenizerService:
    """Handles tokenization using tiktoken (BPE encoding compatible with GPT models)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return self._encoding.n_vocab

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for a list of chat messages.

        Each message has overhead: <|role|> + content + <|sep|> ≈ 4 tokens per message.
        """
        token_count = 0
        for message in messages:
            token_count += 4  # message overhead
            for key, value in message.items():
                token_count += self.count_tokens(str(value))
        token_count += 2  # final assistant priming
        return token_count

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoding.decode(tokens)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])

    def truncate_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        preserve_system: bool = True,
        preserve_last: bool = True,
    ) -> list[dict[str, str]]:
        """Truncate conversation messages to fit within token budget.

        Strategy:
        1. Always keep system prompt (if preserve_system)
        2. Always keep the last user message (if preserve_last)
        3. Remove oldest non-system, non-last messages first
        """
        if not messages:
            return messages

        total_tokens = self.count_message_tokens(
            [{"role": m["role"], "content": m["content"]} for m in messages]
        )
        if total_tokens <= max_tokens:
            return messages

        # Separate protected and removable messages
        protected: list[dict[str, str]] = []
        removable: list[dict[str, str]] = []

        for i, msg in enumerate(messages):
            is_system = preserve_system and msg.get("role") == "system"
            is_last = preserve_last and i == len(messages) - 1
            if is_system or is_last:
                protected.append(msg)
            else:
                removable.append(msg)

        # Start with protected messages and add removable from newest to oldest
        result = [m for m in protected if m.get("role") == "system"]
        remaining_budget = max_tokens - self.count_message_tokens(
            [{"role": m["role"], "content": m["content"]} for m in protected]
        )

        kept_removable: list[dict[str, str]] = []
        for msg in reversed(removable):
            msg_tokens = self.count_message_tokens(
                [{"role": msg["role"], "content": msg["content"]}]
            )
            if remaining_budget >= msg_tokens:
                kept_removable.insert(0, msg)
                remaining_budget -= msg_tokens
            else:
                break

        result.extend(kept_removable)

        # Add the last message (non-system protected)
        for m in protected:
            if m.get("role") != "system":
                result.append(m)

        return result


# Module-level singleton
_tokenizer: Optional[TokenizerService] = None


def get_tokenizer() -> TokenizerService:
    """Return the global tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = TokenizerService()
    return _tokenizer
