"""Content moderation service for input/output filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModerationResult:
    """Result of content moderation check."""
    flagged: bool
    categories: list[str]
    message: Optional[str] = None


class ContentModerator:
    """Checks content against moderation policies.

    In production, this would integrate with a dedicated moderation model.
    This implementation provides pattern-based baseline filtering.
    """

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior\s+instructions",
        r"forget\s+(all\s+)?(your\s+)?instructions",
        r"override\s+(system|your)\s+(prompt|instructions)",
        r"you\s+are\s+now\s+(?:DAN|jailbroken|unrestricted)",
        r"pretend\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|guidelines)",
    ]

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._injection_re = re.compile(
            "|".join(self.INJECTION_PATTERNS), re.IGNORECASE
        )

    def check_input(self, text: str) -> ModerationResult:
        """Check user input for policy violations."""
        if not self._enabled:
            return ModerationResult(flagged=False, categories=[])

        categories: list[str] = []

        # Check for prompt injection attempts
        if self._injection_re.search(text):
            categories.append("prompt_injection")

        if categories:
            return ModerationResult(
                flagged=True,
                categories=categories,
                message="Content flagged by moderation policy",
            )

        return ModerationResult(flagged=False, categories=[])

    def check_output(self, text: str) -> ModerationResult:
        """Check model output for policy violations."""
        if not self._enabled:
            return ModerationResult(flagged=False, categories=[])

        categories: list[str] = []

        # Check for PII patterns (basic email/phone detection)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            categories.append("potential_pii")

        if categories:
            return ModerationResult(
                flagged=True,
                categories=categories,
                message="Output flagged by moderation policy",
            )

        return ModerationResult(flagged=False, categories=[])


_moderator: Optional[ContentModerator] = None


def get_moderator() -> ContentModerator:
    """Return the global content moderator instance."""
    global _moderator
    if _moderator is None:
        from neuronlm.config import get_settings
        _moderator = ContentModerator(enabled=get_settings().enable_content_moderation)
    return _moderator
