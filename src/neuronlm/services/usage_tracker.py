"""Usage tracking service for token consumption and billing."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional


class UsageTracker:
    """Tracks token consumption per user/API key.

    In production, this writes to Kafka + PostgreSQL.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._daily_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
        )

    def record_usage(
        self,
        user_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: Optional[int] = None,
        api_key_prefix: Optional[str] = None,
        status: str = "success",
    ) -> str:
        """Record a usage event. Returns the record ID."""
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        date_key = f"{user_id}:{now.strftime('%Y-%m-%d')}"

        record = {
            "id": record_id,
            "user_id": user_id,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "api_key_prefix": api_key_prefix,
            "status": status,
            "created_at": now.isoformat(),
        }

        self._records.append(record)

        # Update daily totals
        daily = self._daily_totals[date_key]
        daily["prompt_tokens"] += prompt_tokens
        daily["completion_tokens"] += completion_tokens
        daily["total_tokens"] += prompt_tokens + completion_tokens
        daily["requests"] += 1

        return record_id

    def get_daily_usage(self, user_id: str, date: Optional[str] = None) -> dict[str, int]:
        """Get usage for a user on a specific date."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_key = f"{user_id}:{date}"
        return dict(self._daily_totals.get(date_key, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }))

    def check_quota(self, user_id: str, daily_limit: int) -> bool:
        """Check if user is within their daily token quota."""
        usage = self.get_daily_usage(user_id)
        return usage["total_tokens"] < daily_limit

    def get_user_records(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get usage records for a user."""
        user_records = [r for r in self._records if r["user_id"] == user_id]
        user_records.sort(key=lambda r: r["created_at"], reverse=True)
        return user_records[offset : offset + limit]


# Module-level singleton
_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Return the global usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker
