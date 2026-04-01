"""Rate limiter using token bucket algorithm."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    """Token bucket rate limiter."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, count: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed, False if rate limited."""
        self._refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    @property
    def remaining(self) -> int:
        """Number of tokens remaining."""
        self._refill()
        return int(self.tokens)

    @property
    def retry_after(self) -> float:
        """Seconds until at least 1 token is available."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.refill_rate


class RateLimiter:
    """Per-key rate limiter using token bucket algorithm."""

    def __init__(self, default_rpm: int = 60) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._default_rpm = default_rpm

    def _get_bucket(self, key: str, rpm: Optional[int] = None) -> TokenBucket:
        if key not in self._buckets:
            limit = rpm or self._default_rpm
            self._buckets[key] = TokenBucket(
                capacity=limit,
                refill_rate=limit / 60.0,
            )
        return self._buckets[key]

    def check(self, key: str, count: int = 1, rpm: Optional[int] = None) -> bool:
        """Check if a request is allowed for the given key."""
        bucket = self._get_bucket(key, rpm)
        return bucket.consume(count)

    def remaining(self, key: str) -> int:
        """Get remaining requests for the given key."""
        if key not in self._buckets:
            return self._default_rpm
        return self._buckets[key].remaining

    def retry_after(self, key: str) -> float:
        """Get seconds until the key's rate limit resets."""
        if key not in self._buckets:
            return 0.0
        return self._buckets[key].retry_after

    def reset(self, key: str) -> None:
        """Reset the rate limit for a key."""
        self._buckets.pop(key, None)


# Module-level singleton
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        from neuronlm.config import get_settings
        _rate_limiter = RateLimiter(default_rpm=get_settings().rate_limit_requests_per_minute)
    return _rate_limiter
