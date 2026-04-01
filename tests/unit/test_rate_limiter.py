"""Unit tests for rate limiting."""

import time
import pytest

from neuronlm.core.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    """Token bucket algorithm tests."""

    def test_initial_capacity(self):
        """Bucket starts at full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.remaining == 10

    def test_consume_succeeds(self):
        """Consuming within capacity succeeds."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.remaining == 5

    def test_consume_over_capacity_fails(self):
        """Consuming more than available fails."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        bucket.consume(5)
        assert bucket.consume(1) is False

    def test_refill_over_time(self):
        """Tokens refill after waiting."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # 100/sec for fast test
        bucket.consume(10)
        assert bucket.remaining == 0

        time.sleep(0.05)  # Wait 50ms → should refill ~5 tokens
        assert bucket.remaining >= 3  # Some tokens should have refilled

    def test_retry_after_zero_when_available(self):
        """retry_after is 0 when tokens are available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.retry_after == 0.0


class TestRateLimiter:
    """UT-RL-001 through UT-RL-006: Rate limiter tests."""

    def test_under_limit(self, rate_limiter: RateLimiter):
        """UT-RL-001: Requests under limit are allowed."""
        for _ in range(30):
            assert rate_limiter.check("user-1") is True

    def test_at_limit(self, rate_limiter: RateLimiter):
        """UT-RL-002: Requests at limit are allowed."""
        for _ in range(60):
            assert rate_limiter.check("user-1") is True

    def test_over_limit(self, rate_limiter: RateLimiter):
        """UT-RL-003: Request over limit is rejected."""
        for _ in range(60):
            rate_limiter.check("user-1")
        assert rate_limiter.check("user-1") is False

    def test_different_keys_independent(self, rate_limiter: RateLimiter):
        """Different keys have independent limits."""
        for _ in range(60):
            rate_limiter.check("user-1")
        assert rate_limiter.check("user-1") is False
        assert rate_limiter.check("user-2") is True  # Different key, not limited

    def test_remaining_count(self, rate_limiter: RateLimiter):
        """Remaining count decreases correctly."""
        initial = rate_limiter.remaining("new-user")
        assert initial == 60

        rate_limiter.check("new-user")
        assert rate_limiter.remaining("new-user") == 59

    def test_reset(self, rate_limiter: RateLimiter):
        """Reset restores full capacity."""
        for _ in range(60):
            rate_limiter.check("user-1")
        assert rate_limiter.check("user-1") is False

        rate_limiter.reset("user-1")
        assert rate_limiter.check("user-1") is True

    def test_retry_after(self, rate_limiter: RateLimiter):
        """retry_after returns positive value when limited."""
        for _ in range(60):
            rate_limiter.check("user-1")
        retry = rate_limiter.retry_after("user-1")
        assert retry > 0

    def test_consume_multiple(self):
        """Can consume multiple tokens at once."""
        limiter = RateLimiter(default_rpm=10)
        assert limiter.check("user-1", count=5) is True
        assert limiter.remaining("user-1") == 5
        assert limiter.check("user-1", count=6) is False  # Only 5 remaining
