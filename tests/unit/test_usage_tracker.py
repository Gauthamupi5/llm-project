"""Unit tests for usage tracking."""

import pytest

from neuronlm.services.usage_tracker import UsageTracker


class TestUsageTracking:

    def test_record_usage(self, usage_tracker: UsageTracker):
        """Usage is recorded and returns an ID."""
        record_id = usage_tracker.record_usage(
            user_id="user-1",
            model_id="neuronlm-7b",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert record_id is not None

    def test_daily_usage_accumulated(self, usage_tracker: UsageTracker):
        """Multiple records accumulate in daily totals."""
        usage_tracker.record_usage("user-1", "neuronlm-7b", 100, 50)
        usage_tracker.record_usage("user-1", "neuronlm-7b", 200, 100)

        daily = usage_tracker.get_daily_usage("user-1")
        assert daily["prompt_tokens"] == 300
        assert daily["completion_tokens"] == 150
        assert daily["total_tokens"] == 450
        assert daily["requests"] == 2

    def test_check_quota_within_limit(self, usage_tracker: UsageTracker):
        """User within quota returns True."""
        usage_tracker.record_usage("user-1", "neuronlm-7b", 100, 50)
        assert usage_tracker.check_quota("user-1", daily_limit=1_000_000) is True

    def test_check_quota_exceeded(self, usage_tracker: UsageTracker):
        """User exceeding quota returns False."""
        usage_tracker.record_usage("user-1", "neuronlm-7b", 500_000, 500_001)
        assert usage_tracker.check_quota("user-1", daily_limit=1_000_000) is False

    def test_get_user_records(self, usage_tracker: UsageTracker):
        """User records are returned in reverse chronological order."""
        for i in range(5):
            usage_tracker.record_usage("user-1", "neuronlm-7b", i * 10, i * 5)

        records = usage_tracker.get_user_records("user-1")
        assert len(records) == 5

    def test_user_isolation(self, usage_tracker: UsageTracker):
        """Different users have separate usage."""
        usage_tracker.record_usage("user-1", "neuronlm-7b", 100, 50)
        usage_tracker.record_usage("user-2", "neuronlm-7b", 200, 100)

        daily1 = usage_tracker.get_daily_usage("user-1")
        daily2 = usage_tracker.get_daily_usage("user-2")

        assert daily1["total_tokens"] == 150
        assert daily2["total_tokens"] == 300

    def test_empty_user_usage(self, usage_tracker: UsageTracker):
        """User with no usage returns zeros."""
        daily = usage_tracker.get_daily_usage("no-usage-user")
        assert daily["total_tokens"] == 0
        assert daily["requests"] == 0

    def test_user_records_pagination(self, usage_tracker: UsageTracker):
        """Records support pagination."""
        for i in range(10):
            usage_tracker.record_usage("user-1", "neuronlm-7b", 10, 5)

        page1 = usage_tracker.get_user_records("user-1", limit=3, offset=0)
        page2 = usage_tracker.get_user_records("user-1", limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["id"] != page2[0]["id"]
