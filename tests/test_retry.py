"""Tests for retry logic."""

import pytest

from django_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
)
from django_llm.retry import calculate_backoff, should_retry


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_attempt_uses_base_delay(self):
        # With jitter disabled, first attempt should be exactly base_delay
        delay = calculate_backoff(attempt=0, base_delay=1.0, max_delay=60.0, jitter=False)
        assert delay == 1.0

    def test_exponential_growth(self):
        # Without jitter: delay = base * 2^attempt
        delays = [
            calculate_backoff(attempt=i, base_delay=1.0, max_delay=60.0, jitter=False)
            for i in range(5)
        ]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        # Should never exceed max_delay
        delay = calculate_backoff(attempt=10, base_delay=1.0, max_delay=60.0, jitter=False)
        assert delay == 60.0

    def test_jitter_adds_randomness(self):
        # With jitter, delays should vary
        delays = [
            calculate_backoff(attempt=0, base_delay=1.0, max_delay=60.0, jitter=True)
            for _ in range(10)
        ]
        # All delays should be between 0.5 and 1.5 (base * 0.5 to base * 1.5)
        assert all(0.5 <= d <= 1.5 for d in delays)
        # Should have some variation (not all the same)
        assert len(set(delays)) > 1

    def test_custom_base_delay(self):
        delay = calculate_backoff(attempt=0, base_delay=2.0, max_delay=60.0, jitter=False)
        assert delay == 2.0

    def test_custom_max_delay(self):
        delay = calculate_backoff(attempt=10, base_delay=1.0, max_delay=10.0, jitter=False)
        assert delay == 10.0


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_rate_limit_error_should_retry(self):
        error = RateLimitError("Rate limited", provider="openai")
        assert should_retry(error) is True

    def test_auth_error_should_not_retry(self):
        error = AuthenticationError("Invalid API key", provider="openai")
        assert should_retry(error) is False

    def test_invalid_request_should_not_retry(self):
        error = InvalidRequestError("Bad request", provider="openai")
        assert should_retry(error) is False

    def test_connection_error_should_retry(self):
        import httpx
        error = httpx.ConnectError("Connection refused")
        assert should_retry(error) is True

    def test_timeout_error_should_retry(self):
        import httpx
        error = httpx.TimeoutException("Request timed out")
        assert should_retry(error) is True

    def test_generic_exception_should_not_retry(self):
        error = ValueError("Some error")
        assert should_retry(error) is False

    def test_keyboard_interrupt_should_not_retry(self):
        error = KeyboardInterrupt()
        assert should_retry(error) is False
