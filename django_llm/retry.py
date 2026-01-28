"""Retry logic with exponential backoff for API calls."""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from django_llm.exceptions import RateLimitError

logger = logging.getLogger("django_llm")

T = TypeVar("T")


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter to prevent thundering herd.

    Returns:
        Delay in seconds before next retry.
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = min(base_delay * (2**attempt), max_delay)

    if jitter:
        # Add random jitter between 0 and delay
        delay = delay * (0.5 + random.random())

    return delay


def should_retry(exception: Exception) -> bool:
    """Determine if an exception is retryable.

    Retryable errors:
    - Rate limit errors (429)
    - Server errors (5xx)
    - Connection errors

    Non-retryable errors:
    - Authentication errors (401, 403)
    - Bad request errors (400)
    - Not found errors (404)
    """
    import httpx

    from django_llm.exceptions import AuthenticationError, InvalidRequestError

    # Never retry auth or invalid request errors
    if isinstance(exception, (AuthenticationError, InvalidRequestError)):
        return False

    # Always retry rate limit errors
    if isinstance(exception, RateLimitError):
        return True

    # Retry connection errors
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
        return True

    # Check for server errors by status code
    if hasattr(exception, "status_code"):
        status = exception.status_code
        # Retry 5xx server errors and 429 rate limits
        if status >= 500 or status == 429:
            return True

    return False


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable:
    """Decorator to add retry logic to a function.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.

    Example:
        @with_retry(max_retries=3)
        def call_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not should_retry(e):
                        raise

                    if attempt >= max_retries:
                        logger.warning(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    delay = calculate_backoff(attempt, base_delay, max_delay)
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )
                    time.sleep(delay)

            # Should not reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


def with_async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable:
    """Async decorator to add retry logic to a coroutine.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.

    Example:
        @with_async_retry(max_retries=3)
        async def call_api():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not should_retry(e):
                        raise

                    if attempt >= max_retries:
                        logger.warning(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    delay = calculate_backoff(attempt, base_delay, max_delay)
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )
                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Initial backoff delay in seconds (default: 1.0).
        max_delay: Maximum backoff delay in seconds (default: 60.0).
        retry_on_rate_limit: Whether to retry on rate limit errors (default: True).
        retry_on_server_error: Whether to retry on 5xx errors (default: True).
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_on_rate_limit: bool = True,
        retry_on_server_error: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on_rate_limit = retry_on_rate_limit
        self.retry_on_server_error = retry_on_server_error
