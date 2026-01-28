"""Logging and observability hooks for django-llm.

This module provides structured logging and hooks for monitoring LLM interactions.
Configure logging via Django's LOGGING setting or Python's logging module.

Example Django settings:

    LOGGING = {
        'version': 1,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            'django_llm': {
                'handlers': ['console'],
                'level': 'INFO',
            },
        },
    }

For custom observability (metrics, APM, etc.), register hooks:

    from django_llm.logging import add_request_hook, add_response_hook

    def on_request(model, messages, **kwargs):
        statsd.increment('llm.requests', tags=[f'model:{model}'])

    def on_response(model, message, duration_ms, **kwargs):
        statsd.timing('llm.latency', duration_ms, tags=[f'model:{model}'])
        if message.cost:
            statsd.gauge('llm.cost', float(message.cost), tags=[f'model:{model}'])

    add_request_hook(on_request)
    add_response_hook(on_response)
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from django_llm.message import Message

# Main logger for django-llm
logger = logging.getLogger("django_llm")

# Hooks for custom observability
_request_hooks: list[Callable[..., None]] = []
_response_hooks: list[Callable[..., None]] = []
_error_hooks: list[Callable[..., None]] = []


def add_request_hook(hook: Callable[..., None]) -> None:
    """Add a hook that's called before each LLM request.

    The hook receives:
        - model: str - The model being called
        - messages: list[Message] - The conversation messages
        - **kwargs: Additional parameters (tools, temperature, etc.)

    Example:
        def log_request(model, messages, **kwargs):
            print(f"Calling {model} with {len(messages)} messages")

        add_request_hook(log_request)
    """
    _request_hooks.append(hook)


def add_response_hook(hook: Callable[..., None]) -> None:
    """Add a hook that's called after each successful LLM response.

    The hook receives:
        - model: str - The model that was called
        - message: Message - The response message
        - duration_ms: float - Request duration in milliseconds
        - **kwargs: Additional context

    Example:
        def log_response(model, message, duration_ms, **kwargs):
            print(f"{model} responded in {duration_ms:.0f}ms")
            if message.cost:
                print(f"Cost: ${message.cost:.6f}")

        add_response_hook(log_response)
    """
    _response_hooks.append(hook)


def add_error_hook(hook: Callable[..., None]) -> None:
    """Add a hook that's called when an LLM request fails.

    The hook receives:
        - model: str - The model that was called
        - error: Exception - The exception that occurred
        - duration_ms: float - Request duration in milliseconds
        - **kwargs: Additional context

    Example:
        def log_error(model, error, duration_ms, **kwargs):
            sentry_sdk.capture_exception(error)

        add_error_hook(log_error)
    """
    _error_hooks.append(hook)


def remove_request_hook(hook: Callable[..., None]) -> None:
    """Remove a previously added request hook."""
    if hook in _request_hooks:
        _request_hooks.remove(hook)


def remove_response_hook(hook: Callable[..., None]) -> None:
    """Remove a previously added response hook."""
    if hook in _response_hooks:
        _response_hooks.remove(hook)


def remove_error_hook(hook: Callable[..., None]) -> None:
    """Remove a previously added error hook."""
    if hook in _error_hooks:
        _error_hooks.remove(hook)


def clear_hooks() -> None:
    """Remove all hooks. Useful for testing."""
    _request_hooks.clear()
    _response_hooks.clear()
    _error_hooks.clear()


def _call_request_hooks(model: str, messages: list[Message], **kwargs: Any) -> None:
    """Call all registered request hooks."""
    for hook in _request_hooks:
        try:
            hook(model=model, messages=messages, **kwargs)
        except Exception as e:
            logger.warning(f"Request hook error: {e}")


def _call_response_hooks(
    model: str,
    message: Message,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Call all registered response hooks."""
    for hook in _response_hooks:
        try:
            hook(model=model, message=message, duration_ms=duration_ms, **kwargs)
        except Exception as e:
            logger.warning(f"Response hook error: {e}")


def _call_error_hooks(
    model: str,
    error: Exception,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Call all registered error hooks."""
    for hook in _error_hooks:
        try:
            hook(model=model, error=error, duration_ms=duration_ms, **kwargs)
        except Exception as e:
            logger.warning(f"Error hook error: {e}")


def log_request(model: str, messages: list[Message], **kwargs: Any) -> None:
    """Log an LLM request (called before the request)."""
    message_count = len(messages)
    has_tools = bool(kwargs.get("tools"))
    has_schema = bool(kwargs.get("response_schema"))

    logger.info(
        f"LLM request: model={model} messages={message_count} "
        f"tools={has_tools} schema={has_schema}"
    )
    logger.debug(f"Request details: {kwargs}")

    _call_request_hooks(model, messages, **kwargs)


def log_response(
    model: str,
    message: Message,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log an LLM response (called after successful response)."""
    tokens = message.tokens
    cost = message.cost

    log_parts = [f"LLM response: model={model} duration={duration_ms:.0f}ms"]

    if tokens:
        log_parts.append(
            f"tokens={{in={tokens.input_tokens}, out={tokens.output_tokens}}}"
        )

    if cost:
        log_parts.append(f"cost=${cost:.6f}")

    if message.is_tool_call:
        tool_names = [tc.name for tc in message.tool_calls]
        log_parts.append(f"tool_calls={tool_names}")

    logger.info(" ".join(log_parts))

    _call_response_hooks(model, message, duration_ms, **kwargs)


def log_error(
    model: str,
    error: Exception,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log an LLM error (called when request fails)."""
    error_type = type(error).__name__
    logger.error(
        f"LLM error: model={model} duration={duration_ms:.0f}ms "
        f"error={error_type}: {error}"
    )

    _call_error_hooks(model, error, duration_ms, **kwargs)


@contextmanager
def timed_request(model: str, messages: list[Message], **kwargs: Any):
    """Context manager for timing and logging LLM requests.

    Usage:
        with timed_request(model, messages) as ctx:
            response = provider.complete(messages, model)
            ctx['response'] = response

    The context manager:
    - Logs the request before it starts
    - Times the request duration
    - Logs the response or error when complete
    """
    log_request(model, messages, **kwargs)
    start_time = time.perf_counter()
    context: dict[str, Any] = {}

    try:
        yield context
        duration_ms = (time.perf_counter() - start_time) * 1000
        if "response" in context:
            log_response(model, context["response"], duration_ms, **kwargs)
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_error(model, e, duration_ms, **kwargs)
        raise


@contextmanager
def async_timed_request(model: str, messages: list[Message], **kwargs: Any):
    """Context manager for timing and logging async LLM requests.

    Same as timed_request, but for use in async code.
    Note: This is still a sync context manager, as we just need timing.

    Usage:
        with async_timed_request(model, messages) as ctx:
            response = await provider.acomplete(messages, model)
            ctx['response'] = response
    """
    log_request(model, messages, **kwargs)
    start_time = time.perf_counter()
    context: dict[str, Any] = {}

    try:
        yield context
        duration_ms = (time.perf_counter() - start_time) * 1000
        if "response" in context:
            log_response(model, context["response"], duration_ms, **kwargs)
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_error(model, e, duration_ms, **kwargs)
        raise
