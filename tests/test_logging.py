"""Tests for logging and observability hooks."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from django_llm.logging import (
    _error_hooks,
    _request_hooks,
    _response_hooks,
    add_error_hook,
    add_request_hook,
    add_response_hook,
    clear_hooks,
    log_error,
    log_request,
    log_response,
    remove_error_hook,
    remove_request_hook,
    remove_response_hook,
    timed_request,
)
from django_llm.message import Message, Role, TokenUsage


@pytest.fixture(autouse=True)
def clean_hooks():
    """Ensure hooks are cleared before and after each test."""
    clear_hooks()
    yield
    clear_hooks()


class TestHookManagement:
    """Tests for hook registration and removal."""

    def test_add_request_hook(self):
        def my_hook(model, messages, **kwargs):
            pass

        add_request_hook(my_hook)
        assert my_hook in _request_hooks

    def test_add_response_hook(self):
        def my_hook(model, message, duration_ms, **kwargs):
            pass

        add_response_hook(my_hook)
        assert my_hook in _response_hooks

    def test_add_error_hook(self):
        def my_hook(model, error, duration_ms, **kwargs):
            pass

        add_error_hook(my_hook)
        assert my_hook in _error_hooks

    def test_remove_request_hook(self):
        def my_hook(model, messages, **kwargs):
            pass

        add_request_hook(my_hook)
        remove_request_hook(my_hook)
        assert my_hook not in _request_hooks

    def test_remove_response_hook(self):
        def my_hook(model, message, duration_ms, **kwargs):
            pass

        add_response_hook(my_hook)
        remove_response_hook(my_hook)
        assert my_hook not in _response_hooks

    def test_remove_error_hook(self):
        def my_hook(model, error, duration_ms, **kwargs):
            pass

        add_error_hook(my_hook)
        remove_error_hook(my_hook)
        assert my_hook not in _error_hooks

    def test_clear_hooks(self):
        add_request_hook(lambda **k: None)
        add_response_hook(lambda **k: None)
        add_error_hook(lambda **k: None)

        clear_hooks()

        assert len(_request_hooks) == 0
        assert len(_response_hooks) == 0
        assert len(_error_hooks) == 0


class TestHookCalling:
    """Tests for hook invocation."""

    def test_request_hook_called_with_correct_args(self):
        mock_hook = MagicMock()
        add_request_hook(mock_hook)

        messages = [Message.user("Hello")]
        log_request("gpt-4o", messages, tools=None)

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == messages

    def test_response_hook_called_with_correct_args(self):
        mock_hook = MagicMock()
        add_response_hook(mock_hook)

        response = Message(
            role=Role.ASSISTANT,
            content="Hello!",
            model_id="gpt-4o",
            tokens=TokenUsage(input_tokens=10, output_tokens=5),
        )
        log_response("gpt-4o", response, 123.45)

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["message"] == response
        assert call_kwargs["duration_ms"] == 123.45

    def test_error_hook_called_with_correct_args(self):
        mock_hook = MagicMock()
        add_error_hook(mock_hook)

        error = ValueError("Test error")
        log_error("gpt-4o", error, 50.0)

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["error"] == error
        assert call_kwargs["duration_ms"] == 50.0

    def test_hook_error_does_not_propagate(self):
        """Hook errors should be logged but not raised."""

        def bad_hook(**kwargs):
            raise RuntimeError("Hook failed!")

        add_request_hook(bad_hook)

        # Should not raise
        log_request("gpt-4o", [Message.user("Hello")])

    def test_multiple_hooks_all_called(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        add_request_hook(mock1)
        add_request_hook(mock2)

        log_request("gpt-4o", [Message.user("Hello")])

        mock1.assert_called_once()
        mock2.assert_called_once()


class TestTimedRequest:
    """Tests for the timed_request context manager."""

    def test_timed_request_logs_request(self):
        mock_hook = MagicMock()
        add_request_hook(mock_hook)

        with timed_request("gpt-4o", [Message.user("Hello")]) as ctx:
            ctx["response"] = Message.assistant("Hi!", model_id="gpt-4o")

        mock_hook.assert_called_once()

    def test_timed_request_logs_response(self):
        mock_hook = MagicMock()
        add_response_hook(mock_hook)

        response = Message.assistant("Hi!", model_id="gpt-4o")
        with timed_request("gpt-4o", [Message.user("Hello")]) as ctx:
            ctx["response"] = response

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert call_kwargs["message"] == response
        assert call_kwargs["duration_ms"] >= 0

    def test_timed_request_logs_error_on_exception(self):
        mock_hook = MagicMock()
        add_error_hook(mock_hook)

        with pytest.raises(ValueError):
            with timed_request("gpt-4o", [Message.user("Hello")]):
                raise ValueError("Test error")

        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert isinstance(call_kwargs["error"], ValueError)
        assert call_kwargs["duration_ms"] >= 0


class TestLogging:
    """Tests for structured logging output."""

    def test_log_request_logs_info(self, caplog):
        with caplog.at_level("INFO", logger="django_llm"):
            log_request("gpt-4o", [Message.user("Hello")])

        assert "LLM request" in caplog.text
        assert "gpt-4o" in caplog.text

    def test_log_response_logs_info(self, caplog):
        response = Message(
            role=Role.ASSISTANT,
            content="Hello!",
            model_id="gpt-4o",
            tokens=TokenUsage(input_tokens=10, output_tokens=5),
        )

        with caplog.at_level("INFO", logger="django_llm"):
            log_response("gpt-4o", response, 123.45)

        assert "LLM response" in caplog.text
        assert "123ms" in caplog.text
        assert "gpt-4o" in caplog.text

    def test_log_response_includes_cost(self, caplog):
        response = Message(
            role=Role.ASSISTANT,
            content="Hello!",
            model_id="gpt-4o",
            tokens=TokenUsage(input_tokens=1000, output_tokens=500),
        )

        with caplog.at_level("INFO", logger="django_llm"):
            log_response("gpt-4o", response, 100.0)

        # Should include cost since gpt-4o has pricing
        assert "$" in caplog.text or "cost" in caplog.text.lower()

    def test_log_error_logs_error(self, caplog):
        error = ValueError("Test error")

        with caplog.at_level("ERROR", logger="django_llm"):
            log_error("gpt-4o", error, 50.0)

        assert "LLM error" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test error" in caplog.text


class TestExports:
    """Test that logging functions are properly exported."""

    def test_exports_from_package(self):
        import django_llm

        assert hasattr(django_llm, "add_request_hook")
        assert hasattr(django_llm, "add_response_hook")
        assert hasattr(django_llm, "add_error_hook")
        assert hasattr(django_llm, "remove_request_hook")
        assert hasattr(django_llm, "remove_response_hook")
        assert hasattr(django_llm, "remove_error_hook")
        assert hasattr(django_llm, "clear_hooks")
