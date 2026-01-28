"""Tests for Ollama provider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from django_llm.message import Message, Role, TokenUsage, ToolCall
from django_llm.providers.ollama import OllamaProvider
from django_llm.tool import FunctionTool, Tool


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_provider_name(self):
        provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_default_base_url(self):
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434/v1"

    def test_custom_base_url(self):
        provider = OllamaProvider(base_url="http://my-server:11434/v1")
        assert provider.base_url == "http://my-server:11434/v1"

    def test_base_url_strips_trailing_slash(self):
        provider = OllamaProvider(base_url="http://my-server:11434/v1/")
        assert provider.base_url == "http://my-server:11434/v1"

    def test_no_api_key_required(self):
        """Ollama should work without an API key."""
        provider = OllamaProvider()
        headers = provider._get_headers()
        assert "Authorization" not in headers

    def test_longer_default_timeout(self):
        """Ollama should have a longer default timeout for local inference."""
        provider = OllamaProvider()
        assert provider.timeout == 300.0

    def test_format_messages(self):
        provider = OllamaProvider()
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        formatted = provider._format_messages(messages)

        assert len(formatted) == 3
        assert formatted[0] == {"role": "system", "content": "You are helpful."}
        assert formatted[1] == {"role": "user", "content": "Hello"}
        assert formatted[2] == {"role": "assistant", "content": "Hi there!"}

    def test_format_tool_result_message(self):
        provider = OllamaProvider()
        messages = [
            Message.tool_result("call_123", '{"result": "42"}'),
        ]

        formatted = provider._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "tool"
        assert formatted[0]["content"] == '{"result": "42"}'
        assert formatted[0]["tool_call_id"] == "call_123"

    def test_format_tools(self):
        provider = OllamaProvider()

        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: Sunny"

        tool = FunctionTool(get_weather)
        formatted = provider.format_tools([tool])

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"
        assert formatted[0]["function"]["description"] == "Get the current weather for a location."

    def test_parse_response(self):
        provider = OllamaProvider()
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }

        message = provider._parse_response(data, "llama3.2")

        assert message.role == Role.ASSISTANT
        assert message.content == "Hello!"
        assert message.model_id == "ollama:llama3.2"
        assert message.tokens.input_tokens == 10
        assert message.tokens.output_tokens == 5

    def test_parse_response_with_tool_calls(self):
        provider = OllamaProvider()
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }

        message = provider._parse_response(data, "llama3.2")

        assert message.is_tool_call
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "get_weather"
        assert message.tool_calls[0].arguments == {"location": "Paris"}

    def test_build_payload_strips_ollama_prefix(self):
        provider = OllamaProvider()
        messages = [Message.user("Hello")]

        payload = provider._build_payload(
            messages=messages,
            model="ollama:llama3.2",
            temperature=None,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "llama3.2"

    def test_build_payload_no_prefix(self):
        provider = OllamaProvider()
        messages = [Message.user("Hello")]

        payload = provider._build_payload(
            messages=messages,
            model="llama3.2",
            temperature=None,
            tools=None,
            stream=False,
        )

        assert payload["model"] == "llama3.2"


class TestOllamaModelDetection:
    """Tests for Ollama model detection in registry."""

    def test_ollama_prefix_detection(self):
        from django_llm.providers.registry import get_provider_name_for_model

        assert get_provider_name_for_model("ollama:llama3.2") == "ollama"
        assert get_provider_name_for_model("ollama:mistral") == "ollama"
        assert get_provider_name_for_model("ollama:codellama") == "ollama"


class TestOllamaIntegration:
    """Integration tests for Ollama (mocked)."""

    def test_complete_mocked(self):
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm Llama, running locally!",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            result = provider.complete(
                messages=[Message.user("Who are you?")],
                model="llama3.2",
            )

        assert result.content == "I'm Llama, running locally!"
        assert result.model_id == "ollama:llama3.2"

    def test_error_handling_model_not_found(self):
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_response.json.return_value = {"error": "model not found"}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                provider.complete(
                    messages=[Message.user("Hello")],
                    model="nonexistent-model",
                )

            assert "ollama pull" in str(exc_info.value).lower()
