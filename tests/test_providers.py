"""Tests for LLM providers."""

import pytest

from django_llm.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
)
from django_llm.message import Message, Role
from django_llm.providers.anthropic import AnthropicProvider
from django_llm.providers.google import GoogleProvider
from django_llm.providers.openai import OpenAIProvider
from django_llm.providers.registry import get_provider, get_provider_name_for_model
from django_llm.tool import Tool


class MockTool(Tool):
    """Mock tool for testing."""
    name = "get_weather"
    description = "Get weather for a location"

    def execute(self, location: str) -> str:
        return f"Weather in {location}: Sunny"


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_openai_model_detection(self):
        """Test that OpenAI models are correctly identified."""
        assert get_provider_name_for_model("gpt-4o") == "openai"
        assert get_provider_name_for_model("gpt-3.5-turbo") == "openai"
        assert get_provider_name_for_model("o1-preview") == "openai"

    def test_anthropic_model_detection(self):
        """Test that Anthropic models are correctly identified."""
        assert get_provider_name_for_model("claude-sonnet-4-20250514") == "anthropic"
        assert get_provider_name_for_model("claude-3-opus-20240229") == "anthropic"

    def test_google_model_detection(self):
        """Test that Google models are correctly identified."""
        assert get_provider_name_for_model("gemini-2.0-flash") == "google"
        assert get_provider_name_for_model("gemini-pro") == "google"

    def test_unknown_model_raises_error(self):
        """Test that unknown models raise ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            get_provider_name_for_model("unknown-model")

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        from django_llm.configuration import _config

        # Save original and set test key
        original_key = _config.openai_api_key
        _config.openai_api_key = "test-key"

        try:
            provider = get_provider("openai")
            assert isinstance(provider, OpenAIProvider)
            assert provider.api_key == "test-key"
        finally:
            _config.openai_api_key = original_key

    def test_get_provider_without_key_raises_error(self):
        """Test that missing API key raises error."""
        from django_llm.configuration import _config

        # Save original and clear key
        original_key = _config.openai_api_key
        _config.openai_api_key = None

        try:
            with pytest.raises(AuthenticationError):
                get_provider("openai")
        finally:
            _config.openai_api_key = original_key


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_format_messages(self):
        """Test message formatting for OpenAI API."""
        provider = OpenAIProvider(api_key="test")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        formatted = provider._format_messages(messages)

        assert len(formatted) == 3
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"

    def test_format_tool_result_message(self):
        """Test formatting tool result messages."""
        provider = OpenAIProvider(api_key="test")
        messages = [
            Message.tool_result("call_123", "Result data"),
        ]

        formatted = provider._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "tool"
        assert formatted[0]["tool_call_id"] == "call_123"
        assert formatted[0]["content"] == "Result data"

    def test_format_tools(self):
        """Test tool formatting for OpenAI API."""
        provider = OpenAIProvider(api_key="test")
        tools = [MockTool()]

        formatted = provider.format_tools(tools)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"
        assert "parameters" in formatted[0]["function"]

    def test_parse_response(self):
        """Test parsing OpenAI response."""
        provider = OpenAIProvider(api_key="test")
        data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello!",
                        "role": "assistant",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }

        message = provider._parse_response(data, "gpt-4o")

        assert message.role == Role.ASSISTANT
        assert message.content == "Hello!"
        assert message.tokens.input_tokens == 10
        assert message.tokens.output_tokens == 5

    def test_parse_response_with_tool_calls(self):
        """Test parsing response with tool calls."""
        provider = OpenAIProvider(api_key="test")
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }

        message = provider._parse_response(data, "gpt-4o")

        assert message.is_tool_call
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "get_weather"
        assert message.tool_calls[0].arguments == {"location": "NYC"}


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_format_messages_separates_system(self):
        """Test that system message is separated."""
        provider = AnthropicProvider(api_key="test")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]

        system, formatted = provider._format_messages(messages)

        assert system == "You are helpful."
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"

    def test_format_tool_result_message(self):
        """Test formatting tool result messages for Anthropic."""
        provider = AnthropicProvider(api_key="test")
        messages = [
            Message.tool_result("call_123", "Result data"),
        ]

        _, formatted = provider._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"][0]["type"] == "tool_result"

    def test_format_tools(self):
        """Test tool formatting for Anthropic API."""
        provider = AnthropicProvider(api_key="test")
        tools = [MockTool()]

        formatted = provider.format_tools(tools)

        assert len(formatted) == 1
        assert formatted[0]["name"] == "get_weather"
        assert "input_schema" in formatted[0]

    def test_parse_response(self):
        """Test parsing Anthropic response."""
        provider = AnthropicProvider(api_key="test")
        data = {
            "content": [
                {"type": "text", "text": "Hello!"},
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
            },
        }

        message = provider._parse_response(data, "claude-sonnet-4-20250514")

        assert message.role == Role.ASSISTANT
        assert message.content == "Hello!"
        assert message.tokens.input_tokens == 10

    def test_parse_response_with_tool_use(self):
        """Test parsing response with tool use."""
        provider = AnthropicProvider(api_key="test")
        data = {
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                },
            ],
        }

        message = provider._parse_response(data, "claude-sonnet-4-20250514")

        assert message.content == "Let me check the weather."
        assert message.is_tool_call
        assert message.tool_calls[0].name == "get_weather"


class TestGoogleProvider:
    """Tests for Google Gemini provider."""

    def test_format_messages(self):
        """Test message formatting for Gemini API."""
        provider = GoogleProvider(api_key="test")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        system, formatted = provider._format_messages(messages)

        assert system is None
        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "model"

    def test_format_messages_with_system(self):
        """Test that system message is extracted."""
        provider = GoogleProvider(api_key="test")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]

        system, formatted = provider._format_messages(messages)

        assert system == "You are helpful."
        assert len(formatted) == 1

    def test_format_tools(self):
        """Test tool formatting for Gemini API."""
        provider = GoogleProvider(api_key="test")
        tools = [MockTool()]

        formatted = provider.format_tools(tools)

        assert len(formatted) == 1
        assert "functionDeclarations" in formatted[0]
        assert formatted[0]["functionDeclarations"][0]["name"] == "get_weather"

    def test_parse_response(self):
        """Test parsing Gemini response."""
        provider = GoogleProvider(api_key="test")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello!"}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }

        message = provider._parse_response(data, "gemini-2.0-flash")

        assert message.role == Role.ASSISTANT
        assert message.content == "Hello!"
        assert message.tokens.input_tokens == 10

    def test_parse_response_with_function_call(self):
        """Test parsing response with function call."""
        provider = GoogleProvider(api_key="test")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ],
        }

        message = provider._parse_response(data, "gemini-2.0-flash")

        assert message.is_tool_call
        assert message.tool_calls[0].name == "get_weather"
