"""Tests for Azure OpenAI provider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from django_llm.message import Message, Role, TokenUsage, ToolCall
from django_llm.providers.azure_openai import AzureOpenAIProvider
from django_llm.tool import FunctionTool


class TestAzureOpenAIProvider:
    """Tests for AzureOpenAIProvider."""

    def test_provider_name(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        assert provider.name == "azure_openai"

    def test_endpoint_configuration(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        assert provider.endpoint == "https://my-resource.openai.azure.com"

    def test_endpoint_strips_trailing_slash(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com/"
        )
        assert provider.endpoint == "https://my-resource.openai.azure.com"

    def test_default_api_version(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        assert provider.api_version == "2024-08-01-preview"

    def test_custom_api_version(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com",
            api_version="2023-12-01-preview"
        )
        assert provider.api_version == "2023-12-01-preview"

    def test_headers_use_api_key_header(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        headers = provider._get_headers()
        assert headers["api-key"] == "test-key"
        assert "Authorization" not in headers

    def test_get_url(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        url = provider._get_url("my-deployment")
        assert "my-resource.openai.azure.com" in url
        assert "deployments/my-deployment" in url
        assert "api-version=2024-08-01-preview" in url

    def test_get_deployment_with_prefix(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        deployment = provider._get_deployment("azure:my-gpt4")
        assert deployment == "my-gpt4"

    def test_get_deployment_without_prefix(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        deployment = provider._get_deployment("my-gpt4")
        assert deployment == "my-gpt4"

    def test_format_messages(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
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
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
        messages = [
            Message.tool_result("call_123", '{"result": "42"}'),
        ]

        formatted = provider._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "tool"
        assert formatted[0]["content"] == '{"result": "42"}'
        assert formatted[0]["tool_call_id"] == "call_123"

    def test_format_tools(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )

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
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
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

        message = provider._parse_response(data, "my-gpt4")

        assert message.role == Role.ASSISTANT
        assert message.content == "Hello!"
        assert message.model_id == "azure:my-gpt4"
        assert message.tokens.input_tokens == 10
        assert message.tokens.output_tokens == 5

    def test_parse_response_with_tool_calls(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )
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

        message = provider._parse_response(data, "my-gpt4")

        assert message.is_tool_call
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "get_weather"
        assert message.tool_calls[0].arguments == {"location": "Paris"}


class TestAzureOpenAIModelDetection:
    """Tests for Azure OpenAI model detection in registry."""

    def test_azure_prefix_detection(self):
        from django_llm.providers.registry import get_provider_name_for_model

        assert get_provider_name_for_model("azure:my-gpt4") == "azure_openai"
        assert get_provider_name_for_model("azure:gpt-4o-deployment") == "azure_openai"


class TestAzureOpenAIIntegration:
    """Integration tests for Azure OpenAI (mocked)."""

    def test_complete_mocked(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Azure!",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            result = provider.complete(
                messages=[Message.user("Hello")],
                model="azure:my-gpt4",
            )

        assert result.content == "Hello from Azure!"
        assert result.model_id == "azure:my-gpt4"

    def test_error_handling_deployment_not_found(self):
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com"
        )

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "deployment not found"
        mock_response.json.return_value = {"error": {"message": "deployment not found"}}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                provider.complete(
                    messages=[Message.user("Hello")],
                    model="nonexistent-deployment",
                )

            assert "deployment" in str(exc_info.value).lower()
