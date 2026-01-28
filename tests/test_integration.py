"""Integration tests with real API calls using VCR."""

import os

import pytest
import vcr

import django_llm
from django_llm import Chat, configure
from django_llm.message import Role
from django_llm.tool import Tool

# Custom VCR configuration
my_vcr = vcr.VCR(
    cassette_library_dir="tests/cassettes",
    record_mode="once",
    match_on=["method", "scheme", "host", "port", "path", "body"],
    filter_headers=[
        "authorization",
        "x-api-key",
        "api-key",
    ],
    filter_query_parameters=["key"],
    decode_compressed_response=True,
)


class WeatherTool(Tool):
    """Example tool for testing function calling."""

    name = "get_weather"
    description = "Get the current weather for a location"

    def execute(self, location: str) -> str:
        # Mock weather response
        return f"The weather in {location} is sunny with a temperature of 72°F"


def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_google_key():
    """Check if Google API key is available."""
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @my_vcr.use_cassette("openai_basic_chat.yaml")
    def test_basic_chat(self):
        """Test basic chat with OpenAI."""
        chat = Chat(model="gpt-4o-mini")
        response = chat.ask("Say 'Hello, World!' and nothing else.")

        assert response.content is not None
        assert "Hello" in response.content or "hello" in response.content.lower()
        assert response.role == Role.ASSISTANT

    @my_vcr.use_cassette("openai_with_system.yaml")
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        chat = Chat(model="gpt-4o-mini", system="You always respond in exactly 3 words.")
        response = chat.ask("What is Python?")

        assert response.content is not None
        # Should be a short response due to system prompt
        assert len(response.content.split()) <= 10

    @my_vcr.use_cassette("openai_multi_turn.yaml")
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        chat = Chat(model="gpt-4o-mini")

        response1 = chat.ask("My name is Alice.")
        assert response1.content is not None

        response2 = chat.ask("What is my name?")
        assert response2.content is not None
        assert "Alice" in response2.content

    @my_vcr.use_cassette("openai_streaming.yaml")
    def test_streaming(self):
        """Test streaming response."""
        chat = Chat(model="gpt-4o-mini")
        chunks = []

        for chunk in chat.stream("Count from 1 to 5."):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) > 1  # Should have multiple chunks
        assert "1" in full_response
        assert "5" in full_response

    @my_vcr.use_cassette("openai_tool_calling.yaml")
    def test_tool_calling(self):
        """Test tool/function calling."""
        chat = Chat(model="gpt-4o-mini")
        chat.with_tools([WeatherTool()])

        response = chat.ask("What's the weather in San Francisco?")

        assert response.content is not None
        # The response should include weather information
        # (either from tool result or mentioning weather)

    @my_vcr.use_cassette("openai_tokens.yaml")
    def test_token_usage(self):
        """Test that token usage is returned."""
        chat = Chat(model="gpt-4o-mini")
        response = chat.ask("Hi")

        assert response.tokens is not None
        assert response.tokens.input_tokens > 0
        assert response.tokens.output_tokens > 0


@pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicIntegration:
    """Integration tests for Anthropic provider."""

    @my_vcr.use_cassette("anthropic_basic_chat.yaml")
    def test_basic_chat(self):
        """Test basic chat with Anthropic."""
        chat = Chat(model="claude-sonnet-4-20250514")
        response = chat.ask("Say 'Hello, World!' and nothing else.")

        assert response.content is not None
        assert "Hello" in response.content or "hello" in response.content.lower()

    @my_vcr.use_cassette("anthropic_with_system.yaml")
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        chat = Chat(model="claude-sonnet-4-20250514", system="You always respond in exactly 3 words.")
        response = chat.ask("What is Python?")

        assert response.content is not None

    @my_vcr.use_cassette("anthropic_streaming.yaml")
    def test_streaming(self):
        """Test streaming response."""
        chat = Chat(model="claude-sonnet-4-20250514")
        chunks = []

        for chunk in chat.stream("Count from 1 to 5."):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) > 1
        assert "1" in full_response

    @my_vcr.use_cassette("anthropic_tool_calling.yaml")
    def test_tool_calling(self):
        """Test tool/function calling."""
        chat = Chat(model="claude-sonnet-4-20250514")
        chat.with_tools([WeatherTool()])

        response = chat.ask("What's the weather in San Francisco?")
        assert response.content is not None


@pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set")
class TestGoogleIntegration:
    """Integration tests for Google Gemini provider."""

    @my_vcr.use_cassette("google_basic_chat.yaml")
    def test_basic_chat(self):
        """Test basic chat with Gemini."""
        chat = Chat(model="gemini-2.0-flash")
        response = chat.ask("Say 'Hello, World!' and nothing else.")

        assert response.content is not None
        assert "Hello" in response.content or "hello" in response.content.lower()

    @my_vcr.use_cassette("google_with_system.yaml")
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        chat = Chat(model="gemini-2.0-flash", system="You always respond in exactly 3 words.")
        response = chat.ask("What is Python?")

        assert response.content is not None

    @my_vcr.use_cassette("google_streaming.yaml")
    def test_streaming(self):
        """Test streaming response."""
        chat = Chat(model="gemini-2.0-flash")
        chunks = []

        for chunk in chat.stream("Count from 1 to 5."):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) >= 1
        assert "1" in full_response

    @my_vcr.use_cassette("google_tool_calling.yaml")
    def test_tool_calling(self):
        """Test tool/function calling."""
        chat = Chat(model="gemini-2.0-flash")
        chat.with_tools([WeatherTool()])

        response = chat.ask("What's the weather in San Francisco?")
        assert response.content is not None


@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
class TestAsyncIntegration:
    """Async integration tests for OpenAI provider."""

    @my_vcr.use_cassette("openai_async_basic.yaml")
    @pytest.mark.asyncio
    async def test_async_ask(self):
        """Test async ask with OpenAI."""
        chat = Chat(model="gpt-4o-mini")
        response = await chat.aask("Say 'async works' and nothing else.")

        assert response.content is not None
        assert response.role == Role.ASSISTANT

    @my_vcr.use_cassette("openai_async_stream.yaml")
    @pytest.mark.asyncio
    async def test_async_stream(self):
        """Test async streaming response."""
        chat = Chat(model="gpt-4o-mini")
        chunks = []

        async for chunk in chat.astream("Count from 1 to 3."):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) > 1
        assert "1" in full_response


@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
class TestStructuredOutputIntegration:
    """Integration tests for structured output."""

    @my_vcr.use_cassette("openai_structured_output.yaml")
    def test_structured_output_dataclass(self):
        """Test structured output with dataclass."""
        from dataclasses import dataclass

        @dataclass
        class Person:
            name: str
            age: int
            occupation: str

        chat = Chat(model="gpt-4o-mini").with_schema(Person)
        response = chat.ask("Extract: Jane Doe is a 28 year old data scientist.")

        assert response.parsed is not None
        assert isinstance(response.parsed, Person)
        assert response.parsed.name == "Jane Doe"
        assert response.parsed.age == 28
        assert response.parsed.occupation == "data scientist"

    @my_vcr.use_cassette("openai_structured_output_list.yaml")
    def test_structured_output_with_list(self):
        """Test structured output with list field."""
        from dataclasses import dataclass

        @dataclass
        class Tags:
            items: list[str]

        chat = Chat(model="gpt-4o-mini").with_schema(Tags)
        response = chat.ask("List 3 programming languages: Python, JavaScript, Rust")

        assert response.parsed is not None
        assert isinstance(response.parsed, Tags)
        assert len(response.parsed.items) == 3


class TestUnifiedInterface:
    """Tests for the unified interface across providers."""

    @pytest.mark.skipif(
        not (has_openai_key() or has_anthropic_key() or has_google_key()),
        reason="No API keys available",
    )
    def test_django_llm_chat_shortcut(self):
        """Test the django_llm.chat() shortcut."""
        chat = django_llm.chat()
        assert chat is not None

    def test_configuration(self):
        """Test configuration."""
        configure(timeout=60.0)
        from django_llm.configuration import get_config

        assert get_config().timeout == 60.0

        # Reset
        configure(timeout=120.0)

    def test_fluent_interface(self):
        """Test the fluent interface works across all methods."""
        chat = (
            Chat()
            .with_model("gpt-4o")
            .with_temperature(0.7)
            .with_instructions("Be helpful")
            .with_tools([WeatherTool()])
        )

        assert chat.model == "gpt-4o"
        assert chat._temperature == 0.7
        assert len(chat._tools) == 1
        assert len(chat.messages) == 1  # System message
