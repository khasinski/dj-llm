"""Tests for the Chat class."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from django_llm.chat import Chat
from django_llm.message import Message, Role
from django_llm.tool import Tool


class MockTool(Tool):
    """Mock tool for testing."""
    name = "mock_tool"
    description = "A mock tool for testing"

    def execute(self, value: str) -> str:
        return f"Mocked: {value}"


class TestChat:
    """Tests for Chat class."""

    def test_chat_creation(self):
        """Test creating a new chat."""
        chat = Chat()
        assert len(chat.messages) == 0

    def test_chat_with_system_prompt(self):
        """Test creating chat with system prompt."""
        chat = Chat(system="You are a helpful assistant.")
        assert len(chat.messages) == 1
        assert chat.messages[0].role == Role.SYSTEM
        assert chat.messages[0].content == "You are a helpful assistant."

    def test_chat_with_model(self):
        """Test setting model via constructor."""
        chat = Chat(model="gpt-4o")
        assert chat.model == "gpt-4o"

    def test_fluent_model_setting(self):
        """Test fluent interface for setting model."""
        chat = Chat().with_model("claude-sonnet-4-20250514")
        assert chat.model == "claude-sonnet-4-20250514"

    def test_fluent_temperature_setting(self):
        """Test fluent interface for setting temperature."""
        chat = Chat().with_temperature(0.7)
        assert chat._temperature == 0.7

    def test_fluent_instructions_setting(self):
        """Test fluent interface for setting instructions."""
        chat = Chat().with_instructions("Be concise.")
        assert len(chat.messages) == 1
        assert chat.messages[0].content == "Be concise."

    def test_fluent_instructions_replaces_existing(self):
        """Test that with_instructions replaces existing system message."""
        chat = Chat(system="Old instructions")
        chat.with_instructions("New instructions")
        assert len(chat.messages) == 1
        assert chat.messages[0].content == "New instructions"

    def test_fluent_tool_setting(self):
        """Test fluent interface for adding tools."""
        tool = MockTool()
        chat = Chat().with_tool(tool)
        assert len(chat._tools) == 1
        assert chat._tools[0] is tool

    def test_fluent_tools_setting(self):
        """Test fluent interface for setting multiple tools."""
        tools = [MockTool()]
        chat = Chat().with_tools(tools)
        assert len(chat._tools) == 1

    def test_add_message(self):
        """Test adding a message."""
        chat = Chat()
        chat.add_message(Message.user("Hello"))
        assert len(chat.messages) == 1

    def test_add_message_from_dict(self):
        """Test adding a message from dict."""
        chat = Chat()
        chat.add_message({"role": "user", "content": "Hello"})
        assert len(chat.messages) == 1
        assert chat.messages[0].role == Role.USER

    def test_reset_preserves_system(self):
        """Test that reset preserves system message."""
        chat = Chat(system="System prompt")
        chat.add_message(Message.user("Hello"))
        chat.reset()
        assert len(chat.messages) == 1
        assert chat.messages[0].role == Role.SYSTEM

    def test_clear_removes_all(self):
        """Test that clear removes all messages."""
        chat = Chat(system="System prompt")
        chat.add_message(Message.user("Hello"))
        chat.clear()
        assert len(chat.messages) == 0

    def test_iteration(self):
        """Test iterating over messages."""
        chat = Chat()
        chat.add_message(Message.user("One"))
        chat.add_message(Message.user("Two"))

        messages = list(chat)
        assert len(messages) == 2

    def test_len(self):
        """Test getting message count."""
        chat = Chat()
        chat.add_message(Message.user("One"))
        chat.add_message(Message.user("Two"))
        assert len(chat) == 2

    def test_on_new_message_callback(self):
        """Test on_new_message callback is called."""
        callback = Mock()
        chat = Chat().on_new_message(callback)
        chat.add_message(Message.user("Hello"))

        callback.assert_called_once()
        assert callback.call_args[0][0].content == "Hello"

    def test_on_chunk_callback_set(self):
        """Test on_chunk callback is set."""
        callback = Mock()
        chat = Chat().on_chunk(callback)
        assert chat._on_chunk is callback

    @patch("django_llm.chat.get_provider_for_model")
    def test_ask_adds_user_message(self, mock_get_provider):
        """Test that ask adds user message to history."""
        mock_provider = Mock()
        mock_response = Message.assistant("Response")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        chat.ask("Hello")

        # Should have user message and assistant response
        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[0].content == "Hello"
        assert chat.messages[1].role == Role.ASSISTANT

    @patch("django_llm.chat.get_provider_for_model")
    def test_ask_returns_response(self, mock_get_provider):
        """Test that ask returns the response message."""
        mock_provider = Mock()
        mock_response = Message.assistant("Hello there!")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        response = chat.ask("Hi")

        assert response.content == "Hello there!"
        assert response.role == Role.ASSISTANT

    @patch("django_llm.chat.get_provider_for_model")
    def test_say_alias(self, mock_get_provider):
        """Test that say is an alias for ask."""
        mock_provider = Mock()
        mock_response = Message.assistant("Response")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        response = chat.say("Hello")

        assert response.content == "Response"

    @patch("django_llm.chat.get_provider_for_model")
    def test_method_chaining(self, mock_get_provider):
        """Test that fluent methods return self for chaining."""
        mock_provider = Mock()
        mock_response = Message.assistant("Response")
        mock_provider.complete.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        result = (
            Chat()
            .with_model("gpt-4o")
            .with_temperature(0.7)
            .with_instructions("Be helpful")
            .ask("Hello")
        )

        assert result.content == "Response"

    def test_with_schema_sets_schema(self):
        """Test that with_schema sets the schema."""
        from dataclasses import dataclass

        @dataclass
        class Person:
            name: str
            age: int

        chat = Chat().with_schema(Person)
        assert chat._schema is Person


class TestAsyncChat:
    """Tests for async Chat methods."""

    @pytest.mark.asyncio
    @patch("django_llm.chat.get_provider_for_model")
    async def test_aask_adds_user_message(self, mock_get_provider):
        """Test that aask adds user message to history."""
        mock_provider = MagicMock()
        mock_response = Message.assistant("Async response")

        async def mock_acomplete(*args, **kwargs):
            return mock_response

        mock_provider.acomplete = mock_acomplete
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        await chat.aask("Hello async")

        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[0].content == "Hello async"
        assert chat.messages[1].role == Role.ASSISTANT

    @pytest.mark.asyncio
    @patch("django_llm.chat.get_provider_for_model")
    async def test_aask_returns_response(self, mock_get_provider):
        """Test that aask returns the response message."""
        mock_provider = MagicMock()
        mock_response = Message.assistant("Async hello!")

        async def mock_acomplete(*args, **kwargs):
            return mock_response

        mock_provider.acomplete = mock_acomplete
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        response = await chat.aask("Hi")

        assert response.content == "Async hello!"
        assert response.role == Role.ASSISTANT

    @pytest.mark.asyncio
    @patch("django_llm.chat.get_provider_for_model")
    async def test_astream_yields_chunks(self, mock_get_provider):
        """Test that astream yields chunks."""
        mock_provider = MagicMock()

        async def mock_astream(*args, **kwargs):
            for chunk in ["Hello", " ", "world", "!"]:
                yield chunk

        mock_provider.astream = mock_astream
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        chunks = []
        async for chunk in chat.astream("Say hello"):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world", "!"]
        # Should have user message and accumulated assistant response
        assert len(chat.messages) == 2
        assert chat.messages[1].content == "Hello world!"

    @pytest.mark.asyncio
    @patch("django_llm.chat.get_provider_for_model")
    async def test_asay_alias(self, mock_get_provider):
        """Test that asay is an alias for aask."""
        mock_provider = MagicMock()
        mock_response = Message.assistant("Async response")

        async def mock_acomplete(*args, **kwargs):
            return mock_response

        mock_provider.acomplete = mock_acomplete
        mock_get_provider.return_value = mock_provider

        chat = Chat(model="gpt-4o")
        response = await chat.asay("Hello")

        assert response.content == "Async response"
