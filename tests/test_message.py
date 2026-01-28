"""Tests for the Message class."""

from django_llm.message import Message, Role, TokenUsage, ToolCall


class TestMessage:
    """Tests for Message creation and serialization."""

    def test_create_user_message(self):
        """Test creating a simple user message."""
        msg = Message.user("Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"
        assert not msg.is_tool_call
        assert not msg.is_tool_result

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_create_tool_result_message(self):
        """Test creating a tool result message."""
        msg = Message.tool_result("call_123", "The weather is sunny.")
        assert msg.role == Role.TOOL
        assert msg.content == "The weather is sunny."
        assert msg.tool_call_id == "call_123"
        assert msg.is_tool_result

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"}),
        ]
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=tool_calls)
        assert msg.is_tool_call
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"

    def test_message_serialization(self):
        """Test converting message to dict."""
        msg = Message.user("Test message")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"

    def test_message_deserialization(self):
        """Test creating message from dict."""
        data = {
            "role": "assistant",
            "content": "Response",
            "model_id": "gpt-4o",
            "tokens": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            },
        }
        msg = Message.from_dict(data)

        assert msg.role == Role.ASSISTANT
        assert msg.content == "Response"
        assert msg.model_id == "gpt-4o"
        assert msg.tokens.input_tokens == 10
        assert msg.tokens.output_tokens == 20

    def test_role_from_string(self):
        """Test that role can be created from string."""
        msg = Message(role="user", content="Hello")
        assert msg.role == Role.USER


class TestTokenUsage:
    """Tests for TokenUsage."""

    def test_total_tokens(self):
        """Test total token calculation."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        assert tokens.total_tokens == 150


class TestToolCall:
    """Tests for ToolCall."""

    def test_tool_call_serialization(self):
        """Test converting tool call to dict."""
        tc = ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"})
        data = tc.to_dict()

        assert data["id"] == "call_1"
        assert data["name"] == "get_weather"
        assert data["arguments"] == {"location": "NYC"}

    def test_tool_call_deserialization(self):
        """Test creating tool call from dict."""
        data = {"id": "call_2", "name": "calculate", "arguments": {"x": 5, "y": 10}}
        tc = ToolCall.from_dict(data)

        assert tc.id == "call_2"
        assert tc.name == "calculate"
        assert tc.arguments["x"] == 5
