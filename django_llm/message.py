"""Message representation for conversations."""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the assistant."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


@dataclass
class TokenUsage:
    """Token usage statistics for a response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, model: str) -> Decimal | None:
        """Calculate the cost of this token usage for a given model.

        Args:
            model: The model identifier.

        Returns:
            Cost in USD, or None if pricing is unknown for the model.
        """
        from django_llm.pricing import calculate_cost

        return calculate_cost(
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
        )


@dataclass
class Message:
    """A single message in a conversation.

    Represents messages from users, assistants, the system, or tool results.
    """
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    model_id: str | None = None
    tokens: TokenUsage | None = None
    thinking: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)
    parsed: Any = field(default=None, repr=False)  # Structured output result

    def __post_init__(self):
        # Convert string role to enum
        if isinstance(self.role, str):
            self.role = Role(self.role)
        # Convert dict tool_calls to ToolCall objects
        if self.tool_calls and isinstance(self.tool_calls[0], dict):
            self.tool_calls = [ToolCall.from_dict(tc) for tc in self.tool_calls]

    @property
    def is_tool_call(self) -> bool:
        """Check if this message contains tool calls."""
        return bool(self.tool_calls)

    @property
    def is_tool_result(self) -> bool:
        """Check if this message is a tool result."""
        return self.tool_call_id is not None

    @property
    def cost(self) -> Decimal | None:
        """Calculate the cost of this message in USD.

        Returns:
            Cost in USD, or None if model/tokens are not available
            or pricing is unknown.

        Example:
            >>> response = chat.ask("Hello")
            >>> print(f"Cost: ${response.cost:.6f}")
        """
        if not self.tokens or not self.model_id:
            return None
        return self.tokens.cost(self.model_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        data = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.model_id:
            data["model_id"] = self.model_id
        if self.tokens:
            data["tokens"] = {
                "input_tokens": self.tokens.input_tokens,
                "output_tokens": self.tokens.output_tokens,
                "cached_tokens": self.tokens.cached_tokens,
                "reasoning_tokens": self.tokens.reasoning_tokens,
            }
        if self.thinking:
            data["thinking"] = self.thinking
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        tokens = None
        if "tokens" in data:
            tokens = TokenUsage(**data["tokens"])

        return cls(
            role=Role(data["role"]),
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            model_id=data.get("model_id"),
            tokens=tokens,
            thinking=data.get("thinking"),
        )

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, **kwargs) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        """Create a tool result message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)
