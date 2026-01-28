"""Chat interface - the main API for conversations."""

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Self, TypeVar

from django_llm.configuration import get_config
from django_llm.logging import async_timed_request, timed_request
from django_llm.message import Message, Role
from django_llm.providers.registry import get_provider_for_model
from django_llm.schema import parse_json_response, schema_to_json_schema
from django_llm.tool import Tool

T = TypeVar("T")


class Chat:
    """A conversation with an LLM.

    The Chat class provides a fluent interface for building and managing
    conversations with various LLM providers.

    Example:
        >>> chat = Chat()
        >>> response = chat.ask("What is Python?")
        >>> print(response.content)

        # With streaming
        >>> for chunk in chat.stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)

        # With tools
        >>> chat.with_tools([my_tool]).ask("Use the tool")

        # Chained configuration
        >>> chat.with_model("gpt-4o").with_temperature(0.7).ask("Be creative")
    """

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        messages: list[Message] | None = None,
    ):
        """Initialize a new chat.

        Args:
            model: Model identifier. If None, uses config default.
            system: Optional system prompt.
            messages: Optional initial messages (for restoring from persistence).
        """
        self._model = model
        self._temperature: float | None = None
        self._tools: list[Tool] = []
        self._messages: list[Message] = messages or []
        self._schema: type | dict[str, Any] | None = None

        # Event callbacks
        self._on_new_message: Callable[[Message], None] | None = None
        self._on_chunk: Callable[[str], None] | None = None

        # Add system message if provided
        if system and not any(m.role == Role.SYSTEM for m in self._messages):
            self._messages.insert(0, Message.system(system))

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        if self._model:
            return self._model
        return get_config().get_default_model()

    @property
    def messages(self) -> list[Message]:
        """Get all messages in the conversation."""
        return self._messages.copy()

    def __iter__(self):
        """Iterate over messages in the conversation."""
        return iter(self._messages)

    def __len__(self):
        """Get the number of messages in the conversation."""
        return len(self._messages)

    # Fluent configuration methods

    def with_model(self, model: str) -> Self:
        """Set the model for this chat.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514').

        Returns:
            Self for method chaining.
        """
        self._model = model
        return self

    def with_temperature(self, temperature: float) -> Self:
        """Set the sampling temperature.

        Args:
            temperature: Temperature value (0.0-2.0).

        Returns:
            Self for method chaining.
        """
        self._temperature = temperature
        return self

    def with_instructions(self, instructions: str) -> Self:
        """Set or update the system instructions.

        Args:
            instructions: The system prompt.

        Returns:
            Self for method chaining.
        """
        # Remove existing system message if any
        self._messages = [m for m in self._messages if m.role != Role.SYSTEM]
        # Add new system message at the start
        self._messages.insert(0, Message.system(instructions))
        return self

    def with_tools(self, tools: list[Tool]) -> Self:
        """Add tools that the model can call.

        Args:
            tools: List of Tool instances.

        Returns:
            Self for method chaining.
        """
        self._tools = tools
        return self

    def with_tool(self, tool: Tool) -> Self:
        """Add a single tool.

        Args:
            tool: A Tool instance.

        Returns:
            Self for method chaining.
        """
        self._tools.append(tool)
        return self

    def with_schema(self, schema: type[T] | dict[str, Any]) -> Self:
        """Set a schema for structured output.

        The model will be instructed to return JSON matching the schema.
        Use message.parsed to get the parsed result.

        Args:
            schema: A dataclass, Pydantic model, or JSON schema dict.

        Returns:
            Self for method chaining.

        Example:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            ...
            >>> chat = Chat().with_schema(Person)
            >>> response = chat.ask("Extract: John is 30 years old")
            >>> person = response.parsed  # Person(name='John', age=30)
        """
        self._schema = schema
        return self

    def on_new_message(self, callback: Callable[[Message], None]) -> Self:
        """Set callback for new messages.

        Args:
            callback: Function called when a new message is added.

        Returns:
            Self for method chaining.
        """
        self._on_new_message = callback
        return self

    def on_chunk(self, callback: Callable[[str], None]) -> Self:
        """Set callback for streaming chunks.

        Args:
            callback: Function called for each streamed chunk.

        Returns:
            Self for method chaining.
        """
        self._on_chunk = callback
        return self

    # Core methods

    def ask(self, content: str, stream: bool = False) -> Message:
        """Send a message and get a response.

        Args:
            content: The user's message.
            stream: If True, streams the response (use on_chunk callback).

        Returns:
            The assistant's response message.
        """
        # Add user message
        user_msg = Message.user(content)
        self._add_message(user_msg)

        # Get completion
        response = self._complete(stream=stream)

        # Handle tool calls if present
        while response.is_tool_call:
            response = self._handle_tool_calls(response)

        return response

    def say(self, content: str, stream: bool = False) -> Message:
        """Alias for ask()."""
        return self.ask(content, stream=stream)

    def stream(self, content: str) -> Iterator[str]:
        """Send a message and stream the response.

        Args:
            content: The user's message.

        Yields:
            Text chunks as they arrive.

        Example:
            >>> for chunk in chat.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        # Add user message
        user_msg = Message.user(content)
        self._add_message(user_msg)

        # Get provider and stream
        provider = get_provider_for_model(self.model)

        accumulated = []
        for chunk in provider.stream(
            messages=self._messages,
            model=self.model,
            temperature=self._temperature,
            tools=self._tools or None,
        ):
            accumulated.append(chunk)
            if self._on_chunk:
                self._on_chunk(chunk)
            yield chunk

        # Add final assistant message
        final_content = "".join(accumulated)
        self._add_message(Message.assistant(final_content, model_id=self.model))

    def add_message(self, message: Message | dict[str, Any]) -> Self:
        """Add a message to the conversation.

        Args:
            message: A Message instance or dict with message data.

        Returns:
            Self for method chaining.
        """
        if isinstance(message, dict):
            message = Message.from_dict(message)
        self._add_message(message)
        return self

    def reset(self) -> Self:
        """Clear all messages except the system message.

        Returns:
            Self for method chaining.
        """
        system_msg = next((m for m in self._messages if m.role == Role.SYSTEM), None)
        self._messages = [system_msg] if system_msg else []
        return self

    def clear(self) -> Self:
        """Clear all messages including system message.

        Returns:
            Self for method chaining.
        """
        self._messages = []
        return self

    # Async methods

    async def aask(self, content: str, stream: bool = False) -> Message:
        """Async version of ask(). Send a message and get a response.

        Args:
            content: The user's message.
            stream: If True, streams the response (use on_chunk callback).

        Returns:
            The assistant's response message.

        Example:
            >>> response = await chat.aask("What is Python?")
        """
        # Add user message
        user_msg = Message.user(content)
        self._add_message(user_msg)

        # Get completion
        response = await self._acomplete(stream=stream)

        # Handle tool calls if present
        while response.is_tool_call:
            response = await self._ahandle_tool_calls(response)

        return response

    async def asay(self, content: str, stream: bool = False) -> Message:
        """Alias for aask()."""
        return await self.aask(content, stream=stream)

    async def astream(self, content: str) -> AsyncIterator[str]:
        """Async version of stream(). Send a message and stream the response.

        Args:
            content: The user's message.

        Yields:
            Text chunks as they arrive.

        Example:
            >>> async for chunk in chat.astream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        # Add user message
        user_msg = Message.user(content)
        self._add_message(user_msg)

        # Get provider and stream
        provider = get_provider_for_model(self.model)

        accumulated = []
        async for chunk in provider.astream(
            messages=self._messages,
            model=self.model,
            temperature=self._temperature,
            tools=self._tools or None,
        ):
            accumulated.append(chunk)
            if self._on_chunk:
                self._on_chunk(chunk)
            yield chunk

        # Add final assistant message
        final_content = "".join(accumulated)
        self._add_message(Message.assistant(final_content, model_id=self.model))

    async def _acomplete(self, stream: bool = False) -> Message:
        """Async version of _complete()."""
        provider = get_provider_for_model(self.model)

        # Convert schema to JSON schema if set
        json_schema = None
        if self._schema:
            json_schema = schema_to_json_schema(self._schema)

        with async_timed_request(
            model=self.model,
            messages=self._messages,
            tools=self._tools or None,
            response_schema=json_schema,
        ) as ctx:
            response = await provider.acomplete(
                messages=self._messages,
                model=self.model,
                temperature=self._temperature,
                tools=self._tools or None,
                stream=stream,
                on_chunk=self._on_chunk,
                response_schema=json_schema,
            )
            ctx["response"] = response

        # Parse structured output if schema is set
        if self._schema and response.content:
            try:
                response.parsed = parse_json_response(response.content, self._schema)
            except Exception:
                response.parsed = None

        self._add_message(response)
        return response

    async def _ahandle_tool_calls(self, message: Message) -> Message:
        """Async version of _handle_tool_calls()."""
        for tool_call in message.tool_calls:
            # Find the tool
            tool = next((t for t in self._tools if t.name == tool_call.name), None)
            if tool is None:
                result = f"Error: Unknown tool '{tool_call.name}'"
            else:
                try:
                    result = tool.execute(**tool_call.arguments)
                    if not isinstance(result, str):
                        result = str(result)
                except Exception as e:
                    result = f"Error executing tool: {e}"

            # Add tool result message
            self._add_message(Message.tool_result(tool_call.id, result))

        # Get follow-up response
        return await self._acomplete()

    # Private methods

    def _add_message(self, message: Message) -> None:
        """Add a message and trigger callback."""
        self._messages.append(message)
        if self._on_new_message:
            self._on_new_message(message)

    def _complete(self, stream: bool = False) -> Message:
        """Get a completion from the provider."""
        provider = get_provider_for_model(self.model)

        # Convert schema to JSON schema if set
        json_schema = None
        if self._schema:
            json_schema = schema_to_json_schema(self._schema)

        with timed_request(
            model=self.model,
            messages=self._messages,
            tools=self._tools or None,
            response_schema=json_schema,
        ) as ctx:
            response = provider.complete(
                messages=self._messages,
                model=self.model,
                temperature=self._temperature,
                tools=self._tools or None,
                stream=stream,
                on_chunk=self._on_chunk,
                response_schema=json_schema,
            )
            ctx["response"] = response

        # Parse structured output if schema is set
        if self._schema and response.content:
            try:
                response.parsed = parse_json_response(response.content, self._schema)
            except Exception:
                response.parsed = None

        self._add_message(response)
        return response

    def _handle_tool_calls(self, message: Message) -> Message:
        """Execute tool calls and get a follow-up response."""
        for tool_call in message.tool_calls:
            # Find the tool
            tool = next((t for t in self._tools if t.name == tool_call.name), None)
            if tool is None:
                result = f"Error: Unknown tool '{tool_call.name}'"
            else:
                try:
                    result = tool.execute(**tool_call.arguments)
                    if not isinstance(result, str):
                        result = str(result)
                except Exception as e:
                    result = f"Error executing tool: {e}"

            # Add tool result message
            self._add_message(Message.tool_result(tool_call.id, result))

        # Get follow-up response
        return self._complete()
