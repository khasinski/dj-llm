"""Base provider interface for LLM APIs."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from django_llm.message import Message
from django_llm.tool import Tool


class Provider(ABC):
    """Abstract base class for LLM providers.

    Each provider implements the specifics of communicating with their API
    while exposing a unified interface.
    """

    name: str

    def __init__(
        self,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Message:
        """Generate a completion from the model.

        Args:
            messages: The conversation history.
            model: The model identifier to use.
            temperature: Sampling temperature (0.0-2.0).
            tools: Optional list of tools the model can call.
            stream: Whether to stream the response.
            on_chunk: Callback for streaming chunks.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The assistant's response message.
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate a streaming completion from the model.

        Args:
            messages: The conversation history.
            model: The model identifier to use.
            temperature: Sampling temperature (0.0-2.0).
            tools: Optional list of tools the model can call.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Text chunks as they arrive.
        """
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Message:
        """Async version of complete()."""
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async version of stream()."""
        pass

    @abstractmethod
    def format_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Format tools for this provider's API format."""
        pass
