"""
Django LLM - A beautiful unified LLM interface for Django.

One API for OpenAI, Anthropic, Google, Azure OpenAI, and Ollama
with persistence, streaming, and production-ready features.

Features:
    - Unified API across all providers
    - Django model persistence
    - Streaming with callbacks or iterators
    - Tool/function calling
    - Structured output with JSON schema
    - Async support (aask, astream)
    - Cost tracking
    - Observability hooks
    - Retry with exponential backoff
    - Local models via Ollama

Quick Start:
    >>> import django_llm
    >>> response = django_llm.chat().ask("Hello!")
    >>> print(response.content)

    # With specific model
    >>> from django_llm import Chat
    >>> chat = Chat(model="ollama:llama3.2")
    >>> response = chat.ask("Hello!")

    # Streaming
    >>> for chunk in chat.stream("Tell me a story"):
    ...     print(chunk, end="")

For more information, see https://github.com/khasinski/django-llm
"""

from django_llm.chat import Chat
from django_llm.configuration import Config, configure
from django_llm.exceptions import (
    AuthenticationError,
    DjangoLLMError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from django_llm.logging import (
    add_error_hook,
    add_request_hook,
    add_response_hook,
    clear_hooks,
    remove_error_hook,
    remove_request_hook,
    remove_response_hook,
)
from django_llm.message import Message, Role
from django_llm.tool import Tool

__version__ = "1.0.0"
__all__ = [
    "Chat",
    "Message",
    "Role",
    "Config",
    "configure",
    "Tool",
    "DjangoLLMError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    # Observability hooks
    "add_request_hook",
    "add_response_hook",
    "add_error_hook",
    "remove_request_hook",
    "remove_response_hook",
    "remove_error_hook",
    "clear_hooks",
]


def chat(model: str | None = None) -> Chat:
    """Create a new chat instance.

    Args:
        model: Optional model identifier. Supported formats:
            - OpenAI: 'gpt-4o', 'gpt-4o-mini', 'o1'
            - Anthropic: 'claude-sonnet-4-20250514', 'claude-3-5-haiku-20241022'
            - Google: 'gemini-2.0-flash', 'gemini-1.5-pro'
            - Azure: 'azure:your-deployment-name'
            - Ollama: 'ollama:llama3.2', 'ollama:mistral'

            If not provided, uses the default model from configuration.

    Returns:
        A new Chat instance ready for conversation.

    Examples:
        >>> import django_llm

        # Simple usage (uses default model)
        >>> response = django_llm.chat().ask("Hello!")
        >>> print(response.content)

        # With specific cloud model
        >>> response = django_llm.chat("gpt-4o").ask("Hello!")

        # With local Ollama model
        >>> response = django_llm.chat("ollama:llama3.2").ask("Hello!")
    """
    return Chat(model=model)
